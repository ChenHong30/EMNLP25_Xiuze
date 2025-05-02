# generate_hidden_states.py
import os
import sys

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到 Python 的模块搜索路径中
sys.path.append(parent_dir)

import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model # 直接加载 GPT2Model
from utils import DataLoader, Batchify, now_time, ids2tokens

def generate_states_for_bert(args):
    """
    Generates hidden states from a base GPT-2 model for a given dataset
    and saves them along with ratings for BERT pre-fine-tuning.
    """
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(now_time() + f"Using device: {device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(now_time() + f"Created output directory: {args.output_dir}")

    # --- Load Tokenizer and Data ---
    print(now_time() + 'Loading tokenizer and data')
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.llm_model, bos_token=bos, eos_token=eos, pad_token=pad, unk_token=unk)
        pad_token_id = tokenizer.pad_token_id
        print(now_time() + f"Tokenizer loaded. Pad token ID: {pad_token_id}")
    except Exception as e:
        print(now_time() + f"Error loading tokenizer: {e}")
        return

    # Assume DataLoader handles image features appropriately if needed,
    # but they are not directly used by the base GPT-2 transformer here.
    image_dir = os.path.join(os.path.dirname(args.data_path), 'image')
    corpus = DataLoader(
        data_path=args.data_path,
        index_dir=args.index_dir,
        tokenizer=tokenizer,
        seq_len=args.words,
        image_dir=image_dir,
        use_images=True
    )

    # Decide which dataset split to use (e.g., 'train')
    if args.split == 'valid':
        data = Batchify(corpus.valid, tokenizer, bos, eos, args.batch_size)
        print(now_time() + f"Processing validation data")
    elif args.split == 'test':
        data = Batchify(corpus.test, tokenizer, bos, eos, args.batch_size)
        print(now_time() + f"Processing test data")
    else: # Default to train
        data = Batchify(corpus.train, tokenizer, bos, eos, args.batch_size, shuffle=False) # No shuffle needed
        print(now_time() + f"Processing training data")


    # --- Load Base GPT-2 Model ---
    print(now_time() + f'Loading base GPT-2 model from: {args.llm_model}')
    try:
        # Load only the core GPT-2 model, without any custom heads or LoRA adapters
        gpt2_model = GPT2Model.from_pretrained(args.llm_model)
        # Resize embeddings if tokenizer added special tokens (like <bos>, <eos>, <pad>)
        gpt2_model.resize_token_embeddings(len(tokenizer))
        gpt2_model.to(device)
        gpt2_model.eval() # Set to evaluation mode
        print(now_time() + 'Base GPT-2 model loaded successfully.')
    except Exception as e:
        print(now_time() + f"Error loading base GPT-2 model: {e}")
        return

    # --- Generate Hidden States ---
    print(now_time() + 'Starting hidden state generation...')
    batch_num = 0
    # # Reset data iterator
    # data.reset()

    with torch.no_grad(): # Ensure no gradients are computed
        while True:
            # Fetch batch data - assuming Batchify yields user, item, rating, seq, mask, item_embedding
            # We primarily need seq and mask for base GPT-2, and rating for saving.
            try:
                 user, item, rating, seq, mask, item_embedding = data.next_batch()
                 # If next_batch returns None or raises StopIteration when done
                 if user is None: break
            except StopIteration:
                 break # End of data
            except Exception as e:
                 print(now_time() + f"Error getting batch {data.step}: {e}")
                 continue # Skip batch on error

            seq = seq.to(device)   # (batch_size, seq_len)
            mask = mask.to(device) # (batch_size, seq_len)
            rating = rating.to(device) # Keep ratings on the same device initially

            try:
                # Pass sequence and mask through the base GPT-2 model
                outputs = gpt2_model(input_ids=seq, attention_mask=mask)
                # Get the hidden states of the last layer
                last_hidden_states = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_dim)

                # --- Save batch data ---
                # Move to CPU before saving to avoid GPU memory buildup if saving tensors directly
                hidden_states_cpu = last_hidden_states.cpu()
                ratings_cpu = rating.cpu()
                mask_cpu = mask.cpu() # Also save the mask, might be useful for BERT

                # Define filenames for this batch
                state_filename = os.path.join(args.output_dir, f'batch_{batch_num}_states.pt')
                rating_filename = os.path.join(args.output_dir, f'batch_{batch_num}_ratings.pt')
                mask_filename = os.path.join(args.output_dir, f'batch_{batch_num}_masks.pt')

                # Save the tensors
                torch.save(hidden_states_cpu, state_filename)
                torch.save(ratings_cpu, rating_filename)
                torch.save(mask_cpu, mask_filename)

                batch_num += 1
                if batch_num % args.log_interval == 0:
                    print(now_time() + f'Processed and saved batch {batch_num}/{data.total_step}')

            except Exception as e:
                print(now_time() + f"Error processing or saving batch {batch_num}: {e}")
                # Decide if you want to stop or continue on error
                # continue

            # Check if data loading is finished (might be redundant with try/except StopIteration)
            if data.step == data.total_step:
                 break

    print(now_time() + f'Finished generating hidden states. Total batches saved: {batch_num}')
    print(now_time() + f'Saved data can be found in: {args.output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GPT-2 Hidden States for BERT Pre-fine-tuning')
    # --- Arguments from train_gpt_lora_mf_mlp.py that are relevant ---
    parser.add_argument('--data_path', type=str, required=True, help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, required=True, help='load indexes')
    parser.add_argument('--llm_model', type=str, default="./llm/models-gpt2", help='path to the base LLM model (e.g., GPT-2)')
    parser.add_argument('--words', type=int, default=20, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=50, help='report interval')
    parser.add_argument('--use_images', action='store_true', help='whether image data is used/loaded by DataLoader', default=True) # Needed for DataLoader init

    # --- New arguments for this script ---
    parser.add_argument('--output_dir', type=str, default='./bert_finetune_data/', help='directory to save the generated states and ratings')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'], help='Which data split to process (train/valid/test)')

    args = parser.parse_args()

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    generate_states_for_bert(args)