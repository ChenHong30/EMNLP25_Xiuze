# generate_hidden_states_with_bert_input.py
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
# 需要同时加载 GPT2 和 BERT 的 Tokenizer
from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer
from utils import DataLoader, Batchify, now_time, ids2tokens # 确认 ids2tokens 是否需要

def generate_states_for_bert(args):
    """
    Generates GPT-2 hidden states AND BERT-compatible input_ids/attention_mask
    for a given dataset.
    """
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(now_time() + f"Using device: {device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(now_time() + f"Created output directory: {args.output_dir}")

    # --- Load Tokenizers and Data ---
    print(now_time() + 'Loading tokenizers and data')
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    try:
        # 加载 GPT-2 Tokenizer (用于解码和送入 GPT-2 模型)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.llm_model, bos_token=bos, eos_token=eos, pad_token=pad, unk_token=unk)
        gpt2_pad_token_id = gpt2_tokenizer.pad_token_id
        print(now_time() + f"GPT-2 Tokenizer loaded. Pad token ID: {gpt2_pad_token_id}")

        # 加载 BERT Tokenizer (用于生成 BERT 输入)
        # 需要指定 BERT 模型名称，确保与微调时使用的模型匹配
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
        print(now_time() + f"BERT Tokenizer ({args.bert_model_name}) loaded.")

    except Exception as e:
        print(now_time() + f"Error loading tokenizers: {e}")
        return

    # --- Load Data using existing DataLoader and Batchify ---
    # (假设 DataLoader/Batchify 按原样工作，提供 GPT-2 tokenized 'seq' 和 'mask')
    image_dir = os.path.join(os.path.dirname(args.data_path), 'image')
    corpus = DataLoader(
        data_path=args.data_path,
        index_dir=args.index_dir,
        tokenizer=gpt2_tokenizer, # DataLoader 内部使用 GPT-2 tokenizer 处理 'text'
        seq_len=args.words,       # GPT-2 sequence length
        image_dir=image_dir,
        use_images=True # 保持与原始调用一致
    )

    if args.split == 'valid':
        data = Batchify(corpus.valid, gpt2_tokenizer, bos, eos, args.batch_size)
        print(now_time() + f"Processing validation data")
    elif args.split == 'test':
        data = Batchify(corpus.test, gpt2_tokenizer, bos, eos, args.batch_size)
        print(now_time() + f"Processing test data")
    else: # Default to train
        # Batchify 使用 GPT-2 tokenizer 生成 seq 和 mask
        data = Batchify(corpus.train, gpt2_tokenizer, bos, eos, args.batch_size, shuffle=False)
        print(now_time() + f"Processing training data")


    # --- Load Base GPT-2 Model ---
    print(now_time() + f'Loading base GPT-2 model from: {args.llm_model}')
    try:
        gpt2_model = GPT2Model.from_pretrained(args.llm_model)
        gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        gpt2_model.to(device)
        gpt2_model.eval()
        print(now_time() + 'Base GPT-2 model loaded successfully.')
    except Exception as e:
        print(now_time() + f"Error loading base GPT-2 model: {e}")
        return

    # --- Generate Hidden States and BERT Inputs ---
    print(now_time() + 'Starting hidden state and BERT input generation...')
    batch_num = 0

    with torch.no_grad():
        while True:
            try:
                # 获取 GPT-2 格式的批次数据
                user, item, rating, seq_gpt, mask_gpt, item_embedding = data.next_batch()
                if user is None: break
            except StopIteration:
                break
            except Exception as e:
                print(now_time() + f"Error getting batch {data.step}: {e}")
                continue

            seq_gpt = seq_gpt.to(device)   # GPT-2 input IDs
            mask_gpt = mask_gpt.to(device) # GPT-2 attention mask
            rating = rating.to(device)

            try:
                # === Step 1: Generate GPT-2 Hidden States ===
                outputs = gpt2_model(input_ids=seq_gpt, attention_mask=mask_gpt)
                last_hidden_states_gpt = outputs.last_hidden_state # (batch_size, seq_len_gpt, hidden_dim_gpt)

                # === Step 2: Decode GPT-2 IDs back to text ===
                # 需要移动到 CPU 进行解码和后续的 BERT 编码
                seq_gpt_cpu = seq_gpt.cpu()
                mask_gpt_cpu = mask_gpt.cpu()
                texts_batch = []
                for i in range(seq_gpt_cpu.size(0)):
                    # 根据 mask 确定实际长度，避免解码 padding
                    actual_len = mask_gpt_cpu[i].sum().item()
                    if actual_len == 0: # Handle potential empty sequences if necessary
                         texts_batch.append("")
                         continue
                    real_ids = seq_gpt_cpu[i, :actual_len].tolist()
                    # 使用 GPT-2 tokenizer 解码
                    # skip_special_tokens=True 可能会移除重要结构，先不加或测试后决定
                    text = gpt2_tokenizer.decode(real_ids, skip_special_tokens=False)
                    # 可选：进一步清理文本，例如移除可能存在的 <bos>, <eos> 等？
                    # 取决于你的数据和 BERT 是否需要它们
                    # text = text.replace(gpt2_tokenizer.bos_token, "").replace(gpt2_tokenizer.eos_token, "").strip()
                    texts_batch.append(text)

                # === Step 3: Encode text with BERT Tokenizer ===
                bert_encoding = bert_tokenizer(
                    texts_batch,
                    padding='max_length',        # Pad to BERT max length
                    truncation=True,             # Truncate if longer
                    max_length=args.bert_seq_len,# Use specific BERT sequence length
                    return_tensors='pt'          # Return PyTorch tensors
                )
                bert_input_ids = bert_encoding['input_ids']     # (batch_size, seq_len_bert)
                bert_attention_mask = bert_encoding['attention_mask'] # (batch_size, seq_len_bert)

                # === Step 4: Save all required data ===
                # Move results needing save to CPU (BERT inputs already are)
                hidden_states_cpu = last_hidden_states_gpt.cpu()
                ratings_cpu = rating.cpu()
                mask_gpt_cpu_save = mask_gpt.cpu() # GPT mask for hidden states path

                # Define filenames
                state_filename = os.path.join(args.output_dir, f'batch_{batch_num}_states.pt')
                rating_filename = os.path.join(args.output_dir, f'batch_{batch_num}_ratings.pt')
                # Mask for hidden states (using original GPT mask)
                mask_hidden_filename = os.path.join(args.output_dir, f'batch_{batch_num}_masks.pt')
                # BERT input IDs
                input_ids_filename = os.path.join(args.output_dir, f'batch_{batch_num}_input_ids.pt')
                # BERT attention mask
                mask_raw_filename = os.path.join(args.output_dir, f'batch_{batch_num}_attention_mask_raw.pt')

                # Save the tensors
                torch.save(hidden_states_cpu, state_filename)
                torch.save(ratings_cpu, rating_filename)
                torch.save(mask_gpt_cpu_save, mask_hidden_filename) # Save GPT mask
                torch.save(bert_input_ids, input_ids_filename)       # Save BERT input_ids
                torch.save(bert_attention_mask, mask_raw_filename)   # Save BERT attention_mask

                batch_num += 1
                if batch_num % args.log_interval == 0:
                    print(now_time() + f'Processed and saved batch {batch_num}/{data.total_step} (GPT states, ratings, GPT mask, BERT ids, BERT mask)')

            except Exception as e:
                print(now_time() + f"Error processing or saving batch {batch_num}: {e}")
                # Log the problematic text batch?
                # print("Problematic text batch:", texts_batch)


            if data.step == data.total_step:
                break

    print(now_time() + f'Finished generating data. Total batches saved: {batch_num}')
    print(now_time() + f'Saved data can be found in: {args.output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GPT-2 Hidden States & BERT Inputs')
    # --- Arguments from original script ---
    parser.add_argument('--data_path', type=str, required=True, help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, required=True, help='load indexes')
    parser.add_argument('--llm_model', type=str, default="./llm/models-gpt2", help='path to the base GPT-2 model')
    parser.add_argument('--words', type=int, default=20, help='GPT-2 sequence length') # Keep for GPT-2 side
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=50, help='report interval')
    parser.add_argument('--use_images', action='store_true', help='whether image data is used/loaded', default=True)

    # --- Arguments specific to this script ---
    parser.add_argument('--output_dir', type=str, default='./bert_finetune_data/', help='directory to save the generated data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'], help='Which data split to process')

    # === Arguments required for BERT processing ===
    parser.add_argument('--bert_model_name', type=str, required=True, help='Identifier for the BERT model/tokenizer (e.g., bert-base-uncased)')
    parser.add_argument('--bert_seq_len', type=int, default=512, help='Sequence length for BERT tokenizer (padding/truncation)')


    args = parser.parse_args()

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    generate_states_for_bert(args)