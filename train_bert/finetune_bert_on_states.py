# finetune_bert_on_states.py
import os
import sys

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到 Python 的模块搜索路径中
sys.path.append(parent_dir)

import torch
import argparse
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
from utils import now_time # Assuming utils.py with now_time() is available

class HiddenStateDataset(Dataset):
    """
    Custom PyTorch Dataset to load pre-generated hidden states, masks, and ratings
    saved batch by batch.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Find all batch files, assuming they are named correctly
        self.state_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_states.pt')))
        self.rating_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_ratings.pt')))
        self.mask_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_masks.pt')))

        if not (len(self.state_files) == len(self.rating_files) == len(self.mask_files)):
            raise ValueError("Mismatch in number of state, rating, and mask files found!")
        if len(self.state_files) == 0:
             raise ValueError(f"No data files found in {data_dir}. Did generate_hidden_states.py run correctly?")
        print(now_time() + f"Found {len(self.state_files)} batches in {data_dir}")

    def __len__(self):
        # Returns the number of batches
        return len(self.state_files)

    def __getitem__(self, idx):
        # Load the data for the given batch index
        try:
            hidden_states = torch.load(self.state_files[idx])
            ratings = torch.load(self.rating_files[idx])
            masks = torch.load(self.mask_files[idx])
            # Ensure ratings are float for regression loss
            return hidden_states, masks, ratings.float()
        except Exception as e:
            print(f"Error loading batch {idx} from {self.data_dir}: {e}")
            # Return None or dummy data, or re-raise exception
            # Returning None might require careful handling in DataLoader collation
            return None # Or handle appropriately

class BertForRegression(nn.Module):
    """
    BERT model with a regression head on top of the [CLS] token output.
    """
    def __init__(self, bert_model_name, dropout_prob=0.1, input_hidden_dim=768): # 添加 input_hidden_dim 参数
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Use BERT's config to get hidden size
        self.config = self.bert.config
        self.dropout = nn.Dropout(dropout_prob)
        # Regression head: maps BERT's [CLS] embedding to a single rating value
        self.regressor = nn.Linear(self.config.hidden_size, 1)
        input_hidden_dim = 768
        # 添加投影层（如果输入维度与模型维度不匹配）
        self.projection = None
        model_hidden_dim = self.config.hidden_size
        if input_hidden_dim is not None and input_hidden_dim != model_hidden_dim:
            print(f"{now_time()}Input dimension ({input_hidden_dim}) differs from model hidden dimension ({model_hidden_dim}). Adding a projection layer.")
            self.projection = nn.Linear(input_hidden_dim, model_hidden_dim)
        elif input_hidden_dim is None:
             print(f"{now_time()}input_hidden_dim not provided. Assuming input dimension matches model hidden dimension ({model_hidden_dim}).")


    def forward(self, inputs_embeds, attention_mask):
        # inputs_embeds: pre-computed hidden states (batch_size, seq_len, input_hidden_dim)
        # attention_mask: (batch_size, seq_len)

        # 应用投影层（如果存在）
        if self.projection is not None:
            inputs_embeds = self.projection(inputs_embeds) # (batch_size, seq_len, model_hidden_dim)

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # Use the embedding of the first token ([CLS] position equivalent)
        cls_output = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_dim)
        # Apply dropout and the regression head
        pooled_output = self.dropout(cls_output)
        logits = self.regressor(pooled_output) # (batch_size, 1)
        # Remove the last dimension for loss calculation
        return logits.squeeze(-1) # (batch_size,)


def train_bert(args):
    """
    Main function to train the BERT model on generated hidden states.
    """
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(now_time() + f"Using device: {device}")

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
        print(now_time() + f"Created output directory: {args.output_model_dir}")

    # --- Prepare Data ---
    print(now_time() + f"Loading training data from: {args.train_data_dir}")
    train_dataset = HiddenStateDataset(args.train_data_dir)
    # Note: Batch size here is effectively 1 in terms of dataset batches,
    # but each loaded item IS a batch from the generation phase.
    # So, the effective batch size is the one used during generation.
    # We set batch_size=1 for the DataLoader loading these files.
    # We handle potential None items from dataset __getitem__
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0] if x[0] is not None else None)

    # Optional: Load validation data
    val_dataloader = None
    if args.valid_data_dir:
        print(now_time() + f"Loading validation data from: {args.valid_data_dir}")
        try:
             val_dataset = HiddenStateDataset(args.valid_data_dir)
             val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=lambda x: x[0] if x[0] is not None else None)
        except ValueError as e:
             print(now_time() + f"Warning: Could not load validation data: {e}")


    # --- Prepare Model ---
    print(now_time() + f"Loading BERT model: {args.bert_model_name}")
    # 传递 input_hidden_dim 给模型构造函数
    model = BertForRegression(args.bert_model_name, input_hidden_dim=args.input_hidden_dim)
    model.to(device)

    # --- Prepare Optimizer and Scheduler ---
    # 确保优化器能看到所有参数，包括可能添加的 projection 层
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    # Calculate total training steps for scheduler
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    # --- Loss Function ---
    criterion = nn.MSELoss()

    # --- Training Loop ---
    print(now_time() + "Starting BERT fine-tuning...")
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Initialize cumulative counter

    for epoch in range(args.epochs):
        print(now_time() + f"--- Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch_data in progress_bar:
            # Skip if batch loading failed
            if batch_data is None:
                 print(now_time() + f"Skipping a None batch (likely loading error).")
                 continue

            hidden_states, masks, ratings = batch_data
            hidden_states = hidden_states.to(device)
            masks = masks.to(device)
            ratings = ratings.to(device)

            model.zero_grad()
            predictions = model(inputs_embeds=hidden_states, attention_mask=masks)
            loss = criterion(predictions, ratings)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            num_train_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        print(now_time() + f"Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            print(now_time() + "Running Validation...")
            with torch.no_grad():
                for batch_data in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False):
                    if batch_data is None: continue # Skip bad batches
                    hidden_states, masks, ratings = batch_data
                    hidden_states = hidden_states.to(device)
                    masks = masks.to(device)
                    ratings = ratings.to(device)

                    predictions = model(inputs_embeds=hidden_states, attention_mask=masks)
                    loss = criterion(predictions, ratings)
                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            print(now_time() + f"Average Validation Loss: {avg_val_loss:.4f}")

            # --- Early Stopping and Model Saving ---
            if avg_val_loss < best_val_loss:
                print(now_time() + f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                # Save the BERT part of the model
                bert_output_dir = os.path.join(args.output_model_dir, "bert_fine_tuned")
                if not os.path.exists(bert_output_dir):
                    os.makedirs(bert_output_dir)
                model.bert.save_pretrained(bert_output_dir) # Save only BERT weights and config
                # Optionally save the regressor head too if needed later
                # torch.save(model.regressor.state_dict(), os.path.join(args.output_model_dir, "regressor_head.pt"))
                # Removed resetting epochs_no_improve here
            else:
                epochs_no_improve += 1 # Increment cumulative counter
                print(now_time() + f"Validation loss did not improve. Total epochs without improvement: {epochs_no_improve}.")
                if epochs_no_improve >= args.patience:
                    print(now_time() + f"Early stopping triggered after accumulating {args.patience} epochs without improvement.")
                    break # Stop training
        else:
             # If no validation, save model at the end of each epoch or just the last one
             pass # Add logic here if needed, e.g., save last epoch model

    print(now_time() + "Finished BERT fine-tuning.")
    print(now_time() + f"Best validation loss: {best_val_loss:.4f}" if val_dataloader else "No validation performed.")
    print(now_time() + f"Fine-tuned BERT model saved in: {os.path.join(args.output_model_dir, 'bert_fine_tuned')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune BERT on pre-generated hidden states')
    # --- Data Arguments ---
    parser.add_argument('--train_data_dir', type=str, required=True, help='Directory containing the generated training states/ratings (.pt files)')
    parser.add_argument('--valid_data_dir', type=str, default=None, help='Optional: Directory containing the generated validation states/ratings')

    # --- Model Arguments ---
    parser.add_argument('--bert_model_name', type=str, required=True, help='Identifier for the pre-trained BERT model to load') # Made required
    parser.add_argument('--output_model_dir', type=str, default='./fine_tuned_bert/', help='Directory to save the fine-tuned BERT model')
    # 添加 input_hidden_dim 参数
    parser.add_argument('--input_hidden_dim', type=int, default=None, help='Optional: Dimension of the input hidden states if different from BERT model hidden size.')


    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for AdamW optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Linear warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping based on validation loss')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    args = parser.parse_args()

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    train_bert(args)