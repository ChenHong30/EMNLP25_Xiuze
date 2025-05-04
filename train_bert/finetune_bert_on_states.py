# finetune_bert_contrastive.py
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
from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup # BertTokenizer might be needed if you load raw text instead of ids
import torch.nn as nn
from tqdm import tqdm
from utils import now_time # Assuming utils.py with now_time() is available

class ContrastiveHiddenStateDataset(Dataset):
    """
    扩展的 Dataset，用于加载 hidden states, masks, ratings, 以及原始文本的 input_ids 和 attention_mask。
    假设数据生成脚本已保存 'batch_*_input_ids.pt' 和 'batch_*_attention_mask_raw.pt' 文件。
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # 查找所有相关文件
        self.state_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_states.pt')))
        self.rating_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_ratings.pt')))
        self.mask_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_masks.pt'))) # Mask for hidden states
        self.input_id_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_input_ids.pt'))) # Input IDs for raw text
        self.attention_mask_raw_files = sorted(glob.glob(os.path.join(data_dir, 'batch_*_attention_mask_raw.pt'))) # Attention mask for raw text

        num_states = len(self.state_files)
        if not (num_states == len(self.rating_files) == len(self.mask_files) == \
                len(self.input_id_files) == len(self.attention_mask_raw_files)):
            raise ValueError(f"Mismatch in number of files found in {data_dir}! "
                             f"States: {num_states}, Ratings: {len(self.rating_files)}, "
                             f"Masks: {len(self.mask_files)}, Input IDs: {len(self.input_id_files)}, "
                             f"Raw Masks: {len(self.attention_mask_raw_files)}")
        if num_states == 0:
            raise ValueError(f"No data files found in {data_dir}. Check data generation and file naming.")
        print(now_time() + f"Found {num_states} batches with all required data types in {data_dir}")

    def __len__(self):
        # 返回批次数
        return len(self.state_files)

    def __getitem__(self, idx):
        # 加载给定批次索引的所有数据
        try:
            hidden_states = torch.load(self.state_files[idx])
            ratings = torch.load(self.rating_files[idx])
            masks_hidden = torch.load(self.mask_files[idx]) # Mask for hidden states path
            input_ids_raw = torch.load(self.input_id_files[idx]) # Input IDs for raw text path
            masks_raw = torch.load(self.attention_mask_raw_files[idx]) # Mask for raw text path

            # 确保 ratings 是 float 类型
            # 返回一个包含所有数据的元组
            return hidden_states, masks_hidden, input_ids_raw, masks_raw, ratings.float()
        except Exception as e:
            print(f"Error loading batch {idx} from {self.data_dir}: {e}")
            # 返回 None，DataLoader 的 collate_fn 需要处理
            return None

class BertForRegressionContrastive(nn.Module):
    """
    修改后的 BERT 模型，可以接受 inputs_embeds (hidden states) 或 input_ids (raw text)
    """
    def __init__(self, bert_model_name, dropout_prob=0.1, input_hidden_dim=768): # 保持 input_hidden_dim 参数
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.config = self.bert.config
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.config.hidden_size, 1)
        input_hidden_dim = 768
        # 投影层逻辑（仅当 hidden state 输入维度与模型维度不匹配时使用）
        self.projection = None
        model_hidden_dim = self.config.hidden_size
        if input_hidden_dim is not None and input_hidden_dim != model_hidden_dim:
            print(f"{now_time()}Input hidden dimension ({input_hidden_dim}) differs from model ({model_hidden_dim}). Adding projection.")
            self.projection = nn.Linear(input_hidden_dim, model_hidden_dim)
        elif input_hidden_dim is None:
             # 如果提供了 hidden states 输入但未提供 input_hidden_dim，发出警告或假设维度匹配
             print(f"{now_time()}Warning: input_hidden_dim not provided for hidden state input path. Assuming it matches model hidden dimension ({model_hidden_dim}).")


    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # inputs_embeds: 预计算的 hidden states (batch_size, seq_len, input_hidden_dim)
        # input_ids: 原始文本的 token IDs (batch_size, seq_len)
        # attention_mask: 对应的 attention mask (batch_size, seq_len)

        if input_ids is not None:
            # --- Raw Text Path ---
            # 使用 BERT 内置的 embedding 层
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 使用第一个 token ([CLS]) 的输出来进行回归
            cls_output = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_dim)

        elif inputs_embeds is not None:
            # --- Hidden State Path ---
            processed_embeds = inputs_embeds
            # 如果需要，应用投影层
            if self.projection is not None:
                processed_embeds = self.projection(inputs_embeds) # (batch_size, seq_len, model_hidden_dim)

            # 将处理过的 embeds 输入 BERT (注意：BERT 的 embedding 层将被跳过)
            outputs = self.bert(inputs_embeds=processed_embeds, attention_mask=attention_mask)
            # 同样使用第一个位置的输出来进行回归
            cls_output = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_dim)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # --- Common Regression Head ---
        pooled_output = self.dropout(cls_output)
        logits = self.regressor(pooled_output) # (batch_size, 1)
        return logits.squeeze(-1) # (batch_size,)


def train_bert_contrastive(args):
    """
    使用 hidden states 和 raw text 进行对比训练的主函数
    """
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(now_time() + f"Using device: {device}")

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
        print(now_time() + f"Created output directory: {args.output_model_dir}")

    # --- Prepare Data ---
    print(now_time() + f"Loading training data from: {args.train_data_dir}")
    train_dataset = ContrastiveHiddenStateDataset(args.train_data_dir)
    # DataLoader 的 batch_size 仍然是 1，因为每个 Dataset item 本身就是一个预处理好的批次
    # collate_fn 处理可能由 __getitem__ 返回的 None
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0] if x[0] is not None else None)

    val_dataloader = None
    if args.valid_data_dir:
        print(now_time() + f"Loading validation data from: {args.valid_data_dir}")
        try:
            val_dataset = ContrastiveHiddenStateDataset(args.valid_data_dir)
            val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=lambda x: x[0] if x[0] is not None else None)
        except ValueError as e:
            print(now_time() + f"Warning: Could not load validation data: {e}")

    # --- Prepare Model ---
    print(now_time() + f"Loading BERT model: {args.bert_model_name}")
    # 创建可以处理两种输入的模型
    model = BertForRegressionContrastive(args.bert_model_name,
                                        input_hidden_dim=args.input_hidden_dim) # input_hidden_dim 主要用于 hidden state 路径
    model.to(device)

    # --- Prepare Optimizer and Scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    # --- Loss Function ---
    criterion = nn.MSELoss() # 仍然使用 MSE 进行回归

    # --- Training Loop ---
    print(now_time() + "Starting BERT contrastive fine-tuning...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        print(now_time() + f"--- Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        total_train_loss = 0
        total_train_loss_hidden = 0
        total_train_loss_raw = 0
        num_train_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch_data in progress_bar:
            if batch_data is None:
                print(now_time() + f"Skipping a None batch (likely loading error).")
                continue

            # 解包数据
            hidden_states, masks_hidden, input_ids_raw, masks_raw, ratings = batch_data

            # 将数据移动到设备
            hidden_states = hidden_states.to(device)
            masks_hidden = masks_hidden.to(device)
            input_ids_raw = input_ids_raw.to(device)
            masks_raw = masks_raw.to(device)
            ratings = ratings.to(device)

            model.zero_grad()

            # --- Forward Pass 1: Hidden States ---
            predictions_hidden = model(inputs_embeds=hidden_states, attention_mask=masks_hidden)
            loss_hidden = criterion(predictions_hidden, ratings)

            # --- Forward Pass 2: Raw Text ---
            predictions_raw = model(input_ids=input_ids_raw, attention_mask=masks_raw)
            loss_raw = criterion(predictions_raw, ratings)

            # --- Combine Losses ---
            # 使用 args.loss_weight 来控制比例 (例如 0.5 代表 50/50)
            loss = args.loss_weight * loss_hidden + (1.0 - args.loss_weight) * loss_raw

            # --- Backward Pass & Optimization ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_loss_hidden += loss_hidden.item()
            total_train_loss_raw += loss_raw.item()
            num_train_batches += 1
            progress_bar.set_postfix({'Total Loss': loss.item(),
                                      'Hidden Loss': loss_hidden.item(),
                                      'Raw Loss': loss_raw.item()})

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_loss_hidden = total_train_loss_hidden / num_train_batches if num_train_batches > 0 else 0
        avg_train_loss_raw = total_train_loss_raw / num_train_batches if num_train_batches > 0 else 0
        print(now_time() + f"Average Training Loss: {avg_train_loss:.4f} "
                         f"(Hidden: {avg_train_loss_hidden:.4f}, Raw: {avg_train_loss_raw:.4f})")

        # --- Validation Loop ---
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            total_val_loss_hidden = 0
            total_val_loss_raw = 0
            num_val_batches = 0
            print(now_time() + "Running Validation...")
            with torch.no_grad():
                for batch_data in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False):
                    if batch_data is None: continue

                    hidden_states, masks_hidden, input_ids_raw, masks_raw, ratings = batch_data
                    hidden_states = hidden_states.to(device)
                    masks_hidden = masks_hidden.to(device)
                    input_ids_raw = input_ids_raw.to(device)
                    masks_raw = masks_raw.to(device)
                    ratings = ratings.to(device)

                    # Validation Hidden State Path
                    predictions_hidden = model(inputs_embeds=hidden_states, attention_mask=masks_hidden)
                    loss_hidden = criterion(predictions_hidden, ratings)

                    # Validation Raw Text Path
                    predictions_raw = model(input_ids=input_ids_raw, attention_mask=masks_raw)
                    loss_raw = criterion(predictions_raw, ratings)

                    # Combine validation losses (using the same weight for comparison)
                    loss = args.loss_weight * loss_hidden + (1.0 - args.loss_weight) * loss_raw

                    total_val_loss += loss.item()
                    total_val_loss_hidden += loss_hidden.item()
                    total_val_loss_raw += loss_raw.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_loss_hidden = total_val_loss_hidden / num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_loss_raw = total_val_loss_raw / num_val_batches if num_val_batches > 0 else float('inf')
            print(now_time() + f"Average Validation Loss: {avg_val_loss:.4f} "
                             f"(Hidden: {avg_val_loss_hidden:.4f}, Raw: {avg_val_loss_raw:.4f})")

            # --- Early Stopping and Model Saving (based on combined validation loss) ---
            if avg_val_loss < best_val_loss:
                print(now_time() + f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                bert_output_dir = os.path.join(args.output_model_dir, "bert_fine_tuned")
                if not os.path.exists(bert_output_dir):
                    os.makedirs(bert_output_dir)
                # 保存整个模型（包括 BERT 和 回归头），因为它们一起训练
                # 或者只保存 BERT 部分 model.bert.save_pretrained(bert_output_dir)
                # 并单独保存回归头 torch.save(model.regressor.state_dict(), ...)
                # 这里选择保存整个模型状态字典，更灵活
                model.bert.save_pretrained(bert_output_dir)
                # 或者使用 save_pretrained 保存 BERT 部分和 config
                # model.bert.save_pretrained(bert_output_dir)
                # model.config.save_pretrained(bert_output_dir)
                # torch.save(model.regressor.state_dict(), os.path.join(bert_output_dir, "regressor_head.pt"))
                # if model.projection: torch.save(model.projection.state_dict(), ...)

                # epochs_no_improve = 0 # 重置计数器
            else:
                epochs_no_improve += 1
                print(now_time() + f"Validation loss did not improve. Total epochs without improvement: {epochs_no_improve}.")
                if epochs_no_improve >= args.patience:
                    print(now_time() + f"Early stopping triggered after {args.patience} epochs without improvement.")
                    break
        else:
             # 如果没有验证集，可以在每个 epoch 结束时保存模型
             pass
             # model_save_path = os.path.join(args.output_model_dir, f"model_epoch_{epoch+1}.pt")
             # torch.save(model.state_dict(), model_save_path)
             # print(f"{now_time()}Saved model checkpoint to {model_save_path}")


    print(now_time() + "Finished BERT contrastive fine-tuning.")
    if val_dataloader:
        print(now_time() + f"Best validation loss: {best_val_loss:.4f}")
    # 指向正确的保存路径
    print(now_time() + f"Fine-tuned BERT model saved in: {os.path.join(args.output_model_dir, 'bert_fine_tuned')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune BERT contrastively on hidden states and raw text')
    # --- Data Arguments ---
    parser.add_argument('--train_data_dir', type=str, required=True, help='Directory containing generated training data (.pt files for states, masks, ratings, input_ids, raw_masks)')
    parser.add_argument('--valid_data_dir', type=str, default=None, help='Optional: Directory containing generated validation data')

    # --- Model Arguments ---
    parser.add_argument('--bert_model_name', type=str, required=True, help='Identifier for the pre-trained BERT model')
    parser.add_argument('--output_model_dir', type=str, default='./contrastive_fine_tuned_bert/', help='Directory to save the fine-tuned model')
    parser.add_argument('--input_hidden_dim', type=int, default=None, help='Optional: Dimension of input hidden states if different from BERT model hidden size.')

    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for AdamW optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Linear warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    # --- Contrastive Learning Arguments ---
    parser.add_argument('--loss_weight', type=float, default=0.5, help='Weight for the hidden state loss (raw text loss weight will be 1 - loss_weight)')


    args = parser.parse_args()

    # 验证 loss_weight 在合理范围
    if not 0.0 <= args.loss_weight <= 1.0:
        raise ValueError("--loss_weight must be between 0.0 and 1.0")

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    train_bert_contrastive(args)