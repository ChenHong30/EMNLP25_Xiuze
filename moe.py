import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import NewGELUActivation

# --- 适用于 Transformer Block 内部的 MoE FFN ---


class TransformerExpert(nn.Module):
    """模仿 GPT2MLP 结构的专家网络"""
    def __init__(self, embed_dim, intermediate_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, intermediate_dim) # 第一个线性层
        self.c_proj = nn.Linear(intermediate_dim, embed_dim) # 第二个线性层
        self.act = NewGELUActivation() # 使用与 GPT-2 匹配的激活函数
        self.dropout = nn.Dropout(0.1) # 可以根据需要调整 dropout

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerMoELayer(nn.Module):
    """
    用于替换 GPT2Block 中 MLP 的 MoE 层 (Top-k Gating, Qwen 风格稀疏计算)
    """
    def __init__(self, embed_dim, num_experts=4, intermediate_dim=None, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.top_k = top_k
        if intermediate_dim is None:
            intermediate_dim = embed_dim * 4
        self.intermediate_dim = intermediate_dim
        self.experts = nn.ModuleList([TransformerExpert(embed_dim, intermediate_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        self.gate.weight.data.normal_(mean=0.0, std=initrange)
        for expert in self.experts:
            expert.c_fc.weight.data.normal_(mean=0.0, std=initrange)
            expert.c_fc.bias.data.zero_()
            expert.c_proj.weight.data.normal_(mean=0.0, std=initrange)
            expert.c_proj.bias.data.zero_()

    def forward(self, hidden_states): # 输入 hidden_states shape: (batch_size, seq_len, embed_dim)
        batch_size, sequence_length, embed_dim = hidden_states.shape
        k = min(self.top_k, self.num_experts)
        hidden_states_flat = hidden_states.view(-1, embed_dim) # (batch*seq_len, embed_dim)
        num_tokens = hidden_states_flat.shape[0]

        # 1. 计算门控 logits
        gate_logits = self.gate(hidden_states_flat) # (num_tokens, num_experts)

        # 2. 选择 Top-k 专家及其 logits
        #    注意：这里选择 topk logits 而不是 softmax 后的 weights，Softmax 在后面进行
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=k, dim=-1) # (num_tokens, k)

        # 3. 对 Top-k logits 应用 Softmax 得到权重
        top_k_weights = F.softmax(top_k_logits, dim=-1) # (num_tokens, k)

        # 4. 初始化最终输出张量
        final_hidden_states_flat = torch.zeros_like(hidden_states_flat) # (num_tokens, embed_dim)

        # 5. 循环遍历每个专家，进行稀疏计算
        #    构建一个扁平化的 one-hot mask 用于查找 token
        #    zeros: (num_tokens, num_experts)
        #    将 top_k_weights 填充到 top_k_indices 指定的位置
        sparse_weights = torch.zeros_like(gate_logits) # (num_tokens, num_experts)
        sparse_weights.scatter_(dim=-1, index=top_k_indices, src=top_k_weights)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # 找出路由到当前专家的 token 的索引
            # 使用 Qwen 的方法更鲁棒：找出 sparse_weights 中当前专家权重 > 0 的位置
            token_indices_for_expert = torch.where(sparse_weights[:, expert_idx] > 0)[0]

            if token_indices_for_expert.numel() > 0:
                # 选择需要由当前专家处理的 hidden_states
                current_states = hidden_states_flat[token_indices_for_expert] # (num_tokens_for_expert, embed_dim)

                # 直接从 sparse_weights 中获取这些 token 分配给当前专家的权重
                current_weights = sparse_weights[token_indices_for_expert, expert_idx] # (num_tokens_for_expert,)

                # 计算专家输出并加权
                expert_output = expert_layer(current_states) # (num_tokens_for_expert, embed_dim)
                weighted_expert_output = expert_output * current_weights.unsqueeze(-1) # (num_tokens_for_expert, embed_dim)

                # 使用 index_add_ 将结果加回到最终输出张量的对应位置
                final_hidden_states_flat.index_add_(0, token_indices_for_expert, weighted_expert_output.to(hidden_states.dtype))

        # 将输出 reshape 回原始形状
        final_hidden_states = final_hidden_states_flat.view(batch_size, sequence_length, embed_dim)
        return final_hidden_states # Qwen 返回了 router_logits，我们这里只返回 hidden_states