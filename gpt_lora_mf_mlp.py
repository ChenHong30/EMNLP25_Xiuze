import transformers
import models_forward

transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = models_forward.gpt2_attention_forward
transformers.models.gpt2.modeling_gpt2.GPT2Block.forward = models_forward.gpt2_block_forward
transformers.models.gpt2.modeling_gpt2.GPT2Model.forward = models_forward.gpt2_model_forward

from transformers import GPT2Tokenizer,GPT2LMHeadModel, BertModel, BertTokenizer, BertConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch.nn as nn
import torch
import copy
from lora import LoraLinear
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from utils import now_time
from moe import TransformerMoELayer


class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, pre_model, post_att, nuser, nitem, lora_nums, lora_dim,
                        num_heads, pad_token_id, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # -------------------------------- LoRA Replacement --------------------------------

        # Replace targeting linear layers with LoRA layers.
        # get target module name
        target_names = []
        lora_layer_nums = [int(n) for n in lora_nums.split(",")]
        lora_layer = ["transformer.h.{}".format(ll) for ll in lora_layer_nums]
        
        # print("\n" + "="*50)
        # print("LoRA Layer Distribution:")
        # print(f"Layer numbers with LoRA: {lora_layer_nums}")
        # print(f"Layer names with LoRA: {lora_layer}")
        # print("="*50 + "\n")
        
        for name, module in model.named_modules():
            lora_layer_bool = sum([f"{lora_layer_name}." in name for lora_layer_name in lora_layer])
            # lora_layer_bool = sum([lora_layer_name in name for lora_layer_name in lora_layer])
            # if "ln_1" in name or "ln_2" in name, if "mlp.c_fc" in name
            if lora_layer_bool > 0 and "attn.c_attn" in name:
                target_names.append(name)
                # print(f"Found LoRA target: {name}")

        # replace each module with LoRA
        for name in target_names:
            name_struct = name.split(".")
            # get target module
            module_list = [model]
            for struct in name_struct:
                module_list.append(getattr(module_list[-1], struct))
            # build LoRA
            lora = LoraLinear(
                weight=torch.transpose(module_list[-1].weight, 0, 1),
                bias=module_list[-1].bias,
                lora_dim=lora_dim,
            )
            # replace
            module_list[-2].__setattr__(name_struct[-1], lora)

        # -------------------------------- LoRA Replacement --------------------------------

        # # -------------------------------- MoE Replacement --------------------------------
        # config = model.config
        # emsize = config.n_embd
        # intermediate_dim = config.n_inner if config.n_inner is not None else emsize * 4 # 获取或计算中间层维度

        # # --- 替换 Transformer 块中的 MLP 为 MoE ---
        # print(now_time() + "Replacing MLP with MoE in Transformer Blocks")
        # num_experts_transformer = 4 # Transformer 内部 MoE 的专家数
        # num_replaced = 0
        # for i, block in enumerate(model.transformer.h):
        #     # 创建新的 MoE 层实例
        #     moe_layer = TransformerMoELayer(
        #         embed_dim=emsize,
        #         num_experts=num_experts_transformer,
        #         intermediate_dim=intermediate_dim)
        #     # 替换原始的 MLP 层
        #     block.mlp = moe_layer
        #     num_replaced += 1
        # print(now_time() + f"Successfully replaced MLP in {num_replaced} transformer blocks with MoE ({num_experts_transformer} experts each).")
        # # 冻结除FFN以外参数
        # for param in model.parameters():
        #     param.requires_grad = False
        
        # num_unfrozen_tensors = 0
        # total_unfrozen_params = 0
        # unfrozen_param_names_list = [] # 用于记录解冻的参数名，方便验证

        # # 然后，解冻所有属于 MoE 层的参数
        # for name, param in model.named_parameters():
        #     is_moe_param = False
        #     name_parts = name.split('.')
        #     # 检查参数名是否属于被替换的 MLP（现在是 MoE）路径
        #     # 路径通常是 transformer.h.<block_index>.mlp.<internal_moe_layer_name>
        #     # 核心是检查它是否在 transformer.h.<数字>.mlp 这个层级下
        #     if len(name_parts) > 3 and \
        #     name_parts[0] == 'transformer' and \
        #     name_parts[1] == 'h' and \
        #     name_parts[2].isdigit() and \
        #     name_parts[3] == 'mlp':
        #         is_moe_param = True

        #     if is_moe_param:
        #         param.requires_grad = True
        #         num_unfrozen_tensors += 1
        #         total_unfrozen_params += param.numel() # 统计解冻参数的总数量
        #         unfrozen_param_names_list.append(name) # 记录名字用于后续验证
        # # verificatioon
        # trainable_params_found = unfrozen_param_names_list 
        # if not trainable_params_found:
        #     print(now_time() + "🚨 Warning: No parameters ended up being marked as trainable! Check the freezing logic.")
        # else:
        #     # 再次确认数量（应与上面打印的一致）
        #     print(now_time() + f"Verifying the {len(trainable_params_found)} parameter tensors marked as trainable.")
        #     print(now_time() + f"Listing examples:")
            
        #     # 打印前 10 个和最后 10 个名字，排序以便查看
        #     limit = 10 
        #     sorted_trainable_names = sorted(trainable_params_found) 
        #     for i, name in enumerate(sorted_trainable_names):
        #         if i < limit or i >= len(sorted_trainable_names) - limit:
        #             print(f"   - {name}")
        #         elif i == limit:
        #             print("     ...")
                    
        #     # 执行核心的模式检查
        #     all_match_pattern = True
        #     mismatched_names = []
        #     for name in trainable_params_found:
        #         # 再次应用严格的模式检查逻辑
        #         name_parts = name.split('.')
        #         is_expected_pattern = (
        #             len(name_parts) > 3 and
        #             name_parts[0] == 'transformer' and
        #             name_parts[1] == 'h' and
        #             name_parts[2].isdigit() and
        #             name_parts[3] == 'mlp'
        #         )
        #         if not is_expected_pattern:
        #             all_match_pattern = False
        #             mismatched_names.append(name)

        #     # 打印最终验证结果
        #     if all_match_pattern:
        #         print(now_time() + "✅ Verification Check PASSED: All identified trainable parameters correctly follow the 'transformer.h.<i>.mlp.' pattern.")
        #     else:
        #         print(now_time() + "🚨 Verification Check FAILED: Some trainable parameters DO NOT follow the expected 'transformer.h.<i>.mlp.' pattern!")
        #         print(now_time() + "Please review the list above and these potential mismatches:")
        #         for mismatched_name in mismatched_names[:20]: # 最多打印 20 个不匹配的
        #             print(f"      - {mismatched_name}")
        #         if len(mismatched_names) > 20: print("        ...")
        #         print(now_time() + "This might indicate an issue with the freezing logic or unexpected model structure/naming.")
        # # -------------------------------- MoE Replacement --------------------------------

        model.init_prompt(pre_model, post_att, nuser, nitem, lora_layer_nums, num_heads, pad_token_id)
        # Finally, freeze all parameters except for LoRA parameters.
        for name, param in model.named_parameters():
            if ("lora_text_right" in name or
                "lora_text_left" in name or 
                "lora_base_right" in name or 
                "lora_base_left" in name or 
                "lora_img_right" in name or 
                "lora_img_left" in name or 
                "lora_gate_generator" in name or
                "rec" in name or
                "att" in name):
                param.requires_grad = True
                print(now_time() + f"Trainable parameter: {name}")
            else:
                param.requires_grad = False
        return model

    def init_prompt(self, pre_model, post_att, nuser, nitem, lora_nums, num_heads, pad_token_id):
        self.src_len = 2
        self.post_att = post_att
        self.lora_nums = lora_nums
        self.pad_token_id = pad_token_id
        emsize = self.transformer.wte.weight.size(1)  # 768

        # Load the best saved model.
        with open(pre_model, 'rb') as f:
            self.pre_model = torch.load(f)

        self.rec = MLP(emsize)

        # bert_model_name_teacher = '/hpc2hdd/home/hchen763/jhaidata/local_model/bert-base-uncased'
        # bert_model_name_student = 'google/bert_uncased_L-2_H-128_A-2'
        # # 教师模型
        # print(now_time() + f'Loading teacher BERT model: {bert_model_name_teacher}')
        # self.bert_tokenizer_teacher = BertTokenizer.from_pretrained(bert_model_name_teacher)
        # self.bert_teacher = BertModel.from_pretrained(bert_model_name_teacher)
        # self.bert_teacher.eval()
        # # 学生模型
        # print(now_time() + f'Loading student Transformer model')
        # self.bert_student = nn.Transformer(
        #     d_model=768,
        #     nhead=12,
        #     num_encoder_layers=1,
        #     num_decoder_layers=1,
        # )
        # self.bert_student.decoder_cls_token = nn.Parameter(torch.randn(1, 768))
        # # 评分头
        # print(now_time() + 'Initializing trainable rating head MLP...')
        # initrange = 0.1
        # # bert_hidden_size = self.bert_student.config.hidden_size
        # bert_hidden_size = 768
        # self.rating_head_mlp = nn.Linear(bert_hidden_size, 1)
        # self.rating_head_mlp.weight.data.uniform_(-initrange, initrange)
        # self.rating_head_mlp.bias.data.zero_()
        # # 蒸馏损失计算
        # self.distil_criterion = torch.nn.MSELoss()

        self.att = nn.MultiheadAttention(emsize, num_heads, dropout=0.2, batch_first=True)
        self.ui2emsize = nn.Linear(self.pre_model.user_embeddings.weight.size(1), emsize, bias=True)

    def _forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            lora_nums: Optional[torch.LongTensor] = None,
            last_token_index: Optional[torch.LongTensor] = None,
            rating_prediction: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            raw_text = None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            lora_nums=lora_nums,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        distillation_loss = None
        if rating_prediction:
            if self.post_att:
                att_hidden_states, _ = self.att(hidden_states, hidden_states, hidden_states)
                rec_hidden_states = att_hidden_states[
                    torch.arange(att_hidden_states.shape[0], device=att_hidden_states.device), last_token_index]
            else:
                rec_hidden_states = hidden_states[
                    torch.arange(hidden_states.shape[0], device=hidden_states.device), last_token_index]
            rating = self.rec(rec_hidden_states)
            # if attention_mask is None:
            #     print(now_time() + "Warning: attention_mask is None during rating prediction. BERT might behave unexpectedly.")
            
            # # 学生输入
            # bert_input_embeds_student = hidden_states
            # src = bert_input_embeds_student.permute(1, 0, 2)
            # current_batch_size = bert_input_embeds_student.shape[0]
            # tgt = self.bert_student.decoder_cls_token.unsqueeze(1).repeat(1, current_batch_size, 1)
            # if attention_mask is not None:
            #     # attention_mask from HuggingFace is typically (batch_size, src_seq_len)
            #     # where 0 indicates padding.
            #     # src_key_padding_mask for nn.Transformer (with batch_first=False for encoder input)
            #     # expects (batch_size, src_seq_len) where True indicates padding.
            #     transformer_padding_mask = (attention_mask == 0)
            #     # transformer_padding_mask 会是布尔张量，shape (batch_size, src_seq_len)
            # else:
            #     # If no attention_mask is provided, then no padding.
            #     transformer_padding_mask = None
            # bert_outputs_student = self.bert_student(
            #     src=src,
            #     tgt=tgt,
            #     src_key_padding_mask=transformer_padding_mask,
            #     memory_key_padding_mask=transformer_padding_mask
            # )
            # # 提取 Student BERT 输出的第一个 token ([CLS] 位置) 的隐藏状态
            # bert_cls_embedding_student = bert_outputs_student.squeeze(0)
            
            # 教师输入
            if raw_text is not None:
                # bert_input_embeds_teacher = self.bert_tokenizer_teacher(
                #         raw_text,
                #         padding='max_length',        # Pad to BERT max length
                #         truncation=True,             # Truncate if longer
                #         max_length=32,
                #         return_tensors='pt'          # Return PyTorch tensors
                #     )
                # input_ids_teacher = bert_input_embeds_teacher['input_ids'].to(hidden_states.device)
                # attention_mask_teacher = bert_input_embeds_teacher['attention_mask'].to(hidden_states.device)
                
                # with torch.no_grad():
                #     bert_embedding_output_teacher = self.bert_teacher.embeddings(input_ids=input_ids_teacher)
                #     bert_outputs_teacher = self.bert_teacher(
                #         inputs_embeds=bert_embedding_output_teacher,
                #         attention_mask=attention_mask_teacher, # 使用 GPT-2 对应的注意力掩码
                #         return_dict=True
                #     )
                # # 提取 Teacher BERT 输出的第一个 token ([CLS] 位置) 的隐藏状态
                # bert_cls_embedding_teacher = bert_outputs_teacher.last_hidden_state[:, 0, :] # 取第一个位置
                # # 计算蒸馏损失     
                # distillation_loss = self.distil_criterion(bert_cls_embedding_student, bert_cls_embedding_teacher.detach())
                distillation_loss = torch.tensor(0.0, requires_grad=False)

            # # 通过新的可训练 MLP 评分头得到评分
            # # rating_head_mlp 是可训练的，所以这里不需要 no_grad
            # rating = self.rating_head_mlp(bert_cls_embedding_student) # shape: [batch_size, 1]
            # rating = torch.squeeze(rating, dim=-1) # 压缩维度得到 shape: [batch_size,]
        else:
            rating = None

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        if distillation_loss is not None:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            ), rating, distillation_loss
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            ), rating
            

    def forward(self, user, item, text, mask, item_embedding=None, rating_prediction=True, ignore_index=-100, raw_text=None):
        device = user.device

        if rating_prediction:
            # 取最后一个非pad的token
            last_token_index = torch.eq(text, self.pad_token_id).int().argmax(-1) - 1
            last_token_index = last_token_index % text.shape[-1]
        else:
            last_token_index = None

        # embeddings
        u_src = self.pre_model.user_embeddings(user)  # (batch_size, emsize)
        u_src = self.ui2emsize(u_src)
        i_src = self.pre_model.item_embeddings(item)  # (batch_size, emsize)
        i_src = self.ui2emsize(i_src)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)

        # 处理图像嵌入
        processed_item_embedding = None
        if item_embedding is not None:
            # Ensure item_embedding's dimension matches the model's emsize
            # breakpoint()
            if item_embedding.size(-1) != i_src.size(-1):
                # Add a projection layer if it doesn't exist
                if not hasattr(self, 'item_embedding_proj'):
                    self.item_embedding_proj = nn.Linear(item_embedding.size(-1), i_src.size(-1)).to(item_embedding.device)
                processed_item_embedding = self.item_embedding_proj(item_embedding)
            else:
                processed_item_embedding = item_embedding
        
        # src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)
        # src = w_src  # (batch_size, total_len, emsize)
        u_i_src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1)], 1)  # (batch_size, 2, emsize)
        src = [w_src, u_i_src, processed_item_embedding]
        ################ 调试信息 #####################
        # print(f"src: {len(src)}")
        if mask is None:
            # print("Enter first if branch..........")
            # auto-regressive generation
            return self._forward(inputs_embeds=src, lora_nums=self.lora_nums, last_token_index=last_token_index,
                                 rating_prediction=rating_prediction, raw_text=raw_text)
        else:
            # training
            # input padding
            # pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            # pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)
            pad_input = mask
            # prediction for training
            # pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text,
                                     torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            # prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)
            prediction = pred_right

            return self._forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction,
                                 lora_nums=self.lora_nums, last_token_index=last_token_index,
                                 rating_prediction=rating_prediction, raw_text=raw_text)


class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class FeaturePrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, context, explanation, exp_mask, ignore_index=-100):
        device = context.device
        text = torch.cat([context, explanation], 1)  # (batch_size, total_len)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)

        if exp_mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones_like(context, dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, exp_mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full_like(context, ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(exp_mask == 1, explanation,
                                     torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class DiscretePromptLearning(FeaturePrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        rating = torch.sum(user * item, 1)  # (batch_size,)
        return rating


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=2):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, llm_hidden_states):  # (batch_size, emsize)
        # ui_cat = torch.cat([user, item], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(llm_hidden_states))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)
        return rating


class UIPromptWithReg:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, use_mf=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        model.init_prompt(nuser, nitem, use_mf)
        return model

    def init_prompt(self, nuser, nitem, use_mf):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        if use_mf:
            self.rec = MF()
        else:
            self.rec = MLP(emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, rating_prediction=True, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if rating_prediction:
            rating = self.rec(u_src, i_src)  # (batch_size,)
        else:
            rating = None
        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src), rating
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(
                device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text,
                                     torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction), rating


class RecReg(UIPromptWithReg, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
