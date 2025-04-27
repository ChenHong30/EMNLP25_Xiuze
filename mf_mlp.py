import transformers
import models_forward

transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = models_forward.gpt2_attention_forward
transformers.models.gpt2.modeling_gpt2.GPT2Block.forward = models_forward.gpt2_block_forward
transformers.models.gpt2.modeling_gpt2.GPT2Model.forward = models_forward.gpt2_model_forward

from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch.nn as nn
import torch
import copy
from lora import LoraLinear
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss


class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, lora_nums, lora_dim, num_heads, pad_token_id,
                        **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Replace targeting linear layers with LoRA layers.
        # get target module name
        target_names = []
        lora_layer_nums = [int(n) for n in lora_nums.split(",")]
        lora_layer = ["transformer.h.{}".format(ll) for ll in lora_layer_nums]
        for name, module in model.named_modules():
            lora_layer_bool = sum([lora_layer_name in name for lora_layer_name in lora_layer])
            # if "ln_1" in name or "ln_2" in name, if "mlp.c_fc" in name
            if lora_layer_bool > 0 and "attn.c_attn" in name:
                target_names.append(name)

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

        # Finally, freeze all parameters except for LoRA parameters.
        for name, param in model.named_parameters():
            if "lora_right" in name or "lora_left" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.init_prompt(nuser, nitem, lora_layer_nums, num_heads, pad_token_id)
        return model

    def init_prompt(self, nuser, nitem, lora_nums, num_heads, pad_token_id):
        self.src_len = 2
        self.lora_nums = lora_nums
        self.pad_token_id = pad_token_id
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

        self.rec = MLP(emsize)
        self.att = nn.MultiheadAttention(emsize, num_heads, dropout=0.2, batch_first=True)

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

        if rating_prediction:
            att_hidden_states, _ = self.att(hidden_states, hidden_states, hidden_states)
            rec_hidden_states = att_hidden_states[
                torch.arange(att_hidden_states.shape[0], device=att_hidden_states.device), last_token_index]
            rating = self.rec(rec_hidden_states)
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

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        ), rating

    def forward(self, user, item, text, mask, rating_prediction=True, ignore_index=-100):
        device = user.device

        if rating_prediction:
            # 取最后一个非pad的token
            last_token_index = torch.eq(text, self.pad_token_id).int().argmax(-1) - 1
            last_token_index = last_token_index % text.shape[-1]
        else:
            last_token_index = None

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        # src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)
        # src = w_src  # (batch_size, total_len, emsize)
        u_i_src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1)], 1)  # (batch_size, 2, emsize)
        src = [w_src, u_i_src]
        if mask is None:
            # auto-regressive generation
            return self._forward(inputs_embeds=src, lora_nums=self.lora_nums, last_token_index=last_token_index,
                                 rating_prediction=rating_prediction)
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
                                 rating_prediction=rating_prediction)


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

    def forward(self, user, item):  # (batch_size, emsize)
        ui_cat = torch.cat([user, item], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_cat))  # (batch_size, hidden_size)
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


class TradModel(nn.Module):
    def __init__(self, nuser, nitem, emsize, model_name):
        super(TradModel, self).__init__()
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        if model_name == "mf":
            self.rec = MF()
        else:
            self.rec = MLP(emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item):
        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)

        rating = self.rec(u_src, i_src)  # (batch_size,)

        return rating
