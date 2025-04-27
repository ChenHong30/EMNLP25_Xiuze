import torch
from torch import nn
import inspect
import math

class LoraLinear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoraLinear, self).__init__()

        row, column = weight.shape

        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization)
        self.lora_base_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_normal_(self.lora_base_right)  # , a=math.sqrt(5)
        self.lora_base_left = nn.Parameter(torch.zeros(lora_dim, row))

        self.lora_img_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_normal_(self.lora_img_right)  # , a=math.sqrt(5)
        self.lora_img_left = nn.Parameter(torch.zeros(lora_dim, row))

        self.img_norm = nn.LayerNorm(column)

    def forward(self, inputs):
        w_inputs, u_i_inputs, img_inputs = inputs
        
        x = self.linear(w_inputs)
        
        y_base = u_i_inputs @ self.lora_base_right @ self.lora_base_left
        y_base = torch.mean(y_base, dim=1).unsqueeze(1).repeat((1, x.shape[1], 1))

        img_inputs = self.img_norm(img_inputs)

        y_img = img_inputs @ self.lora_img_right @ self.lora_img_left
        y_img = y_img.unsqueeze(1).repeat((1, x.shape[1], 1))

        return x + y_base + y_img
