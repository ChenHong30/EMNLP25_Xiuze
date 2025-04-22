import torch
from torch import nn
import inspect

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
        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_normal_(self.lora_right)  # , a=math.sqrt(5)
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))

    def forward(self, inputs):
        # stack = inspect.stack()

        # # 打印调用栈的前五层信息，格式化输出
        # print("="*50)
        # print("调用栈信息：")
        # for i in range(min(5, len(stack))):  # 至少打印五层，但如果栈帧少于五层，则打印所有栈帧
        #     caller = stack[i]
        #     print(f"层 {i+1}:")
        #     print(f"  文件: {caller.filename}")
        #     print(f"  行号: {caller.lineno}")
        #     print(f"  函数: {caller.function}")
        #     print("-"*50)

        # print("="*50)
        

        w_inputs, u_i_inputs = inputs
        x = self.linear(w_inputs)
        y = u_i_inputs @ self.lora_right @ self.lora_left
        y = torch.mean(y, dim=1).unsqueeze(1).repeat((1, x.shape[1], 1))

        return x + y
