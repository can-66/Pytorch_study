from torch import nn
import torch


class TuDui(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self,input):
        output=input+1
        return output



tudui=TuDui()#实例化模型

x=torch.tensor(1.0)# 创建一个标量张量作为输入
output=tudui(x)#把输入喂给模型（内部会调用 forward（

''' 因为 nn.Module 重载了 __call__ 方法。你写 output = model(x) 时，实际上调用的是 nn.Module.__call__，
它内部会做一堆通用处理（如前后钩子、混合精度、参数/缓冲区管理等），最后再自动调用你自定义的 forward(x)。
所以你不用手动 model.forward(x)，直接 model(x) 就会自动走 forward。'''
print(output)# 打印模型输出



