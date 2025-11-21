'''
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
return_indices=False, ceil_mode=False)
是二维最大池化层，对输入的特征图按窗口取最大值，常用于下采样。
• kernel_size: 窗口大小。可为单值(int)或二元组(h, w)。
• stride: 步幅。为 None 时默认等于 kernel_size。可为单值或二元组。
• padding: 在输入四周补零的像素数。可为单值或二元组。
• dilation: 膨胀系数，扩大感受野，实际窗口感受野变为 effective_kernel = (k - 1) * dilation + 1。一般保持为 1。也被称为空洞卷积
• return_indices: 若为 True，返回最大值的索引（用于 MaxUnpool2d 反池化）。
• ceil_mode: 输出尺寸计算用 ceil 而非 floor，会让输出可能多 1。
'''

'''
torch.Tensor 是 PyTorch 的张量类型（多维数组），在 Python 中它是一个类：<class 'torch.Tensor'>。
dtype: 数据类型，如 torch.float32、torch.int64、torch.bool。
device: 存放设备，如 cpu 或 cuda:0。
shape/ndim: 形状与维度数，如 (2,3,4)、ndim=3。
requires_grad: 是否参与自动求导（训练时对可学习参数设为 True）。

import torch

# 1) 标量（0D）
a = torch.tensor(3.14)              # 默认 float32（受全局默认影响）
print(type(a), a.shape, a.ndim)      # <class 'torch.Tensor'>, torch.Size([]), 0

# 2) 向量（1D）
b = torch.tensor([1, 2, 3], dtype=torch.int64)
print(b.dtype, b.shape)              # torch.int64, torch.Size([3])

# 3) 矩阵（2D）
c = torch.tensor([[1., 2.], [3., 4.]])   # 浮点
print(c.dtype, c.shape)                   # torch.float32, torch.Size([2, 2])

# 4) 4D 张量（典型 NCHW：批次×通道×高×宽）
x = torch.randn(8, 3, 224, 224, device='cpu')  # 随机浮点
print(x.shape, x.device)                   # torch.Size([8, 3, 224, 224]) cpu

# 5) 需要梯度的张量（用于训练的参数或中间量）
w = torch.randn(10, 10, requires_grad=True)
y = (w ** 2).sum()
y.backward()                               # 反向传播后 w.grad 有梯度
print(w.grad.shape)                        # torch.Size([10, 10])

 '''



import torch
# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]],dtype=torch.float32)

# print(input.shape)#torch.Size([5, 5]) 

# input=torch.reshape(input,(-1,1,5,5))
# print(input.shape)#torch.Size([1, 1, 5, 5] )
# N 表示批大小（batch size），也就是一次输入到网络中的样本数量。
# 在 NCHW 中，形状 [N, C, H, W] 分别是：批大小、通道数、高、宽。
# 你的例子里 input4d 形状为 (1, 1, 5, 5)，N=1 表示本次只喂入了 1 张图片（或 1 个样本

'''
torch.reshape(input, shape)
方法式：Tensor.reshape(*shape) 或 Tensor.reshape(shape_tuple)
作用：返回“按给定形状重排视图/拷贝”的新张量（元素不变、只变形状）。尽量返回视图，不能视图时返回拷贝。
参数说明
input: torch.Tensor，源张量。
shape:
可以是整型元组，如 (batch, channels, height, width)；
也可用可变参数，如 x.reshape(b, c, h, w)；
允许最多一个 -1，表示该维度由系统自动推断；
其它维度必须是正整数（可为 0 但通常仅在零元素张量时使用），各维乘积需与 input.numel() 一致（若含 -1 则由其满足）。
数据类型、设备、requires_grad 等属性保持不变（只是形状改变）。
 '''
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(test_data,batch_size=64)


import torch.nn as nn
class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1=nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        output=self.maxpool1(x)
        return output


tuidui=TuiDui()
# output=tuidui(input)
# print(output)
# tensor([[[[2., 3.],
#           [5., 1.]]]])

writer=SummaryWriter("maxpool_logs")
step=0
for data in dataloader:
    img,target=data
    writer.add_images("input",img,step)
    output=tuidui(img)
    #池化不改变通道数,注意与前面的卷积不同
    writer.add_images("output",output,step)
    step=step+1

writer.close()


