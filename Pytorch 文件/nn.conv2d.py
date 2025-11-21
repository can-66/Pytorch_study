''' 
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
in_channels（必填）
输入通道数。灰度=1，RGB=3，或上一层特征图通道数。

out_channels（必填）
卷积核个数=输出通道数。越大特征表达能力越强、参数越多。

in_channels：输入特征图的通道数。就像“输入有几层信息”。例：
灰度图=1（只有明暗）
彩色 RGB=3（R/G/B 三层）
上一层卷积输出了 64 个通道，那这一层的 in_channels 就是 64
out_channels：这一层要学多少个卷积核（滤波器），每个卷积核都会产出一个通道；所以 out_channels 就是“输出有几层信息”。例：
设 out_channels=128，表示这一层学 128 个卷积核，输出 128 个通道
直观理解
输入张量形状：[N, in_channels, H, W]
卷积核权重形状：[out_channels, in_channels, kH, kW]
输出张量形状：[N, out_channels, H_out, W_out]
小例子（RGB 输入 → 64 通道特征）
总结
in_channels 由“输入的通道数”决定（接上游）
out_channels 由“这一层想提取多少组特征”决定（你来选，越大容量越强但更耗算力）





kernel_size（必填）
卷积核大小。int 或 (kH, kW)。常用 3、5、7。3×3 最常见。

stride=1
步幅。int 或 (sH, sW)。控制下采样倍数。2 表示尺寸约减半。

padding=0
边缘填充像素。int 或 (pH, pW)。
k=3 且 stride=1 时，padding=1 可基本保持尺寸不变（“same”近似）。

dilation=1
空洞（膨胀）卷积系数。>1 可扩大感受野，不增参数量；常配合较大 padding。
groups=1
组卷积。1 为普通卷积；
depthwise：groups=in_channels 且 out_channels=in_channels（每通道单独卷积）；
pointwise 常指 kernel_size=1 的 1×1 卷积（不通过该参数控制）。
bias=True
是否为每个输出通道加偏置项。搭配 BatchNorm 可设为 False。

padding_mode='zeros'
填充模式：'zeros'（默认）、'reflect'、'replicate'、'circular'。
device=None, dtype=None
张量所在设备与数据类型，一般用 model.to(device)/.half() 统一管理。
输出形状（单维度公式）
out = floor((in + 2padding - dilat
'''
import torchvision
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
img,target=test_data[0]

#print(type(test_data[0]))
print(img.size())#torch.Size([3, 32, 32])
print(target)#3

dataloader=DataLoader(test_data,batch_size=64)

class TuiDui(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

tuidui=TuiDui()
print(tuidui)
# TuiDui(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )

writer=SummaryWriter("conv_logs")
step=0
for data in dataloader:
    img1,target1=data
    output=tuidui(img1)
    # print(img1.shape)#torch.Size([64, 3, 32, 32])
    # print(output.shape)#torch.Size([64, 6, 30, 30])，卷积后的30*30

    writer.add_images("input",img1,step)
    '''
    因为卷积输出的通道数是 6，而 TensorBoard 的图片要求通道数只能是 1（灰度）或 3（RGB）。
    你的 Conv2d(3→6, k=3, p=0) 输出形状是 [B, 6, 30, 30]，不能直接当图片写入。
    于是把 6 个通道按每 3 个通道拼成一张 RGB 图，变成 [-1, 3, 30, 30]，
    1 在 reshape 里表示“自动推断该维度”，根据总元素个数和其他维度计算出来。只能有一个 -1。
     结合你这行：
    这样就能用 add_images 可视化了：
    
    
     '''

    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("Output",output,step)

    step=step+1

writer.close()
