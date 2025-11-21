import torch

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

kernel=torch.tensor([[1,2,1],
                    [0,1,0],
                    [2,1,0]])

print(input.shape)#torch.Size([5, 5])
print(kernel.shape)#torch.Size([3, 3])

import torch.nn.functional as F

''' 
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) 
input: [N, C_in, H, W]
N：batch size（批大小），一次送入卷积的样本数
C_in：输入通道数，例如灰度图=1，RGB=3，特征图可能更多
H：输入特征图的高度（像素/网格大小）
W：输入特征图的宽度
示例：
一张 RGB 图片（32×32），单张：形状 [1, 3, 32, 32]
64 张 RGB 图片打成批：形状 [64, 3, 32, 32]
灰度图批（28×28，batch=32）：形状 [32, 1, 28, 28]
conv2d 必须接收 4 维张量（NCHW）

灰度图（grayscale）指只有明暗信息、没有颜色的图像，只有1个通道。
每个像素用一个强度值表示（常见为 0–255，0=黑，255=白）。
在深度学习中，灰度图张量形状为 [1, H, W]；批量则是 [N, 1, H, W]。

weight 形状 [C_out, C_in/groups, kH, kW]
C_out: 输出通道数（卷积核个数）
C_in/groups: 每个卷积核关联的输入通道数（被分组后每组的通道数）
kH, kW: 卷积核高、宽

bias 形状 [C_out] 或 None
每个输出通道一个偏置，加在卷积结果上；不需要就设 None

stride 步幅（int 或 (sh, sw)）
卷积窗口每次移动的步长
越大，特征图下采样越多（尺寸更小）
例：stride=2，H/W 大约减半（取决于 paddi

padding 填充（int 或 (ph, pw)）
在输入边缘补零（或其他模式）以控制输出尺寸
常见“same”近似：当 kernel_size=3 且 stride=1，用 padding=1 保持尺寸不变
简单说：padding 就是在输入特征图的四边补像素。补多少完全由你传入的 padding 决定。
整数 p：上下左右各补 p 个像素（对称补）
元组 (ph, pw)：上下各补 ph，左右各补 pw
在 F.conv2d 里，补的像素默认是 0（零填充）
例：
padding=1（整数）：上下左右各补1圈像素
原图 5×5 → 补后 7×7（因为左右各+1、上下各+1）
等价于在四边加一圈“边框”（默认值为0）
padding=(ph, pw)（元组）：高方向补 ph，宽方向补 pw
比如 padding=(2,1)：上下各补2，左右各补1
原图 5×5 → 补后 (5+2+2)×(5+1+1) = 9×7

dilation 空洞系数
在卷积核内部插入空洞，扩大感受野而不增参数量
例：kernel=3, dilation=2 等效感受野为 5（跨步取样）

groups 组卷积
将输入通道分为 groups 组，每组单独做卷积再拼回
普通卷积：groups=1（每个核看见全部 C_in）
depthwise 卷积：groups=C_in 且 C_out=C_in（每个输入通道各自用一个3×3核，极大降耗）
pointwise（1×1 卷积）常与 depthwise 串联成 DW-Conv（MobileNet 风格）
'''

input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))
output=F.conv2d(input,kernel,stride=1)

print(output)
# tensor([[[[10, 12, 12],
#           [18, 16, 16],
#           [13,  9,  3]]]])

output3=F.conv2d(input,kernel,stride=1,padding=1)
print(output3)
# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])




