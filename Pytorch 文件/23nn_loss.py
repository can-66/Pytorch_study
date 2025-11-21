import torch
from torch.nn import L1Loss
from torch import nn

inputs=torch.tensor([1,2,3],dtype=torch.float32)
targets=torch.tensor([1,2,5],dtype=torch.float32)

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss=L1Loss()
result=loss(inputs,targets)

print(result)#tensor(0.6667)  2/3
'''
torch.nn.L1Loss(reduction='mean')；函数式：F.l1_loss(input, target, reduction='mean')
计算逐元素绝对误差：|input - target|，也叫 MAE。
reduction:
'none': 返回与输入同形状的逐元素损失
'mean'(默认): 对全部元素求均值
'sum': 对全部元素求和

输入要求
input 和 target 形状相同或可广播到同形状
dtype 必须是浮点或复数（最常见 float32）；Long 会报错
device 一致（都在 CPU 或同一块 GPU）
'''

loss_mse=nn.MSELoss()
result_mse=loss_mse(inputs,targets)
print(result_mse)#tensor(1.3333),差的平方的均值

''' 
模块式：nn.MSELoss(reduction='mean')
函数式：F.mse_loss(input, target, reduction='mean')
含义：逐元素平方误差 (input - target)^2，再按 reduction 聚合。
参数
reduction:
'none': 不聚合，返回与输入同形状的损失
'mean'(默认): 对所有元素求平均
'sum': 对所有元素求和
输入要求
input 和 target 形状相同或可广播到同形状
dtype 为浮点或复数（常用 float32/float16/bfloat16）；Long 会报错
device 一致（都在 CPU 或同一块 GPU）
常见用法

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) 基础
inp = torch.tensor([1., 2., 3.])
tgt = torch.tensor([1., 2., 5.])
mse = nn.MSELoss(reduction='mean')
print(mse(inp, tgt))            # tensor(1.3333) = ((0^2 + 0^2 + 2^2)/3)

# 2) 不聚合，按样本再聚合
pred = torch.randn(4, 10)
gt   = torch.randn(4, 10)
loss_per_elem = F.mse_loss(pred, gt, reduction='none')   # [4,10]
loss_per_sample = loss_per_elem.mean(dim=1)              # [4]
loss = loss_per_sample.mean()                            # 标量

# 3) 多维张量（如 NCHW）
pred = torch.randn(8, 3, 32, 32)
gt   = torch.randn(8, 3, 32, 32)
loss = F.mse_loss(pred, gt)       # 对全部元素求均值
适用场景
回归任务（连续值预测）：如坐标、深度、重建误差、自动编码器等
'''


x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
print(x.shape)#torch.Size([1, 3])
''' 
这是把 x 变成交叉熵需要的 logits 形状 [N, C]：这里 N=1（1 个样本），C=3（3 个类别）。
nn.CrossEntropyLoss 要求：
输入 x 形状 [N, C]（或 [N, C, H, W]），为未归一化分数 logits（不要先 softmax）。
目标 y 形状 [N]，类型 Long，取值在 [0, C-1]

'''
loss_cross=nn.CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)
#tensor(1.1019)

''' 
nn.CrossEntropyLoss = LogSoftmax + NLLLoss 的组合。直接接收“未归一化分数”（logits），内部完成 softmax 与取负对数。

签名与常用参数
nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)
weight: Tensor[C]，类别权重（处理类别不均衡）。
ignore_index: 目标中等于该值的样本会被忽略（常用于分割中的填充标签）。
reduction: 'none' | 'mean' | 'sum'。
label_smoothing: 标签平滑系数 ∈ [0,1)（缓解过拟合、提升鲁棒性）。

输入/目标要求（非常重要）
输入 input（logits）：形状 [N, C] 或 [N, C, d1, d2, ...]，dtype 浮点（float32 等）。
目标 target（类别索引）：形状 [N] 或 [N, d1, d2, ...]，dtype Long，取值范围 0..C-1。
目标不是 one-hot！若你是概率标签/one-hot，需改用 KLDivLoss 或先取 argmax（不推荐）或用 soft labels 的 CrossEntropy 替代方案。
'''

