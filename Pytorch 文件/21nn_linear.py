import torchvision
import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(test_data,batch_size=64,drop_last=True)
''' 
Linear 层定义与签名
模块式：nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
函数式：F.linear(input, weight, bias=None)
作用：执行线性变换 y = xW^T + b，其中 W 是权重矩阵，b 是偏置向量。

n_features: 输入特征数（输入张量最后一维的大小）
out_features: 输出特征数（输出张量最后一维的大小）
bias: 是否使用偏置，默认 True
device/dtype: 权重和偏置的设备/数据类型
输入输出形状
输入: [*, in_features]，其中 * 可以是任意维度
输出: [*, out_features]，保持除最后一维外的所有维度
import torch
import torch.nn as nn

# 1) 基本用法
linear = nn.Linear(in_features=784, out_features=128)
x = torch.randn(32, 784)  # [batch_size, features]
y = linear(x)             # [32, 128]

# 2) 无偏置版本
linear_no_bias = nn.Linear(784, 128, bias=False)

# 3) 多维输入（保持前面维度）
x_3d = torch.randn(10, 20, 784)  # [batch, seq_len, features]
y_3d = linear(x_3d)              # [10, 20, 128]

# 4) 在神经网络中使用
model = nn.Sequential(
    nn.Flatten(),           # 展平为 [N, features]
    nn.Linear(784, 256),    # 第一层
    nn.ReLU(),
    nn.Linear(256, 128),    # 第二层
    nn.ReLU(),
    nn.Linear(128, 10),     # 输出层
)

'''
class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output


tuidui=TuiDui()

for data in dataloader:
    imgs,targets=data
    # print(imgs.shape)#torch.Size([64, 3, 32, 32])
    # output=torch.reshape(imgs,(1,1,1,-1))
    # print(output.shape)#torch.Size([1, 1, 1, 196608])
    output=torch.flatten(imgs)
    print(output.shape)#torch.Size([196608])
    '''
torch.flatten 用法
函数式：torch.flatten(input, start_dim=0, end_dim=-1)
方法式：tensor.flatten(start_dim=0, end_dim=-1)
作用：将指定维度范围内的张量展平为一维，常用于将多维特征图转换为线性层输入。
核心参数
input: 输入张量
start_dim: 开始展平的维度（包含），默认 0
end_dim: 结束展平的维度（包含），默认 -1（最后一维）
    
    import torch

# 1) 完全展平（默认行为）
x = torch.randn(2, 3, 4, 5)
y = torch.flatten(x)  # 展平所有维度
print(y.shape)        # torch.Size([120])  # 2*3*4*5=120

# 2) 从指定维度开始展平
x = torch.randn(2, 3, 4, 5)
y = torch.flatten(x, start_dim=1)  # 从第1维开始展平
print(y.shape)        # torch.Size([2, 60])  # 保持第0维，展平3*4*5=60

# 3) 展平指定范围
x = torch.randn(2, 3, 4, 5)
y = torch.flatten(x, start_dim=1, end_dim=2)  # 只展平第1、2维
print(y.shape)        # torch.Size([2, 12, 5])  # 3*4=12

# 4) 方法式调用
x = torch.randn(2, 3, 4, 5)
y = x.flatten(1)      # 从第1维开始展平
print(y.shape)        # torch.Size([2, 60])
     '''
    output=tuidui(output)
    print(output.shape)#torch.Size([1, 1, 1, 10])



