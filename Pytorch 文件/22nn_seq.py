'''
torch.nn.Sequential 用来把一串层（模块）按顺序“串起来”，前一层的输出自动作为下一层的输入，省去手写 forward。

基本用法
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*16*16, 10),
)
x = torch.randn(8, 3, 32, 32)
y = model(x)  # [8, 10]

命名子模块（便于访问）
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ("conv", nn.Conv2d(3, 16, 3, padding=1)),
    ("act", nn.ReLU(inplace=True)),
    ("pool", nn.MaxPool2d(2)),
    ("flat", nn.Flatten()),
    ("fc", nn.Linear(16*16*16, 10)),
]))
print(model.fc)  # 通过名字访问


追加/嵌套
# 1) 先空的，再逐层添加
m = nn.Sequential()
m.add_module("conv", nn.Conv2d(3, 8, 3, padding=1))
m.add_module("relu", nn.ReLU())
m.add_module("pool", nn.MaxPool2d(2))

# 2) 顺序块里嵌套另一个 Sequential
block = nn.Sequential(
    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
)
net = nn.Sequential(m, block, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))

'''
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR10 结构（对应配图）：
        # 输入: N×3×32×32
        # 1) Conv(3→32, 5x5, pad=2) → ReLU → 2) MaxPool(2×2)
        # 3) Conv(32→32, 5x5, pad=2) → ReLU → 4) MaxPool(2×2)
        # 5) Conv(32→64, 5x5, pad=2) → ReLU → 6) MaxPool(2×2)
        # 7) Flatten → 8) Linear(1024→64) → ReLU → 9) Linear(64→10)
        self.backbone = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),            # 32×16×16

            Conv2d(32, 32, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),            # 32×8×8

            Conv2d(32, 64, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),            # 64×4×4

            Flatten(),                           # 64*4*4 = 1024
            Linear(64*4*4, 64),
            ReLU(inplace=True),
            Linear(64, 10)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    import torch
    model = TuiDui()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("input:", x.shape)   # torch.Size([8, 3, 32, 32])
    print("output:", y.shape)  # torch.Size([8, 10])

    writer=SummaryWriter("_seq_logs")
    writer.add_graph(model,x)
    #writer.add_graph(model, x): 用输入样   例 x 跑一次前向，追踪并记录 model 的计算图到日志。
    writer.close()
    