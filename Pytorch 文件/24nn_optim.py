''' 
核心用法（通用三步）
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)  # 1) 定义优化器

for imgs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)   # 2) 清梯度（推荐 set_to_none=True 更省显存/更快）
    logits = model(imgs)
    loss = criterion(logits, targets)
    loss.backward()                         # 3) 反向传播，计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可选：梯度裁剪
    optimizer.step()                        # 4) 用梯度更新参数


常用优化器与关键参数
SGD
torch.optim.SGD(params, lr, momentum=0.0, weight_decay=0.0, nesterov=False)
lr: 学习率；momentum: 动量；nesterov: Nesterov 动量；weight_decay: L2 正则（权重衰减）
Adam
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False)
betas: 一阶/二阶动量衰减；eps: 数值稳定项；amsgrad: 可选变体

AdamW（推荐，decoupled weight decay）
torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

参数组（不同层用不同超参）
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 3e-4, "weight_decay": 0.01},
    {"params": model.head.parameters(),     "lr": 1e-3, "weight_decay": 0.0},  # 比如不衰减偏置/Norm
])
不衰减偏置与归一化参数（常见做法）
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or name.endswith("bias"):  # BN/LayerNorm/偏置
        no_decay.append(p)
    else:
        decay.append(p)
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)


学习率调度器（可选）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train_one_epoch(...)
    scheduler.step()  # 每 epoch 调整学习率

典型训练完整示例
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

model.train()
for epoch in range(10):
    for imgs, targets in dataloader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

常见注意
每次更新前一定要清梯度：zero_grad，否则会累加。
推理/验证不需要梯度：model.eval() + with torch.no_grad(): ...
学习率是最敏感超参；爆炸梯度可用梯度裁剪或降低 lr。
与 CrossEntropyLoss 一起用时传“logits”，不要手动 softmax。
'''

from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import torch

class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
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



test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(test_data,batch_size=64)

loss=nn.CrossEntropyLoss()
tuidui=TuiDui()
optim=torch.optim.SGD(tuidui.parameters(),lr=0.01,momentum=0.9)
#lr学习率不宜过大，否则会震荡，momentum动量不宜过大，否则会震荡，lr一般先从大的开始，然后逐渐减小
for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=tuidui(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()#对梯度进行更新，执行完后回到115行，继续执行，对data参数进行更新
        running_loss+=result_loss
    print(running_loss/len(dataloader))
#tensor(2.2488, grad_fn=<DivBackward0>)
# tensor(1.9806, grad_fn=<DivBackward0>)
# tensor(1.7864, grad_fn=<DivBackward0>)
# tensor(1.6103, grad_fn=<DivBackward0>)
