from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

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



test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(test_data,batch_size=1)

loss=nn.CrossEntropyLoss()
tuidui=TuiDui()
for data in dataloader:
    imgs,targets=data
    outputs=tuidui(imgs)
    # print(outputs)
    # print(targets)
    # ''' 
    # tensor([[-0.1060, -0.1272,  0.0566, -0.0177, -0.0518, -0.0703, -0.1107, -0.0815,
    #      -0.0544,  0.1103]], grad_fn=<AddmmBackward0>)
    # tensor([4])
    # '''
    result_loss=loss(outputs,targets)
    print(result_loss)
    result_loss.backward()
    print(result_loss)
    #tensor(2.4013, grad_fn=<NllLossBackward0>)


''' 
grad 是张量的“梯度”（对标量损失的偏导），训练时用来更新参数。
你看到的 grad_fn=<AddmmBackward0> 不是梯度本身，而是“这个张量是由哪些算子产生的”的记录，用于反向传播。真正的梯度存放在叶子张量（模型参数等）的 .grad 里。


outputs = tuidui(imgs) 是 logits，打印出来常带 grad_fn=...，表示它参与了计算图，可反向。
loss = CrossEntropyLoss(outputs, targets) 返回一个标量，同样带 grad_fn。
调 loss.backward() 后，PyTorch 会沿着 grad_fn 记录的计算图自动求导，把每个“叶子参数”（model.parameters()）的梯度写入 param.grad。

loss.backward()只“计算梯度并写入param.grad”，并不会改参数的数值。参数更新需要再调用optimizer.step()；且在更新前要optimizer.zero_grad()清梯度，否则会累加。

最小训练三步
optimizer = torch.optim.SGD(tuidui.parameters(), lr=1e-2)

optimizer.zero_grad()          # 1. 清梯度
result_loss.backward()         # 2. 反向传播，填充 param.grad
optimizer.step()               # 3. 用梯度更新参数
想“看到变化”，可以打印某个权重或其梯度范数：
w = next(tuidui.parameters())
print('before', w.view(-1)[0].item())
optimizer.zero_grad()
result_loss.backward()
print('grad_norm', w.grad.norm().item())
optimizer.step()
print('after ', w.view(-1)[0].item())
'''
