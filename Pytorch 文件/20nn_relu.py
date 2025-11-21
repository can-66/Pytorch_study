import torch
from torch.utils.data import DataLoader
import torchvision
input=torch.tensor([[1,-0.5],
                    [-1,3]])
print(input.shape)#torch.Size([2, 2])
input=torch.reshape(input,(-1,1,2,2))
print(input.shape)#torch.Size([1, 1, 2, 2])

import torch.nn as nn
class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1=nn.ReLU()
        self.sigmoid1=nn.Sigmoid()
        '''
模块式：
torch.nn.ReLU(inplace=False)
参数
inplace: 是否原地覆盖输入，省内存；默认 False。原地可能影响反向传播，通常在 Conv/BN 后使用较安全。
函数式：
torch.nn.functional.relu(input, inplace=False)
参数
input: 输入张量（一般为浮点，支持 CPU/GPU）
inplace: 同上

import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[-1.0, 0.5, 2.0],
                  [ 3.0,-2.0, 0.0]])

# 1) 模块式（常见于 nn.Sequential）
act = nn.ReLU(inplace=False)
y1 = act(x)  # tensor([[0.0, 0.5, 2.0],[3.0, 0.0, 0.0]])

# 2) 函数式
y2 = F.relu(x, inplace=False)

# 3) 在网络中使用
net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),   # 省内存
    nn.MaxPool2d(2)
)
        
        
         '''
    def forward(self,x):
        # x=self.relu1(x)
        # return x
        x=self.sigmoid1(x)
        return x

tuidui=TuiDui()
# output=tuidui(input)
# print(output)
# # tensor([[[[1., 0.],
# #           [0., 3.]]]])

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(test_data,batch_size=64)
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("relu_logs")
step=0
for data in dataloader:
    img,target=data
    writer.add_images("input",img,step)
    output=tuidui(img)
    writer.add_images("output",output,step)
    step=step+1
writer.close()
