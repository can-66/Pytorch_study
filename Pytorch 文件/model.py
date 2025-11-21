#创建网络模型
from torch import nn
import torch


class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    
    def forward(self,x):
        return self.model(x)

if __name__ == "__main__":
    tuidui=TuiDui()
    input=torch.ones((64,3,32,32))
    #输入为batch_size,channel,height,width
    output=tuidui(input)
    print(output.shape)#torch.Size([64, 10])
    #返回64行数据，每一行有10个