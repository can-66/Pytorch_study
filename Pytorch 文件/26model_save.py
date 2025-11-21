import torch
import torchvision
from torch import nn

vgg16=torchvision.models.vgg16(pretrained=False)#创建一个未加载预训练权重的 VGG16 模型实例，也就是随机初始化参数
# 保存方式1 模型结构+参数
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2 模型参数（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#陷阱
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(10,10)

    def forward(self,x):
        return self.linear1(x)
    
model=Model()
torch.save(model,"model.pth")


