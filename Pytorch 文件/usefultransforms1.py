from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_pth=r"data\train\ants_image\6743948_2b8c096dda.jpg"#注意不要写pytorch\
img=Image.open(img_pth)
print(img)
#<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=375x500 at 0x1BF6F9EF9E8>

from torchvision import transforms

#ToTensor
writer=SummaryWriter("logs_new")
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)


#Normalize
'''
transforms.Normalize() 用于对图像进行标准化（零均值、单位方差），通常用于预训练模型的输入预处理
 transforms.Normalize(mean, std, inplace=False)
 mean: 各通道的均值，如 [0.485, 0.456, 0.406] (ImageNet RGB)
std: 各通道的标准差，如 [0.229, 0.224, 0.225] (ImageNet RGB)
inplace: 是否就地修改（默认 False）
output[channel] = (input[channel] - mean[channel]) / std[channel]``

1. ImageNet 标准化（最常用）
from torchvision import transforms

# ImageNet 预训练模型的标准参数
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])

# 组合使用
transform = transforms.Compose([
    transforms.ToTensor(),        # 先转 Tensor，范围 [0,1]
    normalize                     # 再标准化
])



'''


print(img_tensor[0][0][0])#tensor(0.6549)
trans_norm=transforms.Normalize([6,3,2],[9,3,5])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])#tensor(0.3098)
writer.add_image("Normalize",img_norm,2)

writer.close()






