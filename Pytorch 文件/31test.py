import torchvision
from model import TuiDui
from PIL import Image
import torch

# 设备统一管理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_path='./imgs/volun.jpg'
image=Image.open(image_path)
print(image)
#<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=4032x3024 at 0x26832414BA8>
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

image=transform(image)
print(image.shape)
#torch.Size([3, 32, 32])
import torch
model= torch.load("tuidui_device_1.path")
print(model)

image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    image=image.to(device)
    output=model(image)
    


print(output)
# tensor([[ 4.6230e-01,  1.1736e+00, -4.5993e-01,  1.2369e-01, -1.0935e+00,
#           9.8792e-04, -1.1381e+00, -6.3119e-01,  9.5559e-01,  3.0099e-01]],
#        device='cuda:0')
#这里若无image=torch.reshape(image,(1,3,32,32))
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight [32, 3, 5, 5], but got 3-dimensional input of size [3, 32, 32] instead 


print(output.argmax(1))