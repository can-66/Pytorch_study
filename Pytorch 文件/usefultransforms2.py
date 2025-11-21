from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

writer=SummaryWriter("logs_2")
img=Image.open(r"data\train\ants_image\28847243_e79fe052cd.jpg")
 
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)

''' 
transforms.Resize() 用于调整图像尺寸，是数据预处理中非常常用的变换。

transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None)
size: 目标尺寸
  单个整数：短边缩放到该值，长边按比例缩放
  trans_resize = transforms.Resize(512)  # 短边缩放到 512
元组 (height, width)：精确调整到指定尺寸
trans_resize = transforms.Resize((512, 512))  # 调整到 512x512
interpolation: 插值方法（默认双线性插值）



支持的输入类型
PIL.Image：最常见，从文件读取的图片对象
torch.Tensor：已经是张量的图片数据

from PIL import Image
from torchvision import transforms

# PIL Image 输入
img = Image.open("image.jpg")  # PIL.Image.Image
print(type(img))  # <class 'PIL.Image.Image'>

resize = transforms.Resize(256)
resized_img = resize(img)  # 返回 PIL.Image
print(resized_img.size)    # (width, height)


mport torch
from torchvision import transforms

# Tensor 输入（CHW 格式）
tensor_img = torch.randn(3, 224, 224)  # torch.Tensor
print(tensor_img.shape)  # torch.Size([3, 224, 224])

resize = transforms.Resize(256)
resized_tensor = resize(tensor_img)  # 返回 torch.Tensor
print(resized_tensor.shape)  # torch.Size([3, 256, 256])

'''
print(img.size)#(500, 375)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize2=trans_resize(img_tensor)
print(img_resize)#<PIL.Image.Image image mode=RGB size=512x512 at 0x209BC217860>
writer.add_image("Resize",img_resize2,0)



#Compose的使用

''' 
transforms.Compose() 用于将多个图像变换组合成一个变换管道，按顺序依次执行。
transforms.Compose([transform1, transform2, transform3, ...])
1. 基本用法
from torchvision import transforms
from PIL import Image

# 定义变换管道
transform = transforms.Compose([
    transforms.Resize(256),                    # 1. 短边缩放到 256
    transforms.CenterCrop(224),               # 2. 中心裁剪到 224x224
    transforms.ToTensor(),                    # 3. 转成 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # 4. 标准化
])

# 应用变换
img = Image.open("data/train/ants_image/0013035.jpg")
tensor_img = transform(img)
print("最终形状:", tensor_img.shape)  # torch.Size([3, 224, 224])


2. 训练时的数据增强
# 训练时的变换（包含随机性）
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),               # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 验证时的变换（无随机性）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),               # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


换按顺序执行，前一个的输出是后一个的输入
通常 ToTensor() 放在最后几个位置
Normalize() 必须在 ToTensor() 之后
随机变换（如 RandomCrop）只在训练时使用
'''
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resieze",img_resize_2,1)


#RandomCrop
'''
transforms.RandomCrop() 用于随机裁剪图像，是数据增强中常用的变换，可以增加模型的泛化能力。
基本语法
transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
常用参数
size: 裁剪尺寸
单个整数：正方形裁剪 (size, size)
元组 (height, width)：矩形裁剪
padding: 填充像素数（可选）
pad_if_needed: 图像小于裁剪尺寸时是否填充
fill: 填充值（默认 0）
padding_mode: 填充模式

使用示例
from torchvision import transforms
from PIL import Image

# 随机裁剪到 224x224
random_crop = transforms.RandomCrop(224)

# 组合使用
transform = transforms.Compose([
    transforms.Resize(256),        # 先放大
    transforms.RandomCrop(224),    # 再随机裁剪
    transforms.ToTensor()
])
. 使用 pad_if_needed（自动填充）
# 自动填充小图片
trans_random = transforms.RandomCrop(512, pad_if_needed=True)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])

img = Image.open("data/train/ants_image/0013035.jpg")
cropped_img = transform(img)
print("裁剪后形状:", cropped_img.shape)  # torch.Size([3, 224, 224])

 '''
trans_random=transforms.RandomCrop(512)
trans_compose_2=transforms.Compose([transforms.Resize(600),trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)



writer.close()
