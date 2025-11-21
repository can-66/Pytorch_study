from PIL import Image
from torchvision import transforms


''' 
Tensor：PyTorch 的多维数组类型，带自动求导与 GPU 加速能力（类似 NumPy 的 ndarray，但更适合深度学习）
。
transforms.ToTensor()：把图片/数组转成 PyTorch 的 Tensor，并做规范化（HWC→CHW，uint8 0–255 → float32 0–1）

oTensor 的作用与用法
通道顺序：HWC → CHW（TensorBoard/模型期望的输入格式
类型和值域：uint8 → float32，同时除以 255 变成 0–1


'''
img_path='pytorch/data/train/ants_image/0013035.jpg'
img=Image.open(img_path)
#是 PIL 的图像对象
#
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1F38FAA0FD0>
#模式 RGB，尺寸 768x512
# at 0x1F38FAA0FD0 是内存地址
tensor_trans=transforms.ToTensor()#创建一个ToTensor可调用的“变换器”对象
tensor_img=tensor_trans(img)#将img图片转换为tensor_img

''' 
可以，用 OpenCV 读取后转成 PyTorch Tensor。注意：OpenCV 读图是 BGR，需要转成 RGB，再归一化到 0–1。
用 torchvision 的 ToTensor（先转 PIL 更方便复用增强）：
import cv2
from PIL import Image
from torchvision import transforms

img_bgr = cv2.imread("data/train/ants_image/0013035.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
t = transforms.ToTensor()(img_pil)  # [C,H,W], float32, [0,1]

'''

print(tensor_img)




from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("translogs")
writer.add_image("Tensor_img",tensor_img)

''' 
writer.add_image(tag, img_tensor, global_step=None, dataformats='CHW')
mg_tensor 支持 torch.Tensor 或 numpy.ndarray
dataformats 指明维度顺序：常用 'CHW'（C,H,W）、'HWC'（H,W,C）、'HW'

最小示例（随机图像，NumPy→HWC）

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter('logs')
img = (np.random.rand(128,128,3)*255).astype('uint8')  # HWC, uint8
writer.add_image('demo/random', img, 0, dataformats='HWC')
writer.close()
'''

''' 
PIL 读取后写入（自动 HWC）
from PIL import Image
import numpy as np
img = np.array(Image.open('xxx.jpg'))   # HWC, uint8
writer.add_image('img/pil', img, 1, dataformats='HWC')
'''
writer.close()
