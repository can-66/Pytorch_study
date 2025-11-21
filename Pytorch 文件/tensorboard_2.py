from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


'''
writer.add_image(tag, img_tensor, global_step=None, dataformats='CHW') 
tag: 标签名
img_tensor: 图片数据（torch.Tensor 或 numpy.ndarray）
global_step: 步数
dataformats: 数据维度格式，常用 'CHW'、'HWC'、'HW'。注意通道顺序是 RGB。

'''

writer = SummaryWriter('logs_new')#创建一个日志文件
image_path="pytorch\data\train\ants_image\0013035.jpg"#读取图片
image_PIL=Image.open(image_path)#打开图片
image_array=np.array(image_PIL)#将图片转换为数组

writer.add_image('test',image_array,1,dataformats='HWC')#注意这里的HWC是高度，宽度，通道

writer.close()




