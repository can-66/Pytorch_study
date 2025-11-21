import torchvision

from torch.utils.data import DataLoader

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

'''
快速上手 DataLoader
作用：把 Dataset 打包成批次（batch），可打乱、并行加载、自动拼接张量，便于训练循环迭代。
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_set = datasets.CIFAR10(root="./dataset", train=True, transform=transform, download=False)

loader = DataLoader(
    dataset=train_set,
    batch_size=64,     # 每批样本数
    shuffle=True,      # 每个epoch打乱
    num_workers=0,     # 数据加载进程数，Windows先用0
    drop_last=False,   # 是否丢弃最后不足一批
    pin_memory=False,  # GPU训练时可考虑 TrueGPU 训练时设 True 可略提升拷贝效率
)

for images, labels in loader:
    # images: [B, C, H, W], labels: [B]
    pass
 '''
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

img,target=test_data[0]
print(img.shape)#torch.Size([3, 32, 32])
#  通道数（RGB 三通道）
# 32: 高度 32 像素
# 32: 宽度 32 像素
print(target)

from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("dataloader")
step=0

for epoch in range(2):
    for data in test_loader:
        step=0
        imgs,targets=data
        # print(imgs.shape)#torch.Size([4, 3, 32, 32])
        # #4张图片，3通道，32*32
        # print(targets)#tensor([2, 0, 8, 1])。。。。。。
        writer.add_images("Epoch: {}".format(epoch),imgs,step)#注意是add_images

# Python 的字符串格式化，把变量 epoch 填进字符串里。
# 写法1（你现在的）："Epoch: {}".format(epoch) → 当 epoch=3 时结果是 "Epoch: 3"
# 等价更简洁（推荐）：f"Epoch: {epoch}" （Python 3.6+）



        step=step+1
#
#两轮，若是shuffle=False 则两轮照片是一样的。True则不一样
''' 

概念区别
add_image: 写入单张图片
add_images: 一次写入多张图片（批量）
输入形状与参数
add_image(tag, img, step, dataformats='CHW')
img: 单张图
Tensor: [C,H,W]（默认 CHW）
NumPy: [H,W,C] 或 [H,W]（需设 dataformats）
add_images(tag, imgs, step, dataformats='NCHW')
imgs: 多张图的批次
Tensor: [N,C,H,W]（默认 NCHW）
NumPy: [N,H,W,C] 或 [N,H,W]（需设 dataformats）
可选参数 max_images 控制最多显示前多少张
from torch.utils.tensorboard import SummaryWriter
import torch, torchvision.utils as vutils

writer = SummaryWriter('logs')

# 单张
img = torch.rand(3, 64, 64)
writer.add_image('one', img, 0)  # CHW

# 批量
batch = torch.rand(16, 3, 64, 64)  # NCHW
writer.add_images('many', batch, 0)  # 直接多张

# 批量做网格更直观（推荐）
grid = vutils.make_grid(batch, nrow=4)  # CHW
writer.add_image('grid', grid, 0)

writer.close()


常见坑
OpenCV 读图需 BGR→RGB，再写 HWC 或转 CHW
float 类型必须在 0~1；uint8 在 0~255
dataformats 要与实际维度一致（单张 CHW/HWC/HW；多张 NCHW/NHWC/NHW）
'''

writer.close()


