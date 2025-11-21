import torchvision
#classtorchvision.datasets.CIFAR10(root: Union[str, Path], train: bool = True, 
# transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
'''root: Union[str, Path]含义：数据集存储的根目录路径 类型：字符串或 Path 对象
示例：root = "data"  # 数据会下载到 ./data/cifar-10-batches-py/
root = "/path/to/datasets"  # 绝对路径

2. train: bool = True
含义：是否加载训练集
True：加载训练集（50,000 张图片）
False：加载测试集（10,000 张图片）

3transform: Optional[Callable] = None
含义：对图像进行变换的函数
类型：可调用对象（如 transforms.Compose）
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = CIFAR10(root="data", train=True, transform=transform)

4. target_transform: Optional[Callable] = None
含义：对标签进行变换的函数
类型：可调用对象
# 将标签转换为 one-hot 编码
def to_one_hot(label):
    one_hot = torch.zeros(10)
    one_hot[label] = 1
    return one_hot

dataset = CIFAR10(root="data", train=True, target_transform=to_one_hot)
download: bool = False
含义：是否自动下载数据集
True：如果数据集不存在则自动下载
False：不下载，如果不存在会报错

'''
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)

print("PIL类型")

print(test_set)
#     Dataset CIFAR10
#     Number of datapoints: 10000
#     Root location: ./dataset
#     Split: Test
print(test_set[0])#(<PIL.Image.Image image mode=RGB size=32x32 at 0x24EC57BF940>, 3)
#这里返回的是一个元组
#这个 3 是标签（label）所在的下标，表示这张图片属于 CIFAR-10 数据集的第 3 个类别。
print(test_set.classes)
#['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']十个类型

img,target=test_set[0]
#元组的第一个元素给img，第二个元素给target
print(img)
print(target)
#img.show()

#将数据集中的图片类型转换为tensor的
dataset_transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transforms,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transforms,download=True)

print("tensor类型")
print(test_set[0])
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("P10")
for i in range(10):
    img ,target=test_set[i]
    writer.add_image("Test_set",img,i)

writer.close()
