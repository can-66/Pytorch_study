from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision
import time
from torch.utils.tensorboard import SummaryWriter

# 设备统一管理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="./dataset", train=True,
    transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="./dataset", train=False,
    transform=torchvision.transforms.ToTensor(), download=True
)

train_data_length = len(train_data)
test_data_length = len(test_data)
print("训练数据集长度：{}".format(train_data_length))
print("测试数据集长度：{}".format(test_data_length))

# DataLoader
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
class TuiDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


# 实例化并迁移到 device
tuidui = TuiDui().to(device)

# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(tuidui.parameters(), lr=learning_rate)

# 训练配置
total_train_step = 0
total_test_step = 0
epoch = 10

# TensorBoard 与计时
writer = SummaryWriter('./new_logs_train_device')
global_start_time = time.time()

tuidui.train()
for i in range(epoch):
    print("------------第{}轮训练开始-------------".format(i + 1))
    epoch_start_time = time.time()

    # 训练
    for imgs, targets in train_dataloader:
        step_start_time = time.time()
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = tuidui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            step_time = time.time() - step_start_time
            print("训练次数：{}, loss: {}, step_time(s): {:.4f}".format(
                total_train_step, loss.item(), step_time
            ))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            writer.add_scalar("time/step_s", step_time, total_train_step)

    # 验证
    total_test_loss = 0.0
    total_accuracy = 0
    tuidui.eval()
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = tuidui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率 :{}".format(total_accuracy / test_data_length))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_length, total_test_step)
    total_test_step += 1
    tuidui.train()

    # 计时输出
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - global_start_time
    print("本轮耗时(s): {:.2f} | 累计耗时(s): {:.2f}".format(epoch_time, total_time))
    writer.add_scalar("time/epoch_s", epoch_time, i + 1)
    writer.add_scalar("time/total_s", total_time, i + 1)

    # 保存模型
    torch.save(tuidui, "tuidui_device_{}.path".format(i + 1))
    print("模型已保存")

writer.close()


