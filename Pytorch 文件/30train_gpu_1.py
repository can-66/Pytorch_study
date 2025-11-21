from torch import nn
import torch
from torch.cuda import is_available
from torch.utils.data import DataLoader
import torchvision
import time
# from model import TuiDui#引入模型·,一定要是在同一文件夹下的
from torch.utils.tensorboard import SummaryWriter
#准备数据集
train_data=torchvision.datasets.CIFAR10(root="./dataset",train=True,
                                         transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="./dataset",train=False,
                                         transform=torchvision.transforms.ToTensor(),download=True)
#length长度
train_data_length=len(train_data)
test_data_length=len(test_data)
print("训练数据集长度：{}".format(train_data_length))#训练数据集长度：50000
print("测试数据集长度：{}".format(test_data_length))#测试数据集长度：10000

#利用DataLoader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

#创建网络模型
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

#创建网络模型
tuidui=TuiDui()
if torch.cuda.is_available():
    tuidui=tuidui.cuda()#GPU加速

#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()#GPU

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(tuidui.parameters(),lr=learning_rate)


#设置训练网络的一些参数

#记录训练次数
total_train_step=0
#记录测试的次数
total_test_step=0
#训练的轮数
epoch=10

#添加tensorboard
writer=SummaryWriter('./new_logs_train')

# 全局计时起点
global_start_time = time.time()

tuidui.train()
for i in range(epoch):
    print("------------第{}轮训练开始-------------".format(i+1))
    epoch_start_time = time.time()
    #训练开始
    for data in train_dataloader:
        imgs,targets=data
        step_start_time = time.time()
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        outputs=tuidui(imgs)
        # print(outputs)
        loss=loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()#梯度清零
        loss.backward()
        optimizer.step()

        total_train_step+=1
        if total_train_step % 100==0:
            # GPU 异步执行，计时前先同步
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.time() - step_start_time
            print("训练次数：{},loss: {}, step_time(s): {:.4f}".format(total_train_step,loss.item(), step_time))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
            writer.add_scalar("time/step_s", step_time, total_train_step)
        
        
    #测试步骤开始
    #在“评估阶段”计算整套数据的损失总和
    #输出准确率
    total_test_loss=0
    total_accuracy=0




    tuidui.eval()
    with torch.no_grad():
        #用了 with torch.no_grad()，表示推理阶段不建计算图，节省显存/加速；
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs=imgs.cuda()
                targets=targets.cuda()
            outputs=tuidui(imgs)
            
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
   
            total_accuracy+=accuracy
        
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率 :{}".format(total_accuracy/test_data_length))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_length,total_test_step)
    total_test_step+=1

    # 记录单个 epoch 耗时与累计耗时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - global_start_time
    print("本轮耗时(s): {:.2f} | 累计耗时(s): {:.2f}".format(epoch_time, total_time))
    writer.add_scalar("time/epoch_s", epoch_time, i+1)
    writer.add_scalar("time/total_s", total_time, i+1)

    #保存每轮训练后的结果
    torch.save(tuidui,"tuidui{}.path".format(i+1))
    print("模型已保存")


writer.close()