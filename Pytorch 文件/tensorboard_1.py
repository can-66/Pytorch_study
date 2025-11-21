from torch.utils.tensorboard import SummaryWriter
import torch


#writer.add_image()


writer = SummaryWriter('logs_new')

for epoch in range(100):
    # 训练损失 - 更明显的下降趋势
    train_loss = 2.0 * (0.9 ** epoch)
    writer.add_scalar('Loss/Train', train_loss, epoch)
    
    # 验证损失 - 稍微高一些
    val_loss = 2.2 * (0.9 ** epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    
    # 准确率 - 更平滑的上升
    accuracy = min(0.95, 0.5 + epoch * 0.0045)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)
    
    # 添加学习率变化
    lr = 0.01 * (0.95 ** epoch)
    writer.add_scalar('Learning_Rate', lr, epoch)

writer.close()

#现在你可以启动 TensorBoard 来查看可视化结果：
'''
writer.add_scalar 是 PyTorch TensorBoard 中用于记录标量数据的函数
writer.add_scalar(tag, scalar_value, global_step) 


tag (str): 标签名，用于在 TensorBoard 中识别和分组数据
scalar_value (float): 要记录的标量值,y值
global_step (int): 全局步数，通常表示训练轮次或迭代次数x值

tensorboard --logdir=logs
或者指定端口：tensorboard --logdir=logs --port=6006
logdir=事件文件所在的文件夹名称


这是因为 TensorBoard 有缓存机制，即使你删除了日志文件，浏览器和 TensorBoard 服务器可能还在显示缓存的数据。

这里若还想其它图像，可以设置新的文件夹
'''


