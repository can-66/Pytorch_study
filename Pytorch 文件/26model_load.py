import torch
import torchvision


#加载方式1 加载模型结构+参数
vgg16=torch.load("vgg16_method1.pth")
# print(vgg16)

#加载方式2 加载模型参数
vgg16=torchvision.models.vgg16(pretrained=False)#创建一个未加载预训练权重的 VGG16 模型实例，也就是随机初始化参数
model_dict=torch.load("vgg16_method2.pth")#从文件读出“状态字典”（state_dict）
vgg16.load_state_dict(model_dict)#将状态字典加载到模型中，要求模型结构与保存时一致
# print(vgg16)


#陷阱
# from 26model_save import Model,这里不能带数字
model=torch.load("model.pth")
print(model)#AttributeError: Can't get attribute 'Model' on <module '__main__' from 'e:/python_study/pytorch/26model_load.py'>
# 这里需要写一下模块，否则会报错
#from 26model_save import Model
# #新版本可以正常加载


''' 
也可以
from 26model_save import *
from 模块 import *：把模块中“公开的名字”一次性导入当前命名空间。
只能在“模块顶层”使用，函数或类内部使用会报错：SyntaxError: import * only allowed at module level。
若模块定义了 __all__，就只导入 __all__ 列表里的名字；否则导入所有非下划线开头的名字。
不推荐的原因（PEP 8）
污染命名空间，容易与本地变量或其他模块同名冲突。
可读性差：看代码不知道名字来自哪里。
工具/类型检查更难分析。

'''