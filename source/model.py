import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import Res2Net50
from pvtv2 import pvt_v2_b2
from utils import weight_init

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        '''通过继承父类nn.Module的构造函数来初始化，通过super()函数获取父类的对象，这里.__init__()表示调用父类的构造函数
            Fusion表示子类的名称，self是子类的实例对象'''
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        # BatchNorm2d用于进行正态化，这里的参数值64表示通道数为64
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4   = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        # 将x2,x3,x4三个特征分别通过上述线性变化层，该线性变化层由一个卷积层和一个批归一化层
        x4           = F.interpolate(x4, size=x2.size()[2:], mode='bilinear')
        x3           = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        '''这里使用双线性插值将x4和x3的分辨率调整得和x2一样。
            interplate表示插值的意思，x4为输入数据，size为x2的H和W，这里的大小可以理解为[batch_size,channels,height,weight]
            插值模式为bilinear，也就是双线性插值'''
        out          = x2*x3*x4
        return out

    def initialize(self):
        weight_init(self) # 递归初始化权重

class WeakPolyp(nn.Module):
    def __init__(self, cfg):
        super(WeakPolyp, self).__init__()
        if cfg.backbone=='res2net50':
            self.backbone = Res2Net50()
            channels      = [256, 512, 1024, 2048]
            # 直接通过Res2Net50模型四个通道产生四个特征
        if cfg.backbone=='pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            channels      = [64, 128, 320, 512]

        self.fusion       = Fusion(channels) # 创建融合特征的模型
        self.linear       = nn.Conv2d(64, 1, kernel_size=1) # 输入特征64，输出特征1，卷积核1×1
        
        ## initialize
        if cfg.mode=='train':
            # 如果训练模式，那么初始化权重
            weight_init(self)
        elif cfg.mode=='test':
            # 如果测试模式，那么直接加载已经训练完毕
            self.load_state_dict(torch.load(cfg.snapshot)) # 用字典形式加载已经训练完毕的模型参数
            '''参数具体解释如下：
            （1）self：self 是当前对象（类实例）的引用。在这个上下文中，它表示当前类的实例对象，即要加载参数的模型对象。
            （2）load_state_dict：load_state_dict 是 PyTorch 中的一个函数，用于加载模型的参数状态字典。
            它将参数状态字典中的值复制到模型的对应参数上。
            （3）torch.load(cfg.snapshot)：torch.load() 是 PyTorch 中的函数，用于加载保存在文件中的对象。
            cfg.snapshot 是一个文件路径，指定了要加载的参数状态字典所在的文件。
                cfg 是一个配置对象或字典，其中 snapshot 是一个保存参数状态字典的文件路径。
                torch.load(cfg.snapshot) 加载指定文件路径下的对象，这里是加载参数状态字典。
            （4）load_state_dict 函数将加载的参数状态字典应用于模型对象 self，以更新模型的参数。'''
        else:
            raise ValueError

    def forward(self, x):
        x1,x2,x3,x4 = self.backbone(x)
        pred        = self.fusion(x1,x2,x3,x4)
        pred        = self.linear(pred)
        return pred