import os
import sys
import logging

import cv2
import numpy as np
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from albumentations.pytorch import ToTensorV2
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from model import WeakPolyp
from utils import clip_gradient,TrainData,TestData,MyPreprocess


def ImageVisualization(img):
    '''图像可视化函数：可视化预测结果，传入数据类型应该为tensor(H,W)'''
    img_np = img.cpu().detach().numpy()
    plt.imshow(img_np,cmap='gray')
    plt.show(block=True)
    plt.close()

class Config:
    '''配置类：用来进行一些基本的设置'''
    def __init__(self, backbone):
        ## set the backbone type
        self.backbone = backbone  # 传入模型骨干为res2net50

        ## set the path of training dataset
        self.train_image = '../dataset/SUN-SEG-Processed/TrainDataset/Frame'
        self.train_mask = '../dataset/SUN-SEG-Processed/TrainDataset/Box'

        ## set the path of testing dataset
        self.test_image = '../dataset/SUN-SEG-Processed/TestHardDataset/Frame'
        self.test_mask = '../dataset/SUN-SEG-Processed/TestHardDataset/GT'

        ## set the path of logging
        self.log_path = self.backbone + '/log'
        os.makedirs(self.log_path, exist_ok=True)

        ## keep unchanged
        if self.backbone == 'res2net50':  # 关于参数的设置
            self.mode = 'train'
            self.epoch = 16
            self.batch_size = 4
            self.lr = 0.1
            self.num_workers = 4
            self.weight_decay = 1e-3  # 权重衰减（一种正则化技术），用于避免过拟合，提高模型的泛化能力
            self.clip = 0.5  # 梯度裁剪，避免梯度爆炸，将梯度控制在一定阈值内
        if self.backbone == 'pvt_v2_b2':
            self.mode = 'train'
            self.epoch = 16
            self.batch_size = 4
            self.lr = 0.1
            self.num_workers = 4
            self.weight_decay = 1e-4
            self.clip = 1000


class Train:
    def __init__(self, cfg):

        ## parameter
        self.cfg = cfg
        # 通过SummaryWriter进行可视化
        self.logger = SummaryWriter(cfg.log_path)
        # 基本的日志记录配置，方便查看与分析
        logging.basicConfig(level=logging.INFO, filename=cfg.log_path + '/train.log', filemode='a',
                            format='[%(asctime)s | %(message)s]', datefmt='%I:%M:%S')

        ## model
        self.model = WeakPolyp(cfg).cuda()  # 模型为WeakPolyp，传入相关的基础模型（比如说Res2Net，包含一些基本的配置），这里采用GPU进行训练，加快训练速度
        self.model.train()

        ## data
        self.data = TrainData(cfg)
        self.loader = DataLoader(dataset=self.data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        '''数据集为训练数据集，批次为设置好的16，shuffle=True表示在每一个训练周期开始随机打乱数据集的顺序，num_workers设置为配置好的子进程数量'''

        ## optimizer
        base, head = [], []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer = torch.optim.SGD([{'params': base, 'lr': 0.1 * cfg.lr}, {'params': head, 'lr': cfg.lr}], momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
        self.max_dice = 0

    def forward(self):
        global_step = 0
        scaler = torch.cuda.amp.GradScaler()  # 这段代码创建了一个混合精度训练中使用的梯度缩放器(GradScaler)
        for epoch in range(self.cfg.epoch):
            if epoch in [3, 6, 9, 12]:
                # 一共训练16个回合，在上述这些回合中优化器将学习率减半
                self.optimizer.param_groups[0]['lr'] *= 0.5
            for i, (image, mask) in enumerate(self.loader):
                with torch.cuda.amp.autocast():
                    '''这行代码的作用是创建一个自动混合精度计算的上下文环境，其中的操作将自动应用混合精度计算。'''

                    ## pred 1
                    image, mask = image.cuda(), mask.cuda()  # 采用GPU训练
                    size1 = np.random.choice([256, 288, 320, 352, 384, 416, 448])
                    # 从一组给定的选择中随机选择一个作为图像的分辨率大小
                    image1 = F.interpolate(image, size=size1, mode='bilinear')
                    # 将image的大小双线性插值调整大小成预设的size1
                    pred1 = self.model(image1)  # 进行前向传播得到image1的预测掩码1（不是盒状，而是弯弯曲曲的轮廓）
                    pred1 = F.interpolate(pred1, size=352, mode='bilinear')  # 在这一步将预测掩码1通过双线性插值调整大小为352*352，得到论文所说的P1掩码

                    ## pred 2 这里的操作与pred1基本相同，只是初始图像的大小可能不同（随机采样的）
                    size2 = np.random.choice([256, 288, 320, 352, 384, 416, 448])
                    image2 = F.interpolate(image, size=size2, mode='bilinear')
                    pred2 = self.model(image2)
                    pred2 = F.interpolate(pred2, size=352, mode='bilinear')  # 得到有着弯弯曲曲轮廓的P2掩码


                    ## loss_sc 作者提出的损失函数SC
                    loss_sc = (torch.sigmoid(pred1) - torch.sigmoid(pred2)).abs()
                    '''将两个预测掩码P1和P2都通过sigmoid函数，然后相减求绝对值
                        注意sigmoid函数表达式为：f(x) = 1 / (1 + e^(-x))'''
                    loss_sc = loss_sc[mask[:, 0:1] == 1].mean()

                    ## M2B transformation
                    pred = torch.cat([pred1, pred2], dim=0)  # 将两个预测掩码在行方向上进行拼接
                    mask = torch.cat([mask, mask], dim=0)  # 将边界框标记在行方向上进行拼接
                    predW, predH = pred.max(dim=2, keepdim=True)[0], pred.max(dim=3, keepdim=True)[0]
                    '''pred.max(dim=2, keepdim=True)：在维度2上计算 pred 的最大值，dim=2 表示在第2个维度上进行操作。
                    结果是一个元组，包含两个张量：第一个张量是最大值构成的张量，第二个张量是最大值对应的索引构成的张量。
                    [0]：通过索引 [0] 取出元组中的第一个张量，即最大值构成的张量。
                    这个张量的形状是 (batch_size, num_channels, 1, height)，其中 batch_size 是批大小，num_channels 是通道数，height 是高度。'''
                    pred = torch.minimum(predW, predH)
                    pred, mask = pred[:, 0], mask[:, 0]

                    ## loss_ce + loss_dice
                    loss_ce = F.binary_cross_entropy_with_logits(pred, mask)
                    pred = torch.sigmoid(pred)
                    inter = (pred * mask).sum(dim=(1, 2))
                    union = (pred + mask).sum(dim=(1, 2))
                    loss_dice = 1 - (2 * inter / (union + 1)).mean()
                    loss = loss_ce + loss_dice + loss_sc

                ## backward
                self.optimizer.zero_grad()  # 梯度清零
                scaler.scale(loss).backward()  # 反向传播计算梯度，这里的scale()函数将损失值按比例缩放，避免梯度溢出
                scaler.unscale_(self.optimizer)  # 将优化器中的梯度按比例反缩放回原始值。这是为了在梯度裁剪之前，将梯度恢复到原始范围。
                clip_gradient(self.optimizer, self.cfg.clip)
                # 对梯度进行裁剪，限制梯度的范围，以防止梯度爆炸的情况发生。
                # clip_gradient 函数接受一个优化器对象和一个裁剪阈值 self.cfg.clip，并将梯度的范数裁剪到指定的阈值内。
                scaler.step(self.optimizer)  # 使用优化器更新模型的参数。
                scaler.update()  # scaler.update() 方法根据梯度更新的情况，自适应地调整比例因子，以保持梯度的数值范围在一个合适的区间。

                global_step += 1
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'ce': loss_ce.item(), 'dice': loss_dice.item(), 'sc': loss_sc.item()},
                                        global_step=global_step)
                ## print loss
                if global_step % 20 == 0:
                    print(
                        '{} epoch={:03d}/{:03d}, step={:04d}/{:04d}, loss_ce={:0.4f}, loss_dice={:0.4f}, loss_sc={:0.4f}'.format(
                            datetime.now(), epoch, self.cfg.epoch, i, len(self.loader), loss_ce.item(),
                            loss_dice.item(), loss_sc.item()))
            torch.cuda.empty_cache()
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            data = TestData(self.cfg)  # 根据配置读取测试数据
            loader = DataLoader(dataset=data, batch_size=64, shuffle=False,
                                num_workers=self.cfg.num_workers)  # 加载数据，注意这里批次为64
            dice, iou, cnt = 0, 0, 0  # 初始化dice函数、iou（交并比）、cnt（总共的批次）
            for image, mask, name in tqdm(loader):  # 图像像素数据、弱监督标签、图像名字
                image, mask = image.cuda().float(), mask.cuda().float()  # 采用GPU训练，数据类型为float
                B, H, W = mask.shape  # 取出标签的批次B，高度H，宽度W
                pred = self.model(image)  # 通过模型预测
                pred = F.interpolate(pred, size=(H, W), mode='bilinear')  # 调整大小和标签掩码一致
                pred = (pred.squeeze() > 0)
                # 上述代码将预测结果 pred 进行阈值化处理，即将大于0的像素值设为1，小于等于0的像素值设为0。
                # 这样可以将预测结果转换为二值图像。

                # ImageVisualization(pred[0]) # 传入每一批次的第一张预测掩码进行可视化理解

                inter, union = (pred * mask).sum(dim=(1, 2)), (pred + mask).sum(dim=(1, 2))  # 计算交集和并集
                dice += ((2 * inter + 1) / (union + 1)).sum().cpu().numpy()  # 计算dice函数
                iou += ((inter + 1) / (union - inter + 1)).sum().cpu().numpy()  # 计算交并比
                cnt += B  # 计算总的批次数量
            logging.info('epoch=%-8d | dice=%.4f | iou=%.4f | path=%s' % (
            epoch, dice / cnt, iou / cnt, self.cfg.test_image))  # 打印相关数据

        if dice / cnt > self.max_dice:  # 如果当前的dice大于记录的最大dice，那么重新记录最大dice，并且保存模型
            self.max_dice = dice / cnt
            torch.save(self.model.state_dict(), self.cfg.backbone + '/model.pth')
        self.model.train()  # 调整回训练模式

if __name__=="__main__":
    if not os.path.exists('../dataset/SUN-SEG-Processed'): # 对数据进行预处理
        MyPreprocess('../dataset/SUN-SEG/TrainDataset')
        MyPreprocess('../dataset/SUN-SEG/TestEasyDataset')
        MyPreprocess('../dataset/SUN-SEG/TestHardDataset')
    ## training
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置训练的GPU的环境变量
    Train(Config('res2net50')).forward()  # 传入模型并且调用相关的前向传播函数，该函数一经调用，整个训练过程就开始了

