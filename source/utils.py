import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class TrainData(Dataset):
    def __init__(self, cfg):
        self.samples = []
        for name in os.listdir(cfg.train_image):
            image = cfg.train_image + '/' + name
            mask = cfg.train_mask + '/'  + name.replace('.jpg', '.png')
            self.samples.append((image, mask))

        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image, mask = cv2.imread(image_name), cv2.imread(mask_name)
        image, mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.float32(mask > 128)
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'].permute(2, 0, 1)
    '''假设原始的图像和掩膜数据的维度顺序为(H, W, C)，其中H表示高度，W表示宽度，C表示通道数。
    经过.permute(2, 0, 1)操作后，图像和掩膜数据的维度顺序变为(C, H, W)，即通道维度在最前面，然后是高度和宽度维度。'''

    def __len__(self):
        return len(self.samples)

class TestData(Dataset):
    def __init__(self, cfg):
        self.samples  = []
        for name in os.listdir(cfg.test_image):
            image = cfg.test_image+'/' + name
            mask  = cfg.test_mask+'/' + name.replace('.jpg', '.png')
            self.samples.append((image, mask))
        print('Test Data: %s,   Test Samples: %s'%(cfg.test_image, len(self.samples)))

        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(320, 320),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image, mask           = cv2.imread(image_name), cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        image, mask           = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.float32(mask>128)
        pair                  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], mask_name

    def __len__(self):
        return len(self.samples)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def weight_init(module):
    # 权重初始化函数，根据传入的神经网络模块进行初始化
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU)):
            pass
        else:
            m.initialize()


def MyPreprocess(path_src):
    print('process', path_src)  # path_src是图像文件的源路径
    path_dst = path_src.replace('/SUN-SEG/', '/SUN-SEG-Processed/')  # 将字符串中的/SUN-SEG/替换成/SUN-SEG-Processed/
    for name in os.listdir(path_src + '/Frame/' ):
        image = cv2.imread(path_src + '/Frame/'  + name)
        image = cv2.resize(image, (352, 352), interpolation=cv2.INTER_LINEAR)
        '''读取path_src+'/Frame/'+name路径下的图像文件，
        并使用cv2.resize函数调整图像大小为(352, 352)。'''
        mask = cv2.imread(path_src + '/GT/'  + name , cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
        '''读取path_src+'/GT/'+name.replace('.jpg', '.png')路径下的图像文件作为掩码。
        掩码图像被灰度读取(cv2.IMREAD_GRAYSCALE)，然后也被调整大小为(352, 352)。'''
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        box = np.zeros_like(mask)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box[y:y + h, x:x + w] = 255
        '''使用cv2.findContours函数找到掩码图像中的轮廓，
        并使用矩形边界框将轮廓部分置为255，生成一个新的图像box。'''

        os.makedirs(path_dst + '/Frame/' , exist_ok=True)
        '''该函数可以递归创建目录，比如说传入的路径变量名path_dst="../dataset/SUN-SEG/TrainDataset"
        首先..回到上级目录，也就是本代码的上级目录'''
        cv2.imwrite(path_dst + '/Frame/'  +  name, image)
        os.makedirs(path_dst + '/GT/' , exist_ok=True)
        cv2.imwrite(path_dst + '/GT/'  +  name.replace('.jpg', '.png'), mask)
        os.makedirs(path_dst + '/Box/' , exist_ok=True)
        cv2.imwrite(path_dst + '/Box/'   + name.replace('.jpg', '.png'), box)
        '''根据目标路径创建文件夹，然后将预处理后的图像和掩码保存到对应的文件夹中。'''