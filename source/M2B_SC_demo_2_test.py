import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import TestData
from model import WeakPolyp
from PIL import Image


class Config:
    def __init__(self, backbone, testset):
        ## set the backbone type
        self.backbone = backbone  # 设置主干模型
        ## set the path of snapshot model
        self.snapshot = self.backbone + '/model.pth' # 读取已经训练好的模型进行测试
        ## set the path of testing dataset
        self.test_image = '../dataset/SUN-SEG-Processed/' + testset + '/Frame' # 息肉图片路径
        self.test_mask = '../dataset/SUN-SEG-Processed/' + testset + '/GT' # 息肉像素级标签路径
        self.test_set = testset # 读取测试数据集是困难的还是简单的
        ## other settings
        self.mode = 'test'
        self.batch_size = 64
        self.num_workers = 4

class Test(object):
    def __init__(self, cfg):
        ## dataset
        '''传入相关配置，加载测试数据'''
        self.cfg       = cfg
        self.data      = TestData(cfg)
        self.loader    = DataLoader(self.data, batch_size=cfg.batch_size, pin_memory=False, shuffle=False, num_workers=cfg.num_workers)
        ## model
        '''传入相关模型'''
        self.model     = WeakPolyp(cfg).cuda()
        self.model.eval() # 测试模式

    def test_prediction(self):
        with torch.no_grad():
            mae, iou, dice, cnt = 0, 0, 0, 0
            for image, mask, name in self.loader:
                '''这里代码的解释大致与训练模块的一致'''
                B, H, W         = mask.shape
                pred            = self.model(image.cuda().float())
                pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred            = (pred.squeeze()>0).cpu().float()

                cnt            += B
                mae            += np.abs(pred-mask).mean() # 计算预测值和目标值之间的平均绝对误差（Mean Absolute Error，MAE）
                inter, union    = (pred*mask).sum(dim=(1,2)), (pred+mask).sum(dim=(1,2))
                iou            += ((inter+1)/(union-inter+1)).sum() # 计算交并比
                dice           += ((2*inter+1)/(union+1)).sum() # 计算dice
                print('cnt=%10d | mae=%.4f | dice=%.4f | iou=%.4f'%(cnt, mae/cnt, dice/cnt, iou/cnt))

                # 存储预测结果
                path_src = '../dataset/SUN-SEG-Processed/' + self.cfg.test_set + '/pred'
                if not os.path.exists(path_src):
                    os.makedirs(path_src,exist_ok=True)
                for i in range(B):
                    # 创建图像数组对象
                    pred_mask=pred[i]
                    pred_mask=pred_mask*255
                    pred_mask=np.asarray(pred_mask,dtype=np.uint8)

                    # 创建PIL图像对象
                    pil_image=Image.fromarray(pred_mask,'L') # 将numpy数据类型的掩码图像转换为灰度图像

                    # 保存图像
                    file_name=name[i]
                    image_name=os.path.basename(file_name)
                    image_path=os.path.join(path_src,image_name)

                    pil_image.save(image_path)
                print("save prediction test data:{}".format(path_src))
            print('cnt=%10d | mae=%.4f | dice=%.4f | iou=%.4f'%(cnt, mae/cnt, dice/cnt, iou/cnt))




if __name__=='__main__':
    os.environ ["CUDA_VISIBLE_DEVICES"] = '0'
    Test(Config('res2net50', 'TestEasyDataset')).test_prediction()
    Test(Config('res2net50', 'TestHardDataset')).test_prediction()
    # Test(Config('pvt_v2_b2', 'TestEasyDataset')).test_prediction()
    # Test(Config('pvt_v2_b2', 'TestHardDataset')).test_prediction()