# WeakPolyP_demo
This repository is exclusively intended for coding demo tests of the WeakPolyp paper.
## 1. About folder dataset
This directory is used to store the dataset. The SUN-SEG dataset is created by processing 1000 images and masks from the Kvasir-SEG polyp segmentation dataset.

I selected 500 of them for the training set, 250 for the easy test set, and 250 for the challenging test set.

Please note that during training, a new folder named 'SUN-SEG-Processed' is generated. This folder contains the further processed dataset, and I will not upload it here.
## 2. About folder pretrain
Two model files, pvt_v2_b2.pth and res2net50_v1b_26w_4s-3cf99910.pth, need to be stored here. 

Due to the large size of these models, I will not use Git to upload them.

You can download these models into the pretrain folder by following URL.

Downloading pvt_v2_v2:

https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing

Downloading res2net:

https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing
## 3. About folder source
This is the main code file:

Testing code: M2B_SC_demo_2_test.py

Training code: M2B_SC_demo_2_train.py

Paper model code: model.py

Referenced model code: pvtv2.py and res2net.py

Toolbox code: utils.py
## 4. file tree
    MyWeakPolyP
    ├── dataset
    │   └── SUN-SEG
    │       ├── TestEasyDataset
    │       │   ├── Frame
    │       │   └── GT  
    |       |
    │       ├── TestHardDataset
    │       │   ├── Frame
    │       │   └── GT
    │       │         
    │       └── TrainDataset
    │           ├── Frame
    │           └── GT
    |
    ├── pretrain
    │   ├── pvt_v2_b2.pth
    │   └── res2net50_v1b_26w_4s-3cf99910.pth
    |
    ├── readme.md
    |
    └── source
        ├── model.py
        ├── pvtv2.py
        ├── res2net.py
        ├── test.py
        ├── train.py
        └── utils.py

# 中文
本存储库专门用于编写WeakPolyp论文的演示测试代码。
## 1. 关于dataset文件夹
此目录用于存储数据集。SUN-SEG数据集是通过处理Kvasir-SEG息肉分割数据集中的1000张图像和掩码创建的。

我选择了其中的500张用于训练集，250张用于简单测试集，以及250张用于挑战测试集。

请注意，在训练期间，会生成一个名为“SUN-SEG-Processed”的新文件夹。该文件夹包含进一步处理的数据集，我没有在此上传它。
## 2. 关于pretrain文件夹
需要在此处存储两个模型文件，即pvt_v2_b2.pth和res2net50_v1b_26w_4s-3cf99910.pth。

由于这些模型的所占内存较大，我没有使用Git上传它们。

您可以通过以下URL将这些模型下载到pretrain文件夹中。

下载pvt_v2_v2：

https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing

下载res2net：

https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing
## 3. 关于source文件夹
这是主要的代码文件：

测试代码：M2B_SC_demo_2_test.py

训练代码：M2B_SC_demo_2_train.py

论文模型代码：model.py

引用模型代码：pvtv2.py和res2net.py

工具箱代码：utils.py
## 4. 文件树
      MyWeakPolyP
      ├── dataset
      │   └── SUN-SEG
      │       ├── TestEasyDataset
      │       │   ├── Frame
      │       │   └── GT  
      |       |
      │       ├── TestHardDataset
      │       │   ├── Frame
      │       │   └── GT
      │       │         
      │       └── TrainDataset
      │           ├── Frame
      │           └── GT
      |
      ├── pretrain
      │   ├── pvt_v2_b2.pth
      │   └── res2net50_v1b_26w_4s-3cf99910.pth
      |
      ├── readme.md
      |
      └── source
          ├── model.py
          ├── pvtv2.py
          ├── res2net.py
          ├── test.py
          ├── train.py
          └── utils.py
