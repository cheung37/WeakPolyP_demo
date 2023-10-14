# WeakPolyP_demo
This repository is exclusively intended for coding demo tests of the WeakPolyp paper.
# About folder dataset
This directory is used to store the dataset. The SUN-SEG dataset is created by processing 1000 images and masks from the Kvasir-SEG polyp segmentation dataset. 
We selected 500 of them for the training set, 250 for the easy test set, and 250 for the challenging test set.
Please note that during training, a new folder named 'SUN-SEG-Processed' is generated. This folder contains the further processed dataset, and we will not upload it here.
# About folder pretrain
Two model files, pvt_v2_b2.pth and res2net50_v1b_26w_4s-3cf99910.pth, need to be stored here. 
Due to the large size of these models, we will not use Git to upload them.
You can download these models into the pretrain folder by following URL
Downloading pvt_v2_v2:
https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing
Downloading res2net:
https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing
# About folder source
This is the main code file:
Testing code: M2B_SC_demo_2_test
Training code: M2B_SC_demo_2_train
Paper model code: model
Referenced model code: pvtv2 and res2net
Toolbox code: utils
