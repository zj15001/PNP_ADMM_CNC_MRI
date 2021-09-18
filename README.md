# PNP-ADMM-CNC

The implement of the following paper:
 "Plug-and-Play ADMM for MRI Reconstruction with Convex Nonconvex Sparse regularization"

# Scripts

## ADMM-L1

ADMM_L1.py (The definition of ADMM-L1 algorithm)

## PNP-ADMM-L1

PNP_ADMM_L1_BM3D.py (The definition is about the use of BM3D denoising under the PNP_ADMM_L1 framework)

PNP_ADMM_L1_D.py (The definition is about the use of neural network denoising under the PNP_ADMM_L1 framework)

## ADMM-CNC

ADMM_CNC.py (The definition of ADMM_CNC algorithm)

## PNP-ADMM-CNC

PNP_ADMM_CNC_BM3D.py (The definition is about the use of BM3D denoising under the PNP_ADMM_CNC framework)

PNP_ADMM_CNC_D.py (The definition is about the use of neural network denoising under the PNP_ADMM_CNC framework)

# How to run the scripts?

Run with default settings main.py

All parameters and other required functions are explained in the file "utils/utils.py".

The images used in the experiment are all in the file: testsets/set

The noises and sampling templates used in the experiment are all in the file: CS_MRI

The neural network framework was trained using Zhang Kai, et al. If you want to run this code, please put the download file in the folder ''model_zoo'',
Download link: [https://github.com/cszn/KAIR] ,or download directly from the following link.

*  Google drive download link: [https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing)

*  腾讯微云下载链接: [https://share.weiyun.com/5qO32s3](https://share.weiyun.com/5qO32s3)

# Citation

If you find our code helpful in your resarch or work, please cite our paper.