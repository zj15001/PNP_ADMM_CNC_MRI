"""
The definition of ADMM-L1 algorithm
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import os.path
import logging
import argparse
import numpy as np
from collections import OrderedDict
import scipy.io as sio
import torch
from utils import utils_logger
from utils import utils_image as util

def soft(x,c):
    return np.fmax(np.fabs(x) - c, 0) * np.sign(x)

def analyze_parse_ADMM_L1(default_iter_num, default_lambda1, default_reo):
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", type=int, default=default_iter_num, help="Number of iterations")
    parser.add_argument("--lambda1", type=float, default=default_lambda1, help="regularization parameter")
    parser.add_argument("--reo", type=float, default=default_reo, help="Lagrange parameter")
    ADMM_L1_opt = parser.parse_args()
    return ADMM_L1_opt

def ADMM_L1(mask, noises, **ADMM_L1_opts):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    iter_num = ADMM_L1_opts.get('iter_num', 20)
    lambda1 = ADMM_L1_opts.get('lambda1', 0.04)
    reo = ADMM_L1_opts.get('reo', 0.04)
    task_current = 'dn'  # 'dn' for denoising
    testset_name = 'Set1'  # test set, set1
    n_channels = 1
    show_img = False  # default: False
    save_E = True  # save estimated image
    border = 0
    A = np.zeros((256, 256), dtype='uint8')
    out = [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A]
    n = 0
    use_clip = True
    testsets = 'testsets'  # fixed
    results = 'results'  # fixed
    result_name = testset_name + '_' + task_current + '_ADMM_L1'
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['re'] = []

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []
    test_results_ave['ssim'] = []
    test_results_ave['re'] = []

    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_L
        # --------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_H = util.modcrop(img_H, 8)
        img_L = util.uint2single(img_H)

        if use_clip:
            img_L = util.uint2single(util.single2uint(img_L))
        util.imshow(img_L) if show_img else None

        # --------------------------------
        # (2) initialize x
        # --------------------------------

        index = np.nonzero(mask)
        img_L = img_L.squeeze()
        y = np.fft.fft2(img_L) * mask + noises  # observed value
        img_L_init = np.fft.ifft2(y)  # zero fill
        print("zero-filling psnr = %.4f" % util.psnr(img_L_init*255, img_L*255))

        x = np.absolute(np.copy(img_L_init))
        z = np.copy(x)
        w = np.zeros((256, 256), dtype=np.float64)

        # --------------------------------
        # (3) main iterations
        # --------------------------------

        for i in range(iter_num):

            """ Update x """

            xtilde = np.copy(z - w)
            xf = np.fft.fft2(xtilde)
            La2 = 1.0 / 2.0 / reo
            xf[index] = (La2 * xf[index] + y[index]) / (1.0 + La2)
            x = np.real(np.fft.ifft2(xf))
            x = np.absolute(x)

            """ Update z """
            z = soft(x + w, reo * lambda1)

            """ Update w """
            w = w + x - z

        # --------------------------------
        # (4) img_E
        # --------------------------------

        out[n] = x
        img_E = x * 255

        if n_channels == 1:
            img_H = img_H.squeeze()
        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name + '_PDG L1.png'))

        # --------------------------------
        # (5) PSNR SSIM RE
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        re = util.calculate_re(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['re'].append(re)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f} ; RE: {:.4f}.'.format(img_name + ext, psnr, ssim, re))
        util.imshow(np.concatenate([img_E, img_H], axis=1),
                    title='Recovered / Ground-truth') if show_img else None
        n += 1

    # --------------------------------
    # Average PSNR SSIM RE
    # --------------------------------

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_re = sum(test_results['re']) / len(test_results['re'])
    logger.info(
        '------> testset_name: ({}), Average PSNR:({:.3f})dB, Average ssim : ({:.3f}), Average re : ({:.3f}) )'.format(
            testset_name, ave_psnr, ave_ssim, ave_re))
    test_results_ave['psnr'].append(ave_psnr)
    test_results_ave['ssim'].append(ave_ssim)
    test_results_ave['re'].append(ave_re)

    return out

ADMM_L1_opt = analyze_parse_ADMM_L1(50, 0.1, 0.015)
# the arguments are  default_max iteration, default_lambda1, default_reo

with torch.no_grad():

    # ---- load mask matrix ----
    mat = np.array([sio.loadmat('CS_MRI/Q_Random30.mat'),
                    sio.loadmat('CS_MRI/Q_Radial30.mat'),
                    sio.loadmat('CS_MRI/Q_Cartesian30.mat')])
    mask = np.array([mat[0].get('Q1').astype(np.float64),
                     mat[1].get('Q1').astype(np.float64),
                     mat[2].get('Q1').astype(np.float64)])

    # ---- load noises -----
    noises = sio.loadmat('CS_MRI/noises.mat')
    noises = noises.get('noises').astype(np.complex128) * 3.0

    # ---- set options -----
    ADMM_L1_opts = dict(iter_num=ADMM_L1_opt.iter_num, lambda1=ADMM_L1_opt.lambda1, reo=ADMM_L1_opt.reo)

    maskname = ['Q_Random30', 'Q_Radial30', 'Q_Cartesian30' ]
    k = 0
    print("------------------------------>model name = ({}) , mask = ({}) " .format('ADMM_L1', maskname[k]))
    out = ADMM_L1(mask[k], noises, **ADMM_L1_opts)
