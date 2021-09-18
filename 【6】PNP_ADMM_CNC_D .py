"""
The definition is about the use of neural network denoising under the PNP-ADMM-CNC framework
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import os.path
import logging
import argparse
import numpy as np
import scipy.io as sio
import torch
from collections import OrderedDict
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_image as util

def denoising_step2(model_name, x, x8, sigmas, i, model, noises, device, noise_level_model):
    if 'dncnn' in model_name and 'fdncnn' not in model_name:
        if not x8:
            x = model(x)
        else:
            x = utils_model.test_mode(model, x, mode=3)

    elif 'fdncnn' in model_name:
        noises1 = np.absolute(noises)
        noises1 = torch.from_numpy(noises1).float()
        noises1 = np.reshape(noises1, (256, 256, 1))
        noises1 = util.single2tensor4(noises1).to(device) / 255.
        x = torch.cat((x, noises1), dim=1).to(device)
        x = x.to(device)

        if not x8:
            x = model(x)
        else:
            x = utils_model.test_mode(model, x, mode=3)

    elif 'drunet' in model_name:
        if x8:
            x = util.augment_img_tensor4(x, i % 8)

        x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
        x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)

        if x8:
            if i % 8 == 3 or i % 8 == 5:
                x = util.augment_img_tensor4(x, 8 - i % 8)
            else:
                x = util.augment_img_tensor4(x, i % 8)

    elif 'ircnn' in model_name:
        if x8:
            x = util.augment_img_tensor4(x, i % 8)

        x = model(x)

        if x8:
            if i % 8 == 3 or i % 8 == 5:
                x = util.augment_img_tensor4(x, 8 - i % 8)
            else:
                x = util.augment_img_tensor4(x, i % 8)

    elif 'ffdnet' in model_name:
        ffdnet_sigma = torch.full((1, 1, 1, 1), noise_level_model / 255.).type_as(x)
        x = model(x, ffdnet_sigma)

    return x

def analyze_parse_PNP_ADMM_CNC_D(default_alpha, default_iter_num, default_lambda1, default_reo, default_b):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=default_alpha, help="Step size in Plug-and Play")
    parser.add_argument("--iter_num", type=int, default=default_iter_num, help="Number of iterations")
    parser.add_argument("--lambda1", type=float, default=default_lambda1, help="regularization parameter")
    parser.add_argument("--reo", type=float, default=default_reo, help="Lagrange parameter")
    parser.add_argument("--b", type=float, default=default_b, help="convex parameter")
    PNP_ADMM_CNC_D_opt = parser.parse_args()
    return PNP_ADMM_CNC_D_opt

def PNP_ADMM_CNC_D(model_name, mask, noises, **PNP_ADMM_CNC_D_opts):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    alpha = PNP_ADMM_CNC_D_opts.get('alpha', 0.4)
    iter_num = PNP_ADMM_CNC_D_opts.get('iter_num', 46)
    lambda1 = PNP_ADMM_CNC_D_opts.get('lambda1', 2.75)
    reo= PNP_ADMM_CNC_D_opts.get('reo', 1)        # Here reo is 1/bata in the original article
    b = PNP_ADMM_CNC_D_opts.get('b', 1)           # Here b is b^2 in the original article

    task_current = 'dn'  # 'dn' for denoising
    testset_name = 'Set1'  # test set, set1
    x8 = False  # default: False, x8 to boost performance
    n_channels = 1
    sf = 1  # unused for denoising
    show_img = False  # default: False
    border = 0
    n = 0
    sigmas = 0
    A = np.zeros((256, 256), dtype='uint8')
    out = [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A]
    psnr1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    use_clip = True
    save_E = True  # save estimated image
    model_zoo = 'model_zoo'  # fixed
    testsets = 'testsets'  # fixed
    results = 'results'  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
    model_path = os.path.join(model_zoo, model_name + '.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # ----------------------------------------
    # load model
    # ----------------------------------------

    if 'dncnn' in model_name and 'fdncnn' not in model_name:
        from models.network_dncnn import DnCNN as net
        noise_level_img = 15
        noise_level_model = noise_level_img  # noise level of model, default 0
        if model_name in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
            nb = 20  # fixed
        else:
            nb = 17  # fixed
        x8 = False
        border = sf if task_current == 'sr' else 0
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    elif 'fdncnn' in model_name:
        from models.network_dncnn import FDnCNN as net
        border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR and SSIM
        x8 = False
        noise_level_img = 15  # default: 0, noise level for LR image
        noise_level_model = noise_level_img  # noise level of model, default 0

        if 'clip' in model_name:
            use_clip = True  # clip the intensities into range of [0, 1]
        else:
            use_clip = False

        model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=64, nb=20, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    elif 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        noise_level_img = 15 / 255.0  # default: 0, noise level for LR image
        noise_level_model = noise_level_img  # noise level of model, default 0
        n_channels = 3 if 'color' in model_name else 1  # fixed
        modelSigma1 = 49
        modelSigma2 = noise_level_model * 255.
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=iter_num,
                                         modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    elif 'ircnn' in model_name:
        from models.network_dncnn import IRCNN as net
        noise_level_img = 15 / 255.0  # default: 0, noise level for LR image
        noise_level_model = noise_level_img  # noise level of model, default 0
        x8 = False
        modelSigma1 = 49
        modelSigma2 = noise_level_model * 255.
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=iter_num,
                                         modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0

    elif 'ffdnet' in model_name:
        from models.network_ffdnet import FFDNet as net
        noise_level_img = 15  # noise level for noisy image
        noise_level_model = noise_level_img  # noise level for model
        nc = 64  # setting for grayscale image
        nb = 15  # setting for grayscale image
        border = sf if task_current == 'sr' else 0

        if 'clip' in model_name:
            use_clip = True  # clip the intensities into range of [0, 1]
        else:
            use_clip = False

        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['re'] = []

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []
    test_results_ave['ssim'] = []
    test_results_ave['re'] = []

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

        img_L = img_L.squeeze()
        y = np.fft.fft2(img_L) * mask + noises  # observed value
        img_L_init = np.fft.ifft2(y)
        index = np.nonzero(mask)
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

            x = torch.from_numpy(x).float()
            x = np.reshape(x, (256, 256, 1))
            x = util.single2tensor4(x).to(device)

            z = np.absolute(z)
            z = torch.from_numpy(z).float()
            z = np.reshape(z, (256, 256, 1))
            z = util.single2tensor4(z).to(device)

            w = np.absolute(w)
            w = torch.from_numpy(w).float()
            w = np.reshape(w, (256, 256, 1))
            w = util.single2tensor4(w).to(device)

            """ Update z """

            if 'ircnn' in model_name:
                current_idx = np.int(np.ceil(sigmas[i].cpu().numpy() * 255. / 2.) - 1)

                if current_idx != former_idx:
                    model.load_state_dict(model25[str(current_idx)], strict=True)
                    model.eval()
                    for _, v in model.named_parameters():
                        v.requires_grad = False
                    model = model.to(device)
                former_idx = current_idx

            s = denoising_step2(model_name, z, x8, sigmas, i, model, noises, device, noise_level_model)
            t = (1 - alpha) * z + alpha * (x + w) + alpha * reo * lambda1 * b * (z - s)
            z = denoising_step2(model_name, t, x8, sigmas, i, model, noises, device, noise_level_model)

            """ Update w """
            w = w + x - z
            x = x.data.squeeze().float().clamp_(0, 1).cpu().numpy()
            z = z.data.squeeze().float().clamp_(0, 1).cpu().numpy()
            w = w.data.squeeze().float().clamp_(0, 1).cpu().numpy()

        # --------------------------------
        # (4) img_E
        # --------------------------------

        out[n] = x
        img_E = np.uint8((x * 255.0).round())

        if n_channels == 1:
            img_H = img_H.squeeze()
        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name + 'PNP_ADMM_CNC_D.png'))

        # --------------------------------
        # (5) PSNR SSIM RE
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        re = util.calculate_re(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['re'].append(re)
        logger.info('{:s} - PSNR: {:.4f} dB; SSIM: {:.4f} ; RE: {:.4f}.'.format(img_name + ext, psnr, ssim, re))
        util.imshow(np.concatenate([img_E, img_H], axis=1),
                    title='Recovered / Ground-truth') if show_img else None
        psnr1[n] = psnr
        n += 1

    # --------------------------------
    # Average PSNR SSIM RE
    # --------------------------------

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_re = sum(test_results['re']) / len(test_results['re'])
    logger.info('------> testset_name: ({}), alpha: ({:.3f}), Average PSNR:({:.3f})dB, Average ssim : ({:.3f}), Average re : ({:.3f}) )'.format(
            testset_name, alpha, ave_psnr, ave_ssim, ave_re))
    test_results_ave['psnr'].append(ave_psnr)
    test_results_ave['ssim'].append(ave_ssim)
    test_results_ave['re'].append(ave_re)

    return out, psnr1

def denoising_step(model_name, x, x8, model):
    if 'dncnn' in model_name and 'fdncnn' not in model_name:
        if not x8:
            x = model(x)
        else:
            x = utils_model.test_mode(model, x, mode=3)
    return x

def analyze_parse_PNP_ADMM_CNC_DnCNN(default_alpha, default_iter_num, default_lambda1, default_reo, default_b):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=default_alpha, help="Step size in Plug-and Play")
    parser.add_argument("--iter_num", type=int, default=default_iter_num, help="Number of iterations")
    parser.add_argument("--lambda1", type=float, default=default_lambda1, help="regularization parameter")
    parser.add_argument("--reo", type=float, default=default_reo, help="Lagrange parameter")
    parser.add_argument("--b", type=float, default=default_b, help="convex parameter")

    PNP_ADMM_CNC_DnCNN_opt = parser.parse_args()
    return PNP_ADMM_CNC_DnCNN_opt

def PNP_ADMM_CNC_DnCNN(model_name1, model_name2, mask, noises, **PNP_ADMM_CNC_DnCNN_opts):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    alpha = PNP_ADMM_CNC_DnCNN_opts.get('alpha', 0.4)
    iter_num = PNP_ADMM_CNC_DnCNN_opts.get('iter_num', 46)
    lambda1 = PNP_ADMM_CNC_DnCNN_opts.get('lambda1', 2.75)
    reo= PNP_ADMM_CNC_DnCNN_opts.get('reo', 1)        # Here reo is 1/bata in the original article
    b = PNP_ADMM_CNC_DnCNN_opts.get('b', 1)           # Here b is b^2 in the original article
    task_current = 'dn'  # 'dn' for denoising
    testset_name = 'Set1'  # test set, set1
    n_channels = 1
    sf = 1  # unused for denoising
    show_img = False  # default: False
    save_E = True  # save estimated image
    border = 0
    n = 0
    A = np.zeros((256, 256),dtype='uint8')
    out = [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A]
    psnr1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    use_clip = True
    model_zoo = 'model_zoo'  # fixed
    testsets = 'testsets'  # fixed
    results = 'results'  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name1 + '_' + model_name2
    model_path1 = os.path.join(model_zoo, model_name1 + '.pth')
    model_path2 = os.path.join(model_zoo, model_name2 + '.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_dncnn import DnCNN as net
    nb = 17  # fixed
    x8 = False
    border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR and SSIM
    model1 = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model1.load_state_dict(torch.load(model_path1), strict=True)
    model1.eval()
    for k, v in model1.named_parameters():
        v.requires_grad = False
    model1 = model1.to(device)
    logger.info('Model path 1: {:s}'.format(model_path1))
    number_parameters = sum(map(lambda x: x.numel(), model1.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    model2 = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model2.load_state_dict(torch.load(model_path1), strict=True)
    model2.eval()
    for k, v in model2.named_parameters():
        v.requires_grad = False
    model2 = model2.to(device)
    logger.info('Model path 2: {:s}'.format(model_path2))
    number_parameters = sum(map(lambda x: x.numel(), model2.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

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

        img_L = img_L.squeeze()
        y = np.fft.fft2(img_L) * mask + noises  # observed value
        img_L_init = np.fft.ifft2(y)  # zero fill
        print("zero-filling psnr = %.4f" % util.psnr(img_L_init * 255, img_L * 255))

        index = np.nonzero(mask)
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

            x = torch.from_numpy(x).float()
            x = np.reshape(x, (256, 256, 1))
            x = util.single2tensor4(x).to(device)

            z = np.absolute(z)
            z = torch.from_numpy(z).float()
            z = np.reshape(z, (256, 256, 1))
            z = util.single2tensor4(z).to(device)

            w = np.absolute(w)
            w = torch.from_numpy(w).float()
            w = np.reshape(w, (256, 256, 1))
            w = util.single2tensor4(w).to(device)

            """ Update z """
            s = denoising_step(model_name1, z, x8,  model1)
            t = (1 - alpha) * z + alpha * (x + w) + alpha * reo * lambda1 * b * (z - s)
            z = denoising_step(model_name2, t, x8, model2)

            """ Update w """
            w = w + x - z
            x = x.data.squeeze().float().clamp_(0, 1).cpu().numpy()
            z = z.data.squeeze().float().clamp_(0, 1).cpu().numpy()
            w = w.data.squeeze().float().clamp_(0, 1).cpu().numpy()

        # --------------------------------
        # (4) img_E
        # --------------------------------
        out[n] = x
        img_E = np.uint8((x * 255.0).round())

        if n_channels == 1:
            img_H = img_H.squeeze()
        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name + 'PNP_ADMM_CNC_DnCNN.png'))

        # --------------------------------
        # (5) PSNR SSIM RE
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        re = util.calculate_re(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['re'].append(re)
        logger.info('{:s} - PSNR: {:.4f} dB; SSIM: {:.4f} ; RE: {:.4f}.'.format(img_name + ext, psnr, ssim, re))
        util.imshow(np.concatenate([img_E, img_H], axis=1),
                    title='Recovered / Ground-truth') if show_img else None
        psnr1[n] = psnr
        n += 1

    # --------------------------------
    # Average PSNR SSIM RE
    # --------------------------------

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_re = sum(test_results['re']) / len(test_results['re'])
    logger.info('------> testset_name: ({}), alpha: ({:.3f}), Average PSNR:({:.3f})dB, Average ssim : ({:.3f}), Average re : ({:.3f}) )'.format(
            testset_name, alpha, ave_psnr, ave_ssim, ave_re))
    test_results_ave['psnr'].append(ave_psnr)
    test_results_ave['ssim'].append(ave_ssim)
    test_results_ave['re'].append(ave_re)

    return out, psnr1

PNP_ADMM_CNC_D_opt1 = analyze_parse_PNP_ADMM_CNC_D(0.9, 50, 0.2, 0.45, 0.3)   #FDnCNN
# the arguments are default_alpha, default_max iteration, default_lambda1, default_reo, default_b
PNP_ADMM_CNC_DnCNN_opt = analyze_parse_PNP_ADMM_CNC_DnCNN(1.2, 50, 4, 0.45, 0.3)   #DnCNN
# the arguments are default_alpha, default_max iteration, default_lambda1, default_reo, default_b
PNP_ADMM_CNC_D_opt2 = analyze_parse_PNP_ADMM_CNC_D(0.9, 50, 1.35, 0.45,  0.3)   #FFDNet
# the arguments are default_alpha, default_max iteration, default_lambda1, default_reo,default_b
PNP_ADMM_CNC_D_opt3 = analyze_parse_PNP_ADMM_CNC_D(0.5, 50, 1.3, 0.45, 2)   #IRCNN
# the arguments are default_alpha, default_max iteration, default_lambda1, default_reo,default_b
PNP_ADMM_CNC_D_opt4 = analyze_parse_PNP_ADMM_CNC_D(1, 50, 0.8, 0.8, 0.45)  #DRUNet
# the arguments are default_alpha, default_max iteration, default_lambda1, default_reo, default_b

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

    PNP_ADMM_CNC_D_opts1 = dict(alpha=PNP_ADMM_CNC_D_opt1.alpha, iter_num=PNP_ADMM_CNC_D_opt1.iter_num,
                             lambda1=PNP_ADMM_CNC_D_opt1.lambda1, reo=PNP_ADMM_CNC_D_opt1.reo, b=PNP_ADMM_CNC_D_opt1.b)
    PNP_ADMM_CNC_DnCNN_opts = dict(alpha=PNP_ADMM_CNC_DnCNN_opt.alpha, iter_num=PNP_ADMM_CNC_DnCNN_opt.iter_num,
                             lambda1=PNP_ADMM_CNC_DnCNN_opt.lambda1, reo=PNP_ADMM_CNC_DnCNN_opt.reo, b=PNP_ADMM_CNC_DnCNN_opt.b)
    PNP_ADMM_CNC_D_opts2 = dict(alpha=PNP_ADMM_CNC_D_opt2.alpha, iter_num=PNP_ADMM_CNC_D_opt2.iter_num,
                             lambda1=PNP_ADMM_CNC_D_opt2.lambda1, reo=PNP_ADMM_CNC_D_opt2.reo, b=PNP_ADMM_CNC_D_opt2.b)
    PNP_ADMM_CNC_D_opts3 = dict(alpha=PNP_ADMM_CNC_D_opt3.alpha, iter_num=PNP_ADMM_CNC_D_opt3.iter_num,
                             lambda1=PNP_ADMM_CNC_D_opt3.lambda1, reo=PNP_ADMM_CNC_D_opt3.reo, b=PNP_ADMM_CNC_D_opt3.b)
    PNP_ADMM_CNC_D_opts4 = dict(alpha=PNP_ADMM_CNC_D_opt4.alpha, iter_num=PNP_ADMM_CNC_D_opt4.iter_num,
                             lambda1=PNP_ADMM_CNC_D_opt4.lambda1, reo=PNP_ADMM_CNC_D_opt4.reo, b=PNP_ADMM_CNC_D_opt4.b)

    name = ['fdncnn_gray', 'ffdnet_gray', 'ircnn_gray', 'drunet_gray']  #
    maskname = ['Q_Random30', 'Q_Radial30', 'Q_Cartesian30']
    name1 = ['dncnn_15', 'dncnn_25', 'dncnn_50']
    PNP_ADMM_CNC_D_opts = [PNP_ADMM_CNC_D_opts1, PNP_ADMM_CNC_D_opts2, PNP_ADMM_CNC_D_opts3, PNP_ADMM_CNC_D_opts4]

    m = 3
    k = 0
    print("------------------------------>model name = ({}) , mask = ({}) " .format(name[m], maskname[0]))
    out, psnr = PNP_ADMM_CNC_D(name[m], mask[k], noises, **PNP_ADMM_CNC_D_opts[m])
    out1 = out[0]

    print("------------------------------>model1 name = ({}) ,model2 name = ({}) , mask = ({}) " .format(name[1], name[0],maskname[k]))
    out2, psnr2 = PNP_ADMM_CNC_DnCNN(name1[1], name1[0], mask[k], noises, **PNP_ADMM_CNC_DnCNN_opts)
    out2 = out2[0]