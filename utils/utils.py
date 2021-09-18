"""
Some function definitions
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import torch
import cv2
import numpy as np
from PIL import Image

# ---- calculating PSNR (dB) of x -----
def psnr(x,im_orig):
    max = 1
    M, N = np.shape(x)
    mse = (np.sum((np.absolute(x - im_orig)) ** 2)) / (M * N )
    psnr = 10 * np.log10(max * max / mse)
    return psnr

# ---- Selection of denoising step function -----
def Denoisingstep(denoise_func, x,sigma,im_orig):
    m, n = im_orig.shape
    xtilde = np.copy(x)
    mintmp = np.min(xtilde)
    maxtmp = np.max(xtilde)
    xtilde = (xtilde - mintmp) / (maxtmp - mintmp)

    # the reason for the following scaling:
    # our denoisers are trained with "normalized images + noise"
    # so the scale should be 1 + O(sigma)
    scale_range = 1.0 + sigma / 255.0 / 2.0
    scale_shift = (1 - scale_range) / 2.0
    xtilde = xtilde * scale_range + scale_shift

    # pytorch denoising model
    xtilde_torch = np.reshape(xtilde, (1, 1, m, n))
    # print('test0', xtilde_torch)
    xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).cuda()
    # print('test0',type(xtilde_torch))

    r = denoise_func(xtilde_torch).cpu().numpy()
    r = np.reshape(r, (m, n))
    x = xtilde - r

    # scale and shift the denoised image back
    x = (x - scale_shift) / scale_range
    x = x * (maxtmp - mintmp) + mintmp
    return x

# ---- The gradient of f -----
def Df(x,mask,y):
    res = np.fft.fft2(x) * mask
    index = np.nonzero(mask)
    res[index] = res[index] - y[index]
    df = np.fft.ifft2(res)
    return df

# ---- Put a frame on the image -----
def get_mask_image(mask, left_top, right_top, left_bottom, right_bottom):
    # The image order for Anchor to be displayed must be top left, bottom left, bottom right, and top right
    contours = np.array([[left_top, left_bottom, right_bottom, right_top]], dtype=np.int)
    # print(contours)
    """
    The first parameter is which image to display on;
    The second parameter is the contour;
    The third parameter specifies which contours in the contours list are drawn. If -1, all contours in the list are drawn.
    The fourth parameter is the color of the drawing;
    The fifth parameter is the thickness of the line
    """
    mask_image = cv2.drawContours(mask, contours, -1, (1, 1, 1), 2)  # Color: BGR

    return mask_image

# ---- Enlarge the local image for display -----
def enlargement(j,x,region):
    original_image = x
    # print(original_image.shape)
    original_image_width = original_image.shape[1]
    original_image_height = original_image.shape[0]
    print("The image size (width * height) is: {}*{}".format(original_image_width, original_image_height))

    # coordinate
    left_top = region[j]
    right_top = region[j+1]
    left_bottom = region[j+2]
    right_bottom = region[j+3]

    mask2 = original_image.copy()
    mask_image = get_mask_image(mask2, left_top, right_top, left_bottom, right_bottom)

    x1 = min(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    x2 = max(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    y1 = min(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    y2 = max(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_image[y1:y1 + hight, x1:x1 + width] # Get the cut image

    image = Image.fromarray(crop_img)
    image = image.resize((original_image_width, original_image_height))

    # Frame the enlarged picture
    left_top = [0, 0]
    right_top = [original_image_width, 0]
    left_bottom = [0, original_image_height]
    right_bottom = [original_image_width, original_image_height]
    image = np.array(image)
    mask_crop_img = get_mask_image(image, left_top, right_top, left_bottom, right_bottom)
    result = (mask_image, mask_crop_img)

    return result