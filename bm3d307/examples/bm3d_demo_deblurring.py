"""
BM3D deblurring demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
"""


import numpy as np
from bm3d import bm3d_deblurring, BM3DProfile, gaussian_kernel
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from scipy.ndimage.filters import correlate
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # Experiment specifications
    imagename = 'cameraman256.png'

    # Load noise-free image
    y = np.array(Image.open(imagename)) / 255

    # Generate blurry + noisy image
    experiment_number = 3

    if experiment_number == 1:
        sigma = np.sqrt(2) / 255
        v = np.zeros((15, 15))
        for x1 in range(-7, 8, 1):
            for x2 in range(-7, 8, 1):
                v[x1 + 7, x2 + 7] = 1 / (x1 ** 2 + x2 ** 2 + 1)

        v = v / np.sum(v)

    elif experiment_number == 2:
        sigma = np.sqrt(8) / 255
        s1 = 0
        v = np.zeros((15, 15))
        for a1 in range(-7, 8, 1):
            s1 = s1 + 1
            s2 = 0
            for a2 in range(-7, 8, 1):
                s2 = s2 + 1
                v[s1-1, s2-1] = 1 / (a1 ** 2 + a2 ** 2 + 1)

    elif experiment_number == 3:
        bsnr = 40
        sigma = -1  # if "sigma=-1", then the value of sigma deps on the BSNR
        v = np.ones((9, 9))
        v = v / np.sum(v)

    elif experiment_number == 4:
        sigma = 7 / 255
        v = np.atleast_2d(np.array([1, 4, 6, 4, 1])).T @ np.atleast_2d(np.array([1, 4, 6, 4, 1]))
        v = v / np.sum(v)

    elif experiment_number == 5:
        sigma = 2 / 255
        v = gaussian_kernel((25, 25), 1.6)

    else:  # 6 +
        sigma = 8 / 255
        v = gaussian_kernel((25, 25), 0.4)

    y_blur = correlate(np.atleast_3d(y), np.atleast_3d(v), mode='wrap')  # performs blurring (by circular convolution)

    if sigma == -1:  # check whether to use BSNR in order to define value of sigma
        sigma = np.sqrt(np.linalg.norm(np.ravel(y_blur - np.mean(y_blur)), 2) ** 2 / (y.shape[0] * y.shape[1] * 10 ** (bsnr / 10)))

    z = y_blur + sigma * np.random.normal(size=y_blur.shape)

    # Call BM3D deblurring With the default settings.
    y_est = bm3d_deblurring(z, sigma, v)

    # To include refiltering:
    # y_est = bm3d_deblurring(z, sigma, v, 'refilter');

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d_deblurring(z, sigma, v, profile);

    # Note: You may also pass a PSD
    # y_est = bm3d_deblurring(z, psd, v);

    psnr = get_psnr(y, y_est)
    print("PSNR:", psnr)

    # PSNR ignoring 16-pixel wide borders (as used in the paper), due to refiltering potentially leaving artifacts
    # on the pixels near the boundary of the image when noise is not circulant
    psnr_cropped = get_cropped_psnr(y, y_est, [16, 16])
    print("PSNR cropped:", psnr_cropped)

    # Ignore values outside range for display (or plt gives an error for multichannel input)
    y_est = np.minimum(np.maximum(y_est, 0), 1)
    z_rang = np.minimum(np.maximum(z, 0), 1)

    plt.title("y, z, y_est")
    plt.imshow(np.concatenate((y, np.squeeze(z_rang), y_est), axis=1), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
