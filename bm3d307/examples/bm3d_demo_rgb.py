"""
RGB image BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
For color images, block matching is performed on the luminance channel.
"""

import numpy as np
from bm3d import bm3d_rgb, BM3DProfile
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # Experiment specifications
    imagename = 'image_Lena512rgb.png'

    # Load noise-free image
    y = np.array(Image.open(imagename)) / 255

    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'g4'
    noise_var = 0.02  # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    z = np.atleast_3d(y) + np.atleast_3d(noise)

    # Call BM3D With the default settings.
    y_est = bm3d_rgb(z, psd)

    # To include refiltering:
    # y_est = bm3d_rgb(z, psd, 'refilter');

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d_rgb(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d_rgb(z, sqrt(noise_var));

    # If the different channels have varying PSDs, you can supply a MxNx3 PSD or a list of 3 STDs:
    # y_est = bm3d_rgb(z, np.concatenate((psd1, psd2, psd3), 2))
    # y_est = bm3d_rgb(z, [sigma1, sigma2, sigma3])

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
    plt.imshow(y_est)
    plt.show()


if __name__ == '__main__':
    main()
