"""
Multichannel (non-rgb) BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189.
For multichannel images, block matching is performed only on the first channel.
"""


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bm3d import bm3d, BM3DProfile
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from scipy.io import loadmat

def main():
    # Experiment specifications

    # The multichannel example data is acquired from: http://www.bic.mni.mcgill.ca/brainweb/
    # (C.A. Cocosco, V. Kollokian, R.K.-S. Kwan, A.C. Evans,
    #  "BrainWeb: Online Interface to a 3D MRI Simulated Brain Database"
    # NeuroImage, vol.5, no.4, part 2/4, S425, 1997
    # -- Proceedings of 3rd International Conference on Functional Mapping of the Human Brain, Copenhagen, May 1997.
    data_name = 'brainslice.mat'
    table_name = 'slice_sample'

    # Load noise-free image
    # Data should be in same shape as with Image.open, but channel count can be any (M x N x channels)
    # Noise-free data should be between 0 and 1.
    y = loadmat(data_name)[table_name]
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'g2'
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
    # The call is identical to that of the grayscale BM3D.
    y_est = bm3d(z, psd)

    # To include refiltering:
    # y_est = bm3d(z, psd, 'refilter');

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d(z, sqrt(noise_var));

    # Instead of passing a singular PSD, you may also pass equal number of PSDs to the channels:
    # y_est = bm3d(z, np.concatenate((psd1, psd2, psd3, psd4, psd5), 2))
    # y_est = bm3d(z, [sigma1, sigma2, sigma3, sigma4, sigma5])

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
    disp_mat = np.concatenate([np.concatenate((y[:, :, i], np.squeeze(z_rang[:, :, i]), y_est[:, :, i]), axis=1)
                              for i in range(y_est.shape[2])], axis=0)

    plt.imshow(disp_mat)
    plt.show()


if __name__ == '__main__':
    main()
