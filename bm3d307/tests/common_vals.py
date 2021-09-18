import numpy as np
from scipy.fftpack import fft2

TEST_IMAGE_DIMS = (50, 50)
TEST_IMAGE_SIZE = TEST_IMAGE_DIMS[0] * TEST_IMAGE_DIMS[1]
# Allowed error when the procedure could cause small differences
ALLOWED_ERROR_DIFFERENT = 1/255 - 1e-8
# Allowed error when there should be absolutely no difference (minus float errors)
ALLOWED_ERROR_SAME = 1e-6


def generate_input_data(seed=1):
    np.random.seed(seed)
    sigma = 0.1
    return np.random.normal(size=(TEST_IMAGE_DIMS[0], TEST_IMAGE_DIMS[1])) * sigma, sigma


def generate_input_data_3d():
    np.random.seed(1)
    sigma = 0.1
    return np.random.normal(size=(TEST_IMAGE_DIMS[0], TEST_IMAGE_DIMS[1], 3)) * sigma, sigma


def calculate_2d_psd(kernel, size):
    return np.abs(fft2(kernel, shape=size)) ** 2 * size[0] * size[1]
