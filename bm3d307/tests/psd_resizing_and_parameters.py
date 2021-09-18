import numpy as np
from scipy.fftpack import fft2
from .common_vals import *
from bm3d import bm3d, _estimate_parameters_for_psd, _get_kernel_from_psd, _shrink_and_normalize_psd
from bm3d import BM3DProfile, _process_psd


def test_psd_to_kernel_white():
    sigma = 1
    psd = calculate_2d_psd(np.atleast_2d(sigma), (100, 100))
    kernel = _get_kernel_from_psd(psd)
    kernel2 = _get_kernel_from_psd(np.array(sigma), single_dim_psd=True)
    assert np.sum(kernel) - np.sum(kernel2) < ALLOWED_ERROR_SAME


def test_white_noise_parameters65():
    sigma = 1
    psd65 = calculate_2d_psd(np.atleast_2d(sigma), (65, 65))
    p1 = _estimate_parameters_for_psd(psd65)
    lambda_thr3d = 3.0
    lambda_wiener = 0.4
    lambda_thr3d2 = 2.5
    lambda_wiener2 = 3.6
    ref = (lambda_thr3d, lambda_wiener, lambda_thr3d2, lambda_wiener2)
    for f in range(len(ref)):
        assert np.abs(p1[f][0] - ref[f]) < ALLOWED_ERROR_SAME


def test_white_noise_parameters_bigger():
    sigma = 1
    psd = calculate_2d_psd(np.atleast_2d(sigma), (100, 100))
    kernel = _get_kernel_from_psd(np.atleast_3d(psd))
    psd65 = _shrink_and_normalize_psd(kernel, (65, 65))
    p1 = _estimate_parameters_for_psd(psd65)
    lambda_thr3d = 3.0
    lambda_wiener = 0.4
    lambda_thr3d2 = 2.5
    lambda_wiener2 = 3.6
    ref = (lambda_thr3d, lambda_wiener, lambda_thr3d2, lambda_wiener2)
    for f in range(len(ref)):
        assert np.abs(p1[f][0] - ref[f]) < ALLOWED_ERROR_SAME


def test_white_noise_parameters_nonsquare():
    sigma = 1
    psd = calculate_2d_psd(np.atleast_2d(sigma), (100, 80))
    kernel = _get_kernel_from_psd(np.atleast_3d(psd))
    psd65 = _shrink_and_normalize_psd(kernel, (65, 65))
    p1 = _estimate_parameters_for_psd(psd65)
    lambda_thr3d = 3.0
    lambda_wiener = 0.4
    lambda_thr3d2 = 2.5
    lambda_wiener2 = 3.6
    ref = (lambda_thr3d, lambda_wiener, lambda_thr3d2, lambda_wiener2)
    for f in range(len(ref)):
        assert np.abs(p1[f][0] - ref[f]) < ALLOWED_ERROR_SAME


def test_white_noise_parameters_smaller():
    sigma = 1
    psd = calculate_2d_psd(np.atleast_2d(sigma), (10, 10))
    kernel = _get_kernel_from_psd(np.atleast_3d(psd))
    psd65 = _shrink_and_normalize_psd(kernel, (65, 65))
    p1 = _estimate_parameters_for_psd(psd65)
    lambda_thr3d = 3.0
    lambda_wiener = 0.4
    lambda_thr3d2 = 2.5
    lambda_wiener2 = 3.6
    ref = (lambda_thr3d, lambda_wiener, lambda_thr3d2, lambda_wiener2)
    for f in range(len(ref)):
        assert np.abs(p1[f][0] - ref[f]) < ALLOWED_ERROR_SAME
