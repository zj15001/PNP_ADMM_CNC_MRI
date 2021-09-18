import numpy as np
from scipy.fftpack import fft2
from .common_vals import *
from bm3d import bm3d
from bm3d import BM3DProfile, BM3DStages


def test_consistency():
    z, sigma = generate_input_data()
    res1 = bm3d(np.copy(z), np.copy(sigma))
    res2 = bm3d(np.copy(z), np.copy(sigma))
    assert np.max(np.abs(res2 - res1)) < ALLOWED_ERROR_SAME


def test_consistency_traditional():
    z, sigma = generate_input_data()
    p = BM3DProfile()
    p.nf = 0
    res1 = bm3d(np.copy(z), np.copy(sigma), p)
    res2 = bm3d(np.copy(z), np.copy(sigma), p)
    assert np.max(np.abs(res2 - res1)) < ALLOWED_ERROR_SAME


def test_consistency_mismatched_psd():
    z, sigma = generate_input_data()
    res1 = bm3d(np.copy(z), calculate_2d_psd(np.atleast_2d(sigma), (9, 9)))
    res2 = bm3d(np.copy(z), calculate_2d_psd(np.atleast_2d(sigma), (9, 9)))
    assert np.max(np.abs(res2 - res1)) < ALLOWED_ERROR_SAME


def test_multidimensional_consistency_1st_dim():
    z, sigma = generate_input_data_3d()
    res1 = bm3d(z, sigma)
    res2 = bm3d(z[:, :, 0], sigma)
    assert np.max(np.abs(res2 - res1[:, :, 0])) < ALLOWED_ERROR_SAME


def test_multidimensional_consistency_between_dims():
    z, sigma = generate_input_data()
    z = np.atleast_3d(z)
    res1 = bm3d(np.concatenate((z, z, z), axis=2), sigma)
    assert np.max(np.abs(res1[:, :, 0] - res1[:, :, 1])) < ALLOWED_ERROR_SAME


def test_blockmatching_pass():
    z, sigma = generate_input_data()
    res1, bm = bm3d(z, sigma, blockmatches=(True, True))
    res2 = bm3d(z, sigma, blockmatches=bm)
    assert np.max(np.abs(res1 - res2)) < ALLOWED_ERROR_SAME


def test_blockmatching_usage():
    z, sigma = generate_input_data()
    z2 = np.ones(z.shape)
    res_cr, bm = bm3d(z2, sigma, blockmatches=(True, True))
    res1 = bm3d(z, sigma)
    res2 = bm3d(z, sigma, blockmatches=bm)
    assert np.max(np.abs(res1 - res2)) > ALLOWED_ERROR_SAME


def test_split_stages_consistency():
    z, sigma = generate_input_data()
    res1 = bm3d(z, sigma)
    res2_ht = bm3d(z, sigma, stage_arg=BM3DStages.HARD_THRESHOLDING)
    res2 = bm3d(z, sigma, stage_arg=res2_ht)
    assert np.max(np.abs(res1 - res2)) < ALLOWED_ERROR_SAME


def test_blockmatches_consistency():
    z, sigma = generate_input_data()
    res1, bms = bm3d(z, sigma, blockmatches=(True, True))
    res2 = bm3d(z, sigma, blockmatches=bms)
    assert np.max(np.abs(res1 - res2)) < ALLOWED_ERROR_SAME


def test_blockmatches_ht_inconsistency():
    z, sigma = generate_input_data()
    z2, sigma = generate_input_data(2)

    res_ref, bms = bm3d(z2, sigma, blockmatches=(True, True))

    res1 = bm3d(z, sigma, stage_arg=BM3DStages.HARD_THRESHOLDING)
    res2 = bm3d(z, sigma, blockmatches=bms, stage_arg=BM3DStages.HARD_THRESHOLDING)

    assert np.max(np.abs(res1 - res2)) != 0


def test_blockmatches_wie_inconsistency():
    z, sigma = generate_input_data()
    z2, sigma = generate_input_data(2)

    res_ref, bms = bm3d(z2, sigma, blockmatches=(True, True))
    res_ht = bm3d(z, sigma, stage_arg=BM3DStages.HARD_THRESHOLDING)

    res1 = bm3d(z, sigma, stage_arg=res_ht)
    res2 = bm3d(z, sigma, stage_arg=res_ht, blockmatches=bms)

    assert np.max(np.abs(res1 - res2)) != 0
