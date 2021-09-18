# Python wrapper for BM3D denoising - from Tampere with love

Python wrapper for BM3D for stationary correlated noise (including white noise) for color,
grayscale and multichannel images and deblurring.

BM3D is an algorithm for attenuation of additive spatially correlated
stationary (aka colored) Gaussian noise. This package provides a wrapper
for the BM3D binaries for use for grayscale, color and other multichannel images
for denoising and deblurring.

This implementation is based on Y. Mäkinen, L. Azzari, A. Foi,
"Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise",
in Proc. 2019 IEEE Int. Conf. Image Process. (ICIP), pp. 185-189.

This package includes binaries which require an additional
installation of the OpenBLAS library (http://www.openblas.net/).
For Windows and Mac, a version of OpenBLAS is included in the binary.

The package contains the BM3D binaries compiled for:
- Windows (Win10, MinGW-32)
- Linux (Debian 10, 64-bit)
- Mac OSX (El Capitan, 64-bit)

Additionally, library files with an example Cython interface are provided for Linux and Mac builds.

The binaries are available for non-commercial use only. For details, see LICENSE.

For examples, see the examples folder of the full source zip, which also includes the example noise cases demonstrated in the paper.
Alternatively, you can download the examples from http://www.cs.tut.fi/~foi/GCF-BM3D/bm3d_py_demos.zip .

Authors: \
    Ymir Mäkinen <ymir.makinen@tuni.fi> \
    Lucio Azzari \
    Alessandro Foi



