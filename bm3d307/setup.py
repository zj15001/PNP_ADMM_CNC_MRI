from setuptools import setup, Extension
import os
from ctypes.util import find_library
from sys import platform

USE_CYTHON = False
ext_modules = []

if USE_CYTHON:

    if platform != "darwin" and platform != "linux":
        raise NotImplementedError("Cython version is currently not available for your platform.")

    # Compile pyx with Cython if we can
    try:
        from Cython.Build import cythonize
        import numpy as np
    except ImportError:
        cythonize = None
        use_cython = False
        print("Cythonizing off.")
    else:
        os.environ["CC"] = "gcc"

        # OpenBLAS is not needed, but will speed up the thing a lot
        has_openblas = find_library("openblas")
        if not has_openblas:
            print("OpenBLAS not found!")

        # Select correct library files
        if platform == "darwin":
            lib_names = ["bm3d_thr_mac", "bm3d_wie_mac"]
        else:  # linux
            lib_names = ["bm3d_thr", "bm3d_wie"]

        if not has_openblas:
            lib_names[0] += "_noblas"
            lib_names[1] += "_noblas"
        else:
            lib_names += ["openblas"]

        bm3d_source = Extension(
            name="bm3d.bm3d_c",
            sources=["bm3d/bm3d_c.pyx"],
            libraries=lib_names,
            library_dirs=["bm3d"],
            include_dirs=["bm3d", np.get_include()],
        )
        ext_modules = cythonize([bm3d_source], compiler_directives={'language_level': '3'})

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bm3d",
    version='3.0.7',
    description='BM3D for correlated noise.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    author='Ymir Mäkinen',
    author_email='ymir.makinen@tuni.fi',
    packages=['bm3d'],
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'PyWavelets'],
    tests_require=['pytest'],
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Free for non-commercial use',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
