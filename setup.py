from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import torch

setup(
    name="datastar",
    version="0.2.0",
    description="High-performance GPU-first data loading library",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='datastar.loader_cpp',
            sources=['datastar/csrc/loader.cpp'],
            extra_compile_args=['-O3', '-std=c++17'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "transformers",
        "datasets",
        "tqdm",
    ],
)
