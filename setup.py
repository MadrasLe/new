from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Detect compute capability
def get_cuda_arch_flags():
    """Detects available CUDA architectures."""
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return [f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}']
    except:
        pass
    # Fallback for common architectures
    return [
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
        '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
        '-gencode=arch=compute_90,code=sm_90',  # H100
    ]

cuda_arch_flags = get_cuda_arch_flags()

setup(
    name='moe_dispatch',
    version='0.1.0',
    author='Jules',
    description='Efficient MoE CUDA Kernels for PyTorch',
    packages=['moe_dispatch'],
    ext_modules=[
        CUDAExtension(
            name='moe_dispatch._C',
            sources=[
                'moe_dispatch/csrc/bindings.cpp',
                'moe_dispatch/csrc/moe_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                ] + cuda_arch_flags,
            },
            include_dirs=[
                os.path.join(os.getcwd(), 'moe_dispatch/csrc'),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
    ],
    python_requires='>=3.8',
)
