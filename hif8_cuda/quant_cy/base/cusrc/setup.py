from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hifxg_quant',
    ext_modules=[
        CUDAExtension('hifxg_quant', [
            'hifxg_quant.cpp',
            'hifxg_quant_cuda.cu',
        ],
        #extra_compile_args=['-std=c++17'], 
        extra_link_args=['-lgomp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
