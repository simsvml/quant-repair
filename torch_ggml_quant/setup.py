import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torch_ggml_quant',
    ext_modules=[cpp_extension.CppExtension(
        'torch_ggml_quant',
        [
            'dequant.cpp',
            'dequant_cpu.cpp',
        ],
        include_dirs = [
            os.environ.get('LLAMA_CPP_INCLUDEDIR', '../llama.cpp'),
        ],
        library_dirs = [
            os.environ.get('LLAMA_CPP_LIBDIR', '../llama.cpp'),
        ],
        libraries = [
            #'llama',
        ],
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
