from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compiler directives for optimization
compiler_directives = {
    'language_level': '3',
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'initializedcheck': False,
    'nonecheck': False,
}

# Extensions with OpenMP support
extensions = [
    Extension(
        "example_cy",
        ["example_cy.pyx"],
    ),
    Extension(
        "algorithms_cy",
        ["algorithms_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='Cython Performance Suite',
    version='2.0',
    description='Advanced Cython optimization examples with comprehensive benchmarking',
    author='Your Name',
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True  # Generate HTML annotation files
    ),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas',
    ],
)