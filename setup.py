from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy


# compile with: python setup.py build_ext --inplace


ext_modules = [
    Extension(
        "c_dist", ["c_dist.pyx"],
        include_dirs=[numpy.get_include(), '/opt/local/include'],
        extra_compile_args=["-O3"]
    )
]
setup(
    name="c_dist",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include(), '/opt/local/include/']
)
