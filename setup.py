from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
import os

os.environ["CC"] = "gcc-mp-9"
os.environ["CXX"] = "g++-9"

# compile with: python setup.py build_ext --inplace

ext_modules = [
    Extension(
        "c_dist", ["c_dist.pyx"],
        include_dirs=[numpy.get_include(), '/opt/local/include'],
        extra_compile_args=["-O3"]
        # library_dirs=["/opt/local/lib/", "."],
    )
]
setup(
    name="c_dist",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include(), '/opt/local/include/']
)
