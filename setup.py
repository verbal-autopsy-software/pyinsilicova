import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

if platform.system() == "Windows":
    ext_modules = [
        Pybind11Extension(
            "insilicova._sampler._sampler",
            ["src/insilicova/_sampler/sampler.cpp"],
            include_dirs=["C:\\Program Files\\boost\\boost_1_82_0"],
        ),
    ]
else:
    ext_modules = [
        Pybind11Extension(
            "insilicova._sampler._sampler",
            ["src/insilicova/_sampler/sampler.cpp"],
            # extra_compile_args=["-g"]
        ),
    ]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
