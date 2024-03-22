import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

WIN_BOOST_PATH = ""
if platform.system() == "Windows":
    WIN_BOOST_PATH = "C:\\boost_1_82_0"

ext_modules = [
    Pybind11Extension(
        "insilicova._sampler._sampler",
        ["src/insilicova/_sampler/sampler.cpp"],
        # extra_compile_args=["-g"]
        include_dirs=[WIN_BOOST_PATH],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
