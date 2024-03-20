from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

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
