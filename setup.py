from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "orderbook_engine",
        [
            "orderbook_bindings.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            "..",
        ],
        cxx_std=14,
    ),
]

setup(
    name="orderbook_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)