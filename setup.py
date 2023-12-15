# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.

# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds

ext_modules = [
    Pybind11Extension(
        "vipsenv",
        sources=["src/py_envpool.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=[
            "-pthread",
            "-std=c++11",
        ],
        libraries=["vips-cpp", "vips", "gio-2.0", "gobject-2.0", "glib-2.0"],
    ),
]

setup(
    name="vipsenv",
    version=__version__,
    author="Rochan Avlur",
    author_email="rochan.avlur@gmail.com",
    url="https://github.com/Rochan-A/GymVips",
    description="Vectorized Image-based RL Gym Library with Vips",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["pybind11>=2.11.1"],
)
