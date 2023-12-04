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
        sources=["src/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs = ['/usr/local/include', '/usr/include/pango-1.0', '/usr/include/harfbuzz', '/usr/include/pango-1.0', '/usr/include/fribidi', '/usr/include/cairo', '/usr/include/pixman-1', '/usr/include/harfbuzz', '/usr/include/uuid', '/usr/include/freetype2', '/usr/include/libpng16', '/usr/include/x86_64-linux-gnu', '/usr/include/libmount', '/usr/include/blkid', '/usr/include/glib-2.0', '/usr/lib/x86_64-linux-gnu/glib-2.0/include'],
        library_dirs = ['/usr/local/lib/x86_64-linux-gnu'],
        extra_compile_args = ['-std=c++17', '-pthread', '-lvips-cpp', '-lvips', '-lgio-2.0', '-lgobject-2.0', '-lglib-2.0'],
        extra_link_args=['-pthread', '-L/usr/local/lib/x86_64-linux-gnu', '-L/usr/lib/x86_64-linux-gnu', '-lvips-cpp', '-lvips', '-lgio-2.0', '-lgobject-2.0', '-lglib-2.0']
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
)
