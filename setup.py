# Available at setup time due to pyproject.toml
import subprocess
import re
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


def parse_text(text):
    # Initialize lists to store parsed information
    include_dirs = []
    library_dirs = []
    libraries = []

    # Define regular expressions for -I, -L, and -l patterns
    include_pattern = re.compile(r"-I(\S+)")
    library_pattern = re.compile(r"-L(\S+)")

    # Extract -I and -L options
    include_dirs = include_pattern.findall(text)
    library_dirs = library_pattern.findall(text)

    # Extract -l options
    libraries = re.findall(r"-l(\S+)", text)

    return (
        list(set(include_dirs)),
        list(set(library_dirs)),
        ["-l" + l for l in list(set(libraries))],
    )


gobject = subprocess.run(
    ["pkg-config", "gobject-2.0", "--libs", "--cflags"], capture_output=True, text=True
)
glib = subprocess.run(
    ["pkg-config", "glib-2.0", "--libs", "--cflags"], capture_output=True, text=True
)
vips_cpp = subprocess.run(
    ["pkg-config", "vips-cpp", "--libs", "--cflags"], capture_output=True, text=True
)

if len(gobject.stdout) == 0:
    raise RuntimeError(f"Issue with pkg-config gobject-2.0 --libs --cflags!")

if len(glib.stdout) == 0:
    raise RuntimeError(f"Issue with pkg-config glib-2.0 --libs --cflags!")

if len(vips_cpp.stdout) == 0:
    raise RuntimeError(f"Issue with pkg-config vips-cpp --libs --cflags!")

include_dirs, library_dirs, extra_compile_args = parse_text(
    vips_cpp.stdout # + gobject.stdout + glib.stdout
)

ext_modules = [
    Pybind11Extension(
        "vipsenv",
        sources=["src/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=["-lvips", "-lvips-cpp"] + extra_compile_args + ["-pthread"],
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
    install_requires=["pybind11>=2.11.1"]
)
