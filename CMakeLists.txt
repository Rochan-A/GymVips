cmake_minimum_required(VERSION 3.4...3.18)
project(vipsenv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/include/libmount -I/usr/include/blkid -I/usr/include/uuid -I/usr/include/libpng16 -I/usr/include/pango-1.0 -I/usr/include/fribidi -I/usr/include/harfbuzz -I/usr/include/freetype2 -I/usr/include/cairo -I/usr/include/pixman-1 -pthread -lvips-cpp -lvips -lgio-2.0 -lgobject-2.0 -lglib-2.0")

add_subdirectory(pybind11)
pybind11_add_module(vipsenv src/region.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(vipsenv
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})