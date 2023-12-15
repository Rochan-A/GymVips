GymVips
==============

## Requirements

* Python >=3.7
* pybind11 (via pip)
* [Libvips](https://www.libvips.org/)

## Install

```(bash)
$ git clone --recursive https://github.com/Rochan-A/GymVips.git
$ cd GymVips
$ CFLAGS="$(pkg-config --cflags vips-cpp --libs)" pip install .
```

Note: Installation may sometimes fail if `pkg-config` cannot find the following:
* `vips-cpp.pc`
* `vips.pc`
* `glib-2.0.pc`
* `gobject-2.0.pc`

You can fix this by setting environment variable `PKG_CONFIG_PATH` to point to the directory containing
the above files.
```(bash)
$ export PKG_CONFIG_PATH=/path/to/(dir containing vips.pc)
```

Ensure the output of `pkg-config --cflags vips-cpp --libs` looks something like below:
```(bash)
$ pkg-config --cflags vips-cpp --libs
-pthread -I/usr/local/include -I/usr/include/pango-1.0 -I/usr/include/harfbuzz -I/usr/include/pango-1.0 
-I/usr/include/fribidi -I/usr/include/cairo -I/usr/include/pixman-1 -I/usr/include/harfbuzz -I/usr/include/uuid 
-I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/x86_64-linux-gnu -I/usr/include/libmount 
-I/usr/include/blkid -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include 
-L/usr/local/lib/x86_64-linux-gnu -lvips-cpp -lvips -lgio-2.0 -lgobject-2.0 -lglib-2.0
```
