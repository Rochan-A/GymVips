GymVips
==============

## Requirements

* Python >=3.7
* [Libvips](https://www.libvips.org/)

## Install

```(bash)
$ git clone https://github.com/Rochan-A/GymVips.git
$ cd GymVips
$ pip install .
```

Note: Installation my sometimes fail if `pkg-config` cannot find the following:
* `vips-cpp.pc`
* `vips.pc`
* `glib-2.0.pc`
* `gobject-2.0.pc`

You can fix this by setting `PKG_CONFIG_PATH` to point to the directory containing
the above files.