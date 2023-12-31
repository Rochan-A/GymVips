GymVips
==============

GymVips is a C++-based batched environment pool with pybind11 and thread pool. Unlike other reinforcement learning environment executors, GymVips focuses on I/O intensive environments (e.g., environment configuration correspond to files saved to disk). For instance, the environment configuration may depend on an image collection.

GymVips comes in two parts: a core environment pool implementation that supports asynchronous and synchronous environments; and a TIFF image environment built on [Libvips](https://www.libvips.org/). The code is structured to be easy-to-read and modifiable to build your own custom environments.

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

## Performance Benchmarks (TODO)

All benchmarks were launched on 45 parallel executors (CPU threads or processes) and each executor ran for 5 episodes of length 100 steps each.

### SyncEnv

| Method | Time (ms) | Memory (MB) |
|--------|-------------|-------------|
| OpenCV + Gymnasium.vector.VectorEnv (Python) | 61412 | 14000 |
| GymVips (Python) | 4324 | 5030 |
| GymVips (C++) | 2723 | - |

### AsyncEnv

| Method | Time (ms) | Memory (MB) |
|--------|-------------|-------------|
| OpenCV + Gymnasium.vector.AsyncVectorEnv (Python) | - | - |
| GymVips (Python) | - | - |
| GymVips (C++) | - | - |

## Comparison with [EnvPool](http://envpool.readthedocs.io)

EnvPool focuses on compute-only environments where threading and parallel execution is relatively simple. I/O latency makes environments dealing with files saved to disk more challenging and tricky to handle.