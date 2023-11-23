/* compile with:
 *      g++ -g -Wall example.cc `pkg-config vips-cpp --cflags --libs`
 */

#include <iostream>
#include <random>
#include <vector>
#include <utility>
#include <cstdlib>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "image_manip.hpp"

using namespace std;
using namespace pybind11::literals;

namespace py = pybind11;

/* Box-type for upper-left and lower-right coordinates */
typedef pair<pair<int, int>, pair<int, int>> box_t;

/* Function ported from Python to convert continuous coordinates of action to
  upper-left and lower-right coordinates given the image size and view size */
box_t continuous_to_coords(pair<float, float> action, pair<int, int> img_sz, pair<int, int> view_sz)
{
  float x = action.first;
  float y = action.second;

  x = (x + 1) / 2;
  y = (y + 1) / 2;

  pair<int, int> up_left{int((img_sz.first - view_sz.first) * x), int((img_sz.second - view_sz.second) * y)};
  pair<int, int> lower_right{up_left.first + view_sz.first, up_left.second + view_sz.second};

  return {up_left, lower_right};
}

/* Initialize a 3D array to work with Python Numpy of template type
   Numpy array has shape (c, h, w)
*/
template <typename T>
py::array_t<T> init_3darray(int h, int w, int c)
{
  const size_t _h = (size_t)h;
  const size_t _w = (size_t)w;
  const size_t _c = (size_t)c;

  constexpr size_t elsize = sizeof(T);
  size_t shape[3]{_c, _h, _w};
  size_t strides[3]{_c * _h * elsize, _w * elsize, elsize};
  auto a = py::array_t<T>(shape, strides);
  auto view = a.template mutable_unchecked<3>();

  for (size_t i = 0; i < a.shape(0); i++)
  {
    for (size_t j = 0; j < a.shape(1); j++)
    {
      for (size_t k = 0; k < a.shape(2); k++)
      {
        /* type-cast */
        view(i, j, k) = 0;
      }
    }
  }
  return a;
}


/* BaseEnv class */
class BaseEnv
{

public:
  vector<string> files;
  pair<int, int> view_sz;
  ImageContainer ic;

  /* Initialize files that we need to read */
  BaseEnv(py::list file_paths, py::tuple view_sz)
  {
    BaseEnv::files = file_paths.cast<vector<string>>();
    BaseEnv::view_sz = view_sz.cast<pair<int, int>>();
  }

  /* Select a random file and initialize the ImageContainer */
  void _init_random_image()
  {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, files.size());

    ic.read_file(files[dis(gen)]);
  }

  /* Initalizes py::array_t of type with pixels of the region in memory */
  template <typename T>
  py::array_t<T> get_region(int h, int w, int c, box_t patch)
  {
    const size_t _h = (size_t)h;
    const size_t _w = (size_t)w;
    const size_t _c = (size_t)c;

    constexpr size_t elsize = sizeof(T);
    size_t shape[3]{_c, _h, _w};
    size_t strides[3]{_c * _h * elsize, _w * elsize, elsize};
    auto a = py::array_t<T>(shape, strides);
    auto view = a.template mutable_unchecked<3>();

    for (size_t i = 0; i < a.shape(0); i++)
    {
      for (size_t j = 0; j < a.shape(1); j++)
      {
        for (size_t k = 0; k < a.shape(2); k++)
        {
          // TODO: implement way to get pixel value
          view(i, j, k) = 0; // ic.get_pixel(i, j, k);
        }
      }
    }
    return a;
  }

  /* reset the environment: select a random image and location to read */
  py::array_t<int> reset()
  {
    _init_random_image();

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0f, 1.0f);

    /* random (x, y) coordinates */
    pair<float, float> points{float(dis(gen)), float(dis(gen))};
    box_t patch = continuous_to_coords(points, {ic.width, ic.height}, BaseEnv::view_sz);

    return init_3darray<int>(10, 10, 3);//get_region<int>(BaseEnv::view_sz.first, BaseEnv::view_sz.second, 3, patch);
  }

  py::array_t<int> step(py::tuple action)
  {
    float action_x = action.attr("__getitem__")(0).attr("__float__")().cast<float>();
    float action_y = action.attr("__getitem__")(1).attr("__float__")().cast<float>();

    /* Get box based on action */
    box_t patch = continuous_to_coords({action_x, action_y}, {ic.width, ic.height}, BaseEnv::view_sz);

    return get_region<int>(BaseEnv::view_sz.first, BaseEnv::view_sz.second, 3, patch);
  }
};


PYBIND11_MODULE(vipsenv, m) {
    py::class_<BaseEnv>(m, "BaseEnv")
        .def(py::init<py::list, py::tuple>(), py::arg("file_paths"), py::arg("view_sz"))
        .def("reset", &BaseEnv::reset, "reset method of environment")
        .def("step", &BaseEnv::step, "step method of environment", py::arg("action"));
        // .def_readwrite_static("files", &BaseEnv::files)
        // .def_readwrite_static("view_sz", &BaseEnv::view_sz);
}
