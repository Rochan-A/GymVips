/* compile with:
 *      g++ -g -Wall example.cc `pkg-config vips-cpp --cflags --libs`
 */

#include <iostream>
#include <random>
#include <vector>
#include <utility>
#include <cstdlib>
#include <vips/vips8>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
using namespace vips;
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

/* BaseEnv class */
class BaseEnv
{

public:
  vector<string> files;
  vector<int> classes;
  pair<int, int> view_sz;
  int max_episode_len = 100;
  int timestep = 0;
  int dataset_index = 0;

  /* Hack to initialize */
  VImage image = VImage::new_from_memory((void *)NULL, (size_t)1, 1, 1, 1, VIPS_FORMAT_UCHAR);

  /* Initialize files that we need to read */
  BaseEnv(py::dict dataset, py::tuple view_sz, int max_episode_len)
  {
    for (auto &item : dataset) {
        /* dict keys are file paths, values are class_idx */
        const std::string &key = item.first.cast<std::string>();
        const int &value = item.second.cast<int>();

        BaseEnv::files.push_back(key);
        BaseEnv::classes.push_back(value);
    }

    BaseEnv::view_sz = view_sz.cast<pair<int, int>>();
    BaseEnv::max_episode_len = max_episode_len;
  }

  void _init_random_image()
  {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, BaseEnv::files.size()-1);

    BaseEnv::dataset_index = dis(gen);

    /* pick random image from file list */
    BaseEnv::image = VImage::new_from_file(&BaseEnv::files[BaseEnv::dataset_index][0], VImage::option()->set("access", VIPS_ACCESS_RANDOM));
  }

  /* Initalizes 3D py::array_t of type with pixels of the region in memory */
  py::array_t<int> get_region(box_t patch)
  {
    /* (view_sz.first, view_sz.second, 3) */
    const size_t _h = (size_t)BaseEnv::view_sz.first;
    const size_t _w = (size_t)BaseEnv::view_sz.second;
    const size_t _c = (size_t)3;

    constexpr size_t elsize = sizeof(int);
    size_t shape[3]{_c, _h, _w};
    size_t strides[3]{_w * _h * elsize, _w * elsize, elsize};
    auto a = py::array_t<int>(shape, strides);
    auto view = a.mutable_unchecked<3>();

    VipsRect r{patch.first.first, patch.first.second, BaseEnv::view_sz.first, BaseEnv::view_sz.second};

    VipsRegion *region;
    if(!(region = vips_region_new(BaseEnv::image.get_image())))
        vips_error_exit(NULL);

    if (vips_region_prepare(region, &r))
        vips_error("vips_region_prepare error!", NULL);

    for (int y = 0; y < r.height; y++) {
        VipsPel *p = VIPS_REGION_ADDR(region, r.left, r.top + y);
        for (int x = 0; x < r.width; x++){
            for (int b = 0; b < BaseEnv::image.bands(); b++){
                view(b, y, x) = *p++;
            }
        }
    }
    return a;
  }

  /* reset the environment: select a random image and location to read */
  py::tuple reset()
  {
    _init_random_image();

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0f, 1.0f);

    /* random (x, y) coordinates */
    pair<float, float> points{float(dis(gen)), float(dis(gen))};
    box_t patch = continuous_to_coords(points, {BaseEnv::image.width(), BaseEnv::image.height()}, BaseEnv::view_sz);

    BaseEnv::timestep = 0;

    py::dict info("timestep"_a=timestep, "target"_a=BaseEnv::classes[BaseEnv::dataset_index]);
    /* return (obs, info) */
    return py::make_tuple(get_region(patch), info);
  }

  /* step the environment: select new observation and return */
  py::tuple step(py::tuple action)
  {
    float action_x = action.attr("__getitem__")(0).attr("__float__")().cast<float>();
    float action_y = action.attr("__getitem__")(1).attr("__float__")().cast<float>();

    /* Get box based on action */
    box_t patch = continuous_to_coords({action_x, action_y}, {BaseEnv::image.width(), BaseEnv::image.height()}, BaseEnv::view_sz);

    BaseEnv::timestep += 1;

    py::dict info("timestep"_a=timestep, "target"_a=BaseEnv::classes[BaseEnv::dataset_index]);
    /* return (next_obs, reward, done, truncated, info) */
    return py::make_tuple(get_region(patch), 0, (timestep >= BaseEnv::max_episode_len ? py::bool_(true) : py::bool_(false)), py::bool_(false), info);
  }
};

PYBIND11_MODULE(vipsenv, m)
{
  py::class_<BaseEnv>(m, "BaseEnv")
      .def(py::init<py::dict, py::tuple, int>(), py::arg("dataset"), py::arg("view_sz"), py::arg("max_episode_len"))
      .def("reset", &BaseEnv::reset, "reset method of environment")
      .def("step", &BaseEnv::step, "step method of environment", py::arg("action"))
      .def_readwrite("files", &BaseEnv::files)
      .def_readwrite("classes", &BaseEnv::classes)
      .def_readwrite("view_sz", &BaseEnv::view_sz)
      .def_readwrite("max_episode_len", &BaseEnv::max_episode_len)
      .def_readonly("timestep", &BaseEnv::timestep)
      .def_readwrite("dataset_index", &BaseEnv::dataset_index);
}
