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

/**
 * @brief Converts continuous coordinates of action to upper-left coordinates
 * given the image size and view size.
 *
 * @param action The continuous coordinates to convert.
 * @param img_sz The size of the image.
 * @param view_sz The size of the view.
 * @return VipsRect containing upper-left and rectangle height and width.
 */
VipsRect continuous_to_coords(pair<float, float> action, pair<int, int> img_sz, pair<int, int> view_sz)
{
    // Normalize coordinates to the range [0, 1]
    float x = (action.first + 1) / 2;
    float y = (action.second + 1) / 2;

    // Calculate upper-left coordinates
    int up_left_x = static_cast<int>((img_sz.first - view_sz.first) * x);
    int up_left_y = static_cast<int>((img_sz.second - view_sz.second) * y);
    pair<int, int> up_left{};

    return VipsRect{up_left_x, up_left_y, view_sz.first, view_sz.second};
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
    py::array_t<int> a;

    /* Hack to initialize */
    VImage image = VImage::new_from_memory((void *)NULL, (size_t)1, 1, 1, 1, VIPS_FORMAT_UCHAR);

    /**
     * @brief Constructor initalizing environment.
     *
     * @param py::dict dataset with keys (path to image) and values (classes).
     * @param py::tuple view size
     * @param int maximum episode length
     */
    BaseEnv(py::dict dataset, py::tuple view_sz, int max_episode_len)
    {

        try
        {
            if (dataset.size() == 0 || view_sz.size() != 2 || max_episode_len <= 0)
            {
                throw invalid_argument("Invalid arguments. 'dataset' must not be empty, 'view_sz' must be a tuple of size 2, and 'max_episode_len' must be greater than 0.");
            }

            for (auto &item : dataset)
            {
                /* dict keys are file paths, values are class_idx */
                const string &key = item.first.cast<string>();
                const int &value = item.second.cast<int>();

                files.push_back(key);
                classes.push_back(value);
            }

            this->view_sz = view_sz.cast<pair<int, int>>();
            this->max_episode_len = max_episode_len;


            const size_t _h = static_cast<size_t>(this->view_sz.first);
            const size_t _w = static_cast<size_t>(this->view_sz.second);
            const size_t _c = static_cast<size_t>(3);

            constexpr size_t elsize = sizeof(int);
            size_t shape[3]{_c, _h, _w};
            size_t strides[3]{_w * _h * elsize, _w * elsize, elsize};
            a = py::array_t<int>(shape, strides);
        }
        catch (const exception &e)
        {
            // Catch any C++ exceptions and convert them to Python exceptions
            throw py::value_error(e.what());
        }
    }

    /**
     * @brief Initialize the image for the episode.
     */
    void _init_random_image()
    {
        try
        {
            if (files.empty())
            {
                throw runtime_error("No files available for initialization");
            }

            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dis(0, files.size() - 1);

            dataset_index = dis(gen);

            /* pick random image from file list */
            image = VImage::new_from_file(&files[dataset_index][0], VImage::option()->set("access", VIPS_ACCESS_RANDOM));

            if (image.width() <= 0 || image.height() <= 0)
            {
                throw runtime_error("Failed to load image. Ensure that the image file is valid and accessible.");
            }
        }
        catch (const exception &e)
        {
            // Catch any C++ exceptions and convert them to Python exceptions
            throw py::value_error(e.what());
        }
    }

    /**
     * @brief Given a rectangular region within the image, prepare an array with
     * pixel values from the region.
     *
     * @param VipsRect* Pointer to a VipRect struct specifying the rectangular
     *      region to prepare as array_t.
     * @return py::array_t<int> containing pixels of the rectangular region.
     */
    py::array_t<int> get_region(VipsRect *patch)
    {
        try
        {
            auto view = a.mutable_unchecked<3>();

            VipsRegion *region;
            if (!(region = vips_region_new(image.get_image())))
                throw runtime_error("Failed to create VipsRegion");

            if (vips_region_prepare(region, patch))
            {
                throw runtime_error("Failed to prepare VipsRegion. Ensure that the region coordinates are within the image boundaries.");
            }

            for (int y = 0; y < patch->height; y++)
            {
                VipsPel *p = VIPS_REGION_ADDR(region, patch->left, patch->top + y);
                for (int x = 0; x < patch->width; x++)
                {
                    for (int b = 0; b < image.bands(); b++)
                    {
                        view(b, y, x) = *p++;
                    }
                }
            }
            return a;
        }
        catch (const exception &e)
        {
            // Catch any C++ exceptions and convert them to Python exceptions
            throw py::value_error(e.what());
        }
    }

    /**
     * @brief Reset the environment.
     *
     * @return py::tuple containing observation (py::array_t<int>) and info
     *      (py::dict).
     */
    py::tuple reset()
    {
        _init_random_image();

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0f, 1.0f);

        /* random (x, y) coordinates */
        pair<float, float> points{static_cast<float>(dis(gen)), static_cast<float>(dis(gen))};
        VipsRect patch = continuous_to_coords(points, {image.width(), image.height()}, view_sz);

        timestep = 0;

        py::dict info("timestep"_a = timestep, "target"_a = classes[dataset_index]);
        /* return (obs, info) */
        return py::make_tuple(get_region(&patch), info);
    }

    /**
     * @brief Step in the environment.
     *
     * @param py::tuple Action tuple.
     * @return py::tuple Contains
     *      next observation (py::array_t<int>), reward (int),
     *      done (py::bool_(false)), truncated (py::bool), info (py::dict)
     */
    py::tuple step(py::tuple action)
    {
        try
        {
            if (action.size() != 2)
            {
                throw invalid_argument("Action must be a tuple of size 2 and values must be in (0, 1)");
            }
            float action_x = action[0].cast<float>();
            float action_y = action[1].cast<float>();

            /* Get box based on action */
            VipsRect patch = continuous_to_coords({action_x, action_y}, {image.width(), image.height()}, view_sz);

            timestep += 1;

            py::dict info("timestep"_a = timestep, "target"_a = classes[dataset_index]);
            /* return (next_obs, reward, done, truncated, info) */
            return py::make_tuple(get_region(&patch), 0, py::bool_(false), (timestep >= max_episode_len ? py::bool_(true) : py::bool_(false)), info);
        }
        catch (const exception &e)
        {
            // Catch any C++ exceptions and convert them to Python exceptions
            throw py::value_error(e.what());
        }
    }

    /**
     * @brief Close the environment.
     *
     * @return py::none.
     */
    py::none close()
    {
        // TODO: memory cleanup!

        return py::none();
    }
};

PYBIND11_MODULE(vipsenv, m)
{
    py::class_<BaseEnv>(m, "BaseEnv")
        .def(py::init<py::dict, py::tuple, int>(), py::arg("dataset"), py::arg("view_sz"), py::arg("max_episode_len"))
        .def("reset", &BaseEnv::reset, "Reset environment.")
        .def("step", &BaseEnv::step, "Step in environment.", py::arg("action"))
        .def("close", &BaseEnv::close, "Close environment.")
        .def_readwrite("files", &BaseEnv::files)
        .def_readwrite("classes", &BaseEnv::classes)
        .def_readwrite("view_sz", &BaseEnv::view_sz)
        .def_readwrite("max_episode_len", &BaseEnv::max_episode_len)
        .def_readonly("timestep", &BaseEnv::timestep)
        .def_readwrite("dataset_index", &BaseEnv::dataset_index);
}
