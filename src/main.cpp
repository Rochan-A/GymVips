#include <iostream>
#include <random>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdint>
#include <vips/vips8>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "envpool.h"
#include "vipsenv.h"

using namespace std;
using namespace vips;
using namespace pybind11::literals;

namespace py = pybind11;

/* AsyncVipsEnv class */
class AsyncVipsEnv
{

public:
    py::array_t<uint8_t> a;
    init_t a{};
    EnvPool<VipsEnv, action_t, data_t, init_t, 10> env_pool(a);

    AsyncVipsEnv(py::dict dataset, py::tuple view_sz, int max_episode_len)
    {

        try
        {
            if (dataset.size() == 0 || view_sz.size() != 2 || max_episode_len <= 0)
            {
                throw invalid_argument("Invalid arguments. 'dataset' must not be empty, 'view_sz' must be a tuple of size 2, and 'max_episode_len' must be greater than 0.");
            }

            init_t init;
            for (auto &item : dataset)
            {
                /* dict keys are file paths, values are class_idx */
                const string &key = item.first.cast<string>();
                const int &value = item.second.cast<int>();

                init.files.push_back(key);
                init.classes.push_back(value);
            }

            init.view_sz = view_sz.cast<pair<int, int>>();
            init.max_episode_len = max_episode_len - 1;

            this->env_pool = EnvPool<VipsEnv, action_t, data_t, init_t, 10>(init);

            const size_t _h = static_cast<size_t>(init.view_sz.first);
            const size_t _w = static_cast<size_t>(init.view_sz.second);
            const size_t _c = static_cast<size_t>(3);

            constexpr size_t elsize = sizeof(uint8_t);
            size_t shape[3]{_c, _h, _w};
            size_t strides[3]{_w * _h * elsize, _w * elsize, elsize};
            a = py::array_t<uint8_t>(shape, strides);
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
     * @return py::tuple containing observation (py::array_t<uint8_t>) and info
     *      (py::dict).
     */
    py::tuple reset()
    {
        this->env_pool.reset();
        std::vector<data_t> data = this->env_pool.recv()

        auto view = a.mutable_unchecked<3>();

        // Fill view with data.obs
        for (int i = 0; i < )

        py::dict info("timestep"_a = data.info.timestep, "target"_a = data.info.target);
        return py::make_tuple(get_region(&patch), info);
    }

    /**
     * @brief Step in the environment.
     *
     * @param py::tuple Action tuple.
     * @return py::tuple Contains
     *      next observation (py::array_t<uint8_t>), reward (int),
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

            if (dataset_index == -1)
            {
                throw runtime_error("Called step() before reset()!");
            }

            float action_x = action[0].cast<float>();
            float action_y = action[1].cast<float>();

            /* Get box based on action */
            VipsRect patch = continuous_to_coords({action_x, action_y}, {width, height}, view_sz);

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
    py::class_<AsyncVipsEnv>(m, "AsyncVipsEnv")
        .def(py::init<py::dict, py::tuple, int>(), py::arg("dataset"), py::arg("view_sz"), py::arg("max_episode_len"))
        .def("reset", &AsyncVipsEnv::reset, "Reset environment.")
        .def("step", &AsyncVipsEnv::step, "Step in environment.", py::arg("action"))
        .def("close", &AsyncVipsEnv::close, "Close environment.")
        .def_readwrite("files", &AsyncVipsEnv::files)
        .def_readwrite("classes", &AsyncVipsEnv::classes)
        .def_readwrite("view_sz", &AsyncVipsEnv::view_sz)
        .def_readwrite("max_episode_len", &AsyncVipsEnv::max_episode_len)
        .def_readonly("timestep", &AsyncVipsEnv::timestep)
        .def_readwrite("dataset_index", &AsyncVipsEnv::dataset_index);
}
