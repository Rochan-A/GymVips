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

#include "vipsenv.h"
#include "envpool.h"

using namespace vips;
using namespace pybind11::literals;

namespace py = pybind11;

/** AsyncVipsEnv class
 *
 * Class binds C++ Envpool<VipsEnv> with PyBind11 types.
 */
class AsyncVipsEnv
{

public:
    /**
     * Hack to initialize variable.
     * TODO: see if there is a better way to do this.
     */
    EnvPool<VipsEnv, action_t, data_t, init_t> env_pool;
    py::array_t<uint8_t> a;
    AsyncVipsEnv(const int num_env, const py::dict dataset, const py::tuple view_sz, const int max_episode_len)
        : env_pool([num_env, &dataset, &view_sz, max_episode_len]() {
            init_t init;
            try
            {
                if (num_env <= 0 || dataset.size() == 0 || view_sz.size() != 2 || max_episode_len <= 0)
                {
                    throw std::invalid_argument("Invalid arguments. 'dataset' must not be empty, 'view_sz' must be a tuple of size 2, 'num_env' and 'max_episode_len' must be greater than 0.");
                }

                for (auto &item : dataset)
                {
                    /* dict keys are file paths, values are class_idx */
                    const std::string &key = item.first.cast<std::string>();
                    const int &value = item.second.cast<int>();

                    init.files.push_back(key);
                    init.classes.push_back(value);
                }

                init.view_sz = view_sz.cast<std::pair<int, int>>();
                init.max_episode_len = max_episode_len - 1;
                init.num_env = num_env;

                // Do any additional initialization if needed

                return init;
            }
            catch (const std::exception &e)
            {
                // Catch any C++ exceptions and convert them to Python exceptions
                throw py::value_error(e.what());
            }
        }())
    {

        // ... rest of your constructor code
        const size_t _h = static_cast<size_t>(env_pool.init.view_sz.first);
        const size_t _w = static_cast<size_t>(env_pool.init.view_sz.second);
        const size_t _c = static_cast<size_t>(3);
        const size_t _num_env = static_cast<size_t>(num_env);

        constexpr size_t elsize = sizeof(uint8_t);
        size_t shape[4]{_num_env, _c, _h, _w};
        size_t strides[4]{_c * _w * _h * elsize, _w * _h * elsize, _w * elsize, elsize};
        a = py::array_t<uint8_t>(shape, strides);
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
        auto view = a.mutable_unchecked<4>();
        std::vector<data_t> data = this->env_pool.recv();

        // Fill view with data.obs
        // for (int e = 0; e < num_env; e++){
        //     for (int )
        // }

        py::dict info;//"timestep"_a = data.info.timestep, "target"_a = data.info.target);

        return py::make_tuple(a, info);
    }

    /**
     * @brief Step in the environment.
     *
     * @param py::tuple Action tuple.
     * @return py::tuple Contains
     *      next observation (py::array_t<uint8_t>), reward (int),
     *      done (py::bool_(false)), truncated (py::bool), info (py::dict)
     */
    py::tuple step(py::tuple actions)
    {
        try
        {
            if (actions.size() != static_cast<long unsigned int>(env_pool.num_env_))
            {
                throw std::invalid_argument("Action length must be equal to the number of envs");
            }

            std::vector<action_t> act(env_pool.num_env_);
            for (int i = 0; i < env_pool.num_env_; i++){
                act[i].val = std::make_pair(actions[i].cast<float>(), actions[i].cast<float>());
            }

            this->env_pool.send(act);
            auto view = a.mutable_unchecked<4>();
            std::vector<data_t> data = this->env_pool.recv();

            py::dict info;//"timestep"_a = timestep, "target"_a = classes[dataset_index]);

            return py::make_tuple(a, 0, 0, 0, info);
        }
        catch (const std::exception &e)
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

void init(const std::string file_name)
{
    VIPS_INIT(&(file_name[0]));
}

void shutdown(void)
{
    vips_shutdown();
}

PYBIND11_MODULE(vipsenv, m)
{
    m.doc() = "AsyncVipsEnv";

    m.def("init", &init, "Initialize the Vips environment. Must be called before anything else (set file_name = sys.argv[0]).", py::arg("file_name"));
    m.def("shutdown", &shutdown, "Shutdown the Vips environment. Must be called at the end. Do not use this library beyond this point.");

    py::class_<AsyncVipsEnv>(m, "AsyncVipsEnv")
        .def(py::init<int, py::dict, py::tuple, int>(), py::arg("num_env"), py::arg("dataset"), py::arg("view_sz"), py::arg("max_episode_len"))
        .def("reset", &AsyncVipsEnv::reset, "Reset environment.")
        .def("step", &AsyncVipsEnv::step, "Step in environment.", py::arg("action"))
        .def("close", &AsyncVipsEnv::close, "Close environment.");
//        .def_readwrite("files", &AsyncVipsEnv::files)
//        .def_readwrite("classes", &AsyncVipsEnv::classes)
//        .def_readwrite("view_sz", &AsyncVipsEnv::view_sz)
//        .def_readwrite("max_episode_len", &AsyncVipsEnv::max_episode_len)
//        .def_readonly("timestep", &AsyncVipsEnv::timestep)
//        .def_readwrite("dataset_index", &AsyncVipsEnv::dataset_index);
}
