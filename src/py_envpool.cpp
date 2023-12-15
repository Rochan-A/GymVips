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

/*
template <typename dtype>
Array NumpyToArray(const py::array& arr) {
  using ArrayT = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
  ArrayT arr_t(arr);
  ShapeSpec spec(arr_t.itemsize(),
                 std::vector<int>(arr_t.shape(), arr_t.shape() + arr_t.ndim()));
  return {spec, reinterpret_cast<char*>(arr_t.mutable_data())};
}
*/


/**
 * AsyncVipsEnv class
 *
 * Class binds C++ Envpool<VipsEnv> with PyBind11 types.
 */
class AsyncVipsEnv
{
public:
    EnvPool<VipsEnv, action_t, data_t, init_t> env_pool; ///< Environment pool instance.
    py::array_t<uint8_t> a;                              ///< Numpy array for observations.

    /**
     * Constructor for AsyncVipsEnv.
     *
     * @param num_env Number of environments in the pool.
     * @param dataset Dictionary containing file paths and corresponding class indices.
     * @param view_sz Tuple representing the view size (height, width).
     * @param max_episode_len Maximum length of an episode.
     */
    AsyncVipsEnv(const int num_env, const py::dict dataset, const py::tuple view_sz, const int max_episode_len)
        : env_pool([num_env, &dataset, &view_sz, max_episode_len]()
                   {
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

                           return init;
                       }
                       catch (const std::exception &e)
                       {
                           // Catch any C++ exceptions and convert them to Python exceptions
                           throw py::value_error(e.what());
                       } }())
    {
        // Initialize numpy array for observations
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
     * Reset the environment.
     *
     * @return py::tuple containing observation (py::array_t<uint8_t>) and info (py::dict).
     */
    py::tuple reset()
    {
        this->env_pool.reset();
        auto view = a.mutable_unchecked<4>();
        std::vector<data_t> data = this->env_pool.recv();

        // Create info dictionary
        py::dict info_dict;

        // Stack observations arrays
        for (int e = 0; e < env_pool.num_env_; e++)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < env_pool.init.view_sz.first; h++)
                {
                    for (int w = 0; w < env_pool.init.view_sz.second; w++)
                    {
                        view(e, c, h, w) = data[e].obs(c, h, w);
                    }
                }
            }
            std::string key = "env_" + std::to_string(e);
            info_dict[&key[0]] = py::dict("timestep"_a = static_cast<int>(data[e].info.timestep), "target"_a = static_cast<int>(data[e].info.target));
        }

        return py::make_tuple(a, info_dict);
    }

    /**
     * Step in the environment.
     *
     * @param actions Nested action tuple.
     * @return py::tuple Contains next observation (py::array_t<uint8_t>), reward (int),
     *      done (py::bool_(false)), truncated (py::bool), info (py::dict).
     */
    py::tuple step(py::tuple actions)
    {
        try
        {
            // Check if the outer tuple size matches the number of environments
            if (actions.size() != static_cast<long unsigned int>(env_pool.num_env_))
            {
                throw std::invalid_argument("Outer tuple length must be equal to the number of envs");
            }

            std::vector<action_t> act(env_pool.num_env_);

            // Iterate through the outer tuple
            for (int i = 0; i < env_pool.num_env_; i++)
            {
                py::tuple innerTuple = actions[i].cast<py::tuple>();

                // Check if the inner tuple size matches your inner structure
                if (innerTuple.size() != 2)
                {
                    throw std::invalid_argument("Inner tuple length must be 2");
                }

                // Extract values from the inner tuple
                act[i].val = std::make_pair(innerTuple[0].cast<float>(), innerTuple[1].cast<float>());
            }

            this->env_pool.send(act);
            auto view = a.mutable_unchecked<4>();
            std::vector<data_t> data = this->env_pool.recv();

            // Create reward, done, and truncated arrays
            py::array_t<float> rewards({env_pool.num_env_}, {sizeof(float)});
            py::array_t<bool> dones({env_pool.num_env_}, {sizeof(bool)});
            py::array_t<bool> truncated({env_pool.num_env_}, {sizeof(bool)});

            auto rewards_unchecked = rewards.mutable_unchecked<1>();
            auto dones_unchecked = dones.mutable_unchecked<1>();
            auto truncated_unchecked = truncated.mutable_unchecked<1>();

            // Create info dictionary
            py::dict info_dict;

            // Stack observations arrays
            for (int e = 0; e < env_pool.num_env_; e++)
            {
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < env_pool.init.view_sz.first; h++)
                    {
                        for (int w = 0; w < env_pool.init.view_sz.second; w++)
                        {
                            view(e, c, h, w) = data[e].obs(c, h, w);
                        }
                    }
                }
                // Populate reward, done, and truncated arrays
                rewards_unchecked(e) = data[e].reward;
                dones_unchecked(e) = data[e].done;
                truncated_unchecked(e) = data[e].truncated;

                std::string key = "env_" + std::to_string(e);
                info_dict[&key[0]] = py::dict("timestep"_a = static_cast<int>(data[e].info.timestep), "target"_a = static_cast<int>(data[e].info.target));
            }

            return py::make_tuple(a, rewards, dones, truncated, info_dict);
        }
        catch (const std::exception &e)
        {
            // Catch any C++ exceptions and convert them to Python exceptions
            throw py::value_error(e.what());
        }
    }

    /**
     * py api
     */
/*
  void PySend(const std::vector<py::array>& action) {
        std::vector<action_t> act;
        act.reserve(action.size());
        ToAction(action, &act);
        py::gil_scoped_release release;
        this->env_pool.send(act);
    }
*/
    /**
     * py api
     */
/*
    std::vector<py::array> PyRecv() {
        std::vector<data_t> data;
        {
            py::gil_scoped_release release;
            data = this->env_pool.recv();
            DCHECK_EQ(data.size(), this->env_pool.num_env_);
        }
        std::vector<py::array> ret;
        ret.reserve(this->env_pool.num_env_);
        ToNumpy(data, &ret);
        return ret;
    }
*/
    /**
     * py api
     */
/*
    void PyReset(const py::array& env_ids) {
        auto arr = NumpyToArray<int>(env_ids);
        py::gil_scoped_release release;
        this->env_pool.reset(arr);
    }
*/
    /**
     * Close the environment.
     *
     * @return py::none.
     */
    py::none close()
    {
        // TODO: memory cleanup!

        return py::none();
    }
};

/**
 * Initialize the Vips environment. Must be called before anything else.
 *
 * @param file_name Name of the file to initialize Vips.
 */
void init(const std::string file_name)
{
    VIPS_INIT(&(file_name[0]));
}

/**
 * Shutdown the Vips environment. Must be called at the end.
 * Do not use this library beyond this point.
 */
void shutdown(void)
{
    vips_shutdown();
}

/**
 * Pybind11 module definition for the AsyncVipsEnv class.
 */
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
}
