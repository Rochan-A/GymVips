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

template <typename action_t>
void ToActionT(const std::vector<py::array> &py_arrs, std::vector<action_t> &ret)
{
    // Iterate through the outer tuple
    for (int i = 0; i < py_arrs.size(); ++i)
    {
        py::tuple innerTuple = py_arrs[i].cast<py::tuple>();

        // Extract values from the inner tuple
        ret[i].val = std::make_pair(innerTuple[0].cast<float>(), innerTuple[1].cast<float>());
    }
}

py::array_t<float> convert_to_pyarray(const std::vector<uint8_t> &input_vector, size_t height, size_t width, size_t num_channels)
{
    // Ensure the vector size matches the expected size
    size_t expected_size = height * width * num_channels;
    if (input_vector.size() != expected_size)
    {
        throw std::runtime_error("Input vector size does not match the expected size.");
    }

    // Assuming the data is stored contiguously
    std::vector<size_t> shape = {num_channels, height, width};
    std::vector<size_t> strides = {height * width * sizeof(uint8_t), width * sizeof(uint8_t), sizeof(uint8_t)};

    // Create a new py::array_t<float> with the correct dimensions and strides
    // The data pointer is obtained from the address of the first element of the input vector
    return py::array_t<uint8_t>(
        shape,
        strides,
        input_vector.data() // pointer to the first element of the input vector
    );
}

void ToNumpy(std::vector<py::array> &dat, const std::vector<data_t> &input_vector, size_t height, size_t width, size_t num_channels)
{
    for (int i = 0; i < input_vector.size(); ++i)
    {
        dat[i] = convert_to_pyarray(input_vector[i].obs.array, height, width, num_channels);
    }
}


/**
 * AsyncVipsEnv class
 *
 * Class binds C++ Envpool<VipsEnv> with PyBind11 types.
 */
class AsyncVipsEnv
{
public:
    EnvPool<VipsEnv, action_t, data_t, init_t> env_pool; ///< Environment pool instance.

    /**
     * Constructor for AsyncVipsEnv.
     *
     * @param num_env Number of environments in the pool.
     * @param dataset Dictionary containing file paths and corresponding class indices.
     * @param view_sz Tuple representing the view size (height, width).
     * @param max_episode_len Maximum length of an episode.
     */
    AsyncVipsEnv(const int &num_env, const py::dict &dataset, const py::tuple &view_sz, const int &max_episode_len)
        : env_pool([num_env, dataset, view_sz, max_episode_len]()
                   {
                        init_t init;
                        for (auto &item : dataset)
                        {
                            /* dict keys are file paths, values are class_idx */
                            const std::string &key = item.first.cast<std::string>();
                            const int &value = item.second.cast<int>();

                            init.files.emplace_back(key);
                            init.classes.emplace_back(value);
                        }

                        init.view_sz = view_sz.cast<std::pair<int, int>>();
                        init.max_episode_len = max_episode_len - 1;
                        init.num_env = num_env;

                        return init; }()) {}

    /**
     * py api
     */
    void PySend(const std::vector<py::array> &action)
    {
        std::vector<action_t> arr(env_pool.num_env_);
        ToActionT(action, arr);
        py::gil_scoped_release release;
        env_pool.send(arr); // delegate to the c++ api
    }

    /**
     * py api
     */
    std::vector<py::array> PyRecv(void)
    {
        py::gil_scoped_release release;

        std::vector<data_t> arr(env_pool.num_env_);
        {
            arr = env_pool.recv();
        }

        // // Create reward, done, and truncated arrays
        // py::array_t<float> rewards({env_pool.num_env_}, {sizeof(float)});
        // py::array_t<bool> dones({env_pool.num_env_}, {sizeof(bool)});
        // py::array_t<bool> truncated({env_pool.num_env_}, {sizeof(bool)});

        // auto rewards_unchecked = rewards.mutable_unchecked<1>();
        // auto dones_unchecked = dones.mutable_unchecked<1>();
        // auto truncated_unchecked = truncated.mutable_unchecked<1>();

        // Create info dictionary
        // py::dict info_dict;
        std::vector<py::array> ret;
        ret.reserve(env_pool.num_env_);
        // py::gil_scoped_acquire acquire;
        // ToNumpy(ret, arr, (size_t) env_pool.init.view_sz.first, (size_t) env_pool.init.view_sz.second, 3);

        // // Stack observations arrays
        // for (int e = 0; e < env_pool.num_env_; e++)
        // {
        //     rewards_unchecked(e) = arr[e].reward;
        //     dones_unchecked(e) = arr[e].done;
        //     truncated_unchecked(e) = arr[e].truncated;

        //     std::string key = "env_" + std::to_string(e);
        //     info_dict[&key[0]] = py::dict("timestep"_a = static_cast<int>(arr[e].info.timestep), "target"_a = static_cast<int>(arr[e].info.target));
        // }

        // return py::make_tuple(ret, info_dict);
        return ret;
    }

    /**
     * py api
     */
    void PyReset(void)
    {
        py::gil_scoped_release release;
        env_pool.reset();
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
PYBIND11_MODULE(compiled, m)
{
    m.doc() = "AsyncVipsEnv";

    m.def("init", &init, "Initialize the Vips environment. Must be called before anything else (set file_name = sys.argv[0]).", py::arg("file_name"));
    m.def("shutdown", &shutdown, "Shutdown the Vips environment. Must be called at the end. Do not use this library beyond this point.");

    py::class_<AsyncVipsEnv>(m, "AsyncVipsEnv")
        .def(py::init<int, py::dict, py::tuple, int>(), py::arg("num_env"), py::arg("dataset"), py::arg("view_sz"), py::arg("max_episode_len"))
        .def("send", &AsyncVipsEnv::PySend, "Send action vector to environment pool.", py::arg("action"))
        .def("recv", &AsyncVipsEnv::PyRecv, "Receive step from environment pool.")
        .def("reset", &AsyncVipsEnv::PyReset, "Reset environment pool.");
}
