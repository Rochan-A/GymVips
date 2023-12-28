#include <random>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdint>
#include <vips/vips8>

using namespace vips;

/** Keep these datatypes in C++ for portability!
 * Write functions to convert them to pybind11 datatypes
 */

/**
 * @brief ImageArray Class
 *
 * Represents a 3D array for storing image data with channels, height, and width dimensions.
 *
 * @class ImageArray
 */
class ImageArray
{
public:
    std::vector<uint8_t> array; ///< Internal storage for image data.
    int C = 0, H = 0, W = 0;    ///< Dimensions of the image array (channels, height, width).

    /**
     * @brief Default constructor for ImageArray.
     *
     * @note Use this constructor to create an ImageArray instance without specific dimensions.
     * Call the init() method to initialize the array with specific dimensions before use.
     */
    ImageArray(void) {}

    /**
     * @brief Initialize the ImageArray with specified dimensions.
     *
     * @param c Number of channels.
     * @param h Height of the image.
     * @param w Width of the image.
     *
     * @note Use this method to initialize the ImageArray with specific dimensions before use.
     */
    void init(const int c, const int h, const int w)
    {
        C = c, H = h, W = w;
        array = std::vector<uint8_t>(c * w * h);
    }

    /**
     * @brief Accessor method for getting or setting pixel values in the image array.
     *
     * @param c Channel index.
     * @param h Height index.
     * @param w Width index.
     * @return Reference to the pixel value at the specified indices.
     * @throws std::runtime_error if the indices are out of bounds.
     */
    inline uint8_t &operator()(const int c, const int h, const int w)
    {
        return this->array[(h * W + w) * C + c];
    }
};
typedef ImageArray image_t;

typedef struct action
{
    std::pair<float, float> val = std::make_pair(0.0f, 0.0f); ///< Pair of float values for the action.
    bool force_reset = false;                                 ///< Flag indicating whether a forceful reset is necessary.

    action() = default;
    action(bool force_reset) : force_reset(force_reset) {}
} action_t;

typedef struct info
{
    int timestep = 0; ///< Current timestep in the simulation.
    int target = 0;   ///< Target value associated with the simulation.

    info() = default;
    info(int timestep, int target) : timestep(timestep), target(target) {}
} info_t;

typedef struct data
{
    image_t obs;            ///< Observation Array
    float reward = 0.0f;    ///< Reward
    bool done = false;      ///< Done
    bool truncated = false; ///< Truncated Episode
    info_t info;            ///< Info

    data() = default;  // Ensure a valid default constructor
} data_t;

struct init_t
{
    std::vector<std::string> files{};                   ///< File paths
    std::vector<int> classes{};                         ///< Class Label
    std::pair<int, int> view_sz = std::make_pair(0, 0); ///< View size
    int max_episode_len = 0;                            ///< Max episode length
    int num_env = 0;                                    ///< Number of environments
};


/**
 * @brief VipsEnv Class
 *
 * Represents an environment using the VIPS image processing library.
 *
 * @class VipsEnv
 */
class VipsEnv
{

public:
    const std::vector<std::string> files; ///< File paths
    const std::vector<int> classes;       ///< Class Label
    const std::pair<int, int> view_sz;    ///< View size
    const int max_episode_len;      ///< Max episode length

    int timestep = 0;               ///< Current timestep in the simulation
    int dataset_index = -1;         ///< Index of the current dataset

    VImage image; ///< VIPS image object

    int height = 0; ///< Height of the image
    int width = 0;  ///< Width of the image
    int bands = 0;  ///< Number of bands in the image

    /**
     * @brief Constructor for VipsEnv
     *
     * @param i Initialization parameters for the environment.
     */
    VipsEnv(const init_t &i) : files(i.files), classes(i.classes), max_episode_len(i.max_episode_len), view_sz(i.view_sz) {}

    /**
     * @brief Initializes a random image from the dataset.
     *
     * @note This method initializes a random image from the dataset for the environment.
     */
    void _init_random_image()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, files.size() - 1);

        dataset_index = dis(gen);

        /* pick a random image from the file list */
        image = VImage::new_from_file(files[dataset_index].c_str(), VImage::option()->set("access", VIPS_ACCESS_RANDOM));
        height = image.height();
        width = image.width();
        bands = image.bands();
    }

    /**
     * @brief Gets a region from the current image and stores it in the provided image_t object.
     *
     * @param patch VipsRect object specifying the region to retrieve.
     * @param img Reference to the image_t object to store the retrieved region.
     */
    void get_region(VipsRect &patch, image_t &img)
    {
        VRegion v = image.region(&patch);

        img.init(this->bands, this->view_sz.first, this->view_sz.second);
        for (int y = 0; y < patch.height; y++)
        {
            VipsPel *p = v.addr(patch.left, patch.top + y);
            for (int x = 0; x < patch.width; x++)
            {
                for (int b = 0; b < this->bands; b++)
                {
                    img(b, y, x) = *p++;
                }
            }
        }
    }

    /**
     * @brief Resets the environment by initializing a random image and creating the initial data_t object.
     *
     * @return Initial data_t object representing the state after the reset.
     */
    data_t reset()
    {
        _init_random_image();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0f, 1.0f);

        // Normalize coordinates to the range [0, 1]
        VipsRect patch = VipsRect{
            static_cast<int>((width - view_sz.first) * static_cast<float>(dis(gen))),
            static_cast<int>((height - view_sz.second) * static_cast<float>(dis(gen))),
            view_sz.first,
            view_sz.second
        };

        timestep = 0;

        data_t d;
        get_region(patch, d.obs);
        d.info = info_t(timestep, classes[dataset_index]);

        return d;
    }

    /**
     * @brief Takes a step in the environment based on the provided action.
     *
     * @param action Action to take in the environment.
     * @return data_t object representing the state after the step.
     */
    data_t step(action_t action)
    {
        VipsRect patch = VipsRect{
            static_cast<int>((width - view_sz.first) * (action.val.first + 1) / 2),
            static_cast<int>((height - view_sz.second) * (action.val.second + 1) / 2),
            view_sz.first,
            view_sz.second
        };

        timestep += 1;

        data_t d;
        get_region(patch, d.obs);
        d.done = this->is_done();
        d.truncated = d.done;

        return d;
    }

    /**
     * @brief Checks if the episode is done based on the current timestep and maximum episode length.
     *
     * @return true if the episode is done, false otherwise.
     */
    bool is_done(void)
    {
        if (timestep >= max_episode_len)
        {
            return true;
        }
        return false;
    }

    /**
     * @brief Closes the environment.
     *
     * @note This method is a placeholder and currently does not perform any specific actions.
     */
    void close(void)
    {
        // TODO: memory cleanup!
    }
};
