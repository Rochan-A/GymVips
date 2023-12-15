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
    std::vector<uint8_t> array; ///< Internal storage for image data.
    int C = 0, H = 0, W = 0;    ///< Dimensions of the image array (channels, height, width).

public:
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
    uint8_t &operator()(const int c, const int h, const int w)
    {
        if ((c < 0 || c >= C) || (h < 0 || h >= H) || (w < 0 || w >= W))
        {
            throw std::runtime_error("ImageArray: Attempt to access region beyond buffer size!");
        }
        return this->array[(c * w * h) + (h * w) + (w)];
    }
};
typedef ImageArray image_t;

typedef struct action_t
{
    std::pair<float, float> val = std::make_pair(0.0f, 0.0f); ///< Pair of float values for the action.
    bool force_reset = false;                                 ///< Flag indicating whether a forceful reset is necessary.
} action_t;

typedef struct info_t
{
    int timestep = 0; ///< Current timestep in the simulation.
    int target = 0;   ///< Target value associated with the simulation.
} info_t;

typedef struct data_t
{
    image_t obs;            ///< Observation Array
    float reward = 0.0f;    ///< Reward
    bool done = false;      ///< Done
    bool truncated = false; ///< Truncated Episode
    info_t info{};          ///< Info
} data_t;

struct init_t
{
    std::vector<std::string> files{};                   ///< File paths
    std::vector<int> classes{};                         ///< Class Label
    std::pair<int, int> view_sz = std::make_pair(0, 0); ///< View size
    int max_episode_len = 100;                          ///< Max episode length
    int num_env = 0;                                    ///< Number of environments
};

/**
 * @brief Converts continuous coordinates of action to upper-left coordinates
 * given the image size and view size.
 *
 * @param action The continuous coordinates to convert.
 * @param img_sz The size of the image.
 * @param view_sz The size of the view.
 * @return VipsRect containing upper-left and rectangle height and width.
 */
VipsRect continuous_to_coords(std::pair<float, float> action, std::pair<int, int> img_sz, std::pair<int, int> view_sz)
{
    // Normalize coordinates to the range [0, 1]
    float x = (action.first + 1) / 2;
    float y = (action.second + 1) / 2;

    // Calculate upper-left coordinates
    int up_left_x = static_cast<int>((img_sz.first - view_sz.first) * x);
    int up_left_y = static_cast<int>((img_sz.second - view_sz.second) * y);

    return VipsRect{up_left_x, up_left_y, view_sz.first, view_sz.second};
}

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
    std::vector<std::string> files; ///< File paths
    std::vector<int> classes;       ///< Class Label
    std::pair<int, int> view_sz;    ///< View size
    int max_episode_len = 100;      ///< Max episode length
    int timestep = 0;               ///< Current timestep in the simulation
    int dataset_index = -1;         ///< Index of the current dataset

    VImage image; ///< VIPS image object
    bool init = true;

    int height = 0; ///< Height of the image
    int width = 0;  ///< Width of the image
    int bands = 0;  ///< Number of bands in the image

    /**
     * @brief Constructor for VipsEnv
     *
     * @param i Initialization parameters for the environment.
     */
    VipsEnv(init_t i)
    {
        files = i.files;
        classes = i.classes;
        max_episode_len = i.max_episode_len;
        view_sz = i.view_sz;
    }

    /**
     * @brief Initializes a random image from the dataset.
     *
     * @note This method initializes a random image from the dataset for the environment.
     */
    void _init_random_image()
    {
        try
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, files.size() - 1);

            dataset_index = dis(gen);

            /* pick a random image from the file list */
            image = VImage::new_from_file(&(files[dataset_index][0]), VImage::option()->set("access", VIPS_ACCESS_RANDOM));
            height = image.height();
            width = image.width();
            bands = image.bands();

            init = false;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Issue with reading Image!");
        }
    }

    /**
     * @brief Gets a region from the current image and stores it in the provided image_t object.
     *
     * @param patch VipsRect object specifying the region to retrieve.
     * @param img Reference to the image_t object to store the retrieved region.
     */
    void get_region(VipsRect *patch, image_t &img)
    {
        VRegion v = image.region(patch);

        img.init(this->bands, this->view_sz.first, this->view_sz.second);
        for (int y = 0; y < patch->height; y++)
        {
            VipsPel *p = v.addr(patch->left, patch->top + y);
            for (int x = 0; x < patch->width; x++)
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

        /* random (x, y) coordinates */
        std::pair<float, float> points{static_cast<float>(dis(gen)), static_cast<float>(dis(gen))};
        VipsRect patch = continuous_to_coords(points, {width, height}, view_sz);

        timestep = 0;

        data_t data;
        get_region(&patch, data.obs);
        info_t info;
        info.timestep = timestep;
        info.target = classes[dataset_index];
        data.info = info;

        return data;
    }

    /**
     * @brief Takes a step in the environment based on the provided action.
     *
     * @param action Action to take in the environment.
     * @return data_t object representing the state after the step.
     */
    data_t step(action_t action)
    {
        float action_x = action.val.first;
        float action_y = action.val.second;

        /* Get box based on action */
        VipsRect patch = continuous_to_coords({action_x, action_y}, {width, height}, view_sz);

        timestep += 1;

        data_t data;
        get_region(&patch, data.obs);
        data.done = this->is_done();
        info_t info;
        info.timestep = timestep;
        info.target = classes[dataset_index];
        data.info = info;

        return data;
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
