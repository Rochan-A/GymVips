#include <chrono>

#include "envpool.h"
#include "vipsenv.h"

/**
 * @brief Main function to demonstrate the usage of the asynchronous environment pool.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit code.
 */
int main(int argc, char **argv)
{
    // Initialize the Vips environment
    if (VIPS_INIT(argv[0]))
        vips_error_exit(NULL);

    // Set the number of environments
    const size_t num_env = 8;

    // Initialize parameters for environment pool
    init_t i;
    i.classes = {0, 1};
    std::string f_name = std::to_string(argv[1]);
    i.files = {f_name, f_name};
    i.view_sz = std::make_pair(256, 256);
    i.num_env = num_env;

    // Create an environment pool
    EnvPool<VipsEnv, action_t, data_t, init_t> pool(i);
    pool.reset();

    // Receive initial data from the environment pool
    std::vector<data_t> data = pool.recv();
    std::vector<action_t> act(num_env);

    // Set up timers for performance measurement
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    auto t2 = high_resolution_clock::now();
    /* Getting the number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    // Perform 1 Million steps
    int step = num_env;
    while (step < 1'000'000)
    {
        // Send actions and receive data from the environment pool
        pool.send(act);
        data = pool.recv();
        step += num_env;

        // Print progress every 10,000 steps
        if (step % 10'000 == 0)
        {
            t2 = high_resolution_clock::now();

            /* Getting the number of milliseconds as an integer. */
            ms_int = duration_cast<milliseconds>(t2 - t1);

            std::cout << step << " , " << ms_int.count() << "ms\n";
        }
    }
    t2 = high_resolution_clock::now();

    /* Getting the number of milliseconds as an integer. */
    ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting the number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    // Print the total time taken in milliseconds
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    // Shutdown the Vips environment
    vips_shutdown();

    return 0;
}
