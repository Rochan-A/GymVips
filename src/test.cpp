#include "envpool.h"
#include <chrono>
#include "vipsenv.h"

int main(int argc, char **argv){
    if (VIPS_INIT (argv[0])) 
        vips_error_exit (NULL);

    const size_t num_env = 8;

    init_t i;
    i.classes = {0, 1};
    i.files = {"../../20220930_230209.jpeg", "../../20220930_230209_1.jpeg"};
    i.view_sz = std::make_pair(256, 256);
    i.t_idx = 0;

    EnvPool<VipsEnv, action_t, data_t, init_t, num_env> pool(i);

    pool.reset();

    std::vector<data_t> data = pool.recv();
    std::vector<action_t> act(num_env);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int step = num_env;

    auto t1 = high_resolution_clock::now();
    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    // 1 Million steps
    while (step < 1'000'000){
        pool.send(act);
        data = pool.recv();
        step += num_env;
        if (step % 10'000 == 0){
            t2 = high_resolution_clock::now();

            /* Getting number of milliseconds as an integer. */
            ms_int = duration_cast<milliseconds>(t2 - t1);

            std::cout << step << " , " << ms_int.count() << "ms\n";
        }
    }
    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    vips_shutdown();

    return 0;
}