#include <iostream>
#include "concurrentqueue/blockingconcurrentqueue.h"

/**
 * Async EnvPool
 *
 * Represents an asynchronous environment pool designed for parallel
 * simulation of environments.
 *
 * batch-action -> action queue -> threadpool -> state queue -> buffer-state
 *
 * The environment steps asynchronously. Class manages a fixed-size pool of
 * environments.
 *
 * @tparam env_t      The type representing the environment.
 * @tparam action_t   The type representing actions to be taken in the environment.
 * @tparam data_t     The type representing the data associated with each environment.
 * @tparam init_t     The type representing the initialization parameters for environments.
 */
template <class env_t, typename action_t, typename data_t, typename init_t>
class EnvPool
{
private: // or public, depending on your design
    EnvPool(const EnvPool&) = delete;
    EnvPool& operator=(const EnvPool&) = delete;

public:
    int num_env_ = 0;
    int stop_ = 0;

    init_t init;

    // vector of envs
    std::vector<env_t> envs_;

    // vector to hold pointers instead of instances of BlockingConcurrentQueue
    std::vector<moodycamel::BlockingConcurrentQueue<action_t, moodycamel::ConcurrentQueueDefaultTraits>*> action_bcq;
    std::vector<moodycamel::BlockingConcurrentQueue<data_t, moodycamel::ConcurrentQueueDefaultTraits>*> data_bcq;

    // thread workers
    std::vector<std::thread> workers_;

    EnvPool(void){}

    /**
     * Constructor for EnvPool
     *
     * Initializes an asynchronous environment pool with the specified initialization parameters.
     *
     * @param init_params   The initialization parameters for setting up the environments.
     */
    EnvPool(const init_t init_params)
    {
        init = init_params;
        num_env_ = init_params.num_env;
        for (int i = 0; i < num_env_; i++)
        {
            envs_.emplace_back(env_t(init_params));

            // Change the initialization to use pointers
            action_bcq.emplace_back(new moodycamel::BlockingConcurrentQueue<action_t, moodycamel::ConcurrentQueueDefaultTraits>());
            data_bcq.emplace_back(new moodycamel::BlockingConcurrentQueue<data_t, moodycamel::ConcurrentQueueDefaultTraits>());
        }

        for (int i = 0; i < num_env_; i++)
        {
            workers_.emplace_back([this, i]
            {
                for (;;)
                { // runs until stop_ == 1
                    action_t raw_action;
                    action_bcq[i]->wait_dequeue(raw_action);
                    if (stop_ == 1)
                    {
                        break;
                    }

                    data_t data;
                    if (raw_action.force_reset || envs_[i].is_done())
                    {
                        data = envs_[i].reset();
                    } else {
                        data = envs_[i].step(raw_action);
                    }
                    data_bcq[i]->enqueue(data);
                }
            });
        }
    }

    /**
     * Send Method
     *
     * Enqueues a batch of actions to be processed asynchronously by the environment pool.
     *
     * @param action   A vector of actions to be processed by the environment pool.
     */
    void send(const std::vector<action_t> action)
    {
        for (int i = 0; i < num_env_; i++)
        {
            action_bcq[i]->enqueue(action[i]);
        }
    }

    /**
     * Receive Method
     *
     * Retrieves the latest batch of states resulting from the asynchronous processing
     * of actions by the environment pool.
     *
     * @return A vector of data_t representing the current states of the environments.
     */
    std::vector<data_t> recv(void)
    {
        std::vector<data_t> states(num_env_);
        for (int i = 0; i < num_env_; i++)
        {
            data_bcq[i]->wait_dequeue(states[i]);
        }
        return states;
    }

    /**
     * Reset Method
     *
     * Initiates a reset for all environments in the pool.
     */
    void reset(void)
    {
        action_t empty_action;
        empty_action.force_reset = true;
        for (int i = 0; i < num_env_; i++)
        {
            action_bcq[i]->enqueue(empty_action);
        }
    }

    ~EnvPool()
    {
        stop_ = 1;
        action_t empty_actions;
        empty_actions.force_reset = true;
        for (int i = 0; i < num_env_; i++)
        {
            action_bcq[i]->enqueue(empty_actions);
        }
        for (auto &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }
};
