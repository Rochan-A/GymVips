#include <iostream>
#include "concurrentqueue/blockingconcurrentqueue.h"

// action_t must expose force_reset of type bool!
template <class env_t, typename action_t, typename data_t, typename init_t, size_t size>
class EnvPool
{
public:
    const int num_envs_ = size;
    int stop_ = 0;

    // vector of envs
    std::vector<env_t> envs_;
    // buffers for action returned by policy
    std::vector<moodycamel::BlockingConcurrentQueue<action_t>> action_bcq;
    // buffers for data returned by env
    std::vector<moodycamel::BlockingConcurrentQueue<data_t>> data_bcq;
    // thread workers
    std::vector<std::thread> workers_;

    EnvPool(init_t init)
    {
        for (int i = 0; i < num_envs_; i++)
        {
            envs_.emplace_back(env_t(init));
            action_bcq.emplace_back(moodycamel::BlockingConcurrentQueue<action_t>());
            data_bcq.emplace_back(moodycamel::BlockingConcurrentQueue<data_t>());
        }

        for (int i = 0; i < num_envs_; i++)
        {
            workers_.emplace_back([this, i]
                                  {
                for (;;)
                { // runs infinitely until stop_ == 1
                    std::cout << "Start of thread loop" << std::endl;
                    action_t raw_action;
                    std::cout << "Waiting to receive action" << std::endl;
                    action_bcq[i].wait_dequeue(raw_action);
                    if (stop_ == 1)
                    {
                        break;
                    }
                    std::cout << "Got action" << std::endl;

                    data_t data;
                    if (raw_action.force_reset || envs_[i].is_done())
                    {
                        data = envs_[i].reset();
                        std::cout << "Reset env" << std::endl;
                    } else {
                        data = envs_[i].step(raw_action);
                        std::cout << "Stepped env" << std::endl;
                    }
                    data_bcq[i].enqueue(data);
                    std::cout << "Enqueued data" << std::endl;
                }
            });
        }
    }

    void send(const std::vector<action_t> action)
    {
        for (int i = 0; i < num_envs_; i++)
        {
            std::cout << "Adding to: " << i << std::endl;
            action_bcq[i].enqueue(action[i]);
            std::cout << "Added to: " << i << std::endl;
        }
    }

    std::vector<data_t> recv(void)
    {
        std::vector<data_t> states(num_envs_);
        for (int i = 0; i < num_envs_; i++)
        {
            data_bcq[i].wait_dequeue(states[i]);
        }
        return states;
    }

    void reset(void)
    {
        action_t empty_action;
        empty_action.force_reset = true;
        for (int i = 0; i < num_envs_; i++)
        {
            action_bcq[i].enqueue(empty_action);
        }
    }

    ~EnvPool()
    {
        stop_ = 1;
        action_t empty_actions;
        empty_actions.force_reset = true;
        for (int i = 0; i < num_envs_; i++)
        {
            action_bcq[i].enqueue(empty_actions);
        }
        for (auto &worker : workers_)
        {
            worker.join();
        }
    }
};