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
 * @tparam env_t      The type representing the environment.
 * @tparam action_t   The type representing actions to be taken in the environment.
 * @tparam data_t     The type representing the data associated with each environment.
 * @tparam init_t     The type representing the initialization parameters for environments.
 *
 * @details The environment steps asynchronously, and the class manages a fixed-size pool of environments.
 * It orchestrates the flow from batched actions to action queues, thread pools, state queues, and buffer states.
 *
 * @note This class is templated to accommodate various types for environments, actions, data, and initialization parameters.
 * The private section contains deleted copy constructor and assignment operator to prevent unintended copies.
 *
 * @see EnvPool::~EnvPool() for the controlled shutdown of the environment pool.
 * @see EnvPool::send() for sending actions to the environment pool.
 * @see EnvPool::recv() for retrieving states resulting from asynchronous processing.
 * @see EnvPool::reset() for initiating a reset in a controlled manner.
 */
template <class env_t, typename action_t, typename data_t, typename init_t>
class EnvPool
{
private:
    EnvPool(const EnvPool &) = delete;
    EnvPool &operator=(const EnvPool &) = delete;

public:
    int num_env_ = 0; /**< Number of environments in the pool. */
    int stop_ = 0;    /**< Flag to signal worker threads to stop processing. */

    init_t init; /**< Initialization parameters for setting up the environments. */

    // vector of envs
    std::vector<env_t> envs_; /**< Vector of environments in the pool. */

    // vector to hold pointers instead of instances of BlockingConcurrentQueue
    std::vector<moodycamel::BlockingConcurrentQueue<action_t, moodycamel::ConcurrentQueueDefaultTraits> *> action_bcq; /**< Vector of action queues for each environment. */
    std::vector<moodycamel::BlockingConcurrentQueue<data_t, moodycamel::ConcurrentQueueDefaultTraits> *> data_bcq;     /**< Vector of data queues for each environment. */

    // thread workers
    std::vector<std::thread> workers_; /**< Vector of worker threads for processing actions asynchronously. */

    /**
     * @brief Default constructor for EnvPool
     *
     * Constructs an EnvPool instance with default values.
     *
     * @note Use this constructor when creating an EnvPool without specific initialization parameters.
     * The instance must be properly initialized before use.
     */
    EnvPool(void) {}

    /**
     * @brief Constructor for EnvPool
     *
     * Initializes an asynchronous environment pool with the specified initialization parameters.
     *
     * @param init_params   The initialization parameters for setting up the environments.
     *
     * @note Use this constructor to create an instance of EnvPool with the given initialization parameters.
     * The constructor initializes environments, action queues, and data queues based on the provided parameters.
     * It also creates worker threads to process actions asynchronously.
     *
     * @see EnvPool::~EnvPool() for the controlled shutdown of the environment pool.
     * @see EnvPool::send() for sending actions to the environment pool.
     * @see EnvPool::recv() for retrieving states resulting from asynchronous processing.
     * @see EnvPool::reset() for initiating a reset in a controlled manner.
     */
    EnvPool(const init_t init_params)
    {
        init = init_params;
        num_env_ = init_params.num_env;

        // Initialize environments, action queues, and data queues
        for (int i = 0; i < num_env_; i++)
        {
            envs_.emplace_back(env_t(init_params));

            // Change the initialization to use pointers
            action_bcq.emplace_back(new moodycamel::BlockingConcurrentQueue<action_t, moodycamel::ConcurrentQueueDefaultTraits>());
            data_bcq.emplace_back(new moodycamel::BlockingConcurrentQueue<data_t, moodycamel::ConcurrentQueueDefaultTraits>());
        }

        // Create worker threads for asynchronous processing of actions
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
                } });
        }
    }

    /**
     * @brief Send Method
     *
     * Enqueues a batch of actions to be processed asynchronously by the environment pool.
     *
     * @param action   A vector of actions to be processed by the environment pool.
     *
     * @note Use this method to send a batch of actions for asynchronous processing
     * by the environments in the pool.
     *
     * @see EnvPool::recv() for retrieving states resulting from the asynchronous processing.
     * @see EnvPool::reset() for initiating a reset in a controlled manner.
     * @see EnvPool::~EnvPool() for the controlled shutdown of the environment pool.
     */
    void send(const std::vector<action_t> action)
    {
        for (int i = 0; i < num_env_; i++)
        {
            action_bcq[i]->enqueue(action[i]);
        }
    }

    /**
     * @brief Receive Method
     *
     * Retrieves the latest batch of states resulting from asynchronous processing
     * of actions by the environment pool.
     *
     * @return A vector of data_t representing the current states of the environments.
     *
     * @note Use this method to obtain the current states of all environments
     * after sending actions using EnvPool::send().
     *
     * @see EnvPool::send() for sending actions to the environment pool.
     * @see EnvPool::reset() for initiating a reset in a controlled manner.
     * @see EnvPool::~EnvPool() for the controlled shutdown of the environment pool.
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
     * @brief Reset Method
     *
     * Initiates a reset for all environments in the pool by enqueuing
     * actions with the force_reset flag set for each environment.
     *
     * @note Use this method to trigger a controlled reset of environments.
     *
     * @see EnvPool::~EnvPool() for the controlled shutdown of the environment pool.
     * @see EnvPool::send() for sending actions to the environment pool.
     * @see EnvPool::recv() for receiving states from the environment pool.
     */
    void reset(void)
    {
        action_t empty_action{.force_reset = true};

        for (int i = 0; i < num_env_; i++)
        {
            action_bcq[i]->enqueue(empty_action);
        }
    }

    /**
     * @brief Destructor for EnvPool
     *
     * Initiates a controlled shutdown of the asynchronous environment pool.
     * Signals worker threads to stop, enqueues termination actions, and joins threads.
     *
     * @note Call explicitly to ensure proper resource release and shutdown.
     *
     * @warning Do not use the EnvPool instance after calling the destructor.
     */
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
