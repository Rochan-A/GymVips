import random
from tqdm import tqdm
import vipsenv

if __name__=='__main__':
    # initialize vips
    vipsenv.init(__file__)

    dataset = {"./img.jpg": 1}
    num_envs = 2
    view_sz = (256, 256)
    max_episode_len = 100

    envs = vipsenv.AsyncVipsEnv(num_envs, dataset, view_sz, max_episode_len)

    for i in tqdm(range(100)):
        obs, infos = envs.reset()
        assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

        while True:
            obs, reward, dones, truncateds, infos = envs.step(((random.random(), random.random()), (random.random(), random.random())))
            assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

            if dones.any() == True:
                break

    envs.close()
    del envs

    vipsenv.shutdown()