import time
import random
from tqdm import tqdm
from vipsenvpool import vipsenv
import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

try:
    import pyvips
    PYVIPS = True
except ImportError:
    PYVIPS = False
    print(f"Missing Pyvips, continuing to benchmark OpenCV and AsyncVipsEnv!")


def continuous_to_coords(action, img_sz, view_sz):
    x, y = action
    x, y = (x + 1) / 2, (y + 1) / 2

    up_left = [
        int((img_sz[0] - view_sz[0]) * x),
        int((img_sz[1] - view_sz[1]) * y),
    ]
    lower_right = [up_left[0] + view_sz[0], up_left[1] + view_sz[1]]

    return [up_left, lower_right]


class Cv2Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dataset, view_sz, max_episode_len):
        super().__init__()
        self.np_random = np.random.default_rng(0)
        self.seed = lambda seed: seed
        self.render_mode = "rgb_array"

        self.max_episode_len = max_episode_len
        self.view_sz = view_sz
        self.action_space = Box(-1, 1, (2,), dtype=np.float32)

        self.observation_space = Box(
            0,
            1,
            (3, *self.view_sz),
            dtype=np.float32,
        )

        self.files = dataset

    def reset(self):
        """Reset environment."""
        # Get next image, target, and objects from iterator
        img_pth = self.np_random.choice(list(self.files.keys()))
        self.A, self.target = cv2.imread(img_pth), self.files[img_pth]

        # Reset env variables
        self.done = False
        self.step_count = 1

        action = self.action_space.sample()
        up_left, lower_right = continuous_to_coords(
            action, self.A.shape, self.view_sz
        )
        crop = self.A[up_left[0] : lower_right[0], up_left[1] : lower_right[1], :]
        self.frame = np.transpose(crop, (2, 0, 1)).astype(np.float32)

        return self.frame, {
            "target": self.target,
        }

    def render(self, mode="human"):
        """Render region"""
        pass

    def close(self):
        """Close the environment"""
        return super().close()

    def _terminate(self, up_left, lower_right):
        self.done = True
        return (
            self.frame,
            0,
            self.done,
            True,
            {
                "target": self.target,
            },
        )

    def step(self, action):
        self.step_count += 1

        up_left, lower_right = continuous_to_coords(
            action, self.A.shape, self.view_sz
        )
        crop = self.A[up_left[0] : lower_right[0], up_left[1] : lower_right[1], :]
        self.frame = np.transpose(crop, (2, 0, 1)).astype(np.float32)

        if self.max_episode_len == self.step_count:
            return self._terminate(up_left, lower_right)

        return (
            self.frame,
            0,
            self.done,
            False,
            {
                "target": self.target,
            },
        )

    def render(self, mode="human"):
        return (self._render_copy * 255).astype(np.uint8)


def st_time(func):
    """Simple decorator to calculate total function exection time"""
    def st_func(*args, **kwargs):
        t1 = time.time()
        r = func(*args, **kwargs)
        t2 = time.time()
        print(f"Function = {func.__name__}, Time = {t2 - t1}")
        return r
    return st_func

@st_time
def async_vips_env(num_episodes, num_envs, dataset, view_sz, max_episode_len):
    # initialize vips
    # vipsenv.init(__file__)

    envs = vipsenv.VipsEnvPool(num_envs, dataset, view_sz, max_episode_len)

    for i in tqdm(range(num_episodes)):
        obs, infos = envs.reset()
        assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

        while True:
            obs, reward, dones, truncateds, infos = envs.step(np.random.rand(num_envs, 2))
            assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

            if dones.any() == True:
                break

    envs.close()
    # vipsenv.shutdown()

@st_time
def async_cv2_env(num_episodes, num_envs, dataset, view_sz, max_episode_len):
    envs = gym.vector.AsyncVectorEnv([lambda: Cv2Env(dataset, view_sz, max_episode_len) for _ in range(num_envs)])

    for i in tqdm(range(num_episodes)):
        obs, infos = envs.reset()
        assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

        while True:
            obs, reward, dones, truncateds, infos = envs.step(tuple([(random.random(), random.random()) for _ in range(num_envs)]))
            assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

            if dones.any() == True:
                break

    envs.close()
    del envs

if PYVIPS:
    class PyVipsEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, dataset, view_sz, max_episode_len):
            super().__init__()
            self.np_random = np.random.default_rng(0)
            self.seed = lambda seed: seed
            self.render_mode = "rgb_array"

            self.max_episode_len = max_episode_len
            self.view_sz = view_sz
            self.action_space = Box(-1, 1, (2,), dtype=np.float32)

            self.observation_space = Box(
                0,
                1,
                (3, *self.view_sz),
                dtype=np.float32,
            )

            self.files = dataset

        def reset(self):
            """Reset environment."""
            # Get next image, target, and objects from iterator
            img_pth = self.np_random.choice(list(self.files.keys()))
            self.A, self.target = pyvips.Image.new_from_file(img_pth, access='random'), self.files[img_pth]

            self.regs = pyvips.Region.new(self.A)

            # Reset env variables
            self.done = False
            self.step_count = 1

            action = self.action_space.sample()
            up_left, lower_right = continuous_to_coords(
                action, (self.A.width, self.A.height), self.view_sz
            )
            crop = np.array(self.regs.fetch(up_left[0], up_left[1], self.view_sz[0], self.view_sz[1])).reshape(*self.view_sz, 3)
            #self.A[up_left[0] : lower_right[0], up_left[1] : lower_right[1], :]

            self.frame = np.transpose(crop, (2, 0, 1)).astype(np.float32)

            return self.frame, {
                "target": self.target,
            }

        def render(self, mode="human"):
            """Render region"""
            pass

        def close(self):
            """Close the environment"""
            return super().close()

        def _terminate(self, up_left, lower_right):
            self.done = True
            return (
                self.frame,
                0,
                self.done,
                True,
                {
                    "target": self.target,
                },
            )

        def step(self, action):
            self.step_count += 1

            up_left, lower_right = continuous_to_coords(
                action, (self.A.width, self.A.height), self.view_sz
            )
            # crop = self.A[up_left[0] : lower_right[0], up_left[1] : lower_right[1], :]
            crop = np.array(self.regs.fetch(up_left[0], up_left[1], self.view_sz[0], self.view_sz[1])).reshape(*self.view_sz, 3)

            self.frame = np.transpose(crop, (2, 0, 1)).astype(np.float32)

            if self.max_episode_len == self.step_count:
                return self._terminate(up_left, lower_right)

            return (
                self.frame,
                0,
                self.done,
                False,
                {
                    "target": self.target,
                },
            )

        def render(self, mode="human"):
            return (self._render_copy * 255).astype(np.uint8)


    @st_time
    def async_pyvips_env(num_episodes, num_envs, dataset, view_sz, max_episode_len):
        envs = gym.vector.AsyncVectorEnv([lambda: PyVipsEnv(dataset, view_sz, max_episode_len) for _ in range(num_envs)])

        for i in tqdm(range(num_episodes)):
            obs, infos = envs.reset()
            assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

            while True:
                obs, reward, dones, truncateds, infos = envs.step(tuple([(random.random(), random.random()) for _ in range(num_envs)]))
                assert obs.shape == (num_envs, 3, *view_sz), f"Obs size did not match! Expected {(num_envs, 3, *view_sz)}, got {obs.shape}!"

                if dones.any() == True:
                    break

        envs.close()
        del envs


if __name__=='__main__':
    dataset = {"/workspace/rochan/c_envs/000920ad0b612851f8e01bcc880d9b3d.tiff": 1, "/workspace/rochan/c_envs/000920ad0b612851f8e01bcc880d9b3d.tiff": 1}
    num_envs = 45
    view_sz = (256, 256)
    max_episode_len = 100
    
    num_episodes = 5

    async_vips_env(num_episodes, num_envs, dataset, view_sz, max_episode_len)

    # async_cv2_env(num_episodes, num_envs, dataset, view_sz, max_episode_len)

    # if PYVIPS:
    #     async_pyvips_env(num_episodes, num_envs, dataset, view_sz, max_episode_len)