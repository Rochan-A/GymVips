import pprint
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import gymnasium as gym

from .envpool import EnvPool
from vipsenvpool.compiled import AsyncVipsEnv as _AsyncVipsEnvCPP
from vipsenvpool.compiled import init, shutdown

init(__file__)

class VipsEnvPool(EnvPool):
    def __init__(self, num_envs: int, dataset: dict, view_sz: tuple, max_episode_len: int) -> None:
        assert num_envs > 0, f"Number envs must be >= 1, got {num_envs}!"
        assert isinstance(dataset, dict), f"dataset must be of type dict, got {type(dataset)}!"
        assert len(dataset) > 0, f"Got empty dataset!"
        for k, v in dataset.items():
            if not (isinstance(k, str) and isinstance(v, int)):
                raise ValueError(f"dataset key, value pair must be filename (type str) and class id (type int)!")

        assert isinstance(view_sz, tuple) and len(view_sz) == 2, f"view_sz must be (height, width) tuple of integer values, got {type(view_sz)}, with element type(s) {[type(i) for i in view_sz]} and shape {len(view_sz)}"
        view_sz = tuple([int(i) for i in view_sz])
        assert isinstance(max_episode_len, int) and max_episode_len > 1, f"max_episode_len must be integer >= 2!"

        self.config = {
            "num_envs": num_envs,
            "dataset_size": len(dataset),
            "view_sz": view_sz,
            "max_episode_len": max_episode_len 
        }
        self.action_array_spec = {str(i): np.zeros((2,), dtype=np.float32) for i in range(num_envs)}
        self._cpp_cls = _AsyncVipsEnvCPP(num_envs, dataset, view_sz, max_episode_len)

    def _check_action(self, actions: List[np.ndarray]) -> None:
        for a, (k, v) in zip(actions, self.action_array_spec.items()):
            if v.dtype != a.dtype:
                raise RuntimeError(
                    f'Expected dtype {v.dtype} with action "{k}", got {a.dtype}'
                )
            shape = tuple(v.shape)
            if shape != a.shape:
                raise RuntimeError(
                    f"Expected shape {shape} with action \"{k}\", "
                    f"got {a.shape}"
                )

    def _to(
      self: Any, state_values: List[np.ndarray], reset: bool
    ) -> Union[
      Any,
      Tuple[Any, Any],
      Tuple[Any, np.ndarray, np.ndarray, Any],
      Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Any],
    ]:
      info = {}
      if reset:
        return np.zeros((self.config['num_envs'], 3, 256, 256)), info
      terminated = False if self._step < 100 else True
      terminated = np.array([terminated for i in range(self.config['num_envs'])], dtype=np.bool_)
      return np.zeros((self.config['num_envs'], 3, 256, 256)), np.zeros((self.config['num_envs'], 1), dtype=np.float32), terminated, terminated, info

    def _from(
        self,
        action: Union[Dict[str, Any], np.ndarray],
        env_id: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Convert action to C++-acceptable format."""
        assert isinstance(action, np.ndarray), f"Expected action of type np.ndarray, got {type(action)}"
        return list(map(lambda x: x.astype(np.float32), action))  # type: ignore

    def __len__(self) -> int:
        """Return the number of environments."""
        return self.config["num_envs"]

    @property
    def all_env_ids(self) -> np.ndarray:
        """All env_id in numpy ndarray with dtype=np.int32."""
        if not hasattr(self, "_all_env_ids"):
            self._all_env_ids = np.arange(self.config["num_envs"], dtype=np.int32)
        return self._all_env_ids  # type: ignore

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Set the seed for all environments (abandoned)."""
        warnings.warn("The `seed` function in envpool is abandoned. ", stacklevel=2)

    def send(
        self,
        action: Union[Dict[str, Any], np.ndarray],
        env_id: Optional[np.ndarray] = None,
    ) -> None:
        """Send actions into EnvPool."""
        action = self._from(action, env_id)
        self._check_action(action)
        self._cpp_cls.send(action)

    def recv(
        self,
        reset: bool = False,
    ) -> Tuple:
        """Recv a batch state from EnvPool."""
        state_list = self._cpp_cls.recv()
        return self._to(state_list, reset)

    def step(
        self,
        action: Union[Dict[str, Any], np.ndarray],
        env_id: Optional[np.ndarray] = None,
    ) -> Tuple:
        """Perform one step with multiple environments in EnvPool."""
        # TODO: remove
        self._step += 1
        self.send(action, env_id)
        return self.recv(reset=False)

    def reset(
        self,
    ) -> Tuple:
        """Follows the async semantics, reset the envs in env_ids."""
        # TODO: Remove later
        self._step = 0
        self._cpp_cls.reset()
        return self.recv(reset=True)

    def __repr__(self) -> str:
        """Prettify the debug information."""
        config = self.config
        config_str = ", ".join([f"{k}={pprint.pformat(v)}" for k, v in config.items()])
        return f"{self.__class__.__name__}({config_str})"

    def __str__(self) -> str:
        """Prettify the debug information."""
        return self.__repr__()

    def close(self) -> None:
        shutdown()