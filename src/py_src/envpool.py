from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Protocol
)

import gymnasium as gym

import numpy as np

class EnvPool(Protocol):
    """Cpp PyEnvpool class interface."""

    def __init__(self, spec):
        """Constructor of EnvPool."""

    def __len__(self) -> int:
        """Return the number of environments."""

    def _check_action(self, actions: List) -> None:
        """Check action shapes."""

    def _recv(self) -> List[np.ndarray]:
        """Cpp private _recv method."""

    def _send(self, action: List[np.ndarray]) -> None:
        """Cpp private _send method."""

    def _reset(self, env_id: np.ndarray) -> None:
        """Cpp private _reset method."""

    def _to(
      self: Any, state_values: List[np.ndarray], reset: bool
    ) -> Union[
      Any,
      Tuple[Any, Any],
      Tuple[Any, np.ndarray, np.ndarray, Any],
      Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Any],
    ]:
        """Conversion for output obs, reward, done, trunc, info."""

    def _from(
        self,
        action: Union[Dict[str, Any], np.ndarray],
        env_id: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Convertion for input action."""

    @property
    def observation_space(self) -> Union[gym.Space, Dict[str, Any]]:
        """Gym observation space."""

    @property
    def action_space(self) -> Union[gym.Space, Dict[str, Any]]:
        """Gym action space."""

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Set the seed for all environments."""

    def send(
        self,
        action: Union[Dict[str, Any], np.ndarray],
        env_id: Optional[np.ndarray] = None,
    ) -> None:
        """Envpool send wrapper."""

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> Tuple:
        """Envpool recv wrapper."""

    def step(
        self,
        action: Union[Dict[str, Any], np.ndarray],
        env_id: Optional[np.ndarray] = None,
    ) -> Tuple:
        """Envpool step interface that performs send/recv."""

    def reset(
        self,
        env_id: Optional[np.ndarray] = None,
    ) -> Tuple:
        """Envpool reset interface."""

    def close(
            self,
    ) -> None:
        """Shutdown the env server (must be called only once)"""
        pass