from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, x_obs: np.ndarray, t: float) -> float:
        """Callable interface required by sim.run_episode()."""
        return self.compute(x_obs, t)

    @abstractmethod
    def compute(self, x_obs: np.ndarray, t: float) -> float:
        """Compute and return control torque [Nm] given observed state and time."""

    def reset(self) -> None:
        """Reset internal state between episodes. Override if stateful."""

    def state_error(self, x_obs: np.ndarray) -> np.ndarray:
        """Return x_obs - x* with angles wrapped to [-pi, pi]."""
        e = x_obs - self.upright
        e[0] = self._wrap(e[0])
        e[1] = self._wrap(e[1])
        return e

    @staticmethod
    def _wrap(a: float | np.ndarray) -> float | np.ndarray:
        """Wrap angle(s) to [-pi, pi]."""
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    @property
    def upright(self) -> np.ndarray:
        """Upright equilibrium x* = [0, pi/2, 0, 0]."""
        return np.array([0.0, np.pi / 2, 0.0, 0.0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"