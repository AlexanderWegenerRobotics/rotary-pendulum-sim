from abc import ABC, abstractmethod
import time
import numpy as np


class BaseController(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.last_solve_time: float = 0.0

    @abstractmethod
    def _compute(self, state: np.ndarray, t: float) -> float:
        """Return scalar torque given state [q1, q2, dq1, dq2] and time t."""

    def compute(self, state: np.ndarray, t: float) -> float:
        t0 = time.perf_counter()
        torque = self._compute(state, t)
        self.last_solve_time = time.perf_counter() - t0
        return torque

    def reset(self):
        pass

    def _clip(self, torque: float, limit: float) -> float:
        return float(np.clip(torque, -limit, limit))