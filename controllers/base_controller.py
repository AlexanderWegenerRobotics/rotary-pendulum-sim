from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.torque_limit = config["simulation"]["torque_limit"]

    @abstractmethod
    def compute(self, state: np.ndarray, t: float) -> float:
        """Return scalar torque given state [q1, q2, dq1, dq2] and time t."""

    def reset(self):
        pass

    def _clip(self, torque: float) -> float:
        return float(np.clip(torque, -self.torque_limit, self.torque_limit))
