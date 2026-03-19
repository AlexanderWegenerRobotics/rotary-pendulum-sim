import numpy as np
from controllers.base_controller import BaseController


class SpringController(BaseController):
    """Simple PD spring around the upright equilibrium. Demo only."""

    def __init__(self, config: dict, kp: float = 50.0, kd: float = 5.0):
        super().__init__(config)
        self.kp = kp
        self.kd = kd

    def compute(self, state: np.ndarray, t: float) -> float:
        q1, q2, dq1, dq2 = state
        torque  = self.kp * (0.0 - q1) - self.kd * dq1
        torque += self.kp * (np.pi - q2) - self.kd * dq2
        return self._clip(torque)
