import numpy as np
from controllers.base_controller import BaseController


class LQRController(BaseController):
    def __init__(self, config: dict):
        super().__init__(config)
        self.torque_limit = config["simulation"].get("torque_limit", np.inf)

        # Placeholder gain vector
        # state = [q1, q2, dq1, dq2]
        self.K = np.array([0.15, 0.8, 0.05, 0.15], dtype=float)

        # desired upright equilibrium
        self.x_ref = np.array([np.pi / 2, 0.0, 0.0, 0.0], dtype=float)

    def _compute(self, state: np.ndarray, t: float) -> float:
        x = np.array(state, dtype=float)
        error = x - self.x_ref
        torque = -float(self.K @ error)
        return self._clip(torque, self.torque_limit)