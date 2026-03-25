import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are

from controllers.base_controller import BaseController


class LQRController(BaseController):
    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.model_path = config["simulation"]["model_path"]
        self.physics_hz = config["simulation"]["physics_hz"]
        self.dt = 1.0 / self.physics_hz

        # Upright equilibrium in this simulation
        self.x_ref = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)
        self.u_ref = 0.0

        # Softer LQR weights to reduce aggressiveness
        self.Q = np.diag([80.0, 120.0, 4.0, 8.0])
        self.R = np.array([[3.0]])

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)

        # Linearize discrete dynamics around equilibrium
        self.Ad, self.Bd = self._linearize_discrete(self.x_ref, self.u_ref)

        # Solve discrete-time LQR
        self.K = self._solve_dlqr(self.Ad, self.Bd, self.Q, self.R)

        print("Ad =\n", self.Ad)
        print("Bd =\n", self.Bd)
        print("K =\n", self.K)

    def _set_state_and_input(self, data, x, u):
        data.qpos[:] = x[:2]
        data.qvel[:] = x[2:]
        data.ctrl[:] = [u]
        mujoco.mj_forward(self.model, data)

    def _step_dynamics(self, x, u):
        data = mujoco.MjData(self.model)
        self._set_state_and_input(data, x, u)
        mujoco.mj_step(self.model, data)
        x_next = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        return x_next

    def _linearize_discrete(self, x_eq, u_eq):
        n = 4
        eps_x = 1e-5
        eps_u = 1e-5

        Ad = np.zeros((n, n))
        Bd = np.zeros((n, 1))

        # Finite-difference w.r.t. state
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps_x
            f_plus = self._step_dynamics(x_eq + dx, u_eq)
            f_minus = self._step_dynamics(x_eq - dx, u_eq)
            Ad[:, i] = (f_plus - f_minus) / (2.0 * eps_x)

        # Finite-difference w.r.t. input
        f_plus = self._step_dynamics(x_eq, u_eq + eps_u)
        f_minus = self._step_dynamics(x_eq, u_eq - eps_u)
        Bd[:, 0] = (f_plus - f_minus) / (2.0 * eps_u)

        return Ad, Bd

    def _solve_dlqr(self, A, B, Q, R):
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _compute(self, state: np.ndarray, t: float) -> float:
        x = np.array(state, dtype=float)

        # Error around upright equilibrium
        err = x - self.x_ref
        err[0] = self._wrap_angle(err[0])
        err[1] = self._wrap_angle(err[1])

        raw_torque = -float((self.K @ err.reshape(-1, 1)).item())
        clipped_torque = self._clip(raw_torque, self.torque_limit)

        # Print torque for debugging during first second
        if t < 1.0:
            print(
                f"t={t:.3f}, "
                f"err={err}, "
                f"raw={raw_torque:.4f}, "
                f"clipped={clipped_torque:.4f}"
            )

        return clipped_torque