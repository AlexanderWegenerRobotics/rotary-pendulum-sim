import numpy as np
from scipy.linalg import expm, solve_discrete_are

from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class LQRController(BaseController):
    ARMATURE = 0.002

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.physics_hz = config["simulation"]["physics_hz"]
        self.dt = 1.0 / self.physics_hz

        self.params = get_system_params(config)

        # Final goal: upright
        self.x_ref = np.array(
            config["test"].get("target_state", [np.pi / 2.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.u_ref = 0.0

        # Hanging-down reference for swing-up
        self.q_down = -np.pi / 2.0

        # LQR near upright
        self.Q = np.diag([80.0, 120.0, 8.0, 10.0])
        self.R = np.array([[1.0]])

        # Swing-up gains
        self.k_energy = 1.2
        self.k_q2 = 0.25
        self.k_dq2 = 0.08
        self.k_kick = 0.8

        # Catch region
        self.catch_angle_q1 = 0.45
        self.catch_angle_q2 = 0.40
        self.catch_vel_q1 = 4.0
        self.catch_vel_q2 = 5.0

        # Release region
        self.release_angle_q1 = 0.65
        self.release_angle_q2 = 0.60
        self.release_vel_q1 = 5.0
        self.release_vel_q2 = 6.0

        self.mode = "swingup"
        self._printed_debug = True

        # Linearize around upright for LQR
        A_c, B_c = self._linearize_continuous(self.x_ref, self.u_ref)
        self.Ad, self.Bd = self._discretize(A_c, B_c, self.dt)
        self.K = self._solve_dlqr(self.Ad, self.Bd, self.Q, self.R)

        Acl = self.Ad - self.Bd @ self.K
        eigvals = np.linalg.eigvals(Acl)

    def _continuous_dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        q1, q2, dq1, dq2 = x
        p: SystemParams = self.params

        c2 = np.cos(q2)
        s2 = np.sin(q2)
        a = self.ARMATURE

        m11 = (
            p.I1
            + p.I2
            + p.m1 * p.lc1**2
            + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2)
            + a
        )
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
        m22 = p.I2 + p.m2 * p.lc2**2 + a
        M = np.array([[m11, m12], [m12, m22]], dtype=float)

        h = p.m2 * p.l1 * p.lc2 * s2
        C_vec = np.array(
            [
                -h * (2.0 * dq1 * dq2 + dq2**2),
                h * dq1**2,
            ],
            dtype=float,
        )

        g_vec = np.array(
            [
                (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.cos(q1)
                + p.m2 * p.lc2 * p.g * np.cos(q1 + q2),
                p.m2 * p.lc2 * p.g * np.cos(q1 + q2),
            ],
            dtype=float,
        )

        D_vec = np.array([p.b1 * dq1, p.b2 * dq2], dtype=float)

        rhs = np.array([u, 0.0], dtype=float) - C_vec - g_vec - D_vec
        ddq = np.linalg.solve(M, rhs)

        return np.array([dq1, dq2, ddq[0], ddq[1]], dtype=float)

    def _linearize_continuous(self, x_eq: np.ndarray, u_eq: float):
        n = 4
        eps_x = 1e-6
        eps_u = 1e-6

        A = np.zeros((n, n))
        B = np.zeros((n, 1))

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps_x
            f_plus = self._continuous_dynamics(x_eq + dx, u_eq)
            f_minus = self._continuous_dynamics(x_eq - dx, u_eq)
            A[:, i] = (f_plus - f_minus) / (2.0 * eps_x)

        f_plus = self._continuous_dynamics(x_eq, u_eq + eps_u)
        f_minus = self._continuous_dynamics(x_eq, u_eq - eps_u)
        B[:, 0] = (f_plus - f_minus) / (2.0 * eps_u)

        return A, B

    def _discretize(self, A_c: np.ndarray, B_c: np.ndarray, dt: float):
        nx = A_c.shape[0]
        nu = B_c.shape[1]

        top = np.hstack([A_c, B_c])
        bot = np.zeros((nu, nx + nu))
        M = np.vstack([top, bot])

        Md = expm(M * dt)
        Ad = Md[:nx, :nx]
        Bd = Md[:nx, nx:]
        return Ad, Bd

    def _solve_dlqr(self, A, B, Q, R):
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K

    def _wrap_angle(self, angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _energy(self, x: np.ndarray) -> float:
        """
        Energy relative to the hanging-down equilibrium q1 = -pi/2.
        """
        q1, _, dq1, _ = x
        p = self.params

        J_eff = p.I1 + p.m1 * p.lc1**2 + self.ARMATURE
        mgl_eff = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g

        V = mgl_eff * (np.sin(q1) + 1.0)
        T = 0.5 * J_eff * dq1**2
        return T + V

    def _desired_upright_energy(self) -> float:
        p = self.params
        mgl_eff = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g
        return 2.0 * mgl_eff

    def _swing_up_control(self, x: np.ndarray) -> float:
        q1, q2, dq1, dq2 = x

        E = self._energy(x)
        E_des = self._desired_upright_energy()
        E_err = E_des - E

        # Smooth pumping
        phase = dq1 * np.sin(q1)
        pump = (
            self.k_energy
            * self.torque_limit
            * np.tanh(0.30 * E_err)
            * np.tanh(10.0 * phase)
        )

        # Small kick-start when stuck near hanging down
        q_down_err = self._wrap_angle(q1 - self.q_down)
        if abs(dq1) < 0.08 and abs(q_down_err) < 0.20:
            pump += self.k_kick * np.sign(q_down_err if abs(q_down_err) > 1e-4 else 1.0)

        shape = -self.k_q2 * self._wrap_angle(q2) - self.k_dq2 * dq2

        u = pump + shape
        return self._clip(u, self.torque_limit)

    def _lqr_control(self, x: np.ndarray) -> float:
        err = x - self.x_ref
        err[0] = self._wrap_angle(err[0])
        err[1] = self._wrap_angle(err[1])

        u = -float((self.K @ err.reshape(-1, 1)).item())
        return self._clip(u, self.torque_limit)

    def _should_switch_to_lqr(self, x: np.ndarray) -> bool:
        q1, q2, dq1, dq2 = x
        e1 = self._wrap_angle(q1 - self.x_ref[0])
        e2 = self._wrap_angle(q2 - self.x_ref[1])

        return (
            abs(e1) < self.catch_angle_q1
            and abs(e2) < self.catch_angle_q2
            and abs(dq1) < self.catch_vel_q1
            and abs(dq2) < self.catch_vel_q2
        )

    def _should_fall_back_to_swingup(self, x: np.ndarray) -> bool:
        q1, q2, dq1, dq2 = x
        e1 = self._wrap_angle(q1 - self.x_ref[0])
        e2 = self._wrap_angle(q2 - self.x_ref[1])

        return (
            abs(e1) > self.release_angle_q1
            or abs(e2) > self.release_angle_q2
            or abs(dq1) > self.release_vel_q1
            or abs(dq2) > self.release_vel_q2
        )

    def _compute(self, state: np.ndarray, t: float) -> float:
        x = np.array(state, dtype=float)

        if self.mode == "swingup" and self._should_switch_to_lqr(x):
            self.mode = "lqr"
        elif self.mode == "lqr" and self._should_fall_back_to_swingup(x):
            self.mode = "swingup"

        if self.mode == "lqr":
            u = self._lqr_control(x)
        else:
            u = self._swing_up_control(x)

        if (not self._printed_debug) and (t > 0.05):
            print("\n----- HYBRID DEBUG -----")
            print(f"t      = {t:.3f}")
            print("mode   =", self.mode)
            print("state  =", x)
            print("x_ref  =", self.x_ref)
            print(f"torque = {u:.6f}")
            print("------------------------\n")
            self._printed_debug = True

        return u