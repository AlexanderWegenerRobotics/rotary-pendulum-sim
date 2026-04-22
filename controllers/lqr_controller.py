import numpy as np
from scipy.linalg import expm, solve_discrete_are

from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class LQRController(BaseController):
    ARMATURE = 0.002  # from MuJoCo XML / Alex's MPC dynamics

    # swing-up tuning
    SWING_UP_GAIN = 60.0

    # catch earlier, but require slower entry
    LQR_CATCH_ANGLE_Q1 = 0.2
    LQR_CATCH_ANGLE_Q2 = 0.85
    LQR_CATCH_VEL = 2.5

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.physics_hz = config["simulation"]["physics_hz"]
        self.dt = 1.0 / self.physics_hz

        self.params = get_system_params(config)

        # upright target
        self.x_ref = np.array(config["test"]["target_state"])
        self.u_ref = 0.0

        # LQR tuning
        self.Q = np.diag([80.0, 120.0, 8.0, 10.0])
        self.R = np.array([[1.0]])

        # startup guard
        self._startup_done = False
        self._startup_calls = 0
        self._printed_debug = False

        # continuous-time linearization around upright
        A_c, B_c = self._linearize_continuous(self.x_ref, self.u_ref)

        # exact ZOH discretization
        self.Ad, self.Bd = self._discretize(A_c, B_c, self.dt)

        # discrete-time LQR
        self.K = self._solve_dlqr(self.Ad, self.Bd, self.Q, self.R)

        # energy at upright target
        self._E_upright = self._total_energy(self.x_ref)

        Acl = self.Ad - self.Bd @ self.K
        eigvals = np.linalg.eigvals(Acl)
        self.mode = "SWING-UP"

    # ------------------------------------------------------------------
    # energy computation for swing-up
    # ------------------------------------------------------------------

    def _total_energy(self, x: np.ndarray) -> float:
        q1, q2, dq1, dq2 = x
        p = self.params
        a = self.ARMATURE
        c2 = np.cos(q2)

        m11 = (
            p.I1
            + p.I2
            + p.m1 * p.lc1**2
            + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2)
            + a
        )
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
        m22 = p.I2 + p.m2 * p.lc2**2 + a

        T = 0.5 * (m11 * dq1**2 + 2.0 * m12 * dq1 * dq2 + m22 * dq2**2)

        U = (
            (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.sin(q1)
            + p.m2 * p.lc2 * p.g * np.sin(q1 + q2)
        )

        return T + U

    # ------------------------------------------------------------------
    # energy-based swing-up
    # ------------------------------------------------------------------

    def _swingup_torque(self, state: np.ndarray) -> float:
        E_err = self._total_energy(state) - self._E_upright

        q1, q2, dq1, dq2 = state
        q1_err = self._wrap_angle(q1 - self.x_ref[0])

        # if nearly stationary and far from upright, kick hard
        #if abs(dq1) < 0.5 and abs(q1_err) > 0.8:
        #    return self._clip(np.sign(-q1_err) * self.torque_limit, self.torque_limit)
        if self._startup_calls < 50:
            self._startup_calls += 1
            return self.torque_limit * 0.5  # gentle push to break symmetry

        # energy pumping term
        pump_signal = dq1 * np.sin(q1 - self.x_ref[0])
        if abs(pump_signal) > 1e-6:
            tau_energy = -self.SWING_UP_GAIN * E_err * np.sign(pump_signal)
        else:
            tau_energy = 0.0

        # shape both links near the top
        if abs(q1_err) < 1.2:
            tau_position = (-25.0 * q1_err - 5.0 * dq1 - 8.0 * q2 - 2.0 * dq2)
        else:
            tau_position = 0.0

        # energy dominates far away, shaping helps near the top
        alpha = max(0.0, 1.0 - abs(q1_err) / 1.2)
        tau = (1.0 - alpha) * tau_energy + alpha * tau_position

        return self._clip(tau, self.torque_limit)

    # ------------------------------------------------------------------
    # catch condition
    # ------------------------------------------------------------------

    def _near_upright(self, state: np.ndarray) -> bool:
        err = state - self.x_ref
        angle_err_q1 = abs(self._wrap_angle(err[0]))
        angle_err_q2 = abs(self._wrap_angle(err[1]))
        vel_norm = np.sqrt(err[2] ** 2 + err[3] ** 2)

        return (
            angle_err_q1 < self.LQR_CATCH_ANGLE_Q1
            and angle_err_q2 < self.LQR_CATCH_ANGLE_Q2
            and vel_norm < self.LQR_CATCH_VEL
        )

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # linearization & LQR
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # main control law
    # ------------------------------------------------------------------

    def _compute(self, state: np.ndarray, t: float) -> float:
        x = np.array(state, dtype=float)

        if self._near_upright(x):
            err = x - self.x_ref
            err[0] = self._wrap_angle(err[0])
            err[1] = self._wrap_angle(err[1])
            u = -float((self.K @ err.reshape(-1, 1)).item())
            mode = "LQR"
        else:
            u = self._swingup_torque(x)
            mode = "SWING-UP"

        if mode != self.mode:
            print(f"Mode change to: {mode}")
        self.mode = mode

        u = self._clip(u, self.torque_limit)


        return u