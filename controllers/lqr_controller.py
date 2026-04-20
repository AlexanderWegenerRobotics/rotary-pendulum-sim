import numpy as np
from scipy.linalg import expm, solve_discrete_are

from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class LQRController(BaseController):
    ARMATURE = 0.002  # from MuJoCo XML / Alex's MPC dynamics

    # swing-up tuning
    SWING_UP_GAIN = 12.0

    # full LQR catch region
    LQR_CATCH_ANGLE_Q1 = 0.55
    LQR_CATCH_ANGLE_Q2 = 0.85
    LQR_CATCH_VEL = 6.0

    # start blending before full catch
    BLEND_ANGLE_Q1 = 1.00
    BLEND_ANGLE_Q2 = 1.20
    BLEND_VEL = 10.0

    # less restrictive torque ramping
    MAX_TORQUE_RATE = 400.0  # Nm/sec

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.physics_hz = config["simulation"]["physics_hz"]
        self.dt = 1.0 / self.physics_hz

        self.params = get_system_params(config)

        # upright target
        self.x_ref = np.array(
            config["test"].get("target_state", [np.pi / 2.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.u_ref = 0.0

        # LQR tuning
        self.Q = np.diag([80.0, 120.0, 8.0, 10.0])
        self.R = np.array([[1.0]])

        # startup guard
        self._startup_done = False
        self._startup_calls = 0
        self._printed_debug = False

        # store previous torque for ramping
        self._last_u = 0.0

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

        print("LQR / Swing-up controller initialized")
        print("x_ref =", self.x_ref)
        print("u_ref =", self.u_ref)
        print("A_c =\n", A_c)
        print("B_c =\n", B_c)
        print("Ad =\n", self.Ad)
        print("Bd =\n", self.Bd)
        print("K =\n", self.K)
        print("Closed-loop eigvals =", eigvals)
        print("Spectral radius =", np.max(np.abs(eigvals)))

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

    def _swingup_torque(self, state: np.ndarray) -> float:
        E_err = self._total_energy(state) - self._E_upright

        q1, q2, dq1, dq2 = state
        q1_err = self._wrap_angle(q1 - self.x_ref[0])

        # softer energy pumping
        pump_signal = dq1 * np.cos(q1)
        if abs(pump_signal) > 1e-6:
            tau_energy = -self.SWING_UP_GAIN * np.tanh(0.25 * E_err) * np.sign(pump_signal)
        else:
            tau_energy = 0.0

        # shape both links near the top
        if abs(q1_err) < 1.2:
            tau_position = (
                -25.0 * q1_err
                - 4.0 * dq1
                - 8.0 * q2
                - 2.0 * dq2
            )
        else:
            tau_position = 0.0

        alpha = max(0.0, 1.0 - abs(q1_err) / 1.2)
        tau = (1.0 - alpha) * tau_energy + alpha * tau_position

        return self._clip(tau, self.torque_limit)

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

    def _blend_weight(self, state: np.ndarray) -> float:
        err = state - self.x_ref
        e1 = abs(self._wrap_angle(err[0]))
        e2 = abs(self._wrap_angle(err[1]))
        v = np.sqrt(err[2] ** 2 + err[3] ** 2)

        if e1 > self.BLEND_ANGLE_Q1 or e2 > self.BLEND_ANGLE_Q2 or v > self.BLEND_VEL:
            return 0.0

        if self._near_upright(state):
            return 1.0

        w1 = 1.0 - (e1 - self.LQR_CATCH_ANGLE_Q1) / (
            self.BLEND_ANGLE_Q1 - self.LQR_CATCH_ANGLE_Q1 + 1e-9
        )
        w2 = 1.0 - (e2 - self.LQR_CATCH_ANGLE_Q2) / (
            self.BLEND_ANGLE_Q2 - self.LQR_CATCH_ANGLE_Q2 + 1e-9
        )
        wv = 1.0 - (v - self.LQR_CATCH_VEL) / (
            self.BLEND_VEL - self.LQR_CATCH_VEL + 1e-9
        )

        return float(np.clip(min(w1, w2, wv), 0.0, 1.0))

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

    def _compute(self, state: np.ndarray, t: float) -> float:
        if not self._startup_done:
            self._startup_calls += 1
            if t == 0.0 and self._startup_calls < 100:
                return 0.0
            self._startup_done = True

        x = np.array(state, dtype=float)

        err = x - self.x_ref
        err[0] = self._wrap_angle(err[0])
        err[1] = self._wrap_angle(err[1])

        u_lqr = -float((self.K @ err.reshape(-1, 1)).item())
        u_swing = self._swingup_torque(x)

        w = self._blend_weight(x)
        u_raw = (1.0 - w) * u_swing + w * u_lqr

        # torque ramping
        du_max = self.MAX_TORQUE_RATE * self.dt
        du = np.clip(u_raw - self._last_u, -du_max, du_max)
        u = self._last_u + du

        u = self._clip(u, self.torque_limit)
        self._last_u = u

        if (not self._printed_debug) and (t > 0.05):
            mode = "LQR" if w >= 0.999 else ("BLEND" if w > 0.0 else "SWING-UP")
            E = self._total_energy(x)
            print(f"\n----- LQR DEBUG ({mode}) -----")
            print(f"t      = {t:.3f}")
            print("state  =", x)
            print("x_ref  =", self.x_ref)
            print(f"E_current = {E:.4f}, E_upright = {self._E_upright:.4f}")
            print(f"blend  = {w:.3f}")
            print(f"u_raw  = {u_raw:.6f}")
            print(f"torque = {u:.6f}")
            print("---------------------\n")
            self._printed_debug = True

        return u