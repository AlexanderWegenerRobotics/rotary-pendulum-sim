import numpy as np
from scipy.linalg import expm, solve_discrete_are

from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class LQRController(BaseController):
    ARMATURE = 0.002  # from MuJoCo XML / Alex's MPC dynamics

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.physics_hz = config["simulation"]["physics_hz"]
        self.dt = 1.0 / self.physics_hz

        # Use shared project system parameters
        self.params = get_system_params(config)

        # Correct upright equilibrium from project convention
        self.x_ref = np.array(
            config["test"].get("target_state", [-np.pi / 2.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.u_ref = 0.0

        # Best stable tuning
        self.Q = np.diag([80.0, 120.0, 8.0, 10.0])
        self.R = np.array([[1.0]])

        self._printed_debug = False

        # Continuous-time linearization around equilibrium
        A_c, B_c = self._linearize_continuous(self.x_ref, self.u_ref)

        # Exact ZOH discretization
        self.Ad, self.Bd = self._discretize(A_c, B_c, self.dt)

        # Discrete-time LQR
        self.K = self._solve_dlqr(self.Ad, self.Bd, self.Q, self.R)

        Acl = self.Ad - self.Bd @ self.K
        eigvals = np.linalg.eigvals(Acl)

        print("LQR initialized")
        print("x_ref =", self.x_ref)
        print("u_ref =", self.u_ref)
        print("A_c =\n", A_c)
        print("B_c =\n", B_c)
        print("Ad =\n", self.Ad)
        print("Bd =\n", self.Bd)
        print("K =\n", self.K)
        print("Closed-loop eigvals =", eigvals)
        print("Spectral radius =", np.max(np.abs(eigvals)))

    def _continuous_dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        q1, q2, dq1, dq2 = x
        p: SystemParams = self.params

        c2 = np.cos(q2)
        s2 = np.sin(q2)
        a = self.ARMATURE

        # Mass matrix
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

        # Coriolis / centripetal
        h = p.m2 * p.l1 * p.lc2 * s2
        C_vec = np.array(
            [
                -h * (2.0 * dq1 * dq2 + dq2**2),
                h * dq1**2,
            ],
            dtype=float,
        )

        # Gravity
        g_vec = np.array(
            [
                (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.cos(q1)
                + p.m2 * p.lc2 * p.g * np.cos(q1 + q2),
                p.m2 * p.lc2 * p.g * np.cos(q1 + q2),
            ],
            dtype=float,
        )

        # Damping
        D_vec = np.array([p.b1 * dq1, p.b2 * dq2], dtype=float)

        # Dynamics: M ddq = [u, 0] - C - g - D
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
        x = np.array(state, dtype=float)

        err = x - self.x_ref
        err[0] = self._wrap_angle(err[0])
        err[1] = self._wrap_angle(err[1])

        u = -float((self.K @ err.reshape(-1, 1)).item())
        u = self._clip(u, self.torque_limit)

        if (not self._printed_debug) and (t > 0.05):
            print("\n----- LQR DEBUG -----")
            print(f"t      = {t:.3f}")
            print("state  =", x)
            print("x_ref  =", self.x_ref)
            print("err    =", err)
            print(f"torque = {u:.6f}")
            print("---------------------\n")
            self._printed_debug = True

        return u