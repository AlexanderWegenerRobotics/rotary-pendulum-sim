import numpy as np
import yaml
from scipy.linalg import expm, solve_discrete_are

from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class LQRController(BaseController):
    ARMATURE = 0.002

    LQR_Q_DIAG = [30.0, 300.0, 5.0, 40.0]
    LQR_R = 1.0

    ENTER_ERR_Q1 = 0.40
    ENTER_ERR_Q2 = 0.45
    ENTER_VEL_NORM = 3.00
    ENTER_ENERGY_ERR = 0.60

    DEPART_ANGLE = 0.08
    DEPART_VEL = 0.25
    DEPART_TIME = 0.22
    DEPART_TORQUE_FRAC = 0.38

    ENERGY_GAIN = 0.60
    ENERGY_SCALE = 1.20
    ENERGY_Q2_BIAS = 0.12
    ENERGY_Q2_DAMP = 0.08

    APPROACH_Q1 = 0.85
    APPROACH_Q2 = 1.20
    APPROACH_ENERGY = 0.90
    APPROACH_K_Q1 = 7.5
    APPROACH_K_Q2 = 17.0
    APPROACH_K_DQ1 = 2.8
    APPROACH_K_DQ2 = 4.0
    APPROACH_BRAKE = 0.55
    APPROACH_BLEND_POWER = 1.6

    def __init__(self, config: dict):
        """Initialize model data, LQR gain, switching thresholds, and energy targets."""
        super().__init__(config)
        self.params = get_system_params(config)
        self.dt = 1.0 / config["simulation"].get("control_hz", config["simulation"]["physics_hz"])
        # The sim dispatches _compute at ~400 Hz regardless of physics_hz, so pinning dt.
        self.dt = 0.0025
        self.x_ref = np.array(config["test"].get("target_state", [np.pi / 2, 0.0, 0.0, 0.0]), dtype=float)
        self.x_down = np.array(config["test"].get("initial_state", [-np.pi / 2, 0.0, 0.0, 0.0]), dtype=float)
        self.torque_limit = self._read_torque_limit(config)

        self.Q = np.diag(self.LQR_Q_DIAG)
        self.R = np.array([[self.LQR_R]], dtype=float)

        A_c, B_c = self._linearize_continuous(self.x_ref, 0.0)
        self.Ad, self.Bd = self._discretize(A_c, B_c, self.dt)
        self.K = self._solve_dlqr(self.Ad, self.Bd, self.Q, self.R)

        self.E_upright = self._total_energy(self.x_ref)
        self._depart_done = False
        self._in_lqr = False
        self._mode = "depart"

    def _read_torque_limit(self, config: dict) -> float:
        """Read the active torque limit from the scenario and fall back to simulation defaults."""
        limit = config["simulation"].get("torque_limit", np.inf)
        test_cfg = config.get("test", {})
        path = test_cfg.get("test_cases_path")
        name = test_cfg.get("scenario")

        if path and name:
            try:
                with open(path) as f:
                    scenario = yaml.safe_load(f)["scenarios"][name]
                scenario_limit = scenario.get("torque_limit")
                if scenario_limit is not None:
                    limit = float(scenario_limit)
            except Exception:
                pass

        return float(limit)

    def _wrap_angle(self, angle: float) -> float:
        """Wrap one angle to the interval [-pi, pi)."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _angle_error(self, q: float, q_ref: float) -> float:
        """Compute the wrapped angular error relative to a reference angle."""
        return self._wrap_angle(q - q_ref)

    def _state_error(self, state: np.ndarray) -> np.ndarray:
        """Build the wrapped state error around the upright equilibrium."""
        err = np.array(state, dtype=float) - self.x_ref
        err[0] = self._angle_error(state[0], self.x_ref[0])
        err[1] = self._angle_error(state[1], self.x_ref[1])
        return err

    def _continuous_dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """Evaluate the continuous dynamics using the same model structure as the MPC controller."""
        q1, q2, dq1, dq2 = x
        p: SystemParams = self.params

        c2 = np.cos(q2)
        s2 = np.sin(q2)

        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2) + self.ARMATURE
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
        m22 = p.I2 + p.m2 * p.lc2**2 + self.ARMATURE
        M = np.array([[m11, m12], [m12, m22]], dtype=float)

        h = p.m2 * p.l1 * p.lc2 * s2
        C_vec = np.array([
            -h * (2.0 * dq1 * dq2 + dq2**2),
            h * dq1**2,
        ], dtype=float)

        g_vec = np.array([
            (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.cos(q1) + p.m2 * p.lc2 * p.g * np.cos(q1 + q2),
            p.m2 * p.lc2 * p.g * np.cos(q1 + q2),
        ], dtype=float)

        D_vec = np.array([p.b1 * dq1, p.b2 * dq2], dtype=float)
        rhs = np.array([u, 0.0], dtype=float) - C_vec - g_vec - D_vec
        ddq = np.linalg.solve(M, rhs)
        return np.array([dq1, dq2, ddq[0], ddq[1]], dtype=float)

    def _linearize_continuous(self, x_eq: np.ndarray, u_eq: float):
        """Compute numerical Jacobians of the continuous dynamics at the operating point."""
        n = 4
        eps_x = 1e-5
        eps_u = 1e-4
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
        """Apply exact zero-order-hold discretization to the linear model."""
        nx = A_c.shape[0]
        nu = B_c.shape[1]
        M = np.block([[A_c, B_c], [np.zeros((nu, nx + nu))]])
        Md = expm(M * dt)
        return Md[:nx, :nx], Md[:nx, nx:]

    def _solve_dlqr(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve the discrete algebraic Riccati equation and return the optimal state-feedback gain."""
        P = solve_discrete_are(A, B, Q, R)
        return np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)

    def _total_energy(self, x: np.ndarray) -> float:
        """Compute total mechanical energy using the simulator coordinate convention."""
        q1, q2, dq1, dq2 = x
        p = self.params
        c2 = np.cos(q2)

        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2) + self.ARMATURE
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
        m22 = p.I2 + p.m2 * p.lc2**2 + self.ARMATURE

        kinetic = 0.5 * (m11 * dq1**2 + 2.0 * m12 * dq1 * dq2 + m22 * dq2**2)
        potential = (
            (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.sin(q1)
            + p.m2 * p.lc2 * p.g * np.sin(q1 + q2)
        )
        return float(kinetic + potential)

    def _departure_torque(self, state: np.ndarray, t: float) -> float:
        """Apply a short bounded pulse to break the exact hanging symmetry at startup."""
        q1_err_down = self._angle_error(state[0], self.x_down[0])
        moving = abs(state[2]) > self.DEPART_VEL or abs(q1_err_down) > self.DEPART_ANGLE
        if moving or t > self.DEPART_TIME:
            self._depart_done = True
            return 0.0

        pulse = 0.90 + 0.10 * np.cos(10.0 * t)
        return self._clip(self.DEPART_TORQUE_FRAC * self.torque_limit * pulse, self.torque_limit)

    def _energy_pump_torque(self, state: np.ndarray) -> float:
        """Passivity-based energy shaping. Sigma aligns u*dq1 with desired energy rate."""
        q1, q2, dq1, dq2 = state
        err = self._state_error(state)
        energy_err = self._total_energy(state) - self.E_upright
        sigma = (dq1 + 0.30 * dq2) * np.cos(q1) + 0.15 * dq2 * np.cos(q1 + q2)
        if abs(sigma) < 1e-5:
            sigma = np.sign(err[0]) if abs(err[0]) > 1e-5 else 1.0

        tau_energy = -self.ENERGY_GAIN * self.torque_limit * np.tanh(energy_err / self.ENERGY_SCALE) * np.sign(sigma)
        tau_bias = -self.ENERGY_Q2_BIAS * err[1] - self.ENERGY_Q2_DAMP * err[3]
        return self._clip(tau_energy + tau_bias, self.torque_limit)

    def _approach_weight(self, state: np.ndarray) -> float:
        """Return a smooth weight describing how close the state is to the terminal approach region."""
        err = self._state_error(state)
        w_q1 = np.clip(1.0 - abs(err[0]) / self.APPROACH_Q1, 0.0, 1.0)
        w_q2 = np.clip(1.0 - abs(err[1]) / self.APPROACH_Q2, 0.0, 1.0)
        w_e = np.clip(1.0 - abs(self._total_energy(state) - self.E_upright) / self.APPROACH_ENERGY, 0.0, 1.0)
        return float((w_q1 * w_q2 * w_e) ** self.APPROACH_BLEND_POWER)

    def _approach_torque(self, state: np.ndarray) -> float:
        """PD toward upright plus a passivity-consistent energy brake."""
        q1, q2, dq1, dq2 = state
        err = self._state_error(state)
        energy_err = self._total_energy(state) - self.E_upright
        u_pd = (
            -self.APPROACH_K_Q1 * err[0]
            -self.APPROACH_K_Q2 * err[1]
            -self.APPROACH_K_DQ1 * err[2]
            -self.APPROACH_K_DQ2 * err[3]
        )
        sigma = dq1 * np.cos(q1)
        sign_sigma = np.sign(sigma) if abs(sigma) > 1e-5 else 1.0
        u_brake = -self.APPROACH_BRAKE * self.torque_limit * np.tanh(energy_err / 0.25) * sign_sigma
        return self._clip(u_pd + u_brake, self.torque_limit)

    def _should_enter_lqr(self, state: np.ndarray) -> bool:
        """Check whether the state is inside the local linear capture region."""
        err = self._state_error(state)
        vel_norm = np.linalg.norm(err[2:])
        energy_err = abs(self._total_energy(state) - self.E_upright)
        return (
            abs(err[0]) < self.ENTER_ERR_Q1
            and abs(err[1]) < self.ENTER_ERR_Q2
            and vel_norm < self.ENTER_VEL_NORM
            and energy_err < self.ENTER_ENERGY_ERR
        )

    def _lqr_torque(self, state: np.ndarray) -> float:
        """Apply the discrete-time LQR feedback law around the upright equilibrium."""
        err = self._state_error(state)
        u = -float((self.K @ err.reshape(-1, 1)).item())
        return self._clip(u, self.torque_limit)

    def _outer_loop_torque(self, state: np.ndarray, t: float) -> float:
        """Run startup departure, then blend pure energy pumping with terminal approach shaping."""
        if not self._depart_done:
            tau_depart = self._departure_torque(state, t)
            if not self._depart_done:
                self._mode = "depart"
                return tau_depart

        w = self._approach_weight(state)
        tau_energy = self._energy_pump_torque(state)
        tau_approach = self._approach_torque(state)
        self._mode = "approach" if w > 0.5 else "energy"
        tau = (1.0 - w) * tau_energy + w * tau_approach
        return self._clip(tau, self.torque_limit)

    def _compute(self, state: np.ndarray, t: float) -> float:
        """One-way latch: swing-up until capture, then LQR forever."""
        x = np.array(state, dtype=float)

        if self._in_lqr:
            self._mode = "lqr"
            return self._lqr_torque(x)

        if self._should_enter_lqr(x):
            self._in_lqr = True
            self._mode = "lqr"
            return self._lqr_torque(x)

        return self._outer_loop_torque(x, t)