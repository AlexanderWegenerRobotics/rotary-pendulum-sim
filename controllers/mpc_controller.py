from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from controllers.base_controller import BaseController
from src.system_params import SystemParams, get_system_params


class Dynamics:

    def __init__(self, p: SystemParams, armature: float = 0.002):
        self.params = p
        self._armature = armature

        self._c_M11 = p.I1 + p.m1 * p.lc1**2 + p.I2 + p.m2 * (p.l1**2 + p.lc2**2)
        self._c_M12 = p.I2 + p.m2 * p.lc2**2
        self._c_Mcos = p.m2 * p.l1 * p.lc2

        self._c_h = p.m2 * p.l1 * p.lc2
        self._c_g1a = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g
        self._c_g2 = p.m2 * p.lc2 * p.g
        self._damping = 0.005

    def forward(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> np.ndarray:
        c2 = np.cos(q[:, 1])
        s2 = np.sin(q[:, 1])

        m_cos = self._c_Mcos * c2
        M11 = self._c_M11 + 2.0 * m_cos + self._armature
        M12 = self._c_M12 + m_cos
        M22 = self._c_M12 + self._armature
        det = M11 * M22 - M12**2

        h = -self._c_h * s2
        Cdq0 = h * dq[:, 1] * (2.0 * dq[:, 0] + dq[:, 1])
        Cdq1 = -h * dq[:, 0]**2

        s1 = np.sin(q[:, 0])
        c1 = np.cos(q[:, 0])
        s12 = np.sin(q[:, 0] + q[:, 1])
        c12 = np.cos(q[:, 0] + q[:, 1])
        g1 = self._c_g1a * c1 + self._c_g2 * c12
        g2 = self._c_g2 * c12

        rhs0 = u - Cdq0 - g1 - self._damping * dq[:, 0]
        rhs1 = 0.0 - Cdq1 - g2 - self._damping * dq[:, 1]

        qdd0 = (M22 * rhs0 - M12 * rhs1) / det
        qdd1 = (-M12 * rhs0 + M11 * rhs1) / det
        return np.stack([qdd0, qdd1], axis=1)

    def potential_energy(self, q1: float, q2: float) -> float:
        p = self.params
        V = self._c_g1a * np.sin(q1) + self._c_g2 * np.sin(q1 + q2)
        return float(V)

    def total_energy(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        p = self.params
        c2 = np.cos(q[:, 1])

        M11 = self._c_M11 + 2.0 * self._c_Mcos * c2
        M12 = self._c_M12 + self._c_Mcos * c2
        M22 = self._c_M12

        T = 0.5 * M11 * dq[:, 0]**2 + M12 * dq[:, 0] * dq[:, 1] + 0.5 * M22 * dq[:, 1]**2
        V = self._c_g1a * np.sin(q[:, 0]) + self._c_g2 * np.sin(q[:, 0] + q[:, 1])
        return T + V

    def total_energy_scalar(self, q: np.ndarray, dq: np.ndarray) -> float:
        return float(self.total_energy(q[None, :], dq[None, :])[0])


class MPCController(BaseController):

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.params = get_system_params(config=config)
        self.dynamics = Dynamics(self.params)

        self._goal = np.array([np.pi / 2, 0.0, 0.0, 0.0])
        self._E_goal = self.dynamics.potential_energy(self._goal[0], self._goal[1])

        self._init_swing_up_params()
        self._init_mppi_params()
        self._init_diagnostics()

    def _init_swing_up_params(self):
        self._k_energy = 1.0
        self._k_bias = 0.3

        self._catch_angle_tol = 0.5
        self._catch_vel_tol = 4.0
        self._release_angle_tol = 0.6
        self._release_vel_tol = 10.0
        self._stabilizing = False

    def _init_mppi_params(self):
        self.K = 512
        self.dt = 0.004
        self.lam = 5.0
        self.sigma = 0.8
        self.N_roll = 125
        self.block_size = 5

        self.N_ctrl = int(np.ceil(self.N_roll / self.block_size))
        self._U_nom = np.zeros(self.N_ctrl)
        self._block_step = 0
        self._u_current = 0.0

        self._Q = np.diag([30.0, 40.0, 5.0, 8.0])
        self._Qf = 5.0 * self._Q
        self._r = 0.01

    def _init_diagnostics(self):
        self.compute_times = []
        self.compute_cnt = 0

    def reset(self):
        self._U_nom[:] = 0.0
        self._block_step = 0
        self._u_current = 0.0
        self._stabilizing = False

    def _angle_error(self, q: np.ndarray) -> np.ndarray:
        e1 = np.arctan2(np.sin(q[0] - self._goal[0]), np.cos(q[0] - self._goal[0]))
        e2 = np.arctan2(np.sin(q[1] - self._goal[1]), np.cos(q[1] - self._goal[1]))
        return np.array([e1, e2])

    def _is_near_goal(self, state: np.ndarray) -> bool:
        e = self._angle_error(state[:2])
        angle_ok = np.abs(e[0]) < self._catch_angle_tol and np.abs(e[1]) < self._catch_angle_tol
        vel_ok = np.abs(state[2]) < self._catch_vel_tol and np.abs(state[3]) < self._catch_vel_tol
        return angle_ok and vel_ok

    def _has_left_goal(self, state: np.ndarray) -> bool:
        e = self._angle_error(state[:2])
        angle_far = np.abs(e[0]) > self._release_angle_tol or np.abs(e[1]) > self._release_angle_tol
        vel_far = np.abs(state[2]) > self._release_vel_tol or np.abs(state[3]) > self._release_vel_tol
        return angle_far or vel_far

    # ---- energy-based swing-up ----

    def _energy_swing_up(self, state: np.ndarray) -> float:
        q, dq = state[:2], state[2:]
        E = self.dynamics.total_energy_scalar(q, dq)
        E_err = E - self._E_goal
        E_err_clipped = np.clip(E_err, -2.0, 2.0)
        u = -self._k_energy * E_err_clipped * np.sign(dq[0])

        e1 = np.arctan2(np.sin(q[0] - self._goal[0]), np.cos(q[0] - self._goal[0]))
        u += self._k_bias * np.sign(e1) * (1.0 - np.exp(-2.0 * e1**2))

        angle_dist = abs(e1)
        if angle_dist < 0.8 and abs(dq[0]) > 3.0:
            u -= 0.8 * dq[0]

        return float(np.clip(u, -self.torque_limit, self.torque_limit))

    # ---- MPPI stabilization ----

    def _rk4_step(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> tuple:
        dt = self.dt
        fwd = self.dynamics.forward

        k1_dq = dq
        k1_ddq = fwd(q, dq, u)

        q2 = q + 0.5 * dt * k1_dq
        dq2 = dq + 0.5 * dt * k1_ddq
        k2_ddq = fwd(q2, dq2, u)

        q3 = q + 0.5 * dt * dq2
        dq3 = dq + 0.5 * dt * k2_ddq
        k3_ddq = fwd(q3, dq3, u)

        q4 = q + dt * dq3
        dq4 = dq + dt * k3_ddq
        k4_ddq = fwd(q4, dq4, u)

        q_next = q + (dt / 6.0) * (k1_dq + 2.0 * dq2 + 2.0 * dq3 + dq4)
        dq_next = dq + (dt / 6.0) * (k1_ddq + 2.0 * k2_ddq + 2.0 * k3_ddq + k4_ddq)

        np.clip(dq_next, -50.0, 50.0, out=dq_next)
        return q_next, dq_next

    def _state_error_batch(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e_q1 = np.arctan2(np.sin(q[:, 0] - self._goal[0]), np.cos(q[:, 0] - self._goal[0]))
        e_q2 = np.arctan2(np.sin(q[:, 1] - self._goal[1]), np.cos(q[:, 1] - self._goal[1]))
        e_dq1 = dq[:, 0] - self._goal[2]
        e_dq2 = dq[:, 1] - self._goal[3]
        return np.stack([e_q1, e_q2, e_dq1, e_dq2], axis=1)

    def _running_cost(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> np.ndarray:
        e = self._state_error_batch(q, dq)
        tracking = (e @ self._Q * e).sum(axis=1)
        control = self._r * u**2
        return tracking + control

    def _terminal_cost(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e = self._state_error_batch(q, dq)
        return (e @ self._Qf * e).sum(axis=1)

    def _rollout(self, state: np.ndarray, V: np.ndarray) -> np.ndarray:
        q = np.tile(state[:2], (self.K, 1))
        dq = np.tile(state[2:], (self.K, 1))
        costs = np.zeros(self.K)

        for t in range(self.N_roll):
            j = t // self.block_size
            u = np.clip(V[:, j], -self.torque_limit, self.torque_limit)
            q, dq = self._rk4_step(q, dq, u)
            costs += self._running_cost(q, dq, u)

        costs += self._terminal_cost(q, dq)
        np.nan_to_num(costs, nan=1e10, posinf=1e10, neginf=0.0, copy=False)
        costs /= self.N_roll
        return costs

    def _mppi_compute(self, state: np.ndarray) -> float:
        self._block_step += 1
        if self._block_step < self.block_size:
            return self._u_current

        self._block_step = 0

        eps = np.random.randn(self.K, self.N_ctrl) * self.sigma
        V = self._U_nom[None, :] + eps

        S = self._rollout(state, V)

        beta = S.min()
        log_w = -(S - beta) / self.lam
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum() + 1e-30

        self._U_nom += (weights[:, None] * eps).sum(axis=0)
        self._U_nom = np.clip(self._U_nom, -self.torque_limit, self.torque_limit)

        self._u_current = float(self._U_nom[0])
        self._U_nom[:-1] = self._U_nom[1:]
        self._U_nom[-1] = 0.0

        return self._u_current

    # ---- main entry point ----

    def _compute(self, state: np.ndarray, t: float) -> float:
        start = time.time()

        if self._stabilizing:
            if self._has_left_goal(state):
                self._stabilizing = False
                self._U_nom[:] = 0.0
                self._block_step = 0
                u = self._energy_swing_up(state)
            else:
                u = self._mppi_compute(state)
        else:
            if self._is_near_goal(state):
                self._stabilizing = True
                self._U_nom[:] = 0.0
                self._block_step = 0
                u = self._mppi_compute(state)
            else:
                u = self._energy_swing_up(state)

        elapsed = time.time() - start
        self.compute_times.append(elapsed)
        self.compute_cnt += 1
        if self.compute_cnt == 100:
            tt = np.array(self.compute_times)
            mode = "MPPI" if self._stabilizing else "SWING"
            print(f"[{mode}] Mean: {np.mean(tt):.4f}s, std: {np.std(tt):.4f}")
            self.compute_cnt = 0
            self.compute_times = []

        return u