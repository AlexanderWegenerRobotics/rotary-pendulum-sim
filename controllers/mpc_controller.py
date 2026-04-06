from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from controllers.base_controller import BaseController
from src.system_params import SystemParams, get_system_params


class Dynamics:
    def __init__(self, p: SystemParams, armature: float = 0.002):
        self.params = p
        self._c_M11  = p.I1 + p.m1*p.lc1**2 + p.I2 + p.m2*(p.l1**2 + p.lc2**2)
        self._c_M12  = p.I2 + p.m2*p.lc2**2
        self._c_Mcos = p.m2*p.l1*p.lc2
        self._armature = armature

        self._c_h   = p.m2*p.l1*p.lc2
        self._c_g1a = (p.m1*p.lc1 + p.m2*p.l1) * p.g
        self._c_g2  = p.m2*p.lc2*p.g

    def forward(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> np.ndarray:
        # M entries — shape (K,)
        m_cos = self._c_Mcos * np.cos(q[:, 1])
        M11   = self._c_M11 + 2.0*m_cos + self._armature
        M12   = self._c_M12 + m_cos
        M22   = self._c_M12 + self._armature
        det   = M11*M22 - M12**2

        # C @ dq — shape (K,)
        h     = -self._c_h * np.sin(q[:, 1])
        Cdq0  = h * dq[:, 1] * (2.0*dq[:, 0] + dq[:, 1])
        Cdq1  = -h * dq[:, 0]**2

        # g — shape (K,)
        g1    = self._c_g1a*np.cos(q[:, 0]) + self._c_g2*np.cos(q[:, 0] + q[:, 1])
        g2    = self._c_g2*np.cos(q[:, 0] + q[:, 1])

        damp0 = 0.005 * dq[:, 0]
        damp1 = 0.005 * dq[:, 1]

        # rhs = Bu - Cdq - g
        rhs0  = u   - Cdq0 - g1 - damp0
        rhs1  = 0.0 - Cdq1 - g2 - damp1

        qdd0  = ( M22*rhs0 - M12*rhs1) / det
        qdd1  = (-M12*rhs0 + M11*rhs1) / det
        return np.stack([qdd0, qdd1], axis=1)
    
    def goal_energy(self, goal:np.ndarray) -> float:
        q1, q2 = goal[0], goal[1]
        p = self.params
        V = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.sin(q1) + p.m2 * p.lc2 * p.g * np.sin(q1 + q2)
        return float(V)

    def total_energy(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        p = self.params

        M11 = p.I1 + p.m1 * p.lc1**2 + p.I2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * np.cos(q[:, 1]))
        M12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * np.cos(q[:, 1]))
        M22 = p.I2 + p.m2 * p.lc2**2

        T = 0.5 * M11 * dq[:, 0]**2 + M12 * dq[:, 0] * dq[:, 1] + 0.5 * M22 * dq[:, 1]**2
        V = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.sin(q[:, 0]) + p.m2 * p.lc2 * p.g * np.sin(q[:, 0] + q[:, 1])

        return T + V


class MPCController(BaseController):

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.params       = get_system_params(config=config)
        self.dynamics     = Dynamics(self.params)

        self.K     = 800
        self.N     = 100
        self.dt    = 0.002
        self.lam = 200
        self.sigma = 0.9

        self._Q    = np.diag([10.0, 20.0, 2.0, 5.0])
        self._Qf   = 10.0 * self._Q
        self._r    = 0.001

        self._wE = 0.05
        self._alpha_E = 2.0
        self._use_energy_gate = True

        self._goal  = np.array([np.pi/2, 0.0, 0.0, 0.0])
        self._U_nom = np.zeros(self.N)
        self._E_goal = self.dynamics.goal_energy(self._goal)

        self.compute_times = []
        self.compute_cnt = 0

    def _rk4_step(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = self.dt
        fwd = self.dynamics.forward

        k1_dq  = dq;                    k1_ddq = fwd(q, dq, u)
        k2_dq  = dq + 0.5*dt*k1_ddq;    k2_ddq = fwd(q + 0.5*dt*k1_dq, k2_dq, u)
        k3_dq  = dq + 0.5*dt*k2_ddq;    k3_ddq = fwd(q + 0.5*dt*k2_dq, k3_dq, u)
        k4_dq  = dq + dt*k3_ddq;        k4_ddq = fwd(q + dt*k3_dq, k4_dq, u)

        q_next  = q  + (dt/6.0)*(k1_dq  + 2.0*k2_dq  + 2.0*k3_dq  + k4_dq)
        dq_next = dq + (dt/6.0)*(k1_ddq + 2.0*k2_ddq + 2.0*k3_ddq + k4_ddq)

        np.clip(dq_next, -50.0, 50.0, out=dq_next)
        return q_next, dq_next

    def _rollout(self, state: np.ndarray, V: np.ndarray) -> np.ndarray:
        q  = np.tile(state[:2], (self.K, 1))
        dq = np.tile(state[2:], (self.K, 1))

        costs = np.zeros(self.K)

        for t in range(self.N):
            u     = np.clip(V[:, t], -self.torque_limit, self.torque_limit)
            q, dq = self._rk4_step(q, dq, u)
            costs += self._running_cost(q, dq, u)

        costs += self._terminal_cost(q, dq)
        np.clip(costs, 0.0, 1e6, out=costs)
        np.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=0.0, copy=False)
        return costs

    def _state_error(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e_q1  = np.arctan2(np.sin(q[:, 0] - self._goal[0]), np.cos(q[:, 0] - self._goal[0]))
        e_q2  = np.arctan2(np.sin(q[:, 1] - self._goal[1]), np.cos(q[:, 1] - self._goal[1]))
        e_dq1 = dq[:, 0] - self._goal[2]
        e_dq2 = dq[:, 1] - self._goal[3]
        return np.stack([e_q1, e_q2, e_dq1, e_dq2], axis=1)

    def _running_cost(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> np.ndarray:
        e = self._state_error(q, dq)
        Qe = e @ self._Q
        tracking = (Qe * e).sum(axis=1)

        E = self.dynamics.total_energy(q, dq)
        eE = E - self._E_goal

        if self._use_energy_gate:
            gate = self._energy_gate(q, dq)
        else:
            gate = 1.0
        energy_cost = self._wE * gate * eE**2
        control_cost = self._r * u**2
        return tracking + energy_cost + control_cost

    def _terminal_cost(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e   = self._state_error(q, dq)
        Qfe = e @ self._Qf
        return (Qfe * e).sum(axis=1)
    
    def reset(self):
        self._U_nom[:] = 0.0

    def _energy_gate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e = self._state_error(q, dq)
        angle_err_sq = e[:, 0]**2 + e[:, 1]**2
        return 1.0 - np.exp(-self._alpha_E * angle_err_sq)

    def _compute(self, state: np.ndarray, t: float) -> float:
        start = time.time()
        eps = np.random.randn(self.K, self.N) * self.sigma
        V   = self._U_nom[None, :] + eps
        S   = self._rollout(state, V)
        beta    = S.min()
        weights = np.exp(-(S - beta) / self.lam)
        weights /= weights.sum()

        self._U_nom += (weights[:, None] * eps).sum(axis=0)
        self._U_nom  = np.clip(self._U_nom, -self.torque_limit, self.torque_limit)

        u                = float(self._U_nom[0])
        self._U_nom[:-1] = self._U_nom[1:]
        self._U_nom[-1]  = 0.0
        self.compute_times.append(time.time() - start)
        self.compute_cnt += 1
        if self.compute_cnt == 50:
            t = np.array(self.compute_times)
            print(f"Mean time: {np.mean(t)}, std: {np.std(t)}")
            self.compute_cnt = 0
            self.compute_times = []
        return u


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    mpc = MPCController(config)

    state = np.array([0.0, np.pi/4, 0.1, -0.2])
    V     = np.random.randn(mpc.K, mpc.N) * mpc.sigma

    # batch forward (K samples)
    q  = np.tile(state[:2], (mpc.K, 1))
    dq = np.tile(state[2:], (mpc.K, 1))
    u  = V[:, 0]
    qdd = mpc.dynamics.forward(q, dq, u)
    print(f"batch forward output shape  : {qdd.shape}")

    # single forward (K=1)
    q1  = state[:2][None, :]
    dq1 = state[2:][None, :]
    u1  = np.array([1.0])
    qdd1 = mpc.dynamics.forward(q1, dq1, u1)
    print(f"single forward output shape : {qdd1.shape}")

    q_next, dq_next = mpc._rk4_step(q, dq, u)
    print(f"_rk4_step q  shape          : {q_next.shape}")
    print(f"_rk4_step dq shape          : {dq_next.shape}")
    print(f"sample q_next[0]            : {q_next[0]}")

    print("End of file")