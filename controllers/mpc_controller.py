from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from controllers.base_controller import BaseController
from src.system_params import SystemParams, get_system_params


class Dynamics:

    def __init__(self, p: SystemParams):
        self._c_M11  = p.I1 + p.m1*p.lc1**2 + p.I2 + p.m2*(p.l1**2 + p.lc2**2)
        self._c_M12  = p.I2 + p.m2*p.lc2**2
        self._c_M22  = p.I2 + p.m2*p.lc2**2
        self._c_Mcos = p.m2*p.l1*p.lc2

        self._c_h    = p.m2*p.l1*p.lc2

        self._c_g1a  = (p.m1*p.lc1 + p.m2*p.l1) * p.g
        self._c_g2   = p.m2*p.lc2*p.g

    def M(self, q: np.ndarray) -> np.ndarray:
        c2 = np.cos(q[1])
        m_cos = self._c_Mcos * c2
        M11 = self._c_M11 + 2.0*m_cos
        M12 = self._c_M12 + m_cos
        M22 = self._c_M22
        return np.array([[M11, M12],
                         [M12, M22]])

    def C(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        h = -self._c_h * np.sin(q[1])
        return np.array([[h*dq[1],          h*(dq[0]+dq[1])],
                         [-h*dq[0],         0.0            ]])

    def g(self, q: np.ndarray) -> np.ndarray:
        c1  = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        g1  = self._c_g1a*c1 + self._c_g2*c12
        g2  = self._c_g2*c12
        return np.array([g1, g2])

    def forward(self, q: np.ndarray, dq: np.ndarray, u: float) -> np.ndarray:
        Bu  = np.array([u, 0.0])
        rhs = Bu - self.C(q, dq) @ dq - self.g(q)
        return np.linalg.solve(self.M(q), rhs)


class MPCController(BaseController):

    def __init__(self, config: dict):
        super().__init__(config)

        self.torque_limit = config["simulation"].get("torque_limit", np.inf)
        self.params       = get_system_params(config=config)
        self.dynamics     = Dynamics(self.params)

        self.K     = 512
        self.N     = 40
        self.dt    = 0.02
        self.lam = 5000.0
        self.sigma = 0.5

        self._Q    = np.diag([5.0, 20.0, 0.1, 0.5])
        self._Qf   = 10.0 * self._Q
        self._r    = 0.01

        self._goal  = np.array([np.pi/2, 0.0, 0.0, 0.0])
        self._U_nom = np.zeros(self.N)

    def _batch_forward(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> np.ndarray:
        d   = self.dynamics
        c2, s2  = np.cos(q[:, 1]), np.sin(q[:, 1])
        c1, c12  = np.cos(q[:, 0]), np.cos(q[:, 0] + q[:, 1])

        m_cos = d._c_Mcos * c2
        M11   = d._c_M11 + 2.0*m_cos
        M12   = d._c_M12 + m_cos
        M22   = d._c_M22

        det   = M11*M22 - M12**2

        h     = -d._c_h * s2
        dq0, dq1 = dq[:, 0], dq[:, 1]

        Cdq0  = h*dq0*dq1 + h*(dq0 + dq1)*dq1
        Cdq1  = -h*dq0**2

        g1    = d._c_g1a*c1 + d._c_g2*c12
        g2    = d._c_g2*c12

        rhs0  = u   - Cdq0 - g1
        rhs1  = 0.0 - Cdq1 - g2

        qdd0  = ( M22*rhs0 - M12*rhs1) / det
        qdd1  = (-M12*rhs0 + M11*rhs1) / det

        return np.stack([qdd0, qdd1], axis=1)

    def _rk4_step(self, q:  np.ndarray, dq: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = self.dt

        k1_dq = dq
        k1_ddq = self._batch_forward(q, dq, u)

        k2_dq  = dq + 0.5*dt*k1_ddq
        k2_ddq = self._batch_forward(q + 0.5*dt*k1_dq, k2_dq, u)

        k3_dq  = dq + 0.5*dt*k2_ddq
        k3_ddq = self._batch_forward(q + 0.5*dt*k2_dq, k3_dq, u)

        k4_dq  = dq + dt*k3_ddq
        k4_ddq = self._batch_forward(q + dt*k3_dq, k4_dq, u)

        q_next  = q  + (dt/6.0)*(k1_dq  + 2.0*k2_dq  + 2.0*k3_dq  + k4_dq)
        dq_next = dq + (dt/6.0)*(k1_ddq + 2.0*k2_ddq + 2.0*k3_ddq + k4_ddq)

        return q_next, dq_next

    def _rollout(self, state: np.ndarray, V: np.ndarray) -> np.ndarray:
        q  = np.tile(state[:2], (self.K, 1))
        dq = np.tile(state[2:], (self.K, 1))

        costs = np.zeros(self.K)

        for t in range(self.N):
            u          = np.clip(V[:, t], -self.torque_limit, self.torque_limit)
            q, dq      = self._rk4_step(q, dq, u)
            costs     += self._running_cost(q, dq, V[:, t])

        costs += self._terminal_cost(q, dq)
        return costs

    def _state_error(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e_q1  = np.arctan2(np.sin(q[:, 0] - self._goal[0]), np.cos(q[:, 0] - self._goal[0]))
        e_q2  = np.arctan2(np.sin(q[:, 1] - self._goal[1]), np.cos(q[:, 1] - self._goal[1]))
        e_dq1 = dq[:, 0] - self._goal[2]
        e_dq2 = dq[:, 1] - self._goal[3]
        return np.stack([e_q1, e_q2, e_dq1, e_dq2], axis=1)

    def _running_cost(self, q: np.ndarray, dq: np.ndarray, u: np.ndarray) -> np.ndarray:
        e   = self._state_error(q, dq)
        Qe  = e @ self._Q
        return (Qe * e).sum(axis=1) + self._r * u**2

    def _terminal_cost(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        e   = self._state_error(q, dq)
        Qfe = e @ self._Qf
        return (Qfe * e).sum(axis=1)
    
    def reset(self):
        self._U_nom[:] = 0.0

    def _compute(self, state: np.ndarray, t: float) -> float:
        eps = np.random.randn(self.K, self.N) * self.sigma
        V   = self._U_nom[None, :] + eps

        S   = self._rollout(state, V)

        beta    = S.min()
        weights = np.exp(-(S - beta) / self.lam)
        weights /= weights.sum()

        self._U_nom += (weights[:, None] * eps).sum(axis=0)
        self._U_nom  = np.clip(self._U_nom, -self.torque_limit, self.torque_limit)

        u               = float(self._U_nom[0])
        self._U_nom[:-1] = self._U_nom[1:]
        self._U_nom[-1]  = 0.0

        return u


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    mpc = MPCController(config)

    state = np.array([0.0, np.pi/4, 0.1, -0.2])
    V     = np.random.randn(mpc.K, mpc.N) * mpc.sigma

    q  = np.tile(state[:2], (mpc.K, 1))
    dq = np.tile(state[2:], (mpc.K, 1))
    u  = V[:, 0]

    qdd = mpc._batch_forward(q, dq, u)
    print(f"_batch_forward output shape : {qdd.shape}")

    q_next, dq_next = mpc._rk4_step(q, dq, u)
    print(f"_rk4_step q  shape          : {q_next.shape}")
    print(f"_rk4_step dq shape          : {dq_next.shape}")
    print(f"sample q_next[0]            : {q_next[0]}")

    print("End of file")