"""
iLQR controller for the planar double pendulum.

No LQR or MPC.  Pure iLQR (Iterative Linear Quadratic Regulator) used in
two phases:

Phase 1 — Offline swing-up trajectory
    On construction the controller solves a long-horizon iLQR problem
    (N = 80 steps × 20 ms = 1.6 s) starting from the initial state given
    in the config (the user's expected starting state).  Many iterations
    are run because this happens *before* simulation begins, so wall time
    is unconstrained.  The result is a nominal trajectory plus the time-
    varying feedback gains K_fb produced by iLQR's backward pass.

Phase 2 — Online tracking + receding horizon
    During simulation:
      a) While the offline trajectory is still active, apply
             u_k = u_offline[k] + K_offline[k] @ (x − x_offline[k])
         This is iLQR's own time-varying feedback control law — no LQR.
      b) Once the offline trajectory is exhausted (system at upright),
         switch to a short-horizon receding-horizon iLQR (N = 20, 2
         iterations per re-solve) that re-plans every 5 control cycles.
         The receding plan also uses time-varying feedback gains for
         the steps it provides.

Latency handling
    The receding-horizon iLQR predicts the state forward by
    ``ceil(last_solve_time / dt)`` cycles before each solve so the new
    plan lines up with the future state.  Plans are warm-started with
    the previous plan shifted left.

Compute
    * Backward pass uses finite-difference Jacobians of the discrete
      RK4 step (5 RK4 calls per stage).
    * Short horizons during online phase keep solves at ~20 ms.
    * Offline phase runs once in __init__, so its 1–3 s wall time is
      paid before the sim clock starts.
"""

import numpy as np
import time
from scipy import linalg as sla
from controllers.base_controller import BaseController
from src.system_params import get_system_params


class iLQRController(BaseController):

    # ── iLQR design parameters ──────────────────────────────────────────
    PLAN_DT = 0.020

    OFFLINE_HORIZON  = 84         # 1.6 s lookahead for swing-up
    OFFLINE_MAX_ITER = 10

    ONLINE_HORIZON   = 16          # 0.4 s lookahead for stabilisation
    ONLINE_MAX_ITER  =1
    REPLAN_INTERVAL  = 1           # control cycles between re-solves

    Q_DIAG = ( 4.272113777294933, 5.398427944681879, 5.633859125309996, 4.0509479754039734)
    R_VAL  =  0.34281579648552424
    P_TERM_DIAG = (0.36625801499646554, 36.69276164410996, 0.012383228461722158, 0.48510091243688835)
    R_TERM_VAL  = 0.611033578943546

    OFFLINE_DONE_TOL = 0.7670835612721277         # |state error| below which offline ends

    # ════════════════════════════════════════════════════════════════════
    def __init__(self, config: dict):
        super().__init__(config)

        p = get_system_params(config)
        self.m1, self.m2   = p.m1, p.m2
        self.l1            = p.l1
        self.lc1, self.lc2 = p.lc1, p.lc2
        self.I1, self.I2   = p.I1, p.I2
        self.g             = p.g

        self.a1   = self.I1 + self.m1*self.lc1**2 + self.m2*self.l1**2
        self.a2   = self.I2 + self.m2*self.lc2**2
        self.beta = self.m2 * self.l1 * self.lc2

        self.arm1, self.arm2, self.damp1, self.damp2 = self._read_joint_params(config)

        sim_cfg = config["simulation"]
        control_hz = sim_cfg.get("control_hz", sim_cfg["physics_hz"])
        self.dt = 1.0 / control_hz
        self.torque_limit = sim_cfg.get("torque_limit", 160.0)

        # Target = both links pointing up
        self.x_goal = np.array([np.pi / 2.0, 0.0, 0.0, 0.0])

        # Terminal cost: solve a Riccati at upright to shape the value
        # function (no controller, just a quadratic terminal weight).
        self.P_term = self._terminal_cost_matrix()

        # ── Offline swing-up trajectory ──────────────────────────────────
        x0 = np.array(sim_cfg.get("initial_state",
                                   config["test"].get("initial_state",
                                                      [-np.pi/2, 0, 0, 0])),
                      dtype=float)
        if x0.size != 4:
            x0 = np.array([-np.pi/2, 0, 0, 0])

        self._u_off, self._x_off, self._K_off = self._solve_offline(x0)
        self._off_idx = 0
        self._offline_done = False

        # ── Online (receding horizon) state ──────────────────────────────
        self._u_on   = None
        self._x_on   = None
        self._K_on   = None
        self._on_idx = self.ONLINE_HORIZON
        self._last_solve_time = self.dt
        self._ctrl_count = 0

        # ── Startup race guard ──────────────────────────────────────────
        self._startup_done  = False
        self._startup_calls = 0

    # ════════════════════════════════════════════════════════════════════
    def _read_joint_params(self, config):
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_path(config["simulation"]["model_path"])
            dof_arm  = np.asarray(model.dof_armature).copy()
            dof_damp = np.asarray(model.dof_damping ).copy()
            arm1  = float(dof_arm[0])  if dof_arm.size  >= 1 else 0.002
            arm2  = float(dof_arm[1])  if dof_arm.size  >= 2 else 0.002
            damp1 = float(dof_damp[0]) if dof_damp.size >= 1 else 0.02
            damp2 = float(dof_damp[1]) if dof_damp.size >= 2 else 0.02
            return arm1, arm2, damp1, damp2
        except Exception:
            return 0.002, 0.002, 0.02, 0.02

    def _terminal_cost_matrix(self):
        """
        Solve discrete Riccati at upright to use as terminal cost.
        Not used as a controller — only as a quadratic weight on the
        last state of every iLQR rollout.
        """
        a1, a2, b = self.a1, self.a2, self.beta
        m1, m2 = self.m1, self.m2
        l1, lc1, lc2, g = self.l1, self.lc1, self.lc2, self.g

        M = np.array([[a1+a2+2*b+self.arm1, a2+b],
                      [a2+b,                a2+self.arm2]])
        Mi = np.linalg.inv(M)
        dGdq = np.array([[-(m1*lc1+m2*l1)*g - m2*g*lc2, -m2*g*lc2],
                         [-m2*g*lc2,                     -m2*g*lc2]])

        A_c = np.zeros((4, 4))
        A_c[0, 2] = 1.0
        A_c[1, 3] = 1.0
        A_c[2:, :2] = -Mi @ dGdq
        A_c[2:, 2:] = -Mi @ np.diag([self.damp1, self.damp2])
        B_c = np.zeros((4, 1))
        B_c[2:, :] = Mi @ np.array([[1.0], [0.0]])

        Z = np.zeros((5, 5))
        Z[:4, :4] = A_c * self.PLAN_DT
        Z[:4, 4:] = B_c * self.PLAN_DT
        eZ = sla.expm(Z)
        A_d = eZ[:4, :4]
        B_d = eZ[:4, 4:]
        Q = np.diag(self.P_TERM_DIAG)
        R = np.array([[self.R_TERM_VAL]])
        return sla.solve_discrete_are(A_d, B_d, Q, R)

    # ═══ helpers ════════════════════════════════════════════════════════
    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _state_error(self, x):
        e = x - self.x_goal
        e[0] = self._wrap(e[0])
        e[1] = self._wrap(e[1])
        return e

    # ═══ dynamics ═══════════════════════════════════════════════════════
    def _continuous_dynamics(self, x, u):
        q1, q2, dq1, dq2 = x
        a1, a2, b = self.a1, self.a2, self.beta
        m1, m2 = self.m1, self.m2
        l1, lc1, lc2, g = self.l1, self.lc1, self.lc2, self.g

        s2  = np.sin(q2); c2  = np.cos(q2)
        cq1 = np.cos(q1); c12 = np.cos(q1 + q2)

        M11 = a1 + a2 + 2*b*c2 + self.arm1
        M12 = a2 + b*c2
        M22 = a2 + self.arm2
        det = M11*M22 - M12**2

        C1 = -2*b*s2*dq1*dq2 - b*s2*dq2**2
        C2 = b*s2*dq1**2
        G1 = (m1*lc1 + m2*l1)*g*cq1 + m2*g*lc2*c12
        G2 = m2*g*lc2*c12

        rhs1 = u - self.damp1*dq1 - C1 - G1
        rhs2 =    - self.damp2*dq2 - C2 - G2

        ddq1 = ( M22*rhs1 - M12*rhs2) / det
        ddq2 = (-M12*rhs1 + M11*rhs2) / det
        return np.array([dq1, dq2, ddq1, ddq2])

    def _step(self, x, u, dt):
        uc = float(np.clip(u, -self.torque_limit, self.torque_limit))
        k1 = self._continuous_dynamics(x, uc)
        k2 = self._continuous_dynamics(x + 0.5*dt*k1, uc)
        k3 = self._continuous_dynamics(x + 0.5*dt*k2, uc)
        k4 = self._continuous_dynamics(x + dt*k3, uc)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # ═══ iLQR core ══════════════════════════════════════════════════════
    def _rollout(self, x0, u_traj):
        N = len(u_traj)
        x_traj = np.empty((N + 1, 4))
        x_traj[0] = x0
        for k in range(N):
            x_traj[k+1] = self._step(x_traj[k], u_traj[k], self.PLAN_DT)
        return x_traj

    def _traj_cost(self, x_traj, u_traj):
        Q = np.diag(self.Q_DIAG)
        R = self.R_VAL
        J = 0.0
        for k in range(len(u_traj)):
            e = self._state_error(x_traj[k])
            J += 0.5 * e @ Q @ e + 0.5 * R * u_traj[k]**2
        e = self._state_error(x_traj[-1])
        J += 0.5 * e @ self.P_term @ e
        return J

    def _ilqr(self, x0, u_init, max_iter):
        """
        Generic iLQR solve.
        Returns (u_traj, x_traj, K_fb, J_final).
        """
        Q = np.diag(self.Q_DIAG)
        R = self.R_VAL
        dt = self.PLAN_DT
        N = len(u_init)

        u_traj = u_init.copy()
        x_traj = self._rollout(x0, u_traj)
        J = self._traj_cost(x_traj, u_traj)
        reg = 1e-3
        K_fb_best = np.zeros((N, 4))

        for _ in range(max_iter):
            # ── backward pass ────────────────────────────────────────
            Vx  = self.P_term @ self._state_error(x_traj[N])
            Vxx = self.P_term.copy()
            k_ff = np.zeros(N)
            K_fb = np.zeros((N, 4))
            ok = True

            for i in range(N - 1, -1, -1):
                eps = 1e-6
                A = np.empty((4, 4))
                for j in range(4):
                    xp = x_traj[i].copy(); xp[j] += eps
                    xm = x_traj[i].copy(); xm[j] -= eps
                    A[:, j] = (self._step(xp, u_traj[i], dt)
                               - self._step(xm, u_traj[i], dt)) / (2*eps)
                bv = ((self._step(x_traj[i], u_traj[i] + eps, dt)
                        - self._step(x_traj[i], u_traj[i] - eps, dt)) / (2*eps))

                e   = self._state_error(x_traj[i])
                Qx  = Q @ e + A.T @ Vx
                Qu  = R * u_traj[i] + bv @ Vx
                Qxx = Q + A.T @ Vxx @ A
                Qux = bv @ Vxx @ A
                Quu = R + bv @ Vxx @ bv

                Quu_r = Quu + reg
                
                # FIX 2: Check for NaN properly to prevent warnings and division by zero
                if not (Quu_r > 1e-12):
                    ok = False
                    break

                inv = 1.0 / Quu_r
                k_ff[i] = -inv * Qu
                K_fb[i] = -inv * Qux

                Vx  = Qx - Qux * (Qu * inv)
                Vxx = Qxx - np.outer(Qux, Qux) * inv
                Vxx = 0.5 * (Vxx + Vxx.T)

            if not ok:
                reg *= 10.0
                continue

            # ── forward line search ──────────────────────────────────
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.01, 0.001):
                xn = np.empty_like(x_traj)
                un = np.empty_like(u_traj)
                xn[0] = x0
                for k in range(N):
                    dx    = xn[k] - x_traj[k]
                    
                    # FIX 1: Wrap angles so differences over the pi boundary don't explode
                    dx[0] = self._wrap(dx[0])
                    dx[1] = self._wrap(dx[1])
                    
                    un[k] = u_traj[k] + alpha * k_ff[k] + K_fb[k] @ dx
                    un[k] = np.clip(un[k], -self.torque_limit, self.torque_limit)
                    xn[k+1] = self._step(xn[k], un[k], dt)
                Jn = self._traj_cost(xn, un)
                if Jn < J:
                    x_traj, u_traj, J = xn, un, Jn
                    K_fb_best = K_fb.copy()
                    reg = max(reg / 10.0, 1e-6)
                    improved = True
                    break

            if not improved:
                reg *= 10.0
                if reg > 1e8:
                    break

        return u_traj, x_traj, K_fb_best, J

    # ═══ offline solve ══════════════════════════════════════════════════
    def _solve_offline(self, x0):
        """
        Long-horizon iLQR solve at construction time.  Used for swing-up
        and any large initial transient.  Wall time is unconstrained
        because this runs before the simulation clock starts.
        """
        u_init = np.zeros(self.OFFLINE_HORIZON)
        u_off, x_off, K_off, _ = self._ilqr(x0, u_init,
                                             max_iter=self.OFFLINE_MAX_ITER)
        return u_off, x_off, K_off

    # ═══ online MPC (receding horizon iLQR) ═════════════════════════════
    def _predict_forward(self, x_now, lag_cycles):
        """Forward-predict by ``lag_cycles`` PLAN_DT steps using current plan."""
        x = x_now.copy()
        for i in range(lag_cycles):
            if self._u_on is not None and (self._on_idx + i) < self.ONLINE_HORIZON:
                u = float(self._u_on[self._on_idx + i])
            else:
                u = 0.0
            x = self._step(x, u, self.PLAN_DT)
        return x

    def _online_step(self, state):
        """Receding horizon iLQR with latency-compensated re-planning."""
        need_replan = (self._ctrl_count % self.REPLAN_INTERVAL == 0
                       or self._on_idx >= self.ONLINE_HORIZON
                       or self._u_on is None)

        if need_replan:
            lag_cycles = max(1, int(np.ceil(self._last_solve_time / self.PLAN_DT)))
            lag_cycles = min(lag_cycles, self.ONLINE_HORIZON - 1)
            x_pred = self._predict_forward(state, lag_cycles)

            if self._u_on is not None:
                u_init = np.concatenate([self._u_on[lag_cycles:],
                                         np.zeros(lag_cycles)])
            else:
                u_init = np.zeros(self.ONLINE_HORIZON)

            t0 = time.perf_counter()
            try:
                u_new, x_new, K_new, _ = self._ilqr(x_pred, u_init,
                                                     max_iter=self.ONLINE_MAX_ITER)
                if (np.all(np.isfinite(u_new))
                        and np.all(np.isfinite(x_new))
                        and np.all(np.isfinite(K_new))):
                    self._u_on = u_new
                    self._x_on = x_new
                    self._K_on = K_new
                    self._on_idx = 0
            except Exception:
                pass
            self._last_solve_time = time.perf_counter() - t0

        if (self._u_on is not None
                and self._on_idx < self.ONLINE_HORIZON
                and self._x_on is not None):
            x_nom = self._x_on[self._on_idx]
            dx = state - x_nom
            dx[0] = self._wrap(dx[0])
            dx[1] = self._wrap(dx[1])
            u_ff = float(self._u_on[self._on_idx])
            u_fb = float(self._K_on[self._on_idx] @ dx)
            u    = u_ff + u_fb
        else:
            u = 0.0

        self._on_idx += 1
        self._ctrl_count += 1
        return float(np.clip(u, -self.torque_limit, self.torque_limit))

    # ═══ main entry ═════════════════════════════════════════════════════
    def _compute(self, state: np.ndarray, t: float) -> float:
        # Startup race guard.
        if not self._startup_done:
            self._startup_calls += 1
            if t == 0.0 and self._startup_calls < 100:
                return 0.0
            self._startup_done = True

        # Phase 1: track the offline trajectory
        if not self._offline_done and self._off_idx < self.OFFLINE_HORIZON:
            x_nom = self._x_off[self._off_idx]
            dx = state - x_nom
            dx[0] = self._wrap(dx[0])
            dx[1] = self._wrap(dx[1])
            u_ff = float(self._u_off[self._off_idx])
            u_fb = float(self._K_off[self._off_idx] @ dx)
            u    = u_ff + u_fb
            self._off_idx += 1

            # Mark offline as done once we reach the end
            if self._off_idx >= self.OFFLINE_HORIZON:
                self._offline_done = True

            # FIX 3: Replaced self._clip with standard np.clip for safety
            return float(np.clip(u, -self.torque_limit, self.torque_limit))

        # Phase 2: receding horizon iLQR
        u = self._online_step(state)
        return float(np.clip(u, -self.torque_limit, self.torque_limit))