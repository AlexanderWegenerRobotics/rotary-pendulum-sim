import numpy as np
import casadi as ca
import yaml
import time
from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class CasADiDynamics:
    def __init__(self, params: SystemParams):
        self.p = params
        self._build_symbolic_model()

    def _build_symbolic_model(self):
        q1 = ca.SX.sym("q1")
        q2 = ca.SX.sym("q2")
        dq1 = ca.SX.sym("dq1")
        dq2 = ca.SX.sym("dq2")
        tau = ca.SX.sym("tau")

        x = ca.vertcat(q1, q2, dq1, dq2)
        p = self.p

        c2 = ca.cos(q2)
        s2 = ca.sin(q2)

        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2)
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
        m22 = p.I2 + p.m2 * p.lc2**2
        M = ca.vertcat(ca.horzcat(m11, m12), ca.horzcat(m12, m22))

        h = p.m2 * p.l1 * p.lc2 * s2
        C_vec = ca.vertcat(-h * (2 * dq1 * dq2 + dq2**2), h * dq1**2)

        g_vec = ca.vertcat(
            (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * ca.cos(q1) + p.m2 * p.lc2 * p.g * ca.cos(q1 + q2),
            p.m2 * p.lc2 * p.g * ca.cos(q1 + q2)
        )

        D_vec = ca.vertcat(p.b1 * dq1, p.b2 * dq2)

        ddq = ca.solve(M, ca.vertcat(tau, 0) - C_vec - g_vec - D_vec)
        xdot = ca.vertcat(dq1, dq2, ddq)

        self.x = x
        self.u = tau
        self.xdot = xdot
        self.f_cont = ca.Function("f_cont", [x, tau], [xdot])

    def build_integrator_rk4(self, dt):
        x, u, f = self.x, self.u, self.f_cont
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return ca.Function("F_rk4", [x, u], [x_next])


class MPCController(BaseController):
    N = 40
    MPC_DT = 0.02
    TAU_MAX = 5.0

    Q_DIAG = [10.0, 40.0, 2.0, 2.0]
    R_WEIGHT = 0.1
    QF_SCALE = 50.0

    IPOPT_MAX_ITER = 80
    IPOPT_TOL = 1e-4
    WARM_START = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.params = get_system_params(config)

        test_cfg = config["test"]
        with open(test_cfg["test_cases_path"]) as f:
            scenarios = yaml.safe_load(f)
        scenario = scenarios["scenarios"][test_cfg["scenario"]]
        tl = scenario.get("torque_limit")
        if tl is not None:
            self.TAU_MAX = float(tl)

        self.x_ref = np.array(config["test"].get("target_state", [np.pi / 2, 0.0, 0.0, 0.0]))

        self.dynamics = None
        self._F = None
        self._F_fine = None
        self._solver = None

        self._prev_w0 = None
        self._last_tau = 0.0
        self._avg_solve_ms = 10.0

        self._solve_times = []
        self._solve_count = 0

    def _ensure_solver(self):
        if self._solver is not None:
            return
        self.dynamics = CasADiDynamics(self.params)
        self._F = self.dynamics.build_integrator_rk4(self.MPC_DT)
        self._F_fine = self.dynamics.build_integrator_rk4(0.002)
        self._build_nlp()

    def _build_nlp(self):
        N = self.N
        nx, nu = 4, 1

        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        J = 0

        R = ca.DM(self.R_WEIGHT)

        x0_param = ca.SX.sym("x0", nx)
        X_vars, U_vars = [], []

        for k in range(N + 1):
            xk = ca.SX.sym(f"X_{k}", nx)
            w.append(xk)
            X_vars.append(xk)
            lbw += [-ca.inf] * nx
            ubw += [ca.inf] * nx
            w0 += [float(self.x_ref[i]) for i in range(nx)]

            if k < N:
                uk = ca.SX.sym(f"U_{k}", nu)
                w.append(uk)
                U_vars.append(uk)
                lbw += [-self.TAU_MAX]
                ubw += [self.TAU_MAX]
                w0 += [0.0]

        g.append(X_vars[0] - x0_param)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        w_q1 = self.Q_DIAG[0]
        w_q2 = self.Q_DIAG[1]
        w_dq1 = self.Q_DIAG[2]
        w_dq2 = self.Q_DIAG[3]

        q1_ref = self.x_ref[0]
        q2_ref = self.x_ref[1]

        for k in range(N):
            q1k = X_vars[k][0]
            q2k = X_vars[k][1]
            dq1k = X_vars[k][2]
            dq2k = X_vars[k][3]

            J += w_q1 * (1 - ca.sin(q1k))**2
            J += w_q2 * (1 - ca.cos(q2k))**2

            J += 2.0 * w_q1 * (q1k - q1_ref)**2
            J += 2.0 * w_q2 * (q2k - q2_ref)**2

            J += w_dq1 * dq1k**2
            J += w_dq2 * dq2k**2
            J += R * U_vars[k]**2

            x_next = self._F(X_vars[k], U_vars[k])
            g.append(X_vars[k + 1] - x_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        q1N = X_vars[N][0]
        q2N = X_vars[N][1]
        dq1N = X_vars[N][2]
        dq2N = X_vars[N][3]
        Sf = self.QF_SCALE
        J += Sf * w_q1 * (1 - ca.sin(q1N))**2
        J += Sf * w_q2 * (1 - ca.cos(q2N))**2
        J += Sf * 2.0 * w_q1 * (q1N - q1_ref)**2
        J += Sf * 2.0 * w_q2 * (q2N - q2_ref)**2
        J += Sf * w_dq1 * dq1N**2
        J += Sf * w_dq2 * dq2N**2

        w_all = ca.vertcat(*w)
        g_all = ca.vertcat(*g)

        opts = {
            "ipopt.max_iter": self.IPOPT_MAX_ITER,
            "ipopt.tol": self.IPOPT_TOL,
            "ipopt.acceptable_tol": self.IPOPT_TOL * 10,
            "ipopt.acceptable_iter": 5,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": False,
            "ipopt.warm_start_init_point": "yes" if self.WARM_START else "no",
        }

        self._solver = ca.nlpsol("mpc", "ipopt", {"f": J, "x": w_all, "g": g_all, "p": x0_param}, opts)

        self._nx = nx
        self._nu = nu
        self._w0_default = np.array(w0, dtype=np.float64)
        self._lbw = np.array(lbw, dtype=np.float64)
        self._ubw = np.array(ubw, dtype=np.float64)
        self._lbg = np.array(lbg, dtype=np.float64)
        self._ubg = np.array(ubg, dtype=np.float64)
        self._u0_idx = nx

    def _make_initial_guess(self, state):
        nx, nu, N = self._nx, self._nu, self.N
        stride = nx + nu

        if self._prev_w0 is not None:
            w0 = self._prev_w0.copy()
            for k in range(N - 1):
                w0[k * stride:(k + 1) * stride] = w0[(k + 1) * stride:(k + 2) * stride]
            w0[N * stride:N * stride + nx] = w0[(N - 1) * stride:(N - 1) * stride + nx]
            w0[:nx] = state
            return w0

        w0 = self._w0_default.copy()
        for k in range(N + 1):
            alpha = k / N
            x_interp = (1 - alpha) * state + alpha * self.x_ref
            idx = k * stride if k < N else N * stride
            w0[idx:idx + nx] = x_interp
            if k < N:
                w0[idx + nx] = 0.0
        return w0

    def _solve_mpc(self, state):
        t0 = time.perf_counter()
        
        w0 = self._make_initial_guess(state)
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()
        lbw[:self._nx] = state
        ubw[:self._nx] = state

        try:
            sol = self._solver(x0=w0, lbx=lbw, ubx=ubw, lbg=self._lbg, ubg=self._ubg, p=state)
            w_opt = sol["x"].full().flatten()
            self._prev_w0 = w_opt
            tau = float(w_opt[self._u0_idx])

        except RuntimeError:
            self._prev_w0 = None
            tau = self._last_tau

        solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_times.append(solve_ms)
        self._solve_count += 1
        if self._solve_count >= 100:
            t = np.array(self._solve_times)
            print(f"MPC solve [ms]: mean={t.mean():.1f}  std={t.std():.1f} min={t.min():.1f}  max={t.max():.1f}")
            self._solve_times = []; self._solve_count = 0

        self._last_tau = tau
        return tau

    def _compute(self, state: np.ndarray, t: float) -> float:
        self._ensure_solver()
        return self._solve_mpc(state)

    def reset(self):
        self._prev_w0 = None
        self._last_tau = 0.0
        self._avg_solve_ms = 10.0