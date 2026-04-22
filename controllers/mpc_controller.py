import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import casadi as ca
import yaml
import time
from scipy.linalg import expm, solve_discrete_are
from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams


class CasADiDynamics:
    """Symbolic EOM for the rotary double pendulum using CasADi.

    Derives M(q)*ddq + C(q,dq)*dq + g(q) + D*dq = B*tau
    from Euler-Lagrange, matching MuJoCo's sign conventions and
    including joint armature inertia.
    """

    ARMATURE = 0.002  # joint rotor inertia [kg*m^2], from MuJoCo XML

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

        # --- mass matrix M(q), with armature on diagonal ---
        a = self.ARMATURE
        m11 = p.I1 + p.I2 + p.m1*p.lc1**2 + p.m2*(p.l1**2 + p.lc2**2 + 2*p.l1*p.lc2*c2) + a
        m12 = p.I2 + p.m2*(p.lc2**2 + p.l1*p.lc2*c2)
        m22 = p.I2 + p.m2*p.lc2**2 + a
        M = ca.vertcat(ca.horzcat(m11, m12), ca.horzcat(m12, m22))

        # --- Coriolis/centripetal: C(q,dq)*dq ---
        h = p.m2 * p.l1 * p.lc2 * s2
        C_vec = ca.vertcat(-h*(2*dq1*dq2 + dq2**2), h*dq1**2)

        # --- gravity torque g(q) ---
        # sign convention: g(q) enters as M*ddq = tau - C*dq - g(q) - D*dq
        # positive g at q1=0 means gravity pulls q1 negative (link falls down)
        # this matches MuJoCo's qfrc_bias
        g_vec = ca.vertcat(
            (p.m1*p.lc1 + p.m2*p.l1)*p.g*ca.cos(q1) + p.m2*p.lc2*p.g*ca.cos(q1 + q2),
            p.m2*p.lc2*p.g*ca.cos(q1 + q2))

        # --- viscous joint damping ---
        D_vec = ca.vertcat(p.b1*dq1, p.b2*dq2)

        # --- EOM: M*ddq = B*tau - C*dq - g - D*dq ---
        ddq = ca.solve(M, ca.vertcat(tau, 0) - C_vec - g_vec - D_vec)
        xdot = ca.vertcat(dq1, dq2, ddq)

        self.x = x
        self.u = tau
        self.xdot = xdot
        self.f_cont = ca.Function("f_cont", [x, tau], [xdot])

    def build_integrator_rk4(self, dt):
        """Returns CasADi function x_next = F(x, u) via 4th-order Runge-Kutta."""
        x, u, f = self.x, self.u, self.f_cont
        k1 = f(x, u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return ca.Function("F_rk4", [x, u], [x_next])


class MPCController(BaseController):
    """Nonlinear MPC via direct multiple-shooting with CasADi + IPOPT.

    Stage cost:    l(x,u) = (x-x_ref)^T Q (x-x_ref) + u^T R u + du^T S du
    Terminal cost:  V_f(x) = (x-x_ref)^T P (x-x_ref)

    P is the solution to the discrete algebraic Riccati equation (DARE)
    for the system linearized at the upright equilibrium. This is the
    standard infinite-horizon LQR cost-to-go, which provides a local
    Lyapunov function guaranteeing closed-loop stability when the
    terminal state lies within the region of attraction of the LQR
    controller (Rawlings & Mayne, Ch. 2).
    """

    # --- horizon ---
    N = 20                   # prediction steps
    MPC_DT = 0.04            # discretization [s] => 0.8s lookahead

    # --- default actuator limit (overridden by scenario) ---
    TAU_MAX = 5.0            # [Nm]

    # --- stage cost weights ---
    Q_DIAG = [10.0, 40.0, 2.0, 2.0]   # [q1, q2, dq1, dq2]
    R_WEIGHT = 0.1                      # control effort
    DU_WEIGHT = 0.02                    # control rate penalty

    # --- solver ---
    IPOPT_MAX_ITER = 80
    IPOPT_TOL = 1e-4

    def __init__(self, config: dict):
        super().__init__(config)
        self.params = get_system_params(config)

        # read torque limit from test scenario
        test_cfg = config["test"]
        with open(test_cfg["test_cases_path"]) as f:
            scenarios = yaml.safe_load(f)
        scenario = scenarios["scenarios"][test_cfg["scenario"]]
        tl = scenario.get("torque_limit")
        if tl is not None:
            self.TAU_MAX = float(tl)

        self.x_ref = np.array(
            config["test"].get("target_state", [np.pi/2, 0.0, 0.0, 0.0]))

        # CasADi objects built lazily in _ensure_solver() for pickle safety
        self.dynamics = None
        self._F = None
        self._solver = None
        self._P_term = None

        self._prev_w0 = None
        self._last_tau = 0.0

        # diagnostics
        self._solve_times = []
        self._solve_count = 0

    # ------------------------------------------------------------------ #
    #  lazy init (required for Windows spawn multiprocessing)             #
    # ------------------------------------------------------------------ #

    def _ensure_solver(self):
        if self._solver is not None:
            return
        self.dynamics = CasADiDynamics(self.params)
        self._F = self.dynamics.build_integrator_rk4(self.MPC_DT)
        self._compute_terminal_cost()
        self._build_nlp()

    # ------------------------------------------------------------------ #
    #  terminal cost via DARE                                             #
    # ------------------------------------------------------------------ #

    def _compute_terminal_cost(self):
        """Solve the DARE for the linearized discrete-time system at x_ref.

        The continuous-time Jacobians A_c, B_c are computed symbolically
        via CasADi, then discretized using the matrix exponential (exact
        ZOH discretization). The DARE is solved with the same Q and R
        used in the stage cost, so P is the true infinite-horizon LQR
        cost-to-go for the linearized system.
        """
        # continuous-time linearization at the equilibrium
        A_fn = ca.Function("A", [self.dynamics.x, self.dynamics.u],
                           [ca.jacobian(self.dynamics.xdot, self.dynamics.x)])
        B_fn = ca.Function("B", [self.dynamics.x, self.dynamics.u],
                           [ca.jacobian(self.dynamics.xdot, self.dynamics.u)])

        A_c = np.array(A_fn(self.x_ref, 0.0), dtype=np.float64)
        B_c = np.array(B_fn(self.x_ref, 0.0), dtype=np.float64)

        nx = A_c.shape[0]
        nu = B_c.shape[1]

        # exact ZOH discretization via matrix exponential
        #   [Ad Bd] = expm([Ac Bc; 0 0] * dt)[:nx, :]
        top = np.hstack([A_c, B_c])
        bot = np.zeros((nu, nx + nu))
        M_cont = np.vstack([top, bot])
        M_disc = expm(M_cont * self.MPC_DT)

        A_d = M_disc[:nx, :nx]
        B_d = M_disc[:nx, nx:]

        # DARE with same Q, R as stage cost — this is the textbook choice
        Q = np.diag(self.Q_DIAG).astype(np.float64)
        R = np.array([[self.R_WEIGHT]], dtype=np.float64)

        self._P_term = solve_discrete_are(A_d, B_d, Q, R)

    # ------------------------------------------------------------------ #
    #  NLP construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_nlp(self):
        """Construct the multiple-shooting NLP.

        Decision variables: X_0, U_0, X_1, U_1, ..., X_{N-1}, U_{N-1}, X_N
        Parameters: [x0 (4), u_prev (1)]

        The parameter u_prev is the torque applied at the previous step,
        used for the rate-of-change penalty on the first control input.
        """
        N = self.N
        nx, nu = 4, 1

        # cost matrices
        Q = ca.diag(ca.DM(self.Q_DIAG))
        R = ca.DM(self.R_WEIGHT)
        P = ca.DM(self._P_term)
        x_ref = ca.DM(self.x_ref)

        # NLP parameter: initial state + previous control
        x0_param = ca.SX.sym("x0", nx)
        u_prev_param = ca.SX.sym("u_prev", nu)
        p_nlp = ca.vertcat(x0_param, u_prev_param)

        # decision variable lists
        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        J = 0
        X_vars, U_vars = [], []

        for k in range(N + 1):
            # state node
            xk = ca.SX.sym(f"X_{k}", nx)
            w.append(xk)
            X_vars.append(xk)
            lbw += [-ca.inf] * nx
            ubw += [ca.inf] * nx
            w0 += [float(self.x_ref[i]) for i in range(nx)]

            # control node (not at final step)
            if k < N:
                uk = ca.SX.sym(f"U_{k}", nu)
                w.append(uk)
                U_vars.append(uk)
                lbw += [-self.TAU_MAX]
                ubw += [self.TAU_MAX]
                w0 += [0.0]

        # --- initial state constraint ---
        #g.append(X_vars[0] - x0_param)
        #lbg += [0.0] * nx
        #ubg += [0.0] * nx

        # --- stage costs + dynamics constraints ---
        for k in range(N):
            x_err = X_vars[k] - x_ref

            # quadratic state + control cost
            J += ca.mtimes([x_err.T, Q, x_err]) + R * U_vars[k]**2

            # control rate penalty (smooths torque profile)
            u_prev_k = u_prev_param if k == 0 else U_vars[k - 1]
            du = U_vars[k] - u_prev_k
            J += self.DU_WEIGHT * du**2

            # dynamics: X_{k+1} = F(X_k, U_k)
            x_next = self._F(X_vars[k], U_vars[k])
            g.append(X_vars[k + 1] - x_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        # --- terminal cost: x_N^T P x_N ---
        # P from DARE is the infinite-horizon LQR cost-to-go
        x_err_N = X_vars[N] - x_ref
        J += ca.mtimes([x_err_N.T, P, x_err_N])

        # --- assemble and create solver ---
        w_all = ca.vertcat(*w)
        g_all = ca.vertcat(*g)

        opts = {
            "ipopt.max_iter": self.IPOPT_MAX_ITER,
            "ipopt.tol": self.IPOPT_TOL,
            "ipopt.acceptable_tol": self.IPOPT_TOL * 10,
            "ipopt.acceptable_iter": 5,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter":20,
            "print_time": False,
            "ipopt.warm_start_init_point": "yes",
        }

        self._solver = ca.nlpsol(
            "mpc", "ipopt",
            {"f": J, "x": w_all, "g": g_all, "p": p_nlp},
            opts)

        # store dimensions and default arrays
        self._nx = nx
        self._nu = nu
        self._stride = nx + nu
        self._w0_default = np.array(w0, dtype=np.float64)
        self._lbw = np.array(lbw, dtype=np.float64)
        self._ubw = np.array(ubw, dtype=np.float64)
        self._lbg = np.array(lbg, dtype=np.float64)
        self._ubg = np.array(ubg, dtype=np.float64)
        self._u0_idx = nx  # U_0 sits right after X_0 in the decision vector

    # ------------------------------------------------------------------ #
    #  warm starting                                                      #
    # ------------------------------------------------------------------ #

    def _make_initial_guess(self, state):
        """Build initial guess for the NLP solver.

        Warm start: shift previous solution forward by one step.
        Cold start: linear interpolation from current state to target.
        """
        nx, nu, N = self._nx, self._nu, self.N
        s = self._stride

        if self._prev_w0 is not None:
            w0 = self._prev_w0.copy()
            # shift: copy segment k+1 into slot k
            for k in range(N - 1):
                w0[k*s : (k+1)*s] = w0[(k+1)*s : (k+2)*s]
            # fill last state node by copying from second-to-last
            w0[N*s : N*s + nx] = w0[(N-1)*s : (N-1)*s + nx]
            # pin X_0 to measured state
            w0[:nx] = state
            return w0

        # cold start: interpolate state, zero controls
        w0 = self._w0_default.copy()
        for k in range(N + 1):
            alpha = k / N
            x_interp = (1 - alpha) * state + alpha * self.x_ref
            idx = k * s if k < N else N * s
            w0[idx : idx + nx] = x_interp
            if k < N:
                w0[idx + nx] = 0.0
        return w0

    # ------------------------------------------------------------------ #
    #  solve                                                              #
    # ------------------------------------------------------------------ #

    def _solve_mpc(self, state):
        """Solve the NLP at the current state, return first control action."""
        t0 = time.perf_counter()

        w0 = self._make_initial_guess(state)

        # fix X_0 to measured state via bounds
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()
        lbw[:self._nx] = state
        ubw[:self._nx] = state

        # NLP parameter: [x0, u_prev]
        p_val = np.concatenate([state, [self._last_tau]])

        try:
            sol = self._solver(x0=w0, lbx=lbw, ubx=ubw, lbg=self._lbg, ubg=self._ubg, p=p_val)
            w_opt = sol["x"].full().flatten()
            self._prev_w0 = w_opt
            tau = float(w_opt[self._u0_idx])

        except RuntimeError:
            # solver failure: hold previous torque, reset warm start
            self._prev_w0 = None
            tau = self._last_tau

        # update last applied torque for next rate penalty
        self._last_tau = tau

        # diagnostics
        if False:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            self._solve_times.append(solve_ms)
            self._solve_count += 1
            if self._solve_count >= 100:
                t = np.array(self._solve_times)
                print(f"MPC solve [ms]: mean={t.mean():.1f}  std={t.std():.1f}  "
                    f"min={t.min():.1f}  max={t.max():.1f}")
                self._solve_times = []
                self._solve_count = 0

        return tau

    # ------------------------------------------------------------------ #
    #  BaseController interface                                           #
    # ------------------------------------------------------------------ #

    def _compute(self, state: np.ndarray, t: float) -> float:
        self._ensure_solver()
        return self._solve_mpc(state)

    def reset(self):
        self._prev_w0 = None
        self._last_tau = 0.0