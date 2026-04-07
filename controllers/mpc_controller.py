import numpy as np
import casadi as ca
from controllers.base_controller import BaseController
from src.system_params import get_system_params, SystemParams
import yaml, time

class CasADiDynamics:
    """Builds symbolic equations of motion and integrator using CasADi."""

    def __init__(self, params: SystemParams):
        self.p = params
        self._build_symbolic_model()

    def _build_symbolic_model(self):
        # symbolic state and input
        q1 = ca.SX.sym("q1")
        q2 = ca.SX.sym("q2")
        dq1 = ca.SX.sym("dq1")
        dq2 = ca.SX.sym("dq2")
        tau = ca.SX.sym("tau")

        x = ca.vertcat(q1, q2, dq1, dq2)
        p = self.p

        # shorthand trig terms
        s2 = ca.sin(q2)
        c2 = ca.cos(q2)
        s1 = ca.sin(q1)
        s12 = ca.sin(q1 + q2)

        # mass matrix M(q) [2x2]
        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2)
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2)
        m22 = p.I2 + p.m2 * p.lc2**2

        M = ca.vertcat(ca.horzcat(m11, m12),
                       ca.horzcat(m12, m22))

        # coriolis/centripetal C(q, dq) * dq
        h = p.m2 * p.l1 * p.lc2 * s2
        C_vec = ca.vertcat(-h * (2 * dq1 * dq2 + dq2**2),
                            h * dq1**2)
        # note: c_vec is C(q,dq)*dq directly, not C matrix

        # gravity g(q)
        g_vec = ca.vertcat(-(p.m1 * p.lc1 + p.m2 * p.l1) * 9.81 * ca.cos(q1) - p.m2 * p.lc2 * 9.81 * ca.cos(q1 + q2),
                           -p.m2 * p.lc2 * 9.81 * ca.cos(q1 + q2))

        # damping
        D_vec = ca.vertcat(p.b1 * dq1, p.b2 * dq2)

        # B * tau
        B_tau = ca.vertcat(tau, 0)

        # M(q) * ddq = B*tau - C*dq - g(q) - D*dq
        ddq = ca.solve(M, B_tau - C_vec - g_vec - D_vec)

        xdot = ca.vertcat(dq1, dq2, ddq)

        # store symbolic quantities
        self.x = x
        self.u = tau
        self.xdot = xdot

        # continuous dynamics function
        self.f_cont = ca.Function("f_cont", [x, tau], [xdot])

    def build_integrator_rk4(self, dt):
        """Returns a CasADi function: x_next = F(x, u) using RK4."""
        x = self.x
        u = self.u
        f = self.f_cont

        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return ca.Function("F_rk4", [x, u], [x_next])

    def compute_gravity_torques(self, q1_val, q2_val):
        """Numeric gravity torque vector for a given configuration."""
        p = self.p
        g1 = -(p.m1 * p.lc1 + p.m2 * p.l1) * 9.81 * np.cos(q1_val) - p.m2 * p.lc2 * 9.81 * np.cos(q1_val + q2_val)
        g2 = -p.m2 * p.lc2 * 9.81 * np.cos(q1_val + q2_val)
        return g1, g2


class MPCController(BaseController):
    """Nonlinear MPC using direct collocation with CasADi + IPOPT."""

    # --- tuning parameters ---
    N = 20                    # prediction horizon steps
    MPC_DT = 0.02            # MPC discretization [s] (=> 0.4s lookahead)
    TAU_MAX = 5.0             # default torque limit [Nm]

    # cost weights for stabilization (state = [q1, q2, dq1, dq2])
    Q_DIAG = [15.0, 25.0, 1.0, 1.0]    # higher weight on q2 (passive joint)
    R_WEIGHT = 0.01
    QF_SCALE = 5.0                       # terminal cost = QF_SCALE * Q

    # swing-up parameters
    SWINGUP_KE_GAIN = 2.0     # energy shaping gain (lower = gentler approach)
    SWINGUP_KD = 0.5          # damping injection on dq1 during swing-up
    SWINGUP_ANGLE_TOL = 0.8   # switch to MPC when q1 within this of upright [rad]

    # solver tuning
    IPOPT_MAX_ITER = 50       # keep low for real-time feasibility
    IPOPT_TOL = 1e-4
    WARM_START = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.params = get_system_params(config)
        self.TAU_MAX = config["simulation"].get("torque_limit", 5.0)
        self.x_ref = np.array(config["test"]["target_state"])
        self._E_target = self._total_energy(self.x_ref)

        self.dynamics = None
        self._F = None
        self._solver = None

        self.mpc_cmpt_times = []
        self.mpc_cmpt_cnt = 0

        # warm start storage
        self._prev_w0 = None
        self._mode = "swingup"  # "swingup" or "mpc"

    def _ensure_solver(self):
        """Build CasADi dynamics + NLP on first use (deferred for pickle safety)."""
        if self._solver is None:
            self.dynamics = CasADiDynamics(self.params)
            self._F = self.dynamics.build_integrator_rk4(self.MPC_DT)
            self._build_nlp()

    def _total_energy(self, state):
        """Compute total mechanical energy (KE + PE) for swing-up controller."""
        q1, q2, dq1, dq2 = state
        p = self.params

        # potential energy (zero at mount height)
        # link1 CoM height: lc1 * sin(q1)
        # link2 CoM height: l1 * sin(q1) + lc2 * sin(q1 + q2)
        PE = (p.m1 * p.lc1 * np.sin(q1) + p.m2 * (p.l1 * np.sin(q1) + p.lc2 * np.sin(q1 + q2))) * 9.81

        # kinetic energy using mass matrix evaluated numerically
        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * np.cos(q2))
        m12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * np.cos(q2))
        m22 = p.I2 + p.m2 * p.lc2**2

        dq = np.array([dq1, dq2])
        M = np.array([[m11, m12], [m12, m22]])
        KE = 0.5 * dq @ M @ dq

        return KE + PE

    def _swingup_control(self, state):
        """Energy-based swing-up: pump energy toward upright level, then hand off."""
        q1, q2, dq1, dq2 = state

        E_curr = self._total_energy(state)
        E_err = E_curr - self._E_target

        # initial kick if nearly stationary
        if abs(dq1) < 0.05 and abs(E_err) > 0.1:
            return self._clip(self.TAU_MAX * 0.5, self.TAU_MAX)
        tau = -self.SWINGUP_KE_GAIN * E_err * dq1 - self.SWINGUP_KD * dq1

        return self._clip(tau, self.TAU_MAX)

    def _should_switch_to_mpc(self, state):
        """Check if we're close enough to upright for MPC to take over."""
        q1, q2, dq1, dq2 = state

        q1_err = abs(q1 - self.x_ref[0])
        vel_ok = abs(dq1) < 8.0 and abs(dq2) < 15.0

        return q1_err < self.SWINGUP_ANGLE_TOL and vel_ok

    def _should_fallback_to_swingup(self, state):
        """If MPC drifts too far from upright, fall back to swing-up."""
        q1_err = abs(state[0] - self.x_ref[0])
        return q1_err > 1.5

    def _build_nlp(self):
        """Set up the direct multiple-shooting NLP solved at each MPC step."""
        N = self.N
        nx, nu = 4, 1

        # decision variables: controls U_0..U_{N-1} and states X_0..X_N
        w = []      # all decision vars
        w0 = []     # initial guess
        lbw = []    # lower bounds
        ubw = []    # upper bounds
        g = []      # constraints
        lbg = []
        ubg = []
        J = 0       # objective

        Q = ca.diag(ca.DM(self.Q_DIAG))
        R = ca.DM(self.R_WEIGHT)
        Qf = self.QF_SCALE * Q
        x_ref = ca.DM(self.x_ref)

        # initial state (parameter, set at each solve)
        x0_param = ca.SX.sym("x0", nx)

        # build shooting nodes
        X_vars = []
        U_vars = []

        for k in range(N + 1):
            xk = ca.SX.sym(f"X_{k}", nx)
            w.append(xk)
            X_vars.append(xk)
            # state bounds: no hard limits except what's physical
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

        # initial state constraint: X_0 == x0_param
        g.append(X_vars[0] - x0_param)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        # dynamics constraints + stage costs
        for k in range(N):
            x_err = X_vars[k] - x_ref
            J += ca.mtimes([x_err.T, Q, x_err]) + R * U_vars[k]**2

            # shooting constraint: X_{k+1} == F(X_k, U_k)
            x_next = self._F(X_vars[k], U_vars[k])
            g.append(X_vars[k + 1] - x_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        # terminal cost
        x_err_N = X_vars[N] - x_ref
        J += ca.mtimes([x_err_N.T, Qf, x_err_N])

        # stack everything
        w_all = ca.vertcat(*w)
        g_all = ca.vertcat(*g)

        # NLP dict
        nlp = {"f": J, "x": w_all, "g": g_all, "p": x0_param}

        # solver options tuned for real-time
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

        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        # store sizes for extracting solution
        self._nw = w_all.shape[0]
        self._nx = nx
        self._nu = nu
        self._w0_default = np.array(w0, dtype=np.float64)
        self._lbw = np.array(lbw, dtype=np.float64)
        self._ubw = np.array(ubw, dtype=np.float64)
        self._lbg = np.array(lbg, dtype=np.float64)
        self._ubg = np.array(ubg, dtype=np.float64)

        # index bookkeeping: w = [X0(4), U0(1), X1(4), U1(1), ..., X_N(4)]
        self._u0_idx = self._nx  # first control starts after X_0

    def _make_initial_guess(self, state):
        """Build warm-started initial guess by shifting previous solution."""
        if self._prev_w0 is not None:
            w0 = self._prev_w0.copy()
            nx, nu = self._nx, self._nu
            stride = nx + nu

            # shift: move k+1 -> k for k=0..N-2
            for k in range(self.N - 1):
                src = (k + 1) * stride
                dst = k * stride
                w0[dst:dst + stride] = w0[src:src + stride]

            last_src = (self.N - 1) * stride
            x_n_idx = self.N * stride
            w0[x_n_idx:x_n_idx + nx] = w0[last_src:last_src + nx]
            w0[:nx] = state
            return w0
        else:
            # cold start: interpolate linearly from current state to target
            w0 = self._w0_default.copy()
            nx, nu = self._nx, self._nu
            for k in range(self.N + 1):
                alpha = k / self.N
                x_interp = (1 - alpha) * state + alpha * self.x_ref
                if k < self.N:
                    idx = k * (nx + nu)
                    w0[idx:idx + nx] = x_interp
                    w0[idx + nx] = 0.0
                else:
                    idx = k * (nx + nu)
                    w0[idx:idx + nx] = x_interp
            return w0

    def _solve_mpc(self, state):
        """Solve the NLP and return optimal first control."""
        start_time = time.time()
        w0 = self._make_initial_guess(state)
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()
        lbw[:self._nx] = state
        ubw[:self._nx] = state

        try:
            sol = self._solver(x0=w0,lbx=lbw,ubx=ubw,lbg=self._lbg,ubg=self._ubg,p=state)
            w_opt = sol["x"].full().flatten()
            self._prev_w0 = w_opt
            tau = float(w_opt[self._u0_idx])
            self.mpc_cmpt_times.append(time.time() - start_time)
            self.mpc_cmpt_cnt += 1
            if self.mpc_cmpt_cnt > 100:
                t = self.mpc_cmpt_times
                print(f"MPC mean compute time: {np.mean(t)}, std: {np.std(t)}, min: {np.min(t)}, max: {np.max(t)}")
                self.mpc_cmpt_cnt = 0; self.mpc_cmpt_times = []
            return tau

        except RuntimeError:
            self._prev_w0 = None
            return 0.0

    def _compute(self, state: np.ndarray, t: float) -> float:
        """Main control dispatch: swing-up or MPC stabilization."""
        self._ensure_solver()

        if self._mode == "swingup":
            if self._should_switch_to_mpc(state):
                self._mode = "mpc"
                self._prev_w0 = None
                print("Mode change to MPC")
                return self._solve_mpc(state)
            return self._swingup_control(state)

        else:  # mpc mode
            if self._should_fallback_to_swingup(state):
                self._mode = "swingup"
                self._prev_w0 = None
                print("Mode change to SWINGUP")
                return self._swingup_control(state)
            return self._solve_mpc(state)

    def reset(self):
        self._prev_w0 = None
        self._mode = "swingup"