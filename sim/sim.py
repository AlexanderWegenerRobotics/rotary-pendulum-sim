"""
Rotary double pendulum simulation — dp-benchmark.
"""

from __future__ import annotations

import time, threading
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
import mujoco

from sim.test_cases import TestCaseConfig, get_test_case


_MODEL_XML   = Path(__file__).parent / "assets" / "double_pendulum.xml"
_SIM_DT      = 0.001
_FALL_THRESH = 0.8
_VEL_NOISE   = 5.0
_UPRIGHT     = np.array([0.0, np.pi / 2, 0.0, 0.0])

_CAM_WIDTH   = 640
_CAM_HEIGHT  = 480
_RENDER_FPS  = 30


class DoublePendulumSim:

    VERSION = "0.1.0"

    def __init__(
        self,
        test_case: str | TestCaseConfig = "nominal",
        seed: int = 42,
    ) -> None:

        self._cfg   = get_test_case(test_case) if isinstance(test_case, str) else test_case
        self._rng   = np.random.default_rng(seed)
        self._lock  = threading.Lock()

        self._model = mujoco.MjModel.from_xml_path(str(_MODEL_XML))
        self._data  = mujoco.MjData(self._model)
        self._apply_mismatch()

        self._renderer = mujoco.Renderer(self._model, height=_CAM_HEIGHT, width=_CAM_WIDTH)

        self._running      = False
        self._control_fn:  Optional[Callable] = None
        self._results:     Optional[dict]      = None
        self._episode_done = threading.Event()

    def run_episode(
        self,
        controller: Callable[[np.ndarray, float], float],
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        render: bool = False,
        video_path: Optional[Path | str] = None,
    ) -> dict:
        """Run a full episode. If render=True, must be called from main thread."""
        if hasattr(controller, "reset"):
            controller.reset()

        self._reset(x0)

        n_steps = int(self._cfg.duration / self._cfg.control_dt)
        buf     = _EpisodeBuffer(n_steps)

        if render:
            self._run_rendered(controller, buf, verbose, video_path)
        else:
            self._run_headless(controller, buf, verbose)

        return buf.pack(self._cfg.id)

    def get_nominal_params(self) -> dict:
        """Return nominal model parameters for controller use."""
        return {
            "l1": 0.3, "l2": 0.3,
            "m1": 0.5, "m2": 0.3,
            "lc1": 0.15, "lc2": 0.15,
            "I1": 0.004, "I2": 0.002,
            "g": 9.81,
        }

    # ── Run modes ─────────────────────────────────────────────────────────────

    def _run_headless(self, controller, buf: "_EpisodeBuffer", verbose: bool) -> None:
        """Step control and physics synchronously."""
        x_obs = self._observe()

        for i in range(buf.n):
            t  = self._data.time
            t0 = time.perf_counter()
            tau = controller(x_obs, t)
            buf.solve_t[i] = time.perf_counter() - t0

            x_obs, info = self._step(tau)
            buf.record(i, info, x_obs)

            if info["done"]:
                buf.truncate(i, verbose)
                break

            if verbose and i % int(1.0 / self._cfg.control_dt) == 0:
                print(f"  t={t:.1f}s  |e|={np.linalg.norm(info['x_true'] - _UPRIGHT):.4f}")

    def _run_rendered(
        self,
        controller,
        buf: "_EpisodeBuffer",
        verbose: bool,
        video_path: Optional[Path | str],
    ) -> None:
        """
        Control runs in background thread, rendering owns main thread.
        Optionally saves video to video_path.
        """

        video_writer = None
        if video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, _RENDER_FPS, (_CAM_WIDTH, _CAM_HEIGHT)
            )
            if not video_writer.isOpened():
                print(f"[Video] Could not open writer at {video_path}")
                video_writer = None

        self._running = True
        ctrl_thread   = threading.Thread(
            target=self._control_loop,
            args=(controller, buf, verbose),
            daemon=True,
        )
        ctrl_thread.start()

        frame_dt = 1.0 / _RENDER_FPS

        while self._running:
            t0 = time.time()

            with self._lock:
                self._renderer.update_scene(self._data)
            frame_rgb = self._renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("dp-benchmark", frame_bgr)

            if video_writer is not None:
                video_writer.write(frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False

            elapsed = time.time() - t0
            sleep   = frame_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        ctrl_thread.join()
        cv2.destroyAllWindows()

        if video_writer is not None:
            video_writer.release()
            print(f"[Video] Saved to {video_path}")

    def _control_loop(self, controller, buf: "_EpisodeBuffer", verbose: bool) -> None:
        """Background thread — runs control at control_dt rate."""
        x_obs = self._observe()

        for i in range(buf.n):
            t  = self._data.time
            t0 = time.perf_counter()
            tau = controller(x_obs, t)
            buf.solve_t[i] = time.perf_counter() - t0

            x_obs, info = self._step(tau)
            buf.record(i, info, x_obs)

            if info["done"]:
                buf.truncate(i, verbose)
                break

            if verbose and i % int(1.0 / self._cfg.control_dt) == 0:
                print(f"  t={t:.1f}s  |e|={np.linalg.norm(info['x_true'] - _UPRIGHT):.4f}")

            time.sleep(self._cfg.control_dt)

        self._running = False

    # ── Physics ───────────────────────────────────────────────────────────────

    def _reset(self, x0: Optional[np.ndarray]) -> None:
        mujoco.mj_resetData(self._model, self._data)

        if x0 is not None:
            q, dq = np.asarray(x0[:2]), np.asarray(x0[2:])
        elif self._cfg.x0 is not None:
            arr = np.asarray(self._cfg.x0)
            q, dq = arr[:2], arr[2:]
        else:
            q  = np.array([0.0, np.pi / 2 + 0.05])
            dq = np.zeros(2)

        self._data.qpos[:] = q
        self._data.qvel[:] = dq
        mujoco.mj_forward(self._model, self._data)

    def _step(self, tau: float) -> tuple[np.ndarray, dict]:
        """Apply torque, advance physics, return (x_obs, info)."""
        u_desired = float(tau)
        u_applied = float(np.clip(tau, -self._cfg.tau_max, self._cfg.tau_max))
        n_sub     = max(1, round(self._cfg.control_dt / _SIM_DT))

        with self._lock:
            for _ in range(n_sub):
                t_now = self._data.time
                extra = sum(
                    mag / _SIM_DT
                    for t_imp, mag in self._cfg.impulses
                    if t_now <= t_imp < t_now + _SIM_DT
                )
                self._data.ctrl[0] = u_applied + extra
                mujoco.mj_step(self._model, self._data)

        x_true = self._true_state()
        x_obs  = self._observe()

        return x_obs, {
            "t":         self._data.time,
            "x_true":    x_true,
            "u_desired": u_desired,
            "u_applied": u_applied,
            "done":      self._is_done(x_true),
        }

    def _true_state(self) -> np.ndarray:
        with self._lock:
            return np.array([*self._data.qpos[:2], *self._data.qvel[:2]])

    def _observe(self) -> np.ndarray:
        """Return state with noise applied per test case config."""
        x = self._true_state()
        if self._cfg.noise_std > 0.0:
            n     = self._rng.normal(0.0, self._cfg.noise_std, size=4)
            n[2:] *= _VEL_NOISE
            x     = x + n
        return x

    def _is_done(self, x_true: np.ndarray) -> bool:
        """True if link2 has fallen too far from upright."""
        dev = abs(x_true[1] - np.pi / 2) % (2 * np.pi)
        return bool(min(dev, 2 * np.pi - dev) > _FALL_THRESH)

    def _apply_mismatch(self) -> None:
        """Scale body masses in MuJoCo model per test case config."""
        for name, scale in [("link1", self._cfg.mass_scale_link1),
                             ("link2", self._cfg.mass_scale_link2)]:
            if scale != 1.0:
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    self._model.body_mass[bid] *= scale


# ── Episode buffer ─────────────────────────────────────────────────────────────

class _EpisodeBuffer:
    """Pre-allocated arrays for one episode."""

    def __init__(self, n: int) -> None:
        self.n       = n
        self.t       = np.zeros(n)
        self.x_true  = np.zeros((n, 4))
        self.x_obs   = np.zeros((n, 4))
        self.u_des   = np.zeros(n)
        self.u_app   = np.zeros(n)
        self.solve_t = np.zeros(n)
        self.success = True

    def record(self, i: int, info: dict, x_obs: np.ndarray) -> None:
        self.t[i]      = info["t"]
        self.x_true[i] = info["x_true"]
        self.x_obs[i]  = x_obs
        self.u_des[i]  = info["u_desired"]
        self.u_app[i]  = info["u_applied"]

    def truncate(self, i: int, verbose: bool) -> None:
        """Pad remaining steps after failure."""
        self.success   = False
        self.t[i+1:]   = self.t[i]
        self.x_true[i+1:] = self.x_true[i]
        self.x_obs[i+1:]  = self.x_obs[i]
        if verbose:
            print(f"  Failed at t={self.t[i]:.2f}s")

    def pack(self, test_case_id: str) -> dict:
        return {
            "t":            self.t,
            "x_true":       self.x_true,
            "x_obs":        self.x_obs,
            "u_desired":    self.u_des,
            "u_applied":    self.u_app,
            "solve_time":   self.solve_t,
            "success":      self.success,
            "test_case_id": test_case_id,
        }