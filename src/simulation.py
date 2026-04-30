import threading
import time
import numpy as np
import mujoco
import yaml
import cv2
import os
from datetime import datetime

from src.logger import Logger


def _load_scenario(config: dict) -> dict:
    path = config["test"]["test_cases_path"]
    name = config["test"]["scenario"]
    with open(path) as f:
        all_scenarios = yaml.safe_load(f)
    return all_scenarios["scenarios"][name]


class Simulation:
    def __init__(self, config: dict, shared_state, shared_torque, shared_solve_time, done_event, controller_name: str):
        self.cfg = config
        self.sim_cfg = config["simulation"]
        self.test_cfg = config["test"]
        self.scenario = _load_scenario(config)

        self.shared_state      = shared_state
        self.shared_torque     = shared_torque
        self.shared_solve_time = shared_solve_time
        self.done_event        = done_event

        self.model = mujoco.MjModel.from_xml_path(self.sim_cfg["model_path"])
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 1.0 / self.sim_cfg["physics_hz"]
        self.physics_dt = self.model.opt.timestep
        self.render_dt = 1.0 / self.sim_cfg["render_fps"]

        self._set_initial_state(self.test_cfg.get("initial_state", [0, 0, 0, 0]))

        init_state = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy(), [self.data.time]])
        with self.shared_state.get_lock():
            np.frombuffer(self.shared_state.get_obj(), dtype=np.float64)[:] = init_state

        self.headless = self.sim_cfg.get("headless", False)
        self.video_log = self.sim_cfg.get("video_log", False)
        self.duration = self.test_cfg["duration"]

        self.torque_limit     = self.scenario["torque_limit"] or np.inf
        self.noise_std        = self.scenario.get("noise_std", 0.0)
        self.impulse_mag      = self.scenario.get("impulse_magnitude", 0.0)
        self.impulse_times    = self.test_cfg.get("impulse_times", [])
        self._fired_impulses  = set()

        log_path = self.sim_cfg["log_path"]
        self.logger = Logger(log_path, controller_name, config)

        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self._camera)
        self._camera.lookat[2] += 0.3  # shift up in z, tune this value
        self._camera.lookat[0] -= 0.3

        self._video_writer = None
        if self.video_log:
            self._init_video_writer(log_path)

        self._renderer = None
        if not self.headless:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.sim_cfg["render_height"],
                width=self.sim_cfg["render_width"],
            )

    def _set_initial_state(self, state):
        self.data.qpos[:] = state[:2]
        self.data.qvel[:] = state[2:]
        mujoco.mj_forward(self.model, self.data)

    def _init_video_writer(self, log_path):
        os.makedirs(log_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_path, f"{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            path, fourcc,
            self.sim_cfg["render_fps"],
            (self.sim_cfg["render_width"], self.sim_cfg["render_height"]),
        )

    def _apply_impulses(self):
        if self.impulse_mag == 0.0:
            return
        for t_imp in self.impulse_times:
            if t_imp not in self._fired_impulses and self.data.time >= t_imp:
                self.data.qvel[0] += self.impulse_mag
                self._fired_impulses.add(t_imp)

    def _noisy_state(self, state: np.ndarray) -> np.ndarray:
        if self.noise_std == 0.0:
            return state
        noisy = state.copy()
        noisy[:4] += np.random.normal(0.0, self.noise_std, 4)
        return noisy

    def _physics_loop(self):
        t_wall_start = time.perf_counter()

        while self.data.time < self.duration and not self.done_event.is_set():
            with self.shared_torque.get_lock():
                torque = self.shared_torque.value

            self._apply_impulses()

            self.data.ctrl[0] = np.clip(torque, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.model, self.data)

            state = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy(), [self.data.time]])
            noisy_state = self._noisy_state(state)

            with self.shared_state.get_lock():
                np.frombuffer(self.shared_state.get_obj(), dtype=np.float64)[:] = noisy_state

            with self.shared_solve_time.get_lock():
                solve_time = self.shared_solve_time.value

            self.logger.log_step(self.data.time, state[:4], self.data.ctrl[0], solve_time)

            deadline = t_wall_start + self.data.time
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        self.done_event.set()

    def _render_frame(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self._camera)
        frame = self._renderer.render()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def run(self):
        physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        physics_thread.start()

        last_render = time.time()

        while not self.done_event.is_set():
            now = time.time()
            if now - last_render >= self.render_dt:
                frame = self._render_frame()
                if frame is not None:
                    cv2.imshow("Double Pendulum", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.done_event.set()
                    if self._video_writer is not None:
                        self._video_writer.write(frame)
                last_render = now
            else:
                time.sleep(0.001)

        physics_thread.join()

        if self._video_writer:
            self._video_writer.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self.logger.close()