import threading
import time
import numpy as np
import mujoco
import cv2
import os
from datetime import datetime

from src.logger import Logger


class Simulation:
    def __init__(self, config: dict, shared_state, shared_torque, done_event, controller_name: str):
        self.cfg = config
        self.sim_cfg = config["simulation"]
        self.test_cfg = config["test"]

        self.shared_state = shared_state
        self.shared_torque = shared_torque
        self.done_event = done_event

        self.model = mujoco.MjModel.from_xml_path(self.sim_cfg["model_path"])
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 1.0 / self.sim_cfg["physics_hz"]
        self.physics_dt = self.model.opt.timestep
        self.render_dt = 1.0 / self.sim_cfg["render_fps"]

        self._set_initial_state(self.test_cfg.get("initial_state", [0, 0, 0, 0]))

        self.headless = self.sim_cfg.get("headless", False)
        self.video_log = self.sim_cfg.get("video_log", False)
        self.duration = self.test_cfg["duration"]

        log_path = self.sim_cfg["log_path"]
        self.logger = Logger(log_path, controller_name, config)

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

    def _physics_loop(self):
        t_wall_start = time.perf_counter()

        while self.data.time < self.duration and not self.done_event.is_set():
            with self.shared_torque.get_lock():
                torque = self.shared_torque.value

            self.data.ctrl[0] = np.clip(torque, -self.sim_cfg["torque_limit"], self.sim_cfg["torque_limit"])
            mujoco.mj_step(self.model, self.data)

            state = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy(), [self.data.time]])
            with self.shared_state.get_lock():
                np.frombuffer(self.shared_state.get_obj(), dtype=np.float64)[:] = state

            self.logger.log_step(self.data.time, state[:4], self.data.ctrl[0])

            deadline = t_wall_start + self.data.time
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        self.done_event.set()

    def _render_frame(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data)
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