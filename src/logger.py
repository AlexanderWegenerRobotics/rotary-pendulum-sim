import h5py
import json
import os
import time
from datetime import datetime


class Logger:
    def __init__(self, log_path: str, controller_name: str, config: dict):
        os.makedirs(log_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hdf5_path = os.path.join(log_path, f"{ts}.h5")
        self.meta_path = os.path.join(log_path, f"{ts}_meta.json")

        self._meta = {
            "controller": controller_name,
            "start_time": datetime.now().isoformat(),
            "config": config,
        }

        self._file = h5py.File(self.hdf5_path, "w")
        max_steps = int(config["test"]["duration"] * config["simulation"]["physics_hz"]) + 1000
        self._t          = self._file.create_dataset("t",          shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._q1         = self._file.create_dataset("q1",         shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._q2         = self._file.create_dataset("q2",         shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._dq1        = self._file.create_dataset("dq1",        shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._dq2        = self._file.create_dataset("dq2",        shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._u          = self._file.create_dataset("u",          shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._solve_time = self._file.create_dataset("solve_time", shape=(0,), maxshape=(max_steps,), dtype="f8")
        self._idx = 0
        self._start_wall = time.time()

    def log_step(self, t: float, state, torque: float, solve_time: float):
        i = self._idx
        for ds in (self._t, self._q1, self._q2, self._dq1, self._dq2, self._u, self._solve_time):
            ds.resize((i + 1,))
        self._t[i]          = t
        self._q1[i]         = state[0]
        self._q2[i]         = state[1]
        self._dq1[i]        = state[2]
        self._dq2[i]        = state[3]
        self._u[i]          = torque
        self._solve_time[i] = solve_time
        self._idx += 1

    def close(self):
        self._file.close()
        self._meta["end_time"] = datetime.now().isoformat()
        self._meta["duration_s"] = round(time.time() - self._start_wall, 3)
        self._meta["steps_logged"] = self._idx
        with open(self.meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)