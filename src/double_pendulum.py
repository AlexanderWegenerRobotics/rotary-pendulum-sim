import multiprocessing as mp
import ctypes
import numpy as np

from src.simulation import Simulation


def _sim_process(config, shared_state, shared_torque, done_event, controller_name):
    sim = Simulation(config, shared_state, shared_torque, done_event, controller_name)
    sim.run()


def _control_process(controller, config, shared_state, shared_torque, done_event):
    import time
    control_hz = config["simulation"].get("control_hz", config["simulation"]["physics_hz"])
    control_dt = 1.0 / control_hz

    while not done_event.is_set():
        t_start = time.perf_counter()

        with shared_state.get_lock():
            state = np.frombuffer(shared_state.get_obj(), dtype=np.float64).copy()

        t = state[4]
        torque = controller.compute(state[:4], t)

        with shared_torque.get_lock():
            shared_torque.value = torque

        elapsed = time.perf_counter() - t_start
        sleep = control_dt - elapsed
        if sleep > 0:
            time.sleep(sleep)


class DoublePendulum:
    def __init__(self, config: dict, controller):
        self.config = config
        self.controller = controller

    def run(self):
        shared_state = mp.Array(ctypes.c_double, 5)  # [q1, q2, dq1, dq2, t]
        shared_torque = mp.Value(ctypes.c_double, 0.0)
        done_event = mp.Event()
        controller_name = type(self.controller).__name__

        sim_proc = mp.Process(target=_sim_process, args=(self.config, shared_state, shared_torque, done_event, controller_name), daemon=True)
        ctrl_proc = mp.Process(target=_control_process, args=(self.controller, self.config, shared_state, shared_torque, done_event), daemon=True)

        sim_proc.start()
        ctrl_proc.start()

        sim_proc.join()
        ctrl_proc.join()