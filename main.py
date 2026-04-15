import yaml, time
import multiprocessing as mp

from src.double_pendulum import DoublePendulum
from controllers.spring_controller import SpringController
from controllers.lqr_controller import LQRController
from controllers.mpc_controller import MPCController
from controllers.ilqr_controller import iLQRController
import numpy as np

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = load_config()
    controller = iLQRController(config)
    env = DoublePendulum(config, controller)
    start_time = time.time()
    print("Start test")
    env.run()
    print(f"Finished. Duration: {time.time() - start_time:.4f}s")
