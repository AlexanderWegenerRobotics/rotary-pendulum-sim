import yaml
import multiprocessing as mp

from src.double_pendulum import DoublePendulum
from controllers.ilqr_controller import ILQRController


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = load_config()
    controller = ILQRController(config)

    env = DoublePendulum(config, controller)
    env.run()
