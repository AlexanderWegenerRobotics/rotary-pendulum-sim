import yaml
import multiprocessing as mp

from src.double_pendulum import DoublePendulum
from controllers.spring_controller import SpringController


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = load_config()
    controller = SpringController(config, kp=0.0, kd=2.0)

    env = DoublePendulum(config, controller)
    env.run()
