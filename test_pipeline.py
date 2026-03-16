"""
Smoke test — verifies model loading, control, rendering and logging.

Run without rendering:  python test_pipeline.py
Run with rendering:     python test_pipeline.py --render
Run with video:         python test_pipeline.py --render --video logs/test.mp4
"""

import argparse
from pathlib import Path
import numpy as np

from sim.sim import DoublePendulumSim
from sim.test_cases import list_test_cases
from common.logging import save_episode, load_episode
from common.metrics import compute_metrics, print_metrics
from common.base_controller import BaseController


class PController(BaseController):

    def __init__(self, kp: float = 5.0) -> None:
        super().__init__(name="PController")
        self.kp = kp

    def compute(self, x_obs: np.ndarray, t: float) -> float:
        """Drive joint1 to 90 deg — sanity check for control commands."""
        return float(self.kp * (np.pi / 2 - x_obs[0]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open CV2 viewer")
    parser.add_argument("--video",  type=str, default=None, help="Save video to path")
    args = parser.parse_args()

    print("── Test cases ───────────────────────────────────────")
    list_test_cases()

    print("\n── Model loading ────────────────────────────────────")
    sim = DoublePendulumSim(test_case="nominal", seed=0)
    print(f"  OK — nominal params: {sim.get_nominal_params()}")

    print("\n── Episode ──────────────────────────────────────────")
    ctrl    = PController(kp=5.0)
    results = sim.run_episode(
        ctrl,
        verbose=True,
        render=args.render,
        video_path=Path(args.video) if args.video else None,
    )
    print(f"  Steps        : {len(results['t'])}")
    print(f"  Success      : {results['success']}")
    print(f"  joint1 final : {np.degrees(results['x_true'][-1, 0]):.2f} deg  (target 90)")

    print("\n── Metrics ──────────────────────────────────────────")
    print_metrics(compute_metrics(results), label=ctrl.name)

    print("\n── Logging ──────────────────────────────────────────")
    h5, js = save_episode(
        results,
        metadata={"controller_name": ctrl.name, "controller_params": {"kp": ctrl.kp}},
        output_dir="logs",
    )
    print(f"  Saved  : {h5}")
    print(f"         : {js}")

    r2, meta = load_episode(h5)
    assert r2["success"] == results["success"]
    assert meta["controller_name"] == ctrl.name
    print(f"  Loaded : OK — {meta}")

    print("\n── All checks passed ────────────────────────────────")