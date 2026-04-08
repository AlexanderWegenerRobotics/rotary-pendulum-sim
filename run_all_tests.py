"""
Batch evaluation runner — runs all test scenarios for a controller and
produces a results summary table.

Usage:
    python run_all_tests.py                    # defaults to MPCController
    python run_all_tests.py --controller lqr   # use LQRController
    python run_all_tests.py --headless         # no render window
    python run_all_tests.py --controller mpc --headless --flush     # flushes the csv even when stoped
"""

import yaml
import multiprocessing as mp
import argparse
import os
import time
import csv
from pathlib import Path
from copy import deepcopy

from src.double_pendulum import DoublePendulum
from src.metrics import compute_metrics


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_controller(name: str, config: dict):
    """Instantiate controller by name."""
    if name == "mpc":
        from controllers.mpc_controller import MPCController
        return MPCController(config)
    elif name == "lqr":
        from controllers.lqr_controller import LQRController
        return LQRController(config)
    elif name == "ilqr":
        from controllers.ilqr_controller import iLQRController
        return iLQRController(config)
    else:
        raise ValueError(f"Unknown controller: {name}")


def find_latest_log(log_dir: str) -> str:
    """Return path to the most recently created .h5 file in log_dir."""
    logs = sorted(Path(log_dir).glob("*.h5"), key=os.path.getmtime)
    if not logs:
        raise FileNotFoundError(f"No .h5 files in {log_dir}")
    return str(logs[-1])


def run_scenario(base_config: dict, scenario_name: str, controller_name: str,
                 headless: bool = True) -> dict:
    """Run one scenario and return its metrics."""
    config = deepcopy(base_config)
    config["test"]["scenario"] = scenario_name
    config["simulation"]["headless"] = headless
    config["simulation"]["video_log"] = not headless

    controller = get_controller(controller_name, config)
    env = DoublePendulum(config, controller)

    print(f"  Running {scenario_name}...", end=" ", flush=True)
    t0 = time.time()
    env.run()
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    log_path = find_latest_log(config["simulation"]["log_path"])
    upright = config["test"].get("target_state", [1.5708, 0.0, 0.0, 0.0])
    metrics = compute_metrics(log_path, upright=upright)
    metrics["scenario"] = scenario_name

    return metrics


def print_results_table(results: list):
    """Print formatted results table to console."""
    if not results:
        print("No results.")
        return

    # column definitions: (header, key, width, value_format)
    columns = [
        ("Scenario",        "scenario",           20, "s"),
        ("RMS Err [rad]",   "rms_error",          14, ".4f"),
        ("RMS Pos [rad]",   "rms_pos_error",      14, ".4f"),
        ("RMS Vel [r/s]",   "rms_vel_error",      14, ".4f"),
        ("RMS Tau [Nm]",    "rms_torque",         12, ".4f"),
        ("Settle [s]",      "settling_time_s",    12, ".3f"),
        ("Solve avg [ms]",  "mean_solve_time_ms", 16, ".2f"),
        ("Solve max [ms]",  "max_solve_time_ms",  16, ".2f"),
    ]

    # header
    header = " | ".join(f"{name:<{w}s}" if key == "scenario" else f"{name:>{w}s}"
                        for name, key, w, _ in columns)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        parts = []
        for name, key, w, vfmt in columns:
            val = r.get(key, float("nan"))
            if isinstance(val, str):
                parts.append(f"{val:<{w}s}")
            elif val != val:  # nan
                parts.append(f"{'N/A':>{w}s}")
            else:
                parts.append(f"{val:>{w}{vfmt}}")
        print(" | ".join(parts))

    print(sep)


def save_results_csv(results: list, path: str):
    """Save results to CSV for further analysis."""
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation runner")
    parser.add_argument("--controller", type=str, default="mpc", choices=["mpc", "lqr", "ilqr"], help="Controller to evaluate")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Specific scenarios to run (default: all)")
    parser.add_argument("--flush", action="store_true", help="Write CSV after each scenario (recover partial runs)")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    config = load_config()

    # load all available scenarios
    with open(config["test"]["test_cases_path"]) as f:
        all_scenarios = yaml.safe_load(f)["scenarios"]

    # select scenarios to run
    if args.scenarios:
        scenario_names = args.scenarios
    else:
        # run all, sorted by level then name
        scenario_names = sorted(all_scenarios.keys(),
                                key=lambda s: (all_scenarios[s].get("level", 0), s))

    print(f"Controller: {args.controller}")
    print(f"Scenarios:  {len(scenario_names)}")
    print(f"Headless:   {args.headless}")
    print(f"Press Ctrl+C to stop early (partial results will be saved)")
    print()

    csv_path = f"results_{args.controller}.csv"
    results = []
    aborted = False

    try:
        for i, name in enumerate(scenario_names):
            print(f"[{i+1}/{len(scenario_names)}]", end=" ")
            try:
                metrics = run_scenario(config, name, args.controller, headless=args.headless)
                results.append(metrics)
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"scenario": name, "rms_error": float("nan")})

            if args.flush and results:
                save_results_csv(results, csv_path)

    except KeyboardInterrupt:
        aborted = True
        print(f"\n\nAborted after {len(results)}/{len(scenario_names)} scenarios.")

    print()
    print_results_table(results)
    save_results_csv(results, csv_path)

    if aborted:
        remaining = scenario_names[len(results):]
        print(f"\nSkipped: {', '.join(remaining)}")


if __name__ == "__main__":
    main()