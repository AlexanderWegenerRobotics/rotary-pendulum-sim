import argparse
import csv
import json
import multiprocessing as mp
import platform
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.double_pendulum import DoublePendulum
from controllers.mpc_controller import MPCController


OUTPUT_DIR_DEFAULT = "horizon_sweep_output"
DATASET_NAME_DEFAULT = "macbook_pro"
HORIZONS_DEFAULT = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400]
ANGLE_TOL_DEG = 7.0
VEL_TOL_DEG_S = 50.0
SETTLE_WINDOW_S = 0.5
FINAL_WINDOW_S = 0.5
SATURATION_RATIO = 0.98
RMS_POS_SUCCESS_MAX_DEG = 90.0


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def find_latest_h5(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No .h5 logs found in {log_dir}")
    return logs[-1]


def load_log_arrays(log_path: Path):
    with h5py.File(log_path, "r") as f:
        t = f["t"][:]
        q1 = f["q1"][:]
        q2 = f["q2"][:]
        dq1 = f["dq1"][:]
        dq2 = f["dq2"][:]
        u = f["u"][:]
        solve_time = f["solve_time"][:]
    return t, q1, q2, dq1, dq2, u, solve_time


def compute_metrics(
    log_path: Path,
    target_state,
    torque_limit: float,
    control_period_ms: float,
    angle_tol_deg: float = ANGLE_TOL_DEG,
    vel_tol_deg_s: float = VEL_TOL_DEG_S,
    settle_window_s: float = SETTLE_WINDOW_S,
    final_window_s: float = FINAL_WINDOW_S,
    saturation_ratio: float = SATURATION_RATIO,
    rms_pos_success_max_deg: float = RMS_POS_SUCCESS_MAX_DEG,
):
    t, q1, q2, dq1, dq2, u, solve_time = load_log_arrays(log_path)
    target = np.asarray(target_state, dtype=float)

    q1_err = wrap_angle(q1 - target[0])
    q2_err = wrap_angle(q2 - target[1])
    dq1_err = dq1 - target[2]
    dq2_err = dq2 - target[3]

    pos_norm = np.sqrt(q1_err**2 + q2_err**2)
    vel_norm = np.sqrt(dq1_err**2 + dq2_err**2)
    state_norm = np.sqrt(q1_err**2 + q2_err**2 + dq1_err**2 + dq2_err**2)

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    settle_n = max(1, int(round(settle_window_s / dt))) if dt > 0 else 1
    final_n = max(1, int(round(final_window_s / dt))) if dt > 0 else 1

    angle_tol_rad = np.deg2rad(angle_tol_deg)
    inside_q1 = np.abs(q1_err) <= angle_tol_rad
    inside_q2 = np.abs(q2_err) <= angle_tol_rad
    inside_both = inside_q1 & inside_q2

    settling_time_s = float("nan")
    if len(inside_both) >= settle_n:
        conv = np.convolve(
            inside_both.astype(np.int32),
            np.ones(settle_n, dtype=np.int32),
            mode="valid",
        )
        hit = np.where(conv == settle_n)[0]
        if len(hit) > 0:
            settling_time_s = float(t[hit[0]])

    final_slice = slice(-final_n, None)
    final_max_q1_deg = float(np.rad2deg(np.max(np.abs(q1_err[final_slice]))))
    final_max_q2_deg = float(np.rad2deg(np.max(np.abs(q2_err[final_slice]))))
    final_max_dq1_deg_s = float(np.rad2deg(np.max(np.abs(dq1_err[final_slice]))))
    final_max_dq2_deg_s = float(np.rad2deg(np.max(np.abs(dq2_err[final_slice]))))

    final_pos_error_deg = float(np.rad2deg(np.mean(pos_norm[final_slice])))
    rms_error = float(np.sqrt(np.mean(state_norm**2)))
    rms_pos_error = float(np.sqrt(np.mean(pos_norm**2)))
    rms_vel_error = float(np.sqrt(np.mean(vel_norm**2)))
    rms_torque = float(np.sqrt(np.mean(u**2)))

    success = (
        (final_max_q1_deg < angle_tol_deg)
        and (final_max_q2_deg < angle_tol_deg)
        and (final_max_dq1_deg_s < vel_tol_deg_s)
        and (final_max_dq2_deg_s < vel_tol_deg_s)
        and np.isfinite(settling_time_s)
        and (np.rad2deg(rms_pos_error) < rms_pos_success_max_deg)
    )

    if np.isfinite(torque_limit) and torque_limit > 0:
        saturation_fraction = float(np.mean(np.abs(u) >= saturation_ratio * torque_limit))
    else:
        saturation_fraction = 0.0

    solve_ms = solve_time * 1e3
    solve_ms_warm = solve_ms[1:] if len(solve_ms) > 1 else solve_ms.copy()

    p95_solve_time_ms = float(np.percentile(solve_ms_warm, 95)) if len(solve_ms_warm) else float("nan")
    mean_solve_time_ms = float(np.mean(solve_ms_warm)) if len(solve_ms_warm) else float("nan")
    median_solve_time_ms = float(np.median(solve_ms_warm)) if len(solve_ms_warm) else float("nan")
    max_solve_time_ms = float(np.max(solve_ms)) if len(solve_ms) else float("nan")
    max_warm_solve_time_ms = float(np.max(solve_ms_warm)) if len(solve_ms_warm) else float("nan")
    first_solve_time_ms = float(solve_ms[0]) if len(solve_ms) else float("nan")

    realtime_feasible = bool(p95_solve_time_ms <= control_period_ms)

    return {
        "horizon": None,
        "success": int(success),
        "realtime_feasible": int(realtime_feasible),
        "settling_time_s": settling_time_s,
        "rms_error": rms_error,
        "rms_pos_error": rms_pos_error,
        "rms_vel_error": rms_vel_error,
        "rms_torque": rms_torque,
        "final_pos_error_deg": final_pos_error_deg,
        "final_max_q1_deg": final_max_q1_deg,
        "final_max_q2_deg": final_max_q2_deg,
        "final_max_dq1_deg_s": final_max_dq1_deg_s,
        "final_max_dq2_deg_s": final_max_dq2_deg_s,
        "saturation_fraction": saturation_fraction,
        "p95_solve_time_ms": p95_solve_time_ms,
        "mean_solve_time_ms": mean_solve_time_ms,
        "median_solve_time_ms": median_solve_time_ms,
        "max_solve_time_ms": max_solve_time_ms,
        "max_warm_solve_time_ms": max_warm_solve_time_ms,
        "first_solve_time_ms": first_solve_time_ms,
        "control_period_ms": float(control_period_ms),
        "log_path": str(log_path),
    }


def save_csv(rows, path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(path: Path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_one_horizon(base_config: dict, horizon: int, dataset_dir: Path, headless: bool):
    case_dir = dataset_dir / f"N_{horizon}"
    case_dir.mkdir(parents=True, exist_ok=True)

    config = deepcopy(base_config)
    config["simulation"]["headless"] = headless
    config["simulation"]["video_log"] = False
    config["simulation"]["log_path"] = str(case_dir)
    config["test"]["scenario"] = "nominal"

    control_hz = config["simulation"].get("control_hz", config["simulation"]["physics_hz"])
    control_period_ms = 1e3 / float(control_hz)

    controller = MPCController(config)
    controller.N = int(horizon)

    env = DoublePendulum(config, controller)
    env.run()

    log_path = find_latest_h5(case_dir)
    metrics = compute_metrics(
        log_path=log_path,
        target_state=config["test"]["target_state"],
        torque_limit=float(getattr(controller, "TAU_MAX", np.inf)),
        control_period_ms=control_period_ms,
    )
    metrics["horizon"] = int(horizon)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME_DEFAULT)
    parser.add_argument("--horizons", type=int, nargs="*", default=HORIZONS_DEFAULT)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    base_config = load_config(args.config)
    root_out_dir = Path(args.output_dir)
    dataset_dir = root_out_dir / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    horizons = sorted(set(int(h) for h in args.horizons))

    metadata = {
        "dataset_name": args.dataset_name,
        "device": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "config_path": args.config,
        "horizons": horizons,
        "angle_tol_deg": ANGLE_TOL_DEG,
        "vel_tol_deg_s": VEL_TOL_DEG_S,
        "settle_window_s": SETTLE_WINDOW_S,
        "final_window_s": FINAL_WINDOW_S,
        "rms_pos_success_max_deg": RMS_POS_SUCCESS_MAX_DEG,
        "simulation": base_config.get("simulation", {}),
        "test": base_config.get("test", {}),
    }
    save_metadata(dataset_dir / "metadata.json", metadata)

    results = []
    total = len(horizons)
    for i, horizon in enumerate(horizons, start=1):
        print(f"[{i}/{total}] Running horizon N = {horizon}")
        metrics = run_one_horizon(base_config, horizon, dataset_dir, headless=args.headless)
        results.append(metrics)

        settle_str = f"{metrics['settling_time_s']:.3f}" if np.isfinite(metrics["settling_time_s"]) else "nan"
        print(
            f"  success={metrics['success']} | "
            f"p95={metrics['p95_solve_time_ms']:.2f} ms | "
            f"max={metrics['max_solve_time_ms']:.2f} ms | "
            f"settling={settle_str} s | "
            f"rms_pos={np.rad2deg(metrics['rms_pos_error']):.2f} deg | "
            f"final_max_dq=({metrics['final_max_dq1_deg_s']:.1f}, {metrics['final_max_dq2_deg_s']:.1f}) deg/s"
        )

    results = sorted(results, key=lambda r: r["horizon"])
    save_csv(results, dataset_dir / "horizon_sweep_metrics.csv")
    print(f"\nSaved results to: {dataset_dir}")


if __name__ == "__main__":
    main()
