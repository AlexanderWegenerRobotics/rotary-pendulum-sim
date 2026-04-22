import h5py
import numpy as np


def compute_metrics(log_path: str, upright: np.ndarray = None) -> dict:
    """Compute performance metrics from an HDF5 log file.

    Args:
        log_path: path to .h5 log
        upright: target state [q1, q2, dq1, dq2], defaults to [pi/2, 0, 0, 0]
    """
    if upright is None:
        upright = np.array([np.pi / 2, 0.0, 0.0, 0.0])

    with h5py.File(log_path, "r") as f:
        t   = f["t"][:]
        q1  = f["q1"][:]
        q2  = f["q2"][:]
        dq1 = f["dq1"][:]
        dq2 = f["dq2"][:]
        u   = f["u"][:]
        solve_time = f["solve_time"][:]

    # angle error with wrapping (handles q1 crossing +/-pi)
    q1_err = np.arctan2(np.sin(q1 - upright[0]), np.cos(q1 - upright[0]))
    q2_err = np.arctan2(np.sin(q2 - upright[1]), np.cos(q2 - upright[1]))
    dq1_err = dq1 - upright[2]
    dq2_err = dq2 - upright[3]

    error = np.stack([q1_err, q2_err, dq1_err, dq2_err], axis=1)

    # RMS of full state error (positions in rad, velocities in rad/s)
    rms_error = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))

    # separate position and velocity RMS for clearer reporting
    rms_pos = float(np.sqrt(np.mean(q1_err**2 + q2_err**2)))
    rms_vel = float(np.sqrt(np.mean(dq1_err**2 + dq2_err**2)))

    rms_torque = float(np.sqrt(np.mean(u**2)))

    settling_time = _settling_time(t, error)

    return {
        "rms_error":          rms_error,
        "rms_pos_error":      rms_pos,
        "rms_vel_error":      rms_vel,
        "rms_torque":         rms_torque,
        "settling_time_s":    settling_time,
        "mean_solve_time_ms": float(np.mean(solve_time) * 1e3),
        "max_solve_time_ms":  float(np.max(solve_time) * 1e3),
    }


def _settling_time(t, error, angle_tol_rad=np.deg2rad(5.0), window_s=0.5):
    if len(t) < 2:
        return float("nan")

    dt = float(np.median(np.diff(t)))
    window_n = max(1, int(round(window_s / dt)))

    q1_ok = np.abs(error[:, 0]) <= angle_tol_rad
    q2_ok = np.abs(error[:, 1]) <= angle_tol_rad
    inside = q1_ok & q2_ok

    if len(inside) < window_n:
        return float("nan")

    conv = np.convolve(inside.astype(np.int32), np.ones(window_n, dtype=np.int32), mode="valid")
    hit = np.where(conv == window_n)[0]

    if len(hit) == 0:
        return float("nan")

    return float(t[hit[0]])

def _success_flag(t, error, angle_tol_rad=np.deg2rad(5.0), final_window_s=0.5):
    if len(t) < 2:
        return False

    dt = float(np.median(np.diff(t)))
    window_n = max(1, int(round(final_window_s / dt)))

    q1_ok = np.abs(error[:, 0]) <= angle_tol_rad
    q2_ok = np.abs(error[:, 1]) <= angle_tol_rad
    inside = q1_ok & q2_ok

    return bool(np.all(inside[-window_n:]))