import h5py
import numpy as np

def compute_metrics(log_path: str, upright: np.ndarray = None) -> dict:
    if upright is None:
        upright = np.array([0.0, np.pi, 0.0, 0.0])

    with h5py.File(log_path, "r") as f:
        t   = f["t"][:]
        q1  = f["q1"][:]
        q2  = f["q2"][:]
        dq1 = f["dq1"][:]
        dq2 = f["dq2"][:]
        u   = f["u"][:]
        solve_time = f["solve_time"][:]

    state = np.stack([q1, q2, dq1, dq2], axis=1)
    error = state - upright

    rms_error  = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    rms_torque = float(np.sqrt(np.mean(u ** 2)))

    settling_time = _settling_time(t, error)

    return {
        "rms_error":         rms_error,
        "rms_torque":        rms_torque,
        "settling_time_s":   settling_time,
        "mean_solve_time_ms": float(np.mean(solve_time) * 1e3),
        "max_solve_time_ms":  float(np.max(solve_time)  * 1e3),
    }


def _settling_time(t: np.ndarray, error: np.ndarray, threshold: float = 0.05) -> float:
    norm = np.sqrt(np.sum(error ** 2, axis=1))
    settled = np.where(norm < threshold)[0]
    if len(settled) == 0:
        return float("nan")
    # first index after which the system stays settled
    for i in settled:
        if np.all(norm[i:] < threshold):
            return float(t[i])
    return float("nan")
