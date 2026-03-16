"""
Performance metrics computed from episode results.
"""

from __future__ import annotations
import numpy as np

_UPRIGHT     = np.array([0.0, np.pi / 2, 0.0, 0.0])
_SETTLE_THR  = 0.05   # [rad] state norm threshold for settling
_SETTLE_WIN  = 0.5    # [s]   must stay below threshold for this long


def compute_metrics(results: dict) -> dict:
    """Compute all metrics for one episode and return as dict."""
    t          = results["t"]
    x_true     = results["x_true"]
    u_desired  = results["u_desired"]
    u_applied  = results["u_applied"]
    solve_time = results["solve_time"]
    success    = results["success"]

    dt = t[1] - t[0] if len(t) > 1 else 0.01

    e           = x_true - _UPRIGHT
    e[:, :2]    = _wrap(e[:, :2])
    state_norms = np.linalg.norm(e, axis=1)

    settling_time   = _compute_settling(t, state_norms, dt)
    valid_solve     = solve_time[solve_time > 0]
    saturated       = np.abs(u_desired) > np.abs(u_applied) + 1e-6

    return {
        "rms_state_error":    float(np.sqrt(np.mean(state_norms ** 2))),
        "settling_time":      settling_time,
        "rms_torque":         float(np.sqrt(np.mean(u_applied ** 2))),
        "max_torque":         float(np.max(np.abs(u_applied))),
        "saturation_frac":    float(np.mean(saturated)),
        "mean_solve_time_ms": float(np.mean(valid_solve) * 1e3) if len(valid_solve) else 0.0,
        "max_solve_time_ms":  float(np.max(valid_solve)  * 1e3) if len(valid_solve) else 0.0,
        "success":            success,
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print a metrics dict."""
    print(f"── {label or 'Metrics'} " + "─" * 40)
    print(f"  Success          : {metrics['success']}")
    print(f"  RMS state error  : {metrics['rms_state_error']:.4f}")
    print(f"  Settling time    : {metrics['settling_time']:.2f} s")
    print(f"  RMS torque       : {metrics['rms_torque']:.3f} Nm")
    print(f"  Max torque       : {metrics['max_torque']:.3f} Nm")
    print(f"  Saturation frac  : {metrics['saturation_frac']:.1%}")
    print(f"  Mean solve time  : {metrics['mean_solve_time_ms']:.3f} ms")
    print(f"  Max solve time   : {metrics['max_solve_time_ms']:.3f} ms")


def _compute_settling(t, state_norms, dt) -> float:
    """Return first time state stays below threshold for SETTLE_WIN seconds."""
    win = max(1, int(_SETTLE_WIN / dt))
    below = state_norms < _SETTLE_THR
    for i in range(len(t) - win):
        if np.all(below[i: i + win]):
            return float(t[i])
    return np.inf


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi