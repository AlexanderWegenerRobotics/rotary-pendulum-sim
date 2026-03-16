from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


_LOG_DIR = Path("logs")


def save_episode(results: dict, metadata: dict, output_dir: Path | str = _LOG_DIR, run_id: str | None = None) -> tuple[Path, Path]:
    """Save episode results to .h5 and metadata to .json, return both paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{metadata.get('controller_name', 'unknown')}_{results.get('test_case_id', 'unknown')}_{ts}"

    h5_path   = out / f"{run_id}.h5"
    json_path = out / f"{run_id}.json"

    with h5py.File(h5_path, "w") as f:
        f.attrs["run_id"]       = run_id
        f.attrs["test_case_id"] = results["test_case_id"]
        f.attrs["success"]      = bool(results["success"])

        grp = f.create_group("data")
        for key in ("t", "x_true", "x_obs", "u_desired", "u_applied", "solve_time"):
            grp.create_dataset(key, data=results[key], compression="gzip")

    meta_out = {
        "run_id":       run_id,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "test_case_id": results["test_case_id"],
        "success":      bool(results["success"]),
        **metadata,
    }

    with open(json_path, "w") as f:
        json.dump(meta_out, f, indent=2, default=_serialise)

    return h5_path, json_path


def load_episode(path: Path | str) -> tuple[dict, dict]:
    """Load episode from .h5 + .json, return (results, metadata)."""
    p       = Path(path)
    h5_path = p.with_suffix(".h5")
    js_path = p.with_suffix(".json")

    with h5py.File(h5_path, "r") as f:
        grp     = f["data"]
        results = {key: grp[key][:] for key in grp}
        results["success"]      = bool(f.attrs["success"])
        results["test_case_id"] = str(f.attrs["test_case_id"])

    with open(js_path) as f:
        metadata = json.load(f)

    return results, metadata


def _serialise(obj: Any) -> Any:
    """JSON serialiser for numpy scalar and array types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")