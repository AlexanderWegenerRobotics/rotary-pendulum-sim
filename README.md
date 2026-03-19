# Rotary Double Pendulum — Simulation & Controller Benchmark

Shared simulation infrastructure for ME/SE 701. Provides a MuJoCo-based environment for evaluating LQR, iLQR, and MPC on an underactuated rotary double pendulum. The environment is a black box — you implement a controller, pass it in, and the rest is handled automatically.

---

## Installation

**Requirements:** conda, Python 3.10

```bash
conda create -n dp-control python=3.10
conda activate dp-control
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel (for the analysis notebook):

```bash
python -m ipykernel install --user --name dp-control --display-name "dp-control"
```

---

## Project Structure

```
root/
├── main.py                  # entry point
├── config.yaml              # simulation and test configuration
├── test_cases.yaml          # scenario definitions (noise, limits, perturbations)
├── requirements.txt
├── model/
│   └── scene.xml            # MuJoCo model (include double_pendulum.xml)
├── analysis/
│   └── general_analysis.ipynb
├── controller/
│   ├── base_controller.py   # abstract base — implement this
│   └── spring_controller.py # example controller
└── src/                     # simulation internals — do not modify
    ├── double_pendulum.py
    ├── simulation.py
    ├── logger.py
    ├── metrics.py
    └── system_params.py
```

---

## Running a Simulation

1. Place your controller in `controller/`
2. Set the active scenario in `config.yaml` under `test.scenario`
3. Edit `main.py` to import and instantiate your controller
4. Run:

```bash
conda activate dp-control
python main.py
```

Logs are written to `log/` automatically. Each run produces a `.h5` trajectory file and a `_meta.json` sidecar.

Press `q` in the render window to stop early. Set `headless: true` in `config.yaml` to run without rendering (recommended on Raspberry Pi).

---

## Implementing a Controller

Subclass `BaseController` and implement `_compute()`. That is the only requirement.

```python
import numpy as np
from controller.base_controller import BaseController

class MyController(BaseController):
    def __init__(self, config: dict):
        super().__init__(config)
        # initialise your controller here

    def _compute(self, state: np.ndarray, t: float) -> float:
        # state = [q1, q2, dq1, dq2]
        # t     = simulation time in seconds
        # return a scalar torque in Nm
        return 0.0
```

**Do not override `compute()`** — the base class wraps `_compute()` automatically to measure solve time. Call `self._clip(torque, limit)` to saturate your output if needed.

### Getting System Parameters

To access physical model parameters (masses, link lengths, CoM positions, inertias) for linearisation or dynamics models, use:

```python
from src.system_params import get_system_params

params = get_system_params(config)
# params.m1, params.m2   — link masses [kg]
# params.l1, params.l2   — link lengths [m]
# params.lc1, params.lc2 — CoM distances from joint [m]
# params.I1, params.I2   — inertias about swing axis [kg·m²]
# params.g               — gravitational acceleration [m/s²]
```

Parameters are read directly from the MuJoCo XML. If the active scenario includes a `mass_perturbation`, it is applied automatically — the controller receives perturbed values while the simulator runs the true model.

---

## Configuration

`config.yaml` controls everything about a run:

```yaml
simulation:
  model_path: "model/scene.xml"   # path to MuJoCo model
  headless: false                 # true = no render window (use on Pi)
  render_width: 1280
  render_height: 720
  render_fps: 20
  physics_hz: 500                 # physics update frequency [Hz]
  control_hz: 500                 # control update frequency [Hz] (optional, defaults to physics_hz)
  log_path: "log/"
  video_log: true                 # save .mp4 alongside .h5

test:
  scenario: "nominal"             # must match a key in test_cases.yaml
  test_cases_path: "test_cases.yaml"
  duration: 10.0                  # simulation duration [s]
  initial_state: [0.0, 3.1, 0.0, 0.0]  # [q1, q2, dq1, dq2]
  impulse_times: [3.0]            # wall times at which impulse is applied [s]
```

---

## Test Scenarios

Scenarios are defined in `test_cases.yaml` and selected by name in `config.yaml`. They correspond directly to the evaluation levels in the project proposal (Table 1).

| Scenario | Level | Description |
|---|---|---|
| `nominal` | 1 | Perfect model, no noise, no torque limit |
| `noise_low` | 2 | Sensor noise σ = 0.01 rad |
| `noise_mid` | 2 | Sensor noise σ = 0.05 rad |
| `noise_high` | 2 | Sensor noise σ = 0.10 rad |
| `torque_limit_mid` | 3 | Torque capped at 2 Nm |
| `torque_limit_low` | 3 | Torque capped at 1 Nm |
| `mismatch_low` | 4 | Controller sees link masses +10% |
| `mismatch_mid` | 4 | Controller sees link masses +20% |
| `mismatch_high` | 4 | Controller sees link masses +30% |
| `impulse_low` | 5 | Impulse disturbance of 0.5 Nm·s at `impulse_times` |
| `impulse_mid` | 5 | Impulse disturbance of 1.0 Nm·s at `impulse_times` |
| `impulse_high` | 5 | Impulse disturbance of 2.0 Nm·s at `impulse_times` |
| `combined_stress` | 6 | noise_mid + torque_limit_mid + mismatch_mid combined |

**How perturbations work:**

- **Sensor noise** is added to the state written to shared memory each physics step. The controller sees noisy state; the logger records ground truth.
- **Torque limit** is enforced in the physics loop via `np.clip` before applying to the actuator.
- **Model mismatch** is applied inside `get_system_params()` — the controller is initialised with scaled masses while MuJoCo runs the true model.
- **Impulse** is a velocity kick applied to `dq1` at the specified simulation times. Set `impulse_times` in `config.yaml`.

---

## Viewing Results

Open `analysis/general_analysis.ipynb` in Jupyter:

```bash
conda activate dp-control
jupyter notebook analysis/general_analysis.ipynb
```

The notebook automatically discovers all `.h5` files in `log/`, lets you select a run, plots joint angles, velocities, and control torque, and computes the four benchmark metrics:

| Metric | Description |
|---|---|
| RMS state error | Root mean square of `‖x(t) − x*‖` over the run |
| RMS torque | Root mean square of control input |
| Settling time | First time the state error norm drops below 0.05 rad and stays there |
| Mean / max solve time | Controller `_compute()` wall time in milliseconds |

To compute metrics in your own script:

```python
from src.metrics import compute_metrics

metrics = compute_metrics("log/20260319_121053.h5")
```

---

## System Architecture

The environment runs two independent OS processes connected via shared memory.

**Sim process** contains two threads:
- Physics thread (background): steps MuJoCo at `physics_hz`, reads torque from shared memory, writes noisy state to shared memory, applies perturbations
- Render thread (main thread): reads state and renders at `render_fps` via OpenCV. Must run on the main thread for macOS compatibility

**Control process** runs a single loop at `control_hz`: reads state from shared memory, calls `_compute()`, writes torque and solve time to shared memory. It runs fully independently — if `_compute()` is slow it simply applies a stale torque for the next physics steps, exactly as a real embedded controller would.

Shared memory layout:
- `shared_state`: `[q1, q2, dq1, dq2, t]` — written by sim, read by control
- `shared_torque`: scalar — written by control, read by sim
- `shared_solve_time`: scalar — written by control, read by sim for logging

This decoupling means the physics clock never stalls waiting for the controller. It is particularly important for MPC where `_compute()` may take tens of milliseconds.
