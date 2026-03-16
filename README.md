# rotary-pendulum-sim

Comparative evaluation of LQR, iLQR, and MPC on an underactuated rotary double pendulum, simulated in MuJoCo. Part of ME/SE 701 — Optimal and Robust Control, Boston University SP2026.

The package provides a shared simulation environment and standardised test suite so all three controllers can be evaluated under identical conditions and compared fairly.

---

## Structure

```
dp-benchmark/
├── dpendulum/
│   ├── sim.py          # core simulation environment
│   ├── test_cases.py   # evaluation scenario registry
│   ├── metrics.py      # shared performance metrics
│   ├── logging.py      # h5py data + JSON metadata (output → logs/)
│   └── assets/
│       └── double_pendulum.xml   # MuJoCo model
├── examples/
│   └── lqr_example.py  # minimal controller integration example
└── setup.py
```

---

## Installation

Requires **Python 3.10+**. Tested on macOS, Linux, and Windows.

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/dp-benchmark.git
cd dp-benchmark
```

**2. Create a virtual environment** (recommended)

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install**
```bash
pip install -e .
```

---

## Quickstart

```python
from dpendulum import DoublePendulumSim, compute_metrics, print_metrics

# any callable with signature: tau = controller(x_obs, t) -> float
def my_controller(x_obs, t):
    return 0.0

sim     = DoublePendulumSim(test_case="nominal")
results = sim.run_episode(my_controller)

print_metrics(compute_metrics(results))
```

The controller receives the **observed state** `x_obs = [θ₁, θ₂, θ̇₁, θ̇₂]` and current time, and returns a scalar torque. Noise, saturation, and model mismatch are applied by the sim — the controller does not need to handle them.

---

## Test cases

```python
from dpendulum import list_test_cases
list_test_cases()
```

| Level | IDs | What varies |
|-------|-----|-------------|
| 1 | `nominal` | Baseline — perfect model, no noise |
| 2 | `noise_low/med/high` | Sensor noise σ ∈ {0.01, 0.05, 0.10} rad |
| 3 | `torque_5nm/2nm/1nm` | Torque saturation ±{5, 2, 1} Nm |
| 4 | `mismatch_p/m10/20/30` | Link mass ±{10, 20, 30}% vs nominal |
| 5 | `impulse_soft/med/hard` | Impulse disturbance at t=3s |
| — | `stress_full` | All perturbations combined |

---

## Logging

Episodes are saved to `logs/` (git-ignored). Each episode produces two files:

```
logs/
├── <run_id>.h5     # states, controls, solve times
└── <run_id>.json   # sim version, test case, controller params, timestamp
```

```python
from dpendulum.logging import save_episode, load_episode

save_episode(results, metadata={"controller_name": "LQR", "controller_params": {...}})
results, meta = load_episode("logs/<run_id>")
```

---

## Rendering

Disabled by default. Enable for debugging — not recommended during batch evaluation.

```python
sim = DoublePendulumSim(test_case="nominal", render=True)
```

The renderer runs in a separate thread at 60 Hz and does not affect control timing.

---

## Notes

- Controllers should use `sim.get_nominal_model_params()` for their internal models — the sim may apply mass/length perturbations that the controller does not see.
- `test_cases.py` is the single source of truth for evaluation conditions. Do not modify existing entries.
- All logged data uses true state (`x_true`) and observed state (`x_obs`) separately, as well as requested (`u_desired`) and applied (`u_applied`) torque, to allow post-hoc diagnosis of failures.