"""
Please do NOT change without discussion in the group! Otherwise we operate on different tests...
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class TestCaseConfig:

    id:                str
    description:       str
    x0:                Optional[tuple]  = None
    noise_std:         float            = 0.0
    tau_max:           float            = np.inf
    mass_scale_link1:  float            = 1.0
    mass_scale_link2:  float            = 1.0
    impulses:          tuple            = field(default_factory=tuple)
    duration:          float            = 10.0
    control_dt:        float            = 0.01


TEST_CASES: dict[str, TestCaseConfig] = {

    "nominal": TestCaseConfig(
        id="nominal",
        description="Perfect model, no noise, no constraints.",
    ),

    # noise
    "noise_low": TestCaseConfig(
        id="noise_low",
        description="Sensor noise sigma=0.01 rad.",
        noise_std=0.01,
    ),
    "noise_med": TestCaseConfig(
        id="noise_med",
        description="Sensor noise sigma=0.05 rad.",
        noise_std=0.05,
    ),
    "noise_high": TestCaseConfig(
        id="noise_high",
        description="Sensor noise sigma=0.10 rad.",
        noise_std=0.10,
    ),

    # torque limits
    "torque_5nm": TestCaseConfig(
        id="torque_5nm",
        description="Torque limited to ±5 Nm.",
        tau_max=5.0,
    ),
    "torque_2nm": TestCaseConfig(
        id="torque_2nm",
        description="Torque limited to ±2 Nm.",
        tau_max=2.0,
    ),
    "torque_1nm": TestCaseConfig(
        id="torque_1nm",
        description="Torque limited to ±1 Nm.",
        tau_max=1.0,
    ),

    # model mismatch
    "mismatch_p10": TestCaseConfig(
        id="mismatch_p10",
        description="Link masses +10%.",
        mass_scale_link1=1.10, mass_scale_link2=1.10,
    ),
    "mismatch_p20": TestCaseConfig(
        id="mismatch_p20",
        description="Link masses +20%.",
        mass_scale_link1=1.20, mass_scale_link2=1.20,
    ),
    "mismatch_p30": TestCaseConfig(
        id="mismatch_p30",
        description="Link masses +30%.",
        mass_scale_link1=1.30, mass_scale_link2=1.30,
    ),
    "mismatch_m10": TestCaseConfig(
        id="mismatch_m10",
        description="Link masses -10%.",
        mass_scale_link1=0.90, mass_scale_link2=0.90,
    ),
    "mismatch_m20": TestCaseConfig(
        id="mismatch_m20",
        description="Link masses -20%.",
        mass_scale_link1=0.80, mass_scale_link2=0.80,
    ),
    "mismatch_m30": TestCaseConfig(
        id="mismatch_m30",
        description="Link masses -30%.",
        mass_scale_link1=0.70, mass_scale_link2=0.70,
    ),

    # impulses
    "impulse_soft": TestCaseConfig(
        id="impulse_soft",
        description="Impulse at t=3s, 0.5 Nm·s.",
        impulses=((3.0, 0.5),),
    ),
    "impulse_med": TestCaseConfig(
        id="impulse_med",
        description="Impulse at t=3s, 1.0 Nm·s.",
        impulses=((3.0, 1.0),),
    ),
    "impulse_hard": TestCaseConfig(
        id="impulse_hard",
        description="Impulse at t=3s, 2.0 Nm·s.",
        impulses=((3.0, 2.0),),
    ),
    "impulse_repeated": TestCaseConfig(
        id="impulse_repeated",
        description="Three impulses at t=2,5,8s, 0.5 Nm·s each.",
        impulses=((2.0, 0.5), (5.0, 0.5), (8.0, 0.5)),
    ),

    # combined
    "stress_noise_torque": TestCaseConfig(
        id="stress_noise_torque",
        description="Medium noise + tight torque limit.",
        noise_std=0.05,
        tau_max=2.0,
    ),
    "stress_full": TestCaseConfig(
        id="stress_full",
        description="All perturbations combined.",
        noise_std=0.05,
        tau_max=2.0,
        mass_scale_link1=1.15,
        mass_scale_link2=1.15,
        impulses=((4.0, 0.5),),
    ),
}


def get_test_case(test_id: str) -> TestCaseConfig:
    """Return test case by ID, raise KeyError with helpful message if missing."""
    if test_id not in TEST_CASES:
        raise KeyError(f"Unknown test case '{test_id}'. Available: {sorted(TEST_CASES)}")
    return TEST_CASES[test_id]


def list_test_cases() -> None:
    """Print all registered test cases."""
    print(f"{'ID':<25} {'Description'}")
    print("-" * 65)
    for tc in TEST_CASES.values():
        print(f"{tc.id:<25} {tc.description}")