import mujoco
import numpy as np
import yaml
from dataclasses import dataclass


@dataclass
class SystemParams:
    m1: float   # link1 mass [kg]
    m2: float   # link2 mass [kg]
    l1: float   # link1 length [m]  (joint1 -> joint2)
    l2: float   # link2 length [m]  (joint2 -> tip)
    lc1: float  # link1 CoM distance from joint1 [m]
    lc2: float  # link2 CoM distance from joint2 [m]
    I1: float   # link1 inertia about swing axis [kg·m²]
    I2: float   # link2 inertia about swing axis [kg·m²]
    g: float    # gravitational acceleration [m/s²]


def _load_scenario(config: dict) -> dict:
    path = config["test"]["test_cases_path"]
    name = config["test"]["scenario"]
    with open(path) as f:
        all_scenarios = yaml.safe_load(f)
    return all_scenarios["scenarios"][name]


def get_system_params(config: dict) -> SystemParams:
    model = mujoco.MjModel.from_xml_path(config["simulation"]["model_path"])
    scenario = _load_scenario(config)

    g = abs(model.opt.gravity[2])

    link1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link1")
    link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")

    m1 = float(model.body_mass[link1_id])
    m2 = float(model.body_mass[link2_id])

    l1 = float(np.linalg.norm(model.body_pos[link2_id]))

    link2_rod_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link2_rod")
    geom_midpoint_x = model.geom_pos[link2_rod_id][0]
    half_length     = model.geom_size[link2_rod_id][1]
    l2 = float(geom_midpoint_x + half_length)

    lc1 = float(np.linalg.norm(model.body_ipos[link1_id]))
    lc2 = float(np.linalg.norm(model.body_ipos[link2_id]))

    I1 = float(model.body_inertia[link1_id][1])
    I2 = float(model.body_inertia[link2_id][1])

    alpha = scenario.get("mass_perturbation", 0.0)
    if alpha != 0.0:
        m1 *= (1.0 + alpha)
        m2 *= (1.0 + alpha)

    return SystemParams(m1=m1, m2=m2, l1=l1, l2=l2, lc1=lc1, lc2=lc2, I1=I1, I2=I2, g=g)
