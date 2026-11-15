"""Explicit X30 simulation parameters with legacy and native position PD backends."""

from __future__ import annotations

import json
import math
from pathlib import Path

from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER, load_x30_spec

JOINT_FIELDS = ("kp", "kd", "effort_limit", "velocity_limit", "armature",
                "frictionloss", "default_q", "action_scale")


def load_training_spec(parameters: dict):
  """Apply the configured nominal mass policy to a fresh vendor model."""
  spec = load_x30_spec()
  if parameters.get("mass_scaling", "source") == "uniform":
    source_mass = float(spec.compile().body_mass.sum())
    scale = parameters["model_mass_kg"] / source_mass
    for body in spec.bodies:
      if body.mass > 0:
        body.mass *= scale
        body.inertia = [float(value) * scale for value in body.inertia]
  return spec


def validate_parameters(data: dict) -> dict:
  """Validate complete joint-side SI parameters, including model limit compatibility."""
  if not isinstance(data, dict):
    raise ValueError("X30 robot config must be a JSON object")
  if not isinstance(data.get("parameter_source"), str) or not data["parameter_source"].strip():
    raise ValueError("X30 parameter_source must identify the motor/model data source")
  if data.get("actuator_model", "ideal_pd") not in ("ideal_pd", "builtin_position"):
    raise ValueError("X30 actuator_model must be 'ideal_pd' or 'builtin_position'")
  if type(data.get("delay_physics_steps")) is not int or data["delay_physics_steps"] != 0:
    raise ValueError("Initial X30 world model supports delay_physics_steps=0 only; delayed control needs actuator history")
  for name in ("standing_height", "model_mass_kg"):
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
      raise ValueError(f"X30 {name} must be a positive finite number with an explicit source or assumption")
  joints = data.get("joints", {})
  if not isinstance(joints, dict) or set(joints) != set(SDK_JOINT_ORDER):
    raise ValueError("X30 joints must contain all 12 SDK names (FL, FR, HL, HR / HipX, HipY, Knee)")
  model = load_x30_spec().compile()
  mass = float(model.body_mass.sum())
  policy = data.get("mass_scaling", "source")
  if policy not in ("source", "uniform"):
    raise ValueError("X30 mass_scaling must be 'source' or 'uniform'")
  if policy == "source" and abs(data["model_mass_kg"] - mass) > 0.05:
    raise ValueError(f"Configured model_mass_kg differs from MJCF mass {mass:.5f}; reconcile model/payload first")
  import mujoco

  for name in SDK_JOINT_ORDER:
    row = joints[name]
    if not isinstance(row, dict):
      raise ValueError(f"X30 {name} parameters must be an object")
    for field in JOINT_FIELDS:
      value = row.get(field)
      if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"X30 {name}.{field} is missing or not finite")
      if field in ("kp", "effort_limit", "velocity_limit", "action_scale") and value <= 0:
        raise ValueError(f"X30 {name}.{field} must be positive")
      if field in ("kd", "armature", "frictionloss") and value < 0:
        raise ValueError(f"X30 {name}.{field} must be nonnegative")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    if not lo <= row["default_q"] - row["action_scale"] < row["default_q"] + row["action_scale"] <= hi:
      raise ValueError(f"{name}: default_q +/- action_scale must fit MJCF limits [{lo}, {hi}]")
  return data


def load_parameters(path: str) -> dict:
  if not path:
    raise ValueError("X30 requires --robot-config <JSON>; fill configs/x30_robot.template.json with your motor parameters")
  return validate_parameters(json.loads(Path(path).read_text(encoding="utf-8")))


def robot_cfg(parameters: dict):
  from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
  from mjlab.utils.spec_config import CollisionCfg

  p = validate_parameters(parameters)
  def training_spec():
    return load_training_spec(p)

  actuator_cfg = (BuiltinPositionActuatorCfg if p.get("actuator_model", "ideal_pd") == "builtin_position"
                  else IdealPdActuatorCfg)
  actuators = tuple(actuator_cfg(
    target_names_expr=(name,), stiffness=p["joints"][name]["kp"],
    damping=p["joints"][name]["kd"], effort_limit=p["joints"][name]["effort_limit"],
    armature=p["joints"][name]["armature"], frictionloss=p["joints"][name]["frictionloss"],
  ) for name in SDK_JOINT_ORDER)
  return EntityCfg(
    spec_fn=training_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, p["standing_height"]),
      joint_pos={name: p["joints"][name]["default_q"] for name in SDK_JOINT_ORDER},
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    collisions=(CollisionCfg(geom_names_expr=(".*_collision.*",), contype=1, conaffinity=1, priority=0,
                             friction=(1.0, 0.005, 0.0005), condim=3),),
  )
