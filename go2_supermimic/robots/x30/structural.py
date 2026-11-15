"""Structural mjlab configuration for the DeepRobotics X30 candidate model.

This configuration intentionally has no actuators. The vendor actuator model
and control gains must be confirmed before it is used by a training task.
"""

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from go2_supermimic.robots.x30.model import load_x30_spec


def get_x30_spec() -> mujoco.MjSpec:
  """Load the X30 MJCF as an editable MuJoCo specification for mjlab."""
  return load_x30_spec()


def get_x30_structural_cfg() -> EntityCfg:
  """Return a loadable X30 entity without guessed actuator dynamics.

  This is for model compilation and index inspection only. It is deliberately
  not wired into the Go2 training environment until X30 actuator parameters
  have been confirmed.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.63),
      joint_pos={".*_Knee_joint": 0.349},
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_x30_spec,
    articulation=EntityArticulationInfoCfg(actuators=(), soft_joint_pos_limit_factor=0.9),
  )
