"""Nominal X30 flat-ground velocity task. Motor configuration is explicitly supplied."""

import math
import torch

from mjlab.managers import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from go2_supermimic.robots.x30.robot import robot_cfg
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.supermimic import state as st
from go2_supermimic.robots.x30 import policy as vel

TASK = "X30-Velocity-Flat"


def fell_over(env, min_height):
  d = env.scene["robot"].data
  return (-d.projected_gravity_b[:, 2] < math.cos(math.radians(70))) | (d.root_link_pos_w[:, 2] < min_height)


def velocity_reward(env, velocity_limits, min_height, weights):
  if not hasattr(env, "_x30_reward_consts"):
    env._x30_reward_consts = st.robot_consts(env)
  return vel.rewards(
    vel.robot_state(env), env.command_manager.get_term("twist").command,
    env.action_manager.action, env.action_manager.prev_action, env._x30_reward_consts,
    torch.as_tensor(velocity_limits, device=env.device), fell_over(env, min_height), weights,
  )[0]


def x30_velocity_env_cfg(parameters, play=False):
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": robot_cfg(parameters)}
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = ()
  cfg.events = {k: cfg.events[k] for k in ("reset_base", "reset_robot_joints")}
  cfg.events["reset_base"].params["pose_range"] = {"yaw": (-math.pi, math.pi)}
  cfg.curriculum, cfg.metrics = {}, {}
  cfg.commands = {"twist": UniformVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(3.0, 6.0), rel_standing_envs=0.2,
    rel_heading_envs=0.0, rel_forward_envs=0.0, heading_command=False, debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-0.1, 0.3), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)),
  )}
  cfg.actions["joint_pos"].scale = {n: parameters["joints"][n]["action_scale"] for n in SDK_JOINT_ORDER}
  terms = {
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
  }
  cfg.observations = {"actor": ObservationGroupCfg(terms=terms, concatenate_terms=True,
                                                  enable_corruption=False, history_length=vel.HISTORY)}
  min_height = parameters["standing_height"] * 0.45
  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "fell_over": TerminationTermCfg(func=fell_over, params={"min_height": min_height}),
  }
  cfg.rewards = {"velocity": RewardTermCfg(func=velocity_reward, weight=1.0, params={
    "velocity_limits": [parameters["joints"][n]["velocity_limit"] for n in SDK_JOINT_ORDER],
    "min_height": min_height, "weights": vel.VelocityRewards(),
  })}
  cfg.episode_length_s = 1e9 if play else 20.0
  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 2.5
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  return cfg


def register_x30_task(parameters):
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from mjlab.tasks.registry import register_mjlab_task, list_tasks

  if TASK in list_tasks():
    raise ValueError("X30 task is already configured in this process")
  register_mjlab_task(TASK, x30_velocity_env_cfg(parameters), x30_velocity_env_cfg(parameters, play=True),
                     RslRlOnPolicyRunnerCfg())
