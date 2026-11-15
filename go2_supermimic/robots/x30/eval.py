"""Deterministic simulator evaluation of X30 velocity checkpoints."""

import math
import time
import torch

from go2_supermimic.robots.x30.env import TASK, x30_velocity_env_cfg
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.robot import load_parameters, validate_parameters
from go2_supermimic.supermimic import state as st
from go2_supermimic.robots.x30 import policy as vel
from go2_supermimic.robots.x30.train import VelocityActor, contract


def evaluate(args):
  from mjlab.envs import ManagerBasedRlEnv

  if args.mode or args.terrain:
    raise ValueError("X30-Velocity-Flat uses --command VX VY WZ; --mode/--terrain are for chase tasks")
  if not math.isfinite(args.seconds) or args.seconds < 0.02 or args.num_envs < 1:
    raise ValueError("Evaluation requires seconds>=0.02 and num_envs>=1")
  if not all(math.isfinite(v) for v in args.command):
    raise ValueError("Command values must be finite")
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  saved = ckpt.get("contract", {})
  if saved.get("task") != TASK:
    raise ValueError("An X30 velocity checkpoint is required; Go2 policies cannot be loaded")
  parameters = validate_parameters(saved["parameters"])
  if args.robot_config and load_parameters(args.robot_config) != parameters:
    raise ValueError("--robot-config differs from the checkpoint motor contract")
  if contract(parameters) != saved:
    raise ValueError("Checkpoint contract differs from current model/observation/action/reward definitions")
  cfg = x30_velocity_env_cfg(parameters, play=True)
  cfg.scene.num_envs, cfg.seed, cfg.auto_reset = args.num_envs, args.seed, False
  cmd = cfg.commands["twist"]
  cmd.ranges.lin_vel_x = (args.command[0], args.command[0])
  cmd.ranges.lin_vel_y = (args.command[1], args.command[1])
  cmd.ranges.ang_vel_z = (args.command[2], args.command[2])
  cmd.rel_standing_envs = 0.0
  cfg.viewer.width, cfg.viewer.height = args.width, args.height
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
  writer = None
  try:
    obs = env.reset()[0]["actor"]
    actor = VelocityActor(vel.OBS_DIM, tuple(ckpt["cfg"]["actor_hidden"]), 12).to(env.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    c = st.robot_consts(env)
    limits = torch.tensor([parameters["joints"][n]["velocity_limit"] for n in SDK_JOINT_ORDER], device=env.device)
    command = torch.tensor(args.command, device=env.device).expand(args.num_envs, 3)
    if args.video:
      import mediapy
      writer = mediapy.VideoWriter(args.video, shape=(args.height, args.width), fps=round(1/env.step_dt))
      writer.__enter__()
    steps = int(args.seconds / env.step_dt)
    sums = torch.zeros(6, device=env.device)
    start = last = time.monotonic()
    with torch.inference_mode():
      for i in range(steps):
        a = actor(obs)
        out, _, died, timed_out, _ = env.step(a)
        s = vel.robot_state(env)
        q = s[:, st.Q] + c.default_q
        tau = c.kp * (c.action_scale * a - s[:, st.Q]) - c.kd * s[:, st.QD]
        sums += torch.stack([
          (s[:, st.V_YAW][:, :2]-command[:, :2]).square().sum(),
          (s[:, st.W_B][:, 2]-command[:, 2]).square().sum(),
          s[:, st.V_YAW][:, :2].norm(dim=-1).sum(), died.float().sum(),
          (tau.abs() >= c.effort).float().mean(-1).sum(),
          ((q < c.soft_limits[..., 0]) | (q > c.soft_limits[..., 1]) | (s[:, st.QD].abs() > limits)).float().mean(-1).sum(),
        ])
        if writer:
          writer.add_image(env.render())
        done = died | timed_out
        obs = env.reset(env_ids=done.nonzero().flatten())[0]["actor"] if done.any() else out["actor"]
        now = time.monotonic()
        if now-last >= 1 or i+1 == steps:
          elapsed = now-start
          print(f"\rX30 evaluation: {i+1}/{steps} steps ({100*(i+1)/steps:.1f}%) | elapsed {elapsed:.0f}s | ETA {elapsed*(steps-i-1)/(i+1):.0f}s", end="\n" if i+1==steps else "", flush=True)
          last = now
    count = steps*args.num_envs
    linear, angular, speed, falls, saturation, limits = sums.cpu().tolist()
    print(f"{TASK} command={tuple(args.command)} seed={args.seed} duration={steps*env.step_dt:.2f}s")
    print(f"  planar velocity RMSE: {math.sqrt(linear/count):.4f} m/s")
    print(f"  body wz RMSE: {math.sqrt(angular/count):.4f} rad/s")
    print(f"  mean planar speed (zero-command drift when command=0): {speed/count:.4f} m/s")
    print(f"  falls per env-minute: {falls/(count*env.step_dt/60):.4f}")
    print(f"  PD torque saturation estimate: {saturation/count:.4f}")
    print(f"  joint soft-limit or speed-limit fraction: {limits/count:.4f}")
  finally:
    if writer:
      writer.__exit__(None, None, None)
    env.close()
  if args.video:
    print(f"  video: {args.video}")
