"""SuperMimic trainer for X30 velocity tracking, independent of ball state."""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from go2_supermimic.robots.x30.env import TASK, register_x30_task
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER, X30_XML, X30_ASSET_DIR
from go2_supermimic.robots.x30.robot import load_parameters
from go2_supermimic.supermimic import state as st
from go2_supermimic.robots.x30 import policy as vel
from go2_supermimic.supermimic.buffer import ReplayBuffer, WindowArchive
from go2_supermimic.supermimic.imagine import ImagineCfg
from go2_supermimic.supermimic.train import Actor, Critic, TrainCfg, Trainer
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS


class VelocityActor(Actor):
  def forward(self, obs):
    return super().forward(obs).tanh()


def model_hash():
  h = hashlib.sha256(X30_XML.read_bytes())
  for p in sorted(X30_ASSET_DIR.glob("*.STL")):
    h.update(p.name.encode())
    h.update(p.read_bytes())
  return h.hexdigest()


def contract(parameters):
  return {"version": 1, "task": TASK, "parameters": parameters, "model_sha256": model_hash(),
          "joint_order": list(SDK_JOINT_ORDER), "obs_terms": [list(x) for x in vel.OBS_TERMS],
          "history": vel.HISTORY, "action": "tanh actor; exploration clamped [-1,1]; q=default+scale*a",
          "dt": 0.02, "commands": "yaw-frame vx,vy; body-frame wz; fixed within imagination",
          "randomization": "none; nominal friction=1, COM offset=0, zero observation noise, zero delay",
          "reward_weights": asdict(vel.VelocityRewards())}


def defaults():
  return TrainCfg(task=TASK, num_envs=64, buffer_len=200, wm_hidden=(256, 256),
                  wm_batch=128, imag_batch=256, horizon=8, archive_size=5000,
                  iterations=500, warmup_iterations=100, compile=False,
                  save_interval=100, run_name="x30_velocity")


class VelocityTrainer(Trainer):
  task_id = TASK
  policy_module = vel
  task_contract = staticmethod(contract)
  register_task = staticmethod(register_x30_task)
  task_state_dim = 3

  def __init__(self, cfg):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    if cfg.driver_policy or cfg.expert_policy:
      raise ValueError("Go2 data drivers cannot be used by the X30 velocity task")
    if cfg.buffer_len <= cfg.wm_window + 2 or cfg.archive_size < 1:
      raise ValueError("buffer_len must exceed wm_window+2 and archive_size must be positive")
    if cfg.imag_batch * cfg.horizon < cfg.critic_minibatches:
      raise ValueError("imag_batch*horizon must be >= critic_minibatches")
    if min(cfg.collect_steps, cfg.wm_updates, cfg.wm_batch, cfg.num_envs) < 1:
      raise ValueError("collection and world-model batch settings must be positive")
    if not 0 <= cfg.wm_prioritized_frac + cfg.wm_archive_frac < 1:
      raise ValueError("prioritized+archive fractions must be in [0,1)")
    self.cfg = cfg
    policy = self.policy_module
    parameters = load_parameters(cfg.robot_config)
    self.contract = self.task_contract(parameters)
    self.register_task(parameters)
    torch.manual_seed(cfg.seed)
    self.env_cfg = load_env_cfg(self.task_id)
    self.env_cfg.scene.num_envs, self.env_cfg.seed = cfg.num_envs, cfg.seed
    self.env = ManagerBasedRlEnv(cfg=self.env_cfg, device="cuda:0")
    self.dev, self.history, self.field = self.env.device, policy.HISTORY, None
    self.terms = policy.OBS_TERMS
    self.term = self.env.command_manager.get_term("twist")
    self.obs = self.env.reset()[0]["actor"]
    if self.obs.shape[-1] != policy.OBS_DIM:
      raise ValueError(f"X30 actor requires {policy.OBS_DIM} inputs, got {self.obs.shape[-1]}")
    self.consts = st.robot_consts(self.env)
    self.xi = torch.zeros(cfg.num_envs, st.CONTEXT_DIM, device=self.dev)
    self.xi[:, 0] = 1.0
    self.enc_bias = torch.zeros(cfg.num_envs, 12, device=self.dev)
    self.wm = self._make_world_model(parameters).to(self.dev)
    self.actor = self._make_actor(parameters).to(self.dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
    self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    self.buf = ReplayBuffer(cfg.num_envs, cfg.buffer_len, policy.OBS_DIM, self.dev, gen_dim=self.task_state_dim)
    self.archive = WindowArchive(cfg.archive_size, cfg.wm_window, st.CONTEXT_DIM, self.dev)
    self.a_prev = torch.zeros(cfg.num_envs, 12, device=self.dev)
    self.ou = torch.zeros_like(self.a_prev)
    self.explore = torch.arange(cfg.num_envs, device=self.dev) < int(cfg.explore_frac * cfg.num_envs)
    limits = torch.tensor([parameters["joints"][n]["velocity_limit"] for n in SDK_JOINT_ORDER], device=self.dev)
    self.imag = self._make_imagination(parameters, limits)
    self.critic = Critic(self.imag.critic_dim, cfg.critic_hidden).to(self.dev)
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    self.log_dir = Path(cfg.log_root) / f"{stamp}_{cfg.run_name}"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(str(self.log_dir))
    metadata = {"training": asdict(cfg), "contract": self.contract, "environment": asdict(self.env_cfg)}
    from go2_supermimic.paths import PROJECT_ROOT
    root = PROJECT_ROOT
    metadata["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.rglob("*.py")) if not {"__pycache__", ".backups", ".venv"}.intersection(p.parts)}
    (self.log_dir / "run_metadata.json").write_text(
      json.dumps(metadata, indent=2, default=lambda obj: f"{obj.__module__}.{obj.__qualname__}" if callable(obj) else str(obj)),
      encoding="utf-8")
    self.ep_stats = defaultdict(list)
    self.env_steps = self.actor_frozen_until = self.ramp_start = self.nonfinite_steps = self.start_it = 0
    self.policy_start = cfg.warmup_iterations
    self.lr_scale, self.last_ckpt, self.collapse_strikes = 1.0, None, 0
    if cfg.resume:
      self.load(cfg.resume)

  def _make_imagination(self, parameters, limits):
    return vel.VelocityImagination(
      ImagineCfg(horizon=self.cfg.horizon, dt=self.env.step_dt, compile=self.cfg.compile), self.consts,
      limits, parameters["standing_height"] * 0.45, self.env_cfg.rewards["velocity"].params["weights"])

  def _make_world_model(self, parameters):
    return WorldModel(self.cfg.wm_hidden)

  def _make_actor(self, parameters):
    return VelocityActor(self.policy_module.OBS_DIM, self.cfg.actor_hidden, 12)

  def _update_actor_norm(self):
    self.actor.norm.update(self.obs)

  def _task_state(self, command):
    return command

  def _record_task_stats(self):
    pass

  def load(self, path):
    ckpt = torch.load(path, map_location=self.dev, weights_only=False)
    if ckpt.get("contract") != self.contract:
      raise ValueError(f"{self.task_id} resume requires an identical task, hardware, observation, action and model contract; Go2 weights cannot resume this task")
    super().load(path)

  @torch.no_grad()
  def collect(self):
    cfg, env = self.cfg, self.env
    for _ in range(cfg.collect_steps):
      s, pose = self.policy_module.robot_state(env), st.planar_pose(env)
      command = self.term.command.clone()
      task_state = self._task_state(command)
      a = self.actor(self.obs)
      dt = env.step_dt
      self.ou = self.ou * (1 - cfg.explore_theta * dt) + cfg.explore_noise * (2 * cfg.explore_theta * dt)**0.5 * torch.randn_like(a)
      noise = torch.where(self.explore[:, None], self.ou, cfg.action_noise * torch.randn_like(a))
      a = (a + noise).clamp(-1, 1)
      obs, _, terminated, timeout, extras = env.step(a)
      done = terminated | timeout
      relief = self._relief(pose)
      self.buf.add(s, pose, a, self.a_prev, self.obs, task_state, done, relief)
      keep = ~done
      if keep.any():
        s1 = self.policy_module.robot_state(env)
        delta = st.pose_delta(pose, st.planar_pose(env))
        self.wm.update_stats(s[keep], a[keep], self.a_prev[keep], self.xi[keep], relief[keep], s1[keep], delta[keep])
        err = s1[keep, st.V_YAW][:, :2] - command[keep, :2]
        self.ep_stats["Metrics/velocity/linear_mse"].append(float(err.square().sum(-1).mean()))
        self.ep_stats["Metrics/velocity/angular_mse"].append(float((s1[keep, st.W_B][:, 2] - command[keep, 2]).square().mean()))
      self.ep_stats["Metrics/velocity/fall_fraction"].append(float(terminated.float().mean()))
      self._update_actor_norm()
      self.a_prev = torch.where(done[:, None], 0.0, a)
      self.ou = torch.where(done[:, None], 0.0, self.ou)
      self.obs = obs["actor"]
      self.env_steps += env.num_envs
      self._record_task_stats()
      for k, v in extras.get("log", {}).items():
        if k.startswith(("Episode_Reward", "Episode_Termination")):
          self.ep_stats[k].append(float(v))

  def _relief(self, pose):
    return torch.zeros(len(pose), SCAN_POINTS, device=self.dev)

  @staticmethod
  def _rarity(w):
    s = w["s"]
    contacts = (st.feet_world_z(s.reshape(-1, st.STATE_DIM)) < 0.05).reshape(*s.shape[:2], 4).float()
    return (contacts[:, 1:] - contacts[:, :-1]).abs().sum((1, 2)) + 4 * (1 - w["valid"][:, -1])

  @staticmethod
  def _archive_worthy(w):
    return VelocityTrainer._rarity(w) > 0

  def save(self, it):
    path = super().save(it)
    ckpt = torch.load(path, map_location=self.dev, weights_only=False)
    ckpt["contract"] = self.contract
    torch.save(ckpt, path)
    return path
