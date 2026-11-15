"""X30 velocity-task observation contract, rewards, and differentiable imagination."""

from dataclasses import dataclass
import math

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import Imagination
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS

OBS_TERMS = st.OBS_TERMS[:-1] + (("command", 3),)
HISTORY = 5
OBS_DIM = sum(d for _, d in OBS_TERMS) * HISTORY
LIMB_NAMES = (tuple(f"{leg}_THIGH" for leg in ("FL", "FR", "HL", "HR")),
              tuple(f"{leg}_SHANK" for leg in ("FL", "FR", "HL", "HR")),
              ("FL", "FR", "HL", "HR"))


def robot_state(env):
  return st.robot_state(env, limb_names=LIMB_NAMES)


def obs_frame(s, action, command):
  return {"base_ang_vel": s[:, st.W_B], "projected_gravity": s[:, st.GRAV],
          "joint_pos": s[:, st.Q], "joint_vel": s[:, st.QD],
          "actions": action, "command": command}


@dataclass
class VelocityRewards:
  linear: float = 1.5
  angular: float = 1.0
  tilt: float = -0.5
  action_rate: float = -0.01
  torque_norm: float = -0.02
  limits: float = -1.0
  overspeed: float = -0.1
  terminated: float = -20.0
  linear_sigma: float = 0.25
  angular_sigma: float = 0.3


def rewards(s, command, action, previous_action, c, velocity_limits, failure, weights):
  """Shared SI-unit rewards. Torque is the same saturated zero-velocity PD proxy on both sides."""
  q = s[:, st.Q] + c.default_q
  torque = (c.kp * (c.action_scale * action - s[:, st.Q]) - c.kd * s[:, st.QD]).clamp(-c.effort, c.effort)
  terms = {
    "linear": torch.exp(-(s[:, st.V_YAW][:, :2] - command[:, :2]).square().sum(-1) / weights.linear_sigma**2),
    "angular": torch.exp(-(s[:, st.W_B][:, 2] - command[:, 2]).square() / weights.angular_sigma**2),
    "tilt": s[:, st.GRAV][:, :2].square().sum(-1),
    "action_rate": (action - previous_action).square().sum(-1),
    "torque_norm": (torque / c.effort).square().sum(-1),
    "limits": ((c.soft_limits[..., 0] - q).clamp(min=0) + (q - c.soft_limits[..., 1]).clamp(min=0)).sum(-1),
    "overspeed": (s[:, st.QD].abs() / velocity_limits - 1).clamp(min=0).square().sum(-1),
    "terminated": failure.float(),
  }
  return sum(getattr(weights, k) * v for k, v in terms.items()), terms


def failure_probability(s, min_height):
  tilt_failure = torch.sigmoid((math.cos(math.radians(70)) + s[:, 3]) / 0.05)
  height_failure = torch.sigmoid((min_height - s[:, 0]) / 0.02)
  return 1 - (1 - tilt_failure) * (1 - height_failure)


class VelocityImagination(Imagination):
  """Reuse TD(lambda); command is held constant across each short imagination horizon."""

  def __init__(self, cfg, consts, velocity_limits, min_height, weights):
    self.cfg, self.c = cfg, consts
    self.velocity_limits, self.min_height, self.weights = velocity_limits, min_height, weights
    self.critic_dim = st.STATE_DIM + 3 + st.CONTEXT_DIM
    self._step_fn = torch.compile(self._step_velocity, dynamic=False) if cfg.compile else self._step_velocity

  def _step_velocity(self, wm, actor, s, hist, previous, command, xi, c):
    a = actor(hist)
    relief = torch.zeros(len(s), SCAN_POINTS, device=s.device)
    s1, _ = wm(s, a, previous, xi, relief)
    failure = failure_probability(s1, self.min_height)
    reward, _ = rewards(s1, command, a, previous, c, self.velocity_limits, failure, self.weights)
    hist1 = st.push_history(hist, obs_frame(s1, a, command), HISTORY, OBS_TERMS)
    return s1, hist1, a, reward * self.cfg.dt, failure

  def rollout(self, wm, actor, critic, seed, hint_weight=0.0):
    cfg = self.cfg
    ids = seed["env"]
    c = st.RobotConsts(self.c.default_q[ids], self.c.action_scale[ids], self.c.kp,
                       self.c.kd, self.c.effort, self.c.soft_limits[ids])
    s, hist, previous = seed["s"], seed["obs"], seed["a_prev"]
    command, xi = seed["gen"], seed["xi"]
    survival, total = torch.ones(len(s), device=s.device), torch.zeros(len(s), device=s.device)
    rs, ss, xs = [], [], []
    discount = 1.0
    for _ in range(cfg.horizon):
      xs.append(torch.cat([s, command, xi], -1))
      ss.append(survival)
      s, hist, previous, r, failure = self._step_fn(wm, actor, s, hist, previous, command, xi, c)
      total = total + discount * survival * r
      survival = survival * (1 - failure)
      rs.append(r)
      discount *= cfg.gamma
    xs.append(torch.cat([s, command, xi], -1))
    ss.append(survival)
    objective = total + discount * survival * critic(xs[-1]).squeeze(-1)
    return objective, {"rewards": torch.stack(rs, 1), "surv": torch.stack(ss, 1),
                       "critic_in": torch.stack(xs, 1)}
