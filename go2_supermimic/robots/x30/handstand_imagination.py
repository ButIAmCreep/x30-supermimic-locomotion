"""Short world-model rollouts for timed walking, front support and recovery."""

import torch

from go2_supermimic.robots.x30 import handstand as hs, locomotion as loc
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import Imagination


class HandstandImagination(Imagination):
    def __init__(self, cfg, constants, velocity_limits, weights, gait, foot_radii):
        self.cfg = cfg
        self.c = constants
        self.velocity_limits = velocity_limits
        self.weights = weights
        self.gait = gait
        self.foot_radii = tuple(foot_radii)
        self.critic_dim = st.STATE_DIM + 6 + st.CONTEXT_DIM
        self._step_fn = (torch.compile(self._step, dynamic=False) if cfg.compile else self._step)

    def _step(self, wm, actor, s, hist, previous, command, phase, age, xi, constants, valid):
        action = actor(hist)
        relief = torch.zeros(len(s), SCAN_POINTS, device=s.device)
        s1, pose_delta = wm(s, action, previous, xi, relief)
        age1 = age + self.cfg.dt
        phase1 = loc.advance_phase(phase, command, self.cfg.dt)
        failure = hs.failure_probability(s1)
        reward, _ = hs.rewards(s, s1, pose_delta, action, previous, command,
                               phase1, age1.squeeze(-1), constants, self.velocity_limits,
                               failure, self.weights, self.gait, self.foot_radii,
                               valid=valid, dt=self.cfg.dt)
        command1 = hs.command_for_age(age1.squeeze(-1))
        frame = hs.obs_frame(s1, action, command1, phase1, age1.squeeze(-1))
        hist1 = st.push_history(hist, frame, hs.HISTORY, hs.OBS_TERMS)
        return s1, hist1, action, command1, phase1, age1, reward * self.cfg.dt, failure

    def rollout(self, wm, actor, critic, seed, hint_weight=0.):
        ids = seed['env']
        c = st.RobotConsts(self.c.default_q[ids], self.c.action_scale[ids],
                           self.c.kp, self.c.kd, self.c.effort, self.c.soft_limits[ids])
        s, hist, previous = seed['s'], seed['obs'], seed['a_prev']
        command, phase, age = seed['gen'][:, :3], seed['gen'][:, 3:5], seed['gen'][:, 5:6]
        xi = seed['xi']
        valid = seed.get('reward_valid', torch.ones(len(s), dtype=torch.bool, device=s.device))
        survival = torch.ones(len(s), device=s.device)
        total = torch.zeros_like(survival)
        discount = 1.
        rewards, survivals, inputs = [], [], []
        for _ in range(self.cfg.horizon):
            inputs.append(torch.cat([s, command, phase, age, xi], -1))
            survivals.append(survival)
            s, hist, previous, command, phase, age, r, failure = self._step_fn(
                wm, actor, s, hist, previous, command, phase, age, xi, c, valid)
            valid = torch.ones_like(valid)
            total = total + discount * survival * r
            survival = survival * (1 - failure)
            rewards.append(r)
            discount *= self.cfg.gamma
        inputs.append(torch.cat([s, command, phase, age, xi], -1))
        survivals.append(survival)
        objective = total + discount * survival * critic(inputs[-1]).squeeze(-1)
        return objective, {'rewards': torch.stack(rewards, 1),
                           'surv': torch.stack(survivals, 1),
                           'critic_in': torch.stack(inputs, 1)}
