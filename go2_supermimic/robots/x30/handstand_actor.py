"""Frozen walking branch plus a separately trained front-support action head."""

import torch
from torch import nn

from go2_supermimic.robots.x30 import handstand as hs, locomotion as loc
from go2_supermimic.robots.x30.locomotion_transfer import action_gain
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.supermimic.world_model import RunningNorm, mlp


class HandstandActor(nn.Module):
    def __init__(self, hidden, source_parameters, parameters, reference_action):
        super().__init__()
        old = torch.tensor([source_parameters['joints'][n]['action_scale'] for n in SDK_JOINT_ORDER])
        new = torch.tensor([parameters['joints'][n]['action_scale'] for n in SDK_JOINT_ORDER])
        reference = torch.as_tensor(reference_action, dtype=torch.float32)
        if reference.shape != (12,) or not bool(torch.isfinite(reference).all()) or not bool((reference.abs() <= 1).all()):
            raise ValueError('Reference handstand action must be 12 finite normalized joint targets')
        self.walk = loc.LocomotionActor(loc.OBS_DIM, hidden, 12, action_gain(source_parameters))
        for parameter in self.walk.parameters():
            parameter.requires_grad_(False)
        self.register_buffer('walk_to_new', old / new)
        self.register_buffer('history_to_old', (new / old).repeat(hs.HISTORY))
        self.head_norm = RunningNorm(hs.OBS_DIM)
        self.head = mlp(hs.OBS_DIM, (256, 128), 12, act=nn.ELU)
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.atanh(reference.clamp(-.98, .98)))

    def walk_observation(self, observation):
        old = observation[:, :loc.OBS_DIM].clone()
        # Action histories are normalized against different position-offset scales.
        old[:, 150:210] *= self.history_to_old
        return old

    def update_norm(self, observation):
        self.head_norm.update(observation)

    def forward(self, observation):
        if observation.shape[-1] != hs.OBS_DIM:
            raise ValueError(f'Handstand actor requires {hs.OBS_DIM} observations')
        walk = self.walk(self.walk_observation(observation)) * self.walk_to_new
        front = self.head(self.head_norm(observation)).tanh()
        gate = observation[:, -3].clamp(0, 1)[:, None]
        return ((1 - gate) * walk + gate * front).clamp(-1, 1)
