"""Front-foot support skill targets and differentiable task reward."""

from dataclasses import dataclass, field
import torch
from go2_supermimic.robots.x30 import locomotion as loc, policy as vel
from go2_supermimic.supermimic import state as st


TASK = 'X30-Front-Handstand'
HISTORY = 5
OBS_TERMS = (('base_ang_vel', 3), ('projected_gravity', 3), ('joint_pos', 12),
             ('joint_vel', 12), ('actions', 12), ('command', 3), ('gait_phase', 2),
             ('skill', 3))
OBS_DIM = sum(dim for _, dim in OBS_TERMS) * HISTORY
robot_state = loc.robot_state


@dataclass
class HandstandRewards:
    walking: loc.LocomotionRewards = field(default_factory=loc.LocomotionRewards)
    pitch_track: float = 2.
    front_support: float = 3.
    rear_lift: float = 3.
    stillness: float = 1.
    front_penetration: float = -2.
    torso_low: float = -3.
    action_rate: float = -.02
    torque_norm: float = -.02
    limits: float = -1.
    overspeed: float = -.1
    terminated: float = -20.
    target_pitch_deg: float = 40.


def skill_state(age):
    enter = ((age - 2.) / 2.).clamp(0, 1)
    exit_ = ((8. - age) / 2.).clamp(0, 1)
    gate = enter * exit_
    hold = ((age >= 4.) & (age < 6.)).to(age.dtype)
    recovery = ((age - 6.) / 2.).clamp(0, 1) * (age < 8.).to(age.dtype)
    return torch.stack((gate, hold, recovery), -1)


def command_for_age(age):
    command = torch.zeros(len(age), 3, device=age.device, dtype=age.dtype)
    command[:, 0] = .3 * ((age < 2.) | (age >= 8.)).to(age.dtype)
    return command


def support_terms(s, foot_radii):
    feet = st.feet_world_z(s) - s.new_tensor(foot_radii)
    front = feet[:, :2]
    rear = feet[:, 2:]
    front_contact = torch.sigmoid((.06 - front) / .01).prod(-1)
    rear_lift = front_contact * torch.sigmoid((rear - .12) / .02).mean(-1)
    front_penetration = ((-front - .012).clamp(min=0) / .01).square().mean(-1)
    return {'front_contact': front_contact, 'rear_lift': rear_lift,
            'front_penetration': front_penetration,
            'rear_height_m': rear.mean(-1)}


def torso_bottom(s):
    gravity = s[:, st.GRAV]
    return s[:, 0] - .23 * gravity[:, 0].abs() - .13 * gravity[:, 1].abs() - .09 * gravity[:, 2].abs()


def failure_from_state(s):
    return (torso_bottom(s) < .045) | (s[:, 0] < .17) | (s[:, st.GRAV][:, 1].abs() > .6)


def failure_probability(s):
    bottom = torch.sigmoid((.045 - torso_bottom(s)) / .015)
    roll = torch.sigmoid((s[:, st.GRAV][:, 1].abs() - .6) / .05)
    low = torch.sigmoid((.17 - s[:, 0]) / .015)
    return 1 - (1 - bottom) * (1 - roll) * (1 - low)


def obs_frame(s, action, command, phase, age):
    return loc.obs_frame(s, action, command, phase) | {'skill': skill_state(age)}


def rewards(s0, s1, pose_delta, action, previous_action, command, phase, age,
            constants, velocity_limits, failure, weights, gait, foot_radii, valid=None, dt=.02):
    """Walking reward blends into front-foot support as the skill gate rises."""
    skill = skill_state(age)
    gate = skill[:, 0]
    walk, _ = loc.rewards(s0, s1, pose_delta, action, previous_action,
                          command, phase, constants, velocity_limits, failure,
                          weights.walking, gait, foot_radii, valid=valid, dt=dt)
    support = support_terms(s1, foot_radii)
    gravity = s1[:, st.GRAV]
    pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
    desired = -torch.deg2rad(s1.new_tensor(weights.target_pitch_deg)) * gate
    _, safety = vel.rewards(s1, torch.zeros_like(command), action, previous_action,
                            constants, velocity_limits, failure, weights.walking)
    terms = {
        'pitch_track': torch.exp(-((pitch - desired) / .22).square()),
        'front_support': support['front_contact'],
        'rear_lift': support['rear_lift'],
        'stillness': torch.exp(-(s1[:, st.V_YAW][:, :2].square().sum(-1) / .25**2 +
                                 s1[:, st.W_B].square().sum(-1) / .5**2)),
        'front_penetration': support['front_penetration'],
        'torso_low': ((.12 - torso_bottom(s1)).clamp(min=0) / .05).square(),
        'action_rate': safety['action_rate'],
        'torque_norm': safety['torque_norm'],
        'limits': safety['limits'],
        'overspeed': safety['overspeed'],
        'terminated': safety['terminated'],
    }
    skill_reward = sum(getattr(weights, name) * value for name, value in terms.items())
    return (1 - gate) * walk + gate * skill_reward, terms | {'walk': walk, 'gate': gate}


def reward_contributions(terms, weights):
    """Per-term contributions to the actually blended training reward."""
    gate = terms['gate']
    result = {'gate': gate.mean(), 'walk': ((1 - gate) * terms['walk']).mean()}
    for name, value in terms.items():
        if name not in ('gate', 'walk'):
            result[name] = (gate * getattr(weights, name) * value).mean()
    return result
