"""Ten-second walk, front-support, recovery task in nominal flat MuJoCo."""

from dataclasses import dataclass

import torch
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

from go2_supermimic.robots.x30 import handstand as hs, trot
from go2_supermimic.robots.x30.curriculum_command import CurriculumCommand, CurriculumCommandCfg
from go2_supermimic.robots.x30.locomotion_env import locomotion_env_cfg, phase_observation
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.trot_env import model_foot_radii
from go2_supermimic.robots.x30.handstand_feasibility import find_pose
from go2_supermimic.supermimic import state as st


class HandstandCommand(CurriculumCommand):
    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.
        self.time_left[env_ids] = 1e9

    def _update_command(self, env_ids=None):
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids)
        age = effective_age(self._env)[ids]
        self.vel_command_b[ids] = hs.command_for_age(age)


@dataclass(kw_only=True)
class HandstandCommandCfg(CurriculumCommandCfg):
    def build(self, env):
        return HandstandCommand(self, env)


def effective_age(env):
    age = env.episode_length_buf * env.step_dt
    return age + getattr(env, '_handstand_age_offset', torch.zeros_like(age))


def reset_handstand_stage(env, env_ids, pose):
    """Train from walking, a feasible front-foot pose, and the start of recovery."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not hasattr(env, '_handstand_age_offset'):
        env._handstand_age_offset = torch.zeros(env.num_envs, device=env.device)
    env._handstand_age_offset[ids] = 0.
    draw = torch.rand(len(ids), device=env.device)
    hold = ids[(draw >= .5) & (draw < .75)]
    recover = ids[draw >= .75]
    env._handstand_age_offset[hold] = 4.
    env._handstand_age_offset[recover] = 6.
    staged = torch.cat((hold, recover))
    if staged.numel() == 0:
        return
    robot = env.scene['robot']
    root = robot.data.default_root_state[staged].clone()
    root[:, :3] = env.scene.env_origins[staged]
    root[:, 2] += pose['base_z_m']
    angle = torch.deg2rad(root.new_tensor(-pose['pitch_deg'])) / 2
    root[:, 3:7] = root.new_tensor([torch.cos(angle), 0., torch.sin(angle), 0.])
    root[:, 7:] = 0.
    joints = root.new_tensor(pose['joint_q']).expand(len(staged), -1)
    robot.write_root_state_to_sim(root, env_ids=staged)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=staged)


def skill_observation(env):
    return hs.skill_state(effective_age(env))


def handstand_time_out(env):
    return effective_age(env) >= 10.


def handstand_fell(env):
    return hs.failure_from_state(hs.robot_state(env))


def handstand_reward(env, velocity_limits, weights, gait, foot_radii):
    if not hasattr(env, '_handstand_reward_consts'):
        env._handstand_reward_consts = st.robot_consts(env)
    s = hs.robot_state(env)
    pose = st.planar_pose(env)
    if not hasattr(env, '_trot_reward_cache'):
        env._trot_reward_cache = {'s': s.detach().clone(), 'pose': pose.detach().clone(),
                                  'valid': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)}
    cache = env._trot_reward_cache
    age = effective_age(env)
    reward, terms = hs.rewards(
        cache['s'], s, st.pose_delta(cache['pose'], pose), env.action_manager.action,
        env.action_manager.prev_action, env.command_manager.get_term('twist').command,
        phase_observation(env), age, env._handstand_reward_consts,
        torch.as_tensor(velocity_limits, device=env.device), handstand_fell(env),
        weights, gait, foot_radii, valid=cache['valid'], dt=env.step_dt)
    env._handstand_reward_terms = {key: value.detach() for key, value in terms.items()}
    cache['s'].copy_(s.detach())
    cache['pose'].copy_(pose.detach())
    cache['valid'].fill_(True)
    return reward


def handstand_env_cfg(parameters, play=False):
    cfg = locomotion_env_cfg(parameters, play=play, evaluation=True)
    cfg.commands['twist'] = HandstandCommandCfg(
        entity_name='robot', resampling_time_range=(10., 10.), heading_command=False,
        rel_heading_envs=0., evaluation=True,
        ranges=HandstandCommandCfg.Ranges(lin_vel_x=(0., .3),
                                          lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)))
    cfg.observations['actor'].terms['skill'] = ObservationTermCfg(func=skill_observation)
    if not play:
        pose = find_pose(parameters, angle_deg=40.)
        if not pose['geometry_passed']:
            raise ValueError('Handstand training reset pose is geometrically invalid')
        cfg.events['handstand_stage_reset'] = EventTermCfg(
            func=reset_handstand_stage, mode='reset', params={'pose': pose})
    cfg.terminations = {
        'time_out': TerminationTermCfg(func=handstand_time_out, time_out=True),
        'handstand_fell': TerminationTermCfg(func=handstand_fell),
    }
    cfg.rewards = {'handstand': RewardTermCfg(func=handstand_reward, weight=1., params={
        'velocity_limits': [parameters['joints'][n]['velocity_limit'] for n in SDK_JOINT_ORDER],
        'weights': hs.HandstandRewards(), 'gait': trot.GaitCfg(),
        'foot_radii': model_foot_radii(),
    })}
    cfg.episode_length_s = 10.
    cfg.auto_reset = True
    return cfg


def register_handstand_task(parameters):
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if hs.TASK in list_tasks():
        raise ValueError('X30 handstand task already registered')
    register_mjlab_task(hs.TASK, handstand_env_cfg(parameters), handstand_env_cfg(parameters, play=True),
                        RslRlOnPolicyRunnerCfg())
