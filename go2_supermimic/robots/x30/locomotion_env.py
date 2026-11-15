"""Independent environment for automatically sampled omnidirectional locomotion."""
import torch
from mjlab.managers import EventTermCfg,ObservationTermCfg,RewardTermCfg
from go2_supermimic.robots.x30 import locomotion as loc,trot
from go2_supermimic.robots.x30.trot_env import trot_env_cfg,model_foot_radii
from go2_supermimic.robots.x30.curriculum_command import CurriculumCommandCfg
from go2_supermimic.robots.x30.env import fell_over
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.supermimic import state as st


def reset_phase(env,env_ids):
    if not hasattr(env,'_loc_phase'):
        env._loc_phase=torch.zeros(env.num_envs,2,device=env.device);env._loc_phase[:,1]=1.
        env._loc_phase_step=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
    ids=slice(None) if env_ids is None else env_ids
    env._loc_phase[ids]=env._loc_phase.new_tensor([0.,1.]);env._loc_phase_step[ids]=0


def phase_observation(env):
    if not hasattr(env,'_loc_phase'):reset_phase(env,None)
    steps=env.episode_length_buf-env._loc_phase_step
    ids=(steps>0).nonzero().flatten()
    if len(ids):
        command=env.command_manager.get_term('twist').command[ids]
        env._loc_phase[ids]=loc.advance_phase(env._loc_phase[ids],command,steps[ids]*env.step_dt)
        env._loc_phase_step[ids]=env.episode_length_buf[ids]
    return env._loc_phase.clone()


def locomotion_reward(env,velocity_limits,min_height,weights,gait,foot_radii):
    if not hasattr(env,'_trot_reward_consts'):env._trot_reward_consts=st.robot_consts(env)
    s=loc.robot_state(env);pose=st.planar_pose(env)
    if not hasattr(env,'_trot_reward_cache'):
        env._trot_reward_cache={'s':s.detach().clone(),'pose':pose.detach().clone(),
                               'valid':torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)}
    cache=env._trot_reward_cache
    reward,terms=loc.rewards(cache['s'],s,st.pose_delta(cache['pose'],pose),env.action_manager.action,
        env.action_manager.prev_action,env.command_manager.get_term('twist').command,phase_observation(env),
        env._trot_reward_consts,torch.as_tensor(velocity_limits,device=env.device),fell_over(env,min_height),
        weights,gait,foot_radii,valid=cache['valid'],dt=env.step_dt)
    env._trot_reward_terms={key:value.detach() for key,value in terms.items()}
    cache['s'].copy_(s.detach());cache['pose'].copy_(pose.detach());cache['valid'].fill_(True)
    return reward


def locomotion_env_cfg(parameters,play=False,evaluation=False):
    cfg=trot_env_cfg(parameters,play=play)
    cfg.commands['twist']=CurriculumCommandCfg(entity_name='robot',resampling_time_range=(3.,6.),
        heading_command=False,rel_heading_envs=0.,evaluation=evaluation,
        ranges=CurriculumCommandCfg.Ranges(lin_vel_x=(-.6,3.),lin_vel_y=(-1.,1.),ang_vel_z=(-1.2,1.2)))
    cfg.observations['actor'].terms['gait_phase']=ObservationTermCfg(func=phase_observation)
    cfg.events['locomotion_reset_phase']=EventTermCfg(func=reset_phase,mode='reset')
    cfg.rewards={'velocity':RewardTermCfg(func=locomotion_reward,weight=1.,params={
        'velocity_limits':[parameters['joints'][n]['velocity_limit'] for n in SDK_JOINT_ORDER],
        'min_height':parameters['standing_height']*.45,'weights':loc.LocomotionRewards(),
        'gait':trot.GaitCfg(),'foot_radii':model_foot_radii()})}
    cfg.episode_length_s=1e9 if play else 40.
    return cfg


def register_locomotion_task(parameters):
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.tasks.registry import list_tasks,register_mjlab_task
    if loc.TASK in list_tasks():raise ValueError('Locomotion task already registered')
    register_mjlab_task(loc.TASK,locomotion_env_cfg(parameters),locomotion_env_cfg(parameters,play=True),RslRlOnPolicyRunnerCfg())
