"""Independent phase-conditioned X30 flat-ground environment."""
import torch
from mjlab.managers import EventTermCfg,ObservationTermCfg,RewardTermCfg
from go2_supermimic.robots.x30 import trot
from go2_supermimic.robots.x30.env import x30_velocity_env_cfg,fell_over
from go2_supermimic.robots.x30.model import load_x30_spec,SDK_JOINT_ORDER
from go2_supermimic.supermimic import state as st

TRAINING_COMMAND_RANGES={"lin_vel_x":(.3,.5),"lin_vel_y":(0.,0.),"ang_vel_z":(-.2,.2)}


def model_foot_radii():
    import mujoco
    model=load_x30_spec().compile()
    return tuple(float(model.geom_size[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'{leg}_foot_collision'),0])
                 for leg in ('FL','FR','HL','HR'))


def phase_observation(env):
    return trot.phase_from_steps(env.episode_length_buf,env.step_dt)


def reset_reward_cache(env,env_ids):
    if hasattr(env,'_trot_reward_cache'):
        env._trot_reward_cache['valid'][slice(None) if env_ids is None else env_ids]=False


def trot_reward(env,velocity_limits,min_height,weights,gait,foot_radii):
    if not hasattr(env,'_trot_reward_consts'):
        env._trot_reward_consts=st.robot_consts(env)
    s=trot.robot_state(env);pose=st.planar_pose(env)
    if not hasattr(env,'_trot_reward_cache'):
        env._trot_reward_cache={'s':s.detach().clone(),'pose':pose.detach().clone(),
                               'valid':torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)}
    cache=env._trot_reward_cache
    reward,terms=trot.rewards(cache['s'],s,st.pose_delta(cache['pose'],pose),
        env.action_manager.action,env.action_manager.prev_action,
        env.command_manager.get_term('twist').command,phase_observation(env),env._trot_reward_consts,
        torch.as_tensor(velocity_limits,device=env.device),fell_over(env,min_height),weights,gait,foot_radii,
        valid=cache['valid'],dt=env.step_dt)
    env._trot_reward_terms={key:value.detach() for key,value in terms.items()}
    cache['s'].copy_(s.detach());cache['pose'].copy_(pose.detach());cache['valid'].fill_(True)
    return reward


def trot_env_cfg(parameters,play=False):
    cfg=x30_velocity_env_cfg(parameters,play=play)
    cmd=cfg.commands['twist']
    for name,bounds in TRAINING_COMMAND_RANGES.items():setattr(cmd.ranges,name,bounds)
    cmd.rel_standing_envs=.1
    cfg.observations['actor'].terms['gait_phase']=ObservationTermCfg(func=phase_observation)
    cfg.events['trot_reset_cache']=EventTermCfg(func=reset_reward_cache,mode='reset')
    cfg.rewards={'velocity':RewardTermCfg(func=trot_reward,weight=1.,params={
        'velocity_limits':[parameters['joints'][name]['velocity_limit'] for name in SDK_JOINT_ORDER],
        'min_height':parameters['standing_height']*.45,'weights':trot.TrotRewards(),
        'gait':trot.GaitCfg(),'foot_radii':model_foot_radii()})}
    return cfg


def register_trot_task(parameters):
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.tasks.registry import list_tasks,register_mjlab_task
    if trot.TASK in list_tasks():raise ValueError('X30 trot task is already configured in this process')
    register_mjlab_task(trot.TASK,trot_env_cfg(parameters),trot_env_cfg(parameters,play=True),RslRlOnPolicyRunnerCfg())
