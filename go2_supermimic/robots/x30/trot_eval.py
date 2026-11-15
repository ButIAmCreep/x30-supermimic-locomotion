"""Deterministic evaluation and native live viewing of X30-Trot-Flat checkpoints."""
from dataclasses import dataclass
from pathlib import Path
import argparse,json,math,time
import torch
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.robots.x30 import trot
from go2_supermimic.robots.x30.trot_env import trot_env_cfg,phase_observation
from go2_supermimic.robots.x30.trot_train import contract
from go2_supermimic.robots.x30.train import VelocityActor
from go2_supermimic.robots.x30.robot import load_parameters,validate_parameters
from go2_supermimic.supermimic import state as st


def parse_sequence(text,dt=.02):
    if text is None:return None
    try:rows=json.loads(text)
    except (TypeError,json.JSONDecodeError) as exc:raise ValueError('command-sequence must be a JSON array') from exc
    if not isinstance(rows,list) or not rows:raise ValueError('command-sequence must not be empty')
    schedule=[];start=0
    for row in rows:
        if not isinstance(row,list) or len(row)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in row):
            raise ValueError('Each command segment must be [seconds,VX,VY,WZ] with finite numbers')
        if row[0]<dt:raise ValueError('Each command duration must be at least one control step')
        count=max(1,round(row[0]/dt));schedule.append((start,count,tuple(float(v) for v in row[1:])))
        start+=count
    return schedule


def scheduled_command(schedule,step):
    return next(row[2] for row in reversed(schedule) if row[0]<=step)


def apply_command_schedule(env,env_ids,schedule):
    # Runs after reward but before command update and new observation history.
    command=scheduled_command(schedule,env.common_step_counter)
    term=env.command_manager.get_term('twist')
    term.cfg.ranges.lin_vel_x=(command[0],command[0])
    term.cfg.ranges.lin_vel_y=(command[1],command[1])
    term.cfg.ranges.ang_vel_z=(command[2],command[2])
    term.command[:]=term.command.new_tensor(command)


def tracking_summary(values,count):
    linear,angular,speed,falls,bound=values
    return {'samples_after_discard':count,
        'planar_velocity_rmse_m_s':math.sqrt(linear/count) if count else None,
        'body_wz_rmse_rad_s':math.sqrt(angular/count) if count else None,
        'mean_planar_speed_m_s':speed/count if count else None,
        'falls':int(falls),'action_near_bound_fraction':bound/count if count else None}

@dataclass
class Evaluation:
    env: object
    actor: object
    checkpoint: dict


def prepare(checkpoint,command,num_envs,seed,robot_config='',video=False,width=640,height=360,auto_reset=False,schedule=None):
    if num_envs<1 or len(command)!=3 or not all(math.isfinite(v) for v in command):
        raise ValueError('Evaluation requires positive num_envs and three finite command values')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    saved=ckpt.get('contract',{})
    if saved.get('task')!=trot.TASK:raise ValueError('A compatible X30-Trot-Flat checkpoint is required; task differs')
    p=validate_parameters(saved['parameters'])
    if saved!=contract(p):raise ValueError('Checkpoint contract differs from current trot model, phase, reward or action definitions')
    if robot_config and load_parameters(robot_config)!=p:raise ValueError('robot-config differs from checkpoint contract')
    if schedule:command=schedule[0][2]
    cfg=trot_env_cfg(p,play=True);cfg.scene.num_envs=num_envs;cfg.seed=seed;cfg.auto_reset=auto_reset
    term=cfg.commands['twist'];term.rel_standing_envs=0.
    term.ranges.lin_vel_x=(command[0],command[0]);term.ranges.lin_vel_y=(command[1],command[1]);term.ranges.ang_vel_z=(command[2],command[2])
    if schedule:
        from mjlab.managers import EventTermCfg
        cfg.events['evaluation_command_schedule']=EventTermCfg(func=apply_command_schedule,mode='step',params={'schedule':schedule})
    cfg.viewer.width=width;cfg.viewer.height=height;cfg.viewer.distance=2.;cfg.viewer.elevation=-22.;cfg.viewer.azimuth=135.
    cfg.viewer.origin_type=cfg.viewer.OriginType.ASSET_ROOT;cfg.viewer.entity_name='robot'
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array' if video else None)
    try:
        env.reset()
        actor=VelocityActor(trot.OBS_DIM,tuple(ckpt['cfg']['actor_hidden']),12).to(env.device)
        actor.load_state_dict(ckpt['actor']);actor.eval()
        return Evaluation(env,actor,ckpt)
    except BaseException:
        env.close();raise


def evaluate(args):
    from go2_supermimic.robots.x30.trot_metrics import GaitMetrics,world_model_errors
    if getattr(args,'mode',None) or getattr(args,'terrain',None):raise ValueError('Trot task uses --command VX VY WZ')
    discard=getattr(args,'discard_seconds',2.)
    if not math.isfinite(args.seconds) or args.seconds<.02 or not math.isfinite(discard) or discard<0:
        raise ValueError('seconds must be >=0.02 and discard-seconds finite and nonnegative')
    schedule=parse_sequence(getattr(args,'command_sequence',None))
    b=prepare(args.checkpoint,tuple(args.command),args.num_envs,args.seed,getattr(args,'robot_config',''),
              bool(args.video),args.width,args.height,schedule=schedule)
    env=b.env;observer=None;writer=None;transitions=[]
    try:
        obs=env.get_observations()['actor'];previous=torch.zeros(env.num_envs,12,device=env.device)
        observer=GaitMetrics(env,discard)
        steps=sum(row[1] for row in schedule) if schedule else int(args.seconds/env.step_dt)
        sums=torch.zeros(5,device=env.device);measurements=0
        segment_sums=torch.zeros(len(schedule),5,device=env.device) if schedule else None
        segment_counts=[0]*len(schedule) if schedule else []
        start=last=time.monotonic()
        if args.video:
            import mediapy
            Path(args.video).parent.mkdir(parents=True,exist_ok=True)
            writer=mediapy.VideoWriter(args.video,shape=(args.height,args.width),fps=round(1/env.step_dt));writer.__enter__()
        with torch.no_grad():
            for i in range(steps):
                s0=trot.robot_state(env).clone();pose0=st.planar_pose(env).clone()
                age0=env.episode_length_buf.clone()*env.step_dt
                command=env.command_manager.get_term('twist').command.clone()
                action=b.actor(obs);out,_,died,timed,_=env.step(action)
                s1=trot.robot_state(env);delta=st.pose_delta(pose0,st.planar_pose(env));done=died|timed
                valid=(age0>=discard)&(~done)
                increment=torch.stack([
                    ((s1[:,st.V_YAW][:,:2]-command[:,:2]).square().sum(-1)*valid).sum(),
                    ((s1[:,9]-command[:,2]).square()*valid).sum(),
                    (s1[:,st.V_YAW][:,:2].norm(dim=-1)*valid).sum(),died.float().sum(),
                    ((action.abs()>.95).float().mean(-1)*valid).sum()])
                sums+=increment;count=int(valid.sum());measurements+=count
                if schedule:
                    index=next(j for j in reversed(range(len(schedule))) if schedule[j][0]<=i)
                    segment_sums[index]+=increment;segment_counts[index]+=count
                if getattr(args,'wm_check',False):
                    transitions.append((s0,action.clone(),previous.clone(),s1.clone(),delta.clone(),done.clone(),age0))
                previous=torch.where(done[:,None],0.,action)
                if writer:writer.add_image(env.render())
                obs=env.reset(env_ids=done.nonzero().flatten())[0]['actor'] if done.any() else out['actor']
                now=time.monotonic()
                if now-last>=1 or i+1==steps:
                    elapsed=now-start;eta=elapsed*(steps-i-1)/(i+1)
                    print(f'\rTrot evaluation {i+1}/{steps} ({100*(i+1)/steps:.1f}%) | elapsed {elapsed:.0f}s | ETA {eta:.0f}s',end='\n' if i+1==steps else '',flush=True);last=now
        values=sums.cpu().tolist();falls=values[3]
        result={'task':trot.TASK,'checkpoint':str(args.checkpoint),'iteration':b.checkpoint['iteration'],
            'command':list(args.command) if schedule is None else None,'seed':args.seed,'num_envs':env.num_envs,'seconds':steps*env.step_dt,
            **tracking_summary(values,measurements),
            'falls_per_env_minute':falls/(steps*env.step_dt*env.num_envs/60),'gait':observer.summary()}
        if schedule:
            result['segments']=[{'start_seconds':row[0]*env.step_dt,'seconds':row[1]*env.step_dt,'command':list(row[2]),
                **tracking_summary(segment_sums[j].cpu().tolist(),segment_counts[j])} for j,row in enumerate(schedule)]
        if getattr(args,'wm_check',False):result['world_model']=world_model_errors(b.checkpoint,transitions,env.step_dt,discard)
        report=getattr(args,'report',None)
        if report:
            path=Path(report);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+'\n');observer.plot(path.with_suffix('.png'))
        print(json.dumps(result,indent=2),flush=True)
        return result
    finally:
        if writer:writer.__exit__(None,None,None)
        if observer:observer.close()
        env.close()


def play(args):
    from mjlab.viewer import NativeMujocoViewer
    b=prepare(args.checkpoint,tuple(args.command),1,args.seed,args.robot_config,auto_reset=True)
    env=b.env;count=0
    try:
        def policy(obs):
            nonlocal count
            count+=1
            if count%50==0:
                s=trot.robot_state(env)[0]
                print(f'[LIVE] sim {count*env.step_dt:.1f}s | command {tuple(args.command)} | actual vx={float(s[4]):.3f} vy={float(s[5]):.3f} wz={float(s[9]):.3f}',flush=True)
            return b.actor(obs['actor'])
        print(f'Live learned trot actor: {args.checkpoint}; Space=pause, Enter=reset, -/= speed, close window to exit.',flush=True)
        viewer=NativeMujocoViewer(env,policy,frame_rate=30.,enable_perturbations=False,verbosity=1)
        viewer.run(num_steps=round(args.seconds/env.step_dt) if args.seconds is not None else None)
        if viewer.get_status().last_error:raise RuntimeError(viewer.get_status().last_error)
    finally:env.close()


def main(live=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--robot-config',default='')
    parser.add_argument('--command',nargs=3,type=float,default=(.4,0.,0.),metavar=('VX','VY','WZ'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--seconds',type=float,default=None if live else 30.)
    if not live:
        parser.add_argument('--num-envs',type=int,default=16)
        parser.add_argument('--video',default=None)
        parser.add_argument('--width',type=int,default=640);parser.add_argument('--height',type=int,default=360)
        parser.add_argument('--discard-seconds',type=float,default=2.)
        parser.add_argument('--report',default=None);parser.add_argument('--wm-check',action='store_true')
        parser.add_argument('--command-sequence',default=None,help='JSON [[seconds,VX,VY,WZ],...]; total duration replaces --seconds without resetting phase/history')
    args=parser.parse_args()
    if live:
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<.02):parser.error('seconds must be finite and >=0.02')
        play(args)
    else:evaluate(args)

if __name__=='__main__':main()
