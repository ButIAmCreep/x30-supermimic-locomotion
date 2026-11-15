"""Evaluate or watch a curriculum checkpoint with its own physical contract."""
import argparse,json,math,time
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from go2_supermimic.robots.x30 import locomotion as loc
from go2_supermimic.robots.x30.locomotion_env import locomotion_env_cfg
from go2_supermimic.robots.x30.locomotion_train import contract
from go2_supermimic.robots.x30.locomotion_transfer import action_gain
from go2_supermimic.robots.x30.robot import validate_parameters
from go2_supermimic.robots.x30.trot_eval import parse_sequence,scheduled_command,Evaluation
from go2_supermimic.robots.x30.curriculum_eval import adaptive_phase_targets
from go2_supermimic.robots.x30.trot_metrics import GaitMetrics,world_model_errors
from go2_supermimic.supermimic import state as st


def sequence_event(env,env_ids,schedule):
    term=env.command_manager.get_term('twist')
    cmd=scheduled_command(schedule,env.common_step_counter)
    term.override=torch.tensor(cmd,device=env.device).expand(env.num_envs,-1).clone()


def prepare(checkpoint,command,num_envs,seed,video=False,width=640,height=360,schedule=None,uturn=None,
            task=loc.TASK,contract_fn=contract,env_fn=locomotion_env_cfg,levels=None):
    if num_envs<1 or len(command)!=3 or not all(math.isfinite(v) for v in command):raise ValueError('Positive env count and finite VX VY WZ required')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);saved=ckpt.get('contract',{})
    if saved.get('task')!=task:raise ValueError(f'Requires {task} checkpoint')
    parameters=validate_parameters(saved['parameters'])
    if saved!=contract_fn(parameters):raise ValueError('Checkpoint differs from current curriculum contract')
    actor=loc.LocomotionActor(235,tuple(ckpt['cfg']['actor_hidden']),12,action_gain(parameters))
    actor.load_state_dict(ckpt['actor']);actor.eval()
    cfg=env_fn(parameters,play=True,evaluation=True)
    cfg.scene.num_envs=num_envs;cfg.seed=seed;cfg.auto_reset=True
    cfg.viewer.width=width;cfg.viewer.height=height;cfg.viewer.distance=2.5
    cfg.viewer.origin_type=cfg.viewer.OriginType.ASSET_ROOT;cfg.viewer.entity_name='robot'
    if schedule:cfg.events['evaluation_sequence']=EventTermCfg(func=sequence_event,mode='step',params={'schedule':schedule})
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array' if video else None)
    try:
        term=env.command_manager.get_term('twist')
        term.override=torch.tensor(command,device=env.device).expand(num_envs,-1).clone()
        env.reset()
        if schedule:sequence_event(env,None,schedule);term._update_command()
        if uturn:
            cap=ckpt.get('curriculum',{}).get('controller',{}).get('axes',{}).get('yaw',{}).get('level',0)
            if levels is None:
                from go2_supermimic.robots.x30.curriculum import LEVELS
                levels=LEVELS
            signed=levels['yaw'][cap]*(1 if uturn=='left' else -1)
            term.begin_uturn(torch.arange(num_envs,device=env.device),torch.full((num_envs,),signed,device=env.device))
            term._update_command()
        return Evaluation(env,actor.to(env.device),ckpt)
    except BaseException:env.close();raise


def evaluate(args,task=loc.TASK,contract_fn=contract,env_fn=locomotion_env_cfg,levels=None):
    schedule=parse_sequence(args.command_sequence)
    if not math.isfinite(args.seconds) or args.seconds<.02:raise ValueError('seconds must be finite and >=.02')
    b=prepare(args.checkpoint,args.command,args.num_envs,args.seed,bool(args.video),args.width,args.height,schedule,args.uturn,
              task,contract_fn,env_fn,levels)
    env=b.env;writer=None;observer=None;transitions=[]
    try:
        observer=GaitMetrics(env,2.,phase_targets=adaptive_phase_targets)
        if args.video:
            import mediapy
            Path(args.video).parent.mkdir(parents=True,exist_ok=True)
            writer=mediapy.VideoWriter(args.video,shape=(args.height,args.width),fps=50);writer.__enter__()
        obs=env.get_observations()['actor'];steps=sum(r[1] for r in schedule) if schedule else round(args.seconds/.02)
        totals=torch.zeros(6,device=env.device);count=falls=0;previous=torch.zeros(env.num_envs,12,device=env.device)
        start=last=time.monotonic()
        with torch.no_grad():
            for i in range(steps):
                s0=loc.robot_state(env).clone();p0=st.planar_pose(env).clone();age=env.episode_length_buf.clone()*.02
                cmd=env.command_manager.get_term('twist').command.clone();a=b.actor(obs)
                out,_,died,timed,_=env.step(a);done=died|timed;s1=loc.robot_state(env).clone()
                valid=(age>=2)&(~done);count+=int(valid.sum());falls+=int(died.sum())
                error=(s1[:,4:6]-cmd[:,:2]).square().sum(-1)
                values=torch.cat([s1[:,4:6],s1[:,9:10],error[:,None],(s1[:,9]-cmd[:,2]).square()[:,None],s1[:,4:6].norm(dim=-1)[:,None]],-1)
                totals+=(values*valid[:,None]).sum(0)
                if args.wm_check:transitions.append((s0,a.clone(),previous.clone(),s1,st.pose_delta(p0,st.planar_pose(env)),done.clone(),age))
                previous=torch.where(done[:,None],0.,a);obs=out['actor']
                if writer:writer.add_image(env.render())
                now=time.monotonic()
                if now-last>=1 or i+1==steps:
                    elapsed=now-start;eta=elapsed*(steps-i-1)/(i+1)
                    print(f'Evaluation {i+1}/{steps} ({100*(i+1)/steps:.1f}%) elapsed={elapsed:.0f}s ETA={eta:.0f}s',flush=True);last=now
        means=(totals/max(1,count)).cpu().tolist()
        report={'task':task,'checkpoint':str(args.checkpoint),'iteration':b.checkpoint['iteration'],
            'command':list(args.command),'seed':args.seed,'seconds':steps*.02,'samples':count,'falls':falls,
            'mean_vx_m_s':means[0] if count else None,'mean_vy_m_s':means[1] if count else None,
            'mean_wz_rad_s':means[2] if count else None,'linear_rmse_m_s':math.sqrt(means[3]) if count else None,
            'yaw_rmse_rad_s':math.sqrt(means[4]) if count else None,'mean_speed_m_s':means[5] if count else None,
            'gait':observer.summary(),'curriculum':b.checkpoint.get('curriculum',{}).get('controller')}
        if args.wm_check:
            wm=loc.ActionMappedWorldModel(tuple(b.checkpoint['cfg']['wm_hidden']),1/action_gain(b.checkpoint['contract']['parameters'])).to(env.device)
            wm.load_state_dict(b.checkpoint['world_model'])
            report['world_model']=world_model_errors(None,transitions,.02,2.,world_model=wm)
        if args.report:
            path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(report,indent=2)+'\n');observer.plot(path.with_suffix('.png'))
        print(json.dumps(report,indent=2));return report
    finally:
        if writer:writer.__exit__(None,None,None)
        if observer:observer.close()
        env.close()


def play(args,task=loc.TASK,contract_fn=contract,env_fn=locomotion_env_cfg,levels=None):
    from mjlab.viewer import NativeMujocoViewer
    b=prepare(args.checkpoint,args.command,1,args.seed,uturn=args.uturn,
              task=task,contract_fn=contract_fn,env_fn=env_fn,levels=levels);count=0
    try:
        def policy(obs):
            nonlocal count
            count+=1
            if count%50==0:
                s=loc.robot_state(b.env)[0];cmd=b.env.command_manager.get_term('twist').command[0]
                print(f'[LIVE] {count*.02:.1f}s command={cmd.tolist()} actual vx={float(s[4]):.3f} vy={float(s[5]):.3f} wz={float(s[9]):.3f}',flush=True)
            return b.actor(obs['actor'])
        viewer=NativeMujocoViewer(b.env,policy,frame_rate=30.,enable_perturbations=False,verbosity=1)
        viewer.run(num_steps=round(args.seconds/.02) if args.seconds is not None else None)
    finally:b.env.close()


def main(live=False,task=loc.TASK,contract_fn=contract,env_fn=locomotion_env_cfg,levels=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--command',type=float,nargs=3,default=(.4,0.,0.),metavar=('VX','VY','WZ'))
    parser.add_argument('--seconds',type=float,default=None if live else 30.)
    parser.add_argument('--uturn',choices=('left','right'))
    if not live:
        parser.add_argument('--num-envs',type=int,default=1);parser.add_argument('--video')
        parser.add_argument('--width',type=int,default=640);parser.add_argument('--height',type=int,default=360)
        parser.add_argument('--report');parser.add_argument('--wm-check',action='store_true');parser.add_argument('--command-sequence')
    args=parser.parse_args()
    if live:
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<.02):parser.error('seconds must be finite and >=.02')
        play(args,task,contract_fn,env_fn,levels)
    else:evaluate(args,task,contract_fn,env_fn,levels)
