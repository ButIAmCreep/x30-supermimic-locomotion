"""Frozen-policy exams in an independent physics environment, never imagination."""
from contextlib import contextmanager
import math,time,os,random
import numpy as np
import torch
from mjlab.managers import EventTermCfg
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.robots.x30 import locomotion as loc
from go2_supermimic.robots.x30.locomotion_env import locomotion_env_cfg
from go2_supermimic.robots.x30.trot_metrics import GaitMetrics,world_model_errors
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.robot import load_parameters
from go2_supermimic.supermimic import state as st


def case_passed(row,minimum_samples=50):
    required=('samples','falls','tracking_normalized_rmse','yaw_rmse','max_tilt_deg',
        'torque_saturation_fraction','overspeed_fraction','phase_agreement','phase_labels',
        'unscheduled_flight_fraction','linear_rmse')
    if any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in required):return False
    ok=(row['samples']>=minimum_samples and row['falls']==0 and row['tracking_normalized_rmse']<=1 and
        row['yaw_rmse']<=.12 and row['max_tilt_deg']<=15 and row['torque_saturation_fraction']<=.05 and
        row['overspeed_fraction']<=.05 and row['phase_labels']>=20 and row['phase_agreement']>=.8 and
        row['unscheduled_flight_fraction']<=.02)
    if row.get('kind')=='stop':
        ok=ok and all(isinstance(row.get(k),(float,int)) and math.isfinite(row[k]) for k in ('stop_speed','stop_drift'))
        ok=ok and row['stop_speed']<=.03 and row['stop_drift']<=.05
    if row.get('kind')=='uturn':
        ok=ok and row.get('uturn_complete') is True and isinstance(row.get('heading_error'),(int,float))
        ok=ok and math.isfinite(row['heading_error']) and row['heading_error']<=.08
    if row.get('kind')=='transition':
        ok=ok and isinstance(row.get('segments'),list) and len(row['segments'])==len(SWITCHES)
        ok=ok and all(s.get('passed') is True for s in row['segments'])
    return bool(ok)


def world_model_passed(report):
    for h in ('1','8','16'):
        row=report.get(h,{})
        if row.get('finite') is not True or row.get('windows',0)<10:return False
        for key,floor,ceiling in (('joint_angle_deg',.3,5.),('foot_height_mm',2.,20.)):
            x=row.get(key,{})
            error,base=x.get('rmse'),x.get('persistence_rmse')
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (error,base)):return False
            if error>min(ceiling,max(floor,.95*base)):return False
        for key,limit in (('linear_velocity_m_s',.25),):
            x=row.get(key,{})
            if not isinstance(x.get('rmse'),(int,float)) or not math.isfinite(x['rmse']) or x['rmse']>limit:return False
        yaw=row.get('yaw_displacement_rmse_rad')
        if not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or yaw>.08:return False
    return True


def make_cases(curriculum):
    rows=[]
    for grade in ('mastered','probe'):
        caps=curriculum.caps(probe=grade=='probe')
        for axis in ('forward','backward','lateral','yaw'):
            for sign in ((-1,1) if axis in ('lateral','yaw') else (1,)):
                cmd=[0.,0.,0.]
                cmd[{'forward':0,'backward':0,'lateral':1,'yaw':2}[axis]]=caps[axis]*sign*(-1 if axis=='backward' else 1)
                rows.append(dict(name=f'{axis}_{grade}_{sign}',axis=axis,grade=grade,kind=axis,command=cmd))
    for name,kind,cmd in [('retention_slow','forward',[.3,0.,0.]),('retention_fast','forward',[.5,0.,0.]),
                           ('stop','stop',[.3,0.,0.]),('left_turn','turn',[.4,0.,.15]),
                           ('right_turn','turn',[.4,0.,-.15]),('switches','transition',[.3,0.,0.]),
                           ('uturn_left','uturn',[0.,0.,curriculum.caps()['yaw']]),
                           ('uturn_right','uturn',[0.,0.,-curriculum.caps()['yaw']])]:
        axis='yaw' if kind=='uturn' else None
        grade='probe' if kind=='uturn' else 'maneuver' if kind=='transition' else 'retention'
        rows.append(dict(name=name,axis=axis,grade=grade,kind=kind,command=cmd))
    # Newly unlocked forward speed also has to support bounded curved travel.
    speed=curriculum.caps(True)['forward'];w=min(.15,curriculum.caps()['yaw'],1.5/speed)
    for sign in (-1,1):rows.append(dict(name=f'probe_curve_{sign}',axis='forward',grade='probe',kind='turn',command=[speed,0.,sign*w]))
    return rows


SWITCHES=((.3,0.,0.),(0.,0.,0.),(.5,0.,0.),(.4,0.,.15),(.4,0.,-.15),(-.2,0.,0.),(0.,.2,0.),(0.,-.2,0.))


def exam_commands(env,env_ids,cases):
    term=env.command_manager.get_term('twist');t=env.common_step_counter*env.step_dt
    commands=[]
    for case in cases:
        if case['kind']=='stop':cmd=case['command'] if t<3 else (0.,0.,0.)
        elif case['kind']=='transition':cmd=SWITCHES[min(int(t/3),len(SWITCHES)-1)]
        else:cmd=case['command']
        commands.append(cmd)
    term.override=torch.tensor(commands,device=env.device)


def exam_environment_config(parameters,seed,cases,env_factory=None):
    cfg=(env_factory or locomotion_env_cfg)(parameters,play=True,evaluation=True)
    cfg.scene.num_envs=len(cases);cfg.seed=seed;cfg.auto_reset=True
    cfg.events['exam_commands']=EventTermCfg(func=exam_commands,mode='step',params={'cases':cases})
    return cfg


def adaptive_phase_targets(env,substep):
    phase=env._loc_phase;command=env.command_manager.get_term('twist').command
    frequency,duty,_=loc.profile(command)
    fraction=torch.remainder(torch.atan2(phase[:,0],phase[:,1])/(2*math.pi)+frequency*substep*env.physics_dt,1.)
    return fraction.cpu().numpy(),duty.cpu().numpy(),(1/frequency).cpu().numpy()


@contextmanager
def isolated_evaluation(actor,world_model,seed):
    actor_mode,wm_mode=actor.training,world_model.training
    python_rng=random.getstate();numpy_rng=np.random.get_state();hash_seed=os.environ.get('PYTHONHASHSEED')
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            torch.manual_seed(seed);actor.eval();world_model.eval()
            with torch.no_grad():yield
        finally:
            actor.train(actor_mode);world_model.train(wm_mode)
            random.setstate(python_rng);np.random.set_state(numpy_rng)
            if hash_seed is None:os.environ.pop('PYTHONHASHSEED',None)
            else:os.environ['PYTHONHASHSEED']=hash_seed


class ExamMeasurements:
    def __init__(self,env,cases):
        self.env,self.cases=env,cases;n=len(cases)
        self.labels=np.zeros(n);self.correct=np.zeros(n);self.air=np.zeros(n);self.physics=np.zeros(n)
        self.samples=np.zeros(n);self.linear=np.zeros(n);self.normalized=np.zeros(n);self.yaw=np.zeros(n)
        self.torque=np.zeros(n);self.overspeed=np.zeros(n);self.fallen=np.zeros(n,dtype=bool)
        self.max_tilt=np.zeros(n);self.stop_speeds=[];self.positions=[]
        self.safety_samples=np.zeros(n)
        self.segment={key:np.zeros((n,len(SWITCHES))) for key in (
            'samples','linear','normalized','yaw','labels','correct','air','physics','torque','overspeed',
            'safety_samples','max_tilt','transient_samples','transient_linear','transient_yaw')}
        self.segment_fallen=np.zeros((n,len(SWITCHES)),dtype=bool)

    def mask(self,age):
        mask=age>2.
        for i,c in enumerate(self.cases):
            if c['kind']=='stop':mask[i]=age[i]>5
            if c['kind']=='transition':mask[i]=(age[i]>2 and (age[i]>=24 or age[i]%3>2))
        return mask

    def contacts(self,found,expected,label_valid,valid):
        env=self.env;sub=(env._sim_step_counter-1)%env.cfg.decimation+1
        age=env.episode_length_buf.cpu().numpy()*.02+sub*env.physics_dt
        mask=self.mask(age)&valid
        label_valid=label_valid&mask[:,None]
        self.labels+=label_valid.sum(-1);self.correct+=((found==expected)&label_valid).sum(-1)
        self.air+=((~found.any(-1))&expected.any(-1)&mask)
        self.physics+=mask
        index=min(int((env.common_step_counter*.02+sub*env.physics_dt)/3),len(SWITCHES)-1)
        for i,case in enumerate(self.cases):
            if case['kind']!='transition':continue
            self.segment['labels'][i,index]+=label_valid[i].sum()
            self.segment['correct'][i,index]+=((found[i]==expected[i])&label_valid[i]).sum()
            self.segment['physics'][i,index]+=mask[i]
            self.segment['air'][i,index]+=int((not found[i].any()) and expected[i].any() and mask[i])

    def control(self,state,command,died,effort,velocity):
        env=self.env;age=env.episode_length_buf.cpu().numpy()*.02;mask=self.mask(age)
        s=state.cpu().numpy();cmd=command.cpu().numpy()
        linear=((s[:,4:6]-cmd[:,:2])**2).sum(-1)
        tolerance=np.maximum(.08,.15*np.linalg.norm(cmd[:,:2],axis=1))
        self.samples+=mask;self.linear+=linear*mask;self.normalized+=linear/tolerance**2*mask
        self.yaw+=(s[:,9]-cmd[:,2])**2*mask
        robot=env.scene['robot']
        torque=(robot.data.actuator_force.abs()>=effort*.98).float().mean(-1).cpu().numpy()
        overspeed=(robot.data.joint_vel.abs()>velocity).float().mean(-1).cpu().numpy()
        self.torque+=torque;self.overspeed+=overspeed;self.safety_samples+=1
        self.fallen|=died.cpu().numpy()
        tilt=np.degrees(np.arccos(np.clip(-s[:,3],-1,1)));self.max_tilt=np.maximum(self.max_tilt,tilt)
        self.stop_speeds.append(np.linalg.norm(s[:,4:6],axis=1))
        self.positions.append(st.planar_pose(env)[:,:2].cpu().numpy().copy())
        index=min(int(max(0,env.common_step_counter-1)*.02/3),len(SWITCHES)-1)
        for i,case in enumerate(self.cases):
            if case['kind']!='transition':continue
            self.segment_fallen[i,index]|=bool(died[i])
            values={'samples':mask[i],'linear':linear[i]*mask[i],
                'normalized':linear[i]/tolerance[i]**2*mask[i],'yaw':(s[i,9]-cmd[i,2])**2*mask[i],
                'torque':torque[i],'overspeed':overspeed[i],'safety_samples':1,
                'transient_samples':not mask[i],'transient_linear':linear[i]*(not mask[i]),
                'transient_yaw':(s[i,9]-cmd[i,2])**2*(not mask[i])}
            for k,value in values.items():self.segment[k][i,index]+=value
            self.segment['max_tilt'][i,index]=max(self.segment['max_tilt'][i,index],tilt[i])

    def segment_summary(self,i):
        rows=[]
        for j,command in enumerate(SWITCHES):
            v={k:a[i,j] for k,a in self.segment.items()};count=max(1,v['samples']);safety=max(1,v['safety_samples'])
            row={'command':list(command),'kind':'segment','samples':int(v['samples']),'falls':int(self.segment_fallen[i,j]),
                'linear_rmse':float(np.sqrt(v['linear']/count)),
                'tracking_normalized_rmse':float(np.sqrt(v['normalized']/count)),
                'yaw_rmse':float(np.sqrt(v['yaw']/count)),'max_tilt_deg':float(v['max_tilt']),
                'torque_saturation_fraction':float(v['torque']/safety),'overspeed_fraction':float(v['overspeed']/safety),
                'phase_labels':int(v['labels']),'phase_agreement':float(v['correct']/max(1,v['labels'])),
                'unscheduled_flight_fraction':float(v['air']/max(1,v['physics'])),
                'transient_samples':int(v['transient_samples']),
                'transient_linear_rmse':float(np.sqrt(v['transient_linear']/max(1,v['transient_samples']))),
                'transient_yaw_rmse':float(np.sqrt(v['transient_yaw']/max(1,v['transient_samples'])))}
            row['passed']=case_passed(row,minimum_samples=40);rows.append(row)
        return rows

    def summary(self):
        rows=[];term=self.env.command_manager.get_term('twist')
        for i,case in enumerate(self.cases):
            count=self.samples[i];denom=max(1,count)
            row=case|{'samples':int(count),'falls':int(self.fallen[i]),
                'linear_rmse':float(np.sqrt(self.linear[i]/denom)),
                'tracking_normalized_rmse':float(np.sqrt(self.normalized[i]/denom)),
                'yaw_rmse':float(np.sqrt(self.yaw[i]/denom)),'max_tilt_deg':float(self.max_tilt[i]),
                'torque_saturation_fraction':float(self.torque[i]/max(1,self.safety_samples[i])),
                'overspeed_fraction':float(self.overspeed[i]/max(1,self.safety_samples[i])),
                'safety_samples':int(self.safety_samples[i]),
                'phase_labels':int(self.labels[i]),'phase_agreement':float(self.correct[i]/max(1,self.labels[i])),
                'unscheduled_flight_fraction':float(self.air[i]/max(1,self.physics[i]))}
            if case['kind']=='stop':
                row|={'stop_speed':float(np.mean(self.stop_speeds[-100:],axis=0)[i]),
                       'stop_drift':float(np.linalg.norm(self.positions[-1][i]-self.positions[-101][i]))}
            if case['kind']=='uturn':row|={'uturn_complete':bool(term.completed[i]&(~term.timed_out[i])),
                'heading_error':float(term.remaining[i].abs()),'timed_out':bool(term.timed_out[i])}
            if case['kind']=='transition':row['segments']=self.segment_summary(i)
            # Failed/numerically invalid experiments must remain JSON-serializable.
            for k,v in list(row.items()):
                if isinstance(v,float) and not math.isfinite(v):row[k]=None
            row['passed']=case_passed(row);rows.append(row)
        return rows


def evaluate_curriculum(trainer,iteration,env_factory=None):
    seed=100000+trainer.cfg.seed+(trainer.evaluation_round%3)*1009
    cases=make_cases(trainer.term.curriculum);parameters=load_parameters(trainer.cfg.robot_config)
    report={'iteration':iteration,'seed':seed,'evaluation_round':trainer.evaluation_round,
            'mastered_caps':trainer.term.curriculum.caps(),'probe_caps':trainer.term.curriculum.caps(True)}
    env=None;observer=None
    with isolated_evaluation(trainer.actor,trainer.wm,seed):
        try:
            cfg=exam_environment_config(parameters,seed,cases,env_factory)
            env=ManagerBasedRlEnv(cfg=cfg,device=trainer.dev);env.reset()
            term=env.command_manager.get_term('twist');exam_commands(env,None,cases)
            for i,c in enumerate(cases):
                if c['kind']=='uturn':term.begin_uturn(torch.tensor([i],device=env.device),torch.tensor([c['command'][2]],device=env.device))
            term._update_command();obs=env.get_observations()['actor']
            measure=ExamMeasurements(env,cases)
            observer=GaitMetrics(env,2.,phase_targets=adaptive_phase_targets,sample_callback=measure.contacts)
            seconds=max(trainer.cfg.evaluation_seconds,24.,math.pi/trainer.term.curriculum.caps()['yaw']+8)
            steps=math.ceil(seconds/env.step_dt);transitions=[];previous=torch.zeros(env.num_envs,12,device=env.device)
            effort=torch.tensor([parameters['joints'][k]['effort_limit'] for k in SDK_JOINT_ORDER],device=env.device)
            velocity=torch.tensor([parameters['joints'][k]['velocity_limit'] for k in SDK_JOINT_ORDER],device=env.device)
            start=last=time.monotonic()
            for step in range(steps):
                s0=loc.robot_state(env).clone();pose0=st.planar_pose(env).clone();age0=env.episode_length_buf.clone()*.02
                command=term.command.clone();action=trainer.actor(obs)
                out,_,died,timeout,_=env.step(action);s1=loc.robot_state(env).clone()
                delta=st.pose_delta(pose0,st.planar_pose(env));done=died|timeout
                measure.control(s1,command,died,effort,velocity)
                transitions.append((s0,action.clone(),previous.clone(),s1,delta.clone(),done.clone(),age0))
                previous=action;obs=out['actor']
                now=time.monotonic()
                if now-last>=2 or step+1==steps:
                    elapsed=now-start;eta=elapsed*(steps-step-1)/(step+1)
                    print(f'[EXAM] it={iteration} {step+1}/{steps} ({100*(step+1)/steps:.1f}%) elapsed={elapsed:.0f}s ETA={eta:.0f}s',flush=True);last=now
            rows=measure.summary();wm=world_model_errors(None,transitions,.02,2.,world_model=trainer.wm)
            report|={'seconds':steps*.02,'cases':rows,'world_model':wm,'world_model_passed':world_model_passed(wm),
                'retention_passed':all(r['passed'] for r in rows if r['grade']=='retention'),
                'axes':{k:{g:all(r['passed'] for r in rows if r['axis']==k and r['grade']==g)
                           for g in ('mastered','probe')} for k in ('forward','backward','lateral','yaw')}}
        finally:
            if observer:observer.close()
            if env:env.close()
    return report
