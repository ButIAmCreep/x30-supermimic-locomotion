"""Automatic curriculum training, explicit transfer, and resumable course state."""
from dataclasses import dataclass,asdict
import argparse,hashlib,json,math
from pathlib import Path
import torch
from go2_supermimic.robots.x30 import locomotion as loc,trot
from go2_supermimic.robots.x30.curriculum import Curriculum,LEVELS,MODE_PROBABILITIES
from go2_supermimic.robots.x30.trot_train import TrotTrainer
from go2_supermimic.robots.x30.train import contract as velocity_contract
from go2_supermimic.robots.x30.locomotion_transfer import action_gain,transfer_from_trot
from go2_supermimic.robots.x30.locomotion_env import register_locomotion_task,phase_observation
from go2_supermimic.robots.x30.robot import load_parameters
from go2_supermimic.supermimic.train import TrainCfg
from go2_supermimic.supermimic.imagine import ImagineCfg


@dataclass(kw_only=True)
class CurriculumTrainCfg(TrainCfg):
    init_from:str=''
    evaluation_interval:int=1000
    evaluation_seconds:float=12.


def defaults():
    return CurriculumTrainCfg(task=loc.TASK,robot_config='configs/x30/robot.curriculum.json',
        num_envs=64,buffer_len=200,wm_hidden=(256,256),wm_window=16,wm_batch=128,wm_updates=2,
        archive_size=5000,imag_batch=128,horizon=16,iterations=50000,warmup_iterations=500,
        action_noise=.05,explore_noise=.15,
        compile=False,save_interval=5000,log_interval=50,continue_run=True,
        log_root='logs/x30_curriculum',run_name='x30_omnidirectional')


def contract(parameters):
    return velocity_contract(parameters)|{'version':3,'task':loc.TASK,'obs_terms':[list(t) for t in loc.OBS_TERMS],
        'history':5,'reward_weights':asdict(loc.LocomotionRewards()),'gait':asdict(trot.GaitCfg()),
        'profile':'speed=norm(vx,1.5*vy)+.35*abs(wz); frequency=1/.6+1.2*clamp(speed-.5,0,2.5); duty=.6-.2*blend; clearance=.04+.05*blend',
        'phase':'integrate pre-action command; reward/obs successor phase; fixed command in short imagination',
        'action':'scaled old actor plus zero-output residual; base action histories converted to legacy units',
        'world_model_action':'both action inputs mapped to legacy units',
        'gain':action_gain(parameters).tolist(),'levels':{k:list(v) for k,v in LEVELS.items()},
        'mode_probabilities':list(MODE_PROBABILITIES),'curriculum_version':1,'evaluation_version':1,
        'critic_dim':79,'task_state_dim':5}


def validate_optimizer_state(saved,optimizer):
    groups=saved['param_groups'];current=optimizer.param_groups
    if len(groups)!=len(current):raise ValueError('Invalid optimizer groups')
    parameters={}
    for old,new in zip(groups,current):
        ids=old['params']
        if len(ids)!=len(new['params']) or any(type(i) is not int or i in parameters for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('Invalid optimizer parameter mapping')
        parameters.update(zip(ids,new['params']))
        for key in ('lr','eps','weight_decay'):
            if key not in old or not math.isfinite(old[key]) or old[key]<0:raise ValueError('Invalid optimizer scalar')
        if len(old['betas'])!=2 or any(not math.isfinite(v) or not 0<=v<1 for v in old['betas']):
            raise ValueError('Invalid Adam betas')
    for key,values in saved['state'].items():
        if key not in parameters or not isinstance(values,dict):raise ValueError('Invalid optimizer state keys')
        if not {'step','exp_avg','exp_avg_sq'}<=set(values):raise ValueError('Incomplete Adam state')
        for name,value in values.items():
            if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite optimizer state')
            if value.shape!=parameters[key].shape and not (name=='step' and value.numel()==1):
                raise ValueError('Invalid optimizer state shape')


class LocomotionTrainer(TrotTrainer):
    task_id=loc.TASK;policy_module=loc
    task_contract=staticmethod(contract);register_task=staticmethod(register_locomotion_task)
    curriculum_type=Curriculum
    initialization_transfer=staticmethod(transfer_from_trot)
    evaluation_env_factory=None

    def __init__(self,cfg):
        if cfg.task!=self.task_id or (cfg.init_from and cfg.resume):raise ValueError('Select this task and only one of init-from/resume')
        if cfg.evaluation_interval<1 or not math.isfinite(cfg.evaluation_seconds) or cfg.evaluation_seconds<6:
            raise ValueError('evaluation-interval >=1 and evaluation-seconds >=6 required')
        if cfg.resume and not cfg.continue_run:raise ValueError('Curriculum resume requires continue-run=true')
        if cfg.resume and cfg.reset_critic:raise ValueError('Same-course resume restores its critic; reset-critic is not supported')
        self.next_evaluation=cfg.evaluation_interval;self.evaluation_round=0;self.provenance=None
        try:self._initialize(cfg)
        except BaseException:
            if hasattr(self,'env'):self.env.close()
            if hasattr(self,'writer'):self.writer.close()
            raise

    def _initialize(self,cfg):
        super().__init__(cfg)
        if cfg.init_from:
            ckpt=torch.load(cfg.init_from,map_location=self.dev,weights_only=False)
            self.provenance=self.initialization_transfer(ckpt,self.actor,self.wm,load_parameters(cfg.robot_config))
            self._after_init_transfer(ckpt)
            self.provenance|={'path':str(Path(cfg.init_from).resolve()),'sha256':hashlib.sha256(Path(cfg.init_from).read_bytes()).hexdigest()}
            print(f'Initialized actor and physical world model from {cfg.init_from}; critic/optimizers/replay are fresh',flush=True)
        (self.log_dir/'initialization.json').write_text(json.dumps(self.provenance,indent=2)+'\n')

    def _after_init_transfer(self,ckpt):
        pass

    def _make_actor(self,parameters):return loc.LocomotionActor(loc.OBS_DIM,self.cfg.actor_hidden,12,action_gain(parameters))
    def _make_world_model(self,parameters):return loc.ActionMappedWorldModel(self.cfg.wm_hidden,1/action_gain(parameters))
    def _update_actor_norm(self):self.actor.norm.update(self.actor.base_observation(self.obs))

    def _make_imagination(self,parameters,limits):
        p=self.env_cfg.rewards['velocity'].params
        return loc.LocomotionImagination(ImagineCfg(horizon=self.cfg.horizon,dt=.02,compile=self.cfg.compile),
            self.consts,limits,p['min_height'],p['weights'],p['gait'],p['foot_radii'])

    def _task_state(self,command):
        cache=getattr(self.env,'_trot_reward_cache',None)
        self._pre_reward_valid=cache['valid'].clone() if cache else torch.zeros(self.env.num_envs,dtype=torch.bool,device=self.dev)
        return torch.cat([command,phase_observation(self.env)],-1)

    def save(self,it):
        path=self.log_dir/f'model_{it}.pt'
        state={'controller':self.term.curriculum.state_dict(),'sampler_rng':self.term.generator.get_state().cpu(),
            'next_evaluation':self.next_evaluation,'evaluation_round':self.evaluation_round,
            'evaluation_interval':self.cfg.evaluation_interval,'evaluation_seconds':self.cfg.evaluation_seconds,
            'provenance':self.provenance,'env_steps':self.env_steps,
            'guard':{k:getattr(self,k) for k in ('lr_scale','ramp_start','actor_frozen_until','policy_start','collapse_strikes')},
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        ckpt={'supermimic':True,'iteration':it,'cfg':asdict(self.cfg),'actor':self.actor.state_dict(),
            'critic':self.critic.state_dict(),'world_model':self.wm.state_dict(),
            'opt_actor':self.opt_actor.state_dict(),'opt_critic':self.opt_critic.state_dict(),'opt_wm':self.opt_wm.state_dict(),
            'obs_dim':loc.OBS_DIM,'contract':self.contract,'curriculum':state}
        temporary=path.with_suffix('.tmp');torch.save(ckpt,temporary);temporary.replace(path)
        return path

    def load(self,path):
        ckpt=torch.load(path,map_location=self.dev,weights_only=False)
        if ckpt.get('contract')!=self.contract:raise ValueError('Curriculum resume requires an identical task/physical/model/reward contract')
        try:
            s=ckpt['curriculum'];validation=self.curriculum_type();validation.load_state_dict(s['controller'])
            if 'provenance' not in s or (s['provenance'] is not None and not isinstance(s['provenance'],dict)):
                raise ValueError('Missing/invalid initialization provenance')
            if type(ckpt['iteration']) is not int or ckpt['iteration']<0:raise ValueError('Invalid iteration')
            if type(s['next_evaluation']) is not int or s['next_evaluation']<=ckpt['iteration']:raise ValueError('Invalid evaluation schedule')
            if type(s['evaluation_round']) is not int or s['evaluation_round']<0:raise ValueError('Invalid evaluation round')
            if s['evaluation_interval']!=self.cfg.evaluation_interval or s['evaluation_seconds']!=self.cfg.evaluation_seconds:
                raise ValueError('Resume evaluation settings differ from saved course')
            rng=torch.Generator(device=self.dev);rng.set_state(s['sampler_rng'].cpu())
            cpu_rng=torch.Generator();cpu_rng.set_state(s['torch_rng'].cpu())
            if not isinstance(s['cuda_rng'],list) or any(x.dtype!=torch.uint8 or x.ndim!=1 or x.numel()!=16 for x in s['cuda_rng']):
                raise ValueError('Invalid CUDA RNG state')
            if str(self.dev).startswith('cuda') and len(s['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA device topology differs')
            if type(s['env_steps']) is not int or s['env_steps']<0:raise ValueError('Invalid environment-step counter')
            if set(s['guard'])!={'lr_scale','ramp_start','actor_frozen_until','policy_start','collapse_strikes'}:
                raise ValueError('Incomplete training guard state')
            for key,value in s['guard'].items():
                if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid training guard state')
                if key!='lr_scale' and type(value) is not int:raise ValueError('Invalid integer guard counter')
            if s['guard']['lr_scale']<=0:raise ValueError('Invalid learning-rate scale')
            for model,key in ((self.actor,'actor'),(self.wm,'world_model'),(self.critic,'critic')):
                target=model.state_dict();source=ckpt[key]
                if set(source)!=set(target) or any(source[k].shape!=v.shape or not bool(torch.isfinite(source[k]).all()) for k,v in target.items()):
                    raise ValueError(f'Invalid {key} checkpoint')
                for name,value in source.items():
                    if name.endswith('.var') and not bool((value>=0).all()):raise ValueError('Negative normalization variance')
                    if name.endswith('.count') and not bool((value>0).all()):raise ValueError('Invalid normalization count')
                    if name in ('action_gain','history_gain','action_ratio') and not torch.equal(value,target[name]):
                        raise ValueError('Physical action mapping differs from contract')
            for opt,key in ((self.opt_actor,'opt_actor'),(self.opt_wm,'opt_wm'),(self.opt_critic,'opt_critic')):
                validate_optimizer_state(ckpt[key],opt)
        except (KeyError,TypeError,RuntimeError) as exc:raise ValueError('Malformed curriculum checkpoint') from exc
        super().load(path)
        self.term.curriculum.load_state_dict(s['controller']);self.term.generator.set_state(s['sampler_rng'].cpu())
        self.next_evaluation=s['next_evaluation'];self.evaluation_round=s['evaluation_round'];self.provenance=s['provenance']
        self.env_steps=int(s['env_steps'])
        self.term.turn_start.fill_(self.env.common_step_counter*self.env.step_dt)
        for k,v in s['guard'].items():setattr(self,k,v)
        torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda_rng']])
        self.last_ckpt=Path(path)

    def _after_iteration(self,it):
        if it<self.next_evaluation:return
        from go2_supermimic.robots.x30.curriculum_eval import evaluate_curriculum
        report=evaluate_curriculum(self,it,env_factory=self.evaluation_env_factory)
        decisions=self.term.curriculum.observe(report)
        self.evaluation_round+=1;self.next_evaluation=it+self.cfg.evaluation_interval
        report|={'decisions':decisions,'curriculum':self.term.curriculum.state_dict()}
        directory=self.log_dir/'evaluations';directory.mkdir(exist_ok=True)
        (directory/f'iteration_{it}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (self.log_dir/'curriculum.jsonl').open('a') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
        caps=self.term.curriculum.caps();print(f'[CURRICULUM] it={it} mastered={caps} probe={self.term.curriculum.caps(True)} decisions={decisions}',flush=True)
        for k,v in caps.items():self.writer.add_scalar(f'curriculum/{k}_cap',v,it)
        for k,row in report['axes'].items():
            for level,passed in row.items():self.writer.add_scalar(f'curriculum/{k}_{level}_passed',int(passed),it)


def main():
    ap=argparse.ArgumentParser(description='X30 automatic omnidirectional curriculum')
    for name,value in asdict(defaults()).items():
        flag='--'+name.replace('_','-')
        if isinstance(value,tuple):ap.add_argument(flag,type=int,nargs='+',default=value)
        elif isinstance(value,bool):ap.add_argument(flag,type=lambda v:v.lower() in ('true','1'),default=value)
        else:ap.add_argument(flag,type=type(value),default=value)
    args=vars(ap.parse_args());cfg=CurriculumTrainCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in args.items()})
    torch.set_float32_matmul_precision('high');trainer=LocomotionTrainer(cfg)
    try:trainer.run()
    finally:trainer.env.close();trainer.writer.close()
