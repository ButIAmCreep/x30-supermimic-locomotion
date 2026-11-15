"""Independent phase-conditioned trot trainer, sharing the physical dynamics learner."""
from dataclasses import asdict,replace
import torch
from go2_supermimic.robots.x30 import trot
from go2_supermimic.robots.x30.trot_env import register_trot_task,model_foot_radii,phase_observation,TRAINING_COMMAND_RANGES
from go2_supermimic.robots.x30.train import VelocityTrainer,contract as velocity_contract,defaults as velocity_defaults
from go2_supermimic.supermimic.imagine import ImagineCfg


def contract(parameters):
    return velocity_contract(parameters)|{
        'version':2,'task':trot.TASK,'obs_terms':[list(term) for term in trot.OBS_TERMS],
        'history':trot.HISTORY,'reward_weights':asdict(trot.TrotRewards()),'gait':asdict(trot.GaitCfg()),
        'foot_radii':list(model_foot_radii()),'critic_dim':79,'task_state_dim':5,
        'training_commands':{name:list(bounds) for name,bounds in TRAINING_COMMAND_RANGES.items()},
        'phase_semantics':'sin/cos episode time; reset=0; clock advances during stop; reward at successor phase; commands fixed within imagination',
        'slip_semantics':'foot-center world displacement / dt, target stance gated, first post-reset delta masked',
    }


def defaults():
    return replace(velocity_defaults(),task=trot.TASK,robot_config='configs/x30/robot.trot.json',
        wm_window=16,horizon=16,imag_batch=128,warmup_iterations=500,iterations=15000,
        log_root='logs/x30_trot',run_name='x30_trot')


class TrotTrainer(VelocityTrainer):
    task_id=trot.TASK
    policy_module=trot
    task_contract=staticmethod(contract)
    register_task=staticmethod(register_trot_task)
    task_state_dim=5

    def __init__(self,cfg):
        super().__init__(cfg)
        self.reward_valid=torch.zeros(cfg.num_envs,cfg.buffer_len,dtype=torch.bool,device=self.dev)

    def _prepare_imagination_seed(self,seed,slots):
        return seed|{'reward_valid':self.reward_valid[seed['env'],slots]}

    def _make_imagination(self,parameters,limits):
        params=self.env_cfg.rewards['velocity'].params
        return trot.TrotImagination(
            ImagineCfg(horizon=self.cfg.horizon,dt=self.env.step_dt,compile=self.cfg.compile),self.consts,
            limits,params['min_height'],params['weights'],params['gait'],params['foot_radii'])

    def _task_state(self,command):
        cache=getattr(self.env,'_trot_reward_cache',None)
        self._pre_reward_valid=(cache['valid'].clone() if cache is not None else
                               torch.zeros(self.env.num_envs,dtype=torch.bool,device=self.dev))
        return torch.cat([command,phase_observation(self.env)],-1)

    def _record_task_stats(self):
        self.reward_valid[:,(self.buf.ptr-1)%self.buf.L]=self._pre_reward_valid
        for name,value in getattr(self.env,'_trot_reward_terms',{}).items():
            contribution=getattr(self.imag.weights,name)*value.mean()
            self.ep_stats[f'Metrics/velocity/reward_{name}'].append(float(contribution))
