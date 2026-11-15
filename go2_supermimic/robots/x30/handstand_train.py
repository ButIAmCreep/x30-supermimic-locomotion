"""Train a distinct X30 front-foot support skill from a speed-course policy."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from go2_supermimic.robots.x30 import handstand as hs, trot
from go2_supermimic.robots.x30.handstand_actor import HandstandActor
from go2_supermimic.robots.x30.handstand_env import effective_age, register_handstand_task
from go2_supermimic.robots.x30.handstand_feasibility import find_pose
from go2_supermimic.robots.x30.handstand_imagination import HandstandImagination
from go2_supermimic.robots.x30.handstand_transfer import transfer_from_speed, validate_fixed_buffers
from go2_supermimic.robots.x30.locomotion_env import phase_observation
from go2_supermimic.robots.x30.locomotion_train import validate_optimizer_state
from go2_supermimic.robots.x30.locomotion_transfer import action_gain
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.robot import load_parameters
from go2_supermimic.robots.x30.train import contract as velocity_contract
from go2_supermimic.robots.x30.trot_train import TrotTrainer
from go2_supermimic.supermimic.imagine import ImagineCfg
from go2_supermimic.supermimic.train import TrainCfg


@dataclass(kw_only=True)
class HandstandTrainCfg(TrainCfg):
    init_from: str = ''


def _walking_parameters():
    from go2_supermimic.paths import PROJECT_ROOT
    return load_parameters(str(PROJECT_ROOT / 'configs/x30/robot.curriculum.json'))


def contract(parameters):
    reference = find_pose(parameters, angle_deg=40.)
    if not reference['geometry_passed']:
        raise ValueError('Handstand pose failed geometry check')
    walking = _walking_parameters()
    return velocity_contract(parameters) | {
        'version': 5, 'task': hs.TASK,
        'obs_terms': [list(x) for x in hs.OBS_TERMS], 'history': hs.HISTORY,
        'task_state_dim': 6, 'critic_dim': 80,
        'reward_weights': asdict(hs.HandstandRewards()),
        'stage_schedule_s': [2., 4., 6., 8., 10.],
        'training_stage_reset': {'walk': .5, 'front_support_at_4s': .25,
                                 'recovery_at_6s': .25},
        'reference_action': [round(float(x), 6) for x in reference['action']],
        'actor': 'frozen 235-input speed walk branch; 250-input normalized handstand head; stage gate blends normalized physical joint targets',
        'world_model_action': 'both action inputs converted to legacy trot joint-offset units',
        'walking_source_action_scale': [walking['joints'][n]['action_scale'] for n in SDK_JOINT_ORDER],
        'termination': 'body collision-box bottom <.045 m, base z <.17 m, or projected-gravity lateral magnitude >.6',
    }


def defaults():
    return HandstandTrainCfg(task=hs.TASK, robot_config='configs/x30/robot.handstand.json',
                            num_envs=64, buffer_len=600, wm_hidden=(256, 256),
                            wm_window=16, wm_batch=128, wm_updates=2,
                            archive_size=5000, imag_batch=64, horizon=32,
                            iterations=30000, warmup_iterations=1000,
                            action_noise=.08, explore_noise=.2,
                            compile=False, save_interval=5000, log_interval=50,
                            log_root='logs/x30_handstand', run_name='x30_front_handstand')


class HandstandTrainer(TrotTrainer):
    task_id = hs.TASK
    policy_module = hs
    task_contract = staticmethod(contract)
    register_task = staticmethod(register_handstand_task)
    task_state_dim = 6

    def __init__(self, cfg):
        if cfg.task != hs.TASK or (cfg.init_from and cfg.resume):
            raise ValueError('Choose the handstand task and only one of init-from/resume')
        if cfg.resume and not cfg.continue_run:
            raise ValueError('Handstand resume requires --continue-run true')
        self.provenance = None
        try:
            super().__init__(cfg)
            if cfg.init_from:
                checkpoint = torch.load(cfg.init_from, map_location=self.dev, weights_only=False)
                parameters = load_parameters(cfg.robot_config)
                self.provenance = transfer_from_speed(checkpoint, self.actor, self.wm, parameters)
                path = Path(cfg.init_from)
                self.provenance.update(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                self.actor_frozen_until = max(self.actor_frozen_until,
                                              cfg.warmup_iterations + cfg.critic_warmup_iterations)
                print(f'Initialized frozen walk and world model from {cfg.init_from}; handstand head/critic/replay fresh',
                      flush=True)
            (self.log_dir / 'initialization.json').write_text(json.dumps(self.provenance, indent=2) + '\n')
        except BaseException:
            if hasattr(self, 'env'):
                self.env.close()
            if hasattr(self, 'writer'):
                self.writer.close()
            raise

    def _make_actor(self, parameters):
        reference = find_pose(parameters, angle_deg=40.)
        if not reference['geometry_passed']:
            raise ValueError('Handstand pose failed geometry check')
        return HandstandActor(self.cfg.actor_hidden, _walking_parameters(), parameters, reference['action'])

    def _make_world_model(self, parameters):
        from go2_supermimic.robots.x30.locomotion import ActionMappedWorldModel
        return ActionMappedWorldModel(self.cfg.wm_hidden, 1 / action_gain(parameters))

    def _update_actor_norm(self):
        self.actor.update_norm(self.obs)

    def _task_state(self, command):
        cache = getattr(self.env, '_trot_reward_cache', None)
        self._pre_reward_valid = (cache['valid'].clone() if cache is not None else
                                  torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.dev))
        age = effective_age(self.env).unsqueeze(-1)
        return torch.cat([command, phase_observation(self.env), age], -1)

    def _make_imagination(self, parameters, limits):
        terms = self.env_cfg.rewards['handstand'].params
        return HandstandImagination(ImagineCfg(horizon=self.cfg.horizon, dt=self.env.step_dt,
                                              compile=self.cfg.compile),
                                     self.consts, limits, terms['weights'],
                                     terms['gait'], terms['foot_radii'])

    def _record_task_stats(self):
        self.reward_valid[:, (self.buf.ptr - 1) % self.buf.L] = self._pre_reward_valid
        terms = getattr(self.env, '_handstand_reward_terms', None)
        if terms is None:
            return
        for name, contribution in hs.reward_contributions(terms, self.imag.weights).items():
            self.ep_stats[f'Metrics/handstand/reward_{name}'].append(float(contribution))

    def save(self, it):
        path = super().save(it)
        checkpoint = torch.load(path, map_location=self.dev, weights_only=False)
        checkpoint['handstand_provenance'] = self.provenance
        checkpoint['handstand_training_state'] = {
            'actor_frozen_until': self.actor_frozen_until,
            'policy_start': self.policy_start,
        }
        tmp = path.with_suffix('.tmp')
        torch.save(checkpoint, tmp)
        tmp.replace(path)
        return path

    def load(self, path):
        """Exact same-task resume; the generic loader assumes a `norm.mean` actor."""
        checkpoint = torch.load(path, map_location=self.dev, weights_only=False)
        if checkpoint.get('contract') != self.contract:
            raise ValueError('Handstand resume requires an identical physical and task contract')
        validate_fixed_buffers(checkpoint, self.actor, load_parameters(self.cfg.robot_config))
        for key, module in (('actor', self.actor), ('critic', self.critic),
                            ('world_model', self.wm)):
            saved = checkpoint[key]
            current = module.state_dict()
            if set(saved) != set(current):
                raise ValueError(f'Handstand resume {key} keys differ')
            for name, value in saved.items():
                if value.shape != current[name].shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'Invalid handstand {key}.{name}')
        for key, optimizer in (('opt_actor', self.opt_actor), ('opt_critic', self.opt_critic),
                               ('opt_wm', self.opt_wm)):
            validate_optimizer_state(checkpoint[key], optimizer)
        for key, module in (('actor', self.actor), ('critic', self.critic),
                            ('world_model', self.wm)):
            module.load_state_dict(checkpoint[key])
        for key, optimizer in (('opt_actor', self.opt_actor), ('opt_critic', self.opt_critic),
                               ('opt_wm', self.opt_wm)):
            optimizer.load_state_dict(checkpoint[key])
        guard = checkpoint.get('handstand_training_state', {})
        self.actor_frozen_until = int(guard.get(
            'actor_frozen_until', checkpoint['cfg']['warmup_iterations'] +
            checkpoint['cfg']['critic_warmup_iterations']))
        self.policy_start = int(guard.get('policy_start', checkpoint['cfg']['warmup_iterations']))
        self.provenance = checkpoint.get('handstand_provenance')
        self.start_it = int(checkpoint['iteration'])
        self.env.common_step_counter = self.start_it
        print(f'Resumed handstand from {path} at iteration {self.start_it}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train X30 front-foot handstand and walk transitions')
    for name, value in asdict(defaults()).items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, tuple):
            parser.add_argument(flag, type=int, nargs='+', default=value)
        elif isinstance(value, bool):
            parser.add_argument(flag, type=lambda v: v.lower() in ('true', '1'), default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args = vars(parser.parse_args())
    cfg = HandstandTrainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()})
    torch.set_float32_matmul_precision('high')
    trainer = HandstandTrainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()
        trainer.writer.close()
