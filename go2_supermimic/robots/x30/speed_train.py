"""Resume-safe second-course training from the 50k locomotion policy."""

from dataclasses import asdict, replace
import argparse
import torch

from go2_supermimic.robots.x30.curriculum import MODE_PROBABILITIES
from go2_supermimic.robots.x30.locomotion_train import (
    CurriculumTrainCfg, LocomotionTrainer, contract as old_contract, defaults as old_defaults,
)
from go2_supermimic.robots.x30.speed_curriculum import (
    SpeedCurriculum, SPEED_LEVELS, SPEED_MODE_PROBABILITIES,
)
from go2_supermimic.robots.x30.speed_env import TASK, register_speed_task, speed_env_cfg
from go2_supermimic.robots.x30.speed_transfer import transfer_from_curriculum


def contract(parameters):
    result = old_contract(parameters)
    result.update(task=TASK, version=4,
                  levels={k: list(v) for k, v in SPEED_LEVELS.items()},
                  mode_probabilities=list(SPEED_MODE_PROBABILITIES),
                  curriculum_version=2, evaluation_version=2)
    return result


def defaults():
    return replace(old_defaults(), task=TASK, iterations=40000,
                   log_root='logs/x30_speed2', run_name='x30_speed2',
                   evaluation_interval=1000, save_interval=5000)


class SpeedTrainer(LocomotionTrainer):
    task_id = TASK
    task_contract = staticmethod(contract)
    register_task = staticmethod(register_speed_task)
    curriculum_type = SpeedCurriculum
    initialization_transfer = staticmethod(transfer_from_curriculum)
    evaluation_env_factory = staticmethod(speed_env_cfg)

    def _after_init_transfer(self, ckpt):
        try:
            course = SpeedCurriculum.from_prior(ckpt['curriculum']['controller'])
        except (KeyError, TypeError) as exc:
            raise ValueError('Source checkpoint lacks a valid prior curriculum') from exc
        self.term.curriculum = course
        self.provenance['mastered_caps_inherited'] = course.caps()


def main():
    ap = argparse.ArgumentParser(description='X30 lateral 2 m/s and yaw 2 rad/s course')
    for name, value in asdict(defaults()).items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, tuple):
            ap.add_argument(flag, type=int, nargs='+', default=value)
        elif isinstance(value, bool):
            ap.add_argument(flag, type=lambda v: v.lower() in ('true', '1'), default=value)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    args = vars(ap.parse_args())
    cfg = CurriculumTrainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()})
    torch.set_float32_matmul_precision('high')
    trainer = SpeedTrainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()
        trainer.writer.close()
