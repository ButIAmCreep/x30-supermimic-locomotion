"""Numerical and live evaluation of the X30 speed-2 course."""

from go2_supermimic.robots.x30 import locomotion_eval
from go2_supermimic.robots.x30.speed_curriculum import SPEED_LEVELS
from go2_supermimic.robots.x30.speed_env import TASK, speed_env_cfg
from go2_supermimic.robots.x30.speed_train import contract


def prepare(checkpoint, command, num_envs, seed, **kwargs):
    return locomotion_eval.prepare(checkpoint, command, num_envs, seed,
                                   task=TASK, contract_fn=contract, env_fn=speed_env_cfg,
                                   levels=SPEED_LEVELS, **kwargs)


def main(live=False):
    locomotion_eval.main(live, task=TASK, contract_fn=contract,
                         env_fn=speed_env_cfg, levels=SPEED_LEVELS)
