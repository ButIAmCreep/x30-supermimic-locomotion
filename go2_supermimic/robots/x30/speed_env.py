"""Independent environment and command term for the wider speed course."""

from dataclasses import dataclass

from go2_supermimic.robots.x30 import speed_curriculum
from go2_supermimic.robots.x30.curriculum_command import CurriculumCommand, CurriculumCommandCfg
from go2_supermimic.robots.x30.locomotion_env import locomotion_env_cfg


TASK = 'X30-Locomotion-Speed2'


class SpeedCommand(CurriculumCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.curriculum = speed_curriculum.SpeedCurriculum()


@dataclass(kw_only=True)
class SpeedCommandCfg(CurriculumCommandCfg):
    def build(self, env):
        return SpeedCommand(self, env)


def speed_env_cfg(parameters, play=False, evaluation=False):
    cfg = locomotion_env_cfg(parameters, play=play, evaluation=evaluation)
    cfg.commands['twist'] = SpeedCommandCfg(
        entity_name='robot', resampling_time_range=(3., 6.), heading_command=False,
        rel_heading_envs=0., evaluation=evaluation,
        ranges=SpeedCommandCfg.Ranges(lin_vel_x=(-.6, 3.),
                                      lin_vel_y=(-2., 2.), ang_vel_z=(-2., 2.)))
    return cfg


def register_speed_task(parameters):
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if TASK in list_tasks():
        raise ValueError('X30 speed task is already registered')
    register_mjlab_task(TASK, speed_env_cfg(parameters), speed_env_cfg(parameters, play=True),
                        RslRlOnPolicyRunnerCfg())
