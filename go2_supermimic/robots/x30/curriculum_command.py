"""Command sampling and closed-loop 180-degree turns."""
from dataclasses import dataclass
import math
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand,UniformVelocityCommandCfg
from go2_supermimic.robots.x30.curriculum import Curriculum


def wrap(angle):return torch.atan2(angle.sin(),angle.cos())


def heading_control(remaining,yaw_cap,resume_speed):
    complete=remaining.abs()<=.08
    command=torch.zeros(len(remaining),3,device=remaining.device)
    command[:,0]=torch.where(complete,resume_speed,0.)
    command[:,2]=torch.where(complete,0.,(1.5*remaining).clamp(-yaw_cap,yaw_cap))
    return command,complete


class CurriculumCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.curriculum=Curriculum()
        self.generator=torch.Generator(device=self.device).manual_seed((env.cfg.seed or 0)+1729)
        self.mode=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.remaining=torch.zeros(self.num_envs,device=self.device)
        self.last_heading=torch.zeros_like(self.remaining);self.yaw_cap=torch.ones_like(self.remaining)*.2
        self.completed=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.timeouts=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.turn_start=torch.zeros_like(self.remaining);self.timed_out=torch.zeros_like(self.completed)
        self.heading_initialized=torch.zeros_like(self.completed)
        self.override=None

    def _resample_command(self,env_ids):
        if self.cfg.evaluation:
            self.vel_command_b[env_ids]=0.;self.time_left[env_ids]=1e9;self.mode[env_ids]=0
            return
        self.timeouts[env_ids]+=((self.mode[env_ids]==6)&(~self.completed[env_ids])).long()
        command,modes=self.curriculum.sample(len(env_ids),self.generator,self.device)
        self.vel_command_b[env_ids]=command;self.mode[env_ids]=modes
        duration=3+3*torch.rand(len(env_ids),device=self.device,generator=self.generator)
        self.time_left[env_ids]=duration
        ids=env_ids[modes==6]
        if len(ids):self.begin_uturn(ids,command[modes==6,2])

    def begin_uturn(self,ids,signed_cap):
        self.mode[ids]=6;self.completed[ids]=False
        self.timed_out[ids]=False;self.turn_start[ids]=self._env.common_step_counter*self._env.step_dt
        self.heading_initialized[ids]=False
        self.remaining[ids]=torch.where(signed_cap>=0,math.pi,-math.pi)
        self.yaw_cap[ids]=signed_cap.abs().clamp(min=.2)
        # Reset commands run before forward(); capture fresh derived yaw on update.
        self.time_left[ids]=1e9 if self.cfg.evaluation else math.pi/self.yaw_cap[ids]+6
        self.vel_command_b[ids]=0.
        self.vel_command_b[ids,2]=signed_cap

    def _update_command(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        if self.override is not None:
            direct=ids[self.mode[ids]!=6]
            self.vel_command_b[direct]=self.override[direct]
        ids=ids[self.mode[ids]==6]
        if len(ids):
            heading=self.robot.data.heading_w[ids]
            fresh=~self.heading_initialized[ids]
            self.last_heading[ids[fresh]]=heading[fresh];self.heading_initialized[ids]=True
            self.remaining[ids]-=wrap(heading-self.last_heading[ids]);self.last_heading[ids]=heading
            speed=torch.full_like(self.remaining[ids],min(.3,self.curriculum.caps()['forward']))
            cmd,done=heading_control(self.remaining[ids],self.yaw_cap[ids],speed)
            elapsed=self._env.common_step_counter*self._env.step_dt-self.turn_start[ids]
            in_time=elapsed<=math.pi/self.yaw_cap[ids]+6
            self.completed[ids]|=done&in_time&(~self.timed_out[ids])
            # Once completed, keep travelling instead of re-entering the turn on drift.
            cmd[self.completed[ids],0]=speed[self.completed[ids]];cmd[self.completed[ids],2]=0.
            expired=(~in_time)&(~self.completed[ids])
            self.timed_out[ids]|=expired
            cmd[self.timed_out[ids]]=0.
            self.vel_command_b[ids]=cmd

    def reset(self,env_ids):
        self.mode[env_ids]=0;self.completed[env_ids]=False
        return super().reset(env_ids)


@dataclass(kw_only=True)
class CurriculumCommandCfg(UniformVelocityCommandCfg):
    evaluation:bool=False

    def build(self,env):return CurriculumCommand(self,env)
