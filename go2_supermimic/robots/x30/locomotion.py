"""Omnidirectional residual policy and command-conditioned diagonal gait."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from go2_supermimic.robots.x30 import trot,policy as vel
from go2_supermimic.robots.x30.train import VelocityActor
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.world_model import WorldModel,mlp
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS

TASK='X30-Locomotion-Curriculum'
HISTORY,OBS_TERMS,OBS_DIM=trot.HISTORY,trot.OBS_TERMS,trot.OBS_DIM
robot_state,obs_frame=trot.robot_state,trot.obs_frame


class LocomotionActor(VelocityActor):
    """Zero residual preserves old physical targets, including action histories."""
    def __init__(self,obs_dim,hidden,act_dim,gain):
        super().__init__(obs_dim,hidden,act_dim)
        self.register_buffer('action_gain',torch.as_tensor(gain).float().clone())
        history_gain=torch.ones(obs_dim);offset=0
        for name,dim in OBS_TERMS:
            if name=='actions':history_gain[offset:offset+dim*HISTORY]=1/self.action_gain.repeat(HISTORY)
            offset+=dim*HISTORY
        self.register_buffer('history_gain',history_gain)
        self.residual=mlp(obs_dim,(128,64),act_dim,act=nn.ELU)
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def base_observation(self,obs):return obs*self.history_gain

    def forward(self,obs):
        x=self.norm(self.base_observation(obs))
        return self.action_gain*self.net(x).tanh()+(1-self.action_gain)*self.residual(x).tanh()


class ActionMappedWorldModel(WorldModel):
    """Keep old physical action units inside the learned dynamics model."""
    def __init__(self,hidden,action_ratio):
        super().__init__(hidden)
        self.register_buffer('action_ratio',torch.as_tensor(action_ratio).float().clone())

    def _inputs(self,s,a,a_prev,xi,relief):
        return super()._inputs(s,a*self.action_ratio,a_prev*self.action_ratio,xi,relief)


@dataclass
class LocomotionRewards(trot.TrotRewards):
    stop_motion:float=-2.


def profile(command):
    speed=torch.stack([command[:,0],1.5*command[:,1]],-1).norm(dim=-1)+.35*command[:,2].abs()
    blend=((speed-.5)/2.5).clamp(0,1)
    frequency=1/.6+1.2*(speed-.5).clamp(0,2.5)
    return frequency,.6-.2*blend,.04+.05*blend


def advance_phase(phase,command,dt,gait=None):
    angle=2*math.pi*profile(command)[0]*dt
    return torch.stack([phase[:,0]*angle.cos()+phase[:,1]*angle.sin(),
                        phase[:,1]*angle.cos()-phase[:,0]*angle.sin()],-1)


def gait_targets(phase,command,gait=trot.GaitCfg()):
    _,duty,clearance=profile(command)
    fraction=torch.remainder(torch.atan2(phase[:,0],phase[:,1])/(2*math.pi),1.)
    u=torch.remainder(fraction[:,None]+phase.new_tensor([0.,.5,.5,0.]),1.)
    duty=duty[:,None]
    stance=torch.sigmoid(((2*math.pi*(u-duty/2)).cos()-torch.cos(math.pi*duty))/gait.phase_softness)
    height=clearance[:,None]*torch.sin(math.pi*((u-duty)/(1-duty)).clamp(0,1))
    moving=command.square().sum(-1)>gait.stop_threshold**2
    return torch.where(moving[:,None],stance,1.),torch.where(moving[:,None],height,0.)


def unscheduled_flight(contact,stance):return (1-contact).prod(-1)*stance.max(-1).values


def rewards(s0,s1,pose_delta,action,previous_action,command,phase1,c,velocity_limits,
            failure,weights,gait=trot.GaitCfg(),foot_radii=(.036,)*4,valid=None,dt=.02):
    _,terms=vel.rewards(s1,command,action,previous_action,c,velocity_limits,failure,weights)
    speed=command[:,:2].norm(dim=-1)
    terms['linear']=torch.exp(-(s1[:,st.V_YAW][:,:2]-command[:,:2]).square().sum(-1)/(.2+.12*speed).square())
    terms['angular']=torch.exp(-(s1[:,9]-command[:,2]).square()/(.3+.1*command[:,2].abs()).square())
    height=st.feet_world_z(s1)-s1.new_tensor(foot_radii)
    stance,target_height=gait_targets(phase1,command,gait)
    contact=torch.sigmoid((gait.contact_height-height)/gait.contact_softness)
    slip=(stance*(trot.foot_world_velocity(s0,s1,pose_delta,dt)/.2).square().sum(-1)).mean(-1)
    if valid is not None:slip=slip*valid.to(s1.dtype)
    clearance=profile(command)[2]
    stop=command.square().sum(-1)<gait.stop_threshold**2
    terms.update(contact_phase=(contact-stance).square().mean(-1),
        foot_height=((height-target_height)/clearance[:,None]).square().mean(-1),
        flight=unscheduled_flight(contact,stance),stance_slip=slip,
        vertical_velocity=s1[:,6].square(),tilt_rate=s1[:,7:9].square().sum(-1),
        base_height=((s1[:,0]-gait.base_height)/.05).square(),
        penetration=((-gait.penetration_tolerance-height).clamp(min=0)/.01).square().mean(-1),
        stop_motion=stop*(s1[:,4:6].square().sum(-1)+.25*s1[:,9].square()))
    return sum(getattr(weights,name)*value for name,value in terms.items()),terms


class LocomotionImagination(trot.TrotImagination):
    def _step_trot(self,wm,actor,s,hist,previous,command,phase,xi,c,valid):
        a=actor(hist);relief=torch.zeros(len(s),SCAN_POINTS,device=s.device)
        s1,dp=wm(s,a,previous,xi,relief)
        phase1=advance_phase(phase,command,self.cfg.dt)
        failure=vel.failure_probability(s1,self.min_height)
        reward,_=rewards(s,s1,dp,a,previous,command,phase1,c,self.velocity_limits,failure,
                         self.weights,self.gait,self.foot_radii,valid=valid,dt=self.cfg.dt)
        hist1=st.push_history(hist,obs_frame(s1,a,command,phase1),HISTORY,OBS_TERMS)
        return s1,hist1,a,phase1,reward*self.cfg.dt,failure
