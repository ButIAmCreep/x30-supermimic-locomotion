"""Phase-conditioned diagonal gait: shared simulator/imagination math in SI units."""
from dataclasses import dataclass
import math
import torch
from go2_supermimic.robots.x30 import policy as vel
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import Imagination
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS

TASK='X30-Trot-Flat'
HISTORY=5
OBS_TERMS=vel.OBS_TERMS+(('gait_phase',2),)
OBS_DIM=sum(dim for _,dim in OBS_TERMS)*HISTORY
robot_state=vel.robot_state

@dataclass(frozen=True)
class GaitCfg:
    period: float=.6
    duty: float=.6
    clearance: float=.04
    contact_height: float=.005
    contact_softness: float=.003
    phase_softness: float=.1
    stop_threshold: float=.05
    base_height: float=.50
    penetration_tolerance: float=.012

@dataclass
class TrotRewards(vel.VelocityRewards):
    linear: float=2.
    angular: float=.5
    linear_sigma: float=.2
    contact_phase: float=-2.
    foot_height: float=-1.
    flight: float=-2.
    stance_slip: float=-.1
    vertical_velocity: float=-1.5
    tilt_rate: float=-.05
    base_height: float=-.5
    penetration: float=-.2


def phase_from_steps(steps,dt,gait=GaitCfg()):
    angle=torch.remainder(steps.float()*dt,gait.period)*(2*math.pi/gait.period)
    return torch.stack([angle.sin(),angle.cos()],-1)


def advance_phase(phase,dt,gait=GaitCfg()):
    angle=2*math.pi*dt/gait.period
    sine,cosine=math.sin(angle),math.cos(angle)
    return torch.stack([phase[:,0]*cosine+phase[:,1]*sine,
                        phase[:,1]*cosine-phase[:,0]*sine],-1)


def gait_targets(phase,command,gait=GaitCfg()):
    fraction=torch.remainder(torch.atan2(phase[:,0],phase[:,1])/(2*math.pi),1.)
    offsets=phase.new_tensor([0.,.5,.5,0.])  # FL, FR, HL, HR
    u=torch.remainder(fraction[:,None]+offsets,1.)
    stance=torch.sigmoid(((2*math.pi*(u-gait.duty/2)).cos()-math.cos(math.pi*gait.duty))/gait.phase_softness)
    swing=((u-gait.duty)/(1-gait.duty)).clamp(0,1)
    height=gait.clearance*torch.sin(math.pi*swing)
    moving=command.square().sum(-1)>gait.stop_threshold**2
    return torch.where(moving[:,None],stance,1.),torch.where(moving[:,None],height,0.)


def obs_frame(s,action,command,phase):
    return vel.obs_frame(s,action,command)|{'gait_phase':phase}


def foot_world_velocity(s0,s1,pose_delta,dt):
    """Foot-center world displacement / dt in the previous yaw frame (B,4,2)."""
    old=s0[:,st.FEET].reshape(-1,4,3)[...,:2]
    new=s1[:,st.FEET].reshape(-1,4,3)[...,:2]
    c=pose_delta[:,2].cos()[:,None];sn=pose_delta[:,2].sin()[:,None]
    rotated=torch.stack([c*new[...,0]-sn*new[...,1],sn*new[...,0]+c*new[...,1]],-1)
    return (pose_delta[:,:2,None].transpose(1,2)+rotated-old)/dt


def rewards(s0,s1,pose_delta,action,previous_action,command,phase1,c,velocity_limits,
            failure,weights,gait=GaitCfg(),foot_radii=(.036,)*4,valid=None,dt=.02):
    _,terms=vel.rewards(s1,command,action,previous_action,c,velocity_limits,failure,weights)
    radii=torch.as_tensor(foot_radii,device=s1.device,dtype=s1.dtype)
    height=st.feet_world_z(s1)-radii
    stance,target_height=gait_targets(phase1,command,gait)
    contact=torch.sigmoid((gait.contact_height-height)/gait.contact_softness)
    velocity=foot_world_velocity(s0,s1,pose_delta,dt)
    slip=(stance*(velocity/.2).square().sum(-1)).mean(-1)
    if valid is not None:
        slip=slip*valid.to(s1.dtype)
    terms.update({
        'contact_phase':(contact-stance).square().mean(-1),
        'foot_height':((height-target_height)/gait.clearance).square().mean(-1),
        'flight':(1-contact).prod(-1),
        'stance_slip':slip,
        'vertical_velocity':s1[:,6].square(),
        'tilt_rate':s1[:,7:9].square().sum(-1),
        'base_height':((s1[:,0]-gait.base_height)/.05).square(),
        'penetration':((-gait.penetration_tolerance-height).clamp(min=0)/.01).square().mean(-1),
    })
    return sum(getattr(weights,name)*value for name,value in terms.items()),terms


class TrotImagination(Imagination):
    def __init__(self,cfg,consts,velocity_limits,min_height,weights,gait=GaitCfg(),foot_radii=(.036,)*4):
        self.cfg,self.c=cfg,consts
        self.velocity_limits,self.min_height,self.weights=velocity_limits,min_height,weights
        self.gait,self.foot_radii=gait,tuple(foot_radii)
        self.critic_dim=st.STATE_DIM+5+st.CONTEXT_DIM
        self._step_fn=torch.compile(self._step_trot,dynamic=False) if cfg.compile else self._step_trot

    def _step_trot(self,wm,actor,s,hist,previous,command,phase,xi,c,valid):
        a=actor(hist)
        relief=torch.zeros(len(s),SCAN_POINTS,device=s.device)
        s1,dp=wm(s,a,previous,xi,relief)
        phase1=advance_phase(phase,self.cfg.dt,self.gait)
        failure=vel.failure_probability(s1,self.min_height)
        reward,_=rewards(s,s1,dp,a,previous,command,phase1,c,self.velocity_limits,failure,
                         self.weights,self.gait,self.foot_radii,valid=valid,dt=self.cfg.dt)
        hist1=st.push_history(hist,obs_frame(s1,a,command,phase1),HISTORY,OBS_TERMS)
        return s1,hist1,a,phase1,reward*self.cfg.dt,failure

    def rollout(self,wm,actor,critic,seed,hint_weight=0.):
        ids=seed['env'];cfg=self.cfg
        c=st.RobotConsts(self.c.default_q[ids],self.c.action_scale[ids],self.c.kp,self.c.kd,
                        self.c.effort,self.c.soft_limits[ids])
        s,hist,previous=seed['s'],seed['obs'],seed['a_prev']
        command,phase=seed['gen'][:,:3],seed['gen'][:,3:5]
        survival=torch.ones(len(s),device=s.device);total=torch.zeros_like(survival)
        discount=1.;xs=[];rs=[];ss=[]
        valid=seed.get("reward_valid",torch.ones(len(s),dtype=torch.bool,device=s.device))
        for _ in range(cfg.horizon):
            xs.append(torch.cat([s,command,phase,seed['xi']],-1));ss.append(survival)
            s,hist,previous,phase,r,failure=self._step_fn(wm,actor,s,hist,previous,command,phase,seed['xi'],c,valid)
            valid=torch.ones_like(valid)
            total=total+discount*survival*r
            survival=survival*(1-failure);discount*=cfg.gamma;rs.append(r)
        xs.append(torch.cat([s,command,phase,seed['xi']],-1));ss.append(survival)
        objective=total+discount*survival*critic(xs[-1]).squeeze(-1)
        return objective,{'rewards':torch.stack(rs,1),'surv':torch.stack(ss,1),'critic_in':torch.stack(xs,1)}
