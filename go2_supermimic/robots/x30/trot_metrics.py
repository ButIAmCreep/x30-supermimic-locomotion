"""Independent solver-contact gait metrics and held-out physical-model errors."""
import math
import numpy as np
import torch
import warp as wp
import mujoco
from go2_supermimic.robots.x30 import trot
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS


def contact_point_velocity(cvel,position,subtree_com):
    return cvel[...,3:]+np.cross(cvel[...,:3],position-subtree_com)


class GaitMetrics:
    """Observe existing physics at 200 Hz; restore the step method on close."""
    def __init__(self,env,discard_seconds=2.,phase_targets=None,sample_callback=None):
        self.env,self.discard=env,discard_seconds
        self.gait=trot.GaitCfg();self.original=env.sim.step
        self.phase_targets=phase_targets;self.sample_callback=sample_callback
        robot=env.scene['robot'];m=env.sim.mj_model
        local,_=robot.find_geoms(tuple(f'{leg}_foot_collision' for leg in ('FL','FR','HL','HR')),preserve_order=True)
        self.feet=robot.indexing.geom_ids[local].cpu().numpy()
        self.bodies=m.geom_bodyid[self.feet];self.root=int(robot.indexing.root_body_id)
        self.floor=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)
        wd=env.sim.wp_data
        self.nacon=wp.to_torch(wd.nacon);self.geom=wp.to_torch(wd.contact.geom)
        self.efc=wp.to_torch(wd.contact.efc_address);self.world=wp.to_torch(wd.contact.worldid)
        self.pos=wp.to_torch(wd.contact.pos)
        self.samples=self.correct=self.labels=self.flight=self.pair_samples=self.pair_correct=0
        self.vz2=self.slip_sum=0.;self.slip_count=0
        self.sync=np.zeros(2);self.counts=np.zeros(5)
        self.z_min=np.full(env.num_envs,np.inf);self.z_max=np.full(env.num_envs,-np.inf)
        self.trace=[];self.ticks=0
        env.sim.step=self._step

    def _step(self):
        self.original();env=self.env;self.ticks+=1
        nc=int(self.nacon[0])
        if nc>len(self.geom):raise RuntimeError('Contact buffer overflow invalidates gait metrics')
        g=self.geom[:nc].cpu().numpy();w=self.world[:nc].cpu().numpy()
        active=(self.efc[:nc,0]>=0).cpu().numpy()
        found=np.zeros((env.num_envs,4),dtype=bool);pairs=[]
        for i,fid in enumerate(self.feet):
            match=(((g[:,0]==fid)&np.isin(g[:,1],self.floor))|((g[:,1]==fid)&np.isin(g[:,0],self.floor)))&active
            rows=np.flatnonzero(match)
            if len(rows):found[w[rows],i]=True;pairs.append((i,rows))
        sub=(env._sim_step_counter-1)%env.cfg.decimation+1
        age=env.episode_length_buf.cpu().numpy()*env.step_dt+sub*env.physics_dt
        valid=age>self.discard
        if self.phase_targets is None:
            phase=np.remainder(age/self.gait.period,1.)[:,None]
            duty=self.gait.duty;period=self.gait.period
        else:
            fraction,duties,periods=self.phase_targets(env,sub)
            phase=fraction[:,None];duty=duties[:,None];period=periods[:,None]
        u=np.remainder(phase+np.array([0.,.5,.5,0.]),1.)
        expected=u<duty
        command=env.command_manager.get_term('twist').command.cpu().numpy()
        moving=np.linalg.norm(command,axis=1)>self.gait.stop_threshold
        expected[~moving]=True
        distance=np.minimum(np.minimum(u,1-u),np.abs(u-duty))
        label_valid=(distance*period>.02)|(~moving[:,None])
        label_valid&=valid[:,None]
        if self.sample_callback is not None:self.sample_callback(found,expected,label_valid,valid)
        self.labels+=int(label_valid.sum());self.correct+=int(((found==expected)&label_valid).sum())
        nfeet=found.sum(1);self.samples+=int(valid.sum())
        self.flight+=int(((nfeet==0)&valid).sum())
        for k in range(5):self.counts[k]+=int(((nfeet==k)&valid).sum())
        self.sync+=np.array([((found[:,0]==found[:,3])&valid).sum(),((found[:,1]==found[:,2])&valid).sum()])
        pair_mask=valid&moving&(expected.sum(1)==2)&label_valid.all(1)
        self.pair_samples+=int(pair_mask.sum())
        self.pair_correct+=int(((found==expected).all(1)&pair_mask).sum())
        z=env.sim.data.qpos[:,2].cpu().numpy();vz=env.sim.data.qvel[:,2].cpu().numpy()
        self.z_min=np.minimum(self.z_min,np.where(valid,z,np.inf))
        self.z_max=np.maximum(self.z_max,np.where(valid,z,-np.inf));self.vz2+=float((vz[valid]**2).sum())
        if pairs:
            cvel=env.sim.data.cvel[:,self.bodies.tolist()].cpu().numpy()
            com=env.sim.data.subtree_com[:,self.root].cpu().numpy();points=self.pos[:nc].cpu().numpy()
            for i,rows in pairs:
                rows=rows[valid[w[rows]]]
                if len(rows):
                    ids=w[rows];v=contact_point_velocity(cvel[ids,i],points[rows],com[ids])
                    speeds=np.linalg.norm(v[:,:2],axis=1)
                    self.slip_sum+=float(speeds.sum());self.slip_count+=len(rows)
        if len(self.trace)<12000:self.trace.append((self.ticks*env.physics_dt,found[0].copy(),float(z[0])))

    def summary(self):
        n=self.samples
        spans=self.z_max-self.z_min;spans=spans[np.isfinite(spans)]
        return {'physics_samples_after_discard':n,'discard_seconds':self.discard,
            'contact_source':'existing foot/plane solver constraints; efc_address>=0; not normal-force thresholded',
            'sampling_hz':1/self.env.physics_dt,'phase_boundary_exclusion_ms':20.,
            'phase_agreement':self.correct/self.labels if self.labels else None,
            'alternating_diagonal_support_fraction':self.pair_correct/self.pair_samples if self.pair_samples else None,
            'diagonal_synchrony':(self.sync/n).tolist() if n else None,
            'all_feet_air_fraction':self.flight/n if n else None,
            'contact_count_fraction':(self.counts/n).tolist() if n else None,
            'base_height_peak_to_peak_mm_per_env':(spans*1000).tolist(),
            'vertical_velocity_rms_m_s':math.sqrt(self.vz2/n) if n else None,
            'contact_point_tangential_speed_mean_m_s':self.slip_sum/self.slip_count if self.slip_count else None,
            'kinematics_note':'Derived body/contact quantities may lag integration by one 5 ms substep.'}

    def close(self):self.env.sim.step=self.original

    def plot(self,path):
        if not self.trace:return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        times=np.array([x[0] for x in self.trace]);contacts=np.stack([x[1] for x in self.trace]);z=np.array([x[2] for x in self.trace])
        fig,axes=plt.subplots(2,1,sharex=True,figsize=(11,5))
        axes[0].imshow(contacts.T,origin='lower',aspect='auto',interpolation='nearest',
            extent=[times[0],times[-1],-.5,3.5],cmap='Greys',vmin=0,vmax=1)
        axes[0].set_yticks(range(4),['FL','FR','HL','HR']);axes[0].set_ylabel('Solver contact')
        axes[1].plot(times,z);axes[1].set_ylabel('Base height (m)');axes[1].set_xlabel('Simulation time (s)')
        axes[1].grid(alpha=.3);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


@torch.no_grad()
def world_model_errors(checkpoint,transitions,dt,discard_seconds,world_model=None):
    if not transitions:return {'status':'no transitions'}
    states,actions,previous,targets,deltas,dones,ages=[torch.stack([row[k] for row in transitions]) for k in range(7)]
    T,N=actions.shape[:2];device=actions.device
    wm=world_model
    if wm is None:
        wm=WorldModel(tuple(checkpoint['cfg']['wm_hidden'])).to(device)
        wm.load_state_dict(checkpoint['world_model'])
    wm.eval();result={}
    for h in [1,8,16]:
        if T<h:continue
        starts=torch.arange(0,T-h+1,4,device=device).repeat_interleave(N)
        ids=torch.arange(N,device=device).repeat(len(starts)//N)
        keep=ages[starts,ids]>=discard_seconds;starts,ids=starts[keep],ids[keep]
        if len(starts)>512:
            indices=torch.linspace(0,len(starts)-1,512,device=device).long();starts,ids=starts[indices],ids[indices]
        if not len(starts):continue
        pred=states[starts,ids].clone();initial=pred.clone();valid=torch.ones(len(starts),dtype=torch.bool,device=device)
        pp=torch.zeros(len(starts),3,device=device);tp=pp.clone()
        xi=torch.zeros(len(starts),st.CONTEXT_DIM,device=device);xi[:,0]=1.
        relief=torch.zeros(len(starts),SCAN_POINTS,device=device)
        delta_xy_errors=[];delta_yaw_errors=[]
        for j in range(h):
            pred,pd=wm(pred,actions[starts+j,ids],previous[starts+j,ids],xi,relief)
            actual_delta=deltas[starts+j,ids]
            delta_xy_errors.append(pd[:,:2]-actual_delta[:,:2])
            yaw_error=pd[:,2]-actual_delta[:,2]
            delta_yaw_errors.append(torch.atan2(yaw_error.sin(),yaw_error.cos()))
            pp=st.apply_pose_delta(pp,pd);tp=st.apply_pose_delta(tp,actual_delta)
            valid&=~dones[starts+j,ids]
        if not valid.any():continue
        truth=targets[starts[valid]+h-1,ids[valid]];p=pred[valid];base=initial[valid]
        finite=bool(torch.isfinite(p).all() and torch.isfinite(pp[valid]).all())
        row={'windows':int(valid.sum()),'finite':finite}
        if finite:
            rmse=lambda x:float(x.square().mean().sqrt())
            for name,sl,scale in [('base_height_mm',st.Z,1000.),('linear_velocity_m_s',st.V_YAW,1.),('joint_angle_deg',st.Q,180/math.pi)]:
                row[name]={'rmse':rmse((p[:,sl]-truth[:,sl])*scale),'persistence_rmse':rmse((base[:,sl]-truth[:,sl])*scale)}
            row['foot_height_mm']={'rmse':rmse((st.feet_world_z(p)-st.feet_world_z(truth))*1000),
                                  'persistence_rmse':rmse((st.feet_world_z(base)-st.feet_world_z(truth))*1000)}
            row['planar_displacement_rmse_mm']=rmse((pp[valid,:2]-tp[valid,:2])*1000)
            yaw_error=pp[valid,2]-tp[valid,2]
            row['yaw_displacement_rmse_rad']=rmse(torch.atan2(yaw_error.sin(),yaw_error.cos()))
            row['pose_delta_xy_rmse_mm']=rmse(torch.stack(delta_xy_errors,1)[valid]*1000)
            row['pose_delta_yaw_rmse_rad']=rmse(torch.stack(delta_yaw_errors,1)[valid])
        result[str(h)]=row
    return result or {'status':'insufficient valid post-transient windows'}
