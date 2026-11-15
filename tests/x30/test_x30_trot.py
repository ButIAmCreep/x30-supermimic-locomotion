"""Trot tests cover timing, diagonal targets, physical rewards and actor gradients."""
import importlib
import math
import unittest
import torch
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.train import Critic
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.supermimic.imagine import ImagineCfg
from go2_supermimic.robots.x30.train import VelocityActor
from tests.x30.test_x30_velocity import constants


class TrotMathTest(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('go2_supermimic.robots.x30.trot')
        except ModuleNotFoundError:
            self.fail('The phase-conditioned trot functions are not implemented yet')

    def test_phase_wrap_and_half_period_are_consistent(self):
        t=self.module()
        phase=t.phase_from_steps(torch.tensor([0,15,30]),.02)
        torch.testing.assert_close(phase,torch.tensor([[0.,1.],[0.,-1.],[0.,1.]]),atol=2e-6,rtol=0)
        predicted=t.advance_phase(t.phase_from_steps(torch.tensor([29]),.02),.02)
        torch.testing.assert_close(predicted,torch.tensor([[0.,1.]]),atol=2e-6,rtol=0)

    def test_diagonal_groups_swap_and_stop_has_no_swing(self):
        t=self.module()
        phase=t.phase_from_steps(torch.tensor([9,24]),.02)  # .3 and .8 cycles
        stance,height=t.gait_targets(phase,torch.tensor([[.4,0.,0.],[.4,0.,0.]]))
        self.assertTrue(torch.equal(stance>.5,torch.tensor([[1,0,0,1],[0,1,1,0]],dtype=torch.bool)))
        torch.testing.assert_close(height,torch.tensor([[0.,.04,.04,0.],[.04,0.,0.,.04]]),atol=2e-6,rtol=0)
        stance,height=t.gait_targets(phase,torch.zeros(2,3))
        torch.testing.assert_close(stance,torch.ones(2,4))
        torch.testing.assert_close(height,torch.zeros(2,4))

    def test_world_foot_velocity_cancels_body_motion_and_rotation(self):
        t=self.module();s0=torch.zeros(1,70);s1=s0.clone()
        s0[:,st.FEET].reshape(1,4,3)[:,:,0]=1
        s1[:,st.FEET].reshape(1,4,3)[:,:,0]=.996
        v=t.foot_world_velocity(s0,s1,torch.tensor([[.004,0.,0.]]),.02)
        torch.testing.assert_close(v,torch.zeros(1,4,2),atol=5e-6,rtol=0)
        s1[:,st.FEET].reshape(1,4,3)[:,:,0]=0
        s1[:,st.FEET].reshape(1,4,3)[:,:,1]=-1
        v=t.foot_world_velocity(s0,s1,torch.tensor([[0.,0.,math.pi/2]]),.02)
        torch.testing.assert_close(v,torch.zeros(1,4,2),atol=5e-6,rtol=0)

    def test_reward_prefers_diagonal_clearance_over_synchronous_bounce(self):
        t=self.module();s=torch.zeros(3,70);s[:,0]=.5;s[:,3]=-1;s[:,4]=.4
        feet=s[:,st.FEET].reshape(3,4,3)
        feet[:,:,2]=.036-.5
        feet[0,[1,2],2]+=.04
        feet[1,:,2]+=.006;s[1,6]=.4
        feet[2,[0,3],2]+=.04
        previous=s.clone();a=torch.zeros(3,12)
        reward,terms=t.rewards(previous,s,torch.zeros(3,3),a,a,torch.tensor([[.4,0.,0.]]).expand(3,-1),
            t.phase_from_steps(torch.full((3,),9),.02),constants(3),torch.full((12,),10.),torch.zeros(3),t.TrotRewards())
        self.assertGreater(float(reward[0]),float(reward[1]))
        self.assertGreater(float(reward[0]),float(reward[2]))
        self.assertLess(float(terms['contact_phase'][0]),float(terms['contact_phase'][2]))
        self.assertTrue(torch.isfinite(reward).all())

    def test_reset_masks_slip_and_air_does_not_hide_expected_stance(self):
        t=self.module();s=torch.zeros(2,70);s[:,0]=.5;s[:,3]=-1
        s[:,st.FEET].reshape(2,4,3)[:,:,2]=.1+.036-.5
        a=torch.zeros(2,12);dp=torch.tensor([[.01,0.,0.],[.01,0.,0.]])
        _,terms=t.rewards(s,s,dp,a,a,torch.tensor([[.4,0.,0.]]).expand(2,-1),
            t.phase_from_steps(torch.full((2,),9),.02),constants(),torch.ones(12)*10,
            torch.zeros(2),t.TrotRewards(),valid=torch.tensor([False,True]))
        self.assertEqual(float(terms['stance_slip'][0]),0)
        self.assertGreater(float(terms['stance_slip'][1]),0)
        self.assertGreater(float(terms['flight'][1]),.9)

    def test_imagination_backpropagates_and_uses_235_input_history(self):
        t=self.module();torch.manual_seed(7)
        wm=WorldModel((32,));actor=VelocityActor(235,(32,),12)
        for p in wm.parameters():p.requires_grad_(False)
        imag=t.TrotImagination(ImagineCfg(horizon=3),constants(),torch.ones(12)*10,.24,t.TrotRewards())
        critic=Critic(79,(32,));s=torch.zeros(2,70);s[:,0]=.5;s[:,3]=-1
        phase=t.phase_from_steps(torch.tensor([0,15]),.02)
        seed={'env':torch.arange(2),'s':s,'obs':torch.zeros(2,235),'a_prev':torch.zeros(2,12),
              'gen':torch.cat([torch.tensor([[.4,0.,0.]]).expand(2,-1),phase],-1),'xi':torch.zeros(2,4)}
        value,data=imag.rollout(wm,actor,critic,seed)
        (-value.mean()).backward()
        grads=[p.grad for p in actor.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
        self.assertEqual(tuple(data['critic_in'].shape),(2,4,79))
        torch.testing.assert_close(data['critic_in'][:,1,73:75],t.advance_phase(phase,.02))
        self.assertEqual(tuple(imag.gae_targets(critic,data).shape),(2,3))

    def test_imagination_preserves_first_post_reset_slip_mask(self):
        t=self.module();s=torch.zeros(2,70);s[:,0]=.5;s[:,3]=-1
        s[:,st.FEET].reshape(2,4,3)[:,:,2]=.036-.5
        imag=t.TrotImagination(ImagineCfg(horizon=1),constants(),torch.ones(12)*10,.24,t.TrotRewards())
        def wm(state,action,previous,context,relief):
            return state,torch.tensor([[.01,0.,0.]]).expand(len(state),-1)
        seed={'env':torch.arange(2),'s':s,'obs':torch.zeros(2,235),'a_prev':torch.zeros(2,12),
              'gen':torch.cat([torch.tensor([[.4,0.,0.]]).expand(2,-1),t.phase_from_steps(torch.full((2,),9),.02)],-1),
              'xi':torch.zeros(2,4),'reward_valid':torch.tensor([False,True])}
        _,data=imag.rollout(wm,lambda obs:torch.zeros(len(obs),12),lambda x:torch.zeros(len(x),1),seed)
        self.assertGreater(float(data['rewards'][0,0]-data['rewards'][1,0]),.005)

if __name__=='__main__':unittest.main()
