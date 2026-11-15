"""Physical equivalence and shared real/imagined gait timing."""
import copy,importlib,unittest
import torch
from go2_supermimic.robots.x30 import trot
from go2_supermimic.robots.x30.train import VelocityActor
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.supermimic import state as st
from go2_supermimic.robots.go2.target.terrain import SCAN_POINTS


class LocomotionMathTest(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('go2_supermimic.robots.x30.locomotion')
        except ModuleNotFoundError:self.fail('Adaptive locomotion policy is missing')

    def test_warm_actor_preserves_targets_with_nonzero_action_histories(self):
        m=self.module();torch.manual_seed(3)
        old=VelocityActor(235,(32,),12);gain=torch.tensor([.4,5/12,7/16]*4)
        new=m.LocomotionActor(235,(32,),12,gain)
        new.load_state_dict(old.state_dict(),strict=False)
        old.norm.update(torch.randn(100,235));new.norm.load_state_dict(old.norm.state_dict())
        old_obs=torch.randn(8,235);new_obs=old_obs.clone();new_obs[:,150:210]*=gain.repeat(5)
        torch.testing.assert_close(new(new_obs),old(old_obs)*gain,atol=2e-6,rtol=2e-6)
        new(new_obs).sum().backward()
        self.assertGreater(float(new.residual[-1].weight.grad.abs().sum()),0.)

    def test_world_model_transfer_preserves_corresponding_physical_actions(self):
        m=self.module();torch.manual_seed(4)
        old=WorldModel((32,));ratio=torch.tensor([2.5,2.4,16/7]*4)
        new=m.ActionMappedWorldModel((32,),ratio);new.load_state_dict(old.state_dict(),strict=False)
        s=torch.randn(5,70);a=torch.randn(5,12);prev=torch.randn(5,12);xi=torch.zeros(5,4);scan=torch.zeros(5,SCAN_POINTS)
        expected=old(s,a,prev,xi,scan);got=new(s,a/ratio,prev/ratio,xi,scan)
        for x,y in zip(got,expected):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)

    def test_phase_integrates_each_command_without_clock_reset(self):
        m=self.module();p=torch.tensor([[0.,1.],[0.,1.]])
        slow=torch.tensor([[.3,0.,0.],[0.,0.,0.]])
        got=m.advance_phase(p,slow,.02)
        torch.testing.assert_close(got,trot.advance_phase(p,.02))
        fast=torch.tensor([[3.,0.,0.],[0.,1.,0.]])
        after=m.advance_phase(got,fast,.02)
        self.assertGreater(float(after[0,0]),float(trot.advance_phase(got,.02)[0,0]))
        torch.testing.assert_close(after.norm(dim=-1),torch.ones(2))

    def test_high_speed_targets_allow_only_scheduled_flight_and_finite_gradients(self):
        m=self.module();cmd=torch.tensor([[3.,0.,0.],[0.,0.,0.]])
        # At 0.45 of a cycle, both diagonals are in swing with high-speed duty=.4.
        phase=torch.tensor([[.309016994,-.951056516],[.309016994,-.951056516]])
        stance,height=m.gait_targets(phase,cmd)
        self.assertLess(float(stance[0].max()),.15)
        torch.testing.assert_close(stance[1],torch.ones(4));torch.testing.assert_close(height[1],torch.zeros(4))
        contact=torch.zeros(2,4,requires_grad=True)
        penalty=m.unscheduled_flight(contact,stance)
        self.assertLess(float(penalty[0]),float(penalty[1])*.15)
        penalty.sum().backward();self.assertTrue(torch.isfinite(contact.grad).all())

    def test_lateral_one_meter_per_second_fits_conservative_stance_workspace(self):
        m=self.module();command=torch.tensor([[0.,1.,0.]])
        frequency,duty,_=m.profile(command)
        # Native model grid found ~.235 m lateral span; retain about 15% margin.
        self.assertLess(float(duty[0]/frequency[0]),.20)

if __name__=='__main__':unittest.main()
