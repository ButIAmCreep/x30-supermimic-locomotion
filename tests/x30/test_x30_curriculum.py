"""Curriculum gates must respond to measured skill, never elapsed iterations."""
import copy
import importlib
import unittest
import torch


class CurriculumTest(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('go2_supermimic.robots.x30.curriculum')
        except ModuleNotFoundError:self.fail('Automatic curriculum controller is missing')

    def report(self, iteration=1000):
        return {'iteration':iteration,'retention_passed':True,'world_model_passed':True,
                'axes':{k:{'mastered':True,'probe':True} for k in ('forward','backward','lateral','yaw')}}

    def test_two_passes_promote_axes_independently(self):
        m=self.module();c=m.Curriculum();r=self.report();r['axes']['lateral']['probe']=False
        c.observe(r);self.assertAlmostEqual(c.caps()['forward'],.5)
        r['iteration']=2000;c.observe(r)
        self.assertAlmostEqual(c.caps()['forward'],.8)
        self.assertAlmostEqual(c.caps()['lateral'],.2)

    def test_probe_failures_do_not_demote_mastered_skill(self):
        m=self.module();c=m.Curriculum()
        for it in (1000,2000):c.observe(self.report(it))
        for it in (3000,4000,5000):
            r=self.report(it);r['axes']['forward']['probe']=False;c.observe(r)
        self.assertAlmostEqual(c.caps()['forward'],.8)
        self.assertLess(c.state_dict()['axes']['forward']['probe_weight'],.2)
        for it in (6000,7000):
            r=self.report(it);r['axes']['forward']['mastered']=False;c.observe(r)
        self.assertAlmostEqual(c.caps()['forward'],.5)

    def test_retention_or_world_model_failure_blocks_graduation(self):
        m=self.module()
        for key in ('retention_passed','world_model_passed'):
            c=m.Curriculum()
            for it in (1000,2000,3000):
                r=self.report(it);r[key]=False;c.observe(r)
            self.assertEqual(c.caps()['forward'],.5)

    def test_bad_reports_are_rejected_before_state_changes(self):
        m=self.module();c=m.Curriculum();before=c.state_dict()
        for value in (float('nan'),1,None):
            r=self.report();r['axes']['yaw']['probe']=value
            with self.assertRaises(ValueError):c.observe(r)
            self.assertEqual(c.state_dict(),before)
        c.observe(self.report())
        with self.assertRaises(ValueError):c.observe(self.report())

    def test_state_and_sampler_rng_round_trip(self):
        m=self.module();c=m.Curriculum();c.observe(self.report());d=m.Curriculum();d.load_state_dict(c.state_dict())
        a=torch.Generator().manual_seed(42);state=a.get_state();x,modes=c.sample(500,a,'cpu')
        b=torch.Generator();b.set_state(state);y,other=d.sample(500,b,'cpu')
        torch.testing.assert_close(x,y);torch.testing.assert_close(modes,other)
        bad=copy.deepcopy(c.state_dict());bad['axes']['lateral']['level']=99
        before=d.state_dict()
        with self.assertRaises(ValueError):d.load_state_dict(bad)
        self.assertEqual(d.state_dict(),before)

    def test_samples_cover_both_signs_and_respect_coupled_turn_bound(self):
        m=self.module();c=m.Curriculum();x,mode=c.sample(30000,torch.Generator().manual_seed(7),'cpu')
        self.assertLess(abs(float((mode==0).float().mean())-.1),.015)
        self.assertTrue((x[mode==0]==0).all())
        for kind,col in ((3,1),(4,2)):
            values=x[mode==kind,col]
            self.assertGreater(float((values>0).float().mean()),.45)
            self.assertLess(float((values>0).float().mean()),.55)
        self.assertLessEqual(float(x[:,0].max()),.8)
        self.assertGreaterEqual(float(x[:,0].min()),-.3)
        self.assertLessEqual(float(x[:,1].abs().max()),.3)
        turn=x[mode==5]
        self.assertLessEqual(float((turn[:,0]*turn[:,2]).abs().max()),1.50001)

    def test_final_caps_are_bounded_and_probe_never_exceeds_goal(self):
        m=self.module();c=m.Curriculum()
        for i in range(1,21):c.observe(self.report(i*1000))
        self.assertEqual(c.caps(),{'forward':3.,'backward':.6,'lateral':1.,'yaw':1.2})
        self.assertEqual(c.caps(probe=True),c.caps())
        x,_=c.sample(10000,torch.Generator().manual_seed(1),'cpu')
        self.assertLessEqual(float(x[:,0].max()),3.)
        self.assertGreaterEqual(float(x[:,0].min()),-.600001)
        self.assertLessEqual(float(x[:,1].abs().max()),1.)

if __name__=='__main__':unittest.main()
