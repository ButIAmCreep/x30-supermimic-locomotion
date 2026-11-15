"""Reproductions of reset, isolation, and exam-measurement review findings."""
import math,os,random,unittest
from types import SimpleNamespace
import numpy as np
import torch
from go2_supermimic.robots.x30.curriculum_command import CurriculumCommand,CurriculumCommandCfg
from go2_supermimic.robots.x30.curriculum_eval import isolated_evaluation,ExamMeasurements,case_passed


class CurriculumBoundaryTest(unittest.TestCase):
    def command(self):
        data=SimpleNamespace(heading_w=torch.zeros(2))
        env=SimpleNamespace(num_envs=2,device='cpu',cfg=SimpleNamespace(seed=0),scene={'robot':SimpleNamespace(data=data)},
                            common_step_counter=0,step_dt=.02)
        cfg=CurriculumCommandCfg(entity_name='robot',resampling_time_range=(3.,6.),evaluation=True,
            heading_command=False,ranges=CurriculumCommandCfg.Ranges(lin_vel_x=(-.6,3.),lin_vel_y=(-1.,1.),ang_vel_z=(-1.2,1.2)))
        return CurriculumCommand(cfg,env),env,data

    def test_initial_and_partial_reset_refresh_is_not_turn_progress(self):
        for ids,after in (([0,1],[2.,-1.]),([1],[1.7])):
            with self.subTest(ids=ids):
                term,env,data=self.command();ids=torch.tensor(ids)
                term.begin_uturn(ids,torch.full((len(ids),),.2))
                data.heading_w[ids]=torch.tensor(after)  # forward() after reset refreshes derived yaw.
                term._update_command(ids)
                torch.testing.assert_close(term.remaining[ids],torch.full((len(ids),),math.pi))
                data.heading_w[ids]+=.1;term._update_command(ids)
                torch.testing.assert_close(term.remaining[ids],torch.full((len(ids),),math.pi-.1))

    def test_turn_completing_after_deadline_stays_failed(self):
        term,env,_=self.command();ids=torch.tensor([0])
        term.begin_uturn(ids,torch.tensor([.2]));term._update_command(ids)
        env.common_step_counter=1086;term.remaining[0]=.077
        term._update_command(ids)
        self.assertFalse(bool(term.completed[0]));self.assertTrue(bool(term.timed_out[0]))

    def test_isolation_restores_all_rngs_and_hash_seed_on_error(self):
        a=torch.nn.Linear(1,1);w=torch.nn.Linear(1,1);w.eval()
        python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
        old_hash=os.environ.get('PYTHONHASHSEED')
        try:
            with self.assertRaisesRegex(RuntimeError,'exam failed'):
                with isolated_evaluation(a,w,123):
                    random.seed(999);np.random.seed(999);os.environ['PYTHONHASHSEED']='999'
                    raise RuntimeError('exam failed')
            self.assertEqual(random.getstate(),python_state)
            after=np.random.get_state();self.assertEqual(after[0],numpy_state[0]);np.testing.assert_array_equal(after[1],numpy_state[1])
            self.assertEqual(after[2:],numpy_state[2:]);self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
            self.assertEqual(os.environ.get('PYTHONHASHSEED'),old_hash)
            self.assertTrue(a.training);self.assertFalse(w.training)
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
            if old_hash is None:os.environ.pop('PYTHONHASHSEED',None)
            else:os.environ['PYTHONHASHSEED']=old_hash

    def measurement(self,kind):
        data=SimpleNamespace(actuator_force=torch.zeros(1,12),joint_vel=torch.zeros(1,12),
                             root_link_pos_w=torch.zeros(1,3),heading_w=torch.zeros(1))
        term=SimpleNamespace(completed=torch.tensor([False]),timed_out=torch.tensor([False]),remaining=torch.zeros(1))
        env=SimpleNamespace(num_envs=1,episode_length_buf=torch.zeros(1,dtype=torch.long),common_step_counter=0,
            scene={'robot':SimpleNamespace(data=data)},command_manager=SimpleNamespace(get_term=lambda key:term))
        case=dict(name=kind,kind=kind,axis=None,grade='retention',command=[.3,0.,0.])
        return ExamMeasurements(env,[case]),env,data

    def test_startup_and_braking_limit_violations_are_not_discarded(self):
        for kind in ('forward','stop'):
            measure,env,data=self.measurement(kind)
            s=torch.zeros(1,70);s[:,3]=-1
            for step in range(1,401):
                env.episode_length_buf[:]=step;env.common_step_counter=step
                data.actuator_force[:]=1 if step<=100 else 0;data.joint_vel[:]=2 if step<=100 else 0
                measure.control(s,torch.zeros(1,3),torch.zeros(1,dtype=torch.bool),torch.ones(12),torch.ones(12))
            row=measure.summary()[0]
            self.assertAlmostEqual(row['torque_saturation_fraction'],.25)
            self.assertAlmostEqual(row['overspeed_fraction'],.25)

    def test_stop_speed_uses_two_seconds(self):
        measure,env,data=self.measurement('stop');s=torch.zeros(1,70);s[:,3]=-1
        for step in range(1,401):
            env.episode_length_buf[:]=step;env.common_step_counter=step
            s[:,4]=.08 if 301<=step<=350 else 0.
            measure.control(s,torch.zeros(1,3),torch.zeros(1,dtype=torch.bool),torch.ones(12),torch.ones(12))
        self.assertAlmostEqual(measure.summary()[0]['stop_speed'],.04,places=6)

    def test_one_failed_switch_segment_cannot_be_diluted_by_the_others(self):
        row=dict(samples=100,falls=0,tracking_normalized_rmse=.5,yaw_rmse=.02,max_tilt_deg=3.,
            torque_saturation_fraction=0.,overspeed_fraction=0.,phase_agreement=.96,phase_labels=200,
            unscheduled_flight_fraction=0.,linear_rmse=.02,kind='transition',segments=[{'passed':True} for _ in range(8)])
        self.assertTrue(case_passed(row));row['segments'][3]['passed']=False
        self.assertFalse(case_passed(row))

    def test_switch_fall_is_attributed_to_its_own_segment(self):
        measure,env,data=self.measurement('transition');s=torch.zeros(1,70);s[:,3]=-1
        for step in range(1,1201):
            env.episode_length_buf[:]=step;env.common_step_counter=step
            measure.control(s,torch.zeros(1,3),torch.tensor([step==140]),torch.ones(12),torch.ones(12))
        segments=measure.summary()[0]['segments']
        self.assertEqual(segments[0]['falls'],1)
        self.assertTrue(all(x['falls']==0 for x in segments[1:]))

if __name__=='__main__':unittest.main()
