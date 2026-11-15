import importlib,os,tempfile,unittest
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]

class LocomotionEnvironmentTest(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('go2_supermimic.robots.x30.locomotion_train')
        except ModuleNotFoundError:self.fail('Automatic locomotion trainer is missing')

    def test_heading_controller_accumulates_signed_rotation_and_resumes(self):
        self.module()
        from go2_supermimic.robots.x30.curriculum_command import heading_control
        remaining=torch.tensor([3.14159265,-3.14159265,.04])
        command,complete=heading_control(remaining,torch.tensor([.7,.7,.7]),torch.tensor([.3,.3,.3]))
        torch.testing.assert_close(command,torch.tensor([[0.,0.,.7],[0.,0.,-.7],[.3,0.,0.]]))
        self.assertEqual(complete.tolist(),[False,False,True])

    @unittest.skipUnless(os.environ.get('X30_GPU_SMOKE')=='1','explicit GPU integration')
    def test_real_training_phase_transfer_save_and_resume(self):
        tr=self.module()
        from go2_supermimic.robots.x30.locomotion_env import phase_observation
        from go2_supermimic.robots.x30 import locomotion as loc
        with tempfile.TemporaryDirectory() as d:
            cfg=replace(tr.defaults(),num_envs=4,buffer_len=24,wm_window=2,wm_batch=4,
                wm_updates=1,archive_size=8,imag_batch=4,horizon=2,actor_hidden=(32,),
                critic_hidden=(32,),wm_hidden=(32,),warmup_iterations=0,critic_epochs=1,
                critic_minibatches=1,iterations=2,log_interval=1,save_interval=2,
                evaluation_interval=1000,log_root=d)
            trainer=tr.LocomotionTrainer(cfg)
            try:
                p=phase_observation(trainer.env).clone();cmd=trainer.term.command.clone()
                trainer.collect()
                torch.testing.assert_close(phase_observation(trainer.env),loc.advance_phase(p,cmd,.02),atol=1e-6,rtol=1e-6)
                trainer.run();saved=trainer.last_ckpt
                self.assertTrue(saved.exists());self.assertEqual(trainer.nonfinite_steps,0)
                state=trainer.term.curriculum.state_dict()
                trainer.load(str(saved));self.assertEqual(trainer.start_it,2)
                self.assertEqual(trainer.term.curriculum.state_dict(),state)
                trainer.env.reset(env_ids=torch.tensor([0],device=trainer.dev))
                torch.testing.assert_close(phase_observation(trainer.env)[0],torch.tensor([0.,1.],device=trainer.dev))
            finally:trainer.env.close();trainer.writer.close()

if __name__=='__main__':unittest.main()
