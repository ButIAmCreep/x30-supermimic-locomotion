"""Velocity contract checks; synthetic motor numbers are software fixtures only."""

from copy import deepcopy
from dataclasses import replace
import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from go2_supermimic.robot.x30_model import SDK_JOINT_ORDER, load_x30_spec
from go2_supermimic.robot.x30_training import load_parameters, validate_parameters, robot_cfg
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic import velocity as vel
from go2_supermimic.supermimic.imagine import ImagineCfg
from go2_supermimic.supermimic.train import Critic
from go2_supermimic.supermimic.velocity_train import VelocityActor, VelocityTrainer, defaults
from go2_supermimic.supermimic.world_model import WorldModel


def synthetic_parameters():
  """Not X30 hardware data. Only exercises software contracts."""
  model = load_x30_spec().compile()
  return {"parameter_source": "SYNTHETIC SOFTWARE TEST ONLY", "delay_physics_steps": 0,
          "standing_height": 0.55, "model_mass_kg": float(model.body_mass.sum()),
          "joints": {n: {"kp": 60., "kd": 2., "effort_limit": 50., "velocity_limit": 10.,
                          "armature": 0.1, "frictionloss": 0., "action_scale": 0.1,
                          "default_q": 1.2 if "Knee" in n else (-0.6 if "HipY" in n else 0.)}
                     for n in SDK_JOINT_ORDER}}


def constants(n=2):
  return st.RobotConsts(torch.zeros(n, 12), torch.full((n, 12), 0.1), torch.full((12,), 60.),
                       torch.full((12,), 2.), torch.full((12,), 50.),
                       torch.tensor([-2., 2.]).expand(n, 12, 2).clone())


class VelocityContractTest(unittest.TestCase):
  def test_nominal_59kg_scales_mass_and_inertia_without_changing_geometry(self):
    import numpy as np
    from mjlab.entity.entity import Entity
    raw = load_x30_spec().compile()
    parameters = {**synthetic_parameters(), "model_mass_kg": 59.0, "mass_scaling": "uniform"}
    model = Entity(robot_cfg(parameters)).spec.compile()
    scale = 59.0 / float(raw.body_mass.sum())
    self.assertAlmostEqual(float(model.body_mass.sum()), 59.0, places=6)
    np.testing.assert_allclose(model.body_mass, raw.body_mass * scale)
    np.testing.assert_allclose(model.body_inertia, raw.body_inertia * scale)
    np.testing.assert_allclose(model.body_ipos, raw.body_ipos)
    # Building a training entity must not mutate the structural source asset.
    np.testing.assert_allclose(load_x30_spec().compile().body_mass, raw.body_mass)

  def test_evaluation_does_not_reset_every_training_episode(self):
    from go2_supermimic.env.x30_velocity import x30_velocity_env_cfg
    p = synthetic_parameters()
    self.assertEqual(x30_velocity_env_cfg(p).episode_length_s, 20.)
    self.assertGreater(x30_velocity_env_cfg(p, play=True).episode_length_s, 60.)

  def test_motor_configuration_rejects_missing_or_inconsistent_data(self):
    with self.assertRaisesRegex(ValueError, "robot-config"):
      load_parameters("")
    p = synthetic_parameters()
    validate_parameters(p)
    for field, value in (("kp", None), ("effort_limit", -1), ("default_q", 99)):
      bad = deepcopy(p)
      bad["joints"][SDK_JOINT_ORDER[0]][field] = value
      with self.assertRaises(ValueError):
        validate_parameters(bad)
    with self.assertRaisesRegex(ValueError, "history"):
      validate_parameters({**p, "delay_physics_steps": 2})
    with self.assertRaisesRegex(ValueError, "mass"):
      validate_parameters({**p, "model_mass_kg": 100})

  def test_compiled_actuators_target_sdk_joints(self):
    import mujoco
    from go2_supermimic.robot.x30_inspect import compile_model
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/"synthetic.json"
      path.write_text(json.dumps({**synthetic_parameters(), "model_mass_kg": 59.0, "mass_scaling": "uniform"}))
      model = compile_model("mjlab", robot_config=str(path))
    self.assertEqual((model.nq, model.nv, model.nu), (19, 18, 12))
    targets = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
                    for jid in model.actuator_trnid[:, 0])
    self.assertEqual(targets, SDK_JOINT_ORDER)

  def test_observation_is_225_and_history_is_termwise(self):
    s, a, command = torch.randn(2, 70), torch.randn(2, 12), torch.randn(2, 3)
    frame = vel.obs_frame(s, a, command)
    hist = torch.arange(225).float().expand(2, -1)
    new = st.push_history(hist, frame, vel.HISTORY, vel.OBS_TERMS)
    offset = 0
    for name, dim in vel.OBS_TERMS:
      block = new[:, offset:offset+dim*5]
      torch.testing.assert_close(block[:, :-dim], hist[:, offset+dim:offset+dim*5])
      torch.testing.assert_close(block[:, -dim:], frame[name])
      offset += dim*5
    self.assertEqual(offset, 225)
    self.assertNotIn("target", frame)

  def test_reward_prefers_correct_linear_and_angular_direction(self):
    s, a, command = torch.zeros(2, 70), torch.zeros(2, 12), torch.tensor([[.2, .1, .2]]).expand(2, -1)
    s[:, 3] = -1
    s[0, 4:6], s[0, 9] = command[0, :2], command[0, 2]
    s[1, 4:6], s[1, 9] = -command[1, :2], -command[1, 2]
    r, terms = vel.rewards(s, command, a, a, constants(), torch.ones(12)*10, torch.zeros(2), vel.VelocityRewards())
    self.assertGreater(r[0], r[1])
    self.assertEqual(float(terms["linear"][0]), 1.)
    self.assertEqual(float(terms["angular"][0]), 1.)

  def test_replay_preserves_commands_and_masks_reset_crossings(self):
    from go2_supermimic.supermimic.buffer import ReplayBuffer
    b = ReplayBuffer(1, 6, 225, "cpu", gen_dim=3)
    for i in range(6):
      b.add(torch.ones(1, 70)*i, torch.zeros(1, 3), torch.zeros(1, 12), torch.zeros(1, 12),
            torch.zeros(1, 225), torch.ones(1, 3)*i, torch.tensor([i==2]), torch.zeros(1, 77))
    e, idx, valid = b.sample_windows(64, 3)
    for row, mask in zip(idx, valid):
      for step, slot in enumerate(row[:-1]):
        self.assertEqual(float(mask[step]), float(not any(int(x)==2 for x in row[:step+1])))
        self.assertEqual(float(b.gen[0, slot, 0]), float(b.s[0, slot, 0]))

  def test_imagined_reward_backpropagates_to_actor(self):
    torch.manual_seed(7)
    wm, actor = WorldModel((32,)), VelocityActor(225, (32,), 12)
    for p in wm.parameters():
      p.requires_grad_(False)
    imag = vel.VelocityImagination(ImagineCfg(horizon=3), constants(), torch.ones(12)*10, .2, vel.VelocityRewards())
    critic = Critic(imag.critic_dim, (32,))
    s = torch.randn(2, 70)*.01
    s[:, 0], s[:, 3] = .55, -1
    seed = {"env": torch.arange(2), "s": s, "obs": torch.randn(2, 225)*.1,
            "a_prev": torch.zeros(2, 12), "gen": torch.ones(2, 3)*.1, "xi": torch.zeros(2, 4)}
    result, data = imag.rollout(wm, actor, critic, seed)
    (-result.mean()).backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
    self.assertEqual(tuple(imag.gae_targets(critic, data).shape), (2, 3))

  @unittest.skipUnless(os.environ.get("X30_GPU_SMOKE") == "1", "explicit synthetic GPU smoke only")
  def test_synthetic_gpu_training_loop(self):
    with tempfile.TemporaryDirectory() as d:
      path = Path(d)/"test_only.json"
      path.write_text(json.dumps({**synthetic_parameters(), "model_mass_kg": 59.0, "mass_scaling": "uniform"}))
      cfg = replace(defaults(), robot_config=str(path), num_envs=2, buffer_len=16,
                    wm_window=2, wm_batch=4, wm_updates=1, archive_size=8, imag_batch=4,
                    horizon=2, actor_hidden=(32,), critic_hidden=(32,), wm_hidden=(32,),
                    warmup_iterations=0, critic_epochs=1, critic_minibatches=1,
                    iterations=2, save_interval=2, log_interval=1, log_root=d)
      trainer = VelocityTrainer(cfg)
      try:
        # Actor observation bridge must match manager terms at reset and after stepping.
        trainer.run()
        state = vel.robot_state(trainer.env)
        frame = vel.obs_frame(state, trainer.a_prev, trainer.term.command)
        offset = 0
        for name, dim in vel.OBS_TERMS:
          torch.testing.assert_close(trainer.obs[:, offset+4*dim:offset+5*dim], frame[name], atol=1e-4, rtol=1e-4)
          offset += 5*dim
        self.assertEqual(trainer.nonfinite_steps, 0)
        ckpt = torch.load(trainer.last_ckpt, weights_only=False)
        self.assertEqual(ckpt["contract"]["task"], "X30-Velocity-Flat")
        self.assertAlmostEqual(float(trainer.env.sim.mj_model.body_mass.sum()), 59., places=5)
        from go2_supermimic.supermimic.velocity_eval import evaluate
        video = str(Path(d)/"evaluation.mp4") if os.environ.get("MUJOCO_GL")=="glfw" else None
        evaluate(SimpleNamespace(checkpoint=str(trainer.last_ckpt), task="X30-Velocity-Flat",
                                 command=(.1, 0., 0.), mode=None, terrain=None, robot_config="",
                                 num_envs=1, seed=0, seconds=.04, width=160, height=96, video=video))
        if video:
          import imageio_ffmpeg
          decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", video,
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                   check=True, capture_output=True)
          self.assertEqual(len(decoded.stdout), 2*160*96*3)
      finally:
        trainer.env.close()
        trainer.writer.close()
