"""Independent physical front-foot contact, hold and recovery evaluation."""

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from go2_supermimic.robots.x30 import handstand as hs
from go2_supermimic.robots.x30.handstand_actor import HandstandActor
from go2_supermimic.robots.x30.handstand_env import handstand_env_cfg
from go2_supermimic.robots.x30.handstand_train import _walking_parameters, contract
from go2_supermimic.robots.x30.handstand_transfer import validate_fixed_buffers
from go2_supermimic.robots.x30.robot import validate_parameters
from go2_supermimic.supermimic import state as st


def handstand_passed(row):
    required = ('falls', 'hold_samples', 'front_only_fraction', 'pitch_error_rms_deg',
                'rear_foot_height_min_m', 'base_collision_bottom_min_m',
                'recovery_samples', 'recovery_tilt_max_deg', 'recovery_speed_error_m_s')
    if any(not isinstance(row.get(k), (float, int)) or not math.isfinite(row[k]) for k in required):
        return False
    return (row['falls'] == 0 and row['hold_samples'] >= 200 and
            row['front_only_fraction'] >= .8 and row['pitch_error_rms_deg'] <= 10. and
            row['rear_foot_height_min_m'] >= .1 and
            row['base_collision_bottom_min_m'] >= .08 and
            row['recovery_samples'] >= 50 and row['recovery_tilt_max_deg'] <= 15. and
            row['recovery_speed_error_m_s'] <= .1)


def front_only_contact_mask(geoms, worlds, active, feet, robot_geoms, floor, num_envs):
    """Require both front feet and no other robot geom supported by the floor."""
    support = np.zeros((num_envs, 2), dtype=bool)
    forbidden = np.zeros(num_envs, dtype=bool)
    ground0 = np.isin(geoms[:, 0], floor)
    ground1 = np.isin(geoms[:, 1], floor)
    ground = active & (ground0 | ground1)
    other = np.where(ground0, geoms[:, 1], geoms[:, 0])
    robot_ground = ground & np.isin(other, robot_geoms)
    for index, foot in enumerate(feet[:2]):
        support[worlds[robot_ground & (other == foot)], index] = True
    forbidden[worlds[robot_ground & ~np.isin(other, feet[:2])]] = True
    return support.all(-1) & ~forbidden


class FrontFootContacts:
    """Observe MuJoCo Warp solver contacts at each physics substep."""

    def __init__(self, env):
        self.env = env
        self.original = env.sim.step
        robot = env.scene['robot']
        model = env.sim.mj_model
        local, _ = robot.find_geoms(tuple(f'{leg}_foot_collision' for leg in ('FL', 'FR', 'HL', 'HR')),
                                    preserve_order=True)
        self.feet = robot.indexing.geom_ids[local].cpu().numpy()
        self.robot_geoms = robot.indexing.geom_ids.cpu().numpy()
        self.floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        data = env.sim.wp_data
        self.nacon = wp.to_torch(data.nacon)
        self.geom = wp.to_torch(data.contact.geom)
        self.efc = wp.to_torch(data.contact.efc_address)
        self.world = wp.to_torch(data.contact.worldid)
        self.hold_samples = np.zeros(env.num_envs, dtype=int)
        self.front_only = np.zeros(env.num_envs, dtype=int)
        env.sim.step = self._step

    def _step(self):
        self.original()
        env = self.env
        nc = int(self.nacon[0])
        if nc > len(self.geom):
            raise RuntimeError('Contact buffer overflow invalidates front support metrics')
        geoms = self.geom[:nc].cpu().numpy()
        worlds = self.world[:nc].cpu().numpy()
        active = (self.efc[:nc, 0] >= 0).cpu().numpy()
        front_only = front_only_contact_mask(geoms, worlds, active, self.feet,
                                             self.robot_geoms, self.floor, env.num_envs)
        sub = (env._sim_step_counter - 1) % env.cfg.decimation + 1
        age = env.episode_length_buf.cpu().numpy() * env.step_dt + sub * env.physics_dt
        hold = (age >= 4.) & (age < 6.)
        self.hold_samples += hold.astype(int)
        self.front_only += (hold & front_only).astype(int)

    def close(self):
        self.env.sim.step = self.original


def prepare(checkpoint, num_envs, seed, video=False, width=640, height=360):
    if num_envs < 1:
        raise ValueError('num-envs must be positive')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    old = saved.get('contract', {})
    if old.get('task') != hs.TASK:
        raise ValueError(f'Requires {hs.TASK} checkpoint')
    parameters = validate_parameters(old['parameters'])
    if old != contract(parameters):
        raise ValueError('Handstand checkpoint contract differs from current model or reward')
    actor = HandstandActor(tuple(saved['cfg']['actor_hidden']), _walking_parameters(),
                           parameters, old['reference_action'])
    validate_fixed_buffers(saved, actor, parameters)
    actor.load_state_dict(saved['actor'])
    actor.eval()
    cfg = handstand_env_cfg(parameters, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = True
    cfg.viewer.width, cfg.viewer.height = width, height
    cfg.viewer.distance = 2.5
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = 'robot'
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode='rgb_array' if video else None)
    try:
        env.reset()
        return env, actor.to(env.device), saved
    except BaseException:
        env.close()
        raise


def evaluate(args):
    if not math.isfinite(args.seconds) or args.seconds < 10.:
        raise ValueError('Handstand evaluation needs at least one 10 s skill cycle')
    env, actor, checkpoint = prepare(args.checkpoint, args.num_envs, args.seed,
                                      bool(args.video), args.width, args.height)
    observer = FrontFootContacts(env)
    writer = None
    try:
        if args.video:
            import mediapy
            path = Path(args.video)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = mediapy.VideoWriter(str(path), shape=(args.height, args.width), fps=50)
            writer.__enter__()
        obs = env.get_observations()['actor']
        steps = round(args.seconds / env.step_dt)
        falls = 0
        pitch_error = 0.
        pitch_samples = 0
        rear_min = float('inf')
        bottom_min = float('inf')
        recovery_samples = 0
        recovery_tilt_max = 0.
        recovery_speed_error = 0.
        start = last = time.monotonic()
        with torch.no_grad():
            for i in range(steps):
                age = env.episode_length_buf.clone() * env.step_dt
                action = actor(obs)
                result, _, died, timed, _ = env.step(action)
                falls += int(died.sum())
                state = hs.robot_state(env)
                valid = ~(died | timed)
                hold = (age >= 4.) & (age < 6.) & valid
                if bool(hold.any()):
                    gravity = state[hold, st.GRAV]
                    pitch = -torch.rad2deg(torch.atan2(gravity[:, 0], -gravity[:, 2]))
                    pitch_error += float(((pitch - 40.) ** 2).sum())
                    pitch_samples += int(hold.sum())
                    rear = st.feet_world_z(state[hold])[:, 2:] - .036
                    rear_min = min(rear_min, float(rear.min()))
                    bottom_min = min(bottom_min, float(hs.torso_bottom(state[hold]).min()))
                recover = (age >= 8.) & (age < 10.) & valid
                if bool(recover.any()):
                    gravity = state[recover, st.GRAV]
                    tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
                    recovery_tilt_max = max(recovery_tilt_max, float(tilt.max()))
                    error = (state[recover, st.V_YAW][:, 0] - .3).abs()
                    recovery_speed_error += float(error.sum())
                    recovery_samples += int(recover.sum())
                obs = result['actor']
                if writer:
                    writer.add_image(env.render())
                now = time.monotonic()
                if now - last >= 1 or i + 1 == steps:
                    elapsed = now - start
                    eta = elapsed * (steps - i - 1) / (i + 1)
                    print(f'Handstand evaluation {i+1}/{steps} elapsed={elapsed:.0f}s ETA={eta:.0f}s', flush=True)
                    last = now
        hold_samples = int(observer.hold_samples.sum())
        report = {
            'task': hs.TASK, 'checkpoint': str(args.checkpoint),
            'iteration': checkpoint['iteration'], 'seed': args.seed,
            'seconds': steps * env.step_dt, 'num_envs': env.num_envs,
            'falls': falls, 'hold_samples': hold_samples,
            'front_only_fraction': float(observer.front_only.sum() / hold_samples) if hold_samples else None,
            'pitch_error_rms_deg': math.sqrt(pitch_error / pitch_samples) if pitch_samples else None,
            'rear_foot_height_min_m': rear_min if pitch_samples else None,
            'base_collision_bottom_min_m': bottom_min if pitch_samples else None,
            'recovery_samples': recovery_samples,
            'recovery_tilt_max_deg': recovery_tilt_max if recovery_samples else None,
            'recovery_speed_error_m_s': recovery_speed_error / recovery_samples if recovery_samples else None,
        }
        report['passed'] = handstand_passed(report)
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return report
    finally:
        if writer:
            writer.__exit__(None, None, None)
        observer.close()
        env.close()


def play(args):
    from mjlab.viewer import NativeMujocoViewer
    env, actor, _ = prepare(args.checkpoint, 1, args.seed)
    count = 0
    try:
        def policy(observations):
            nonlocal count
            count += 1
            if count % 50 == 0:
                s = hs.robot_state(env)[0]
                age = float(env.episode_length_buf[0]) * env.step_dt
                pitch = -math.degrees(math.atan2(float(s[1]), -float(s[3])))
                print(f'[LIVE] t={count*.02:.1f}s cycle={age:.1f}s pitch={pitch:.1f}deg '
                      f'base_z={float(s[0]):.3f}m', flush=True)
            return actor(observations['actor'])
        viewer = NativeMujocoViewer(env, policy, frame_rate=30.,
                                    enable_perturbations=False, verbosity=1)
        viewer.run(num_steps=round(args.seconds / env.step_dt) if args.seconds is not None else None)
    finally:
        env.close()


def main(live=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=float, default=None if live else 10.)
    if not live:
        parser.add_argument('--num-envs', type=int, default=1)
        parser.add_argument('--video')
        parser.add_argument('--width', type=int, default=640)
        parser.add_argument('--height', type=int, default=360)
        parser.add_argument('--report')
    args = parser.parse_args()
    if live:
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds < .02):
            parser.error('seconds must be finite and positive')
        play(args)
    else:
        evaluate(args)
