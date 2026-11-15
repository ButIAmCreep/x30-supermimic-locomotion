"""Kinematic front-foot support search and deliberately separate open-loop physics probe."""

import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.robot import load_training_spec


def _model(parameters):
    spec = load_training_spec(parameters)
    spec.worldbody.add_geom(name='handstand_probe_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[10., 10., .1], friction=[1., .005, .0005])
    return spec.compile()


def _indices(model):
    site = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
            for leg in ('FL', 'FR', 'HL', 'HR')]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'base_collision')
    return site, body, geom


def find_pose(parameters, angle_deg=40.):
    """Find a symmetric pose; geometry success is not a balance or motor proof."""
    if not 10 <= angle_deg <= 75 or not math.isfinite(angle_deg):
        raise ValueError('Requested pitch must be finite and between 10 and 75 degrees')
    model = _model(parameters)
    data = mujoco.MjData(model)
    sites, body, geom = _indices(model)
    q0 = np.array([parameters['joints'][n]['default_q'] for n in SDK_JOINT_ORDER])
    scale = np.array([parameters['joints'][n]['action_scale'] for n in SDK_JOINT_ORDER])
    lower, upper = q0 - scale, q0 + scale
    # Search a symmetric joint pose. Four hip-X targets stay at standing values.
    lo = np.array([.1, -1.5, max(lower[1], lower[4]), max(lower[2], lower[5]),
                   max(lower[7], lower[10]), max(lower[8], lower[11])])
    hi = np.array([.9, -.1, min(upper[1], upper[4]), min(upper[2], upper[5]),
                   min(upper[7], upper[10]), min(upper[8], upper[11])])
    if np.any(lo >= hi):
        raise ValueError('No common symmetric X30 handstand joint range')

    def pose(x):
        z, pitch, front_y, front_k, rear_y, rear_k = x
        data.qpos[:3] = [0., 0., z]
        data.qpos[3:7] = [math.cos(pitch / 2), 0., math.sin(pitch / 2), 0.]
        q = q0.copy()
        q[[1, 4]], q[[2, 5]] = front_y, front_k
        q[[7, 10]], q[[8, 11]] = rear_y, rear_k
        data.qpos[7:] = q
        mujoco.mj_forward(model, data)
        feet = data.site_xpos[sites].copy()
        com = data.subtree_com[body].copy()
        matrix = data.geom_xmat[geom].reshape(3, 3)
        bottom = data.geom_xpos[geom, 2] - np.abs(matrix[2]) @ model.geom_size[geom]
        return feet, com, float(bottom), q

    target = -math.radians(angle_deg)

    def residual(x):
        feet, com, bottom, q = pose(x)
        return np.r_[20 * (feet[:2, 2] - .036),
                     20 * (com[0] - feet[:2, 0].mean()),
                     8 * np.maximum(.16 - feet[2:, 2], 0),
                     10 * max(.12 - bottom, 0),
                     4 * (x[1] - target),
                     .02 * ((q - q0) / scale)]

    x0 = np.clip([.43, target, q0[1], q0[2], q0[7], q0[8]], lo + 1e-5, hi - 1e-5)
    solution = least_squares(residual, x0, bounds=(lo, hi), max_nfev=300)
    feet, com, bottom, q = pose(solution.x)
    result = {
        'requested_pitch_deg': float(angle_deg),
        'pitch_deg': float(-math.degrees(solution.x[1])),
        'base_z_m': float(solution.x[0]),
        'front_foot_z_m': feet[:2, 2].tolist(),
        'rear_foot_z_m': feet[2:, 2].tolist(),
        'com_front_foot_x_error_m': float(com[0] - feet[:2, 0].mean()),
        'base_collision_bottom_m': bottom,
        'joint_q': q.tolist(),
        'action': ((q - q0) / scale).tolist(),
        'geometry_only': True,
    }
    result['geometry_passed'] = bool(
        abs(result['pitch_deg'] - angle_deg) <= 5 and
        max(abs(v - .036) for v in result['front_foot_z_m']) <= .015 and
        min(result['rear_foot_z_m']) >= .12 and
        abs(result['com_front_foot_x_error_m']) <= .025 and bottom >= .1 and
        max(abs(v) for v in result['action']) <= 1.0001)
    return result


def open_loop_probe(parameters, pose, seconds=2.):
    """A fixed PD pose baseline; failure here does not prove dynamic infeasibility."""
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError('seconds must be positive and finite')
    model = _model(parameters)
    data = mujoco.MjData(model)
    sites, _, _ = _indices(model)
    target = np.asarray(pose['joint_q'])
    angle = -math.radians(pose['pitch_deg'])
    data.qpos[:3] = [0., 0., pose['base_z_m']]
    data.qpos[3:7] = [math.cos(angle / 2), 0., math.sin(angle / 2), 0.]
    data.qpos[7:] = target
    kp = np.array([parameters['joints'][n]['kp'] for n in SDK_JOINT_ORDER])
    kd = np.array([parameters['joints'][n]['kd'] for n in SDK_JOINT_ORDER])
    effort = np.array([parameters['joints'][n]['effort_limit'] for n in SDK_JOINT_ORDER])
    samples = 0
    for _ in range(round(seconds / model.opt.timestep)):
        torque = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
        data.qfrc_applied[6:] = torque
        mujoco.mj_step(model, data)
        feet = data.site_xpos[sites]
        current = math.degrees(2 * math.atan2(data.qpos[5], data.qpos[3]))
        samples += int(abs(current + pose['pitch_deg']) <= 10 and
                       np.all(np.abs(feet[:2, 2] - .036) <= .04) and
                       np.all(feet[2:, 2] >= .10))
    return {'seconds': seconds, 'hold_fraction': samples / max(1, round(seconds / model.opt.timestep)),
            'fixed_pd_only': True}
