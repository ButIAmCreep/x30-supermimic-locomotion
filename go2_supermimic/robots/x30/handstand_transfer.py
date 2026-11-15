"""Atomic cross-task transfer from the X30 speed course to front-support skill."""

import torch

from go2_supermimic.robots.x30.locomotion_transfer import action_gain
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.speed_env import TASK as SPEED_TASK
from go2_supermimic.robots.x30.speed_train import contract as speed_contract


def _physical_signature(parameters):
    return {
        'standing_height': parameters['standing_height'],
        'model_mass_kg': parameters['model_mass_kg'],
        'mass_scaling': parameters.get('mass_scaling', 'source'),
        'actuator_model': parameters.get('actuator_model', 'ideal_pd'),
        'delay_physics_steps': parameters['delay_physics_steps'],
        'joints': {n: {k: v for k, v in parameters['joints'][n].items() if k != 'action_scale'}
                   for n in SDK_JOINT_ORDER},
    }


def validate_fixed_buffers(checkpoint, actor, parameters):
    """Protect immutable physical unit conversions on same-task load and eval."""
    expected = actor.state_dict()
    try:
        for key in ('walk_to_new', 'history_to_old',
                    'walk.action_gain', 'walk.history_gain'):
            value = checkpoint['actor'][key]
            if not torch.is_tensor(value) or not torch.equal(value.cpu(), expected[key].cpu()):
                raise ValueError(f'Handstand checkpoint changed fixed actor mapping {key}')
        ratio = checkpoint['world_model']['action_ratio']
        physical = 1 / action_gain(parameters)
        if not torch.is_tensor(ratio) or not torch.equal(ratio.cpu(), physical.cpu()):
            raise ValueError('Handstand checkpoint changed world-model action mapping')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Handstand checkpoint lacks fixed physical mappings') from exc


def transfer_from_speed(checkpoint, actor, world_model, parameters):
    try:
        old = checkpoint['contract']
        source_parameters = old['parameters']
        if old.get('task') != SPEED_TASK or old != speed_contract(source_parameters):
            raise ValueError('Source must be an intact X30-Locomotion-Speed2 checkpoint')
        if _physical_signature(source_parameters) != _physical_signature(parameters):
            raise ValueError('Motor, pose or model physics differ from the walking source')
        expected_ratio = 1 / action_gain(source_parameters)
        expected_walk_ratio = (torch.tensor([source_parameters['joints'][n]['action_scale'] for n in SDK_JOINT_ORDER]) /
                               torch.tensor([parameters['joints'][n]['action_scale'] for n in SDK_JOINT_ORDER]))
        if not torch.equal(actor.walk_to_new.cpu(), expected_walk_ratio):
            raise ValueError('Walking action scale mapping differs from source')
        for target, source, name, skipped in (
            (actor.walk.state_dict(), checkpoint['actor'], 'actor', set()),
            (world_model.state_dict(), checkpoint['world_model'], 'world_model', {'action_ratio'}),
        ):
            if set(target) != set(source):
                raise ValueError(f'Incompatible {name} keys')
            for key, value in source.items():
                if value.shape != target[key].shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'Invalid {name}.{key}')
                if key.endswith('.var') and not bool((value >= 0).all()):
                    raise ValueError(f'Negative {name} normalization variance')
                if key.endswith('.count') and not bool((value > 0).all()):
                    raise ValueError(f'Invalid {name} normalization count')
                if key not in skipped and key in ('action_gain', 'history_gain') and not torch.equal(value, target[key]):
                    raise ValueError(f'Invalid physical action map {name}.{key}')
        if not torch.equal(checkpoint['world_model']['action_ratio'].cpu(), expected_ratio.cpu()):
            raise ValueError('Source world model action map differs from contract')
        actor.walk.load_state_dict(checkpoint['actor'])
        target_wm = world_model.state_dict()
        for key, value in checkpoint['world_model'].items():
            if key != 'action_ratio':
                target_wm[key] = value
        world_model.load_state_dict(target_wm)
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('Malformed X30 speed source checkpoint') from exc
    return {'task': old['task'], 'iteration': checkpoint.get('iteration'),
            'walking_branch': 'frozen; observation action history and output actions mapped to new joint offsets',
            'world_model': 'copied; both action inputs mapped to prior physical joint offsets'}
