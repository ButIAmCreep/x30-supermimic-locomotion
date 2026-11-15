"""Validated initialization from the completed X30 locomotion course."""

import torch

from go2_supermimic.robots.x30 import locomotion as loc
from go2_supermimic.robots.x30.locomotion_train import contract as source_contract


def transfer_from_curriculum(checkpoint, actor, world_model, parameters):
    try:
        old = checkpoint['contract']
        if old.get('task') != loc.TASK or old != source_contract(old['parameters']):
            raise ValueError('Source must be an intact X30-Locomotion-Curriculum checkpoint')
        if old['parameters'] != parameters:
            raise ValueError('Source and target physical configuration differ')
        for model, name in ((actor, 'actor'), (world_model, 'world_model')):
            source, target = checkpoint[name], model.state_dict()
            if set(source) != set(target):
                raise ValueError(f'Incompatible {name} keys')
            for key, value in source.items():
                if value.shape != target[key].shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'Incompatible or nonfinite {name}.{key}')
                if key.endswith('.var') and not bool((value >= 0).all()):
                    raise ValueError(f'Invalid {name} normalization variance')
                if key.endswith('.count') and not bool((value > 0).all()):
                    raise ValueError(f'Invalid {name} normalization count')
                if key in ('action_gain', 'history_gain', 'action_ratio') and not torch.equal(value, target[key]):
                    raise ValueError(f'Physical action mapping mismatch: {key}')
        actor.load_state_dict(checkpoint['actor'])
        world_model.load_state_dict(checkpoint['world_model'])
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('Malformed X30 locomotion source checkpoint') from exc
    return {'task': old['task'], 'iteration': checkpoint.get('iteration'),
            'physical_action_mapping': 'identical 12-joint action and observation contract'}
