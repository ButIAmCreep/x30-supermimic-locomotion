"""Explicit, validated old-trot to wider-action initialization."""
from copy import deepcopy
import torch
from go2_supermimic.robots.x30.model import SDK_JOINT_ORDER
from go2_supermimic.robots.x30.trot_train import contract as trot_contract

REFERENCE_SCALES=(.10,.25,.35)*4


def action_gain(parameters):
    scales=torch.tensor([parameters['joints'][k]['action_scale'] for k in SDK_JOINT_ORDER])
    gain=torch.tensor(REFERENCE_SCALES)/scales
    if not bool(((gain>0)&(gain<=1)).all()):raise ValueError('Curriculum action range must contain old trot range')
    return gain


def reference_parameters(parameters):
    old=deepcopy(parameters)
    for key,value in zip(SDK_JOINT_ORDER,REFERENCE_SCALES):old['joints'][key]['action_scale']=value
    return old


def transfer_from_trot(checkpoint,actor,world_model,parameters):
    try:
        old=checkpoint['contract']
        if old.get('task')!='X30-Trot-Flat' or old!=trot_contract(old['parameters']):
            raise ValueError('Initialization requires an unchanged compatible X30-Trot-Flat contract')
        physical=deepcopy(old['parameters']);physical.pop('parameter_source',None)
        expected=reference_parameters(parameters);expected.pop('parameter_source',None)
        if physical!=expected:raise ValueError('Initialization dynamics/pose/action reference differs')
        for model,key in ((actor,'actor'),(world_model,'world_model')):
            target=model.state_dict();source=checkpoint[key]
            required={k for k in target if not k.startswith('residual.') and k not in ('action_gain','history_gain','action_ratio')}
            if set(source)!=required:raise ValueError(f'Incompatible {key} keys')
            if any(source[k].shape!=target[k].shape or not bool(torch.isfinite(source[k]).all()) for k in required):
                raise ValueError(f'Incompatible or nonfinite {key} tensors')
            for name,value in source.items():
                if name.endswith('.var') and not bool((value>=0).all()):raise ValueError('Negative source normalization variance')
                if name.endswith('.count') and not bool((value>0).all()):raise ValueError('Invalid source normalization count')
        actor.load_state_dict(checkpoint['actor'],strict=False)
        world_model.load_state_dict(checkpoint['world_model'],strict=False)
    except (KeyError,TypeError,RuntimeError) as exc:raise ValueError('Malformed initialization checkpoint') from exc
    return {'task':old['task'],'iteration':checkpoint.get('iteration'),'physical_action_mapping':'old/new scales; history and both WM action inputs converted'}
