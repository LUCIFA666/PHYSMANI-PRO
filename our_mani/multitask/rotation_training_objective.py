"""Experimental rotation emphasis; original dynamics implementation unchanged.

The sampler keeps the same draw budget for every original TRAIN parent. New
coefficients change only the existing masked rotation/angular-velocity terms.
"""
from collections import defaultdict
import math

import numpy as np

from .fitting import (balanced_parent_indices as original_parent_indices,
    loss_per_window as original_loss_per_window,masked_window_mean)

SCHEMA='r10_same_data_rotation_emphasis_objective_v1'


def positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<1:
        raise ValueError(name+' must be finite and at least one')
    return float(value)


def moving_rotation_window(row,minimum_angular_speed=.05):
    """TRAIN-only sampling strata; future labels never enter model history."""
    if row['source_key'][0]!='train':raise ValueError('rotation sampler accepts TRAIN only')
    mask=np.asarray(row['rotation_mask'],dtype=bool)
    if not mask.any():return False
    values=np.asarray(row['targets']['angular_velocities'])
    return bool(np.any(np.linalg.norm(values[mask],axis=-1)>=minimum_angular_speed))


def balanced_parent_indices(samples,windows_per_parent,rng,*,rotation_sampling_multiplier=1.):
    multiplier=positive(rotation_sampling_multiplier,'rotation sampling multiplier')
    if multiplier==1.:
        return original_parent_indices(samples,windows_per_parent,rng)
    if windows_per_parent<1:raise ValueError('positive per-parent example budget required')
    groups=defaultdict(list)
    for index,row in enumerate(samples):
        if row['source_key'][0]!='train':raise ValueError('rotation sampler accepts TRAIN only')
        groups[row['source_key']].append(index)
    result=[]
    for _,indices in sorted(groups.items()):
        weights=np.asarray([multiplier if moving_rotation_window(samples[index]) else 1. for index in indices])
        # Retain the original replacement rule and per-parent draw budget.
        selected=rng.choice(indices,windows_per_parent,
            replace=len(indices)<windows_per_parent,p=weights/weights.sum())
        result.append(selected)
    result=np.concatenate(result);rng.shuffle(result)
    return result


def loss_per_window(model,data,steps=22,prediction=None,*,rotation_loss_multiplier=1.,angular_loss_multiplier=1.):
    rmul=positive(rotation_loss_multiplier,'rotation loss multiplier')
    wmul=positive(angular_loss_multiplier,'angular loss multiplier')
    prediction=prediction if prediction is not None else model(data['history'],data['length'],data['state'],steps=steps)
    result=original_loss_per_window(model,data,steps,prediction)
    if rmul==wmul==1.:return result
    mask=data['rotation_mask'][:,:steps+1]
    targets=data['targets']
    rotation=((prediction['rotations']-targets['rotations'][:,:steps+1])/.5).square().mean((-1,-2))
    angular=((prediction['angular_velocities']-targets['angular_velocities'][:,:steps+1])/2.).square().mean(-1)
    return result+.1*(rmul-1)*masked_window_mean(rotation,mask)+.01*(wmul-1)*masked_window_mean(angular,mask)
