"""Causal orientation channel for original no-action buzz hold acquisitions.

Uses public current RGBD centres and a TRAIN-fitted static pair prior only.
The caller must prove this is a no-action hold acquisition. Once a public
contact, close, or non-hold command occurs, this channel is disabled forever.
It does not grant runtime post-action pair rigidity or absolute SO3 accuracy.
"""
import numpy as np
from .two_center_pose import PublicCenter,estimate_two_center_pose
from .perception import _world_twist
from ..contracts import Pose

SCHEMA='r10_causal_public_two_center_no_action_hold_rotation_channel_v76'


def derive_hold_observations(prior,arrays):
    ids=list(arrays['body_ids']);times=np.asarray(arrays['times'],float);n=len(times)
    if set(ids)!={'frame','wand'} or len(ids)!=2:raise ValueError('exact buzz pair required')
    positions=np.asarray(arrays['positions']);valid=np.asarray(arrays['valid'],bool);modes=np.asarray(arrays['modes'])
    mode_valid=np.asarray(arrays['mode_valid'],bool)
    opening=np.asarray(arrays['robot_gripper_open'],bool);contact=np.asarray(arrays['robot_contact'],bool)
    commands=np.asarray(arrays['public_command_kind']).astype(str)
    rotations=np.broadcast_to(np.eye(3),(n,2,3,3)).copy();angular=np.zeros((n,2,3));qualified=np.zeros((n,2),bool)
    history={role:[] for role in ids};action_or_contact_seen=False;reasons=[]
    for i,now in enumerate(times):
        action_or_contact_seen=action_or_contact_seen or bool(contact[i]) or not bool(opening[i]) or commands[i]!='hold'
        if action_or_contact_seen:
            reasons.append('public_action_contact_or_close_disables_pair_channel');[h.clear() for h in history.values()];continue
        centers={role:PublicCenter(positions[i,j],bool(valid[i,j]),bool(mode_valid[i,j]),
            ('external','attached','released')[int(modes[i,j])],float(now)) for j,role in enumerate(ids)}
        estimate=estimate_two_center_pose(prior,centers,current_timestamp=float(now))
        if not estimate.valid:
            reasons.append(estimate.reason);[h.clear() for h in history.values()];continue
        for j,role in enumerate(ids):
            rotations[i,j]=getattr(estimate,role+'_rotation');qualified[i,j]=True
            h=history[role]
            if h and abs(now-h[-1][0]-.05)>1e-8:h.clear()
            h.append((float(now),Pose(positions[i,j],rotations[i,j])));del h[:-20]
            if len(h)>1:angular[i,j]=_world_twist(h)[1]
        reasons.append('conditional_no_action_hold_pair_orientation')
    return dict(rotations=rotations,angular_velocities=angular,rotation_observable=qualified),dict(schema=SCHEMA,
        public_fields_only=True,post_action_pair_channel_allowed=False,qualified_body_frames=int(qualified.sum()),
        total_body_frames=n*2,per_frame_reasons=reasons,static_TRAIN_prior_sha256=prior.model_sha256,
        current_RGBD_unique_SO3_measurement_claimed=False)
