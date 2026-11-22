"""Complete measured robot/RGBD feedback and an explicit legacy NN projection."""
import numpy as np
from ..generic_reference_mpc_20261104_v1.public_observation import (
    CAMERAS,pack_public_observation,validate_policy_packet)

SCHEMA='physmani_public_robot_RGBD_feedback_20261104_v1'
KEYS={'schema','timestamp','physics_timestamp','task_timestamp','physics_time_origin','task_time_origin',
      'rgb','depth_m','K','T','camera_clips','camera_names','render_modes','q','dq','tcp_pose',
      'finger_joint_positions','finger_joint_velocities','finger_open_amounts','finger_open_amounts_within_nominal','input_semantics'}
SEMANTICS='current_public_RGBD_calibration_and_measured_robot_only_no_object_contact_success'


def _clock(value,name):
    if isinstance(value,(bool,np.bool_))or not isinstance(value,(int,float,np.number))or not np.isfinite(value):
        raise ValueError('Actual finite clock readback required: '+name)
    return float(value)


def _vector(value,size,name):
    out=np.asarray(value,dtype=np.float64)
    if out.shape!=(size,)or not np.isfinite(out).all():raise ValueError('Actual finite '+name+' required')
    return out.copy()


def pack_feedback_snapshot(*,physics_timestamp,task_timestamp,physics_time_origin,task_time_origin,
        rgb,depth_m,K,T,camera_clips,q,dq,tcp_pose,finger_joint_positions,finger_joint_velocities,
        finger_open_amounts,render_modes):
    clocks={key:_clock(value,key)for key,value in dict(physics_timestamp=physics_timestamp,
        task_timestamp=task_timestamp,physics_time_origin=physics_time_origin,task_time_origin=task_time_origin).items()}
    timestamp=clocks['physics_timestamp']-clocks['physics_time_origin']
    task_delta=clocks['task_timestamp']-clocks['task_time_origin']
    if timestamp<0 or task_delta<0 or abs(task_delta-timestamp)>2e-5:
        raise ValueError('Original paired physics/task clock deltas must be nonnegative and synchronized')
    old=pack_public_observation(timestamp=timestamp,rgb=rgb,depth_m=depth_m,K=K,T=T,
        camera_clips=camera_clips,q=q,tcp_pose=tcp_pose)
    if not isinstance(render_modes,(tuple,list))or len(render_modes)!=4 or any(type(x)is not str or not x for x in render_modes):
        raise ValueError('Actual renderer name for each original view required')
    out=dict(schema=SCHEMA,timestamp=timestamp,**clocks,
        **{key:old[key].copy()for key in ('rgb','depth_m','K','T','camera_clips')},
        camera_names=CAMERAS,render_modes=tuple(render_modes),q=_vector(q,7,'arm q7'),dq=_vector(dq,7,'arm dq7'),
        tcp_pose=_vector(tcp_pose,7,'TCP xyzw7'),finger_joint_positions=_vector(finger_joint_positions,2,'finger q2'),
        finger_joint_velocities=_vector(finger_joint_velocities,2,'finger dq2'),
        finger_open_amounts=_vector(finger_open_amounts,2,'actual mechanical opening2'),input_semantics=SEMANTICS)
    out['finger_open_amounts_within_nominal']=bool(np.all((out['finger_open_amounts']>=0)&(out['finger_open_amounts']<=1)))
    # Public state is observed, so actual slight joint-limit excursions are
    # retained; action limits belong to the original execution qualifier.
    for value in out.values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    return out


def validate_feedback_snapshot(snapshot):
    if type(snapshot)is not dict or set(snapshot)!=KEYS or snapshot.get('schema')!=SCHEMA or snapshot.get('input_semantics')!=SEMANTICS:
        raise ValueError('Exact public measured-feedback whitelist required; private fields rejected')
    if tuple(snapshot['camera_names'])!=CAMERAS:raise ValueError('Original four real view identities required')
    rebuilt=pack_feedback_snapshot(**{key:snapshot[key]for key in (
        'physics_timestamp','task_timestamp','physics_time_origin','task_time_origin',
        'rgb','depth_m','K','T','camera_clips','q','dq','tcp_pose','finger_joint_positions',
        'finger_joint_velocities','finger_open_amounts','render_modes')})
    if type(snapshot['finger_open_amounts_within_nominal'])is not bool or snapshot['finger_open_amounts_within_nominal']!=rebuilt['finger_open_amounts_within_nominal']:
        raise ValueError('Mechanical opening diagnostic must match actual unmodified readback')
    if _clock(snapshot['timestamp'],'relative physical timestamp')!=rebuilt['timestamp']:raise ValueError('Measured absolute and relative physical clocks disagree')
    return snapshot


def legacy_policy_packet(snapshot):
    """Old trained public ABI: q7+TCP7, contaminated opening still0/maskFalse.

    Extra measured dq/fingers remain available to the controller through the
    full snapshot; they are never silently appended to trained NN features.
    """
    validate_feedback_snapshot(snapshot)
    return validate_policy_packet(pack_public_observation(timestamp=snapshot['timestamp'],
        rgb=snapshot['rgb'],depth_m=snapshot['depth_m'],K=snapshot['K'],T=snapshot['T'],
        camera_clips=snapshot['camera_clips'],q=snapshot['q'],tcp_pose=snapshot['tcp_pose']))
