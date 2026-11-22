"""Public-only teacher shadow replay through the exact newly trained GRU.

This module never opens label files. Only Root's separately authorized CPU
prediction phase calls predict_public_episode; import/default checks do no NN.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
from ..contracts import ObjectState,Pose
from ..local_mpc_20261102.records import plain
from ..cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics


def _public_frame(packet,archived_result):
    """All object quantities here are current public measurements, never labels."""
    target=packet['target'];now=packet['timestamp'];robot=packet['robot']
    if packet.get('Native_labels_in_input')is not False or type(packet.get('close_requested'))is not bool:
        raise ValueError('Actual public-only row and explicit issued close intent required')
    if isinstance(now,bool)or not np.isfinite(now):raise ValueError('Recorded current public timestamp required')
    tcp=np.asarray(robot['tcp_pose'],float)
    if tcp.shape!=(7,)or not np.isfinite(tcp).all():raise ValueError('Actual public TCP required')
    public_robot=dict(timestamp=now,tcp_position=tcp[:3].tolist(),tcp_quaternion=tcp[3:].tolist(),
        q=deepcopy(robot['q']),dq=deepcopy(robot['dq']),finger_open_amounts=deepcopy(robot['finger_open_amounts']))
    obj=None;reason=None
    if target.get('available')is True:
        old=archived_result.get('object_observation')
        if not isinstance(old,dict):raise ValueError('Available current public target needs recorded actual support cloud')
        p=np.asarray(target['position_world'],float);points=np.asarray(old['points_world'],float)
        if p.shape!=(3,)or points.ndim!=2 or points.shape[1]!=3 or len(points)<3 or not np.isfinite(p).all()or not np.isfinite(points).all():
            raise ValueError('Current public position and actual support points required')
        if abs(float(old['timestamp'])-now)>1e-8 or not np.allclose(old['pose']['position'],p,atol=1e-6,rtol=0):
            raise ValueError('Current support must be the same real public measurement, not last body')
        obj=ObjectState('target',now,Pose(p,np.eye(3)),points,
            linear_velocity=np.zeros(3),angular_velocity=np.zeros(3),confidence=target['confidence'],visible=True)
    else:
        if target.get('available')is not False or target.get('position_world')is not None or target.get('confidence')is not None:
            raise ValueError('UNKNOWN remains null, never filled from labels or forecast')
        reason='current_public_unavailable'
    return public_robot,obj,reason


def predict_public_episode(public_packets,archived_public_outputs,dynamics,*,episode_id,expected_forecast_sha256):
    """Return every public frame, with fresh causal predictions only where valid.

    Geometry derives from archived same-frame public support, c16 and future17
    only from the exact new provider. Saved old-ebec predictions are not read
    as feature values. Real private teacher labels are not an argument.
    """
    if type(dynamics)is not CubeResidualMotionDynamics or dynamics.checkpoint_sha256!=expected_forecast_sha256:
        raise TypeError('Exact actual newly trained frozen Cube forecaster required')
    dynamics._assert_live();dynamics.reset()
    if not public_packets or len(public_packets)!=len(archived_public_outputs):raise ValueError('All actual public rows required')
    lookup={float(row['result']['timestamp']):row['result']for row in archived_public_outputs}
    if len(lookup)!=len(archived_public_outputs):raise ValueError('Duplicate public output timestamps')
    rows=[];ledger=[];calls=0;closed=False
    for index,packet in enumerate(public_packets):
        if packet.get('episode_id')!=episode_id or packet.get('index')!=index:raise ValueError('One complete physical teacher episode only')
        now=float(packet['timestamp'])
        if now not in lookup:raise ValueError('Exact same public timestamp join; no interpolation')
        robot,obj,reason=_public_frame(packet,lookup[now]);close=packet['close_requested'];closed=closed or close
        result=dict(timestamp=now,mode='UNKNOWN',available=False,free_forecast_available=False,
            Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
            forecaster_is_injected_test_double=False,object_observation=None if obj is None else plain(obj),
            rotation_observable=False,orientation_is_identity_prior=True)
        if obj is None or closed:
            dynamics.reset();reason=reason or'first_close_latched_no_free_training_forecast'
        else:
            dynamics.observe([obj],modes={'target':'external'},orientation_observability={'target':False})
            if len(dynamics.history.get('target',()))<5:reason='fewer_than_five_real_visible_frames'
            else:
                forecast=dynamics.predict([obj],horizon=.8,dt=.05);calls+=1
                if forecast.metadata['checkpoint_sha256']!=expected_forecast_sha256:raise ValueError('Fresh provider authority changed')
                result.update(mode='FREE',available=True,free_forecast_available=True,forecast=plain(forecast))
        rows.append(dict(episode_id=episode_id,index=index,result=result,public_robot=robot,
            close_requested=close,rgbd=packet['raw_RGBD']))
        ledger.append(dict(index=index,timestamp=now,current_public_available=obj is not None,
            new_forecast_available=result['free_forecast_available'],reason=reason,
            actual_GRU_forward_calls_this_frame=int(result['free_forecast_available'])))
    dynamics._assert_live()
    return rows,dict(episode_id=episode_id,public_frames=len(rows),GRU_forward_batch_calls=calls,
        input_and_forecast_frames='current public world coordinates / unobserved identity orientation',
        new_forecast_checkpoint_sha256=expected_forecast_sha256,private_label_files_read=0,
        old_ebec_forecasts_used_as_inputs=False,unknown_or_postclose_rows_retained=True,ledger=ledger,
        actual_forward_count_source='one exact runtime.predict->one model.forward per eligible frame; no context_snapshot extra forward')


def normalize_private_labels(rows,*,parent_id,episode_id,role):
    """Explicit P0 TRAINLabelRecorder fields; called only after public freeze."""
    result=[]
    for index,row in enumerate(rows):
        if row.get('parent_id')!=parent_id or row.get('episode_id')!=episode_id or row.get('role')!=role or row.get('reset_boundary_id')!=episode_id or row.get('physics_tick')!=index:
            raise ValueError('No cross-parent/reset/pass private joins')
        tcp=np.asarray(row['tcp_pose_world_xyzw'],float)
        if tcp.shape!=(7,):raise ValueError('Actual measured TCP pose required')
        result.append(dict(episode_id=episode_id,time=row['actual_relative_physics_timestamp'],
            physics_time=row['physics_time_absolute'],task_time=row['task_time_absolute'],held=row['held'],
            native_success=row['native_success'],requested_opening=row['requested_opening'],
            object_position=row['target_position_world'],tcp_position=tcp[:3].tolist(),tcp_quaternion=tcp[3:].tolist()))
    if not result:raise ValueError('Initial and all committed actual private states required')
    return result[0],result[1:]
