"""Public feature ABI plus separately privileged actual-contact targets.

No file or model is opened by compilation. The caller supplies already frozen
public forecasts and a same-episode private TRAIN label stream after closure.
"""
from collections import Counter
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from ..local_mpc_p4_20261103.data import finite,rot6,unit_quaternion,PUBLIC_ROW_KEYS,ROBOT_KEYS
from ..local_mpc_20261102.records import HoldMetric
from . import SCHEMA,FRAME

ROOT=Path(__file__).resolve().parents[2]
PARTITION=ROOT/'artifacts/framework/local_mpc_20261102/cube_matched_motion_candidate_partition_20261104_v1/partition.json'
PARTITION_SHA256='38ac76250d382d36b3c28b01cc8551a25cc95edb850a3c21cc9bf1b47d3ade6a'
CLOCK_TOLERANCE=2e-5
FEATURES=('future_tokens','robot','geometry','context_c','task_e')


def load_partition(path=PARTITION):
    path=Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PARTITION_SHA256:
        raise ValueError('Exact pre-capture fixed sixteen-parent partition required')
    partition=json.loads(path.read_text())
    if partition['schema']!='Cube_matched_native_motion_candidate_whole_episode_partition_20261104_v1':
        raise ValueError('Exact partition schema required')
    return partition


def public_features(row, history, *, expected_forecast_sha256):
    """Read only approved current/past public fields; output SI tensors.

    ``history`` contains the actual rows up to and including ``row``. It is
    never interpolated. File existence and NPZ timestamp checks belong to the
    dataset compiler; this function verifies the causal temporal contract.
    """
    if type(expected_forecast_sha256)is not str or len(expected_forecast_sha256)!=64:
        raise ValueError('Explicit actual frozen new forecast SHA required')
    if not isinstance(row, dict) or set(row) != PUBLIC_ROW_KEYS:
        raise ValueError('exact public observer record required, without label channels')
    r, robot = row['result'], row['public_robot']
    if not isinstance(robot, dict) or set(robot) != ROBOT_KEYS:
        raise ValueError('whitelisted public robot fields required')
    if isinstance(robot['timestamp'],(bool,np.bool_)):
        raise ValueError('Actual public timestamp cannot be Boolean')
    now = float(robot['timestamp'])
    if not np.isfinite(now) or now < 0 or abs(float(r['timestamp']) - now) > 1e-8:
        raise ValueError('same-time public record required')
    if (r.get('mode') != 'FREE' or r.get('available') is not True
            or r.get('free_forecast_available') is not True or row['close_requested'] is not False):
        raise ValueError('pre-close available FREE forecast required')
    for flag in ('Native_inputs_used', 'future_observations_used', 'observations_fabricated',
                 'forecaster_is_injected_test_double'):
        if r.get(flag) is not False:
            raise ValueError('public causal frozen forecaster provenance required: ' + flag)
    forecast = r['forecast']; metadata = forecast['metadata']
    if (metadata.get('checkpoint_sha256') != expected_forecast_sha256
            or abs(float(metadata['observed_until']) - now) > 1e-8
            or forecast['object_ids'] != ['target']):
        raise ValueError('frozen target-only causal forecast required')
    n = metadata['history_frames']['target']
    if type(n) is not int or not 5 <= n <= 20 or len(history) < n:
        raise ValueError('5..20 actual causal history frames required')
    selected = history[-n:]
    times = finite([v['public_robot']['timestamp'] for v in selected], (n,), 'history times')
    if abs(times[-1] - now) > 1e-8 or not np.allclose(np.diff(times), .05, atol=CLOCK_TOLERANCE, rtol=0):
        raise ValueError('real contiguous 50ms history ending at anchor required')
    for item, timestamp in zip(selected, times):
        obj = item['result'].get('object_observation')
        if (not isinstance(obj, dict) or obj.get('object_id') != 'target'
                or not obj.get('visible', True) or float(obj.get('confidence', 1.)) < .5
                or abs(float(obj['timestamp']) - timestamp) > 1e-8 or not item.get('rgbd')):
            raise ValueError('actual visible timestamped public RGBD history required')
    expected = now + np.arange(17) * .05
    if not np.allclose(finite(forecast['times'], (17,), 'forecast times'), expected, atol=1e-8, rtol=0):
        raise ValueError('current plus16 frozen future samples required')
    p = finite(forecast['positions'], (17, 1, 3), 'future positions')[:, 0]
    matrices = finite(forecast['rotations'], (17, 1, 3, 3), 'future rotations')[:, 0]
    if (not np.allclose(matrices.transpose(0, 2, 1) @ matrices, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(matrices), 1., atol=1e-5, rtol=0)):
        raise ValueError('proper forecast rotations required')
    v = finite(forecast['linear_velocities'], (17, 1, 3), 'future velocity')[:, 0]
    w = finite(forecast['angular_velocities'], (17, 1, 3), 'future angular velocity')[:, 0]
    obj = r['object_observation']
    current = finite(obj['pose']['position'], (3,), 'current public body')
    if not np.allclose(current, p[0], atol=1e-6, rtol=0):
        raise ValueError('forecast current state differs from real public observation')
    points = np.asarray(obj['points_world'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('public geometry points required')
    c = finite(metadata['contexts']['target'], (16,), 'motion context')
    if np.any(c < 0) or not np.allclose(c.reshape(2, 8).sum(1), 1., atol=1e-5, rtol=0):
        raise ValueError('frozen SimNorm motion context required')
    q = finite(robot['q'], (7,), 'robot q'); dq = finite(robot['dq'], (7,), 'robot dq')
    tcp = finite(robot['tcp_position'], (3,), 'public TCP')
    tcp_rotation = unit_quaternion(robot['tcp_quaternion'], 'TCP quaternion').as_matrix()
    fingers = finite(robot['finger_open_amounts'], (2,), 'two fingers')
    if np.any((fingers < 0) | (fingers > 1)):
        raise ValueError('normalized actual two-finger sensors required')
    features = dict(
        future_tokens=np.concatenate(((expected-now)[:, None]/.8, p-current, rot6(matrices), v, w), axis=1),
        robot=np.r_[q, dq, tcp-current, rot6(tcp_rotation), fingers],
        geometry=np.r_[np.ptp(points, axis=0), float(obj.get('confidence', 1.))],
        context_c=c,
        # One semantic task cannot train a task classifier. This explicit
        # constant is a pilot placeholder, distinct from the motion context.
        task_e=np.full(16, 1./8),
    )
    if (not all(np.isfinite(value).all() for value in features.values())
            or not .5 <= features['geometry'][-1] <= 1.):
        raise ValueError('finite public features and valid confidence required')
    return {k: v.astype(np.float32) for k, v in features.items()}, dict(
        timestamp=now, real_history_timestamps=times.tolist(),
        real_history_rgbd=[item['rgbd'] for item in selected],
        checkpoint_sha256=expected_forecast_sha256, task_e_source='fixed_uniform_single_task_pilot_not_task_classifier',
        geometry_source='public_tracker_proxy_points_not_observed_full_shape', orientation_observed=False)


def _states(initial,ticks,episode_id):
    """Validate observed clock values; never create time from row index."""
    rows=[initial,*ticks];previous=None
    for row in rows:
        if row.get('episode_id')!=episode_id:
            raise ValueError('No cross-reset/pass/episode label joins')
        for name in ('time','physics_time','task_time'):
            value=row.get(name)
            if isinstance(value,bool)or not isinstance(value,(int,float))or not np.isfinite(value):
                raise ValueError('Recorded physical/task timestamps required')
        if any(type(row.get(name))is not bool for name in ('held','native_success')):
            raise ValueError('Actual attachment and original task Boolean labels required')
        finite(row['object_position'],(3,),'actual Cube centre label')
        finite(row['tcp_position'],(3,),'actual TCP label');unit_quaternion(row['tcp_quaternion'],'actual TCP label')
        if isinstance(row.get('requested_opening'),(bool,np.bool_))or row.get('requested_opening')not in(0.,1.):raise ValueError('Actual binary request label required')
        if abs((row['physics_time']-initial['physics_time'])-row['time'])>CLOCK_TOLERANCE:
            raise ValueError('Relative timestamp must come from recorded actual physics origin')
        if previous is not None:
            delta=row['physics_time']-previous['physics_time']
            if abs(delta-.05)>CLOCK_TOLERANCE or row['time']<=previous['time']:
                raise ValueError('Missing or non50ms actual physical transition')
            if abs(row['task_time']-previous['task_time']-delta)>CLOCK_TOLERANCE:
                raise ValueError('Task/physics clock gap cannot supply this candidate stream')
        previous=row
    if abs(initial['time'])>CLOCK_TOLERANCE or initial['held']:
        raise ValueError('Initially free physical origin required')
    return rows


def actual_successful_contacts(initial,ticks,*,episode_id):
    """A contact observation is not a force certificate; no expert-target label."""
    rows=_states(initial,ticks,episode_id);segments=[];active=[];attempt=0;previous=initial
    for row in rows[1:]:
        if previous['requested_opening']>=.5 and row['requested_opening']<.5:attempt+=1
        if row['held'] and row['requested_opening']<.5:
            if not attempt:raise ValueError('Attached label without a preceding explicit close edge')
            active.append(row)
        elif active:
            segments.append((attempt,active));active=[]
        previous=row
    if active:segments.append((attempt,active))
    accepted=[]
    for attempt_id,segment in segments:
        metric=HoldMetric(.5,.04,.02);metric.update(initial)
        for row in segment:metric.update(row)
        if not metric.passed or not any(row['native_success']for row in segment):continue
        first=segment[0]
        accepted.append(dict(episode_id=episode_id,attempt_id=attempt_id,time=first['time'],
            physics_time=first['physics_time'],object_position=first['object_position'],
            tcp_position=first['tcp_position'],tcp_quaternion=first['tcp_quaternion'],
            label_event='first_actual_50ms_attachment_observation_not_force_contact',
            acceptance='later_original_goal_and_P1_lifted_hold_in_same_uninterrupted_closed_attachment_segment',
            hold=metric.receipt()))
    return accepted


def public_reference_target(forecast,contact,*,anchor_time):
    """Actual time is retained, including >.8; only valid near time is supervised."""
    times=np.asarray(forecast['times'],float)
    rotations=finite(forecast['rotations'],(17,1,3,3),'public frozen forecast rotations')[:,0]
    if times.shape!=(17,)or not np.allclose(times,anchor_time+np.arange(17)*.05,atol=CLOCK_TOLERANCE,rtol=0):
        raise ValueError('Actual same-anchor frozen H17 forecast required')
    if not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-5,rtol=0)or not np.allclose(np.linalg.det(rotations),1.,atol=1e-5,rtol=0):
        raise ValueError('Proper public reference rotations required')
    at=float(contact['time']);tau=at-anchor_time
    if not np.isfinite(tau)or tau<=0.:raise ValueError('Future actual contact observation required')
    time_valid=.05-1e-8<=tau<=.8+1e-8 and at<=times[-1]+1e-9
    if at<=times[-1]:
        ref=Slerp(times,Rotation.from_matrix(rotations))([at]).as_matrix()[0]
        basis='causal_forecast_rotation_at_actual_contact'
    else:
        if (forecast.get('metadata',{}).get('full_orientation_observable',{}).get('target')is not False
                or not np.array_equal(rotations,np.broadcast_to(np.eye(3),(17,3,3)))):
            raise ValueError('Beyond-horizon pose label needs explicitly unobserved constant identity public basis')
        ref=rotations[0]
        basis='declared_unobserved_constant_identity_public_reference_not_far_rotation_prediction'
    delta=finite(contact['tcp_position'],(3,),'label TCP')-finite(contact['object_position'],(3,),'label Cube centre')
    tcp=unit_quaternion(contact['tcp_quaternion'],'actual contact TCP').as_matrix()
    return dict(target_position=(ref.T@delta).astype(np.float32),target_rotation=(ref.T@tcp).astype(np.float32),
        target_dt=np.float32(tau),target_time_valid=np.bool_(time_valid),target_frame=FRAME,public_reference_basis=basis,
        Native_object_orientation_used_in_target=False,GT_or_Native_fields_used_by_feature_forward=False)

def compile_contact_episode(public_rows,initial,ticks,*,parent_id,episode_id,expected_forecast_sha256,partition_path=PARTITION):
    """Compile fit/selection only. No contact => empty arrays and honest not_ready."""
    partition=load_partition(partition_path)
    matches=[r for r in partition['rows']if r['parent_id']==parent_id]
    if len(matches)!=1 or matches[0]['role']not in('fit','selection'):
        raise ValueError('Heldout/unknown parents cannot supply candidate fit or selection supervision')
    if not public_rows:raise ValueError('Every recorded public frame required')
    if len(public_rows)!=len(ticks)+1:raise ValueError('Keep initial and all failed/unknown physical frames')
    # Explicit whitelist strips only the same-episode envelope, never private fields.
    if any(set(row)!=PUBLIC_ROW_KEYS|{'episode_id'}for row in public_rows):
        raise ValueError('Exact public row plus episode identity; private/unknown fields forbidden')
    projected=[{key:row[key]for key in PUBLIC_ROW_KEYS}for row in public_rows]
    # Build all possible causal feature rows without reading private contact labels.
    features=[];rejected=Counter()
    for index,row in enumerate(public_rows):
        if row.get('episode_id')!=episode_id:raise ValueError('No cross-pass public feature joins')
        try:value,provenance=public_features(projected[index],projected[:index+1],expected_forecast_sha256=expected_forecast_sha256)
        except (KeyError,ValueError)as error:
            features.append(None);rejected[str(error)]+=1;continue
        features.append((value,provenance))
    contacts=actual_successful_contacts(initial,ticks,episode_id=episode_id)
    values=[];ledger=[]
    first_close=next((state['time']for state in ticks if state['requested_opening']<.5),float('inf'))
    for index,(row,state)in enumerate(zip(public_rows,[initial,*ticks])):
        now=float(row['public_robot']['timestamp'])
        if abs(now-state['time'])>CLOCK_TOLERANCE:raise ValueError('Public/private actual timestamp mismatch; no nearest-frame matching')
        if features[index]is None or now>=first_close-1e-8:continue
        value,provenance=features[index]
        for contact in contacts[:1]:
            if contact['time']<=now+1e-8:continue
            target=public_reference_target(row['result']['forecast'],contact,anchor_time=now)
            values.append(dict(**value,**{k:target[k]for k in('target_position','target_rotation','target_dt','target_time_valid')}))
            ledger.append(dict(parent_id=parent_id,episode_id=episode_id,role=matches[0]['role'],public_index=index,
                timestamp=now,contact_time=contact['time'],attempt_id=contact['attempt_id'],target_frame=FRAME,
                feature_provenance=provenance,target_time_valid=bool(target['target_time_valid']),public_reference_basis=target['public_reference_basis'],Native_object_orientation_used_in_target=False))
    arrays={k:np.stack([v[k]for v in values])for k in values[0]}if values else{}
    return arrays,dict(schema=SCHEMA,parent_id=parent_id,episode_id=episode_id,role=matches[0]['role'],
        candidate_training_ready=bool(values),public_frames_total=len(public_rows),private_frames_total=len(ticks)+1,
        accepted_contact_count=len(contacts),supervised_anchors=len(values),time_supervised_anchors=sum(bool(v['target_time_valid'])for v in values),pose_only_far_anchors=sum(not bool(v['target_time_valid'])for v in values),rejected_feature_reasons=dict(rejected),
        all_failed_and_unknown_frames_retained_in_episode_denominator=True,
        unmatched_candidates_are_negative=False,target_frame=FRAME,ledger=ledger,supervision_scope='All available causal visible FREE anchors strictly before first close in successful episodes; far real tau retained and time loss masked, never clamped',
        frozen_forecast_checkpoint_sha256=expected_forecast_sha256,
        other_contacts_or_failed_attempts_not_success_probability_labels=True)


def source_map():
    files=[Path(__file__).with_name(name)for name in('__init__.py','data.py','losses.py','evaluation.py')]
    files.extend(ROOT/name for name in('our_mani/local_mpc_p4_20261103/data.py','our_mani/local_mpc_p4_20261103/model.py','our_mani/local_mpc_20261102/records.py'))
    files.append(PARTITION)
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}


def combine_episode_results(results):
    """Preserve every declared episode report, including zero-positive failures.

    The caller passes every preassigned same-role episode, not just successes.
    source-parent weighting is computed only after preserving this denominator.
    """
    if not results:raise ValueError('All fixed role episodes required')
    reports=[report for _,report in results]
    roles={r['role']for r in reports}
    if len(roles)!=1 or not roles<=set(('fit','selection')):raise ValueError('Never mix roles or include heldout')
    partition=load_partition();expected=[r['parent_id']for r in partition['rows']if r['role']in roles]
    actual=[r['parent_id']for r in reports]
    if len(set(actual))!=len(actual)or set(actual)!=set(expected):raise ValueError('Every fixed whole parent including failures required')
    counts={report['parent_id']:len(arrays.get('target_dt',[]))for arrays,report in results}
    parents=[p for p in expected if counts[p]]
    nonempty=[arrays for arrays,_ in results if len(arrays.get('target_dt',[]))]
    combined={k:np.concatenate([a[k]for a in nonempty])for k in nonempty[0]}if nonempty else{}
    if combined:
        ids=np.concatenate([np.full(counts[report['parent_id']],parents.index(report['parent_id']),dtype=np.int64)for arrays,report in results if counts[report['parent_id']]])
        combined['parent_index']=ids
        combined['sample_weight']=np.asarray([1./counts[parents[index]]for index in ids],np.float32)
    return combined,dict(schema=SCHEMA,role=next(iter(roles)),declared_parent_denominator=len(expected),
        supervised_parents=parents,failed_or_no_contact_parents=[p for p in expected if not counts[p]],
        candidate_training_ready=bool(combined),supervised_anchors=sum(counts.values()),
        episodes=reports,target_frame=FRAME,all_fixed_episodes_preserved=True,
        no_failed_or_unmatched_candidate_negative_labels=True)
