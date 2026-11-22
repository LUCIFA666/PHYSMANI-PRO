"""Fixed TRAIN sensor-cache replay, with GT used only for final metrics.

No simulator/native imports, model fitting, threshold search, VAL or TEST.
The score configuration is supplied explicitly after the owned tracker hook
is available. Both variants see identical current/past public observations.
"""
import argparse
from collections import Counter
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.contracts import Pose, RobotState
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import (BodyTemplate, GeometryComponent,
    PriorProvenance, RegistrationConfig, MultiBodySE3Tracker)
from our_mani.multitask.mode_observer import PublicModeObserver
from our_mani.multitask.depth_pose_consistency import FreeSpaceConfig, current_depth_pose_cost
from our_mani.multitask.specs import get_task_spec
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c

VERSION = 'r10_fixed_TRAIN_current_depth_pose_replay_v1'
CASES = (('orientation_pilot_v2', 'place_cups_on_rotating_frame'),
    ('orientation_pilot_v2', 'place_cups_on_rotating_frame_high_speed'),
    ('orientation_pilot_v2', 'remove_cups_from_rotating_frame'),
    ('orientation_pilot_v2', 'remove_cups_from_rotating_frame_high_speed'),
    ('rigid_orientation_pilot_v3', 'place_cups_on_rotating_frame'),
    ('rigid_orientation_pilot_v3', 'place_cups_on_rotating_frame_high_speed'),
    ('rigid_orientation_pilot_v3', 'remove_cups_from_rotating_frame'))
ROLES = ('cup0', 'cup1', 'cup2')


def prepare(path, score_configuration):
    common.require(isinstance(score_configuration, dict) and score_configuration,
                   'explicit owned RegistrationConfig score settings required')
    common.require(set(score_configuration) <= {field.name for field in fields(RegistrationConfig)},
                   'tracker score hook unavailable or unknown configuration field')
    baseline = asdict(RegistrationConfig())
    scored = asdict(RegistrationConfig(**score_configuration))
    common.require(scored != baseline, 'scored tracker must differ from fixed original registration')
    inputs = {str(p): c.sha(p) for p in (Path(__file__).resolve(), ROOT/'our_mani/multitask/perception.py',
        ROOT/'our_mani/multitask/depth_pose_consistency.py', ROOT/'our_mani/multitask/axis_evidence.py',
        ROOT/'our_mani/multitask/mode_observer.py',
        ROOT/'our_mani/perception.py', ROOT/'our_mani/contracts.py', ROOT/'our_mani/multitask/specs.py')}
    cases = []
    for directory, task in CASES:
        base = c.BASE/directory
        report_path = base/task/'report.json'
        report, queue = common.read(report_path), common.read(base/'queue_report.json')
        row = next(row for row in queue['results'] if row['task'] == task)
        common.require(queue['complete'] is True and row['completed'] is True
                       and c.sha(report_path) == row['report_sha256'], 'original pilot parent receipt differs')
        sensor_path, geometry_path = ROOT/report['causal_sensor_observations']['path'], base/task/'TRAIN_geometry.json'
        common.require(report['complete'] is True and report['TRAIN_only'] is True and report['TEST_used'] is False
                       and report['teacher_used'] is False and report['actual_frames'] == 16
                       and report['causal_sensor_observations']['native_GT_or_task_outcome_included'] is False
                       and c.sha(sensor_path) == report['causal_sensor_observations']['sha256'], 'TRAIN current sensor cache invalid')
        asset = common.read(geometry_path)
        common.require(asset['split'] == 'train' and asset['task'] == task and asset['shared_with_baseline'] is True
                       and asset['world_pose_or_motion_schedule_in_asset'] is False
                       and asset['asset_sha256'] == hashlib.sha256(json.dumps({k:v for k,v in asset.items()
                           if k != 'asset_sha256'}, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                       'immutable TRAIN local geometry fingerprint differs')
        with np.load(sensor_path, allow_pickle=False) as arrays:
            common.require(len(arrays['times']) == 16 and np.allclose(np.diff(arrays['times']), .05),
                           'fixed16 current frame clock differs')
        for p in (report_path, base/'queue_report.json', sensor_path, geometry_path):
            inputs[str(p.resolve())] = c.sha(p)
        cases.append(dict(case_id=directory+'__'+task, task=task,
            parent_source_group='train:'+task+':'+report['source']['raw_episode_id'],
            report=str(report_path.relative_to(ROOT)), sensor=str(sensor_path.relative_to(ROOT)),
            geometry=str(geometry_path.relative_to(ROOT)), asset_version=asset['version'],
            public_cameras=report['initial_conditions']['public_initial_conditions']['cameras'],
            roles=list(ROLES), frames=16, direct_cup_registration_only=True,
            permanent_cup_rigid_group_assumed=False))
    plan = dict(version=VERSION, created_utc=c.utc(), cases=cases, inputs_sha256=inputs,
        baseline_registration=baseline, scored_registration=scored, score_configuration=score_configuration,
        fixed_free_space_parameters=asdict(FreeSpaceConfig()), expected_cases=7, expected_body_sequences=21,
        selection='all four existing orientation place/remove tasks and all three existing rigid-pilot cup tasks; no outcome selection',
        source_kind='previous TRAIN native-generated hold diagnostic observations, not original HDF5 replay',
        observations='each candidate/tracker receives only current frame RGBD/masks/calibration plus its own past state and current proprioception',
        GT_access='native GT fields accessed only after all replay predictions finish, for full SO3/position metrics',
        coupled_frames_and_duplicate_parent_sources_not_independent=True, TRAIN_only=True,
        no_native_or_GPU=True, no_VAL_TEST=True, no_parameter_search=True, annotation_promotion=False)
    plan['plan_sha256'] = c.canonical_sha(plan)
    c.write_json(path, plan)
    return plan


def verify(path):
    plan = common.read(path)
    common.require(plan['version'] == VERSION and plan['plan_sha256'] == c.canonical_sha(
        {k:v for k,v in plan.items() if k != 'plan_sha256'}), 'replay plan identity changed')
    for path, expected in plan['inputs_sha256'].items():
        common.require(c.sha(path) == expected, 'frozen replay input differs: '+path)
    return plan


def templates_from_original_train_asset(asset, roles):
    """Read historical v1/v2 local visual surfaces without upgrading assets.

    Historical IDs are valid only within their saved same-process TRAIN RGBD
    cache. No name-to-index colors, world poses or permanent cup binding enter.
    """
    result = []
    for role in roles:
        body = asset['bodies'][role]
        provenance = PriorProvenance('declared_cad', 'saved same-process TRAIN CAD:'+asset['asset_sha256'],
            body['frame_definition'], True, 'train', (asset['source_group'],), asset['asset_sha256'])
        components = tuple(GeometryComponent(row['name'], tuple(row['TRAIN_instance_ids']),
            np.asarray(row['points_body'])) for row in body['components'])
        result.append(BodyTemplate(role, components, provenance, orientation_observability='full'))
    return tuple(result)


def predict_case(case, configuration):
    """Deliberately receives no report/native GT namespace."""
    asset = common.read(ROOT/case['geometry'])
    templates = templates_from_original_train_asset(asset, case['roles'])
    tracker = MultiBodySE3Tracker(templates, task_spec=get_task_spec(case['task']),
        config=RegistrationConfig(**configuration))
    observer = PublicModeObserver(case['roles'], case['roles'], .15)
    calibrations = {name: CameraCalibration(np.asarray(row['intrinsics']), Pose.from_xyzw(row['camera_to_world']))
        for name, row in case['public_cameras'].items()}
    outputs, previous = [], []
    with np.load(ROOT/case['sensor'], allow_pickle=False) as stored:
        for i, now in enumerate(stored['times']):
            observation = {'timestamp': float(now), 'observation/calibrations': calibrations,
                'observation/camera_timestamps': {camera:float(now) for camera in calibrations}}
            for channel, prefix in (('observation/depths','depth'),('observation/instance_masks','mask'),('observation/images','rgb')):
                observation[channel] = {camera:stored[prefix+'_'+camera][i] for camera in calibrations}
            robot = RobotState(float(now), Pose.from_xyzw(stored['robot_pose'][i]), np.zeros(3), np.zeros(3),
                stored['robot_joints'][i], stored['robot_joint_velocities'][i],
                bool(stored['robot_gripper_open'][i]), bool(stored['robot_public_contact'][i]))
            modes, invalid = observer.observe(previous, robot)
            states, metadata = tracker.observe(observation, modes=modes)
            by_id = {state.object_id:state for state in states}
            views = {name:(calibrations[name],observation['observation/depths'][name]) for name in calibrations}
            bodies = {}
            for template in templates:
                role = template.role_id
                state = by_id.get(role)
                diagnostic = metadata['bodies'][role]
                cost = current_depth_pose_cost({component.name:component.points_body for component in template.components},
                    state.pose, views, reference_time=float(now), view_timestamps={name:float(now) for name in views}) if state else None
                bodies[role] = dict(pose_xyzw=state.pose.xyzw().tolist() if state else None,
                    visible=bool(state and state.visible), public_mode_identity_invalid=role in invalid,
                    causal_registration=diagnostic, independently_recomputed_current_depth_cost=cost)
            outputs.append(dict(frame=i, time=float(now), bodies=bodies, GT_used=False))
            previous = states
    return outputs


def final_metrics(case, predictions):
    # This is the first access to native target arrays; all variants/cases have
    # already emitted and saved their predictions before this function runs.
    report = common.read(ROOT/case['report'])
    result = {}
    for variant, frames in predictions.items():
        roles = {}
        for role in case['roles']:
            positions, angles, visible, observable, ambiguous, invalid = [], [], 0, 0, 0, 0
            count = Counter()
            per_frame = []
            for prediction, native_frame in zip(frames, report['frames']):
                row = prediction['bodies'][role]
                diagnostic = row['causal_registration']
                visible += int(row['visible'])
                observable += int(diagnostic.get('orientation_observable') is True)
                ambiguous += int(diagnostic.get('rotation_ambiguous') is True)
                invalid += int(diagnostic.get('registration_valid') is not True)
                metric = dict(frame=prediction['frame'], pose_present=row['pose_xyzw'] is not None)
                if row['pose_xyzw'] is not None:
                    gt = native_frame['metric_only']['native_current_ground_truth']
                    j = native_frame['metric_only']['body_ids'].index(role)
                    pose = Pose.from_xyzw(row['pose_xyzw'])
                    distance = float(np.linalg.norm(pose.position-np.asarray(gt['positions'][j])))
                    angle = float(Rotation.from_matrix(pose.rotation@np.asarray(gt['rotations'][j]).T).magnitude())
                    positions.append(distance);angles.append(angle)
                    metric.update(position_error_m=distance, full_SO3_error_rad=angle)
                    cost = row['independently_recomputed_current_depth_cost']
                    count.update({name:value for name,value in cost['counts'].items() if name != 'model_samples'})
                per_frame.append(metric)
            roles[role] = dict(metric_frames=len(positions), direct_visible_rows=visible,
                orientation_observable_rows=observable, ambiguous_rows=ambiguous, invalid_registration_rows=invalid,
                position_error_m=common.scalar_stats(positions), full_SO3_error_degrees=common.scalar_stats(np.rad2deg(angles)),
                full_SO3_error_below10deg_rows=int((np.asarray(angles)<np.deg2rad(10)).sum()),
                current_depth_evidence_counts=dict(count), frame_metrics=per_frame)
        result[variant] = roles
    return result


def run(plan_path, output):
    plan = verify(plan_path)
    output = c.destination(output)
    common.require(not output.exists(), 'fresh replay output required')
    output.mkdir(parents=True)
    predictions = {}
    for case in plan['cases']:
        variants = {}
        for name, configuration in (('original_full_ICP',plan['baseline_registration']),
                                    ('current_depth_scored_full_ICP',plan['scored_registration'])):
            variants[name] = predict_case(case, configuration)
            c.write_json(output/(case['case_id']+'__'+name+'.json'), variants[name])
        predictions[case['case_id']] = variants
        print('predicted '+case['case_id'], flush=True)
    verify(plan_path)
    # GT is evaluated only after the complete fixed prediction inventory.
    metrics = [dict(case_id=case['case_id'], task=case['task'], parent_source_group=case['parent_source_group'],
        metrics=final_metrics(case,predictions[case['case_id']])) for case in plan['cases']]
    result = dict(version=VERSION, complete=True, plan=common.pointer(plan_path),
        plan_sha256=plan['plan_sha256'], source=common.pointer(__file__), cases=metrics,
        all_predictions_completed_before_GT_metric_access=True, no_native_or_GPU=True,
        model_fitted=False, thresholds_tuned=False, annotation_promoted=False, TRAIN_only=True, VAL_TEST_used=False,
        limitation='short held TRAIN observations; repeated frames/parents are correlated; consistent occluded/unsupported poses can have zero free-space cost; this is not full-pose observability or policy SR validation')
    c.write_json(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true');action.add_argument('--run', action='store_true')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--score-configuration', type=Path, help='explicit RegistrationConfig additional settings as JSON')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.prepare:
        common.require(args.score_configuration is not None, 'score configuration JSON required')
        result = prepare(args.plan, common.read(args.score_configuration))
        print(json.dumps(dict(plan_sha256=result['plan_sha256'], cases=len(result['cases']))))
    else:
        common.require(args.out is not None, 'fresh offline output required')
        result = run(args.plan, args.out)
        print(json.dumps(dict(complete=result['complete'], cases=len(result['cases']))))


if __name__ == '__main__':
    main()
