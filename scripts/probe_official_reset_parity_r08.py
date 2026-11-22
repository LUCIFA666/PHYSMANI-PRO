"""Already exposed official-reset input inspection; no policy/outcome evaluation.

Two tasks x two launchers, independent processes. Only the released reset's
own staging steps run. Temporary wrappers read current snapshots after its
stages and do not add physics steps or modify frozen/external files.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.sensor_alignment.author_rgbd import AlignedOfficialSimulation, public_initial_conditions
from our_mani.aligned.simulation import physics_time
from our_mani.benchmark_protocol import file_sha256
from summarize_aligned_r08_v6 import compare_conditions

OUTPUT = ROOT/'artifacts/framework/aligned_r08/official_reset_parity_probe'
CONFIG = ROOT/'config/framework/aligned_protocol_r08.json'
MANIFEST = ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
LAUNCHERS = ('ours_cpu', 'baseline_gpu')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def current_snapshot(env, stage):
    from pyrep.const import ObjectType
    before = physics_time(), float(env.task.t)
    robot_shapes = env.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
    robot_shapes += env.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
    excluded = {shape.get_handle() for shape in robot_shapes} | {
        env.arm.get_handle(), env.gripper.get_handle(), env.task.target_block.get_handle()}
    fixtures = {}
    for shape in env.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
        if shape.get_handle() in excluded or not shape.is_collidable():
            continue
        linear, angular = shape.get_velocity()
        fixtures[shape.get_name()] = dict(pose=shape.get_pose().tolist(),
            dynamic=shape.is_dynamic(), respondable=shape.is_respondable(),
            current_native_linear_velocity_m_per_s=linear.tolist(),
            current_native_angular_velocity_rad_per_s=angular.tolist())
    snapshot = dict(stage=stage, physics_time=before[0], task_time=before[1],
                    robot_joints=env.arm.get_joint_positions(), robot_pose=env.arm.get_tip().get_pose().tolist(),
                    current_native_joint_velocities=env.arm.get_joint_velocities(),
                    target_current_pose_metric_only=env.task.target_block.get_pose().tolist(),
                    fixtures=fixtures,
                    camera_render_modes={name: camera.get_render_mode().name for name, camera in env.cameras.items()})
    if (physics_time(), float(env.task.t)) != before:
        raise RuntimeError('read-current stage snapshot advanced clock')
    snapshot['snapshot_advances_physics_or_task'] = False
    return snapshot


def child_probe(task, launcher):
    config = json.loads(CONFIG.read_text())
    row = next(row for row in json.loads(MANIFEST.read_text())['episodes'] if row['task'] == task)
    random.seed(0);np.random.seed(0)
    report = dict(task=task, launcher=launcher, source=row, seed=0, probe_completed=False,
                  policy_actions=0, counted_episode_ticks=0, neural_policy_used=False,
                  world_model_used=False, GPU_tensor_operations=False, outcome_labels_read=False,
                  source_exposure='first official source already included in completed r04 public TEST manifest',
                  protocol_only=True, stages=[])
    env = None
    restorations = []
    flags = dict(inside_reset_to_demo=False, reset_to_demo_pyrep_step_calls=0)
    try:
        env = AlignedOfficialSimulation(task, row['path'], protocol=config, seed=0, servo_lead=1.)
        original_reset = env.task_environment.reset_to_demo
        def wrapped_reset(*args, **kwargs):
            flags['inside_reset_to_demo'] = True
            try:
                result = original_reset(*args, **kwargs)
            finally:
                flags['inside_reset_to_demo'] = False
            report['stages'].append(current_snapshot(env, 'after_reset_to_demo'))
            return result
        restorations.append((env.task_environment, 'reset_to_demo', original_reset))
        env.task_environment.reset_to_demo = wrapped_reset
        original_init = env.task.init_episode
        def wrapped_init(*args, **kwargs):
            result = original_init(*args, **kwargs)
            if not flags['inside_reset_to_demo']:
                report['stages'].append(current_snapshot(env, 'after_explicit_init_episode'))
            return result
        restorations.append((env.task, 'init_episode', original_init))
        env.task.init_episode = wrapped_init
        original_step = env.pyrep.step
        def wrapped_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            if flags['inside_reset_to_demo']:
                flags['reset_to_demo_pyrep_step_calls'] += 1
            else:
                report['stages'].append(current_snapshot(env, 'after_author_pyrep_step'))
            return result
        restorations.append((env.pyrep, 'step', original_step))
        env.pyrep.step = wrapped_step
        original_alignment = env._start_alignment
        def wrapped_alignment(*args, **kwargs):
            result = original_alignment(*args, **kwargs)
            report['stages'].append(current_snapshot(env, 'after_alignment'))
            return result
        restorations.append((env, '_start_alignment', original_alignment))
        env._start_alignment = wrapped_alignment
        env.reset_official(row)
        before = physics_time(), float(env.task.t)
        observation, robot = env.observe()
        report['public_initial_conditions'] = public_initial_conditions(env, observation, robot)
        report['reset_receipt'] = env.reset_receipt
        report['read_current_physics_delta'] = physics_time()-before[0]
        report['read_current_task_delta'] = float(env.task.t)-before[1]
        report['counted_episode_ticks'] = env.ledger.ticks
        report['reset_to_demo_pyrep_step_calls'] = flags['reset_to_demo_pyrep_step_calls']
        assert report['counted_episode_ticks'] == 0
        assert report['read_current_physics_delta'] == report['read_current_task_delta'] == 0.
        report['probe_completed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)
        write(OUTPUT/(task+'__'+launcher+'.json'), report)
        if env:
            env.close()


def stage_comparisons(left, right, tolerances):
    first = {row['stage']:row for row in left['stages']}
    second = {row['stage']:row for row in right['stages']}
    results = []
    for stage in ('after_reset_to_demo','after_explicit_init_episode','after_author_pyrep_step','after_alignment'):
        if stage not in first or stage not in second:
            results.append(dict(stage=stage, missing_stage=True));continue
        a,b=first[stage],second[stage]
        changes=[]
        for name in a['fixtures'].keys() & b['fixtures'].keys():
            x,y=np.asarray(a['fixtures'][name]['pose']),np.asarray(b['fixtures'][name]['pose'])
            changes.append(dict(name=name, maximum_pose_component_delta=float(np.max(np.abs(x-y))),
                                position_delta_m=float(np.linalg.norm(x[:3]-y[:3])),
                                orientation_delta_rad=float((Rotation.from_quat(x[3:]).inv()*Rotation.from_quat(y[3:])).magnitude()),
                                accepted=bool(np.allclose(x,y,rtol=0,atol=tolerances['fixture_pose_component']))))
        distractors = ['stack_blocks_distractor0','stack_blocks_distractor1']
        overlapping = {}
        for name,source in (('ours_cpu',a),('baseline_gpu',b)):
            if all(key in source['fixtures'] for key in distractors):
                poses=[np.asarray(source['fixtures'][key]['pose']) for key in distractors]
                overlapping[name]=dict(distractor_center_separation_m=float(np.linalg.norm(poses[0][:3]-poses[1][:3])),
                                       both_dynamic=all(source['fixtures'][key]['dynamic'] for key in distractors),
                                       both_respondable=all(source['fixtures'][key]['respondable'] for key in distractors))
        results.append(dict(stage=stage, fixtures=changes, distractor_same_stage_geometry=overlapping,
                            maximum_fixture_position_delta_m=max((x['position_delta_m'] for x in changes),default=0.),
                            maximum_fixture_orientation_delta_rad=max((x['orientation_delta_rad'] for x in changes),default=0.)))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--launcher',choices=LAUNCHERS)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.task:
        if not args.launcher:raise ValueError('child requires launcher label')
        child_probe(args.task,args.launcher);return
    return_codes=[]
    for task in TASKS:
        for launcher in LAUNCHERS:
            wrapper='scripts/run_local_sim.sh' if launcher=='ours_cpu' else 'scripts/run_aligned_gpu.sh'
            command=['bash',wrapper,str(Path(__file__).relative_to(ROOT)),'--task',task,'--launcher',launcher]
            return_codes.append(subprocess.run(command,cwd=ROOT).returncode)
    tolerances_path=ROOT/'artifacts/framework/aligned_r08/acceptance_tolerances.json'
    tolerances=json.loads(tolerances_path.read_text());pairs=[]
    for task in TASKS:
        a,b=[json.loads((OUTPUT/(task+'__'+launcher+'.json')).read_text()) for launcher in LAUNCHERS]
        comparison=compare_conditions(a,b,tolerances) if a['probe_completed'] and b['probe_completed'] else dict(accepted=False,problems=['probe execution incomplete'])
        pairs.append(dict(task=task,source=a['source'],comparison=comparison,
                          stage_comparisons=stage_comparisons(a,b,tolerances),results=[a,b]))
    completed=all(code==0 for code in return_codes)
    accepted=completed and all(pair['comparison']['accepted'] for pair in pairs)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),probe_completed=completed,
                passed=accepted,protocol_acceptance_passed=accepted,protocol_only=True,
                scope='official-reset input inspection only on two already exposed r04 sources; no policy/training/evaluation outcome',
                policy_actions=0,neural_policy_used=False,world_model_used=False,GPU_tensor_operations=False,
                outcome_labels_read=False,counted_episode_ticks=0,
                manifest_sha256=file_sha256(MANIFEST),config_sha256=file_sha256(CONFIG),
                script_sha256=file_sha256(Path(__file__).resolve()),acceptance_tolerances=tolerances,
                tolerances_sha256=file_sha256(tolerances_path),results=pairs,
                source_observation='both author task init_episode functions set every distractor pose using the previous enumerate loop final i; source retained unchanged',
                frozen_core_sha256={name:file_sha256(ROOT/name) for name in (
                    'our_mani/official_simulation.py','our_mani/aligned/simulation.py','our_mani/sensor_alignment/author_rgbd.py')})
    write(OUTPUT/'report.json',report)
    print(json.dumps(dict(probe_completed=completed,protocol_acceptance_passed=accepted,
                         pairs=[dict(task=pair['task'],comparison=pair['comparison'],stage_maxima=[{k:row.get(k) for k in ('stage','maximum_fixture_position_delta_m','maximum_fixture_orientation_delta_rad')} for row in pair['stage_comparisons']]) for pair in pairs])),flush=True)
    if not accepted:raise RuntimeError('official reset paired protocol acceptance failed; do not loosen tolerances')


if __name__=='__main__':main()
