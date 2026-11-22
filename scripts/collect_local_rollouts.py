"""Collect observable-state local grasp rollouts in an owned simulator.

These are perturbed local handover experiments initialized from TRAIN or VAL
kinematics, not restored official benchmark episodes. Simulator object poses
are used only to annotate perception error, stable lift, and reset metadata.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np

from our_mani.benchmark_perception import RedBoxTracker
from our_mani.goal_perception import RedGoalMarker
from our_mani.local_capture import LocalCaptureController, LocalConfig, local_features
from our_mani.pyrep_env import LocalSimulation

TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
STOP = ROOT/'runtime/framework/local_state/STOP'


def initialization(path):
    with h5py.File(path, 'r') as f:
        points = f['low_dim_obs/expert_trajectory_target_position'][:5]
        stages = np.char.lower(f['low_dim_obs/expert_trajectory_stage'][:].astype('U').reshape(-1))
        lifts = np.flatnonzero(stages == 'lift')
        goal = f['low_dim_obs/expert_trajectory'][lifts[0]].reshape(-1)[:3] if len(lifts) else None
    velocities = np.diff(points, axis=0)/.05
    estimates = np.diff(velocities, axis=0)/.05
    a = np.zeros(3)
    for axis in range(3):
        # Boundary reflections are velocity discontinuities, not a large
        # continuous driving acceleration. Estimate a from non-flip segments.
        usable = estimates[:, axis][np.abs(estimates[:, axis]) <= 1.]
        a[axis] = np.median(usable) if len(usable) else 0.
    x, v = points[0].copy(), velocities[0]-.5*a*.05
    x[2], v[2], a[2] = .775, 0., 0.
    return x, v, a, goal


def collect(environment, row, offset, budget, delay, seed, goal_offset, *, full_task, learned_policy=None, video=False, controller_config=None, warmup_match=0., latent_checkpoint=None, context_history_frames=5):
    x, v, a, goal = initialization(row['path'])
    started = time.perf_counter()
    record = {'task': row['task'], 'source_episode': row['raw_episode_id'], 'seed': seed,
              'offset_m': list(offset), 'capture_budget_seconds': budget, 'delay_ticks': delay,
              'warmup_velocity_match_fraction': warmup_match,
              'reset_target_position': x.tolist(), 'reset_velocity': v.tolist(), 'reset_acceleration': a.tolist(),
              'captured': False, 'stable_lift': False, 'official_task_success': False,
              'official_task_success_immediate': False,
              'full_task_executed': bool(full_task),
              'kind': 'perturbed_local_handover_experiment_not_official_episode_replay'}
    if not environment.reset(target_position=x, target_velocity=v, target_acceleration=a,
                             relative_ee=offset, goal_position=goal):
        return {**record, 'failure': 'reset_ik_failed'}, {}, []
    tracker, marker = RedBoxTracker(), RedGoalMarker(goal_offset)
    features, actions, times, tracking_errors, stages = [], [], [], [], []
    frames, controller, initial = [], None, None
    lift_since, success_since = None, None
    capture_start = 0.
    dynamics = None
    if latent_checkpoint:
        from our_mani.context_dynamics import ContextLatentDynamics
        dynamics = ContextLatentDynamics(latent_checkpoint)
    warmup_ticks = max(delay+4,context_history_frames-1) if dynamics else delay+4
    total_ticks = int((budget+4.5 if full_task else budget)/.05)+(warmup_ticks+4 if dynamics else delay+4)
    for step in range(total_ticks):
        if STOP.exists():
            record['failure'] = 'deadline_stop'
            break
        observation, robot = environment.observe()
        if video:
            from PIL import Image, ImageDraw
            names = ('front', '8', '16', '36')
            mosaic = np.concatenate([np.concatenate([observation['observation/images'][n] for n in names[:2]], axis=1),
                                     np.concatenate([observation['observation/images'][n] for n in names[2:]], axis=1)], axis=0)
            image = Image.fromarray(mosaic)
            ImageDraw.Draw(image).text((8, 8), f"{row['task']}  t={environment.time:.2f}s\n{controller.stage if controller else 'initialize'}", fill='white', stroke_width=1, stroke_fill='black')
            frames.append(np.asarray(image))
        if controller is None:
            try:
                goal_pose, marker_meta = marker.observe(observation)
                if dynamics:
                    from our_mani.context_control import ContextCaptureController
                    controller = ContextCaptureController(goal_pose,controller_config,dynamics)
                else:
                    controller = LocalCaptureController(goal_pose, config=controller_config, learned_policy=learned_policy)
                record['goal_visual_metadata'] = marker_meta
                record['goal_visual_error_m'] = float(np.linalg.norm(goal_pose.position-goal)) if goal is not None else None
            except ValueError as error:
                record['failure'] = str(error)
                break
        try:
            obj, metadata = tracker.observe(observation)
        except ValueError as error:
            # Carrying is controlled using proprioception and the previously
            # observed goal; a fully occluded grasped cube is expected.
            if robot.contact and 'last_obj' in locals():
                obj = last_obj
            else:
                environment.advance(robot.pose, opening=1.)
                if step > 5:
                    record['failure'] = str(error)
                    break
                continue
        last_obj = obj
        if dynamics and obj.timestamp == robot.timestamp:
            dynamics.observe([obj])
        perception_error = float(np.linalg.norm(obj.pose.position-environment.metric_target_position()))
        if step < warmup_ticks:
            from our_mani.contracts import Pose
            warmup_velocity = obj.linear_velocity*warmup_match
            environment.advance(Pose(robot.pose.position+warmup_velocity*.05, robot.pose.rotation), opening=1.)
            continue
        if initial is None:
            if dynamics:
                try:
                    record['dynamics_context'] = dynamics.context_snapshot(obj.object_id)
                except ValueError:
                    environment.advance(robot.pose,opening=1.)
                    continue
                record['decision_timestamp'] = robot.timestamp
            initial = np.r_[local_features(robot, obj), budget]
            capture_start = environment.time
        elapsed = environment.time-capture_start
        if dynamics and full_task and elapsed > budget+4.5+1e-8:
            record['failure'] = 'visual_goal_transport_timeout'
            break
        actual_height = environment.metric_target_position()[2]-x[2]
        if robot.contact:
            record['captured'] = True
            record.setdefault('first_contact_seconds', elapsed)
        if robot.contact and actual_height >= .08:
            lift_since = environment.time if lift_since is None else lift_since
            if environment.time-lift_since >= .30-1e-6 and elapsed <= budget:
                record['stable_lift'] = True
                record.setdefault('stable_lift_seconds', elapsed)
        else:
            lift_since = None
        if robot.task_success:
            record['official_task_success_immediate'] = True
            success_since = environment.time if success_since is None else success_since
            if environment.time-success_since >= .30-1e-6:
                record['official_task_success'] = True
                break
        else:
            success_since = None
        if not full_task and (record['stable_lift'] or elapsed >= budget):
            break
        if full_task and elapsed >= budget and not robot.contact:
            record['failure'] = 'capture_budget_exceeded'
            break
        try:
            desired, opening, velocity, angular, details = controller.command(robot, obj)
        except RuntimeError as error:
            record['failure'] = str(error)
            record['failure_state'] = {'robot_velocity': robot.linear_velocity.tolist(),
                                       'object_velocity': obj.linear_velocity.tolist(),
                                       'relative_position': (robot.pose.position-obj.pose.position).tolist()}
            break
        if details['stage'] in ('track', 'close'):
            features.append(local_features(robot, obj))
            actions.append(np.r_[velocity, angular, opening].astype(np.float32))
            times.append(environment.time)
        tracking_errors.append(perception_error)
        stages.append(details['stage'])
        applied = environment.advance(desired, opening=opening)
        if not applied:
            record['rejected_actions'] = record.get('rejected_actions', 0)+1
    record.update({'wall_seconds': time.perf_counter()-started, 'simulation_seconds': environment.time,
                   'steps': step+1, 'perception_mean_error_m': float(np.mean(tracking_errors)) if tracking_errors else None,
                   'initial_features': initial.tolist() if initial is not None else None,
                   'stage_counts': {s: stages.count(s) for s in set(stages)}})
    arrays = {'features': np.asarray(features, dtype=np.float32).reshape(-1, 17),
              'actions': np.asarray(actions, dtype=np.float32).reshape(-1, 7), 'timestamps': np.asarray(times)}
    return record, arrays, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--episodes-per-task', type=int, default=24)
    parser.add_argument('--replicas', type=int, default=4)
    parser.add_argument('--name', default='mpc_train_v1')
    parser.add_argument('--full-task', action='store_true')
    parser.add_argument('--central-only', action='store_true')
    parser.add_argument('--learned-checkpoint')
    parser.add_argument('--latent-checkpoint', help='Pipeline v1 context/SimNorm dynamics and forecast MPC')
    parser.add_argument('--videos', type=int, default=0)
    parser.add_argument('--video-task', choices=TASKS)
    parser.add_argument('--servo-lead', type=float, default=1.4)
    parser.add_argument('--max-speed', type=float, default=1.)
    parser.add_argument('--max-acceleration', type=float, default=4.)
    args = parser.parse_args()
    if args.latent_checkpoint and args.learned_checkpoint:
        raise ValueError('context MPC and learned-servo controller modes are distinct')
    out = ROOT/'artifacts/framework/local_policy'/args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT/f'artifacts/framework/protocol/{args.split}_manifest.json').read_text())['episodes']
    calibration = json.loads((ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json').read_text())
    goal_offset = calibration['offset_train_calibrated_m']
    controller_config = LocalConfig(max_speed=args.max_speed, max_acceleration=args.max_acceleration)
    controller_hash = hashlib.sha256(b''.join((ROOT/'our_mani'/name).read_bytes() for name in
            ('local_capture.py', 'control.py', 'benchmark_perception.py', 'goal_perception.py', 'pyrep_env.py'))
            +json.dumps({'controller_config': asdict(controller_config), 'servo_lead': args.servo_lead}, sort_keys=True).encode()).hexdigest()
    learned_policy = None
    if args.latent_checkpoint:
        from our_mani.context_control import controller_sha256
        controller_hash = controller_sha256(ROOT/args.latent_checkpoint,controller_config,args.servo_lead)
    if args.learned_checkpoint:
        from our_mani.local_learning import LocalServo
        learned_policy = LocalServo.load(args.learned_checkpoint)
        controller_hash += ':'+hashlib.sha256(Path(args.learned_checkpoint).read_bytes()).hexdigest()
    results, video_count = [], 0
    rng = np.random.default_rng(421 if args.split == 'train' else 9401)
    for task in TASKS:
        chosen = [r for r in rows if r['task'] == task][:args.episodes_per_task]
        environment = LocalSimulation(task, chosen[0]['path'], seed=421 if args.split == 'train' else 9401, servo_lead=args.servo_lead)
        try:
            for row in chosen:
                for replica in range(args.replicas):
                    if STOP.exists():
                        return
                    key = f"{task}__{row['raw_episode_id']}__{replica}"
                    if replica == 0 or args.central_only:
                        offset, budget, delay = np.array([0., 0., .07]), 3., 0
                    else:
                        offset = np.r_[rng.uniform(-.20, .20, 2), rng.uniform(.025, .23)]
                        budget, delay = float(rng.choice([1., 1.5, 2., 3.])), int(rng.integers(0, 5))
                    warmup_match = [1., 0., .5, -0.5, 1.25, 1.][replica % 6]
                    context_frames = (5,10,20)[replica%3]
                    if (out/(key+'.json')).exists():
                        saved = json.loads((out/(key+'.json')).read_text(encoding='utf-8'))
                        if saved['controller_sha256'] != controller_hash:
                            raise RuntimeError('refusing to mix controller versions on collection resume')
                        previous_full_task = saved.get('full_task_executed')
                        if previous_full_task is None and (out/'report.json').exists():
                            previous_full_task = json.loads((out/'report.json').read_text()).get('full_task_executed')
                        if previous_full_task is not args.full_task:
                            raise RuntimeError('refusing to mix censored and full-task outcomes on resume')
                        if args.latent_checkpoint and saved.get('requested_context_history_frames') != context_frames:
                            raise RuntimeError('context history sampling differs on resume')
                        if saved['offset_m'] != offset.tolist() or saved['capture_budget_seconds'] != budget or saved['delay_ticks'] != delay:
                            raise RuntimeError('perturbation sampling differs on resume')
                        results.append(saved)
                        continue
                    record, arrays, frames = collect(environment, row, offset, budget, delay, 421+len(results), goal_offset,
                            full_task=args.full_task, learned_policy=learned_policy,
                            video=video_count < args.videos and (args.video_task is None or args.video_task == task),
                            controller_config=controller_config, warmup_match=warmup_match,
                            latent_checkpoint=ROOT/args.latent_checkpoint if args.latent_checkpoint else None,
                            context_history_frames=context_frames)
                    record.update({'key': key, 'split': args.split, 'controller': LocalCaptureController.name,
                                   'controller_sha256': controller_hash, 'controller_config': asdict(controller_config),
                                   'servo_lead': args.servo_lead,
                                   'learned_checkpoint': args.learned_checkpoint, 'formal_test_used': False})
                    if args.latent_checkpoint:
                        record.update({'controller':'context_simnorm_capture_and_visual_goal_v1',
                            'latent_checkpoint':args.latent_checkpoint,'requested_context_history_frames':context_frames})
                    np.savez_compressed(out/(key+'.npz'), **arrays)
                    (out/(key+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                    results.append(record)
                    if frames and record['captured']:
                        from our_mani.video import write_video
                        video_path = out/f'demo_{video_count+1}_{task}.mp4'
                        write_video(video_path, frames)
                        video_count += 1
                    valid = [r for r in results if r.get('initial_features') is not None]
                    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'name': args.name, 'split': args.split,
                        'controller_sha256': controller_hash, 'completed': len(results), 'valid_initializations': len(valid),
                        'captured': sum(r['captured'] for r in valid), 'stable_lift': sum(r['stable_lift'] for r in valid),
                        'official_task_success': sum(r['official_task_success'] for r in valid), 'full_task_executed': args.full_task,
                        'formal_test_used': False, 'kind': 'perturbed_local_handover_experiment_not_official_benchmark',
                        'per_task': {t: {'n': sum(r['task'] == t for r in valid),
                                        'captured': sum(r['task'] == t and r['captured'] for r in valid),
                                        'stable_lift': sum(r['task'] == t and r['stable_lift'] for r in valid),
                                        'official_success': sum(r['task'] == t and r['official_task_success'] for r in valid)} for t in TASKS}}
                    (out/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                    status_name = 'collection_status.json' if args.split == 'train' else 'validation_report.json'
                    if 'smoke' not in args.name:
                        (out.parent/status_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                    print(json.dumps({k:v for k,v in record.items() if k not in ('initial_features', 'controller_config')}, ensure_ascii=False), flush=True)
        finally:
            environment.close()


if __name__ == '__main__':
    main()
