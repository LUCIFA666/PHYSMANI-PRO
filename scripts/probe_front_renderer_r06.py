"""TRAIN same-state front-renderer diagnostic; no TEST or policy tuning."""
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
import torch

from collect_local_rollouts import initialization, TASKS
from our_mani.aligned.simulation import AlignedLocalSimulation, physics_time
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.context_dynamics import ContextLatentDynamics
from our_mani.goal_perception import RedGoalMarker
from our_mani.benchmark_protocol import file_sha256

OUTPUT = ROOT/'artifacts/framework/aligned_r06/front_renderer_probe'
CONFIG = ROOT/'config/framework/aligned_protocol_r06_baseline_v4.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def distribution(values):
    if not len(values):
        return dict(n=0, mean=None, maximum=None)
    return dict(n=len(values), mean=float(np.mean(values)), maximum=float(np.max(values)))


def summarize(frames):
    keys = ('front_rgb_mean_absolute_255', 'front_rgb_max_absolute_255',
            'front_depth_mean_absolute_m', 'front_depth_max_absolute_m',
            'target_position_delta_m', 'target_velocity_delta_m_per_s',
            'goal_position_delta_m', 'context_l2_delta', 'context_max_absolute_delta',
            'forecast_05s_position_delta_m')
    metrics = {key: distribution([frame[key] for frame in frames if frame.get(key) is not None])
               for key in keys}
    failures = {mode: {name: sum(name in frame['modes'][mode].get('failures', {}) for frame in frames)
                       for name in ('target', 'goal', 'context', 'prediction')}
                for mode in ('OPENGL3', 'OPENGL')}
    warmup = {mode: sum(frame['modes'][mode].get('context_warmup', False) for frame in frames)
              for mode in ('OPENGL3', 'OPENGL')}
    return dict(metrics=metrics, failures=failures, expected_context_warmup_frames=warmup,
                renders_never_advance_physics=all(frame['render_physics_delta_seconds'] == 0 for frame in frames),
                renders_never_advance_task=all(frame['render_task_delta_seconds'] == 0 for frame in frames))


def observe_model(observation, tracker, marker, dynamics, frame_index):
    result = dict(failures={})
    try:
        goal, _ = marker.observe(observation)
        result['goal_position_m'] = goal.position.tolist()
    except ValueError as exc:
        result['failures']['goal'] = str(exc)
    try:
        obj, perception = tracker.observe(observation)
        result['target_position_m'] = obj.pose.position.tolist()
        result['target_velocity_m_per_s'] = obj.linear_velocity.tolist()
        result['target_perception'] = perception
        dynamics.observe([obj])
    except ValueError as exc:
        result['failures']['target'] = str(exc)
        return result
    if frame_index < 4:
        result['context_warmup'] = True
        return result
    try:
        result['context'] = dynamics.context_snapshot()
    except ValueError as exc:
        result['failures']['context'] = str(exc)
    try:
        forecast = dynamics.predict([obj], .5, .05)
        result['forecast_05s_position_m'] = forecast.positions[-1, 0].tolist()
    except ValueError as exc:
        result['failures']['prediction'] = str(exc)
    return result


def compare_models(results, field):
    left, right = results['OPENGL3'].get(field), results['OPENGL'].get(field)
    if left is None or right is None:
        return None
    return float(np.linalg.norm(np.asarray(left)-np.asarray(right)))


def task_probe(task, frame_count):
    from pyrep.const import RenderMode
    from PIL import Image
    config = json.loads(CONFIG.read_text())
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    row = next(row for row in rows if row['task'] == task)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.set_num_threads(2)
    goal_offset = json.loads((ROOT/config['goal_calibration']).read_text())['offset_train_calibrated_m']
    checkpoint = ROOT/config['dynamics_checkpoint']
    models = {name: (RedBoxTracker(), RedGoalMarker(goal_offset), ContextLatentDynamics(checkpoint))
              for name in ('OPENGL3', 'OPENGL')}
    report = dict(task=task, source=row['raw_episode_id'], split='train', formal_test_used=False,
                  passed=False, frame_count=frame_count, frames=[], GPU_used=False,
                  experiment='same physical state; only front renderer changes; other views retained',
                  dynamics_checkpoint_sha256=file_sha256(checkpoint))
    output = OUTPUT/task
    output.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = AlignedLocalSimulation(task, row['path'], protocol=config, seed=config['seed'], servo_lead=1.)
        x, velocity, acceleration, goal = initialization(row['path'])
        assert env.reset(target_position=x, target_velocity=velocity,
                         target_acceleration=acceleration, goal_position=goal)
        env.finish_home_reset()
        camera = env.cameras['front']
        original_mode = camera.get_render_mode()
        intrinsics, extrinsics = camera.get_intrinsic_matrix(), camera.get_matrix()
        report['original_front_render_mode'] = original_mode.name
        report['other_camera_modes'] = {name: sensor.get_render_mode().name for name, sensor in env.cameras.items() if name != 'front'}
        report['front_calibration'] = dict(intrinsics=intrinsics.tolist(), extrinsics=extrinsics.tolist(),
                                           near=camera.get_near_clipping_plane(), far=camera.get_far_clipping_plane())
        print(json.dumps(dict(task=task, original_front_render_mode=original_mode.name,
                              other_camera_modes=report['other_camera_modes'])), flush=True)
        for frame_index in range(frame_count):
            before_physics, before_task = physics_time(), float(env.task.t)
            observations = {}
            results = {}
            try:
                for name, mode in (('OPENGL3', RenderMode.OPENGL3), ('OPENGL', RenderMode.OPENGL)):
                    camera.set_render_mode(mode)
                    observation, _ = env.observe()
                    assert np.allclose(camera.get_intrinsic_matrix(), intrinsics, rtol=0, atol=1e-7)
                    assert np.allclose(camera.get_matrix(), extrinsics, rtol=0, atol=1e-7)
                    observations[name] = observation
                    results[name] = observe_model(observation, *models[name], frame_index)
                    if frame_index == 0:
                        Image.fromarray(observation['observation/images']['front']).save(output/(name+'_front_frame0.png'))
                        np.save(output/(name+'_front_depth_frame0.npy'), observation['observation/depths']['front'])
            finally:
                camera.set_render_mode(original_mode)
            physics_delta, task_delta = physics_time()-before_physics, float(env.task.t)-before_task
            assert physics_delta == 0 and task_delta == 0
            rgb_delta = np.abs(observations['OPENGL3']['observation/images']['front'].astype(float)
                               - observations['OPENGL']['observation/images']['front'].astype(float))
            depth_delta = np.abs(observations['OPENGL3']['observation/depths']['front'].astype(float)
                                 - observations['OPENGL']['observation/depths']['front'].astype(float))
            frame = dict(index=frame_index, timestamp=env.time, modes=results,
                         front_rgb_mean_absolute_255=float(np.mean(rgb_delta)),
                         front_rgb_max_absolute_255=float(np.max(rgb_delta)),
                         front_depth_mean_absolute_m=float(np.mean(depth_delta)),
                         front_depth_max_absolute_m=float(np.max(depth_delta)),
                         render_physics_delta_seconds=physics_delta, render_task_delta_seconds=task_delta,
                         target_position_delta_m=compare_models(results, 'target_position_m'),
                         target_velocity_delta_m_per_s=compare_models(results, 'target_velocity_m_per_s'),
                         goal_position_delta_m=compare_models(results, 'goal_position_m'),
                         forecast_05s_position_delta_m=compare_models(results, 'forecast_05s_position_m'),
                         context_l2_delta=None, context_max_absolute_delta=None)
            if all('context' in results[name] for name in models):
                context_delta = np.asarray(results['OPENGL3']['context']['values'])-np.asarray(results['OPENGL']['context']['values'])
                frame['context_l2_delta'] = float(np.linalg.norm(context_delta))
                frame['context_max_absolute_delta'] = float(np.max(np.abs(context_delta)))
            report['frames'].append(frame)
            report['summary'] = summarize(report['frames'])
            write(output/'report.json', report)
            if frame_index in (0, 1, 2, 4, frame_count-1):
                print(json.dumps({key: value for key, value in dict(task=task, **frame).items() if key != 'modes'}), flush=True)
            if frame_index < frame_count-1:
                env.hold_tick()
        report['alignment_receipt'] = env.alignment_receipt()
        report['front_mode_restored'] = camera.get_render_mode() == original_mode
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        write(output/'report.json', report)
        if env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--frames', type=int, default=16)
    args = parser.parse_args()
    if not 10 <= args.frames <= 20:
        raise ValueError('use 10–20 fixed TRAIN diagnostic frames')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.task:
        task_probe(args.task, args.frames)
        return
    statuses = []
    for task in TASKS:
        statuses.append(subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                       '--task', task, '--frames', str(args.frames)]).returncode)
    reports = [json.loads((OUTPUT/task/'report.json').read_text()) for task in TASKS]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  passed=all(code == 0 for code in statuses) and all(result['passed'] for result in reports),
                  split='train', formal_test_used=False, GPU_used=False, task_processes_isolated=True,
                  scope='shared sensor domain diagnostic, not policy success-rate evaluation; no TEST tuning',
                  config_sha256=file_sha256(CONFIG),
                  probe_sha256=file_sha256(Path(__file__).resolve()),
                  combined_summary=summarize([frame for result in reports for frame in result['frames']]),
                  results=reports)
    write(OUTPUT/'report.json', report)
    print(json.dumps(dict(passed=report['passed'], combined_summary=report['combined_summary'])), flush=True)
    if not report['passed']:
        raise RuntimeError('front renderer diagnostic failed; inspect reports')


if __name__ == '__main__':
    main()
