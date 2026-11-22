"""Native TRAIN checks for shared author RGB buffers; no neural policy or GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import torch

from collect_local_rollouts import initialization, TASKS
from our_mani.aligned import physmani as inherited_physmani
from our_mani.aligned.simulation import physics_time
from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation, public_initial_conditions
from our_mani.sensor_alignment.physmani_executor_v5 import tensors, author_tensor_scope
from our_mani.benchmark_protocol import file_sha256

OUTPUT = ROOT/'artifacts/framework/aligned_r07/author_rgbd_probe'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def task_probe(task, config_path):
    from physmani_bench.env import transform
    config = json.loads(config_path.read_text())
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    row = next(row for row in rows if row['task'] == task)
    torch.set_num_threads(2)
    report = dict(task=task, source=row['raw_episode_id'], passed=False,
                  formal_test_used=False, GPU_used=False, split='train', frames=[])
    env = None
    try:
        env = AlignedLocalSimulation(task, row['path'], protocol=config, seed=config['seed'], servo_lead=1.)
        x, velocity, acceleration, goal = initialization(row['path'])
        assert env.reset(target_position=x, target_velocity=velocity,
                         target_acceleration=acceleration, goal_position=goal)
        env.finish_home_reset()
        modes = {name: camera.get_render_mode().name for name, camera in env.cameras.items()}
        assert set(modes.values()) == {'OPENGL'}
        report['actual_render_modes'] = modes
        calibration_before = {name: (camera.get_intrinsic_matrix(), camera.get_matrix())
                              for name, camera in env.cameras.items()}
        report['camera_calibrations'] = {name: dict(intrinsics=values[0].tolist(), extrinsics=values[1].tolist())
                                         for name, values in calibration_before.items()}
        original_tensors = inherited_physmani.tensors
        with author_tensor_scope():
            assert inherited_physmani.tensors is tensors
        assert inherited_physmani.tensors is original_tensors
        try:
            with author_tensor_scope():
                raise LookupError('scope restoration probe')
        except LookupError:
            pass
        assert inherited_physmani.tensors is original_tensors
        report['tensor_binding_restored_on_normal_and_exception_exit'] = True
        for frame_index in range(3):
            before = dict(physics=physics_time(), task=float(env.task.t),
                          joints=np.asarray(env.arm.get_joint_positions()),
                          targets=np.asarray(env.arm.get_joint_target_positions()),
                          pose=np.asarray(env.arm.get_tip().get_pose()))
            observation, robot = env.observe()
            rgb, _, depth, _ = tensors(observation, robot, device='cpu')
            native_errors = {}
            for name, camera in env.cameras.items():
                raw = observation['observation/raw_rgb_255'][name]
                source = camera.capture_rgb()*255.
                expected = np.clip(source.astype(np.uint8), 0, 255) if name == 'front' else source
                native_errors[name] = float(np.max(np.abs(raw.astype(float)-expected.astype(float))))
                assert np.array_equal(raw, expected)
                assert observation['observation/images'][name].dtype == np.uint8
                np.testing.assert_array_equal(observation['observation/images'][name],
                                              np.clip(np.rint(source), 0, 255).astype(np.uint8))
                assert np.allclose(camera.get_intrinsic_matrix(), calibration_before[name][0], rtol=0, atol=1e-7)
                assert np.allclose(camera.get_matrix(), calibration_before[name][1], rtol=0, atol=1e-7)
                assert camera.get_render_mode().name == 'OPENGL'
            author_rgb = transform(dict(rgb=[observation['observation/raw_rgb_255'][name]
                                             for name in inherited_physmani.CAMERAS], pc=[], mask=[]), augmentation=False)
            author_rgb = torch.stack(author_rgb.chunk(4, dim=0))[None]
            transform_error = float(torch.max(torch.abs(rgb-author_rgb)))
            assert torch.equal(rgb, author_rgb)
            assert tuple(depth.shape) == (1, 4, 1, 256, 256)
            assert physics_time() == before['physics'] and float(env.task.t) == before['task']
            np.testing.assert_array_equal(env.arm.get_joint_positions(), before['joints'])
            np.testing.assert_array_equal(env.arm.get_joint_target_positions(), before['targets'])
            np.testing.assert_array_equal(env.arm.get_tip().get_pose(), before['pose'])
            conditions = public_initial_conditions(env, observation, robot)
            assert conditions['sensor_protocol']['camera_render_modes'] == modes
            assert all(values['render_mode'] == 'OPENGL' for values in conditions['cameras'].values())
            report['frames'].append(dict(index=frame_index, timestamp=env.time,
                                         native_buffer_max_delta_by_camera=native_errors,
                                         author_transform_max_delta=transform_error,
                                         raw_dtypes={name: str(value.dtype) for name, value in observation['observation/raw_rgb_255'].items()},
                                         physics_delta_seconds=physics_time()-before['physics'],
                                         task_delta_seconds=float(env.task.t)-before['task'],
                                         joints_pose_targets_calibration_unchanged=True))
            if frame_index == 0:
                report['public_initial_conditions'] = conditions
            if frame_index < 2:
                env.hold_tick()
        report['alignment_receipt'] = env.alignment_receipt()
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        write(OUTPUT/(task+'.json'), report)
        if env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--config', default='config/framework/aligned_protocol_r07.json')
    args = parser.parse_args()
    config_path = ROOT/args.config
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.task:
        task_probe(args.task, config_path)
        return
    statuses = []
    for task in TASKS:
        statuses.append(subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                       '--task', task, '--config', args.config]).returncode)
    reports = [json.loads((OUTPUT/(task+'.json')).read_text()) for task in TASKS]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  passed=all(code == 0 for code in statuses) and all(report['passed'] for report in reports),
                  formal_test_used=False, split='train', GPU_used=False, neural_policy_used=False,
                  world_model_used=False, task_processes_isolated=True,
                  scope='public author-domain sensor/buffer protocol checks, not success-rate evaluation',
                  config=str(config_path.relative_to(ROOT)), config_sha256=file_sha256(config_path),
                  sensor_sha256=file_sha256(ROOT/'our_mani/sensor_alignment/author_rgbd.py'),
                  v5_sha256=file_sha256(ROOT/'our_mani/sensor_alignment/physmani_executor_v5.py'),
                  results=reports)
    write(OUTPUT/'report.json', result)
    print(json.dumps(dict(passed=result['passed'], tasks=[report['task'] for report in reports],
                         frame_checks=sum(len(report['frames']) for report in reports))), flush=True)
    if not result['passed']:
        raise RuntimeError('native author-domain sensor probe failed; inspect reports')


if __name__ == '__main__':
    main()
