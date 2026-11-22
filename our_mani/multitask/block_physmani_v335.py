"""Fresh TRAIN block coverage for the unchanged released R09 V6 baseline.

This fills the two block routes missing from the core14 baseline wrapper.
Neural/WM/RML algorithms and the original legacy block reset are unchanged.
The existing191 observer records every actual tick without changing commands.
"""
from datetime import datetime, timezone
from pathlib import Path
import gc
import random
import time
import numpy as np
import torch

from ..aligned.physmani import build_policy
from ..sensor_alignment.author_rgbd import AlignedLocalSimulation, public_initial_conditions, SensorProtocolInvariantError
from ..sensor_alignment.physmani_executor_v6 import (
    PhysManiAdapter, ProtocolInvariantError, InfrastructureExecutionError, raise_protected_error)
from .native_window_observer_v191 import CountedWindowObserver
from .joint16_tracks_v31 import BLOCKS
from .evaluation import file_sha, digest


def binding(root, config):
    root = Path(root)
    if (config['baseline_executor'] != 'released_waypoints_counted_actuator_author_rgbd_v6'
            or config['baseline_reference_filter'] != 'author_direct_setpoint'
            or config['baseline_decision_ticks'] != 10 or config['dt'] != .05
            or config['simulation_seconds'] != 10. or config['global_joint_acceleration'] != 6.):
        raise ValueError('original released R09 V6 block baseline/clock/native limits required')
    names = ('our_mani/multitask/block_physmani_v335.py',
             'our_mani/multitask/native_window_observer_v191.py',
             'scripts/evaluate_physmani_aligned_r09_v6.py', 'scripts/collect_local_rollouts.py',
             'our_mani/aligned/physmani.py', 'our_mani/aligned/simulation.py',
             'our_mani/sensor_alignment/author_rgbd.py',
             'our_mani/sensor_alignment/physmani_executor_v5.py',
             'our_mani/sensor_alignment/physmani_executor_v6.py',
             'our_mani/ablations/physmani_executor_v4.py')
    return dict(schema='r10_two_block_original_PhysMani_V6_readonly_windows_TRAIN_v335',
        sources_sha256={n: file_sha(root / n) for n in names},
        executor='original released R09 V6 block baseline; core14 R10 uses terminal-aware V7',
        reset='same original HDF5 kinematics/home development reset as both block geometry methods',
        checkpoint=dict(path=config['baseline_checkpoint'], sha256=file_sha(config['baseline_checkpoint'])),
        original_neural_policy_world_model_RML_commands_unchanged=True,
        read_only_actual_joint_windows=True, extra_camera_or_physics_calls=False,
        Value_used=False, NGFF_used=False, FASTER_used=False, formal_VAL_TEST=False)


def run_TRAIN_baseline(root, row, config):
    root = Path(root)
    if row['task'] not in BLOCKS or config.get('block_physmani_v335') != binding(root, config):
        raise ValueError('freeze actual two-block baseline binding')
    from scripts.collect_local_rollouts import initialization
    started = time.perf_counter()
    env = model = adapter = None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
        model = build_policy(root, config, row['task'])  # Original strict author weights.
        # The original R09 runner reseeds after model construction for each case.
        torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
        env = AlignedLocalSimulation(row['task'], row['path'], protocol=config, seed=config['seed'], servo_lead=1.)
        x, v, a, goal = initialization(row['path'])
        if not env.reset(target_position=x, target_velocity=v, target_acceleration=a, goal_position=goal):
            raise RuntimeError('original block baseline development reset IK failed')
        env.finish_home_reset()
        if not np.allclose(env.robot_model.acceleration_limits, 6.):
            raise ProtocolInvariantError('original native six-unit model changed')
        observer = CountedWindowObserver(env)
        observation, robot = env.observe()
        initial = public_initial_conditions(env, observation, robot)
        adapter = PhysManiAdapter(root, config, row['task'], model)
        adapter.reset(env, row['task'], int(row.get('variation', 0)), observation, robot)
        steps, failure = [], None
        while env.ledger.remaining >= env.ledger.dt - 1e-8 and not env._native_success_latched:
            observation, robot = env.observe()
            try:
                action = adapter.predict(observation, robot)
                execution = adapter.execute(env, action)
            except (ProtocolInvariantError, InfrastructureExecutionError, SensorProtocolInvariantError):
                raise
            except (ValueError, RuntimeError, AssertionError) as error:
                raise_protected_error(error)
                if any(t in str(error).lower() for t in (
                        'out of memory', 'illegal memory', 'device-side assert', 'cuda error', 'compilation')):
                    raise
                failure = repr(error)
                break  # Preserve the original baseline policy-failure handling.
            steps.append(dict(time=float(robot.timestamp), action=action.tolist(), execution=execution))
        return dict(created_utc=datetime.now(timezone.utc).isoformat(), task=row['task'],
            source_raw_episode_id=row['raw_episode_id'], raw_episode_id=row['raw_episode_id'],
            split='train', method='physmani', official_task_success=bool(env._native_success_latched),
            policy_failure=failure, steps=steps, public_initial_conditions=initial,
            alignment_receipt=env.alignment_receipt(),
            all_stage_counted_joint_motion=observer.receipt(),
            counted_clock=dict(ticks=int(env.ledger.ticks), dt=.05, seconds=float(env.time)),
            simulation_seconds=float(env.time), wall_seconds=time.perf_counter() - started,
            compute_seconds=adapter.compute_seconds, baseline_executor=config['baseline_executor'],
            block_physmani_v335=config['block_physmani_v335'],
            controller_sha256=digest(config['block_physmani_v335']),
            budget_fully_consumed_or_native_success=bool(env._native_success_latched or env.ledger.ticks == 200),
            original_policy_failure_not_retried=True, demo_only=True, development_demo_only=True,
            formal_validation_metric=False, formal_TEST_metric=False, formal_test_used=False,
            TEST_used_for_fitting=False, Value_used=False, NGFF_used=False, FASTER_used=False)
    finally:
        if env is not None:
            env.close()
        del adapter, model
        gc.collect()
        torch.cuda.empty_cache()
