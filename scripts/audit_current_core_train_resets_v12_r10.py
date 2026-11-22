"""Prepare/run 56 isolated TRAIN reset-only repeats after formal400+10 demos.

Four fresh processes per task share one TRAIN source, variation and seed.
Repeats 0/1 are the fit pair and 2/3 the independent validation pair. No
teacher, policy, body-prediction model, GPU tensor or counted action is used.
Thresholds are preregistered; this entry never adapts or freezes them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.native_tasks import verify_native_sources
from our_mani.multitask.reset_parity import (pending_tolerance_profile, compare_initial_conditions,
                                            capture_recorded_initial_conditions, digest)
BASE = ROOT / 'artifacts/framework/multitask_r10'
PLAN_VERSION = 'r10_TRAIN_reset_repeat_preregistered_v1'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEMO_METHODS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value', 'physmani')
DEMO_TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
DEMO_FIRST_RAW = dict(zip(DEMO_TASKS, ('episode2', 'episode0')))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition: raise ValueError(message)


def write(path, data):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), 'reset-repeat outputs must stay inside new R10 artifacts')
    require(not path.exists(), 'preserve prior reset-repeat evidence; choose a fresh version: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_stat(path):
    item = Path(path).stat()
    return dict(size_bytes=item.st_size, mtime_ns=item.st_mtime_ns, inode=item.st_ino, device=item.st_dev)


def preregistered_profile(legacy):
    profile = pending_tolerance_profile(legacy)
    profile['status'] = 'preregistered_TRAIN_repeat_thresholds_pending_root_review'
    profile['thresholds'].update(body_orientation_rad=1e-4, body_linear_velocity_m_s=1e-4,
        body_angular_velocity_rad_s=1e-4, robot_joint_velocity_rad_s=1e-4, gripper_open_amount=1e-4,
        native_joint_position_units=1e-8, native_joint_velocity_units_s=1e-8,
        schedule_numeric_absolute=1e-8, internal_numeric_absolute=1e-8)
    profile['threshold_selection'] = 'preregistered by root before native TRAIN repeats; no automatic output-dependent fit/relaxation'
    profile['schedule_precision_note'] = 'numeric schedule/internal state 1e-8; driver t0 uses inherited 20us clock bound; discrete values exact'
    return profile


def native_inputs(protocol):
    repo = Path(protocol['baseline_repo']);files = {}
    for spec in TASK_SPECS.values():
        verified = verify_native_sources(repo, spec)
        files.update({str(repo / name): expected for name, expected in verified.items()})
        scene = repo / 'third_party/RLBench/rlbench/task_ttms' / (spec.task_name + '.ttm')
        files[str(scene)] = sha(scene)
    return files


def build_plan(manifest_path, protocol_path, output):
    manifest_path, protocol_path = Path(manifest_path).resolve(), Path(protocol_path).resolve()
    manifest, protocol = read(manifest_path), read(protocol_path)
    from our_mani.multitask.protocol import validate_protocol
    validate_protocol(protocol)
    require(manifest.get('version') == 'multitask_r10_grouped_small_demonstrations_v1'
            and manifest.get('complete') is True and manifest.get('TEST_used_for_training_calibration_or_selection') is False,
            'use the audited original TRAIN/VAL small-cache registry')
    rows = [row for row in manifest['episodes'] if row['split'] == 'train']
    cases = []
    for task in TASK_SPECS:
        candidates = [row for row in rows if row['task'] == task]
        require(candidates, 'missing TRAIN source for ' + task)
        row = candidates[0]  # Stable registry order; never selected by reset/outcome.
        path, cache = Path(row['source_path']).resolve(), (ROOT / row['npz_path']).resolve()
        require('train_package_compressed' in path.parts and path.name == 'data.h5'
                and source_stat(path) == row['source_file_stat'] and sha(cache) == row['npz_sha256'],
                'TRAIN source/cache identity changed')
        require(type(row['variation']) is int and row['variation'] in TASK_SPECS[task].supported_variations,
                'source variation unavailable')
        for repeat in range(4):
            cases.append(dict(case_id=task + '__repeat' + str(repeat), task=task, repeat=repeat,
                pair='fit' if repeat < 2 else 'independent_validation',
                source=dict(task=task, raw_episode_id=row['source_episode'], path=str(path), variation=row['variation']),
                source_file_stat=row['source_file_stat'], original_small_cache=str(cache),
                original_small_cache_sha256=row['npz_sha256'], replica=0,
                seed=int(protocol.get('seed', 0))))
    source_paths = [Path(__file__).resolve(), manifest_path, protocol_path,
                    ROOT / 'artifacts/framework/aligned_r09/acceptance_tolerances.json',
                    ROOT / 'scripts/run_local_sim.sh',
                    *[ROOT / 'our_mani/multitask' / name for name in
                      ('reset_parity.py', 'simulation.py', 'maps.py', 'native_tasks.py', 'specs.py', 'protocol.py')],
                    ROOT / 'our_mani/sensor_alignment/author_rgbd.py',
                    *sorted((ROOT / 'our_mani/aligned').glob('*.py')),
                    *sorted((ROOT / 'our_mani').glob('*.py'))]
    from scripts import collect_multitask_recovery_r10 as recovery
    source_paths.extend(ROOT/name for name in recovery.OWNED_SOURCES)
    inputs = {str(path.resolve()): sha(path) for path in source_paths};inputs.update(native_inputs(protocol))
    for key in ('development_reset_profile','RML_factory_hook'):
        inputs.update(recovery.permission_dependencies(protocol[key]))
    profile = preregistered_profile(read(ROOT / 'artifacts/framework/aligned_r09/acceptance_tolerances.json'))
    plan = dict(version=PLAN_VERSION, created_utc=now(), split='train', cases=cases, protocol=protocol,
        manifest_path=str(manifest_path), protocol_path=str(protocol_path),
        inputs_sha256=inputs, profile=profile, native_started=False, GPU_used=False, teacher_used=False,
        planned_cases=56, tasks=14, fresh_process_per_case=True, maximum_CPU_workers=4,
        counted_actions_per_case=0, future_state_or_success_used_to_select_sources=False,
        repeat_design='same first TRAIN registry source/variation/derived seed/replica per task; 0+1 fit, 2+3 independent fresh-process validation',
        threshold_relaxation_automatic=False, automatic_profile_freeze=False,
        execution_gate='R09 complete hard-protocol formal400 AND canonical ten-demo index complete')
    plan['plan_sha256'] = digest(plan)
    write(output, plan)
    return plan


def verify_plan(path):
    plan = read(path)
    require(plan.get('version') == PLAN_VERSION and plan.get('split') == 'train'
            and plan.get('plan_sha256') == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}), 'reset-repeat plan changed')
    require(len(plan['cases']) == 56 and len({row['case_id'] for row in plan['cases']}) == 56
            and {row['task'] for row in plan['cases']} == set(TASK_SPECS), 'exact 14x4 fresh-repeat budget required')
    require(plan['protocol'] == read(plan['protocol_path']) and plan['profile'] == preregistered_profile(
        read(ROOT / 'artifacts/framework/aligned_r09/acceptance_tolerances.json')),
        'protocol or preregistered thresholds changed inside the plan')
    needed = {str(Path(__file__).resolve()), plan['protocol_path'], plan['manifest_path'],
        str(ROOT / 'scripts/run_local_sim.sh'),
        str(ROOT / 'artifacts/framework/aligned_r09/acceptance_tolerances.json'),
        *[str(ROOT / 'our_mani/multitask' / name) for name in
          ('reset_parity.py', 'simulation.py', 'maps.py', 'native_tasks.py', 'specs.py', 'protocol.py')],
        *native_inputs(plan['protocol'])}
    require(needed <= set(plan['inputs_sha256']), 'reset-repeat source hash table is incomplete')
    registry = read(plan['manifest_path'])['episodes']
    for task in TASK_SPECS:
        selected = [row for row in plan['cases'] if row['task'] == task]
        require(sorted(row['repeat'] for row in selected) == [0, 1, 2, 3]
                and all(row['source'] == selected[0]['source'] and row['seed'] == selected[0]['seed']
                        and row['seed'] == int(plan['protocol'].get('seed', 0))
                        and row['replica'] == 0 and row['pair'] == ('fit' if row['repeat'] < 2 else 'independent_validation')
                        for row in selected), 'repeat source/seed must remain identical')
        original = next(row for row in registry if row['split'] == 'train' and row['task'] == task)
        expected_source = dict(task=task, raw_episode_id=original['source_episode'], path=original['source_path'],
                               variation=original['variation'])
        require(selected[0]['source'] == expected_source, 'repeat source must stay the first fixed TRAIN registry source')
        require(all(row['source_file_stat'] == original['source_file_stat']
                    and row['original_small_cache'] == str((ROOT / original['npz_path']).resolve())
                    and row['original_small_cache_sha256'] == original['npz_sha256'] for row in selected),
                'repeat HDF5/cache provenance differs from the fixed registry source')
    for name, expected in plan['inputs_sha256'].items():
        require(sha(name) == expected, 'reset-repeat source changed: ' + name)
    for row in plan['cases']:
        require(source_stat(row['source']['path']) == row['source_file_stat']
                and sha(row['original_small_cache']) == row['original_small_cache_sha256'], 'TRAIN source identity changed')
    require(sha(Path(__file__)) == SOURCE_SHA256, 'running entrypoint source changed')
    return plan


def require_completed_delivery():
    r09 = ROOT / 'artifacts/framework/aligned_r09'
    report_path, index_path = r09 / 'final_report.json', r09 / 'demo_index.json'
    require(report_path.is_file() and index_path.is_file(), 'wait for formal400 AND canonical ten demos')
    report, index = read(report_path), read(index_path)
    require(report.get('complete') is True and report.get('protocol_acceptance_passed') is True
            and report.get('total_test_runs') == report.get('expected_test_runs') == 400
            and not report.get('acceptance_errors') and report.get('fixed_candidate_evaluation_release'), 'formal400 gate failed')
    require(read(r09 / 'status.json').get('stage') == 'complete', 'R09 orchestrator has not completed')
    require(index.get('version') == 'r09_canonical_ten_demo_index_v1' and index.get('complete') is True
            and index.get('completed') == index.get('expected') == 10
            and index.get('formal_final_report_sha256') == sha(report_path)
            and index.get('original_formal_statistics_replaced') is False
            and index.get('replay_results_in_formal_statistics') is False, 'canonical demo gate failed')
    keys = {(row['method'], row['task'], row['raw_episode_id']) for row in index['entries']}
    require(len(index['entries']) == 10 and keys == {(method, task, DEMO_FIRST_RAW[task])
            for method in DEMO_METHODS for task in DEMO_TASKS},
            'demo inventory differs from fixed ten-case policy')
    video_hashes = {}
    for row in index['entries']:
        path = (ROOT / row['video']).resolve()
        require(path.is_relative_to(r09.resolve()) and sha(path) == row['video_sha256']
                and row['video_validation']['decoded_frame_count'] > 0, 'decoded demo evidence changed or missing')
        video_hashes[str(path)] = row['video_sha256']
    return dict(formal_report_path=str(report_path), formal_report_sha256=sha(report_path),
        demo_index_path=str(index_path), demo_index_sha256=sha(index_path), video_sha256=video_hashes,
        formal400_completed=True, canonical_ten_demos_completed=True, success_outcomes_used_for_thresholds=False)


def validate_case_identity(record, row, plan, execution):
    require(record.get('version') == 'r10_native_TRAIN_reset_repeat_case_v1'
            and record.get('completed') is True and record.get('split') == 'train'
            and all(record.get(key) == row[key] for key in ('case_id', 'task', 'repeat', 'pair', 'source'))
            and record.get('plan_sha256') == plan['plan_sha256']
            and record.get('execution_inputs_sha256') == execution['sha256']
            and record.get('counted_actions') == 0 and record.get('teacher_used') is False
            and record.get('policy_used') is False and record.get('GPU_used') is False
            and record.get('fresh_native_process') is True
            and type(record.get('process_id')) is int and record['process_id'] > 0,
            'native case identity/scope changed')
    from our_mani.multitask.simulation import development_seed
    seed, identity = development_seed(row['task'], row['source'], 'train', seed=row['seed'], replica=0)
    reset = record['recorded_initial_conditions']['reset_receipt']
    require(reset['task'] == row['task'] and reset['raw_episode_id'] == row['source']['raw_episode_id']
            and reset['variation'] == row['source']['variation'] and reset['source_split'] == 'train'
            and reset['seed'] == seed and reset['seed_identity'] == identity and reset['replica'] == 0
            and reset['original_source_replayed'] is False, 'native reset source/derived seed differs')
    receipt = record['alignment_receipt']
    require(receipt['clock']['ticks'] == 0 and abs(receipt['actual_physics_seconds']) < 1e-6
            and abs(receipt['actual_task_seconds']) < 1e-6, 'reset-only child advanced episode physics')


def execution_inputs(plan, guard):
    payload = dict(version='r10_reset_repeat_execution_inputs_v1', plan_sha256=plan['plan_sha256'],
        release_gate=guard, teacher_used=False, GPU_used=False, counted_actions=0)
    payload['sha256'] = digest(payload)
    return payload


def verify_execution(path, plan):
    execution = read(path)
    require(execution.get('sha256') == digest({k: v for k, v in execution.items() if k != 'sha256'})
            and execution.get('plan_sha256') == plan['plan_sha256'], 'native audit execution identity differs')
    require(require_completed_delivery() == execution['release_gate'], 'formal400/demo gate changed')
    return execution


def child(plan_path, execution_path, case_id, output):
    # No native import is reachable before every source and delivery gate.
    plan = verify_plan(plan_path);execution = verify_execution(execution_path, plan)
    rows = [row for row in plan['cases'] if row['case_id'] == case_id]
    require(len(rows) == 1, 'unknown/ambiguous reset repeat')
    row, env = rows[0], None
    try:
        from our_mani.multitask import simulation
        from scripts import collect_multitask_recovery_r10 as recovery
        captures=dict(simulations=[],paths=[])
        hook_plan=dict(protocol=plan['protocol'],inputs=plan['inputs_sha256'])
        with recovery.scoped_owned_hooks(hook_plan,captures):
            env = simulation.AlignedDevelopmentSimulation(row['task'], row['source']['path'], source=row['source'], split='train',
                protocol=plan['protocol'], seed=row['seed'], replica=0,
                servo_lead=plan['protocol'].get('local_servo_lead', 1.4))
            env.reset_development(variation=row['source']['variation'])
            observation, robot = env.observe()
            recorded = capture_recorded_initial_conditions(env, observation, robot)
            receipt = env.alignment_receipt()
            require(receipt['clock']['ticks'] == 0 and env.time == 0
                    and abs(receipt['actual_physics_seconds']) < 1e-6
                    and abs(receipt['actual_task_seconds']) < 1e-6, 'reset-only case advanced a counted episode')
            env.close();env = None
        verify_plan(plan_path);verify_execution(execution_path, plan)
        require(len(captures['simulations'])==1 and getattr(captures['simulations'][0],'reset_recovery_receipt',{}).get('complete') is True,'current-core reset must use the actual shared physical reset subclass')
        record = dict(version='r10_native_TRAIN_reset_repeat_case_v1', completed=True, split='train',
            case_id=case_id, task=row['task'], repeat=row['repeat'], pair=row['pair'], source=row['source'],
            recorded_initial_conditions=recorded, alignment_receipt=receipt, process_id=os.getpid(),
            plan_sha256=plan['plan_sha256'], execution_inputs_sha256=execution['sha256'],
            teacher_used=False, policy_used=False, GPU_used=False, counted_actions=0, fresh_native_process=True,
            initialization_only=True, original_HDF5_motion_replayed=False,
            shared_development_reset_hook_receipts=[getattr(sim,'reset_recovery_receipt',None) for sim in captures['simulations']])
        write(Path(output) / 'case.json', record)
        return record
    finally:
        if env is not None: env.close()


def run_case(plan_path, execution_path, row, output, cancelled):
    require(not cancelled.is_set(), 'reset-repeat pending launch cancelled after infrastructure failure')
    plan = verify_plan(plan_path);execution = verify_execution(execution_path, plan)
    directory = Path(output) / 'cases' / row['case_id'];record_path = directory / 'case.json'
    command = ['bash', 'scripts/run_local_sim.sh', str(Path(__file__).resolve().relative_to(ROOT)),
        '--child', str(Path(plan_path).resolve()), '--execution-inputs', str(Path(execution_path).resolve()),
        '--case-id', row['case_id'], '--out', str(directory.resolve())]
    if record_path.exists():
        record, parent = read(record_path), read(directory / 'process_receipt.json')
        validate_case_identity(record, row, plan, execution)
        require(parent.get('version') == 'r10_reset_repeat_process_receipt_v1'
                and parent.get('case_id') == row['case_id'] and parent.get('launch_command') == command
                and parent.get('plan_sha256') == plan['plan_sha256']
                and parent.get('execution_inputs_sha256') == execution['sha256']
                and parent.get('child_source_sha256') == SOURCE_SHA256
                and parent.get('launcher_source_sha256') == sha(ROOT / 'scripts/run_local_sim.sh')
                and parent.get('fresh_process_per_case') is True
                and parent.get('child_process_id') == record['process_id']
                and parent.get('child_record_sha256') == sha(record_path) and parent.get('exit_code') == 0
                and parent.get('log_sha256') == sha(directory / 'reset.log'),
                'invalid cached native reset repeat')
        return record_path
    require(not directory.exists() or not any(directory.iterdir()), 'preserve partial native attempt; choose a new audit output version')
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', PYTHONHASHSEED='0')
    started = now();log_path = directory / 'reset.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        code = process.wait()
    if code != 0 or not record_path.is_file():
        write(directory / 'infrastructure_error.json', dict(exit_code=code, counted_as_policy_failure=False,
            child_directory=str(directory), log_sha256=sha(log_path), native_reset_evidence_accepted=False))
        raise RuntimeError('native reset infrastructure failure: ' + str(directory))
    record = read(record_path)
    validate_case_identity(record, row, plan, execution)
    write(directory / 'process_receipt.json', dict(version='r10_reset_repeat_process_receipt_v1',
        case_id=row['case_id'], plan_sha256=plan['plan_sha256'], execution_inputs_sha256=execution['sha256'],
        child_source_sha256=SOURCE_SHA256, launcher_source_sha256=sha(ROOT / 'scripts/run_local_sim.sh'),
        launch_command=command, launcher_process_id=process.pid,
        child_process_id=record['process_id'], child_record_sha256=sha(record_path), exit_code=code,
        started_utc=started, finished_utc=now(), log_sha256=sha(log_path), fresh_process_per_case=True))
    return record_path


def summarize(plan, record_paths, execution_path):
    require(len(record_paths) == 56, 'all 56 reset cases required before repeat evidence')
    records = {read(path)['case_id']: (path, read(path)) for path in record_paths}
    require(len(records) == 56, 'duplicate reset case')
    per_task, checks = {}, []
    for task in TASK_SPECS:
        summary = {}
        for purpose, indices in (('fit', (0, 1)), ('independent_validation', (2, 3))):
            a, b = [records[task + '__repeat' + str(index)] for index in indices]
            require(all(row['task'] == task and row['recorded_initial_conditions']['reset_receipt']['task'] == task
                        for _, row in (a, b)), 'repeat case metric task differs from grouped task')
            check = compare_initial_conditions(a[1], b[1], plan['profile'], require_frozen=False)
            checks.append(dict(task=task, purpose=purpose, a_path=str(a[0].resolve()), a_sha256=sha(a[0]),
                b_path=str(b[0].resolve()), b_sha256=sha(b[0]), **check))
            summary[purpose + '_passed'] = check['recorded_conditions_match']
        per_task[task] = dict(fit_pairs=1, independent_validation_pairs=1,
            passed=summary['fit_passed'] and summary['independent_validation_passed'], **summary)
    return dict(version='r10_native_TRAIN_reset_repeat_pending_report_v1', created_utc=now(), complete=True,
        split='train', TEST_used=False, native_repeat_execution_complete=True, synthetic_mock_data_used=False,
        actual_cases=56, tasks=14, fresh_process_per_case=True, teacher_used=False, GPU_used=False,
        fit_repeat_data_passed=all(row['fit_passed'] for row in per_task.values()),
        independent_repeat_validation_passed=all(row['independent_validation_passed'] for row in per_task.values()),
        per_task=per_task, checks=checks, thresholds=plan['profile']['thresholds'],
        thresholds_sha256=digest(plan['profile']['thresholds']), threshold_relaxation_automatic=False,
        automatically_frozen=False, formal_comparison_ready=False, pending_root_review=True,
        plan_sha256=plan['plan_sha256'], execution_inputs_sha256=sha(execution_path),
        record_files_sha256={str(path.resolve()): sha(path) for path in record_paths},
        interpretation='preregistered TRAIN reset-only diagnostics; root must review real differences before a new frozen tolerance profile')


def run_queue(plan_path, output, *, workers=4):
    require(1 <= workers <= 4, 'maximum reset-repeat CPU concurrency is four')
    plan = verify_plan(plan_path);guard = require_completed_delivery();output = Path(output).resolve()
    require(output.is_relative_to(BASE.resolve()), 'native audit output must stay in new R10 artifacts')
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'RUNNING.json'
    # Exclusive creation also blocks a second parent from racing the same child directories.
    with marker.open('x') as stream:
        json.dump(dict(parent_process_id=os.getpid(), plan_sha256=plan['plan_sha256'], started_utc=now()), stream)
    try:
        return _run_locked_queue(plan_path, output, plan, guard, workers=workers)
    finally:
        marker.unlink()


def _run_locked_queue(plan_path, output, plan, guard, *, workers):
    path = output / 'execution_inputs.json';expected = execution_inputs(plan, guard)
    if path.exists(): require(read(path) == expected, 'native audit resume inputs changed')
    else: write(path, expected)
    cancelled = threading.Event();results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, plan_path, path, row, output, cancelled) for row in plan['cases']]
        try:
            for future in as_completed(futures): results.append(future.result())
        except Exception:
            cancelled.set()
            for future in futures: future.cancel()
            raise
    verify_plan(plan_path);verify_execution(path, plan)
    report = summarize(plan, results, path)
    write(output / 'pending_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', metavar='PLAN')
    mode.add_argument('--child', metavar='PLAN')
    parser.add_argument('--manifest', default='artifacts/framework/multitask_r10/demonstrations/manifest.json')
    parser.add_argument('--protocol', default='artifacts/framework/multitask_r10/common_training_smoke_config.json')
    parser.add_argument('--execution-inputs')
    parser.add_argument('--case-id')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.prepare: value = build_plan(ROOT / args.manifest, ROOT / args.protocol, ROOT / args.out)
    elif args.child:
        require(args.execution_inputs and args.case_id, 'native child needs frozen execution inputs and exact case')
        value = child(ROOT / args.child, ROOT / args.execution_inputs, args.case_id, ROOT / args.out)
    else: value = run_queue(ROOT / args.run, ROOT / args.out, workers=args.workers)
    print(json.dumps(dict(complete=value.get('complete'), cases=value.get('actual_cases', value.get('planned_cases')),
                         automatically_frozen=False), ensure_ascii=False))


if __name__ == '__main__':
    if '--run' in sys.argv:
        from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair,process_visibility
        inventory=process_visibility()
        require(inventory['host_visibility_available'] and not inventory['live_project_native_processes'] and not inventory['same_uid_inventory_incomplete'],'duplicate native job or missing host process view')
        with exclusive_pair():main()
    else:main()
