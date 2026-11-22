"""One task's fixed CV_geometry, Context_geometry and PhysMani cases.

Original full source manifests remain unchanged. No partial three-way release,
automatic failed-attempt retry, or task-success threshold is introduced here.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_multitask_case_r10 as evaluator
from scripts.run_multitask_fresh_queue_r10 import build_jobs
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, verify_inputs, validate_completed_case
from our_mani.multitask.ready_candidate_queue import select_original_rows
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.video_validation import validate_video

SELF = Path(__file__).resolve()
CHILD = ROOT / 'scripts/run_supported_geometry_formal_case_v9_r10.py'
OURS = ('cv_geometry', 'context_geometry', 'physmani')
BASE = ROOT / 'artifacts/framework/multitask_r10'
STOP = ROOT / 'runtime/framework/multitask_r10_STOP'
WRAPPER_VERSION = 'r10_trained_supported_three_method_formal_case_v9'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT), 'receipt pointer leaves project')
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def checked(value):
    path = (ROOT / value['path']).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and file_sha(path) == value['sha256'],
        'missing or changed receipt input: ' + str(path))
    return path


def inside(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(BASE) and path != BASE, 'fresh outputs belong under desktop R10 artifacts')
    return path


def rows_for(manifest, split, task):
    require(task in TASK_SPECS, 'only the remaining fourteen declared tasks are supported')
    require(split in ('validation', 'test'), 'formal task queue accepts VAL/TEST only')
    if split == 'validation':
        return select_original_rows(manifest, split, [task])
    values = manifest.get('episodes', manifest.get('entries', []))
    selected = [row for row in values if row.get('task') == task]
    require(selected, 'official task sources missing')
    evaluator.select_source(manifest, 'test', task, selected[0]['raw_episode_id'])
    return selected


def prepare(args):
    config_path, manifest_path, lock_path = [(ROOT / name).resolve() for name in
        (args.config, args.manifest, args.frozen_inputs)]
    config, manifest, lock = map(read, (config_path, manifest_path, lock_path))
    require(lock.get('approved_for_execution') is True, 'freeze actual inputs before formal execution')
    verify_inputs(ROOT, lock['files'])
    require(config.get('task_model_coverage_certificate') is not None,
        'task Context/interaction/geometry/observation models must be ready')
    for path in (SELF, CHILD, config_path, manifest_path,
            ROOT / 'scripts/run_multitask_fresh_queue_r10.py',
            ROOT / 'our_mani/multitask/ready_candidate_queue.py',
            ROOT / args.launcher, ROOT / 'scripts/run_aligned_gpu.sh'):
        evaluator.locked_digest(path, lock['files'])
    rows = rows_for(manifest, args.split, args.task)
    jobs = build_jobs(rows, args.split, OURS, file_sha(config_path), file_sha(lock_path),
        replicas=1, explore_first=False, record_demos=args.split == 'test', seed=config['seed'])
    plan = dict(version='r10_fresh_policy_queue_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), task=args.task, split=args.split,
        config=pointer(config_path), manifest=pointer(manifest_path), frozen_inputs=pointer(lock_path),
        parent_source=pointer(SELF), child_source=pointer(CHILD), output=str(inside(args.output).relative_to(ROOT)),
        source_episodes=len(rows), methods=list(OURS), replicas=1, jobs=jobs, expected_cases=len(OURS) * len(rows),
        case_isolation='fresh_simulator_process_per_episode',
        original_development_source_motion_replayed=False, demo_per_method_per_task=1,
        unexecuted_candidates_labelled=False, infra_failures_are_not_policy_failures=True,
        official_TEST_used_for_fitting=False, automatic_retry=False,
        task_execution_contract=dict(schema='r10_task_formal_case_execution_contract_v1', task=args.task,
            original_evaluator=pointer(evaluator.__file__), wrapper=pointer(CHILD),
            wrapper_schema_version=WRAPPER_VERSION,
            launchers=dict(ours=pointer(ROOT / args.launcher), physmani=pointer(ROOT / 'scripts/run_aligned_gpu.sh'))))
    plan['sha256'] = digest(plan)
    path = inside(args.plan)
    require(not path.exists(), 'preserve prior fixed queue plan')
    atomic_json(path, plan)
    return plan


def verify_plan(path):
    plan = read(path)
    require(plan.get('version') == 'r10_fresh_policy_queue_v1'
        and plan.get('sha256') == digest({key:value for key,value in plan.items() if key != 'sha256'})
        and plan.get('methods') == list(OURS) and plan.get('replicas') == 1
        and plan.get('automatic_retry') is False, 'formal task queue changed')
    for field in ('config', 'manifest', 'frozen_inputs', 'parent_source', 'child_source'):
        checked(plan[field])
    require(plan['parent_source'] == pointer(SELF) and plan['child_source'] == pointer(CHILD),
        'actual formal parent/wrapper source differs')
    lock = read(checked(plan['frozen_inputs']))
    require(lock.get('approved_for_execution') is True, 'actual input freeze lacks execution approval')
    verify_inputs(ROOT, lock['files'])
    config = read(checked(plan['config']))
    rows = rows_for(read(checked(plan['manifest'])), plan['split'], plan['task'])
    expected = build_jobs(rows, plan['split'], OURS, plan['config']['sha256'], plan['frozen_inputs']['sha256'],
        replicas=1, explore_first=False, record_demos=plan['split'] == 'test', seed=config['seed'])
    require(plan['jobs'] == expected and plan['source_episodes'] == len(rows)
        and plan['expected_cases'] == len(OURS) * len(rows), 'original complete task/method coverage differs')
    contract = plan['task_execution_contract']
    require(contract['wrapper'] == plan['child_source'] and contract['task'] == plan['task']
        and contract['wrapper_schema_version'] == WRAPPER_VERSION, 'formal wrapper execution contract differs')
    for item in [contract['original_evaluator'], contract['wrapper'], *contract['launchers'].values()]:
        evaluator.locked_pointer(item, lock['files'])
    return plan, config


def validate_child(plan, job, path):
    record = read(path)
    validate_completed_case(record, job['identity'])
    contract = plan['task_execution_contract']
    witness = record.get('task_execution_witness', {})
    require(witness.get('actual_wrapper') == contract['wrapper']
        and witness.get('original_evaluator') == contract['original_evaluator']
        and witness.get('task') == plan['task'] and witness.get('split') == plan['split']
        and witness.get('actual_process_id') == record['process_id'], 'actual formal case witness differs')
    base = read(checked(witness['base_evaluator_receipt']))
    require(base == {key:value for key,value in record.items() if key != 'task_execution_witness'},
        'wrapper changed original evaluator results')
    hook = read(checked(witness['hook_receipt']))
    require(hook.get('version') == WRAPPER_VERSION and hook.get('complete') is True
        and hook.get('identity') == job['identity'] and hook.get('actual_wrapper') == contract['wrapper']
        and hook.get('actual_process_id') == record['process_id']
        and hook.get('original_TEST_reset_unmodified') is (plan['split'] == 'test')
        and hook.get('shared_development_hooks_applied') is (plan['split'] == 'validation'),
        'formal physical cleanup/reset witness differs')
    require(not record.get('demo_only') and not record.get('development_demo_execution'),
        'development demo cannot enter formal task results')
    if job['record_demo']:
        validate_video(ROOT, record.get('video'))
        require(record.get('recorded_video_in_formal_metrics') is True, 'first TEST demo must remain a formal result')
    else:
        require(record.get('video') is None and record.get('recorded_video_in_formal_metrics') is False,
            'only the fixed first TEST case records video')
    return record


def run(plan_path, workers, case_timeout=900):
    require(1 <= workers <= 2 and case_timeout > 0, 'explicit CPU worker and wall timeout limits required')
    plan, _ = verify_plan(plan_path)
    output = inside(plan['output']); output.mkdir(parents=True, exist_ok=True)
    completed, errors = {}, []
    cancelled = threading.Event(); guard = threading.Lock(); launch_guard = threading.Lock()
    gpu_slot=threading.Semaphore(1)
    started = datetime.now(timezone.utc).isoformat()

    def status(stage):
        return dict(stage=stage, complete=stage == 'complete', split=plan['split'], task=plan['task'],
            completed=len(completed), expected=plan['expected_cases'], errors=list(errors),
            stop_requested=STOP.exists(), plan_sha256=plan['sha256'], started_utc=started,
            formal_TEST_used=plan['split'] == 'test', automatic_retry=False,
            outcomes=[dict(method=method, completed=sum(r['method'] == method for r in completed.values()),
                successes=sum(r['method'] == method and r['official_task_success'] for r in completed.values()))
                for method in OURS])

    def execute_one(job):
        if cancelled.is_set() or STOP.exists():
            return
        verify_plan(plan_path)
        canonical = output / 'results' / (job['job_id'] + '.json')
        if canonical.exists():
            record = read(canonical)
            parent = record['parent_case_process_receipt']
            child = ROOT / parent['child_path']
            require(file_sha(child) == parent['child_sha256'], 'existing raw child changed')
            raw = validate_child(plan, job, child)
            require(raw == {key:value for key,value in record.items() if key != 'parent_case_process_receipt'}
                and parent['plan_sha256'] == plan['sha256'], 'cached parent/raw result differs')
        else:
            case_dir = output / 'cases' / job['job_id']
            child_out = case_dir / 'attempt0000'
            command = ['bash', plan['task_execution_contract']['launchers']['physmani' if job['identity']['method']=='physmani' else 'ours']['path'],
                plan['child_source']['path'], '--config', plan['config']['path'],
                '--manifest', plan['manifest']['path'], '--frozen-inputs', plan['frozen_inputs']['path'],
                '--split', plan['split'], '--task', plan['task'], '--raw-episode-id', job['source_raw_episode_id'],
                '--method', job['identity']['method'], '--replica', '0', '--out', str(child_out.relative_to(ROOT))]
            if job['record_demo']:
                command += ['--record-demo']
            log = case_dir / 'attempt0000.log'
            with launch_guard:
                if cancelled.is_set() or STOP.exists():
                    return
                require(not case_dir.exists(), 'preserve unfinished/failed attempt; no automatic redispatch')
                child_out.mkdir(parents=True)
                stream = log.open('xb')
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                atomic_json(case_dir / 'launch.json', dict(command=command, launcher_pid=process.pid,
                    parent_pid=os.getpid(), plan_sha256=plan['sha256'], actual_parent_source=pointer(SELF)))
            with stream:
                try:
                    code = process.wait(timeout=case_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    raise RuntimeError('native wall timeout; retained as infrastructure error, not policy failure')
            require(code == 0 and (child_out / 'case.json').is_file(),
                'native infrastructure/protocol error, not policy failure: ' + str(log))
            child = child_out / 'case.json'
            record = validate_child(plan, job, child)
            record['parent_case_process_receipt'] = dict(child_path=str(child.relative_to(ROOT)), child_sha256=file_sha(child),
                child_process_id=record['process_id'], launcher_process_id=process.pid, command=command,
                log_path=str(log.relative_to(ROOT)), log_sha256=file_sha(log), plan_sha256=plan['sha256'],
                parent_source_sha256=plan['parent_source']['sha256'], returncode=code)
            verify_plan(plan_path)
            atomic_json(canonical, record)
        with guard:
            completed[job['job_id']] = record
            atomic_json(output / 'status.json', status('running'), overwrite=True)
        print(json.dumps(dict(task=plan['task'], method=record['method'], job=job['job_id'],
            native_success=record['official_task_success'])), flush=True)

    def one(job):
        try:
            if job['identity']['method']=='physmani':
                with gpu_slot:return execute_one(job)
            return execute_one(job)
        except BaseException:
            with launch_guard:
                cancelled.set()
            raise

    with (output / 'queue_owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(output / 'status.json', status('starting'), overwrite=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, job) for job in plan['jobs']]
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    future.result()
                except BaseException as error:
                    cancelled.set()
                    with guard:
                        errors.append(repr(error))
                        atomic_json(output / 'status.json', status('stopping_after_infrastructure_error'), overwrite=True)
                    for pending in futures:
                        pending.cancel()
        result = status('complete' if len(completed) == plan['expected_cases'] and not errors else 'incomplete')
        atomic_json(output / 'status.json', result, overwrite=True)
        require(result['complete'], 'formal task queue incomplete; no three-method release issued')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--config'); parser.add_argument('--manifest'); parser.add_argument('--frozen-inputs')
    parser.add_argument('--task', choices=tuple(TASK_SPECS)); parser.add_argument('--split', choices=('validation', 'test'))
    parser.add_argument('--output'); parser.add_argument('--launcher', default='scripts/run_local_sim_single_thread_r10.sh')
    parser.add_argument('--cpu-workers', type=int, default=4); parser.add_argument('--case-timeout', type=int, default=900)
    args = parser.parse_args()
    if args.prepare and any(getattr(args, field) is None for field in
            ('config', 'manifest', 'frozen_inputs', 'task', 'split', 'output')):
        parser.error('--prepare requires config, manifest, frozen-inputs, task, split and output')
    result = prepare(args) if args.prepare else run(ROOT / args.plan, args.cpu_workers, args.case_timeout)
    print(json.dumps({key:value for key,value in result.items() if key != 'jobs'}), flush=True)


if __name__ == '__main__':
    main()
