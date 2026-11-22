"""Immutable R10 per-source queues: independent CPU ours and GPU baseline."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import (METHODS, atomic_json, case_identity, digest,
    file_sha, validate_completed_case, verify_inputs)
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.video_validation import validate_video

BASE = ROOT / 'artifacts/framework/multitask_r10'
CHILD = ROOT / 'scripts/evaluate_multitask_case_r10.py'
STOP = ROOT / 'runtime/framework/multitask_r10_STOP'


def inside(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError('all new queue outputs must stay in desktop multitask_r10')
    return path


def read_rows(path, split):
    value = json.loads(Path(path).read_text())
    rows = value.get('episodes', value.get('entries'))
    if not isinstance(rows, list):
        raise ValueError('source manifest must contain real episode rows')
    rows = [row for row in rows if row.get('task') in TASK_SPECS
            and row.get('split', split) == split]
    keys = [(row['task'], row['raw_episode_id']) for row in rows]
    if not rows or len(keys) != len(set(keys)) or any(not row.get('path') for row in rows):
        raise ValueError('unique real task/source identities and paths required')
    return rows


def build_jobs(rows, split, methods, config_sha, lock_sha, *, replicas=1, explore_first=False,
               record_demos=False, seed=0):
    if replicas < 1 or (split == 'test' and replicas != 1):
        raise ValueError('official TEST cannot acquire synthetic replicas')
    if explore_first and (split != 'train' or any(not method.endswith('_geometry') for method in methods)):
        raise ValueError('first-candidate exploration is TRAIN geometry collection only')
    if record_demos and split != 'test':
        raise ValueError('fixed demo policy uses official TEST cases')
    first = {}
    for row in rows:
        first.setdefault(row['task'], row['raw_episode_id'])
    jobs = []
    for row in rows:
        for replica in range(replicas):
            for method in methods:
                parent = row['raw_episode_id'] if split != 'test' else None
                raw = f'{parent}__policy_aug_r{replica}' if parent is not None else row['raw_episode_id']
                exploration = (int(digest([seed, row['task'], row['raw_episode_id'], replica, method, 'first'])[:8], 16)
                               if explore_first else None)
                identity = case_identity(split=split, task=row['task'], raw_episode_id=raw, method=method,
                    source_row=row, config_sha256=config_sha, inputs_sha256=lock_sha,
                    parent_raw_episode_id=parent, replica=replica, exploration_seed=exploration)
                jobs.append(dict(job_id=identity['sha256'][:20], identity=identity,
                    source_raw_episode_id=row['raw_episode_id'], row=row, replica=replica,
                    exploration_seed=exploration,
                    record_demo=bool(record_demos and first[row['task']] == row['raw_episode_id'])))
    if len({job['job_id'] for job in jobs}) != len(jobs):
        raise ValueError('duplicate queue job identity')
    return jobs


def prepare(args):
    config_path, manifest_path, lock_path = [ROOT / x for x in (args.config, args.manifest, args.frozen_inputs)]
    config = json.loads(config_path.read_text())
    lock = json.loads(lock_path.read_text())
    verify_inputs(ROOT, lock['files'])
    for path in (config_path, manifest_path, CHILD):
        expected = lock['files'].get(str(path.resolve()), lock['files'].get(str(path.relative_to(ROOT))))
        if expected != file_sha(path):
            raise ValueError('external input lock must bind config/manifest/child: ' + str(path))
    rows = read_rows(manifest_path, args.split)
    if args.tasks:
        if args.split == 'test':
            raise ValueError('prepare the complete official TEST; subset execution is a queue continuation')
        rows = [row for row in rows if row['task'] in args.tasks]
    expected = args.expected_sources if args.expected_sources is not None else {'train': 1400, 'validation': 275, 'test': 560}[args.split]
    if len(rows) != expected:
        raise ValueError(f'source coverage {len(rows)} differs from declared {expected}')
    methods = tuple(args.methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('unique known methods required')
    jobs = build_jobs(rows, args.split, methods, file_sha(config_path), file_sha(lock_path),
        replicas=args.replicas, explore_first=args.explore_first, record_demos=args.record_demos,
        seed=config.get('seed', 0))
    plan = dict(version='r10_fresh_policy_queue_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        split=args.split, config=dict(path=str(config_path.relative_to(ROOT)), sha256=file_sha(config_path)),
        manifest=dict(path=str(manifest_path.relative_to(ROOT)), sha256=file_sha(manifest_path)),
        frozen_inputs=dict(path=str(lock_path.relative_to(ROOT)), sha256=file_sha(lock_path)),
        parent_source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)),
        child_source=dict(path=str(CHILD.relative_to(ROOT)), sha256=file_sha(CHILD)),
        output=str(inside(args.output).relative_to(ROOT)), source_episodes=len(rows),
        methods=list(methods), replicas=args.replicas, jobs=jobs, expected_cases=len(jobs),
        case_isolation='fresh_simulator_process_per_episode',
        original_development_source_motion_replayed=False, demo_per_method_per_task=1,
        unexecuted_candidates_labelled=False, infra_failures_are_not_policy_failures=True,
        official_TEST_used_for_fitting=False)
    plan['sha256'] = digest(plan)
    atomic_json(inside(args.plan), plan)
    return plan


def verify_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('version') != 'r10_fresh_policy_queue_v1' or plan.get('sha256') != digest({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('queue plan changed')
    for field in ('config', 'manifest', 'frozen_inputs', 'parent_source', 'child_source'):
        pointer = plan[field]
        if file_sha(ROOT / pointer['path']) != pointer['sha256']:
            raise ValueError('queue dependency changed: ' + field)
    lock = json.loads((ROOT / plan['frozen_inputs']['path']).read_text())
    verify_inputs(ROOT, lock['files'])
    return plan


def run(plan_path, cpu_workers, gpu_workers):
    if min(cpu_workers, gpu_workers) < 1 or cpu_workers > 8 or gpu_workers > 4:
        raise ValueError('use at most eight CPU and four GPU isolated workers on the shared machine')
    plan = verify_plan(plan_path)
    from scripts.collect_multitask_training_r10 import require_r09_release
    require_r09_release(ROOT / 'artifacts/framework/aligned_r09/final_report.json')
    index = json.loads((ROOT / 'artifacts/framework/aligned_r09/demo_index.json').read_text())
    if (index.get('complete') is not True or index.get('expected') != 10
            or index.get('completed') != 10 or len(index.get('entries', [])) != 10):
        raise ValueError('complete the current two-task ten-demo delivery before new native evaluation')
    output = inside(plan['output']); output.mkdir(parents=True, exist_ok=True)
    marker = output / 'queue_owner.lock'
    import fcntl
    with marker.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cancel = threading.Event()
        status_lock = threading.Lock()
        completed = {}
        errors = []

        def one(job):
            if STOP.exists() or cancel.is_set():
                return
            verify_plan(plan_path)
            canonical = output / 'results' / (job['job_id'] + '.json')
            new_record = False
            if canonical.exists():
                record = json.loads(canonical.read_text())
                validate_completed_case(record, job['identity'])
                receipt = record['parent_case_process_receipt']
                if file_sha(ROOT / receipt['child_path']) != receipt['child_sha256']:
                    raise ValueError('cached raw child changed')
                child_record = json.loads((ROOT / receipt['child_path']).read_text())
                if ({key: value for key, value in record.items() if key != 'parent_case_process_receipt'} != child_record
                        or child_record.get('process_id') != receipt.get('child_process_id')
                        or receipt.get('plan_sha256') != plan['sha256']
                        or receipt.get('parent_source_sha256') != plan['parent_source']['sha256']):
                    raise ValueError('cached parent is not its exact raw child under this plan')
            else:
                job_base = output / 'cases' / job['job_id']; job_base.mkdir(parents=True, exist_ok=True)
                number = 0
                while (job_base / f'attempt{number:04d}').exists():
                    number += 1
                child_out = job_base / f'attempt{number:04d}'
                child_out.mkdir()
                wrapper = 'scripts/run_aligned_gpu.sh' if job['identity']['method'] == 'physmani' else 'scripts/run_local_sim.sh'
                command = ['bash', wrapper, str(CHILD.relative_to(ROOT)),
                    '--config', plan['config']['path'], '--manifest', plan['manifest']['path'],
                    '--frozen-inputs', plan['frozen_inputs']['path'], '--split', plan['split'],
                    '--task', job['identity']['task'], '--raw-episode-id', job['source_raw_episode_id'],
                    '--method', job['identity']['method'], '--replica', str(job['replica']),
                    '--out', str(child_out.relative_to(ROOT))]
                if job['exploration_seed'] is not None:
                    command += ['--exploration-seed', str(job['exploration_seed'])]
                if job['record_demo']:
                    command += ['--record-demo']
                log = job_base / f'attempt{number:04d}.log'
                with log.open('wb') as stream:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    return_code = process.wait()
                child = child_out / 'case.json'
                if return_code != 0 or not child.is_file():
                    raise RuntimeError(f'fresh native job failed, not a policy SR result: {job["job_id"]}; see {log}')
                record = json.loads(child.read_text())
                validate_completed_case(record, job['identity'])
                record['parent_case_process_receipt'] = dict(child_path=str(child.relative_to(ROOT)),
                    child_sha256=file_sha(child), child_process_id=record['process_id'], launcher_process_id=process.pid,
                    command=command, log_path=str(log.relative_to(ROOT)), log_sha256=file_sha(log),
                    plan_sha256=plan['sha256'], parent_source_sha256=plan['parent_source']['sha256'])
                verify_plan(plan_path)
                new_record = True
            if job['record_demo']:
                validate_video(ROOT, record.get('video'))
            elif record.get('video') is not None:
                raise ValueError('only the fixed first case may save a task/method demo')
            if new_record:
                atomic_json(canonical, record)
            with status_lock:
                completed[job['job_id']] = record
                atomic_json(output / 'status.json', dict(stage='running', completed=len(completed),
                    expected=plan['expected_cases'], split=plan['split'], errors=errors,
                    formal_TEST_used=plan['split'] == 'test', plan_sha256=plan['sha256']), overwrite=True)
            print(json.dumps(dict(job=job['job_id'], method=job['identity']['method'],
                task=job['identity']['task'], success=record['official_task_success'])), flush=True)

        pools = [ThreadPoolExecutor(max_workers=cpu_workers), ThreadPoolExecutor(max_workers=gpu_workers)]
        try:
            futures = [pools[int(job['identity']['method'] == 'physmani')].submit(one, job) for job in plan['jobs']]
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    future.result()
                except Exception as exc:
                    cancel.set(); errors.append(repr(exc))
                    for pending in futures:
                        pending.cancel()
        finally:
            for pool in pools:
                pool.shutdown(wait=True, cancel_futures=True)
        report = dict(stage='complete' if len(completed) == plan['expected_cases'] and not errors else 'incomplete',
            complete=len(completed) == plan['expected_cases'] and not errors,
            plan_sha256=plan['sha256'], expected=plan['expected_cases'], completed=len(completed),
            errors=errors, stop_requested=STOP.exists(), no_infrastructure_failure_scored_as_policy_failure=True,
            official_TEST_used_for_fitting=False)
        atomic_json(output / 'status.json', report, overwrite=True)
        if errors:
            raise RuntimeError(errors[0])
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--run', action='store_true')
    p.add_argument('--plan', required=True)
    p.add_argument('--config'); p.add_argument('--manifest'); p.add_argument('--frozen-inputs')
    p.add_argument('--output'); p.add_argument('--split', choices=('train', 'validation', 'test'))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--tasks', nargs='+', choices=tuple(TASK_SPECS))
    p.add_argument('--expected-sources', type=int); p.add_argument('--replicas', type=int, default=1)
    p.add_argument('--explore-first', action='store_true'); p.add_argument('--record-demos', action='store_true')
    p.add_argument('--cpu-workers', type=int, default=8); p.add_argument('--gpu-workers', type=int, default=4)
    args = p.parse_args()
    if args.prepare and any(getattr(args, key) is None for key in ('config', 'manifest', 'frozen_inputs', 'output', 'split')):
        p.error('--prepare requires --config --manifest --frozen-inputs --output --split')
    result = prepare(args) if args.prepare else run(ROOT / args.plan, args.cpu_workers, args.gpu_workers)
    print(json.dumps({k: v for k, v in result.items() if k != 'jobs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
