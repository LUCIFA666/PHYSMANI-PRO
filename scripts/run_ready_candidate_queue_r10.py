"""Execute task-ready geometry VAL and prepare own-policy TRAIN/CAL labels.

1. --prepare-inputs writes a new shared configuration and lock requirements.
2. ROOT freezes/reviews actual inputs; --prepare builds a fixed stage queue.
3. --run executes fresh CPU-native CV/Context geometry cases, once per parent.

No TEST source is accepted. geometry_val is a fresh two-method formal VAL queue;
train/cal collect separate executed critic labels. No GPU/native work occurs
in either preparation command. An infrastructure failure stops pending work;
the failed attempt is preserved and is never a policy negative or retried.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, verify_inputs
from our_mani.multitask.ready_candidate_queue import (METHODS, SCOPE, STAGES, VERSION,
    build_jobs, checked, critic_manifest, pointer, read, require, select_original_rows, validate_raw_child)
from our_mani.multitask.specs import TASK_SPECS

BASE = ROOT / 'artifacts/framework/multitask_r10'
SELF = Path(__file__).resolve()
MODULE = ROOT / 'our_mani/multitask/ready_candidate_queue.py'
EVALUATOR = ROOT / 'scripts/evaluate_multitask_case_r10.py'
LAUNCHER = ROOT / 'scripts/run_local_sim.sh'
STOP = ROOT / 'runtime/framework/multitask_r10_STOP'
MANIFESTS = {split: ROOT / 'artifacts/framework/protocol' / name for split, name in
    (('train', 'train_manifest.json'), ('validation', 'validation_manifest.json'))}


def utc(): return datetime.now(timezone.utc).isoformat()


def inside(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(BASE) and path != BASE, 'new outputs must remain under desktop R10 artifacts')
    return path


def dependency_pointer(path):
    path = (ROOT / path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=file_sha(path))


def locked(path, files):
    path = (ROOT / path).resolve()
    expected = {value for key, value in files.items() if (ROOT / key).resolve() == path}
    require(len(expected) == 1 and path.is_file() and expected == {file_sha(path)},
            'missing or changed frozen input: ' + str(path))


def prepare_inputs(args):
    from our_mani.multitask.factory import artifact_path, continuation_identity
    from our_mani.multitask.interaction import CandidateConfig, InteractionLibrary, MPCConfig
    base = (ROOT / args.base_config).resolve()
    child = (ROOT / args.child).resolve()
    launcher = (ROOT / args.launcher).resolve()
    require(child.parent == ROOT / 'scripts' and child.is_file(), 'explicit local trained-task runner source required')
    require(launcher.parent == ROOT / 'scripts' and launcher.is_file(), 'explicit local CPU-native launcher source required')
    out = inside(args.output)
    require(not out.exists(), 'preserve prior setup; choose a fresh input preparation directory')
    config = deepcopy(read(base))
    for key in ('development_demo_scope', 'task_model_coverage_certificate', 'task_validation_release'):
        config.pop(key, None)
    config['formal_comparison_ready'] = False
    config['execution_scope'] = 'task_ready_geometry_VAL_and_separate_own_policy_TRAIN_CAL'
    config['executed_candidate_collection_scope'] = dict(version=SCOPE, tasks=args.tasks,
        methods=list(METHODS), TRAIN_parents_per_task=100, replicas_per_parent=1,
        TRAIN_first_selection='TRAIN_explore_once', CAL_first_selection='geometry',
        downstream_selection='geometry', TEST_used_for_fitting=False,
        development_demos_used_for_fitting=False, teacher_hold_used_for_critic_fitting=False,
        formal_geometry_VAL_separate_from_critic_CAL=True,
        interaction_library_identities={},
        child_source=pointer(ROOT, child), numerical_controller_tuned_on_VAL_or_TEST=False)
    config['executed_candidate_collection_scope']['cpu_native_launcher'] = pointer(ROOT, launcher)
    counts = {}
    sources = {}
    for stage, split in STAGES.items():
        rows = select_original_rows(read(MANIFESTS[split]), split, args.tasks)
        counts[stage] = dict(Counter(row['task'] for row in rows))
        for row in rows:
            value = sources.setdefault(row['path'], dict(stages=[], task=row['task'],
                raw_episode_id=row['raw_episode_id'], path=row['path'],
                source_row_sha256=digest(row), size_bytes=Path(row['path']).stat().st_size))
            value['stages'].append(stage)
    config.setdefault('continuation_identities', {})
    candidate = CandidateConfig(**config['candidate_config']); mpc = MPCConfig(**config['mpc_config'])
    libraries = {}
    for task in args.tasks:
        library = InteractionLibrary.load(artifact_path(ROOT, config['learned_artifacts'][task]['interaction']))
        require(library.task_spec.task_name == task, 'trained interaction library task differs')
        fit = library.metadata.get('fit_configuration', {})
        require(library.metadata.get('development_support_override') is False
                and fit.get('development_only') is False and fit.get('minimum_completed_parent_sources', 0) >= 8,
                'production fitted interaction support gate is required')
        config['continuation_identities'][task] = {predictor:
            continuation_identity(ROOT, config, task, predictor, library, candidate, mpc)['sha256']
            for predictor in ('cv', 'context')}
        config['executed_candidate_collection_scope']['interaction_library_identities'][task] = library.sha256
        libraries[task] = dict(sha256=library.sha256, phase_order=list(library.phase_order),
            critic_query_phase=library.phase_order[0], downstream_phases='geometry_only')
    out.mkdir(parents=True)
    config_path = out / 'configuration.json'; atomic_json(config_path, config)
    known = {str(path.relative_to(ROOT)):file_sha(path) for path in
        (base, child, SELF, MODULE, EVALUATOR, LAUNCHER, launcher, config_path, *MANIFESTS.values())}
    if args.base_lock:
        for name, checksum in read(ROOT / args.base_lock)['files'].items():
            known.setdefault(name, checksum)
    budget = dict(original_parents=counts,
        TRAIN_native_cases=sum(counts['train'].values()) * 2,
        CAL_native_cases=sum(counts['cal'].values()) * 2,
        formal_geometry_VAL_native_cases=sum(counts['geometry_val'].values()) * 2,
        TRAIN_per_task_predictor=100, CAL_per_task_predictor=counts['cal'],
        TRAIN_exploration_once_per_episode=True, maximum_ticks_per_case=200, dt=.05,
        formal_five_method_VAL_not_in_this_queue=True, formal_TEST_not_in_this_queue=True)
    requirements = dict(version=VERSION, created_utc=utc(), known_dependency_hashes=known,
        source_files_requiring_ROOT_full_byte_freeze=list(sources.values()),
        inherited_lock_is_inventory_not_execution_approval=bool(args.base_lock),
        require_root_approved_fresh_lock=True, source_rows_and_splits_unchanged=True,
        native_started=False, GPU_started=False)
    atomic_json(out / 'source_lock_requirements.json', requirements)
    report = dict(version=VERSION, created_utc=utc(), configuration=pointer(ROOT, config_path),
        dependencies=pointer(ROOT, out / 'source_lock_requirements.json'), budget=budget,
        trained_libraries=libraries, ready_for_ROOT_input_freeze=True,
        trained_critics_exist=False, formal_VAL_started=False, formal_TEST_started=False,
        native_started=False, GPU_started=False,
        prepare_commands={stage:['.conda-env/bin/python', str(SELF.relative_to(ROOT)), '--prepare',
            '--stage', stage, '--config', str(config_path.relative_to(ROOT)), '--frozen-inputs',
            'ROOT_REVIEWED_NEW_LOCK.json', '--output', str((out / (stage + '_execution')).relative_to(ROOT)),
            '--plan', str((out / (stage + '_plan.json')).relative_to(ROOT))] for stage in STAGES},
        critic_fit_commands=[dict(task=task, predictor=predictor, command=[
            '.conda-env/bin/python', 'scripts/train_multitask_candidate_success_r10.py',
            '--train-manifest', str((out / 'train_execution/executed_continuation_manifest.json').relative_to(ROOT)),
            '--validation-manifest', str((out / 'cal_execution/executed_continuation_manifest.json').relative_to(ROOT)),
            '--task', task, '--predictor', predictor, '--interaction-library',
            config['learned_artifacts'][task]['interaction']['path'], '--continuation-sha256',
            config['continuation_identities'][task][predictor], '--out',
            str((out / 'critics' / (task + '_' + predictor)).relative_to(ROOT))])
            for task in args.tasks for predictor in ('cv', 'context')])
    atomic_json(out / 'preparation.json', report)
    return report


def prepare(args):
    config_path = (ROOT / args.config).resolve(); lock_path = (ROOT / args.frozen_inputs).resolve()
    config, lock = read(config_path), read(lock_path)
    require(lock.get('approved_for_execution') is True, 'ROOT must approve fresh actual execution input freeze')
    verify_inputs(ROOT, lock['files'])
    scope = config.get('executed_candidate_collection_scope', {})
    require(scope.get('version') == SCOPE and scope.get('methods') == list(METHODS),
            'explicit executed candidate TRAIN/CAL scope required')
    require(config.get('formal_comparison_ready') is False and scope.get('TEST_used_for_fitting') is False,
            'collection cannot be formal TEST or use TEST fitting')
    child = checked(ROOT, scope['child_source'])
    launcher = checked(ROOT, scope['cpu_native_launcher'])
    split = STAGES[args.stage]; manifest_path = MANIFESTS[split]
    for path in (SELF, MODULE, EVALUATOR, launcher, child, config_path, manifest_path):
        locked(path, lock['files'])
    rows = select_original_rows(read(manifest_path), split, scope['tasks'])
    for row in rows:
        locked(row['path'], lock['files'])
    jobs = build_jobs(rows, args.stage, file_sha(config_path), file_sha(lock_path), config['seed'])
    plan = dict(version=VERSION, created_utc=utc(), stage=args.stage, split=split,
        config=pointer(ROOT, config_path), manifest=pointer(ROOT, manifest_path),
        frozen_inputs=pointer(ROOT, lock_path), parent_source=pointer(ROOT, SELF),
        contracts_source=pointer(ROOT, MODULE), child_source=pointer(ROOT, child),
        original_evaluator=pointer(ROOT, EVALUATOR), launcher=pointer(ROOT, launcher),
        output=str(inside(args.output).relative_to(ROOT)), jobs=jobs,
        expected_cases=len(jobs), per_task_original_parents=dict(Counter(row['task'] for row in rows)),
        methods=list(METHODS), replicas=1, automatic_retry=False,
        teacher_hold_and_development_demo_sources_used=False, unexecuted_candidates_labelled=False,
        formal_VAL_metric=args.stage == 'geometry_val', formal_TEST_used=False, native_started=False)
    plan['sha256'] = digest(plan)
    atomic_json(inside(args.plan), plan)
    return plan


def verify_plan(path):
    plan = read(path)
    require(plan.get('version') == VERSION
            and plan.get('sha256') == digest({key:value for key,value in plan.items() if key != 'sha256'}),
            'fixed queue plan changed')
    for key in ('config', 'manifest', 'frozen_inputs', 'parent_source', 'contracts_source',
                'child_source', 'original_evaluator', 'launcher'):
        checked(ROOT, plan[key])
    lock = read(ROOT / plan['frozen_inputs']['path'])
    require(lock.get('approved_for_execution') is True, 'fresh input freeze lacks ROOT approval')
    verify_inputs(ROOT, lock['files'])
    config = read(ROOT / plan['config']['path'])
    scope = config['executed_candidate_collection_scope']
    rows = select_original_rows(read(MANIFESTS[STAGES[plan['stage']]]), STAGES[plan['stage']], scope['tasks'])
    expected = build_jobs(rows, plan['stage'], plan['config']['sha256'], plan['frozen_inputs']['sha256'], config['seed'])
    require(plan['jobs'] == expected and plan['expected_cases'] == len(expected), 'fixed complete source queue differs')
    return plan, config


def process_start_ticks(pid):
    try: return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError): return None


def run(plan_path, workers):
    require(1 <= workers <= 8, 'use one to eight isolated CPU-native workers')
    plan, config = verify_plan(plan_path)
    out = inside(plan['output']); out.mkdir(parents=True, exist_ok=True)
    cancelled = threading.Event(); guard = threading.Lock(); completed = {}; errors = []
    started_utc = utc()
    with (out / 'queue_owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(stage):
            by_task_method = Counter((record['task'], record['method']) for record in completed.values())
            return dict(version=VERSION, stage=stage, collection_stage=plan['stage'],
                complete=len(completed) == plan['expected_cases'] and not errors,
                expected=plan['expected_cases'], completed=len(completed), errors=list(errors),
                per_task_method=[dict(task=task, method=method, completed=count)
                    for (task, method), count in sorted(by_task_method.items())],
                plan_sha256=plan['sha256'], parent_process_id=os.getpid(),
                parent_process_start_ticks=process_start_ticks(os.getpid()), started_utc=started_utc,
                updated_utc=utc(), automatic_retry=False, no_infrastructure_failure_scored_as_policy_failure=True,
                formal_VAL_metric=plan['stage'] == 'geometry_val', formal_TEST_used=False, GPU_budget_claimed=False)

        def one(job):
            if STOP.exists() or cancelled.is_set(): return
            verify_plan(plan_path)
            canonical = out / 'results' / (job['job_id'] + '.json')
            if canonical.exists():
                record = read(canonical); receipt = record.get('parent_case_process_receipt', {})
                raw = checked(ROOT, receipt.get('raw_child'))
                child_record = validate_raw_child(ROOT, plan, job, raw, config)
                require({key:value for key,value in record.items() if key != 'parent_case_process_receipt'} == child_record
                        and receipt.get('plan_sha256') == plan['sha256']
                        and receipt.get('parent_source') == plan['parent_source'],
                        'cached canonical receipt does not match its original child/parent plan')
            else:
                child_out = out / 'cases' / job['job_id'] / 'attempt0000'
                require(not child_out.exists(), 'prior failed/incomplete attempt preserved; no automatic retry: ' + str(child_out))
                child_out.mkdir(parents=True)
                command = ['bash', plan['launcher']['path'], plan['child_source']['path'],
                    '--config', plan['config']['path'], '--manifest', plan['manifest']['path'],
                    '--frozen-inputs', plan['frozen_inputs']['path'], '--split', plan['split'],
                    '--task', job['identity']['task'], '--raw-episode-id', job['source_raw_episode_id'],
                    '--method', job['identity']['method'], '--replica', '0',
                    '--out', str(child_out.relative_to(ROOT))]
                if job['exploration_seed'] is not None:
                    command += ['--exploration-seed', str(job['exploration_seed'])]
                log = child_out.parent / 'attempt0000.stdout.log'
                with log.open('wb') as stream:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    launch = dict(created_utc=utc(), command=command, launcher_process_id=process.pid,
                        launcher_process_start_ticks=process_start_ticks(process.pid),
                        parent_process_id=os.getpid(), child_source=plan['child_source'],
                        original_evaluator=plan['original_evaluator'], plan_sha256=plan['sha256'])
                    atomic_json(child_out.parent / 'launch.json', launch)
                    code = process.wait()
                raw = child_out / 'case.json'
                require(code == 0 and raw.is_file(),
                        'native infrastructure/protocol failure is not a policy result; see ' + str(log))
                record = validate_raw_child(ROOT, plan, job, raw, config)
                record['parent_case_process_receipt'] = dict(raw_child=pointer(ROOT, raw),
                    child_process_id=record['process_id'], launcher_process_id=process.pid,
                    launch=pointer(ROOT, child_out.parent / 'launch.json'), log=pointer(ROOT, log),
                    plan_sha256=plan['sha256'], parent_source=plan['parent_source'])
                verify_plan(plan_path)
                atomic_json(canonical, record)
            with guard:
                completed[job['job_id']] = record
                atomic_json(out / 'status.json', status('running'), overwrite=True)
            print(json.dumps(dict(job=job['job_id'], task=record['task'], method=record['method'],
                native_success=record['official_task_success'], labelled_decisions=len(record['continuation_attempts'])),
                ensure_ascii=False), flush=True)

        atomic_json(out / 'status.json', status('starting'), overwrite=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, job) for job in plan['jobs']]
            for future in as_completed(futures):
                if future.cancelled(): continue
                try: future.result()
                except BaseException as error:
                    cancelled.set()
                    with guard:
                        errors.append(dict(type=type(error).__name__, message=str(error), utc=utc()))
                        atomic_json(out / 'status.json', status('stopping_after_infrastructure_error'), overwrite=True)
                    for pending in futures: pending.cancel()
        report = status('complete' if len(completed) == plan['expected_cases'] and not errors else 'incomplete')
        report['stop_requested'] = STOP.exists()
        if report['complete']:
            verify_plan(plan_path)
            if plan['stage'] in ('train', 'cal'):
                manifest = critic_manifest(plan, [(job, out / 'results' / (job['job_id'] + '.json'))
                    for job in plan['jobs']], ROOT)
                manifest_path = out / 'executed_continuation_manifest.json'
                if manifest_path.exists():
                    require(read(manifest_path) == manifest, 'previous completed critic manifest differs')
                else:
                    atomic_json(manifest_path, manifest)
                report['executed_manifest'] = pointer(ROOT, manifest_path)
            else:
                report['per_task_method_outcomes'] = [dict(task=task, method=method,
                    completed=sum(record['task'] == task and record['method'] == method for record in completed.values()),
                    successes=sum(record['task'] == task and record['method'] == method
                        and record['official_task_success'] for record in completed.values()))
                    for task in plan['per_task_original_parents'] for method in METHODS]
                report['five_method_task_release_issued'] = False
                report['per_method_TEST_release_issued'] = False
            report['zero_candidate_rollouts'] = [dict(task=record['task'], method=record['method'],
                raw_episode_id=record['raw_episode_id'], native_success=record['official_task_success'])
                for record in completed.values() if not record['continuation_attempts']]
        atomic_json(out / 'status.json', report, overwrite=True)
        if errors: raise RuntimeError(errors[0]['message'])
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-inputs', action='store_true')
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--base-config'); parser.add_argument('--base-lock')
    parser.add_argument('--child', default='scripts/run_trained_task_case_r10.py')
    parser.add_argument('--launcher', default='scripts/run_local_sim.sh')
    parser.add_argument('--tasks', nargs='+', choices=tuple(TASK_SPECS))
    parser.add_argument('--config'); parser.add_argument('--frozen-inputs')
    parser.add_argument('--stage', choices=tuple(STAGES)); parser.add_argument('--output'); parser.add_argument('--plan')
    parser.add_argument('--cpu-workers', type=int, default=2)
    args = parser.parse_args()
    needed = ('base_config', 'tasks', 'output') if args.prepare_inputs else (
        ('config', 'frozen_inputs', 'stage', 'output', 'plan') if args.prepare else ('plan',))
    if any(getattr(args, key) is None for key in needed): parser.error('required fields: ' + ', '.join(needed))
    result = prepare_inputs(args) if args.prepare_inputs else prepare(args) if args.prepare else run(ROOT / args.plan, args.cpu_workers)
    print(json.dumps({key:value for key,value in result.items() if key != 'jobs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
