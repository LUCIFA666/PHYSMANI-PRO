"""Immutable stage receipts and an explicit subprocess queue; default is dry-run.

Data adapters and current-network controllers are explicit hooks, not imports of
the historical five-method R10 suite. Missing hooks are reported as BLOCKED.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .registry import METHOD, TASKS, sha

PLAN_SCHEMA = 'current_network16_execution_plan_20261104_v1'
SPLIT_SCHEMA = 'current_network16_metadata_episode_split_20261104_v1'
STAGES = ('data_check', 'preprocess', 'fit', 'validation', 'freeze', 'test')


class PipelineBlocked(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def pointer(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked_pointer(root, value):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise PipelineBlocked('Exact path/SHA pointer is missing')
    path = (Path(root) / value['path']).resolve(strict=True)
    if not path.is_file() or sha(path) != value['sha256']:
        raise PipelineBlocked('Bound file missing/changed: ' + str(path))
    return path


def verify_sources(root, sources):
    if not isinstance(sources, dict) or not sources:
        raise PipelineBlocked('Explicit source closure required')
    for name, checksum in sources.items():
        checked_pointer(root, dict(path=name, sha256=checksum))


def validate_split(root, value):
    """Only open the split JSON. Episode paths/TEST payloads remain unopened."""
    path = checked_pointer(root, value)
    split = load(path)
    if split.get('schema') != SPLIT_SCHEMA:
        raise PipelineBlocked('Use the new current-network16 source-parent split')
    rows = split.get('episodes')
    if not isinstance(rows, list) or not rows:
        raise PipelineBlocked('Complete metadata episodes required')
    groups, identities, eligible, metadata_groups = {}, set(), [], {}
    for row in rows:
        if (row.get('task') not in TASKS or row.get('split') not in ('train', 'validation', 'test')
                or not isinstance(row.get('source_parent_group'), str) or not row['source_parent_group']
                or not isinstance(row.get('raw_episode_id'), str) or not row['raw_episode_id']
                or not isinstance(row.get('role'), str)
                or type(row.get('eligible_for_pipeline')) is not bool
                or row.get('offline_eligible') is not row['eligible_for_pipeline']):
            raise PipelineBlocked('Explicit task/split/parent/eligibility metadata required')
        ident = (row['task'], row['split'], row['raw_episode_id'])
        if ident in identities:
            raise PipelineBlocked('Duplicate source episode')
        identities.add(ident)
        group = row['source_parent_group']
        metadata_groups.setdefault(group, []).append(row)
        active = row['split'] == 'test' or row['eligible_for_pipeline']
        if active:
            if group in groups and groups[group] != row['split']:
                raise PipelineBlocked('Source-parent leakage across active train/VAL/futureTEST: ' + group)
            groups[group] = row['split']
        # TEST eligibility=False prohibits offline decoding; it must not delete
        # fixed future Native TEST cases from their denominator.
        if row['split'] in ('validation', 'test'):
            if row.get('native_eval_included') is not True:
                raise PipelineBlocked('All fixed Native VAL/TEST rows must be explicitly retained')
            eligible.append(row)
        elif row['eligible_for_pipeline']:
            if row['role'] in ('excluded', 'reserved', 'reserved_fresh_centre_VAL', 'calibration'):
                raise PipelineBlocked('Reserved/calibration/excluded source cannot enter this pipeline')
            eligible.append(row)
    for group, group_rows in metadata_groups.items():
        if len({x['split'] for x in group_rows}) <= 1:
            continue
        affected = [x for x in group_rows if x['split'] != 'test']
        if any(x['eligible_for_pipeline'] or x.get('offline_quarantined') is not True for x in affected):
            raise PipelineBlocked('Cross-split metadata group requires explicit offline quarantine: ' + group)
    for task in TASKS:
        for part in ('train', 'validation', 'test'):
            if not any(r['task'] == task and r['split'] == part for r in eligible):
                raise PipelineBlocked('Missing fixed task/split denominator: ' + task + '/' + part)
    return path, eligible


def validate_plan(root, path):
    root, path = Path(root).resolve(), Path(path).resolve(strict=True)
    plan = load(path)
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('method') != METHOD
            or tuple(plan.get('tasks', ())) != TASKS
            or type(plan.get('preparation_only')) is not bool
            or type(plan.get('execution_authorized')) is not bool
            or plan.get('native_success_authority') != 'task.success()'):
        raise PipelineBlocked('Explicit single-current-network sixteen-task plan required')
    verify_sources(root, plan.get('source_sha256'))
    if plan.get('inventory') is not None:
        declared = load(checked_pointer(root, plan['inventory']))
        if declared.get('method') != METHOD or tuple(t['task'] for t in declared['tasks']) != TASKS:
            raise PipelineBlocked('Exact single-method task inventory changed')
    return plan


def checked_hook(root, plan, stage):
    hook = plan.get('hooks', {}).get(stage)
    if hook is None:
        raise PipelineBlocked('No real ' + stage + ' hook has been connected')
    if (hook.get('method') != METHOD or hook.get('stage') != stage
            or hook.get('resource_lock_owner') != 'pipeline'
            or hook.get('resource') not in ('cpu', 'training', 'native')
            or not isinstance(hook.get('argv'), list) or not hook['argv']
            or any(not isinstance(v, str) or not v for v in hook['argv'])
            or not isinstance(hook.get('timeout_seconds'), (int, float))
            or not math.isfinite(hook['timeout_seconds']) or not 0 < hook['timeout_seconds'] <= 86400):
        raise PipelineBlocked('Actual argv, bounded timeout and pipeline-owned resources required')
    entrypoint = checked_pointer(root, hook.get('entrypoint'))
    if str(entrypoint) not in [str((Path(root) / v).resolve()) for v in hook['argv'] if '{' not in v]:
        raise PipelineBlocked('The frozen entrypoint must occur in the actual command')
    if entrypoint.name == 'evaluate_multitask_case_r10.py':
        raise PipelineBlocked('Historical R10 is not a current-network execution hook')
    verify_sources(root, hook.get('source_sha256'))
    if not entrypoint.is_relative_to(Path(root).resolve()):
        raise PipelineBlocked('Use a versioned project entrypoint')
    if hook['source_sha256'].get(str(entrypoint), hook['source_sha256'].get(str(entrypoint.relative_to(root)))) != hook['entrypoint']['sha256']:
        raise PipelineBlocked('Hook entrypoint must be in its exact source closure')
    if any(plan['source_sha256'].get(name) != value for name, value in hook['source_sha256'].items()):
        raise PipelineBlocked('Every hook dependency must also be in the whole pipeline source closure')
    if stage in ('validation', 'test'):
        readiness = plan.get('task_adapter_readiness', {})
        if any(any(readiness.get(t, {}).get(k) is not True for k in
                   ('public_observation', 'task_phase_controller', 'native_reset')) for t in TASKS):
            raise PipelineBlocked('Every real current-network task adapter must be ready before Native evaluation')
        if (hook['resource'] != 'native' or hook.get('all_tasks') != list(TASKS)
                or hook.get('native_success_authority') != 'task.success()'
                or hook.get('policy_input_contract') != 'current_RGBD_public_K_T_proprioception_only'
                or hook.get('Native_GT_contact_or_instance_masks_as_policy_inputs') is not False):
            raise PipelineBlocked('Current-network public-input and native-success adapter for all16 required')
    elif hook['resource'] == 'native':
        raise PipelineBlocked('Offline data/fit stages cannot launch Native')
    if stage == 'fit' and (hook.get('fit_filter') != dict(split='train', role='fit', offline_eligible=True)
            or hook.get('selection_uses_optimizer') is not False
            or hook.get('TEST_payload_decode_allowed') is not False):
        raise PipelineBlocked('Fit uses only offline-eligible TRAIN rolefit; selection is evaluation-only')
    return hook


def _stage_receipt(root, state_root, stage, plan_sha):
    path = Path(state_root) / stage / 'receipt.json'
    value = load(path)
    if (value.get('complete') is not True or value.get('plan_sha256') != plan_sha
            or value.get('stage') != stage or type(value.get('actual_process_returncode')) is not int
            or value['actual_process_returncode'] != 0):
        raise PipelineBlocked('Completed exact previous stage required: ' + stage)
    for item in value.get('completion_files', []):
        checked_pointer(root, item)
    return value


def stage_admission(root, plan_path, stage, state_root=None):
    if stage not in STAGES:
        raise PipelineBlocked('Unknown stage')
    plan = validate_plan(root, plan_path)
    split_path, rows = validate_split(root, plan.get('split_manifest'))
    plan_sha = sha(plan_path)
    prior = None
    index = STAGES.index(stage)
    if index:
        if state_root is None:
            raise PipelineBlocked('Previous stages have not completed')
        prior = _stage_receipt(root, state_root, STAGES[index - 1], plan_sha)
    hook = None if stage == 'freeze' else checked_hook(root, plan, stage)
    if stage in ('validation', 'test', 'freeze'):
        if state_root is None:
            raise PipelineBlocked('An actual fitted current-network model is required')
        fitted = _stage_receipt(root, state_root, 'fit', plan_sha)
        checked_pointer(root, fitted.get('selected_model'))
        if fitted.get('current_network_end_to_end_ready') is not True:
            raise PipelineBlocked('A component-only pretrain is not a complete current-network candidate/MPC model')
        if stage in ('freeze', 'test'):
            val = _stage_receipt(root, state_root, 'validation', plan_sha)
            expected = [digest(r) for r in rows if r['split'] == 'validation']
            if val.get('ordered_source_rows') != expected or val.get('all_cases_actual_closed') is not True:
                raise PipelineBlocked('All fixed VAL cases, including task failures, must close before freezing')
            if val.get('selected_model') != fitted['selected_model']:
                raise PipelineBlocked('VAL model differs from the fitted selected model')
        if stage == 'test':
            frozen = _stage_receipt(root, state_root, 'freeze', plan_sha)
            if (frozen.get('selected_model') != fitted['selected_model']
                    or frozen.get('source_sha256') != plan['source_sha256']
                    or frozen.get('split_manifest') != plan['split_manifest']):
                raise PipelineBlocked('TEST requires the identical frozen model, method, source and split')
    return dict(plan=plan, plan_sha256=plan_sha, split_path=split_path, rows=rows, hook=hook, prior=prior)


def dry_run(root, plan_path, state_root=None):
    plan = validate_plan(root, plan_path)
    stages = []
    for stage in STAGES:
        try:
            admission = stage_admission(root, plan_path, stage, state_root)
            count = len([r for r in admission['rows'] if r['split'] == ('test' if stage == 'test' else 'validation')]) if stage in ('validation', 'test') else 1
            stages.append(dict(stage=stage, status='READY' if plan['preparation_only'] is False and plan['execution_authorized'] is True else 'PREPARED', jobs=count))
        except (PipelineBlocked, FileNotFoundError) as error:
            stages.append(dict(stage=stage, status='BLOCKED', reason=str(error)))
    return dict(complete=True, method=METHOD, tasks=list(TASKS), stages=stages,
        preparation_only=plan['preparation_only'], Native_started=False, training_started=False,
        subprocesses_started=0, model_forward_calls=0, raw_TEST_files_opened=0,
        pipeline_executable=all(s['status'] == 'READY' for s in stages),
        task_adapter_readiness=plan['task_adapter_readiness'])


@contextmanager
def resource_locks(root, resource):
    """Cooperative locks. Hooks must not reacquire these locks themselves."""
    names = ['multitask_current_network16_pipeline.lock']
    if resource in ('native', 'training'):
        names += ['fixed_TRAIN104_geometry_native.lock', 'TRAIN_postcomplete_diagnostics_v1106.lock']
    if resource == 'training':
        names += ['multitask_current_network16_training.lock']
    handles = []
    try:
        for name in names:
            path = Path(root) / 'runtime/framework' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a')
            handles.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for stream in reversed(handles):
            stream.close()


def _run_command(root, argv, output, timeout, stop_paths):
    """Actual wait/reap on every path, including failed owner writes/STOP/timeout."""
    process = None
    code, reason = None, None
    start = time.monotonic()
    try:
        with (output / 'stdout.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_new(output / 'owner.json', dict(pid=process.pid, argv=argv, started_utc=datetime.now(timezone.utc).isoformat()))
            while process.poll() is None:
                if any(p.exists() for p in stop_paths):
                    reason = 'administrative_STOP'
                    break
                if time.monotonic() - start > timeout:
                    reason = 'wall_timeout'
                    break
                time.sleep(.2)
            if reason is None:
                code = process.wait()
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            code = process.wait()
            write_new(output / 'owner.actual_exit.json', dict(actual_process_returncode=code,
                actual_wait_completed=True, stop_reason=reason,
                wall_seconds=time.monotonic() - start))
    return code, reason


def run_stage(root, plan_path, stage, state_root, expected_plan_sha256):
    """Only an explicit final plan can launch. Preparation never invokes this."""
    root, state_root = Path(root).resolve(), Path(state_root).resolve()
    if sha(plan_path) != expected_plan_sha256:
        raise PipelineBlocked('External final plan SHA required')
    admitted = stage_admission(root, plan_path, stage, state_root)
    plan, hook = admitted['plan'], admitted['hook']
    if plan['preparation_only'] is not False or plan['execution_authorized'] is not True:
        raise PipelineBlocked('Preparation-only/unapproved plan cannot create processes or state')
    stops = [state_root / 'STOP', root / 'runtime/framework/STOP', root / 'runtime/framework/local_mpc_20261102/STOP']
    if any(p.exists() for p in stops):
        raise PipelineBlocked('Administrative STOP is preserved')
    output = state_root / stage
    if output.exists():
        raise FileExistsError('Fresh stage only; no silent resume/retry')
    resource = 'cpu' if stage == 'freeze' else hook['resource']
    with resource_locks(root, resource):
        verify_sources(root, plan['source_sha256'])
        output.mkdir(parents=True, exist_ok=False)
        model = None if stage in ('data_check', 'preprocess', 'fit') else _stage_receipt(root, state_root, 'fit', admitted['plan_sha256'])['selected_model']
        common = dict(stage=stage, method=METHOD, plan_sha256=admitted['plan_sha256'],
            source_sha256=plan['source_sha256'], split_manifest=plan['split_manifest'],
            selected_model=model, raw_TEST_files_opened_by_runner=0)
        if stage == 'freeze':
            receipt = dict(common, complete=True, actual_process_returncode=0,
                completion_files=[], all_cases_actual_closed=True,
                numeric_task_success_gate_invented=False, TEST_used=False)
            write_new(output / 'receipt.json', receipt)
            return receipt
        if stage == 'test':
            claim = admitted['split_path'].parent / 'current_network16_TEST_consumption.json'
            write_new(claim, dict(common, one_attempt=True, created_before_first_TEST_process=True))
        jobs = [r for r in admitted['rows'] if r['split'] == stage] if stage in ('validation', 'test') else [None]
        completed, ordered, fitted_metadata = [], [], {}
        for index, row in enumerate(jobs):
            if any(p.exists() for p in stops):
                raise PipelineBlocked('Administrative STOP before next process')
            verify_sources(root, plan['source_sha256'])
            verify_sources(root, hook['source_sha256'])
            if sha(plan_path) != admitted['plan_sha256']:
                raise PipelineBlocked('Plan changed during the queue')
            case = output / f'job{index:04d}'
            case.mkdir(exist_ok=False)
            # Owner/log files live in case; the actual adapter receives a fresh
            # child path, so its own immutable-output guard is not pre-broken.
            artifact = case / 'artifact'
            replacements = dict(output=str(artifact), owner_output=str(case), split_manifest=str(admitted['split_path']),
                preprocessed_output=str(state_root / 'preprocess/job0000/artifact'),
                preprocess_completion=str(state_root / 'preprocess/job0000/artifact/completion.json'),
                model='' if model is None else str(checked_pointer(root, model)),
                model_sha256='' if model is None else model['sha256'],
                plan_sha256=admitted['plan_sha256'],
                task='' if row is None else row['task'],
                raw_episode_id='' if row is None else row['raw_episode_id'],
                source_parent_group='' if row is None else row['source_parent_group'],
                source_row_sha256='' if row is None else digest(row),
                split=stage, method=METHOD)
            argv = [token.format_map(replacements) for token in hook['argv']]
            code, reason = _run_command(root, argv, case, hook['timeout_seconds'], stops)
            if type(code) is not int or code != 0 or reason is not None:
                raise PipelineBlocked('Infrastructure stop; preserve failed stage and do not retry')
            completion = load(artifact / 'completion.json')
            if (completion.get('complete') is not True or completion.get('method') != METHOD
                    or completion.get('stage') != stage or completion.get('source_unchanged') is not True):
                raise PipelineBlocked('Actual hook completion/identity/source evidence required')
            if row is not None:
                if (completion.get('task') != row['task'] or completion.get('source_row_sha256') != digest(row)
                        or completion.get('native_closed') is not True
                        or completion.get('native_success_source') != 'task.success()'
                        or type(completion.get('native_ever_success')) is not bool
                        or type(completion.get('native_final_success')) is not bool
                        or completion.get('model') != model):
                    raise PipelineBlocked('Closed current-network Native task result required; task False is retained')
                ordered.append(digest(row))
            if stage == 'fit':
                model = completion.get('selected_model')
                checked_pointer(root, model)
                if (not isinstance(completion.get('fitted_component'), str)
                        or type(completion.get('current_network_end_to_end_ready')) is not bool):
                    raise PipelineBlocked('Fit must disclose its actual component and whole-network readiness')
                fitted_metadata = {k: completion[k] for k in
                    ('fitted_component', 'current_network_end_to_end_ready')}
            completed.append(pointer(artifact / 'completion.json'))
        verify_sources(root, plan['source_sha256'])
        verify_sources(root, hook['source_sha256'])
        receipt = dict(common, complete=True, actual_process_returncode=0, selected_model=model,
            ordered_source_rows=ordered, completion_files=completed,
            all_cases_actual_closed=stage in ('validation', 'test'), cases=len(completed),
            task_failure_does_not_stop_or_replace_cases=True, **fitted_metadata)
        write_new(output / 'receipt.json', receipt)
        return receipt
