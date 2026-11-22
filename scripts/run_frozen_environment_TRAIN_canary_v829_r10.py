"""Three isolated original TRAIN canaries for the restored file environment.

Default invocation runs CPU preflight only. --execute-native requires those
unchanged successful CPU receipts, and never retries an existing native leaf.
This is environment validation, not model training or formal VAL/TEST evidence.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME = ROOT / 'runtime/framework/frozen_environment_TRAIN_canary_v829'
OUTPUT = ROOT / 'artifacts/framework/multitask_r10/frozen_environment_TRAIN_canary_v829/development_demo_only'
INVENTORY = ROOT / 'runtime/framework/train_executor_failure_inventory_v790.json'
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
LOCKS = ('runtime/framework/ordered_resume_VAL_TEST_v775.lock',
         'runtime/framework/fixed_TRAIN104_geometry_native.lock')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def stopped(event):
    return event.is_set() or any((ROOT / name).exists() for name in (
        'runtime/framework/multitask_r10_STOP',
        'runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP',
        'runtime/framework/ordered_resume_VAL_TEST_v775/STOP',
        'runtime/framework/formal_VAL_resume_v772/STOP',
        'runtime/framework/frozen_environment_TRAIN_canary_v829/STOP'))


def option(argv, key):
    if argv.count(key) != 1:
        raise ValueError('one exact original TRAIN argument required: ' + key)
    return argv[argv.index(key) + 1]


def selected_rows():
    inventory = json.loads(INVENTORY.read_text())
    rows = [r for r in inventory['cases'] if r['task'] == 'pick_moving_target_on_the_table'
            and r['raw_episode_id'] == 'episode0' and r['method'] in METHODS]
    if len(rows) != 3 or {r['method'] for r in rows} != set(METHODS):
        raise ValueError('exact original same-source three-method TRAIN fixture required')
    result = []
    for method in METHODS:
        row = next(r for r in rows if r['method'] == method)
        old = ROOT / row['case']['path']
        if row['case'] != pointer(old) or row['official_task_success'] is not True:
            raise ValueError('unchanged previously successful TRAIN witness required')
        old_case = json.loads(old.read_text())
        identity = old_case['identity']
        if (old_case.get('official_task_success') is not True or identity.get('split') != 'train'
                or identity.get('method') != method or identity.get('source_raw_episode_id') != 'episode0'
                or identity.get('source_row_sha256') != row['source_row_sha256']):
            raise ValueError('genuine same-method original TRAIN identity required')
        argv = list(row['provenance']['actual_argv'])
        prefix = ['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani' else 'scripts/run_local_sim.sh',
                  'scripts/run_task_executor_smoke_author_outcome_v743_r10.py']
        if argv[:3] != prefix or '--development-demo' not in argv or '--check' in argv:
            raise ValueError('original743 TRAIN path required')
        for key, value in (('--task', 'pick_moving_target_on_the_table'), ('--raw-episode-id', 'episode0'),
                           ('--split', 'train'), ('--method', method),
                           ('--manifest', 'artifacts/framework/protocol/train_manifest.json'),
                           ('--frozen-inputs', 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json')):
            if option(argv, key) != value:
                raise ValueError('TRAIN fixture scope changed: ' + key)
        out = OUTPUT / 'pick_moving_target_on_the_table/episode0' / method
        argv[argv.index('--out') + 1] = str(out.relative_to(ROOT))
        result.append(dict(method=method, argv=argv, out=str(out.relative_to(ROOT)),
                           previous_case=row['case'], source_row_sha256=row['source_row_sha256'],
                           expected_identity=identity))
    return result


def require_CPU_receipt(row, receipt):
    domain = receipt.get('component_domain', {})
    if (receipt.get('complete') is not True or receipt.get('actual_split') != 'train'
            or receipt.get('native_started') is not False or receipt.get('GPU_started') is not False
            or receipt.get('identity') != row['expected_identity']
            or domain.get('purpose') != 'train_qualification' or domain.get('split') != 'train'
            or domain.get('method') != row['method'] or domain.get('raw_episode_id') != 'episode0'
            or domain.get('task') != 'pick_moving_target_on_the_table'
            or row['expected_identity']['config_sha256'] != sha(ROOT / option(row['argv'], '--config'))
            or row['expected_identity']['frozen_inputs_sha256'] != sha(ROOT / option(row['argv'], '--frozen-inputs'))):
        raise ValueError('genuine unchanged same-method CPU-only TRAIN identity required')


def require_CPU_stage(row):
    leaf = RUNTIME / 'CPU' / row['method']
    record = json.loads((leaf / 'terminal.json').read_text())
    launch = json.loads((leaf / 'launch.json').read_text())
    if (record.get('complete') is not True or type(record.get('returncode')) is not int
            or record['returncode'] != 0 or record.get('timed_out') is not False
            or record.get('interrupted') is not False
            or record.get('launch') != pointer(leaf / 'launch.json')
            or record.get('CPU_preflight') != pointer(leaf / 'CPU_preflight.json')
            or launch.get('argv') != row['argv'] + ['--check']
            or launch.get('check_only') is not True
            or launch.get('native_slots') != 0 or launch.get('GPU_slots') != 0):
        raise ValueError('actual clean immutable CPU launch/terminal/receipt required')
    require_CPU_receipt(row, json.loads((leaf / 'CPU_preflight.json').read_text()))


def run_child(row, check, event):
    from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
    leaf = RUNTIME / ('CPU' if check else 'native') / row['method']
    leaf.mkdir(parents=True, exist_ok=False)
    argv = row['argv'] + (['--check'] if check else [])
    process = None
    try:
        with (leaf / 'stdout.log').open('xb') as out, (leaf / 'stderr.log').open('xb') as err:
            if stopped(event):
                raise InterruptedError('STOP before child launch')
            process = subprocess.Popen(argv, cwd=ROOT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                stdout=out, stderr=err, start_new_session=True)
            start = time.monotonic()
            ticks = int(Path('/proc', str(process.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
            atomic(leaf / 'launch.json', dict(created_utc=utc(), argv=argv,
                launcher_pid=process.pid, launcher_start_ticks=ticks, owner_pid=os.getpid(),
                check_only=check, native_slots=0 if check else 1,
                GPU_slots=0 if check else int(row['method'] == 'physmani')))
            timeout = interrupted = False
            while process.poll() is None:
                timeout = time.monotonic() - start > 1800
                interrupted = stopped(event)
                if timeout or interrupted:
                    interrupt_owned_tree(process)
                    break
                event.wait(.5)
            code = process.wait()
            interrupted = interrupted or stopped(event)
        record = dict(complete=code == 0 and not timeout and not interrupted,
                      returncode=code, timed_out=timeout, interrupted=interrupted,
                      finished_utc=utc(), automatic_retry=False, launch=pointer(leaf / 'launch.json'))
        if record['complete'] and check:
            receipt = json.loads((leaf / 'stdout.log').read_text())
            require_CPU_receipt(row, receipt)
            atomic(leaf / 'CPU_preflight.json', receipt)
            record['CPU_preflight'] = pointer(leaf / 'CPU_preflight.json')
        elif record['complete']:
            case_path = ROOT / row['out'] / 'case.json'
            case = json.loads(case_path.read_text())
            if (case.get('identity') != row['expected_identity'] or case.get('cleanup_completed') is not True
                    or case.get('component_domain', {}).get('purpose') != 'train_qualification'
                    or type(case.get('official_task_success')) is not bool):
                raise ValueError('complete original TRAIN case/identity/cleanup required')
            record.update(case=pointer(case_path), native_success=case['official_task_success'])
        if stopped(event):
            record.update(complete=False, interrupted=True)
        atomic(leaf / 'terminal.json', record)
        if not record['complete']:
            raise RuntimeError('environment canary failed; preserve evidence, do not retry: ' + row['method'])
        return record
    except BaseException as error:
        if process is not None and process.poll() is None:
            interrupt_owned_tree(process)
        if not (leaf / 'terminal.json').exists():
            atomic(leaf / 'terminal.json', dict(complete=False, error=repr(error),
                interrupted=stopped(event), finished_utc=utc(), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-manifest', required=True)
    parser.add_argument('--environment-manifest-sha256', required=True)
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args()
    from scripts.run_frozen_system_environment_v820_r10 import require_environment
    from scripts.run_environment_resumed_formal_VAL_owner_v823_r10 import process_visibility
    event = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: event.set())
    if stopped(event):
        raise InterruptedError('user STOP has priority')
    manifest = (ROOT / args.environment_manifest).resolve()
    environment = require_environment(manifest, args.environment_manifest_sha256)
    rows = selected_rows()
    inputs = {name: pointer(ROOT / name) for name in (
        str(Path(__file__).relative_to(ROOT)), 'runtime/framework/train_executor_failure_inventory_v790.json',
        'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py',
        'scripts/run_frozen_system_environment_v820_r10.py',
        'scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
        'scripts/run_local_sim.sh', 'scripts/run_aligned_gpu.sh')}
    plan = dict(schema='r10_environment_TRAIN_canary_plan_v829', rows=rows, inputs=inputs,
                environment_manifest=pointer(manifest),
                training_weights_updated=False, formal_VAL_TEST_evidence=False)
    with ExitStack() as stack:
        for name in LOCKS:
            stream = stack.enter_context((ROOT / name).open('a+'))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = process_visibility()
        if (not inventory['host_visibility_available'] or inventory['live_project_native_processes']
                or inventory['same_uid_inventory_incomplete'] or stopped(event)):
            raise RuntimeError('quiescent native resources and complete host inventory required')
        if args.execute_native:
            if json.loads((RUNTIME / 'plan.json').read_text()) != plan:
                raise ValueError('CPU source/plan/environment changed; preserve previous canary')
            for row in rows:
                require_CPU_stage(row)
                if (RUNTIME / 'native' / row['method']).exists() or (ROOT / row['out']).exists():
                    raise FileExistsError('native canary attempt already exists; no implicit retry')
        else:
            RUNTIME.mkdir(parents=True, exist_ok=False)
            atomic(RUNTIME / 'plan.json', plan)
        stage = 'native' if args.execute_native else 'CPU'
        atomic(RUNTIME / (stage + '_owner_launch.json'), dict(pid=os.getpid(), created_utc=utc(),
            source=pointer(Path(__file__)), environment=environment, resource_inventory=inventory,
            native_opt_in=args.execute_native, training_weights_updated=False))
        results = {}
        for row in rows:
            if stopped(event):
                raise InterruptedError('user STOP has priority')
            results[row['method']] = run_child(row, not args.execute_native, event)
            atomic(RUNTIME / (stage + '_progress.json'), dict(updated_utc=utc(), results=results))
        require_environment(manifest, args.environment_manifest_sha256)
        if stopped(event):
            raise InterruptedError('STOP before canary stage completion')
        atomic(RUNTIME / (stage + '_terminal.json'), dict(complete=True, finished_utc=utc(),
            scope='environment-only original TRAIN same-source fixture', results=results,
            native_started=args.execute_native, training_weights_updated=False,
            formal_VAL_TEST_evidence=False, automatic_retry=False))
        print(json.dumps(dict(complete=True, stage=stage, methods=results), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
