"""Explicit bootstrap for scoped frozen-lookup batching; inactive without a lock.

The original751 formal driver or743 TRAIN worker really executes. Its source,
scientific gates, full verify_inputs calls and physical consumer are retained.
An additional receipt discloses this bootstrap; it is never a751 entry claim.
"""
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import time
from threading import RLock, get_ident, local

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'r10_scoped_frozen_lookup_bootstrap_v901'
LOCK_SCHEMA = 'r10_scoped_frozen_lookup_bootstrap_inputs_v901'
RECEIPT_SCHEMA = 'r10_scoped_frozen_lookup_bootstrap_receipt_v901'
_CALL_LOCK = RLock()
_CALL_STATE = local()
SOURCES = (
    'scripts/run_batched_formal_driver_v901_r10.py',
    'our_mani/multitask/frozen_lookup_batch_v900.py',
    'scripts/run_formal_case_author_outcome_v751_r10.py',
    'our_mani/multitask/formal_suite_admission_v728.py',
    'our_mani/multitask/rml_case_preflight_v742.py',
    'scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
    'scripts/run_task_executor_smoke_author_suite16_v744_r10.py',
)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def point(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError('bootstrap evidence must remain inside project')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha_bytes(path.read_bytes()))


def option(argv, name):
    if argv.count(name) != 1:
        raise ValueError('one forwarded option required: ' + name)
    index = argv.index(name)
    if index + 1 >= len(argv) or argv[index + 1].startswith('--'):
        raise ValueError('forwarded value required: ' + name)
    return argv[index + 1]


def load_control(control_path, lock_path, lock_sha, entry, argv):
    """Read only a small external control lock; original gates check data locks."""
    control_path, lock_path = ((ROOT / name).resolve(strict=True) for name in (control_path, lock_path))
    if not all(p.is_relative_to(ROOT / 'artifacts/framework/multitask_r10') for p in (control_path, lock_path)):
        raise ValueError('isolated project artifact control/lock required')
    payload = lock_path.read_bytes()
    if len(lock_sha) != 64 or sha_bytes(payload) != lock_sha:
        raise ValueError('externally pinned bootstrap lock SHA required')
    lock = json.loads(payload)
    if lock.get('schema') != LOCK_SCHEMA or lock.get('execution_enabled') is not True:
        raise ValueError('explicit enabled bootstrap input lock required')
    files = lock.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('explicit bootstrap input map required')
    canonical = {}
    for name, expected in files.items():
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or str(path) in canonical:
            raise ValueError('unique project control dependency required')
        canonical[str(path)] = expected
    for name in (str(control_path), *(str(ROOT / p) for p in SOURCES)):
        path = Path(name).resolve(strict=True)
        expected = canonical.get(str(path))
        if not isinstance(expected, str) or sha_bytes(path.read_bytes()) != expected:
            raise ValueError('bootstrap source/control changed: ' + str(path))
    control_payload = control_path.read_bytes()
    if sha_bytes(control_payload) != canonical[str(control_path)]:
        raise ValueError('parsed control bytes differ from pinned SHA')
    control = json.loads(control_payload)
    if (control.get('schema') != SCHEMA or control.get('execution_enabled') is not True
            or control.get('entry') != entry or control.get('scientific_policy_changed') is not False
            or control.get('models_or_training_changed') is not False):
        raise ValueError('explicit unchanged-policy bootstrap control required')
    split = option(argv, '--split')
    if entry == 'formal':
        if split != 'test' or '--development-demo' in argv:
            raise ValueError('bootstrap formal continuation is TEST-only')
    elif entry == 'train':
        if split != 'train' or argv.count('--development-demo') != 1:
            raise ValueError('bootstrap resource qualification is TRAIN-only')
        key = [option(argv, k) for k in ('--task', '--raw-episode-id', '--method')]
        if key not in control.get('TRAIN_cases', []):
            raise ValueError('exact frozen original TRAIN resource case required')
    else:
        raise ValueError('explicit formal/train entry required')
    bindings = control.get('argv_bindings')
    required = {'--split', '--manifest', '--frozen-inputs'} if entry == 'train' else {
        '--split', '--manifest', '--suite', '--suite-lock', '--source-inputs', '--validation-release'}
    if not isinstance(bindings, dict) or not required <= set(bindings):
        raise ValueError('complete original execution scope bindings required')
    for name, value in bindings.items():
        if name not in required or option(argv, name) != value:
            raise ValueError('forwarded scope differs from bootstrap control: ' + name)
    return dict(control=point(control_path), lock=point(lock_path), entry=entry,
                original_entry='scripts/run_formal_case_author_outcome_v751_r10.py' if entry == 'formal'
                else 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
                sources={name: point(ROOT / name) for name in SOURCES})


@contextmanager
def scoped_calls(driver, contract, guard, batch_type, phases):
    """Patch only two CPU preparation calls, restoring even after an exception."""
    with _CALL_LOCK:
        if getattr(_CALL_STATE, 'active', False):
            raise RuntimeError('nested bootstrap call scope is not supported')
        _CALL_STATE.active = True
        try:
            with _scoped_calls(driver, contract, guard, batch_type, phases):
                yield
        finally:
            _CALL_STATE.active = False


@contextmanager
def _scoped_calls(driver, contract, guard, batch_type, phases):
    owner_thread = get_ident()
    original_prepare = contract.prepare
    original_bindings = driver.bindings if driver is not None else None

    def invoke(phase, function, args, kwargs):
        started = time.monotonic()
        batch = batch_type(ROOT)
        with batch:
            with batch.scoped_hooks(external=((contract, '_checked'),),
                    pointers=((guard, 'locked_pointer'), (contract, 'locked_pointer')),
                    json_readers=((guard, 'locked_json'), (contract, 'locked_json'))):
                result = function(*args, **kwargs)
        # Successful exit has checked all used inputs and graph boundaries.
        phases.append(dict(phase=phase, wall_seconds=time.monotonic() - started,
                           metrics=dict(batch.metrics), boundary_recheck_complete=True))
        return result

    def prepare(*args, **kwargs):
        if get_ident() != owner_thread:
            return original_prepare(*args, **kwargs)
        return invoke('original742_prepare', original_prepare, args, kwargs)

    def bindings(*args, **kwargs):
        if get_ident() != owner_thread:
            return original_bindings(*args, **kwargs)
        return invoke('original751_bindings', original_bindings, args, kwargs)

    contract.prepare = prepare
    if driver is not None:
        driver.bindings = bindings
    try:
        yield
    finally:
        contract.prepare = original_prepare
        if driver is not None:
            driver.bindings = original_bindings


def run_original(entry, argv, batch_type, phases):
    # Importing this file alone never imports the physical worker or torch.
    from scripts import run_task_executor_smoke_author_outcome_v743_r10 as worker
    from our_mani.multitask import formal_suite_admission_v728 as guard
    from our_mani.multitask.frozen_input_edges_v463 import evaluator_scope, declared_edge_scope
    from scripts import evaluate_multitask_case_r10 as evaluator
    if entry == 'formal':
        from scripts import run_formal_case_author_outcome_v751_r10 as driver
        function = driver.main
    else:
        driver = None
        function = worker.main
    previous = sys.argv
    try:
        sys.argv = [driver.__file__ if driver is not None else worker.__file__, *argv]
        with scoped_calls(driver, worker.contract, guard, batch_type, phases):
            with declared_edge_scope(), evaluator_scope(evaluator):
                function()
    finally:
        sys.argv = previous


def require_phases(entry, phases):
    expected = ['original751_bindings', 'original742_prepare'] if entry == 'formal' else ['original742_prepare']
    if [p.get('phase') for p in phases] != expected:
        raise ValueError('the actual original preparation scopes must complete exactly once')
    for phase in phases:
        metrics = phase.get('metrics', {})
        if (phase.get('boundary_recheck_complete') is not True
                or metrics.get('closed') is not True or metrics.get('complete') is not True
                or type(metrics.get('maps')) is not int or metrics['maps'] < 1
                or type(metrics.get('pointer_checks')) is not int or metrics['pointer_checks'] < 1):
            raise ValueError('complete original-input batch boundary receipt required')


def main():
    outer_argv = list(sys.argv)
    if sys.argv.count('--') != 1:
        raise ValueError('one explicit bootstrap/forwarded argument boundary required')
    boundary = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-control', required=True)
    parser.add_argument('--batch-lock', required=True)
    parser.add_argument('--batch-lock-sha256', required=True)
    parser.add_argument('--entry', choices=('formal', 'train'), required=True)
    args = parser.parse_args(sys.argv[1:boundary])
    argv = sys.argv[boundary + 1:]
    if argv.count('--check') > 1:
        raise ValueError('duplicate CPU-only switch')
    check = '--check' in argv
    if (ROOT / 'runtime/framework/multitask_r10_STOP').exists():
        raise InterruptedError('user STOP before bootstrap preparation')
    proof = load_control(args.batch_control, args.batch_lock, args.batch_lock_sha256, args.entry, argv)
    from our_mani.multitask.frozen_lookup_batch_v900 import LookupBatch
    phases = []
    captured = io.StringIO()
    started = time.monotonic()
    if check:
        with redirect_stdout(captured):
            run_original(args.entry, argv, LookupBatch, phases)
        output = json.loads(captured.getvalue())
    else:
        # Native stdout contains original simulator/RML/WM diagnostics as well
        # as a JSON line. Forward it unchanged instead of parsing it as JSON.
        run_original(args.entry, argv, LookupBatch, phases)
        output = dict(complete=True)
    require_phases(args.entry, phases)
    if load_control(args.batch_control, args.batch_lock, args.batch_lock_sha256, args.entry, argv) != proof:
        raise ValueError('bootstrap source/control changed during execution')
    receipt = dict(schema=RECEIPT_SCHEMA, complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), actual_outer_argv=outer_argv,
        forwarded_original_argv=[proof['original_entry'], *argv], check_only=check,
        phases=phases, wall_seconds=time.monotonic() - started,
        scientific_policy_changed=False, full_verify_inputs_calls_retained=True,
        direct_read_text_JSON_semantics='original742_unchanged_with_batch_exit_rechecks',
        native_entry_claim='original_function_called_under_explicit_v901_bootstrap', **proof)
    if check:
        if output.get('native_started') is not False or output.get('GPU_started') is not False:
            raise ValueError('original CPU proof must not start Native/GPU')
        output['batch_execution'] = receipt
    else:
        out = (ROOT / option(argv, '--out')).resolve(strict=True)
        receipt['case'] = point(out / 'case.json')
        if args.entry == 'formal':
            receipt['original_admission_driver_receipt'] = point(out / 'admission_driver_receipt.json')
        from our_mani.multitask.evaluation import atomic_json
        atomic_json(out / 'batch_driver_receipt.json', receipt)
        output['batch_driver_receipt'] = point(out / 'batch_driver_receipt.json')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
