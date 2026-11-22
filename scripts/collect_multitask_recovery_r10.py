"""Explicit same-case infrastructure recovery; never imported by frozen v11.

The original full roster and collector stay unchanged. This runner owns the
parent command, scopes two hash-bound hooks to one child, and preserves every
historical canonical receipt. Planning and checking start no native runtime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import collect_multitask_training_r10 as collection
from our_mani.multitask import acquisition_lineage as lineage

VERSION = 'r10_explicit_same_case_infrastructure_recovery_runner_v1'
REGISTRATION_SCHEMA = 'r10_explicit_recovery_execution_registration_v1'
RML_PERMISSION_SCHEMA = 'r10_owned_RML_factory_hook_permission_v1'
INPUT_SCHEMA = 'r10_same_case_infrastructure_recovery_input_v1'
HOOK_KEYS = ('development_reset_profile', 'RML_factory_hook', 'recovery_input')
OWNED_SOURCES = (
    'scripts/collect_multitask_recovery_r10.py',
    'scripts/collect_multitask_training_r10.py',
    'scripts/run_local_sim.sh',
    'our_mani/multitask/acquisition_lineage.py',
    'our_mani/multitask/development_reset_recovery.py',
    'our_mani/multitask/rml_path_repair.py',
    'our_mani/multitask/rml_path_recovery.py',
    'our_mani/multitask/train_teacher.py',
    'our_mani/multitask/simulation.py',
)


def require(value, message):
    if not value:
        raise collection.CollectionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path):
    return lineage.pointer(ROOT, path)


def checked(value, files=None):
    return lineage.checked_pointer(ROOT, value, {} if files is None else files)


def source_bound(plan, relative):
    path = (ROOT / relative).resolve()
    require(plan['inputs'].get(str(path)) == collection.sha(path),
            'recovery executable is not bound by actual plan inputs: ' + relative)
    return path


def rml_permission_value():
    from our_mani.multitask.rml_path_recovery import NATIVE_DIAGNOSTIC, recovery_diagnostic_evidence
    recovery_diagnostic_evidence(ROOT)
    names = ('our_mani/multitask/rml_path_repair.py',
             'our_mani/multitask/rml_path_recovery.py', 'our_mani/multitask/train_teacher.py')
    return dict(schema=RML_PERMISSION_SCHEMA,
        factory='our_mani.multitask.rml_path_recovery:recovery_native_path_planner',
        configuration=dict(maximum_virtual_steps=10000),
        physics_step_called_by_factory=False, real_native_step_motion_replaced=False,
        original_native_motion_limits_and_RML_parameters_preserved=True,
        historical_fault_reproduced_in_native_diagnostic=False,
        diagnostic_report=deepcopy(NATIVE_DIAGNOSTIC),
        inputs_sha256={str((ROOT / name).resolve()): collection.sha(ROOT / name) for name in names})


def load_rml_permission(value, plan=None):
    path = checked(value)
    payload = read(path)
    # Exact, explicitly named policy and source fields; no broad numeric guard.
    require(payload == rml_permission_value(), 'owned RML permission scope/source/evidence differs')
    if plan is not None:
        require(plan['inputs'].get(str(path)) == value['sha256'], 'RML permission is not a plan input')
        for name, expected in payload['inputs_sha256'].items():
            require(plan['inputs'].get(name) == expected, 'RML hook dependency is not a plan input')
    return payload


def permission_dependencies(value, result=None, seen=None, *, follow_json=True):
    """Follow only explicit immutable pointers and named source SHA maps."""
    result = {} if result is None else result
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if {'path', 'sha256'} <= set(value):
            path = checked(value)
            name = str(path)
            require(name not in result or result[name] == value['sha256'], 'conflicting permission dependency')
            result[name] = value['sha256']
            if follow_json and path.suffix == '.json' and name not in seen:
                seen.add(name)
                permission_dependencies(read(path), result, seen, follow_json=False)
        for key, item in value.items():
            if key == 'inputs_sha256' and isinstance(item, dict):
                for name, expected in item.items():
                    path = Path(name).resolve()
                    require(collection.sha(path) == expected, 'permission source dependency changed')
                    result[str(path)] = expected
            else:
                permission_dependencies(item, result, seen, follow_json=follow_json)
    elif isinstance(value, list):
        for item in value:
            permission_dependencies(item, result, seen, follow_json=follow_json)
    return result


def prepare_recovery_plan(base_registration, *, physical_permission, rml_permission,
                          recovery_input, protocol_output, plan_output, extra_inputs=()):
    """Freeze the original roster plus explicit hooks; no failure selection or launch.

    The caller supplies actual failed receipt pointers and later creates the
    independent transition declaration, registrations and source archive.
    """
    previous = lineage.load_archived_plan(ROOT, base_registration, {})
    original = previous['plan']
    load_rml_permission(rml_permission)
    from our_mani.multitask.development_reset_recovery import load_permission
    load_permission(physical_permission, root=ROOT, native_repo=original['protocol']['baseline_repo'])
    payload = read(checked(recovery_input))
    require(payload.get('schema') == INPUT_SCHEMA and payload.get('original_plan') == base_registration['plan']
        and payload.get('development_reset_profile') == physical_permission
        and payload.get('RML_factory_hook') == rml_permission
        and payload.get('infrastructure_only') is True,
        'recovery input must bind actual original plan and exact two permissions')
    plan = deepcopy(original)
    plan['created_utc'] = collection.utc()
    plan['protocol'] = deepcopy(original['protocol'])
    plan['protocol'].update(development_reset_profile=deepcopy(physical_permission),
        RML_factory_hook=deepcopy(rml_permission), recovery_input=deepcopy(recovery_input))
    protocol_output = collection.destination(protocol_output)
    plan_output = collection.destination(plan_output)
    require(protocol_output.resolve() != Path(original['protocol_path']).resolve(), 'new protocol path required')
    require(plan_output.resolve() != previous['path'], 'new plan path required')
    # Old non-code inputs retain their immutable SHA. Every actual executable
    # is frozen at its current bytes; the later declaration lists every delta.
    inputs = dict(original['inputs'])
    for name, expected in list(inputs.items()):
        if Path(name).suffix in ('.py', '.sh'):
            inputs[name] = collection.sha(name)
        else:
            require(collection.sha(name) == expected, 'original noncode input changed')
    for relative in OWNED_SOURCES:
        inputs[str((ROOT / relative).resolve())] = collection.sha(ROOT / relative)
    for name in extra_inputs:
        path = Path(name).resolve()
        inputs[str(path)] = collection.sha(path)
    for key in HOOK_KEYS:
        inputs.update(permission_dependencies(plan['protocol'][key]))
    collection.write_json(protocol_output, plan['protocol'])
    inputs[str(protocol_output)] = collection.sha(protocol_output)
    plan['inputs'] = inputs
    plan['protocol_path'] = str(protocol_output)
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = collection.canonical_sha(plan)
    collection.write_json(plan_output, plan)
    collection.verify_plan(plan_output)
    return plan


def verify_execution(plan_path, registration_path, *, output=None, case_ids=None):
    """Check historical archives, current full input closure and same-case failures."""
    registration_path = Path(registration_path).resolve()
    registration_pointer = pointer(registration_path)
    registry = read(registration_path)
    require(set(registry) == {'schema', 'original', 'recovery'}
        and registry['schema'] == REGISTRATION_SCHEMA, 'exact recovery execution registration required')
    files = {}
    previous = lineage.load_archived_plan(ROOT, registry['original'], files)
    current = lineage.load_archived_plan(ROOT, registry['recovery'], files)
    require(current['path'] == Path(plan_path).resolve(), 'actual recovery --plan differs from registration')
    plan = collection.verify_plan(plan_path)
    require(plan == current['plan'], 'live and archived recovery plans differ')
    require(current['collection'] != previous['collection']
        and not current['collection'].is_relative_to(previous['collection'])
        and not previous['collection'].is_relative_to(current['collection']),
        'recovery output must use a separate namespace')
    if output is not None:
        require(Path(output).resolve().is_relative_to(current['collection']),
                'output is outside actual recovery collection')
    for relative in OWNED_SOURCES:
        source_bound(plan, relative)
    for key in HOOK_KEYS:
        path = checked(plan['protocol'].get(key), files)
        require(plan['inputs'].get(str(path)) == collection.sha(path), 'hook permission is not a plan input: ' + key)
    declaration = lineage.verify_recovery_transition(ROOT, previous, current, files)
    allowed = declaration['allowed_recovery_case_ids']
    require(allowed and len(allowed) == len(set(allowed)), 'explicit nonempty failed roster required')
    payload = read(checked(plan['protocol']['recovery_input'], files))
    require(payload.get('schema') == INPUT_SCHEMA and payload.get('infrastructure_only') is True,
            'recovery input is not explicit infrastructure-only metadata')
    require(payload.get('case_ids') == allowed, 'recovery metadata must retain the exact declared fixed case list')
    by_id = {case['case_id']: case for case in plan['cases']}
    require(len(by_id) == len(plan['cases']) and set(allowed) <= set(by_id), 'duplicate or unknown fixed cases')
    failures = {}
    for value in payload['failed_parent_receipts']:
        receipt = read(checked(value, files))
        case_id = receipt.get('case_id')
        require(case_id in allowed, 'failure receipt outside fixed recovery roster')
        path, _ = lineage.verify_attempt_parent(ROOT, value, previous, case_id, files, status='infrastructure_error')
        require(path.is_relative_to(previous['collection'] / 'infrastructure_attempts' / case_id)
            and path.name == 'parent_receipt.json', 'failure receipt outside original actual attempt directory')
        failures.setdefault(case_id, []).append(deepcopy(value))
    require(set(failures) == set(allowed), 'every permitted recovery needs an actual prior infrastructure failure')
    for case_id in allowed:
        original = collection.case_directory(previous['collection'], by_id[case_id]) / 'case.json'
        require(not original.exists(), 'refusing to replace an original canonical completion: ' + case_id)
    if case_ids is not None:
        require(case_ids and len(set(case_ids)) == len(case_ids) and set(case_ids) <= set(allowed),
                'queue requires a unique explicit subset of fixed failed cases')
    load_rml_permission(plan['protocol']['RML_factory_hook'], plan)
    from our_mani.multitask.development_reset_recovery import load_permission
    physical = load_permission(plan['protocol']['development_reset_profile'], root=ROOT,
                               native_repo=plan['protocol']['baseline_repo'])
    # Native reset dependencies must also be covered by the actual new plan.
    for name, expected in physical['native_inputs_sha256'].items():
        path = str((Path(plan['protocol']['baseline_repo']) / name).resolve())
        require(plan['inputs'].get(path) == expected, 'native physical reset source is unbound: ' + path)
    return dict(plan=plan, previous=previous, current=current, allowed=allowed, failures=failures,
                declaration=declaration, registration_pointer=registration_pointer, files=files)


@contextmanager
def scoped_owned_hooks(plan, receipts):
    """Child-only substitutions; both originals are restored even on exception."""
    from our_mani.multitask import simulation, train_teacher
    from our_mani.multitask import rml_path_recovery as adapter
    from our_mani.multitask import rml_path_repair, development_reset_recovery
    NativeRMLHandleFactory = rml_path_repair.NativeRMLHandleFactory
    SharedPhysicalDevelopmentSimulation = development_reset_recovery.SharedPhysicalDevelopmentSimulation
    for module, relative in ((simulation, 'our_mani/multitask/simulation.py'),
            (train_teacher, 'our_mani/multitask/train_teacher.py'),
            (adapter, 'our_mani/multitask/rml_path_recovery.py'),
            (rml_path_repair, 'our_mani/multitask/rml_path_repair.py'),
            (development_reset_recovery, 'our_mani/multitask/development_reset_recovery.py')):
        require(Path(module.__file__).resolve() == source_bound(plan, relative),
                'recovery imported a different executable source: ' + relative)
    require(adapter.native_path_planner is train_teacher.native_path_planner,
            'owned planner must retain the exact original alias; nested/recursive hooks rejected')
    require(SharedPhysicalDevelopmentSimulation.__bases__ == (simulation.AlignedDevelopmentSimulation,),
            'owned physical reset must inherit the actual unmodified base simulation')
    policy = load_rml_permission(plan['protocol']['RML_factory_hook'], plan)
    class CapturedSharedPhysicalSimulation(SharedPhysicalDevelopmentSimulation):
        def __init__(self, *args, **kwargs):
            receipts['simulations'].append(self)
            super().__init__(*args, **kwargs)
    def planner(env):
        factory = NativeRMLHandleFactory(**policy['configuration'])
        original = adapter.recovery_native_path_planner(env, handle_factory=factory, root=ROOT,
                                                        diagnostic_pointer=policy['diagnostic_report'])
        def plan_goal(goal):
            path = original(goal)
            receipts['paths'].append(path)
            return path
        return plan_goal
    with patch.object(simulation, 'AlignedDevelopmentSimulation', CapturedSharedPhysicalSimulation), \
         patch.object(train_teacher, 'native_path_planner', planner):
        yield


def collect_recovery_case(plan_path, case_id, output, guard_report, registration_path):
    context = verify_execution(plan_path, registration_path, output=output, case_ids=[case_id])
    plan = context['plan']
    case = next(case for case in plan['cases'] if case['case_id'] == case_id)
    expected = context['current']['collection'] / 'infrastructure_attempts' / case_id / '0' / 'output'
    require(Path(output).resolve() == expected, 'child output must be its explicit single attempt-0 directory')
    out = collection.case_directory(output, case)
    require(not out.exists(), 'child recovery output already exists; no automatic retry or hidden resume')
    captures = dict(simulations=[], paths=[])
    started = collection.utc()
    failure = None
    record = None
    try:
        with scoped_owned_hooks(plan, captures):
            record = collection.collect_native_case(plan_path, case_id, output, guard_report)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        hooks = dict(version=VERSION, case_id=case_id, plan=pointer(plan_path), plan_sha256=plan['plan_sha256'],
            execution_registration=context['registration_pointer'],
            original_plan=deepcopy(context['previous']['registration']['plan']),
            failed_parent_receipts=context['failures'][case_id],
            actual_runner=pointer(Path(__file__)), started_utc=started, finished_utc=collection.utc(),
            permissions={key: deepcopy(plan['protocol'][key]) for key in HOOK_KEYS},
            reset_recovery_receipts=[deepcopy(getattr(env, 'reset_recovery_receipt', None)) for env in captures['simulations']],
            RML_path_recovery_receipts=[deepcopy(path.recovery_receipt) for path in captures['paths']],
            error=failure, complete=record is not None and failure is None,
            policy_failure_label_assigned=False, historical_v11_outputs_modified=False,
            partial_historical_rows_reconstructed=False, new_attempt_is_original_v11=False)
        hooks['complete'] = bool(hooks['complete']
            and len(hooks['reset_recovery_receipts']) == 1
            and isinstance(hooks['reset_recovery_receipts'][0], dict)
            and hooks['reset_recovery_receipts'][0].get('complete') is True
            and all(row.get('close_complete') is True for row in hooks['RML_path_recovery_receipts']))
        collection.write_json(out / 'recovery_hook_receipt.json', hooks)
    require(hooks['complete'] is True and len(hooks['reset_recovery_receipts']) == 1
        and hooks['reset_recovery_receipts'][0].get('complete') is True,
        'completed recovery requires one measured complete physical reset receipt')
    require(all(row.get('close_complete') is True for row in hooks['RML_path_recovery_receipts']),
            'completed recovery contains an unclosed RML path')
    # Keep the original collector's raw new-plan receipt byte-for-byte, then
    # write our independently identified wrapper receipt, never a v11 receipt.
    base_path = out / 'base_collector_receipt.json'
    (out / 'case.json').rename(base_path)
    record = deepcopy(record)
    record['recovery_execution'] = dict(version=VERSION, hook_receipt=pointer(out / 'recovery_hook_receipt.json'),
        base_collector_receipt=pointer(base_path), execution_registration=context['registration_pointer'])
    for path in (base_path, out / 'recovery_hook_receipt.json'):
        record['files_sha256'][str(path.relative_to(ROOT))] = collection.sha(path)
    collection.write_json(out / 'case.json', record)
    validate_recovery_result(out / 'case.json', context, case)
    return record


def validate_recovery_result(path, context, case):
    record = collection.validate_case_result(path, context['plan'], case)
    own = record.get('recovery_execution', {})
    require(own.get('version') == VERSION
        and own.get('execution_registration') == context['registration_pointer'],
        'completed recovery lacks its actual explicit execution registration')
    hooks = read(checked(own['hook_receipt']))
    base = read(checked(own['base_collector_receipt']))
    require(hooks.get('version') == VERSION and hooks.get('complete') is True and hooks.get('error') is None
        and hooks.get('case_id') == case['case_id'] and hooks.get('plan') == context['current']['registration']['plan']
        and hooks.get('execution_registration') == context['registration_pointer']
        and hooks.get('failed_parent_receipts') == context['failures'][case['case_id']]
        and hooks.get('actual_runner') == pointer(Path(__file__))
        and hooks.get('original_plan') == context['previous']['registration']['plan']
        and hooks.get('plan_sha256') == context['plan']['plan_sha256']
        and hooks.get('permissions') == {key: context['plan']['protocol'][key] for key in HOOK_KEYS}
        and hooks.get('policy_failure_label_assigned') is False
        and hooks.get('historical_v11_outputs_modified') is False
        and hooks.get('partial_historical_rows_reconstructed') is False
        and hooks.get('new_attempt_is_original_v11') is False, 'raw recovery hook identity differs')
    require(len(hooks.get('reset_recovery_receipts', [])) == 1
        and hooks['reset_recovery_receipts'][0].get('complete') is True
        and all(row.get('close_complete') is True for row in hooks.get('RML_path_recovery_receipts', [])),
        'raw recovery hook reset or RML cleanup is incomplete')
    stripped = {key: value for key, value in record.items() if key not in ('recovery_execution', 'parent_process_receipt')}
    stripped['files_sha256'] = {key: value for key, value in stripped['files_sha256'].items()
        if key not in (own['hook_receipt']['path'], own['base_collector_receipt']['path'])}
    require(stripped == base, 'wrapper changed the base collector output or labels')
    for key in ('hook_receipt', 'base_collector_receipt'):
        require(record['files_sha256'].get(own[key]['path']) == own[key]['sha256'],
                'wrapper proof is missing from its output SHA manifest')
    if 'parent_process_receipt' in record:
        parent = record['parent_process_receipt']
        child = read(ROOT / parent['child_case_path'])
        require(child == {key: value for key, value in record.items() if key != 'parent_process_receipt'},
                'canonical recovery changes its preserved raw child')
        attempt_output = Path(lineage.command_argument(parent['command'], '--output')).resolve()
        parent_receipt = read(attempt_output.parent / 'parent_receipt.json')
        require(parent_receipt.get('status') == 'completed' and parent_receipt.get('process_id') == parent['process_id']
            and parent_receipt.get('command') == parent['command']
            and parent_receipt.get('execution_registration') == context['registration_pointer'],
            'recovery canonical lacks its actual completed parent receipt')
    return record


def worker_command(plan_path, case_id, output, guard_report, registration_path):
    return ['bash', str(ROOT / 'scripts/run_local_sim.sh'), str(Path(__file__).resolve()), '--collect-case',
        '--plan', str(Path(plan_path).resolve()), '--case-id', case_id, '--output', str(Path(output).resolve()),
        '--guard-report', str(Path(guard_report).resolve()),
        '--execution-registration', str(Path(registration_path).resolve())]


def run_recovery_queue(plan_path, output, guard_report, registration_path, *, case_ids,
                       workers=2, case_timeout=900):
    context = verify_execution(plan_path, registration_path, output=output, case_ids=case_ids)
    require(Path(output).resolve() == context['current']['collection'], 'parent output must equal registered recovery namespace')
    require(type(workers) is int and 1 <= workers <= 8, 'CPU worker limit is 1..8')
    require(type(case_timeout) is int and case_timeout > 0, 'positive wall timeout required')
    require(context['plan']['protocol'].get('draft') is not True
        and context['plan']['protocol'].get('native_execution_authorized') is not False,
        'draft recovery cannot launch native')
    gate = collection.require_r09_release(guard_report)
    output = collection.destination(output)
    selected = [case for case in context['plan']['cases'] if case['case_id'] in set(case_ids)]
    # Fail the whole submission before starting any child if any selected case
    # has an unresolved prior recovery attempt. Explicit future plans may retry.
    for case in selected:
        canonical = collection.case_directory(output, case) / 'case.json'
        if canonical.exists():
            validate_recovery_result(canonical, context, case)
        else:
            require(not (output / 'infrastructure_attempts' / case['case_id']).exists(),
                    'no automatic recovery retry; existing attempt requires a new explicit recovery plan')
    def one(case):
        out = collection.case_directory(output, case)
        canonical = out / 'case.json'
        if canonical.exists():
            return dict(case_id=case['case_id'], status='resumed_verified',
                        process_id=validate_recovery_result(canonical, context, case)['process_id'])
        if collection.COLLECTION_STOP.exists():
            return dict(case_id=case['case_id'], status='cancelled_stop_marker', policy_failure_label_assigned=False)
        attempt_path = collection.destination(output / 'infrastructure_attempts' / case['case_id'] / '0')
        attempt_path.mkdir(parents=True, exist_ok=False)
        attempt_output = attempt_path / 'output'
        command = worker_command(plan_path, case['case_id'], attempt_output, guard_report, registration_path)
        started, status, detail, code, pid = collection.utc(), 'infrastructure_error', '', None, None
        try:
            with (attempt_path / 'native.log').open('wb') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                             MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
                pid = process.pid
                try:
                    code = process.wait(timeout=case_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(pid, signal.SIGKILL)
                        process.wait()
                    detail = 'native case wall timeout; preserved evidence, no policy outcome assigned'
            if code == 0:
                # Recheck inputs and old canonical absence before promotion.
                fresh = verify_execution(plan_path, registration_path, output=output, case_ids=[case['case_id']])
                child = collection.case_directory(attempt_output, case) / 'case.json'
                record = deepcopy(validate_recovery_result(child, fresh, case))
                # xvfb-run owns the Popen PID; its Python child reports its own
                # PID. Preserve both identities rather than rewriting either.
                require(type(record.get('process_id')) is int and record['process_id'] > 0,
                        'actual native Python child must report its own positive process ID')
                launcher = ROOT / 'scripts/run_local_sim.sh'
                record['parent_process_receipt'] = dict(launcher=str(launcher.relative_to(ROOT)),
                    launcher_sha256=collection.sha(launcher), process_id=pid, command=command,
                    child_case_path=str(child.relative_to(ROOT)), child_case_sha256=collection.sha(child))
                collection.write_json(canonical, record)
                status = 'completed'
        except Exception as error:
            detail = ''.join(traceback.format_exception(error))
        receipt = dict(version=VERSION, case_id=case['case_id'], status=status, detail=detail,
            returncode=code, process_id=pid, started_utc=started, finished_utc=collection.utc(), command=command,
            attempt=0, policy_failure_label_assigned=False, gate=gate,
            execution_registration=context['registration_pointer'], actual_plan=pointer(plan_path),
            prior_infrastructure_receipts=context['failures'][case['case_id']], automatic_retries=False)
        collection.write_json(attempt_path / 'parent_receipt.json', receipt)
        return receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = [future.result() for future in as_completed([pool.submit(one, case) for case in selected])]
    merged = collection.merge_completed(plan_path, output)
    collection.write_json(output / 'queue_report.json', dict(version=VERSION, created_utc=collection.utc(),
        receipts=receipts, workers=workers, expected=len(context['plan']['cases']), selected_case_ids=list(case_ids),
        selected_complete=all(row['status'] in ('completed', 'resumed_verified') for row in receipts),
        complete=merged['complete'], recovery_component_only=True,
        composite_acquisition_coverage_not_claimed=True, automatic_retries=False, GPU_used=False,
        historical_v11_outputs_modified=False, execution_registration=context['registration_pointer']), overwrite=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-rml-permission', action='store_true')
    action.add_argument('--prepare-plan', action='store_true')
    action.add_argument('--check', action='store_true')
    action.add_argument('--collect-case', action='store_true')
    action.add_argument('--run-queue', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--protocol-output', type=Path)
    parser.add_argument('--base-registration', type=Path, help='original historical registration JSON')
    parser.add_argument('--physical-permission', type=Path)
    parser.add_argument('--rml-permission', type=Path)
    parser.add_argument('--recovery-input', type=Path)
    parser.add_argument('--extra-input', type=Path, action='append', default=[])
    parser.add_argument('--execution-registration', type=Path)
    parser.add_argument('--case-id')
    parser.add_argument('--queue-case-id', action='append')
    parser.add_argument('--guard-report', type=Path, default=ROOT / 'artifacts/framework/aligned_r09/final_report.json')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--case-timeout', type=int, default=900)
    args = parser.parse_args()
    def need(*names):
        for name in names:
            if getattr(args, name) is None:
                parser.error('this action requires --' + name.replace('_', '-'))
    if args.prepare_rml_permission:
        need('output')
        collection.write_json(args.output, rml_permission_value())
        value = dict(permission=pointer(args.output), native_started=False)
    elif args.prepare_plan:
        need('plan', 'protocol_output', 'base_registration', 'physical_permission', 'rml_permission', 'recovery_input')
        plan = prepare_recovery_plan(read(args.base_registration), physical_permission=pointer(args.physical_permission),
            rml_permission=pointer(args.rml_permission), recovery_input=pointer(args.recovery_input),
            protocol_output=args.protocol_output, plan_output=args.plan, extra_inputs=args.extra_input)
        value = dict(plan=pointer(args.plan), cases=len(plan['cases']), native_started=False)
    else:
        need('plan', 'execution_registration')
        if args.check:
            context = verify_execution(args.plan, args.execution_registration, output=args.output, case_ids=args.queue_case_id)
            value = dict(plan_sha256=context['plan']['plan_sha256'], fixed_failed_cases=context['allowed'], native_started=False)
        elif args.collect_case:
            need('case_id', 'output')
            record = collect_recovery_case(args.plan, args.case_id, args.output, args.guard_report, args.execution_registration)
            value = dict(case_id=record['case_id'], status=record['status'])
        else:
            need('output', 'queue_case_id')
            merged = run_recovery_queue(args.plan, args.output, args.guard_report, args.execution_registration,
                case_ids=args.queue_case_id, workers=args.workers, case_timeout=args.case_timeout)
            value = dict(completed=merged['completed'], expected=merged['expected'], recovery_component_only=True)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
