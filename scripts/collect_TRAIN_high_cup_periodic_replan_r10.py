"""Separate fixed all100 high-cup periodic10 actual-feedback TRAIN candidate runner.

Preparation is CPU-only. Actual collection requires a fresh execution-ready
registration. Every selected case runs once in a new native process; planning
failure is preserved, while an infrastructure failure stops further dispatch.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import collect_multitask_recovery_r10 as recovery
from our_mani.multitask import periodic_acquisition as audit
from our_mani.multitask import teacher_supplement
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
from our_mani.multitask.train_teacher_periodic_replan import PeriodicReplanningConfig
from our_mani.multitask.train_teacher_periodic_replan_hook import scoped_periodic_replanning

collection = recovery.collection; lineage = recovery.lineage
require = audit.require; read = lineage.read
pointer = lambda p: lineage.pointer(ROOT, p)


def prepare(proposal_path, shared_registration_path, setup, output):
    setup = collection.destination(setup); output = collection.destination(output)
    require(not setup.exists() and not output.exists(), 'fresh independent periodic10 setup/output required')
    proposal_path = lineage.local(ROOT, proposal_path); proposal = read(proposal_path)
    require(proposal.get('schema') == audit.PROPOSAL_VERSION and proposal.get('proposal_only') is True,
            'fixed all100 periodic10 proposal required')
    primary = read(lineage.checked_pointer(ROOT, proposal['original_plan'], {}))
    cases = audit.expected_cases(primary)
    require([r['proposed_case'] for r in proposal['proposed_cases']] == cases,
            'proposal is not exact all100 original-parent derivation')
    shared_pointer = pointer(shared_registration_path)
    shared = teacher_supplement.verify_registration(ROOT, shared_pointer, require_ready=True)
    require(shared['registry']['primary_plan'] == proposal['original_plan'], 'shared initialization belongs to another primary')
    setup.mkdir(parents=True)
    protocol = deepcopy(shared['plan']['protocol'])
    protocol_path = setup / 'protocol.json'; collection.write_json(protocol_path, protocol)
    plan = deepcopy(shared['plan'])
    for key in ('auxiliary_teacher_scope', 'supplement_proposal', 'auxiliary_collection'):
        plan.pop(key, None)
    plan.update(created_utc=collection.utc(), cases=cases, protocol=protocol,
                protocol_path=str(protocol_path), augmentation_budget=deepcopy(audit.BUDGET),
                periodic_replanning_augmentation_scope=audit.PLAN_SCOPE,
                periodic_replanning_proposal=pointer(proposal_path),
                periodic_replanning_collection=str(output.relative_to(ROOT)),
                counted_planning_requery_configuration=dict(wait_ticks=10, maximum_requeries=20, dt=.05),
                periodic_feedback_replanning_configuration=dict(feedback_ticks=10, maximum_periodic_requeries=20, dt=.05),
                periodic_replanning_raw_audit_version=audit.RAW_AUDIT_VERSION,
                primary_K1_roster_unmodified=True, primary_Context_hold_view_unmodified=True,
                native_execution_ready=True)
    plan['teacher_command_semantics'] = (
        'Separate typed TRAIN periodic10 post-query actual-feedback policy: active successful arm plans '
        'are closed and retained censored before fresh original teacher query; failed plans retain '
        'the original counted10 wait; actual gripper actuation and release settling finish uninterrupted. '
        'The shared protocol legacy committed-policy description is superseded only by this explicit '
        'TRAIN query schedule; physics, camera, motor, gripper and source seed rules remain unchanged.')
    plan['periodic_teacher_policy_is_exact_author_schedule'] = False
    plan['replica2_is_exact_r1_scene_seed_pair'] = False
    inputs = dict(shared['plan']['inputs'])
    for name, checksum in list(inputs.items()):
        if Path(name).suffix in ('.py', '.sh'):
            inputs[name] = collection.sha(name)
        else:
            require(collection.sha(name) == checksum, 'shared immutable noncode input changed')
    additions = [proposal_path, protocol_path, lineage.local(ROOT, shared_registration_path),
                 ROOT / 'artifacts/framework/protocol/train_manifest.json',
                 ROOT / 'artifacts/framework/protocol/validation_manifest.json']
    additions.extend(ROOT / name for name in audit.OWNED_SOURCES)
    additions.extend(ROOT / name for name in ('tests/test_multitask_teacher_periodic_replan.py',
        'tests/test_multitask_teacher_periodic_replan_hook.py',
        'tests/test_multitask_periodic_acquisition.py'))
    for path in additions:
        inputs[str(path.resolve())] = collection.sha(path)
    for key in ('development_reset_profile', 'RML_factory_hook'):
        inputs.update(recovery.permission_dependencies(protocol[key]))
    from our_mani.multitask.development_reset_recovery import load_permission
    physical = load_permission(protocol['development_reset_profile'], root=ROOT, native_repo=protocol['baseline_repo'])
    for name, checksum in physical['native_inputs_sha256'].items():
        path = (Path(protocol['baseline_repo']) / name).resolve()
        require(collection.sha(path) == checksum, 'shared native physical executable changed')
        inputs[str(path)] = checksum
    plan['inputs'] = inputs; plan.pop('plan_sha256', None)
    plan['plan_sha256'] = collection.canonical_sha(plan)
    plan_path = setup / 'plan.json'; collection.write_json(plan_path, plan); collection.verify_plan(plan_path)
    archives = []
    for name, checksum in sorted(inputs.items()):
        source = Path(name)
        if source.suffix not in ('.py', '.sh'):
            continue
        require(collection.sha(source) == checksum, 'source changed before archival')
        target = setup / 'sources' / checksum / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        require(collection.sha(target) == checksum, 'archived new source differs')
        archives.append(dict(original_path=str(source.resolve()), **pointer(target)))
    collection.verify_plan(plan_path)
    snapshot = setup / 'source_snapshot.json'
    collection.write_json(snapshot, dict(version=lineage.SOURCE_SCHEMA,
        plan=dict(**pointer(plan_path), internal_sha256=plan['plan_sha256']), source_count=len(archives),
        entries=archives, all_plan_inputs_verified_before_and_after=True,
        source_files_modified=False, case_records_or_labels_modified=False))
    registry = dict(schema=audit.REGISTRATION, proposal=pointer(proposal_path),
        shared_profile_reference_registration=shared_pointer,
        execution=dict(plan=pointer(plan_path), source_snapshot=pointer(snapshot), collection=str(output.relative_to(ROOT))),
        execution_ready=False, fixed_budget=deepcopy(audit.BUDGET))
    registry_path = setup / 'execution_registration_candidate.json'; collection.write_json(registry_path, registry)
    context = audit.verify_registration(ROOT, pointer(registry_path), live=True)
    report = dict(schema='r10_fixed100_periodic_feedback_candidate_preparation_v1',
        plan=pointer(plan_path), execution_registration=pointer(registry_path), source_snapshot=pointer(snapshot),
        first_fixed_case_id=cases[0]['case_id'], fixed_cases=100, fixed_parents=100,
        execution_ready=False, native_or_GPU_started=False, production_fitting_adapter_ready=False,
        typed_raw_auditor_ready=True, prior3000_and_Context_view_changed=False,
        actual_source_archive_count=len(context['current']['sources']))
    collection.write_json(setup / 'preparation_report.json', report)
    return report


def approve_registration(candidate, destination):
    context = audit.verify_registration(ROOT, pointer(candidate), live=True)
    target = collection.destination(destination)
    require(context['registry']['execution_ready'] is False and not target.exists()
            and not context['current']['collection'].exists(), 'fresh approval before native output required')
    value = deepcopy(context['registry']); value['execution_ready'] = True
    collection.write_json(target, value)
    audit.verify_registration(ROOT, pointer(target), live=True, require_ready=True)
    return dict(execution_registration=pointer(target), native_or_GPU_started=False)


def collect_case(plan_path, case_id, output, guard_report, registration_path):
    context = audit.verify_registration(ROOT, pointer(registration_path), live=True, require_ready=True)
    require(context['current']['path'] == Path(plan_path).resolve() and case_id in context['expected_cases'],
            'new actual child plan/case differs')
    expected = context['current']['collection'] / 'infrastructure_attempts' / case_id / '0/output'
    require(Path(output).resolve() == expected, 'new raw child needs its exact attempt0 output')
    case = context['expected_cases'][case_id]; out = collection.case_directory(output, case)
    require(not out.exists(), 'no hidden native child resume or retry')
    captures = dict(simulations=[], paths=[]); followers = []
    started = collection.utc(); record = None; failure = None; factory_receipt = None
    try:
        with recovery.scoped_owned_hooks(context['plan'], captures), scoped_periodic_replanning(case,
                expected_teacher_source_sha256=context['plan']['inputs'][str((ROOT / 'our_mani/multitask/train_teacher.py').resolve())],
                failure_configuration=PlanningRequeryConfig(**context['plan']['counted_planning_requery_configuration']),
                periodic_configuration=PeriodicReplanningConfig(**context['plan']['periodic_feedback_replanning_configuration']),
                captures=followers) as factory_receipt:
            record = collection.collect_native_case(plan_path, case_id, output, guard_report)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        hooks = dict(version=audit.RUNNER_VERSION, case_id=case_id,
            plan=context['current']['registration']['plan'], plan_sha256=context['plan']['plan_sha256'],
            execution_registration=context['registry_pointer'], actual_runner=pointer(Path(__file__)),
            started_utc=started, finished_utc=collection.utc(),
            permissions={key: deepcopy(context['plan']['protocol'][key]) for key in ('development_reset_profile', 'RML_factory_hook')},
            reset_recovery_receipts=[deepcopy(getattr(env, 'reset_recovery_receipt', None)) for env in captures['simulations']],
            RML_path_recovery_receipts=[deepcopy(path.recovery_receipt) for path in captures['paths']],
            periodic_replanning_factory=deepcopy(factory_receipt), error=failure, first_execution=True,
            is_infrastructure_recovery=False, outcome_selected=False, automatic_retry=False,
            policy_failure_label_assigned=False, primary_outputs_modified=False,
            complete=bool(record is not None and failure is None and factory_receipt is not None
                and factory_receipt.get('complete') is True and len(captures['simulations']) == 1
                and isinstance(getattr(captures['simulations'][0], 'reset_recovery_receipt', None), dict)
                and captures['simulations'][0].reset_recovery_receipt.get('complete') is True
                and all(path.recovery_receipt.get('close_complete') is True for path in captures['paths'])))
        collection.write_json(out / 'periodic_replanning_hook_receipt.json', hooks)
    require(hooks['complete'], 'new raw periodic/reset/RML witnesses did not complete')
    base = out / 'base_collector_receipt.json'; (out / 'case.json').rename(base)
    record = deepcopy(record)
    record['periodic_replanning_execution'] = dict(version=audit.RUNNER_VERSION,
        hook_receipt=pointer(out / 'periodic_replanning_hook_receipt.json'), base_collector_receipt=pointer(base),
        execution_registration=context['registry_pointer'])
    for path in (base, out / 'periodic_replanning_hook_receipt.json'):
        record['files_sha256'][str(path.relative_to(ROOT))] = collection.sha(path)
    collection.write_json(out / 'case.json', record)
    audit.validate_result(ROOT, out / 'case.json', context, case)
    return record


def worker_command(plan, cid, output, guard, registration):
    return ['bash', str(ROOT / 'scripts/run_local_sim.sh'), str(Path(__file__).resolve()),
        '--collect-case', '--plan', str(Path(plan).resolve()), '--case-id', cid,
        '--output', str(Path(output).resolve()), '--guard-report', str(Path(guard).resolve()),
        '--execution-registration', str(Path(registration).resolve())]


def run_queue(plan_path, output, guard_report, registration_path, *, case_ids=None, workers=1, case_timeout=900):
    context = audit.verify_registration(ROOT, pointer(registration_path), live=True, require_ready=True)
    require(Path(plan_path).resolve() == context['current']['path']
            and Path(output).resolve() == context['current']['collection'], 'new parent plan/output differs')
    require(type(workers) is int and 1 <= workers <= 8 and type(case_timeout) is int and case_timeout > 0,
            'explicit CPU workers1..8 and positive wall timeout required')
    known = context['expected_cases']; case_ids = list(known) if case_ids is None else case_ids
    require(case_ids and len(case_ids) == len(set(case_ids)) and set(case_ids) <= set(known), 'exact planned case IDs required')
    output = collection.destination(output); gate = collection.require_r09_release(guard_report)
    for path in (output / 'infrastructure_attempts').glob('*/0/parent_receipt.json'):
        require(read(path)['status'] != 'infrastructure_error', 'old infra needs explicit review/new execution; no automatic retry')
    selected = [row for cid, row in known.items() if cid in set(case_ids)]
    for case in selected:
        require((collection.case_directory(output, case) / 'case.json').exists()
                or not (output / 'infrastructure_attempts' / case['case_id']).exists(),
                'unfinished/inflight old attempt requires review; no redispatch')

    def one(case):
        canonical = collection.case_directory(output, case) / 'case.json'
        if canonical.exists():
            audit.validate_result(ROOT, canonical, context, case)
            parent = output / 'infrastructure_attempts' / case['case_id'] / '0/parent_receipt.json'
            require(parent.exists() and read(parent)['status'] == 'completed', 'canonical case lacks actual parent commit')
            return dict(case_id=case['case_id'], status='resumed_verified')
        if collection.COLLECTION_STOP.exists():
            return dict(case_id=case['case_id'], status='cancelled_stop_marker', policy_failure_label_assigned=False)
        attempt = output / 'infrastructure_attempts' / case['case_id'] / '0'
        require(not attempt.parent.exists(), 'no hidden new augmentation retry')
        attempt.mkdir(parents=True); attempt_output = attempt / 'output'
        command = worker_command(plan_path, case['case_id'], attempt_output, guard_report, registration_path)
        started = collection.utc(); status = 'infrastructure_error'; detail = ''; code = None; pid = None
        try:
            with (attempt / 'native.log').open('xb') as log:
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
                        os.killpg(pid, signal.SIGKILL); process.wait()
                    detail = 'counted requery native wall timeout; evidence retained, no policy failure label'
            if code == 0:
                fresh = audit.verify_registration(ROOT, pointer(registration_path), live=True, require_ready=True)
                child = collection.case_directory(attempt_output, case) / 'case.json'
                record = deepcopy(audit.validate_result(ROOT, child, fresh, case))
                require(type(pid) is int and record['process_id'] != pid, 'independent Popen/native child PIDs required')
                launcher = ROOT / 'scripts/run_local_sim.sh'
                record['parent_process_receipt'] = dict(launcher=str(launcher.relative_to(ROOT)), launcher_sha256=collection.sha(launcher),
                    process_id=pid, command=command, child_case_path=str(child.relative_to(ROOT)), child_case_sha256=collection.sha(child))
                collection.write_json(canonical, record); status = 'completed'
        except Exception as error:
            detail = ''.join(traceback.format_exception(error))
        receipt = dict(version=audit.RUNNER_VERSION, case_id=case['case_id'], status=status, detail=detail,
            returncode=code, process_id=pid, process_started=pid is not None, started_utc=started, finished_utc=collection.utc(),
            command=command, attempt=0, policy_failure_label_assigned=False, gate=gate,
            execution_registration=context['registry_pointer'], actual_plan=context['current']['registration']['plan'],
            automatic_retries=False)
        collection.write_json(attempt / 'parent_receipt.json', receipt)
        return receipt

    results = []; iterator = iter(selected); inflight = {}; stopped = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(workers, len(selected))):
            case = next(iterator); inflight[pool.submit(one, case)] = case['case_id']
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                inflight.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(status='infrastructure_error', detail=repr(error), policy_failure_label_assigned=False)
                results.append(result)
                if result['status'] not in ('completed', 'resumed_verified'):
                    stopped = True
            if not stopped:
                for _ in done:
                    case = next(iterator, None)
                    if case is not None:
                        inflight[pool.submit(one, case)] = case['case_id']
    proof = audit.audit_collection(ROOT, pointer(registration_path), require_complete=False)
    collection.write_json(output / 'manifest.json', proof['acquisition_projection'], overwrite=True)
    report = dict(version=audit.RUNNER_VERSION, created_utc=collection.utc(), selected_case_ids=[c['case_id'] for c in selected],
        selected_complete=len(results) == len(selected) and all(r['status'] in ('completed', 'resumed_verified') for r in results),
        receipts=results, workers=workers, expected=100, completed=proof['acquisition_projection']['completed'], complete=proof['complete'],
        stopped_dispatch_after_first_infrastructure_error=stopped, inflight_jobs_allowed_to_finish=True,
        automatic_retries=False, primary_K1_modified=False, GPU_used=False,
        execution_registration=context['registry_pointer'], formal_model_or_policy_release=False)
    name = collection.canonical_sha(report)[:16] + '.json'
    collection.write_json(output / 'queue_reports' / name, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); actions = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'approve-registration', 'check', 'collect-case', 'run-queue', 'audit'):
        actions.add_argument('--' + name, action='store_true')
    for name in ('proposal', 'shared-registration', 'setup', 'output', 'plan', 'execution-registration', 'approved-registration'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--case-id'); parser.add_argument('--queue-case-id', action='append')
    parser.add_argument('--workers', type=int, default=1); parser.add_argument('--case-timeout', type=int, default=900)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--guard-report', type=Path, default=ROOT / 'artifacts/framework/aligned_r09/final_report.json')
    args = parser.parse_args()
    def need(*names):
        for name in names:
            if getattr(args, name) is None:
                parser.error('required --' + name.replace('_', '-'))
    if args.prepare:
        need('proposal', 'shared_registration', 'setup', 'output')
        value = prepare(args.proposal, args.shared_registration, args.setup, args.output)
    elif args.approve_registration:
        need('execution_registration', 'approved_registration')
        value = approve_registration(args.execution_registration, args.approved_registration)
    elif args.check:
        need('execution_registration'); context = audit.verify_registration(ROOT, pointer(args.execution_registration), live=True)
        value = dict(valid=True, fixed_cases=100, execution_ready=context['registry']['execution_ready'], native_or_GPU_started=False)
    elif args.collect_case:
        need('plan', 'case_id', 'output', 'execution_registration')
        record = collect_case(args.plan, args.case_id, args.output, args.guard_report, args.execution_registration)
        value = dict(case_id=record['case_id'], status=record['status'])
    elif args.audit:
        need('execution_registration', 'output')
        proof = audit.audit_collection(ROOT, pointer(args.execution_registration), require_complete=args.require_complete)
        collection.write_json(args.output, proof); value = dict(report=pointer(args.output), complete=proof['complete'])
    else:
        need('plan', 'output', 'execution_registration')
        value = run_queue(args.plan, args.output, args.guard_report, args.execution_registration,
                          case_ids=args.queue_case_id, workers=args.workers, case_timeout=args.case_timeout)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
