"""One graph-bound case under an explicitly owned parallel1213 resource slot.

The actual worker is1206. The unchanged742 preflight uses its recorded1167
entry, and the original744 physical loop remains unchanged. No queue/retry.
"""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask import public_held_repaired_case_v1206 as case


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('graph', 'config', 'split', 'task', 'raw-episode-id', 'method', 'out', 'runtime',
                 'resource-session', 'resource-session-sha256', 'resource-job-id'):
        value.add_argument('--' + name, required=True)
    for name in ('suite', 'suite-lock', 'source-inputs', 'admission'):
        value.add_argument('--' + name)
    value.add_argument('--resource-slot', required=True, type=int)
    value.add_argument('--resource-gpu-slot', type=int)
    value.add_argument('--check', action='store_true')
    return value


def process_identity():
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(fields[19]), argv=list(sys.argv), executable=sys.executable)


def declared_job(args, session):
    jobs = [job for job in session['jobs'] if job.get('job_id') == args.resource_job_id]
    case.require(len(jobs) == 1, 'exactly one predeclared resource job required')
    job = jobs[0]
    actual = dict(task=args.task, raw_episode_id=args.raw_episode_id, method=args.method,
                  split=args.split, out=args.out, runtime=args.runtime)
    case.require(all(job.get(key) == value for key, value in actual.items()),
                 'resource job differs from actual selected case')
    configuration = job['configuration']
    case.require(type(configuration) is dict and configuration.get('path') == args.config,
                 'resource job configuration differs from actual argv')
    return job


def CPU_affinity_proof(root, session, chosen, *, check, slot):
    reference = session['plan']
    name = reference.get('path', '')
    case.require(name.startswith('runtime/framework/public_held_repaired_owner_v1207/' + chosen['split'] + '/')
                 and Path(name).name == 'plan.json', 'typed owning plan required for CPU mask')
    plan, _ = case.graphs.read_project_json(root, reference)
    affinity = plan['CPU_affinity']
    expected_ref = chosen['graph']['control']['static_inputs']['parallel_CPU_affinity_profile']
    case.require(case._same(affinity['profile_ref'], expected_ref), 'CPU mask profile differs from graph')
    declared, _ = case.graphs.read_project_json(root, expected_ref)
    case.require(all(affinity.get(key) == value for key, value in declared.items()),
                 'planned CPU partition differs from frozen profile')
    masks = affinity['native_slot_cpus']
    expected = affinity['control_and_CPU_preflight_cpus'] if check else masks[slot]
    case.require(type(expected) is list and bool(expected) and expected == sorted(set(expected))
                 and all(type(cpu) is int and cpu >= 0 for cpu in expected), 'explicit CPU mask required')
    main = sorted(os.sched_getaffinity(0))
    case.require(main == expected, 'actual main-thread CPU affinity differs from assigned slot')
    threads, vanished = [], []
    for path in Path('/proc/self/task').iterdir():
        try:
            actual = sorted(os.sched_getaffinity(int(path.name)))
        except ProcessLookupError:
            vanished.append(int(path.name))
            continue
        case.require(actual == expected, 'a worker thread escaped its assigned CPU mask')
        threads.append(dict(tid=int(path.name), cpus=actual))
    return dict(profile=expected_ref, expected_cpus=expected, actual_main_cpus=main,
                all_threads_checked=True, thread_cpus=threads, vanished_thread_ids=vanished,
                additional_native_calls=0)


@contextmanager
def private_process_paths(phase):
    """Independent process-local temporary/cache paths; restore on every exit."""
    keys = ('TMPDIR', 'XDG_CACHE_HOME')
    old = {key: os.environ.get(key) for key in keys}
    old_tempdir = tempfile.tempdir
    paths = dict(TMPDIR=phase / 'tmp', XDG_CACHE_HOME=phase / 'cache')
    try:
        for key, path in paths.items():
            path.mkdir(parents=True, exist_ok=False)
            os.environ[key] = str(path)
        tempfile.tempdir = None
        yield {key: str(path) for key, path in paths.items()}
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = old_tempdir


def run_case(args, *, root=ROOT):
    root = Path(root).resolve()
    case.require(root == ROOT, 'actual project root required')
    guard = importlib.import_module('our_mani.multitask.public_held_repaired_resource_guard_v1213')
    session_ref = dict(root='project', path=args.resource_session, sha256=args.resource_session_sha256)
    session = guard.read_session(root, session_ref)
    job = declared_job(args, session)
    chosen = case.selection(root, args)
    case.require(case._same(session['graph'], chosen['graph_ref']),
                 'resource session belongs to another graph')
    case.require(case._same(job['configuration'], chosen['route']['configuration']),
                 'resource configuration identity differs from selected graph')
    phase_name = 'CPU_check' if args.check else 'native'
    phase = chosen['runtime'] / phase_name
    case.require(not phase.exists(), 'fresh CPU/native phase required; no retry')
    permit = None
    with guard.child_scope(root, session_ref=session_ref, job=job, phase=phase_name,
                           slot=args.resource_slot, gpu_slot=args.resource_gpu_slot, enabled=True) as permit:
        resource = guard.require_permit(permit, root)
        system = case.environment_proof(root, chosen)
        phase.mkdir(parents=True, exist_ok=False)
        launch = dict(schema=case.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
            process=process_identity(), graph=chosen['graph_ref'],
            **{key: chosen['graph'][key] for key in case.readiness.HASHES},
            actual_worker=chosen['declaration']['worker'], actual_preparer=chosen['declaration']['preparer'],
            selection={key: chosen[key] for key in ('task', 'method', 'split', 'raw_episode_id')},
            resource_proof=resource, resource_session=session_ref, resource_job_id=job['job_id'],
            resource_slot=args.resource_slot, resource_gpu_slot=args.resource_gpu_slot,
            frozen_OS_environment=system, automatic_retry=False,
            original_745_execution_acceptance_granted=False, formal_score_contribution=args.split == 'validation')
        case.write_json(phase / 'launch.json', launch)
        prepared = None
        try:
            edges = importlib.import_module('our_mani.multitask.frozen_input_edges_v463')
            evaluator = importlib.import_module('scripts.evaluate_multitask_case_r10')
            with private_process_paths(phase) as process_paths, ExitStack() as lifetime:
                affinity_before = CPU_affinity_proof(root, session, chosen, check=args.check, slot=args.resource_slot)
                lifetime.enter_context(edges.declared_edge_scope())
                lifetime.enter_context(edges.evaluator_scope(evaluator))
                lifetime.enter_context(case.admission_scope(root, chosen, args, permit))
                prepared = case.prepare(root, args, chosen=chosen)
                case.verify(root, prepared)
                witness = prepared['graph_witness']
                common = dict(identity=prepared['identity'], graph_witness=witness, actual_split=args.split,
                    component_domain=prepared['component_domain'].receipt(), execution_protocol=prepared['execution_protocol'])
                if args.check:
                    execution = dict(common, mode='CPU_check', complete=True, native_started=False, GPU_started=False,
                        original_substrate_identity=prepared['source_substrate_identity'],
                        source_and_model_preflight=prepared['model_preflight'],
                        physical_factory_constructed=False, graph_source_validation_only=True)
                else:
                    guard.require_native_permit(permit, root)
                    runtime = importlib.import_module('scripts.run_task_executor_smoke_author_suite16_v744_r10')
                    native_start = time.monotonic()
                    with case.runtime_scope(root, prepared, runtime):
                        runtime.run(prepared)
                    native_end = time.monotonic()
                    case_ref = case.pointer(root, args.out + '/case.json')
                    record, _ = case.graphs.read_project_json(root, case_ref)
                    importlib.import_module('our_mani.multitask.evaluation').validate_completed_case(record, prepared['identity'])
                    case.require(record.get('version') == chosen['declaration']['case_schema']
                        and record.get('execution_protocol') == prepared['execution_protocol']
                        and record.get('graph_witness') == witness and record.get('cleanup_completed') is True
                        and type(record.get('process_id')) is int and record['process_id'] == os.getpid(),
                        'clean actual graph-bound case required')
                    case.readiness.validate_sampling_case(record, prepared['expected_author_outcome_protocol'],
                        block=args.task in case.BLOCKS, case_version=chosen['declaration']['case_schema'])
                    restored = prepared['runtime_scope_restoration']
                    case.require(restored.get('complete') is True and restored.get('restored') is True,
                                 'all owned runtime scopes must restore')
                    execution = dict(common, mode='graph_bound_native_TRAIN' if args.split == 'train' else 'graph_bound_native_validation',
                        complete=True, native_started=True, case=case_ref,
                        native_success=record['official_task_success'], native_failure=record['native_terminal_failure'],
                        native_false_preserved=True, runtime_scope_restoration=restored,
                        actual_native_interval=dict(start_monotonic=native_start, end_monotonic=native_end),
                        qualification_granted_by_worker=False)
                case.verify(root, prepared)
                affinity_after = CPU_affinity_proof(root, session, chosen, check=args.check, slot=args.resource_slot)
            guard.require_permit(permit, root)
            result = dict(launch, complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                execution=execution, identity=prepared['identity'], graph_witness=prepared['graph_witness'],
                CPU_affinity_receipt=dict(before=affinity_before, after=affinity_after),
                private_process_paths=process_paths, private_process_paths_restored=True, TEST_used_for_fitting=False)
        except BaseException as error:
            try:
                case.write_json(phase / 'failure.json', dict(schema=case.SCHEMA, complete=False,
                    graph=chosen['graph_ref'], **{key: chosen['graph'][key] for key in case.readiness.HASHES},
                    exception_type=type(error).__name__, message=str(error)[:1500],
                    identity=None if prepared is None else prepared['identity'],
                    runtime_scope_restoration=None if prepared is None else prepared['runtime_scope_restoration'],
                    counted_as_policy_failure=False, original_exception_rethrown=True, automatic_retry=False))
            except Exception:
                pass
            raise
    result['resource_scope_receipt'] = guard.permit_receipt(permit)
    receipt = result['resource_scope_receipt']
    case.require(receipt.get('body_completed') is True and receipt.get('restored') is True
                 and receipt.get('leases_removed') is True and receipt.get('errors') == [],
                 'parallel resource scope must close cleanly before final sidecar')
    case.write_json(phase / 'worker_sidecar.json', result)
    return result


def main(argv=None):
    result = run_case(parser().parse_args(argv))
    print(json.dumps(dict(schema=case.SCHEMA, complete=True, execution=result['execution'],
                         original_745_execution_acceptance_granted=False)))
    return result


if __name__ == '__main__':
    main()
