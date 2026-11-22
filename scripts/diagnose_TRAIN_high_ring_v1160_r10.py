"""Fixed TRAIN highring102 CV/Context passive diagnosis; no queue or retry.

--check performs the inherited source/model CPU preflight without installing
Native/probe/trace/profile hooks. Native uses the unchanged1157/1156 loop with only
passive diagnostic observers and an honestly new identity/case version.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask import train_high_ring_diagnostic_v1160 as overlay


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'raw-episode-id', 'method', 'out', 'runtime', 'diagnostic-lock'):
        value.add_argument('--'+name, required=True)
    value.add_argument('--config')
    value.add_argument('--frozen-inputs')
    value.add_argument('--check', action='store_true')
    return value


def process_identity():
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(fields[19]), argv=list(sys.argv), executable=sys.executable)


def run_case(args, *, root=ROOT):
    root = Path(root).resolve()
    overlay.require(root == ROOT, 'actual project root required')
    chosen = overlay.selection(root, args)
    lock, reference = overlay.read_diagnostic_lock(root, args.diagnostic_lock)
    if not args.check:
        overlay.require(lock['integration_complete'] is True, 'complete diagnostic CPU evidence required before Native')
    out = overlay.parent._path(root, args.out, prefix=overlay.ARTIFACT_PREFIX, must_exist=False)
    overlay.require(not out.exists(), 'fresh diagnostic output required; no overwrite/retry')
    base = overlay.parent._path(root, args.runtime, prefix=overlay.RUNTIME_PREFIX, must_exist=False)
    phase = base/('CPU_check' if args.check else 'native')
    overlay.require(not phase.exists(), 'fresh diagnostic CPU/native phase required; no retry')
    guard = importlib.import_module('our_mani.multitask.post_completed_TEST_resource_guard_v1106')
    with guard.reserved_resources_after_TEST(root, enabled=True) as permit:
        resource = guard.require_permit(permit, root)
        parent_worker = importlib.import_module('scripts.run_TRAIN_ring_terminal_full16_overlay_v1157_r10')
        system = parent_worker.environment_proof(root)
        phase.mkdir(parents=True, exist_ok=False)
        launch = dict(schema=overlay.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
            process=process_identity(), selection=chosen, source_overlay_lock=reference,
            parent_source_overlay_lock=dict(path=overlay.PARENT_LOCK, sha256=overlay.PARENT_LOCK_SHA),
            resource_proof=resource, frozen_OS_environment=system,
            diagnostic_only=True, original_745_execution_acceptance_granted=False,
            formal_score_contribution=False, new_protocol_formal_VAL_TEST_qualification=False,
            automatic_retry=False)
        overlay.parent.write_json(phase/'launch.json', launch)
        prepared = None; observers = {}
        try:
            edges = importlib.import_module('our_mani.multitask.frozen_input_edges_v463')
            evaluator = importlib.import_module('scripts.evaluate_multitask_case_r10')
            with ExitStack() as lifetime:
                lifetime.enter_context(edges.declared_edge_scope())
                lifetime.enter_context(edges.evaluator_scope(evaluator))
                prepared = overlay.prepare(root, args)
                overlay.verify(root, prepared)
                if args.check:
                    execution = dict(mode='CPU_check', complete=True, native_started=False, GPU_started=False,
                        observers_installed=False, native_execution_ready=lock['integration_complete'],
                        unpinned_diagnostic_sources=lock['unpinned_sources'], identity=prepared['identity'],
                        inherited_1157_identity=prepared['inherited_1157_identity'],
                        component_domain=prepared['component_domain'].receipt(),
                        source_and_model_preflight=prepared['model_preflight'], actual_split='train',
                        execution_protocol=prepared['execution_protocol'])
                else:
                    runtime = importlib.import_module('scripts.run_task_executor_smoke_author_suite16_v744_r10')
                    with overlay.runtime_scope(root, prepared, runtime, observers):
                        runtime.run(prepared)
                    case_name = args.out+'/case.json'
                    case = json.loads(overlay.parent.raw(root, case_name))
                    importlib.import_module('our_mani.multitask.evaluation').validate_completed_case(case, prepared['identity'])
                    overlay.require(case.get('version') == overlay.SCHEMA+'_case'
                        and case.get('execution_protocol') == prepared['execution_protocol']
                        and case.get('source_overlay_lock') == reference
                        and case.get('parent_source_overlay_lock') == prepared['source_overlay_pointer']
                        and case.get('cleanup_completed') is True and case.get('process_id') == os.getpid()
                        and type(case.get('official_task_success')) is bool,
                        'clean actual new-diagnostic TRAIN case and identity required')
                    overlay.require(observers.get('restored') is True,
                                    'all diagnostic probe/factory/writer/trace scopes must restore')
                    quality=overlay.require_passive_quality(observers,ticks=case['counted_clock']['ticks'],
                        terminal=case['official_task_success'] or case['native_terminal_failure'])
                    execution = dict(mode='passive_high_ring_native_TRAIN', complete=True,
                        native_started=True, case=overlay.parent.pointer(root, case_name),
                        identity=prepared['identity'], native_success=case['official_task_success'],
                        native_failure=case['native_terminal_failure'], native_false_preserved=True,
                        diagnostic_events=observers,diagnostic_quality=quality,formal_score_contribution=False)
            overlay.verify(root, prepared)
            guard.require_permit(permit, root)
            result = dict(launch, complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                          execution=execution, TEST_used_for_fitting=False)
            overlay.parent.write_json(phase/'worker_sidecar.json', result)
            return result
        except BaseException as error:
            try:
                overlay.parent.write_json(phase/'failure.json', dict(schema=overlay.SCHEMA, complete=False,
                    exception_type=type(error).__name__, message=str(error)[:1000],
                    identity=None if prepared is None else prepared['identity'], diagnostic_events=observers,
                    counted_as_policy_failure=False, original_exception_rethrown=True, automatic_retry=False))
            except Exception:
                pass
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    result = run_case(args)
    print(json.dumps(dict(schema=overlay.SCHEMA, complete=True, execution=result['execution'],
                          diagnostic_only=True, formal_score_contribution=False)))
    return result


if __name__ == '__main__':
    main()
