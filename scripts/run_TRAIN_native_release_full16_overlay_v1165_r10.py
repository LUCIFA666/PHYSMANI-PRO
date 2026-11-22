"""One explicitly enabled full16 TRAIN author first-release repair case.

--check retains original CPU model-loader validation (including CPU Context model construction) without Native/GPU. Native is refused
until root locks completed integration CPU evidence. No queue, retry or source
substitution is provided by this worker. Root owns all launches.
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
from our_mani.multitask import train_native_release_full16_overlay_v1165 as overlay

ENVIRONMENT = dict(
    wrapper=dict(path='scripts/run_frozen_system_environment_v820_r10.py',
                 sha256='3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3'),
    manifest=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json',
                  sha256='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'),
    qualification=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json',
                       sha256='1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa'))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'raw-episode-id', 'method', 'out', 'runtime', 'overlay-lock'):
        value.add_argument('--'+name)
    value.add_argument('--config', help='if supplied, must equal the immutable745 method route')
    value.add_argument('--frozen-inputs', help='if supplied, must equal the immutable745 method route')
    mode=value.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    value.add_argument('--split',default='train');value.add_argument('--prior-stop')
    return value


def environment_proof(root):
    for entry in ENVIRONMENT.values():
        overlay.raw(root, entry['path'], entry['sha256'])
    module = importlib.import_module('scripts.run_frozen_system_environment_v820_r10')
    proof = module.capture_namespace_evidence(root/ENVIRONMENT['manifest']['path'], ENVIRONMENT['manifest']['sha256'],
        root/ENVIRONMENT['qualification']['path'], ENVIRONMENT['qualification']['sha256'])
    overlay.require(proof.get('complete') is True and proof.get('observer_pid') == os.getpid()
        and proof.get('wrapper') == ENVIRONMENT['wrapper']
        and proof.get('manifest') == ENVIRONMENT['manifest']
        and proof.get('namespace_qualification') == ENVIRONMENT['qualification'],
        'actual same-process frozen820 namespace proof required')
    return proof


def process_identity():
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(fields[19]), argv=list(sys.argv), executable=sys.executable)


def run_case(args, *, root=ROOT):
    root = Path(root).resolve()
    overlay.require(root == ROOT, 'actual project root required')
    overlay.validate_modes(args)
    overlay.require(all(getattr(args,k,None) for k in ('raw_episode_id','out','runtime','overlay_lock')),'explicit case fields required')
    prior_stop=overlay.require_prior_stop(root,args.prior_stop) if args.execute else None
    chosen = overlay.selection(root, args)
    source_lock, source_lock_pointer = overlay.read_overlay_lock(root, args.overlay_lock)
    if not args.check:
        overlay.require(source_lock['integration_complete'] is True,'pending integration; Native disabled')
    out = overlay._path(root, args.out, prefix=overlay.ARTIFACT_PREFIX, must_exist=False)
    overlay.require(not out.exists(), 'fresh new-protocol case output required')
    base = overlay._path(root, args.runtime, prefix=overlay.RUNTIME_PREFIX, must_exist=False)
    phase = base/('CPU_check' if args.check else 'native')
    overlay.require(not phase.exists(), 'fresh CPU/native phase required; no automatic retry')
    guard = importlib.import_module('our_mani.multitask.post_completed_TEST_resource_guard_v1106')
    with guard.reserved_resources_after_TEST(root, enabled=True) as permit:
        resource = guard.require_permit(permit, root)
        system = environment_proof(root)
        phase.mkdir(parents=True, exist_ok=False)
        launch = dict(schema=overlay.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
            process=process_identity(), selection=chosen, source_overlay_lock=source_lock_pointer,
            actual_worker=overlay.pointer(root,overlay.WORKER),actual_preparer=overlay.pointer(root,overlay.SOURCE),
            prior_controlled_stop=prior_stop,
            resource_proof=resource, frozen_OS_environment=system,
            original_745_execution_acceptance_granted=False, formal_score_contribution=False,
            new_protocol_formal_VAL_TEST_qualification=False, automatic_retry=False)
        overlay.write_json(phase/'launch.json', launch)
        prepared = None
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
                        native_execution_ready=prepared['source_overlay']['integration_complete'],
                        unpinned_integration_sources=prepared['source_overlay']['unpinned_sources'],
                        identity=prepared['identity'], original_substrate_identity=prepared['source_substrate_identity'],
                        component_domain=prepared['component_domain'].receipt(),
                        source_and_model_preflight=prepared['model_preflight'], actual_split='train',
                        execution_protocol=prepared['execution_protocol'])
                else:
                    runtime = importlib.import_module('scripts.run_task_executor_smoke_author_suite16_v744_r10')
                    with overlay.runtime_scope(root, prepared, runtime):
                        runtime.run(prepared)
                    case_name = args.out+'/case.json'
                    case = json.loads(overlay.raw(root, case_name))
                    importlib.import_module('our_mani.multitask.evaluation').validate_completed_case(case, prepared['identity'])
                    overlay.require(case.get('version') == overlay.SCHEMA+'_case'
                        and case.get('execution_protocol') == prepared['execution_protocol']
                        and case.get('source_overlay_lock') == source_lock_pointer
                        and case.get('cleanup_completed') is True and case.get('process_id') == os.getpid()
                        and type(case.get('official_task_success')) is bool
                        and type(case.get('native_terminal_failure')) is bool, 'clean new-protocol fixed TRAIN case required')
                    execution = dict(mode='new_protocol_native_TRAIN', complete=True, native_started=True,
                        case=overlay.pointer(root, case_name), identity=prepared['identity'],
                        native_success=case['official_task_success'], native_failure=case['native_terminal_failure'],
                        native_false_preserved=True, new_protocol_qualified=False)
            overlay.verify(root, prepared)
            guard.require_permit(permit, root)
            result = dict(launch, complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                          execution=execution, TEST_used_for_fitting=False)
            overlay.write_json(phase/'worker_sidecar.json', result)
            return result
        except BaseException as error:
            try:
                overlay.write_json(phase/'failure.json', dict(schema=overlay.SCHEMA, complete=False,
                    exception_type=type(error).__name__, message=str(error)[:1000],
                    identity=None if prepared is None else prepared['identity'],
                    counted_as_policy_failure=False, original_exception_rethrown=True, automatic_retry=False))
            except Exception:
                pass
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.check and not args.execute:
        result=dict(enabled=False,Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False)
        print(json.dumps(result));return result
    result = run_case(args)
    print(json.dumps(dict(schema=overlay.SCHEMA, complete=True, execution=result['execution'],
                          original_745_execution_acceptance_granted=False, formal_score_contribution=False)))
    return result


if __name__ == '__main__':
    main()
