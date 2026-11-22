"""Fixed original TRAIN development replay of persistent verified-reference feedforward.

This nonfitting smoke preserves the original evaluator, native observations,
physics, task clock, physical reset hooks and video encoder. It cannot execute
formal VAL/TEST, select a successful demonstration or optimize a learned model.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import evaluate_multitask_case_r10 as evaluator
from scripts import collect_multitask_recovery_r10 as recovery
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, validate_completed_case

VERSION = 'r10_bin_next_reference_stop_guard_TRAIN_smoke_v276'
WITNESS_SCHEMA = 'r10_task_formal_case_execution_contract_v1'
OURS = ('cv_geometry', 'context_geometry','physmani')
require = evaluator.require
read = evaluator.read
absolute = evaluator.absolute


def pointer(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'wrapper pointer must remain in desktop project')
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def source_manifest_for(split):
    require(split == 'train', 'persistent-reference smoke accepts only original TRAIN sources')
    return ROOT / 'artifacts/framework/protocol/train_manifest.json'


def verify_development_source(prepared):
    """The subset manifest selects one exact original row, without rewriting it."""
    args = prepared['args']
    source = source_manifest_for(args.split)
    evaluator.locked_digest(source, prepared['files'])
    original = read(source)
    (row, _), values = evaluator.select_source(original, args.split, args.task, args.raw_episode_id)
    require(row == prepared['row'], 'development source row differs from original frozen TRAIN manifest')
    return source, values


def actor_model_preflight(prepared):
    if prepared['args'].method=='physmani':
        return dict(baseline_checkpoint_sha256=evaluator.locked_digest(absolute(prepared['config']['baseline_checkpoint']),prepared['files']),native_started=False)
    from our_mani.multitask.bin_stop_native_v275 import model_preflight as diagnostic_actor_model_preflight
    return diagnostic_actor_model_preflight(ROOT, prepared)


def hook_plan_for(prepared):
    files = {str(absolute(name)): checksum for name, checksum in prepared['files'].items()}
    plan = dict(protocol=prepared['config'], inputs=files)
    for name in recovery.OWNED_SOURCES:
        evaluator.locked_digest(ROOT / name, prepared['files'])
    for key in ('development_reset_profile', 'RML_factory_hook'):
        permission = prepared['config'].get(key)
        require(permission is not None, 'shared development physical reset/RML permission is missing')
        for path, checksum in recovery.permission_dependencies(permission).items():
            require(evaluator.locked_digest(path, prepared['files']) == checksum,
                'shared hook dependency differs from frozen execution lock')
    recovery.load_rml_permission(prepared['config']['RML_factory_hook'], plan)
    from our_mani.multitask.development_reset_recovery import load_permission
    physical = load_permission(prepared['config']['development_reset_profile'], root=ROOT,
        native_repo=prepared['config']['baseline_repo'])
    for name, checksum in physical['native_inputs_sha256'].items():
        require(evaluator.locked_digest(Path(prepared['config']['baseline_repo']) / name,
            prepared['files']) == checksum, 'native development reset dependency changed')
    return plan


def _uncached_preflight(args):
    require(args.split == 'train' and args.development_demo,
        'persistent-reference smoke is TRAIN-only, nonfitting development evidence')
    require(args.method in OURS, 'executor TRAIN smoke accepts exactly the three comparison methods')
    require(not args.record_demo and not args.demo_only,
        'use the explicit --development-demo flag for TRAIN development evidence')
    require(args.replica == 0 and args.exploration_seed is None,
        'fixed TRAIN smoke is not an augmented or explored replica')
    prepared = evaluator.preflight(args)
    evaluator.locked_digest(Path(__file__), prepared['files'])
    for name in ('our_mani/multitask/factory.py', 'our_mani/multitask/interaction.py'):
        evaluator.locked_digest(ROOT / name, prepared['files'])
    source, values = verify_development_source(prepared)
    prepared['shared_hook_plan'] = hook_plan_for(prepared)
    first4 = [row for row in values if row['task'] == args.task][:4]
    require(args.raw_episode_id in [row['raw_episode_id'] for row in first4],
        'qualification requires one of the fixed first four original TRAIN sources')
    require('development_demo_only' in prepared['out'].relative_to(evaluator.BASE.resolve()).parts,
        'development demo requires explicit development_demo_only output namespace')
    evaluator.locked_digest(evaluator.FFMPEG, prepared['files'])
    # Repair diagnostics retain numeric traces; avoid another video per retry.
    prepared['record_demo'] = False
    identity = dict(prepared['identity'], demo_only=True,
        development_demo_only=True, fitting_source=False, formal_validation_metric=False)
    identity.pop('sha256', None)
    identity['sha256'] = digest(identity)
    prepared['identity'] = identity
    prepared['args'] = deepcopy(args)
    prepared['args'].demo_only = True
    prepared['original_source_manifest'] = pointer(source)
    prepared['model_preflight'] = actor_model_preflight(prepared)
    coverage = prepared['config'].get('task_model_coverage_certificate')
    if coverage is not None:
        from our_mani.multitask.task_release import require_task_coverage_certificate
        evaluator.locked_digest(ROOT / 'our_mani/multitask/task_release.py', prepared['files'])
        prepared['task_coverage'] = require_task_coverage_certificate(ROOT, args.task, coverage,
            prepared['config'], prepared['files'])
    else:
        prepared['task_coverage'] = None
    return prepared


def preflight(args):
    from our_mani.multitask.case_output_isolation_v238 import require_development_output
    require_development_output(ROOT,args.out,task=args.task,raw=args.raw_episode_id,method=args.method)
    from our_mani.multitask.frozen_input_index_v93 import evaluator_scope
    files = read(absolute(args.frozen_inputs))['files']
    for name in (Path(__file__), ROOT/'our_mani/multitask/frozen_input_index_v93.py', ROOT/'our_mani/multitask/input_stat_cache_v79.py'):
        evaluator.locked_digest(name, files)
    from our_mani.multitask.cpu_topology_revalidation_v258 import revalidate_cpu_preflight
    def validate_once():
        with evaluator_scope(evaluator):return _uncached_preflight(deepcopy(args))
    prepared,events=revalidate_cpu_preflight(validate_once)
    prepared['model_preflight']['CPU_topology_revalidation_v258']=events
    return prepared


def run_prepared(prepared, *, backend=None, run_case=None, hooks=None):
    """Execute once and preserve the original raw receipt plus hook witness."""
    args, out = prepared['args'], prepared['out']
    runner = evaluator.run_case if run_case is None else run_case
    from our_mani.multitask.bin_stop_native_v275 import SharedJoint16NativeBackend as NativeActuatorDiagnosticBackend
    backend = (evaluator.NativeBackend() if args.method=='physmani' else NativeActuatorDiagnosticBackend()) if backend is None else backend
    captures = dict(simulations=[], paths=[])
    failure = None
    result = None
    try:
        hook_scope = recovery.scoped_owned_hooks if hooks is None else hooks
        with hook_scope(prepared['shared_hook_plan'], captures):
            result = runner(prepared, backend=backend)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        reset_receipts = [deepcopy(getattr(env, 'reset_recovery_receipt', None))
            for env in captures['simulations']]
        path_receipts = [deepcopy(path.recovery_receipt) for path in captures['paths']]
        hooks_complete = (len(reset_receipts) == 1
            and isinstance(reset_receipts[0], dict) and reset_receipts[0].get('complete') is True
            and all(row.get('close_complete') is True for row in path_receipts))
        hook_receipt = dict(version=VERSION, identity=deepcopy(prepared['identity']),
            actual_wrapper=pointer(Path(__file__)), actual_process_id=os.getpid(), split=args.split,
            shared_development_hooks_applied=True, reset_recovery_receipts=reset_receipts,
            RML_path_recovery_receipts=path_receipts,
            complete=bool(result is not None and failure is None and hooks_complete), error=failure,
            permissions={key:deepcopy(prepared['config'].get(key)) for key in
                ('development_reset_profile', 'RML_factory_hook')},
            original_TEST_reset_unmodified=True, source_resampled=False,
            automatic_retry=False, policy_failure_label_assigned=False)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'trained_task_hook_receipt.json', hook_receipt)
    require(hook_receipt['complete'], 'actual development physical reset/RML hooks did not complete')
    evaluator.verify_preflight(prepared)
    case_path = out / 'case.json'
    original = read(case_path)
    validate_completed_case(original, prepared['identity'])
    require(original.get('entrypoint_sha256') == file_sha(evaluator.__file__),
        'original run_case source identity changed')
    base_path = out / 'evaluate_run_case_receipt.json'
    require(not base_path.exists(), 'original raw receipt must not be replaced')
    case_path.rename(base_path)
    record = deepcopy(original)
    record['task_execution_witness'] = dict(schema=WITNESS_SCHEMA, task=args.task,
        original_evaluator=pointer(evaluator.__file__), actual_wrapper=pointer(Path(__file__)),
        actual_process_id=os.getpid(), split=args.split, coverage_certificate=None,
        task_validation_release=None, base_evaluator_receipt=pointer(base_path),
        hook_receipt=pointer(out / 'trained_task_hook_receipt.json'))
    record['development_demo_execution'] = dict(version=VERSION,
        source_manifest=prepared['original_source_manifest'],
        source_selection='fixed first four original TRAIN manifest sources, independent of outcome',
        development_demo_only=True, fitting_source=False, formal_validation_metric=False,
        formal_TEST_metric=False, video_encoded_after_native_cleanup=False,
        video_recording_requested=False,
        original_base_video_selection_text_is_TEST_only_and_inapplicable=True,
        additional_camera_or_physics_calls=False, additional_joint_proprioception_reads=False,
        optional_native_force_contact_diagnostics_enabled=False, diagnostic_values_supplied_to_policy=False,
        native_success_is_development_evidence_only=True, actual_window_limits_required=True)
    atomic_json(case_path, record)
    return record


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'frozen-inputs', 'task', 'raw-episode-id', 'out'):
        value.add_argument('--' + key, required=True)
    value.add_argument('--split', required=True, choices=('train',))
    value.add_argument('--method', required=True, choices=OURS)
    value.add_argument('--replica', type=int, default=0)
    value.add_argument('--exploration-seed', type=int)
    value.add_argument('--development-demo', action='store_true')
    value.add_argument('--record-demo', action='store_true', help=argparse.SUPPRESS)
    value.add_argument('--demo-only', action='store_true', help=argparse.SUPPRESS)
    value.add_argument('--check', action='store_true')
    return value


def main():
    args = parser().parse_args()
    prepared = preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False, GPU_started=False,
            identity=prepared['identity'], original_source_manifest=prepared['original_source_manifest'],
            model_preflight=prepared['model_preflight'], task_coverage=prepared['task_coverage']), ensure_ascii=False))
        return
    record = run_prepared(prepared)
    print(json.dumps(dict(complete=True, case=str(prepared['out'] / 'case.json'),
        native_success=record['official_task_success'], ticks=record['counted_clock']['ticks'],
        development_demo_only=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
