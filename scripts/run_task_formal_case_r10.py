"""One trained task's formal VAL/TEST case; original actor and physics unchanged.

Requires both matching critics through the task model coverage certificate.
TEST additionally requires this task's complete five-method VAL release.
Only the fixed first official TEST source may record its already observed frames.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
from our_mani.multitask.protocol import MultitaskProtocolError

VERSION = 'r10_trained_task_formal_case_wrapper_v3'
WITNESS_SCHEMA = 'r10_task_formal_case_execution_contract_v1'
OURS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value')
require = evaluator.require
read = evaluator.read
absolute = evaluator.absolute


def pointer(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'wrapper pointer must remain in desktop project')
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def source_manifest_for(split):
    return ROOT / 'artifacts/framework/protocol' / (
        'train_manifest.json' if split == 'train' else 'validation_manifest.json')


def verify_development_source(prepared):
    """A subset manifest may select, but cannot invent or rewrite a source row."""
    args = prepared['args']
    source = source_manifest_for(args.split)
    evaluator.locked_digest(source, prepared['files'])
    original = read(source)
    (row, _), values = evaluator.select_source(original, args.split, args.task, args.raw_episode_id)
    require(row == prepared['row'], 'development source row differs from original frozen TRAIN/VAL manifest')
    return source, values


def verify_value_query_heads(library, critic, predictor):
    """Match the fitted one-query critic to the controller's deployed contract."""
    from our_mani.multitask.candidate_success import CONTINUATION_POLICY
    first = library.phase_order[0]
    key = '|'.join((library.task_spec.task_name, predictor, first))
    require(set(critic.heads) == {key}, 'critic must train exactly the first actionable phase head')
    metadata = critic.metadata
    require(metadata.get('continuation_selection_policy') == CONTINUATION_POLICY
        and metadata.get('classifier_and_calibration_population') == 'episode_first_actionable_decision_only'
        and metadata.get('value_use') == 'first actionable selection once per episode including fallback; all later selections geometry',
        'critic query population or downstream geometry continuation differs')
    omitted = metadata.get('omitted_downstream_heads')
    require(isinstance(omitted, list)
        and [row.get('phase') for row in omitted] == list(library.phase_order[1:])
        and all(row.get('reason') == 'unused downstream phase; frozen geometry continuation has no critic query' for row in omitted),
        'critic must explicitly declare all downstream phases geometry-only')
    return dict(first_actionable_phase=first, trained_head=key, downstream_value_queries=False)


def actor_model_preflight(prepared):
    """Load real fitted libraries/critics on CPU before a native process starts."""
    from our_mani.multitask.factory import artifact_path, continuation_identity
    from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
    from our_mani.multitask.candidate_success import CandidateSuccessModel
    args, config = prepared['args'], prepared['config']
    predictor, ranking = args.method.split('_', 1)
    artifacts = config['learned_artifacts'][args.task]
    library = InteractionLibrary.load(artifact_path(ROOT, artifacts['interaction']))
    require(library.task_spec.task_name == args.task, 'fitted interaction task differs')
    fit = library.metadata.get('fit_configuration', {})
    require(library.metadata.get('development_support_override') is False
        and fit.get('development_only') is False
        and fit.get('minimum_completed_parent_sources', 0) >= 8,
        'production actor requires eight completed TRAIN parents per skill')
    candidate = CandidateConfig(**config['candidate_config'])
    mpc = MPCConfig(**config['mpc_config'])
    continuation = continuation_identity(ROOT, config, args.task, predictor, library, candidate, mpc)
    expected = config.get('continuation_identities', {}).get(args.task, {}).get(predictor)
    require(expected is not None and expected == continuation['sha256'],
        'new task runner requires the exact frozen full continuation identity')
    context = None
    if predictor == 'context':
        from our_mani.multitask.context_se3 import MultiBodyContextDynamics
        context = MultiBodyContextDynamics(artifact_path(ROOT, config['context_checkpoint']))
        source = context.metadata['training_source_manifest']
        evaluator.locked_digest(context.path.parent / source['path_relative_to_checkpoint'], prepared['files'])
    critic = None
    if ranking == 'value':
        critic = CandidateSuccessModel(artifact_path(ROOT, artifacts[predictor + '_value']),
            continuation_sha256=continuation['sha256'], library_sha256=library.sha256)
        verify_value_query_heads(library, critic, predictor)
        for key in ('TRAIN_source_manifest_path', 'VAL_calibration_manifest_path'):
            evaluator.locked_digest(critic.metadata[key], prepared['files'])
    return dict(interaction_library_sha256=library.sha256,
        continuation_identity=continuation,
        context_checkpoint_sha256=None if context is None else context.checkpoint_sha256,
        fitted_matching_critic_sha256=None if critic is None else critic.artifact_sha256,
        all_loaders_CPU_only=True, native_started=False)


def hook_plan_for(prepared):
    files = {str(absolute(name)): checksum for name, checksum in prepared['files'].items()}
    plan = dict(protocol=prepared['config'], inputs=files)
    # These are exactly the executable aliases scoped_owned_hooks reads.
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


def test_preflight(args):
    """Retain original TEST source/model/clock checks; use a task VAL release."""
    from our_mani.multitask.protocol import validate_protocol
    from our_mani.multitask.evaluation import case_identity, verify_inputs
    from our_mani.multitask.task_validation_composer import require_task_test_release
    config_path, manifest_path, lock_path = map(absolute,
        (args.config, args.manifest, args.frozen_inputs))
    config, manifest, lock = map(read, (config_path, manifest_path, lock_path))
    validate_protocol(config)
    files = lock.get('files'); verify_inputs(ROOT, files)
    for path in (config_path, manifest_path, *evaluator.required_local_sources(args.method, split='test')):
        evaluator.locked_digest(path, files)
    require(args.replica == 0 and args.exploration_seed is None,
        'official TEST cannot be an augmented/explored replica')
    require(not args.development_demo and not args.demo_only,
        'formal TEST cannot be a development or demo-only case')
    (row, original), values = evaluator.select_source(manifest, 'test', args.task, args.raw_episode_id)
    if args.record_demo:
        first = next(value for value in values if value['task'] == args.task)
        require(row['raw_episode_id'] == first['raw_episode_id'],
            'one TEST demo per method/task uses its fixed first official source')
        evaluator.locked_digest(evaluator.FFMPEG, files)
        policy_path = ROOT / 'config/framework/demo_policy.json'
        evaluator.locked_digest(policy_path, files)
        policy = read(policy_path)
        from our_mani.multitask.evaluation import METHODS
        require(policy.get('videos_per_method_per_task') == 1
            and policy.get('methods') == list(METHODS), 'shared fixed-first-source demo policy differs')
    evaluator.locked_digest(ROOT / 'our_mani/multitask/task_validation_composer.py', files)
    require(Path(row['path']).is_dir() and row.get('mapping_status') == 'verified',
        'official reset alias source missing')
    require(config.get('formal_comparison_ready') is True, 'TEST requires frozen task formal readiness')
    r09 = evaluator.r09_release(config, files)
    release = require_task_test_release(ROOT, args.task, config.get('task_validation_release'),
        config, files, official_test_manifest=manifest_path)
    artifacts = config.get('learned_artifacts', {}).get(args.task, {})
    for key in ('geometry', 'interaction'):
        evaluator.locked_pointer(artifacts.get(key), files)
    for key in ('rigid_layout', 'colour'):
        if artifacts.get(key) is not None: evaluator.locked_pointer(artifacts[key], files)
    if args.task.startswith(('insert_onto_rotating_peg', 'push_moving_button')):
        evaluator.locked_pointer(artifacts.get('colour'), files)
    predictor, ranking = args.method.split('_', 1)
    if predictor == 'context': evaluator.locked_pointer(config.get('context_checkpoint'), files)
    if ranking == 'value': evaluator.locked_pointer(artifacts.get(predictor + '_value'), files)
    out = absolute(args.out)
    require(out.is_relative_to(evaluator.BASE.resolve()) and out != evaluator.BASE.resolve(),
        'fresh child outputs must remain inside desktop R10 artifacts')
    require(not out.exists() or not any(out.iterdir()), 'preserve prior partial/completed case output')
    require('demo_only' not in out.relative_to(evaluator.BASE.resolve()).parts,
        'formal TEST cannot write into demo-only directory')
    identity = case_identity(split='test', task=args.task, raw_episode_id=row['raw_episode_id'],
        method=args.method, source_row=row, config_sha256=file_sha(config_path),
        inputs_sha256=file_sha(lock_path), parent_raw_episode_id=None, replica=0,
        demo_only=False, exploration_seed=None)
    return dict(args=args, config=config, config_path=config_path, manifest_path=manifest_path,
        manifest_sha256=file_sha(manifest_path), row=row, original_row=original, out=out,
        identity=identity, files=files, lock_path=lock_path, lock_sha256=file_sha(lock_path),
        r09_release=r09, r10_validation_release=release, record_demo=bool(args.record_demo))


def preflight(args):
    require(args.method in OURS, 'trained task wrapper executes only the original four ours methods')
    require(args.split in ('validation', 'test') and not args.demo_only and not args.development_demo,
        'formal task entrypoint accepts only VAL/TEST, without development/demo-only scope')
    require(not args.record_demo or args.split == 'test', 'formal VAL records no videos')
    # The original preflight is used verbatim for both development splits.
    prepared = test_preflight(args) if args.split == 'test' else evaluator.preflight(args)
    evaluator.locked_digest(Path(__file__), prepared['files'])
    for name in ('our_mani/multitask/factory.py', 'our_mani/multitask/interaction.py',
            'our_mani/multitask/candidate_success.py'):
        evaluator.locked_digest(ROOT / name, prepared['files'])
    if args.split != 'test':
        source, values = verify_development_source(prepared)
        prepared['shared_hook_plan'] = hook_plan_for(prepared)
    else:
        source, values = prepared['manifest_path'], []
        evaluator.locked_digest(ROOT / 'our_mani/multitask/task_release.py', prepared['files'])
    if args.development_demo:
        first = next(row for row in values if row['task'] == args.task)
        require(first['raw_episode_id'] == args.raw_episode_id,
            'development demo must use fixed first original TRAIN source, independent of outcome')
        require('development_demo_only' in prepared['out'].relative_to(evaluator.BASE.resolve()).parts,
            'development demo requires explicit development_demo_only output namespace')
        evaluator.locked_digest(evaluator.FFMPEG, prepared['files'])
        prepared['record_demo'] = True
        # Rebuild the original identity with its supported demo-only bit. Extra
        # explicit scope remains part of the identity SHA and cannot be mistaken
        # for a fitting or formal metric case.
        identity = dict(prepared['identity'], demo_only=True,
            development_demo_only=True, fitting_source=False, formal_validation_metric=False)
        identity.pop('sha256', None); identity['sha256'] = digest(identity)
        prepared['identity'] = identity
        prepared['args'] = deepcopy(args); prepared['args'].demo_only = True
    elif 'development_demo_only' in prepared['out'].relative_to(evaluator.BASE.resolve()).parts:
        raise MultitaskProtocolError('ordinary training/validation cannot use development demo namespace')
    prepared['original_source_manifest'] = pointer(source)
    prepared['model_preflight'] = actor_model_preflight(prepared)
    coverage = prepared['config'].get('task_model_coverage_certificate')
    if coverage is not None:
        from our_mani.multitask.task_release import require_task_coverage_certificate
        evaluator.locked_digest(ROOT / 'our_mani/multitask/task_release.py', prepared['files'])
        prepared['task_coverage'] = require_task_coverage_certificate(ROOT, args.task, coverage,
            prepared['config'], prepared['files'])
    else:
        raise MultitaskProtocolError('formal task VAL/TEST requires both matching critics and task model coverage')
    return prepared


def run_prepared(prepared, *, backend=None, run_case=None, hooks=None):
    """Execute once; preserve original raw receipt plus separate hook witness."""
    args, out = prepared['args'], prepared['out']
    runner = evaluator.run_case if run_case is None else run_case
    backend = evaluator.NativeBackend() if backend is None else backend
    captures = dict(simulations=[], paths=[])
    failure = None; result = None
    try:
        if args.split == 'test':
            result = runner(prepared, backend=backend)
        else:
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
        hooks_complete = (args.split == 'test' or (len(reset_receipts) == 1
            and isinstance(reset_receipts[0], dict) and reset_receipts[0].get('complete') is True
            and all(row.get('close_complete') is True for row in path_receipts)))
        hook_receipt = dict(version=VERSION, identity=deepcopy(prepared['identity']),
            actual_wrapper=pointer(Path(__file__)), actual_process_id=os.getpid(),
            split=args.split, shared_development_hooks_applied=args.split != 'test',
            reset_recovery_receipts=reset_receipts, RML_path_recovery_receipts=path_receipts,
            complete=bool(result is not None and failure is None and hooks_complete), error=failure,
            permissions={key:deepcopy(prepared['config'].get(key)) for key in
                ('development_reset_profile', 'RML_factory_hook')} if args.split != 'test' else {},
            original_TEST_reset_unmodified=args.split == 'test', source_resampled=False,
            automatic_retry=False, policy_failure_label_assigned=False)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'trained_task_hook_receipt.json', hook_receipt)
    require(hook_receipt['complete'], 'actual trained task physical reset/RML hooks did not complete')
    evaluator.verify_preflight(prepared)
    case_path = out / 'case.json'; original = read(case_path)
    validate_completed_case(original, prepared['identity'])
    require(original.get('entrypoint_sha256') == file_sha(evaluator.__file__),
        'original run_case source identity changed')
    base_path = out / 'evaluate_run_case_receipt.json'
    require(not base_path.exists(), 'original raw receipt must not be replaced')
    case_path.rename(base_path)
    record = deepcopy(original)
    record['task_execution_witness'] = dict(schema=WITNESS_SCHEMA, task=args.task,
        original_evaluator=pointer(evaluator.__file__), actual_wrapper=pointer(Path(__file__)),
        actual_process_id=os.getpid(), split=args.split,
        coverage_certificate=deepcopy(prepared['config'].get('task_model_coverage_certificate')),
        task_validation_release=deepcopy(prepared['config'].get('task_validation_release'))
            if args.split == 'test' else None,
        base_evaluator_receipt=pointer(base_path), hook_receipt=pointer(out / 'trained_task_hook_receipt.json'))
    if args.development_demo:
        record['development_demo_execution'] = dict(version=VERSION,
            source_manifest=prepared['original_source_manifest'],
            source_selection='fixed first original TRAIN manifest source, independent of outcome',
            development_demo_only=True, fitting_source=False, formal_validation_metric=False,
            formal_TEST_metric=False, video_encoded_after_native_cleanup=True,
            original_base_video_selection_text_is_TEST_only_and_inapplicable=True,
            additional_sensor_or_physics_calls=False, native_success_is_development_evidence_only=True)
    atomic_json(case_path, record)
    return record


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'frozen-inputs', 'task', 'raw-episode-id', 'out'):
        value.add_argument('--' + key, required=True)
    value.add_argument('--split', required=True, choices=('train', 'validation', 'test'))
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
        development_demo_only=bool(args.development_demo)), ensure_ascii=False))


if __name__ == '__main__': main()
