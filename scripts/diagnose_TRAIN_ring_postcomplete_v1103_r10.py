"""One original fixed TRAIN case with a separate passive diagnostic sidecar.

All scientific work remains in frozen743/744 and the qualified820 namespace.
No queue, automatic retry, outcome-based source choice, policy edit or extra
physics/collision/getter call is introduced here. Root's new1107 owner chooses
the fixed roster and invokes this process explicitly.
"""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py'
SCHEMA = 'r10_postcomplete_fixed_TRAIN_passive_diagnostic_v1103'
WORKER = 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
RUNTIME_SOURCE = 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
TAP = 'our_mani/multitask/train_public_collision_tap_v934.py'
MANIFEST = 'artifacts/framework/protocol/train_manifest.json'
MANIFEST_SHA = 'c414e70ef4a1daf5ea9b2c8adb247576197a9b2decde97479f9006cfa3ca23a1'
TRAIN_INPUTS = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
TRAIN_INPUTS_SHA = 'd497facb879abb961cc2d2812e9233bccf43f35c927d71e1a1caa6e3f1e9932e'
CHECKPOINT = dict(path='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
                  sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
ENV_SOURCE = 'scripts/run_frozen_system_environment_v820_r10.py'
ENVIRONMENT = dict(
    wrapper=dict(path=ENV_SOURCE, sha256='3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3'),
    manifest=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json', sha256='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'),
    qualification=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json', sha256='1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa'))
PINS = {
    WORKER: '714fb3ba7e5f282cdd9f68618ed2615edbbed0d5ebfb1921ee7d202424ca6174',
    RUNTIME_SOURCE: '1ffb2922521da8b77b4e005ffa0a9b22d9d4bc02a4944253fbc76605ce6f930d',
    TAP: 'aef0493e4cb0b4e5894200dd5ca4699732a8bd07d3a1843d33854497bcdc6bea',
    ENV_SOURCE: ENVIRONMENT['wrapper']['sha256'],
}
FAMILIES = dict(ring='insert_onto_rotating_peg', remove='remove_cups_from_rotating_frame', buzz='beat_the_rotating_buzz')
_ACTIVE = threading.Lock()


def require(value, message):
    if not value:
        raise ValueError('TRAIN1103: ' + message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def _module(name):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(name)


def _raw(root, name, expected=None):
    root = Path(root).resolve()
    path = root / name
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and path.resolve(strict=True) == path, 'canonical existing project input required')
    before = path.stat()
    value = path.read_bytes()
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stamp(before) == stamp(after) and (expected is None or sha(value) == expected), 'input bytes changed: ' + name)
    return value


def _fresh_path(root, name, prefix, *, allow_existing_dir=False):
    require(type(name) is str and name.startswith(prefix) and not Path(name).is_absolute()
            and '..' not in Path(name).parts, 'new1107 diagnostic namespace required')
    path = Path(root) / name
    require(path.resolve() == path and (not path.exists() or allow_existing_dir and path.is_dir()),
            'fresh canonical output only; no overwrite/retry')
    return path


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', choices=tuple(FAMILIES), default='ring')
    for name in ('config', 'task', 'raw-episode-id', 'method', 'out', 'runtime'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--check', action='store_true')
    return p


def _selection(args, root):
    family = FAMILIES[args.diagnostic]
    require(args.task in (family, family + '_high_speed')
            and args.method in ('cv_geometry', 'context_geometry'), 'fixed diagnostic family and two geometry groups only')
    config_name = 'artifacts/framework/multitask_r10/post_training_qualification_v680_setup/' + args.task + '_ours_configuration.json'
    require(args.config == config_name, 'original680 configuration required')
    manifest = json.loads(_raw(root, MANIFEST, MANIFEST_SHA))
    fixed = [r['raw_episode_id'] for r in manifest['episodes'] if r['task'] == args.task][:4]
    require(manifest.get('split') == 'train' and len(fixed) == 4 and args.raw_episode_id in fixed,
            'original first four TRAIN parents required; no replacement')
    _raw(root, TRAIN_INPUTS, TRAIN_INPUTS_SHA)
    config = json.loads(_raw(root, config_name))
    require(config.get('context_checkpoint') == CHECKPOINT, 'actual671 checkpoint binding required')
    lock = json.loads(_raw(root, TRAIN_INPUTS, TRAIN_INPUTS_SHA))['files']
    expected = lock.get(str(Path(root) / config_name), lock.get(config_name))
    require(expected == sha(_raw(root, config_name)), 'original745 configuration SHA binding required')
    return dict(task=args.task, method=args.method, raw_episode_id=args.raw_episode_id)


def _environment(root):
    for p in ENVIRONMENT.values():
        _raw(root, p['path'], p['sha256'])
    module = _module('scripts.run_frozen_system_environment_v820_r10')
    e = ENVIRONMENT
    receipt = module.capture_namespace_evidence(Path(root) / e['manifest']['path'], e['manifest']['sha256'],
        Path(root) / e['qualification']['path'], e['qualification']['sha256'])
    require(receipt.get('complete') is True and receipt.get('observer_pid') == os.getpid()
            and receipt.get('wrapper') == e['wrapper'] and receipt.get('manifest') == e['manifest']
            and receipt.get('namespace_qualification') == e['qualification'], 'actual same-process820 namespace required')
    return receipt


def _identity():
    f = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(f[19]), argv=list(sys.argv), executable=sys.executable)


def summarize_collision(receipt):
    """Classify only complete original call trees; unknown stays explicit."""
    events = receipt['events']
    children = {}
    for e in events:
        children.setdefault(e.get('parent_call_id'), []).append(e)
    def descendants(e):
        result = []
        for c in children.get(e['call_id'], []):
            result.append(c)
            result.extend(descendants(c))
        return result
    trustworthy = not receipt.get('dropped_events') and not receipt.get('error_count')
    rows = []
    for event in events:
        if event['operation'] != 'executor.admit_joint_target':
            continue
        row = dict(call_id=event['call_id'], classification='unknown', physical_collision_claimed=False)
        context = event.get('public_context', {})
        if type(context) is dict:
            row['policy_context'] = context.get('ring_policy_context', {'status': 'unknown'})
        result = event.get('result')
        if type(result) is list and len(result) == 2 and type(result[0]) is bool and type(result[1]) is dict:
            accepted, proof = result
            row.update(accepted=accepted, original_proof=proof)
            if accepted:
                row['classification'] = 'admitted'
            elif proof.get('reason') != 'public_forecast_collision_on_RML_target_sweep':
                row['classification'] = 'original_admission_precheck:' + str(proof.get('reason', 'unknown'))
            elif trustworthy:
                ds = descendants(event)
                models = [e for e in ds if e['operation'] == 'model.collision' and e.get('result') is True]
                if len(models) == 1:
                    model = models[0]
                    qs = [e for e in descendants(model) if e['operation'] == 'model.collision_query']
                    if not qs:
                        row['classification'] = 'model_current_interval_precondition'
                    elif len(qs) == 1 and qs[0].get('result') is True:
                        maps = [e for e in descendants(qs[0]) if e['operation'] == 'map.check']
                        if not maps:
                            row['classification'] = 'public_prediction_or_geometry_unavailable_fail_closed'
                        elif len(maps) == 1 and maps[0].get('result') is True:
                            ms = descendants(maps[0])
                            arm = [e for e in ms if e['operation'] == 'arm.check_arm_collision']
                            ghosts = [e for e in ms if e['operation'] == 'registered_ghost.check_collision' and e.get('result') is True]
                            scenes = [e for e in ms if e['operation'] == 'map.public_scene']
                            if not scenes and not arm:
                                row['classification'] = 'map_allowed_contact_role_precondition'
                            elif len(arm) == 1 and arm[0].get('result') is True:
                                row['classification'] = 'arm_collision_collection'
                            elif len(arm) == 1 and arm[0].get('result') is False and ghosts:
                                row['classification'] = 'held_world_registered_ghost_collision'
                                row['registered_pairs'] = [dict(receiver=e['registered_receiver'], arguments=e.get('arguments')) for e in ghosts]
                            row['map_check_kwargs'] = maps[0].get('arguments', {}).get('kwargs')
        if not trustworthy and row.get('accepted') is False:
            row['classification'] = 'unknown_incomplete_call_tree'
        rows.append(row)
    counts = {}
    for row in rows:
        key = row['classification']
        counts[key] = counts.get(key, 0) + 1
    return dict(admission_call_count=len(rows), branch_counts=counts, admissions=rows,
                exact_robot_member='unknown_not_queried', extra_queries=0,
                physical_equivalence_claimed=False, unknown_not_filled=True)


@contextmanager
def ring_observer_scope(root, runtime, selection, events, *, permit):
    require(_ACTIVE.acquire(blocking=False), 'one observer scope per process')
    tap_module = _module('our_mani.multitask.train_public_collision_tap_v934')
    original = runtime._geometry
    taps = []
    try:
        with ExitStack() as stack:
            def geometry(p, env, descriptions):
                require(not taps, 'one original geometry construction required')
                result = original(p, env, descriptions)
                domain = p['component_domain']
                a = p['args']
                require(a.split == 'train' and a.task == selection['task'] and a.method == selection['method']
                        and a.raw_episode_id == selection['raw_episode_id']
                        and domain.require(a.task, a.method) == 'train_qualification', 'original TRAIN identity required')
                actor, _, _, policy, executor = result
                port = actor.collision_port
                require(port.map is env.map_service and port.model is env.robot_model, 'original public collision assembly required')
                tap = tap_module.TrainPublicCollisionTap(env.map_service, port=port, domain=domain,
                    permit=permit, root=root, enabled=True, development_only=True, max_events=10000)
                original_snapshot = tap._snapshot
                def snapshot():
                    public = original_snapshot()
                    fields = tap_module._namespace(policy)
                    selected = fields.get('selected')
                    policy_fields = {k: fields[k] for k in ('phase_index', 'phase_epoch') if k in fields}
                    if type(selected) is dict:
                        policy_fields['selected'] = {k: selected[k] for k in
                            ('phase', 'reference_id', 'manipulated_ids', 'predicted_reference_time') if k in selected}
                    public['ring_policy_context'] = tap._copy(policy_fields)
                    return public
                tap._snapshot = snapshot
                taps.append(tap)
                stack.enter_context(tap)
                # Wrap the callback already stored by the original executor. No
                # predicate is evaluated twice and the exact result is returned.
                callback = tap_module._safe_method(executor, 'admit_joint_target')
                tap._patch(executor, 'admit_joint_target', callback, 'executor.admit_joint_target',
                           dict(kind='original_geometry_admission_callback'))
                events.update(expected_identity=p['identity'], component_domain=domain.receipt())
                return result
            with patch.object(runtime, '_geometry', geometry):
                yield
    finally:
        try:
            original_error = sys.exc_info()[1]
            try:
                receipts = [tap.receipt() for tap in taps]
                events.update(observer_receipts=receipts, geometry_hook_restored=runtime._geometry is original,
                              restored=runtime._geometry is original and len(receipts) == 1
                                       and receipts[0]['restored'],
                              summary=summarize_collision(receipts[0]) if len(receipts) == 1 else dict(classification='unknown_no_geometry'))
            except BaseException as error:
                events['observer_receipt_error'] = type(error).__name__
                if original_error is None:
                    raise
        finally:
            _ACTIVE.release()


def _default_factory(kind):
    if kind == 'ring':
        return ring_observer_scope
    name = ('scripts.diagnose_TRAIN_remove_postcomplete_v1104_r10' if kind == 'remove'
            else 'our_mani.multitask.train_buzz_failure_tap_v1105')
    return _module(name).observer_context_factory


def run_case(args, observer_context_factory=None, *, root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, 'actual project root required')
    guard = _module('our_mani.multitask.post_completed_TEST_resource_guard_v1106')
    for name, expected in PINS.items():
        _raw(root, name, expected)
    selection = _selection(args, root)
    out = _fresh_path(root, args.out, 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107/development_demo_only/')
    runtime_base = _fresh_path(root, args.runtime, 'runtime/framework/TRAIN_zero_success_diagnosis_v1107/cases/',
                               allow_existing_dir=True)
    # CPU_check leaves the case base reusable by the separately invoked native
    # phase, but every phase directory itself is exclusive and cannot be retried.
    runtime = runtime_base / ('CPU_check' if args.check else 'native')
    require(not runtime.exists(), 'fresh phase runtime required')
    with guard.reserved_resources_after_TEST(root, enabled=True) as permit:
        resource = guard.require_permit(permit, root)
        environment = _environment(root)
        _raw(root, CHECKPOINT['path'], CHECKPOINT['sha256'])
        runtime.mkdir(parents=True, exist_ok=False)
        forwarded = [WORKER, '--config', args.config, '--manifest', MANIFEST, '--frozen-inputs', TRAIN_INPUTS,
                     '--task', args.task, '--raw-episode-id', args.raw_episode_id, '--method', args.method,
                     '--split', 'train', '--development-demo', '--out', args.out]
        if args.check:
            forwarded.append('--check')
        launch = dict(schema=SCHEMA, created_utc=utc(), invocation=_identity(), selection=selection,
                      diagnostic=args.diagnostic, original_worker_argv=forwarded, resource_proof=resource,
                      frozen_environment=environment, runner=dict(path=SOURCE, sha256=sha(_raw(root, SOURCE))),
                      frozen_original_sources=PINS, shared_context_checkpoint=CHECKPOINT,
                      formal_score_contribution=False, automatic_retry=False)
        _write(runtime / 'launch.json', launch)
        events = {}
        try:
            worker = _module('scripts.run_task_executor_smoke_author_outcome_v743_r10')
            edges = _module('our_mani.multitask.frozen_input_edges_v463')
            evaluator = _module('scripts.evaluate_multitask_case_r10')
            with ExitStack() as stack:
                stack.enter_context(edges.declared_edge_scope())
                stack.enter_context(edges.evaluator_scope(evaluator))
                if not args.check:
                    original_geometry = worker.runtime._geometry
                    def capture_prepared_identity(prepared, env, descriptions):
                        events['expected_identity'] = prepared['identity']
                        events['component_domain'] = prepared['component_domain'].receipt()
                        return original_geometry(prepared, env, descriptions)
                    stack.enter_context(patch.object(worker.runtime, '_geometry', capture_prepared_identity))
                    factory = observer_context_factory or _default_factory(args.diagnostic)
                    stack.enter_context(factory(root, worker.runtime, selection, events, permit=permit))
                stack.enter_context(patch.object(sys, 'argv', forwarded))
                if args.check:
                    stream = stack.enter_context(redirect_stdout(io.StringIO()))
                    worker.main()
                    proof = json.loads(stream.getvalue())
                    require(proof.get('complete') is True and proof.get('native_started') is False
                            and proof.get('GPU_started') is False and proof.get('actual_split') == 'train', 'original743 CPU proof required')
                    execution = dict(mode='CPU_check', original_CPU_proof=proof, native_started=False, tap_installed=False)
                else:
                    worker.main()
            if not args.check:
                require(events.get('restored') is True, 'observer restoration required')
                case_bytes = _raw(root, args.out + '/case.json')
                case = json.loads(case_bytes)
                require(case.get('split') == 'train' and case.get('task') == args.task
                        and case.get('method') == args.method and case.get('source_raw_episode_id') == args.raw_episode_id
                        and case.get('parent_raw_episode_id') == args.raw_episode_id
                        and case.get('process_id') == os.getpid() and case.get('cleanup_completed') is True
                        and case.get('shared_context_checkpoint') == CHECKPOINT
                        and case.get('formal_TEST_metric') is False and case.get('formal_validation_metric') is False
                        and case.get('TEST_used_for_fitting') is False, 'original clean fixed TRAIN case required; nativeFalse retained')
                _module('our_mani.multitask.evaluation').validate_completed_case(case, events['expected_identity'])
                execution = dict(mode='instrumented_native_TRAIN', native_started=True,
                                 case=dict(path=args.out + '/case.json', sha256=sha(case_bytes)),
                                 native_success=case['official_task_success'], diagnostic_events=events,
                                 original_case_bytes_unchanged=True)
            guard.require_permit(permit, root)
            result = dict(launch, complete=True, finished_utc=utc(), execution=execution,
                          unqualified=True, native_validated=False, policy_fix_claimed=False,
                          original_prepare_runs_each_invocation=True, source_predicate_files_unchanged=True,
                          instrumentation='pinned_AST_value_taps' if args.diagnostic == 'remove' else 'passive_call_wrappers',
                          TEST_used_for_fitting=False, unknown_values_not_filled=True)
            _write(runtime / 'diagnostic_sidecar.json', result)
            return result
        except BaseException as error:
            try:
                _write(runtime / 'failure.json', dict(schema=SCHEMA, complete=False, exception_type=type(error).__name__,
                    original_exception_rethrown=True, diagnostic_events=events, automatic_retry=False))
            except Exception:
                pass
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    result = run_case(args)
    print(json.dumps(dict(schema=SCHEMA, complete=True, mode=result['execution']['mode'],
                          formal_score_contribution=False, selection=result['selection'])))
    return result


if __name__ == '__main__':
    main()
