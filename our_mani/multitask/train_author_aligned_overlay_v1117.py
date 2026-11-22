"""New fixed-TRAIN execution overlay; the frozen 742/743/744 substrate stays intact.

The old lock certifies original source/model/reset inputs. A separate explicit
lock certifies this new worker and policy overlay. Nothing here grants old745
acceptance or formal VAL/TEST qualification. All physical/getter/outcome calls
remain in the original consumer and the source-pinned integrated components.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'our_mani/multitask/train_author_aligned_overlay_v1117.py'
WORKER = 'scripts/run_TRAIN_author_aligned_overlay_v1117_r10.py'
SCHEMA = 'r10_fixed_TRAIN_author_aligned_overlay_v1117'
LOCK_SCHEMA = SCHEMA + '_source_lock'
CPU_EVIDENCE_SCHEMA = SCHEMA + '_CPU_integration_evidence'
LEGACY_WORKER = 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
LEGACY_RUNTIME = 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
MANIFEST = 'artifacts/framework/protocol/train_manifest.json'
MANIFEST_SHA = 'c414e70ef4a1daf5ea9b2c8adb247576197a9b2decde97479f9006cfa3ca23a1'
TRAIN_INPUTS = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
TRAIN_INPUTS_SHA = 'd497facb879abb961cc2d2812e9233bccf43f35c927d71e1a1caa6e3f1e9932e'
CHECKPOINT = dict(path='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
                  sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
NAME = 'TRAIN_author_aligned_repair_v1117'
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
RUNTIME_PREFIX = 'runtime/framework/' + NAME + '/cases/'
RUNTIME_SETUP_PREFIX = 'runtime/framework/' + NAME + '_setup/'
FAMILIES = ('insert_onto_rotating_peg', 'remove_cups_from_rotating_frame', 'beat_the_rotating_buzz')
TASKS = tuple(t for family in FAMILIES for t in (family, family+'_high_speed'))
METHODS = ('cv_geometry', 'context_geometry')
INTEGRATION_MODULE = 'our_mani.multitask.author_aligned_policy_integration_v1118'
OVERLAY_SOURCES = (
    SOURCE, WORKER,
    'our_mani/multitask/author_aligned_geometry_executor_v1114.py',
    'our_mani/multitask/author_aligned_geometry_integration_v1114b.py',
    'our_mani/multitask/action_conditioned_mode_observer_v1115.py',
    'our_mani/multitask/action_conditioned_mode_observer_v1115b.py',
    'our_mani/multitask/author_aligned_remove_phase_v1116.py',
    'our_mani/multitask/author_aligned_policy_integration_v1118.py',
    'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
)
FROZEN_PINS = {
    LEGACY_WORKER: '714fb3ba7e5f282cdd9f68618ed2615edbbed0d5ebfb1921ee7d202424ca6174',
    LEGACY_RUNTIME: '1ffb2922521da8b77b4e005ffa0a9b22d9d4bc02a4944253fbc76605ce6f930d',
    'our_mani/multitask/author_aligned_geometry_executor_v1114.py': '83199e573969d9d9bcd69cb1e4652edfab010fb815fcd19bfd0dc59d78c7d91e',
    'our_mani/multitask/author_aligned_geometry_integration_v1114b.py': 'fd7aea7488c826c768a0310b5cc12f4ebac8f75cc4fa430b793d5e1c473c420d',
    'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py': '66209c40c36973db5586e1754c1ec6b0c8411f4513720d2ebe566ccb93864d84',
}


def require(value, message):
    if not value:
        raise ValueError('TRAIN1117: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _path(root, name, *, prefix=None, must_exist=True):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and str(Path(name)) == name, 'canonical project-relative path required')
    if prefix is not None:
        require(name.startswith(prefix), 'new explicit TRAIN1117 namespace required')
    path = Path(root)/name
    require(path.resolve(strict=must_exist) == path, 'symlink/alias inputs and outputs are forbidden')
    return path


def raw(root, name, expected=None):
    path = _path(root, name)
    before = path.stat(); value = path.read_bytes(); after = path.stat()
    stamp = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    require(stamp(before) == stamp(after), 'input changed during read: '+name)
    actual = hashlib.sha256(value).hexdigest()
    require(expected is None or expected == actual, 'input SHA changed: '+name)
    return value


def pointer(root, name):
    return dict(path=name, sha256=hashlib.sha256(raw(root, name)).hexdigest())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _cpu_evidence(root, reference, files):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'explicit CPU evidence pointer required')
    require(reference['path'].startswith((SETUP_PREFIX, RUNTIME_SETUP_PREFIX)), 'new TRAIN1117 CPU evidence namespace required')
    evidence = json.loads(raw(root, reference['path'], reference['sha256']))
    require(evidence.get('schema') == CPU_EVIDENCE_SCHEMA and evidence.get('complete') is True
            and type(evidence.get('tests_exit_code')) is int and evidence.get('tests_exit_code') == 0
            and evidence.get('native_started') is False
            and evidence.get('GPU_started') is False and evidence.get('source_hashes') == files,
            'complete CPU integration evidence must bind every actual overlay source')
    return evidence


def write_overlay_lock(root, name, *, integration_complete=False, integration_cpu_evidence=None):
    """Root preparer writes a NEW immutable source lock after code/tests settle.

    Incomplete locks can support --check, but can never start Native. No SHA is
    invented for a missing integration source. To enable Native, root supplies
    a complete CPU evidence pointer whose source_hashes equals this source map.
    """
    root = Path(root).resolve(); path = _path(root, name, prefix=SETUP_PREFIX, must_exist=False)
    require(type(integration_complete) is bool, 'explicit integration state required')
    raw(root, MANIFEST, MANIFEST_SHA); raw(root, TRAIN_INPUTS, TRAIN_INPUTS_SHA)
    for source, expected in FROZEN_PINS.items():
        raw(root, source, expected)
    files = {}; unpinned = []
    for source in OVERLAY_SOURCES:
        if (root/source).exists():
            files[source] = pointer(root, source)['sha256']
        else:
            unpinned.append(source)
    require(not integration_complete or not unpinned, 'all integration sources must exist before Native activation')
    if integration_complete:
        _cpu_evidence(root, integration_cpu_evidence, files)
    else:
        require(integration_cpu_evidence is None, 'incomplete integration cannot claim completed CPU evidence')
    value = dict(schema=LOCK_SCHEMA, required_sources=list(OVERLAY_SOURCES), files=files,
        unpinned_sources=unpinned, integration_complete=integration_complete,
        integration_cpu_evidence=integration_cpu_evidence,
        original_TRAIN_input_lock=dict(path=TRAIN_INPUTS, sha256=TRAIN_INPUTS_SHA),
        original_TRAIN_manifest=dict(path=MANIFEST, sha256=MANIFEST_SHA),
        original_745_execution_acceptance_granted=False, formal_VAL_TEST_qualification=False,
        automatic_retry=False, TEST_used_for_fitting=False)
    write_json(path, value)
    return dict(path=name, sha256=hashlib.sha256(raw(root, name)).hexdigest())


def read_overlay_lock(root, name):
    _path(root, name, prefix=SETUP_PREFIX)
    value = json.loads(raw(root, name))
    require(value.get('schema') == LOCK_SCHEMA and value.get('required_sources') == list(OVERLAY_SOURCES),
            'explicit new source-overlay schema and complete source roster required')
    files = value.get('files')
    require(type(files) is dict and set(files) <= set(OVERLAY_SOURCES)
            and value.get('unpinned_sources') == [s for s in OVERLAY_SOURCES if s not in files], 'honest overlay source map required')
    require(value.get('original_TRAIN_input_lock') == dict(path=TRAIN_INPUTS, sha256=TRAIN_INPUTS_SHA)
            and value.get('original_TRAIN_manifest') == dict(path=MANIFEST, sha256=MANIFEST_SHA)
            and value.get('original_745_execution_acceptance_granted') is False
            and value.get('formal_VAL_TEST_qualification') is False and value.get('automatic_retry') is False
            and value.get('TEST_used_for_fitting') is False, 'TRAIN source-only substrate and new protocol must remain distinct')
    for source, expected in FROZEN_PINS.items():
        raw(root, source, expected)
    for source, expected in files.items():
        raw(root, source, expected)
    require(type(value.get('integration_complete')) is bool, 'explicit integration state required')
    if value['integration_complete']:
        require(not value['unpinned_sources'], 'Native cannot use unpinned integration sources')
        _cpu_evidence(root, value.get('integration_cpu_evidence'), files)
    else:
        require(value.get('integration_cpu_evidence') is None, 'pending integration cannot claim CPU completion')
    return value, pointer(root, name)


def selection(root, args):
    require(args.task in TASKS and args.method in METHODS, 'six fixed TRAIN tasks and two geometry methods only')
    config_name = 'artifacts/framework/multitask_r10/post_training_qualification_v680_setup/' + args.task + '_ours_configuration.json'
    require(args.config == config_name, 'original frozen680 configuration required')
    manifest = json.loads(raw(root, MANIFEST, MANIFEST_SHA))
    fixed = [r['raw_episode_id'] for r in manifest['episodes'] if r['task'] == args.task][:4]
    require(manifest.get('split') == 'train' and len(fixed) == 4 and args.raw_episode_id in fixed,
            'original TRAIN first four parents only; no replacement or outcome selection')
    lock = json.loads(raw(root, TRAIN_INPUTS, TRAIN_INPUTS_SHA))['files']
    config = json.loads(raw(root, config_name))
    expected = lock.get(str(Path(root)/config_name), lock.get(config_name))
    raw(root, config_name, expected)
    require(expected is not None and config.get('context_checkpoint') == CHECKPOINT, 'original745 config and shared671 model identity required')
    return dict(task=args.task, method=args.method, raw_episode_id=args.raw_episode_id,
                fixed_parent_index=fixed.index(args.raw_episode_id), original_first_four=fixed)


def prepare(root, args):
    """Reuse original742 source/model preflight, then bind the NEW worker honestly."""
    root = Path(root).resolve()
    chosen = selection(root, args)
    overlay, overlay_pointer = read_overlay_lock(root, args.overlay_lock)
    require(SOURCE in overlay['files'] and WORKER in overlay['files'], 'actual new preparer/worker must be pinned even for CPU check')
    out = _path(root, args.out, prefix=ARTIFACT_PREFIX, must_exist=False)
    require(not out.exists(), 'fresh output required; no overwrite or retry')
    forwarded = SimpleNamespace(config=args.config, manifest=MANIFEST, frozen_inputs=TRAIN_INPUTS,
        task=args.task, raw_episode_id=args.raw_episode_id, method=args.method, split='train', out=args.out,
        replica=0, exploration_seed=None, development_demo=True, record_demo=False, demo_only=False,
        suite=None, suite_lock=None, validation_release=None, source_inputs=None, check=args.check)
    from . import rml_case_preflight_v742 as substrate
    prepared = substrate.prepare(root, forwarded, root/LEGACY_WORKER)
    from .author_aligned_geometry_integration_v1114b import binding as actuator_binding
    protocol = dict(schema=SCHEMA, source_overlay_lock=overlay_pointer,
        actual_worker=pointer(root, WORKER), actual_preparer=pointer(root, SOURCE),
        inherited_consumer=pointer(root, LEGACY_RUNTIME),
        original743_prepare_worker_check='original_substrate_only_not_new_worker_approval',
        original_source_and_model_preflight=True, original_745_execution_acceptance_granted=False,
        integrated_components=dict(module=INTEGRATION_MODULE, function='integrated_components',
                                  integration_complete=overlay['integration_complete']),
        author_aligned_executor=actuator_binding(root, prepared['config']['baseline_repo']),
        unchanged_action_outcome_sampling=substrate.outcome_protocol_binding(root),
        formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        default_collision_diagnostic_limit=0, additional_native_outcome_or_public_getter_calls=0,
        actual_joint_position_clipping=True, velocity_clipping='original_v6_counterfactual_only',
        pose_action_max_ticks=10, new_gripper_request_on_tenth_tick=True, public_hold_max_ticks=1,
        total_budget_ticks=200, native_physics_collision_and_task_conditions_unchanged=True)
    old_identity = deepcopy(prepared['identity'])
    identity = dict(old_identity); identity.pop('sha256')
    identity.update(execution_protocol_schema=SCHEMA, execution_protocol_sha256=digest(protocol),
        source_overlay_lock_sha256=overlay_pointer['sha256'], original_substrate_identity_sha256=old_identity['sha256'])
    identity['sha256'] = digest(identity)
    prepared.update(identity=identity, source_substrate_identity=old_identity,
                    execution_protocol=protocol, source_overlay=overlay, source_overlay_pointer=overlay_pointer,
                    overlay_selection=chosen, overlay_lock_name=args.overlay_lock)
    return prepared


def verify(root, prepared):
    from . import rml_case_preflight_v742 as substrate
    substrate.verify(root, prepared)
    overlay, reference = read_overlay_lock(root, prepared['overlay_lock_name'])
    require(reference == prepared['source_overlay_pointer'] and overlay == prepared['source_overlay'], 'source overlay changed during case')
    require(digest(prepared['execution_protocol']) == prepared['identity']['execution_protocol_sha256'], 'new execution protocol identity changed')


def require_native_ready(prepared):
    require(prepared['source_overlay']['integration_complete'] is True,
            'integration pending: Native disabled until all sources and CPU evidence are locked')


def annotated_output(prepared, path, value):
    """Tag a new result BEFORE its only write; never rewrite old result files."""
    value = deepcopy(value)
    require(type(value) is dict, 'dictionary result artifact required')
    value.update(execution_protocol=prepared['execution_protocol'],
        source_overlay_lock=prepared['source_overlay_pointer'],
        original_745_execution_acceptance_granted=False, formal_score_contribution=False,
        new_protocol_formal_VAL_TEST_qualification=False)
    if Path(path).name == 'case.json':
        value['inherited_consumer_record_version'] = value.pop('version', None)
        value['version'] = SCHEMA + '_case'
        if 'full16_frozen_execution' in value:
            value['inherited_TRAIN_source_execution_record'] = value.pop('full16_frozen_execution')
    return value


@contextmanager
def runtime_scope(root, prepared, runtime):
    """Owned factory lifetime covers the original consumer's entire run/cleanup."""
    require_native_ready(prepared); verify(root, prepared)
    components = importlib.import_module(INTEGRATION_MODULE)
    from . import rml_component_extensions_v708 as original_actor
    original_write = runtime.atomic_json
    with ExitStack() as lifetime:
        def geometry(current, env, descriptions):
            require(current is prepared, 'geometry factory escaped the prepared case')
            actor, proof = original_actor.create_actor(root, env, current['config'], current['args'].method,
                descriptions, domain=current['component_domain'])
            frontend, policy, executor = lifetime.enter_context(
                components.integrated_components(root, current['config'], current['args'].task, actor))
            proof = dict(proof, new_execution_protocol_schema=SCHEMA,
                         source_overlay_lock=current['source_overlay_pointer'])
            return actor, proof, frontend, policy, executor

        def write(path, value):
            require(Path(path).parent == prepared['out'], 'consumer output escaped new case namespace')
            if Path(path).name == 'case.json':
                verify(root, prepared)
            return original_write(path, annotated_output(prepared, path, value))

        lifetime.enter_context(patch.object(runtime, '_geometry', geometry))
        lifetime.enter_context(patch.object(runtime, 'atomic_json', write))
        yield
    verify(root, prepared)
