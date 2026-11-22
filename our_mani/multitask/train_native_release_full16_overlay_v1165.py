"""Fixed original TRAIN full16 author first-release development repair.

The complete frozen1154 source/model/route graph is inherited. Core1164 retains the original558 basketball and602 Buzz first-release gates. Other core behavior,
block1153 and PhysMani744 remain inherited. No new threshold or formal admission.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import train_block_preclose_full16_overlay_v1154 as comparison_parent
from .evaluation_roster16_v283 import TASKS, METHODS
from .joint16_tracks_v31 import BLOCKS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'our_mani/multitask/train_native_release_full16_overlay_v1165.py'
WORKER = 'scripts/run_TRAIN_native_release_full16_overlay_v1165_r10.py'
TEST_SOURCE = 'tests/test_train_native_release_full16_overlay_v1165.py'
SCHEMA = 'r10_full16_fixed_TRAIN_native_release_overlay_v1165'
LOCK_SCHEMA = SCHEMA + '_source_lock'
CPU_EVIDENCE_SCHEMA = SCHEMA + '_CPU_integration_evidence'
LEGACY_WORKER = 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
LEGACY_RUNTIME = 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
ROUTES = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json'
ROUTES_SHA = '5ea2190dae2b986dc6f7d1b208c0372e2750da68e610639963b697d055e720c0'
MANIFEST = 'artifacts/framework/protocol/train_manifest.json'
MANIFEST_SHA = 'c414e70ef4a1daf5ea9b2c8adb247576197a9b2decde97479f9006cfa3ca23a1'
TRAIN_INPUTS = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
TRAIN_INPUTS_SHA = 'd497facb879abb961cc2d2812e9233bccf43f35c927d71e1a1caa6e3f1e9932e'
CHECKPOINT = dict(path='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
                  sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
NAME = 'TRAIN_native_release_full16_v1165'
PARENT_OVERLAY_LOCK = 'artifacts/framework/multitask_r10/TRAIN_block_preclose_full16_v1154_setup/source_overlay_lock.json'
PARENT_OVERLAY_LOCK_SHA = '0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9'
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
RUNTIME_PREFIX = 'runtime/framework/' + NAME + '/cases/'
RUNTIME_SETUP_PREFIX = 'runtime/framework/' + NAME + '_setup/'
# One explicitly selected matched TRAIN candidate. This is not formal admission.
CORE_INTEGRATION_MODULE = 'our_mani.multitask.author_native_release_core_v1164'
CORE_INTEGRATION_SOURCE = CORE_INTEGRATION_MODULE.replace('.', '/') + '.py'
BASKETS=('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
BUZZ_TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
RELEASE_TASKS=(*BASKETS,*BUZZ_TASKS)
CORE_TEST='tests/test_author_native_release_core_v1164.py'
CORE_POLICY_VARIANT=dict(comparison_parent='our_mani.multitask.author_ring_release_integration_v1140',
    changed_factor='retained_original_author_first_release_qualification',
    applies_to='basketball_deposit_and_Buzz_release_geometry_closed_to_requested_open_only',other_core_tasks='exact_parent1140_delegate',
    block1153_and_baseline744_unchanged=True,new_numeric_thresholds=False)
BLOCK_MODULE = 'our_mani.multitask.retained_preclose_block_integration_v1153'
BLOCK_SOURCE = BLOCK_MODULE.replace('.', '/')+'.py'
BLOCK_TEST = 'tests/test_retained_preclose_block_integration_v1153.py'
BLOCK_PUBLIC_SOURCES = tuple('our_mani/'+name for name in (
    'multitask/physmani_block_policy_v538.py','multitask/physmani_block_receipt_v541.py',
    'multitask/physmani_block_record_v545.py','multitask/block_held_support_v298.py',
    'multitask/joint16_block_context_v97.py','multitask/joint16_tracks_v31.py',
    'benchmark_perception.py','goal_perception.py','contracts.py'))
BLOCK_POLICY_VARIANT = dict(comparison_parent='our_mani.multitask.author_aligned_block_integration_v1120',
    changed_factor='retained_original_block_first_preclose_witness',
    other_block_openings_and_original_frontend_phase_rules_unchanged=True,
    new_numeric_thresholds=False, block_physical_path_unchanged_relative_to1154=True,core_factory_selected_separately=True,physmani744_unchanged=True)
OVERLAY_SOURCES = tuple(dict.fromkeys((*comparison_parent.OVERLAY_SOURCES,
    SOURCE, WORKER, TEST_SOURCE, comparison_parent.SOURCE, comparison_parent.WORKER,
    comparison_parent.TEST_SOURCE, PARENT_OVERLAY_LOCK, BLOCK_SOURCE, BLOCK_TEST,*BLOCK_PUBLIC_SOURCES,CORE_INTEGRATION_SOURCE,CORE_TEST,
    'our_mani/multitask/physmani_basketball_release_v558.py',
    'our_mani/multitask/physmani_buzz_curve_v602.py',
    'our_mani/multitask/repaired_readiness_v1144.py')))
FROZEN_PINS = dict(comparison_parent.FROZEN_PINS, **{
    CORE_INTEGRATION_SOURCE:'14f005b3c5b6b96a63095a8560ec9192624cd3566a59f6e13b36b2a43116ffbd',
    CORE_TEST:'5ec13e57a092297ab888c47a2f976860a51209d257008b833539509f84f23c41',
    PARENT_OVERLAY_LOCK:PARENT_OVERLAY_LOCK_SHA,
    comparison_parent.SOURCE:'8a4b8a2019a979b4483efce9b39f5ebd7671450ed94771dd897fab04d29ca9c0',
    comparison_parent.WORKER:'7e64cb2b7fa66017aeeeca3a7f6d4a684910ee6c1d4093844f46aedd7d2b4d2f',
    comparison_parent.TEST_SOURCE:'1981d432b3ac106adbd38f4682aa742ef2e3f9374324c11fd307c50d114a76a0',
})
PRIOR_OWNER='runtime/framework/repaired_policy_owner_v1149/train/conservative_v1159_train192'
PRIOR_PLAN_SHA='a547505b01632a2eb5cc55ab5d7c4d7202bf2b918615b63321835c2a1711e046'
PRIOR_STOP_SCHEMA='r10_actual_controlled_repair_owner_exit_v1164'
PRIOR_STOP_SHA='2b338d68ad079facad2703d0d1393980f86611d2a64095bf75b77d0eddff5bad'


def require(value, message):
    if not value:
        raise ValueError('TRAIN1165: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _path(root, name, *, prefix=None, must_exist=True):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and str(Path(name)) == name, 'canonical project-relative path required')
    if prefix is not None:
        require(name.startswith(prefix), 'new explicit TRAIN1165 namespace required')
    path = Path(root)/name
    require(path.resolve(strict=must_exist) == path, 'symlink/alias inputs and outputs are forbidden')
    return path


def raw(root, name, expected=None):
    path = _path(root, name)
    before = path.stat(); value = path.read_bytes(); after = path.stat()
    stamp = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    require(stamp(before) == stamp(after), 'input changed during read: '+name)
    actual = hashlib.sha256(value).hexdigest()
    require(expected is None or expected == actual, 'input SHA changed: '+name)
    return value


def pointer(root, name):
    return dict(path=name, sha256=hashlib.sha256(raw(root, name)).hexdigest())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _cpu_evidence(root, reference, files):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'explicit CPU evidence pointer required')
    require(reference['path'].startswith((SETUP_PREFIX, RUNTIME_SETUP_PREFIX)), 'new TRAIN1165 CPU evidence namespace required')
    evidence = json.loads(raw(root, reference['path'], reference['sha256']))
    require(evidence.get('schema') == CPU_EVIDENCE_SCHEMA and evidence.get('complete') is True
            and type(evidence.get('tests_exit_code')) is int and evidence.get('tests_exit_code') == 0
            and evidence.get('native_started') is False and evidence.get('GPU_started') is False
            and evidence.get('source_hashes') == files
            and evidence.get('parent_source_overlay_lock') == dict(path=PARENT_OVERLAY_LOCK,sha256=PARENT_OVERLAY_LOCK_SHA),
            'complete CPU integration evidence must bind every actual overlay source')
    return evidence


def _substrate_pins(root):
    for name, expected in ((MANIFEST, MANIFEST_SHA), (ROUTES, ROUTES_SHA), (TRAIN_INPUTS, TRAIN_INPUTS_SHA)):
        raw(root, name, expected)
    for name, expected in FROZEN_PINS.items():
        raw(root, name, expected)
    locked,reference=comparison_parent.read_overlay_lock(root,PARENT_OVERLAY_LOCK)
    require(reference==dict(path=PARENT_OVERLAY_LOCK,sha256=PARENT_OVERLAY_LOCK_SHA)
        and locked['integration_complete'] is True,'complete frozen1154 comparison parent required')


def _core_binding(files):
    return dict(module=CORE_INTEGRATION_MODULE, function='integrated_components',
                source=CORE_INTEGRATION_SOURCE, sha256=files.get(CORE_INTEGRATION_SOURCE),
                matched_policy_variant=deepcopy(CORE_POLICY_VARIANT))


def _block_binding(files):
    return dict(module=BLOCK_MODULE,function='integrated_block_components',source=BLOCK_SOURCE,
        sha256=files.get(BLOCK_SOURCE),matched_policy_variant=deepcopy(BLOCK_POLICY_VARIANT))


def applied_phase_factor(task,method):
    if method!='physmani' and task in RELEASE_TASKS:
        return 'retained_original_author_first_release_qualification'
    return comparison_parent.applied_phase_factor(task,method)


def phase_component_binding(root, config, task, method):
    """Validate the source graph and precise task rule before Native startup."""
    if method == 'physmani':
        return None
    if task in BLOCKS:
        components=importlib.import_module(BLOCK_MODULE)
        return dict(block_integration=components.binding(root,config['baseline_repo']))
    components = importlib.import_module(CORE_INTEGRATION_MODULE)
    result = dict(integration=components.binding(root, config['baseline_repo']), task_rule=None)
    if task in RELEASE_TASKS:
        result['task_rule']=components.task_rule_binding(root,config,task)
        if task in BUZZ_TASKS:
            curve=importlib.import_module('our_mani.multitask.public_buzz_approach_curve_v1130')
            result['unchanged_Buzz_approach_rule']=curve.binding(root,config,task)
    elif task in ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed'):
        result['task_rule'] = components.ring_rule_binding(root, config, task)
    return result


def write_overlay_lock(root, name, *, integration_complete=False, integration_cpu_evidence=None):
    """Write a pending lock by default; never invent absent integration hashes."""
    root = Path(root).resolve(); path = _path(root, name, prefix=SETUP_PREFIX, must_exist=False)
    require(type(integration_complete) is bool, 'explicit integration state required')
    _substrate_pins(root)
    files = {}; unpinned = []
    for source in OVERLAY_SOURCES:
        if (root/source).exists(): files[source] = pointer(root, source)['sha256']
        else: unpinned.append(source)
    require(not integration_complete or not unpinned, 'all integration sources must exist before Native activation')
    if integration_complete:
        _cpu_evidence(root, integration_cpu_evidence, files)
    else:
        require(integration_cpu_evidence is None, 'pending integration cannot claim CPU completion')
    value = dict(schema=LOCK_SCHEMA, required_sources=list(OVERLAY_SOURCES), files=files,
        core_factory=_core_binding(files),block_factory=_block_binding(files),
        parent_source_overlay_lock=dict(path=PARENT_OVERLAY_LOCK,sha256=PARENT_OVERLAY_LOCK_SHA),
        unpinned_sources=unpinned, integration_complete=integration_complete,
        integration_cpu_evidence=integration_cpu_evidence,
        original_TRAIN_input_lock=dict(path=TRAIN_INPUTS, sha256=TRAIN_INPUTS_SHA),
        original_TRAIN_manifest=dict(path=MANIFEST, sha256=MANIFEST_SHA),
        original_745_routes=dict(path=ROUTES, sha256=ROUTES_SHA),
        original_745_execution_acceptance_granted=False, formal_VAL_TEST_qualification=False,
        automatic_retry=False, TEST_used_for_fitting=False)
    write_json(path, value)
    return pointer(root, name)


def read_overlay_lock(root, name):
    _path(root, name, prefix=SETUP_PREFIX)
    value = json.loads(raw(root, name)); files = value.get('files')
    require(value.get('schema') == LOCK_SCHEMA and value.get('required_sources') == list(OVERLAY_SOURCES),
            'explicit new full16 overlay schema and source roster required')
    require(type(files) is dict and set(files) <= set(OVERLAY_SOURCES)
            and value.get('unpinned_sources') == [s for s in OVERLAY_SOURCES if s not in files], 'honest overlay source map required')
    require(value.get('core_factory') == _core_binding(files), 'selected core factory module/source binding changed')
    require(value.get('block_factory') == _block_binding(files), 'selected block1153 factory/source binding changed')
    require(value.get('parent_source_overlay_lock') == dict(path=PARENT_OVERLAY_LOCK,sha256=PARENT_OVERLAY_LOCK_SHA),
            'frozen1154 comparison-parent source graph required')
    require(value.get('original_TRAIN_input_lock') == dict(path=TRAIN_INPUTS, sha256=TRAIN_INPUTS_SHA)
            and value.get('original_TRAIN_manifest') == dict(path=MANIFEST, sha256=MANIFEST_SHA)
            and value.get('original_745_routes') == dict(path=ROUTES, sha256=ROUTES_SHA)
            and value.get('original_745_execution_acceptance_granted') is False
            and value.get('formal_VAL_TEST_qualification') is False and value.get('automatic_retry') is False
            and value.get('TEST_used_for_fitting') is False, 'source-only substrate must not grant qualification')
    _substrate_pins(root)
    for source, expected in files.items(): raw(root, source, expected)
    require(type(value.get('integration_complete')) is bool, 'explicit integration state required')
    if value['integration_complete']:
        require(not value['unpinned_sources'], 'Native cannot use unpinned sources')
        _cpu_evidence(root, value.get('integration_cpu_evidence'), files)
    else:
        require(value.get('integration_cpu_evidence') is None, 'pending integration cannot claim CPU completion')
    return value, pointer(root, name)


def selection(root, args):
    """Select exact immutable745 route and original first4, never a new roster."""
    validate_modes(args)
    require(args.task in TASKS and args.method in METHODS, 'exactly16 TRAIN tasks and three authorized methods')
    routes = json.loads(raw(root, ROUTES, ROUTES_SHA))
    rows = routes.get('routes', [])
    keys = [(r['task'], r['method']) for r in rows]
    require(len(keys) == 48 and len(set(keys)) == 48
            and set(keys) == {(t, m) for t in TASKS for m in METHODS}, 'complete unique original745 route grid required')
    route = next(r for r in rows if (r['task'], r['method']) == (args.task, args.method))
    require(route.get('GPU_required') is (args.method == 'physmani'), 'original method GPU allocation required')
    require(route.get('frozen_inputs') == dict(path=TRAIN_INPUTS, sha256=TRAIN_INPUTS_SHA), 'original745 source/model lock required')
    for field, name in (('worker', LEGACY_WORKER), ('consumer', LEGACY_RUNTIME),
                        ('source_preflight', 'our_mani/multitask/rml_case_preflight_v742.py')):
        require(route.get(field) == dict(path=name, sha256=FROZEN_PINS[name]), 'original745 route dependency changed: '+field)
    for field in ('configuration', 'frozen_inputs'):
        ref = route[field]
        require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact original route pointer required')
        supplied = getattr(args, 'config' if field == 'configuration' else 'frozen_inputs', None)
        require(supplied is None or supplied == ref['path'], 'argv differs from original745 route: '+field)
        raw(root, ref['path'], ref['sha256'])
    manifest = json.loads(raw(root, MANIFEST, MANIFEST_SHA))
    fixed = [r['raw_episode_id'] for r in manifest['episodes'] if r['task'] == args.task][:4]
    require(manifest.get('split') == 'train' and len(fixed) == 4 and args.raw_episode_id in fixed,
            'original TRAIN first four parents only; no replacement or outcome selection')
    config = json.loads(raw(root, route['configuration']['path'], route['configuration']['sha256']))
    if args.method != 'physmani':
        require(config.get('context_checkpoint') == CHECKPOINT, 'original shared671 model identity required')
    return dict(task=args.task, method=args.method, raw_episode_id=args.raw_episode_id,
        fixed_parent_index=fixed.index(args.raw_episode_id), original_first_four=fixed,
        original_route=route, original_route_sha256=digest(route),
        original_routes=dict(path=ROUTES, sha256=ROUTES_SHA))


def prepare(root, args):
    root = Path(root).resolve();validate_modes(args)
    prior_stop=(require_prior_stop(root,getattr(args,'prior_stop',None)) if args.execute else None)
    chosen = selection(root, args)
    overlay, overlay_pointer = read_overlay_lock(root, args.overlay_lock)
    require(all(s in overlay['files'] for s in (SOURCE, WORKER, TEST_SOURCE)), 'actual new worker/preparer/CPU tests must be pinned')
    out = _path(root, args.out, prefix=ARTIFACT_PREFIX, must_exist=False)
    require(not out.exists(), 'fresh output required; no overwrite or retry')
    route = chosen['original_route']
    forwarded = SimpleNamespace(config=route['configuration']['path'], manifest=MANIFEST,
        frozen_inputs=route['frozen_inputs']['path'], task=args.task, raw_episode_id=args.raw_episode_id,
        method=args.method, split='train', out=args.out, replica=0, exploration_seed=None,
        development_demo=True, record_demo=False, demo_only=False, suite=None, suite_lock=None,
        validation_release=None, source_inputs=None, check=args.check)
    from . import rml_case_preflight_v742 as substrate
    prepared = substrate.prepare(root, forwarded, root/LEGACY_WORKER)
    branch = 'unchanged744_physmani' if args.method == 'physmani' else ('block1153' if args.task in BLOCKS else 'core_selected_integration')
    protocol = dict(schema=SCHEMA, source_overlay_lock=overlay_pointer, actual_worker=pointer(root, WORKER),
        comparison_parent_source_overlay_lock=dict(path=PARENT_OVERLAY_LOCK,sha256=PARENT_OVERLAY_LOCK_SHA),
        actual_worker_is_comparison_parent1154=False,
        actual_preparer=pointer(root, SOURCE), inherited_consumer=pointer(root, LEGACY_RUNTIME),
        original_route=route, original_route_sha256=chosen['original_route_sha256'],
        original743_prepare_worker_check='original_substrate_only_not_new_worker_approval',
        original_source_and_model_preflight=True, original_745_execution_acceptance_granted=False,
        execution_branch=branch, integration_complete=overlay['integration_complete'],
        core_factory=_core_binding(overlay['files']),
        core_changed_relative_to1154=(args.task in RELEASE_TASKS and args.method!='physmani'),
        block_changed_relative_to1154=False,baseline_changed_relative_to1154=False,
        public_phase_component_binding=phase_component_binding(root, prepared['config'], args.task, args.method),
        changed_factor=applied_phase_factor(args.task, args.method),
        release_and_other_opening_dispatch=('retained_original_author558_or602_first_release_qualification'
            if args.task in RELEASE_TASKS and args.method!='physmani' else
            ('ring_original_author_wp4_open_without1129_preseating_guard'
             if applied_phase_factor(args.task, args.method) == 'author_wp4_release_with_confirmed_semantic_commitment'
             else 'unchanged_for_selected_branch')),
        block_factory=_block_binding(overlay['files']),
        unchanged_action_outcome_sampling=substrate.outcome_protocol_binding(root),
        formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        default_collision_diagnostic_limit=0, additional_native_outcome_or_public_getter_calls=0,
        baseline_algorithm_unchanged=True, actual_joint_position_clipping=True,
        velocity_clipping='original_v6_counterfactual_only', pose_action_max_ticks=10,
        new_gripper_request_on_tenth_tick=True, public_hold_max_ticks=1, total_budget_ticks=200,
        native_physics_collision_and_task_conditions_unchanged=True)
    old_identity = deepcopy(prepared['identity']); identity = dict(old_identity); identity.pop('sha256')
    identity.update(execution_protocol_schema=SCHEMA, execution_protocol_sha256=digest(protocol),
        source_overlay_lock_sha256=overlay_pointer['sha256'], original_substrate_identity_sha256=old_identity['sha256'])
    identity['sha256'] = digest(identity)
    prepared.update(identity=identity, source_substrate_identity=old_identity, execution_protocol=protocol,
        source_overlay=overlay, source_overlay_pointer=overlay_pointer, overlay_selection=chosen,
        overlay_lock_name=args.overlay_lock,prior_stop=prior_stop,prior_stop_name=getattr(args,'prior_stop',None))
    return prepared


def verify(root, prepared):
    from . import rml_case_preflight_v742 as substrate
    substrate.verify(root, prepared)
    overlay, reference = read_overlay_lock(root, prepared['overlay_lock_name'])
    require(reference == prepared['source_overlay_pointer'] and overlay == prepared['source_overlay'], 'source overlay changed during case')
    require(digest(prepared['execution_protocol']) == prepared['identity']['execution_protocol_sha256'], 'execution protocol identity changed')


def require_native_ready(prepared):
    require(prepared['source_overlay']['integration_complete'] is True,
            'integration pending: Native disabled until all sources and CPU evidence are locked')
    require(type(prepared.get('prior_stop')) is dict,'actual controlled old-owner exit proof required')


def annotated_output(prepared, path, value):
    value = deepcopy(value); require(type(value) is dict, 'dictionary result artifact required')
    value.update(execution_protocol=prepared['execution_protocol'], source_overlay_lock=prepared['source_overlay_pointer'],
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
    """Adapt only owned geometry instances; preserve original baseline route."""
    require_native_ready(prepared); verify(root, prepared)
    require(require_prior_stop(root,prepared['prior_stop_name'])==prepared['prior_stop'],
            'controlled predecessor exit evidence changed before Native scope')
    original_write = runtime.atomic_json
    with ExitStack() as lifetime:
        if prepared['args'].method != 'physmani':
            def geometry(current, env, descriptions):
                require(current is prepared, 'geometry factory escaped the prepared case')
                if current['args'].task in BLOCKS:
                    components = importlib.import_module(BLOCK_MODULE)
                    frontend, policy, executor = lifetime.enter_context(components.integrated_block_components(
                        root, current['config'], current['args'].task, current['args'].method, env))
                    proof = dict(actual_shared16_block_adapter=current['model_preflight']['actual_shared16_block_adapter'],
                        component_domain=current['component_domain'].receipt(),
                        new_block_integration=components.binding(root, current['config']['baseline_repo']))
                    actor = None
                else:
                    from . import rml_component_extensions_v708 as original_actor
                    actor, proof = original_actor.create_actor(root, env, current['config'], current['args'].method,
                        descriptions, domain=current['component_domain'])
                    components = importlib.import_module(CORE_INTEGRATION_MODULE)
                    frontend, policy, executor = lifetime.enter_context(components.integrated_components(
                        root, current['config'], current['args'].task, actor))
                proof = dict(proof, new_execution_protocol_schema=SCHEMA,
                             source_overlay_lock=current['source_overlay_pointer'])
                return actor, proof, frontend, policy, executor
            lifetime.enter_context(patch.object(runtime, '_geometry', geometry))

        def write(path, value):
            require(Path(path).parent == prepared['out'], 'consumer output escaped new case namespace')
            if Path(path).name == 'case.json': verify(root, prepared)
            return original_write(path, annotated_output(prepared, path, value))
        lifetime.enter_context(patch.object(runtime, 'atomic_json', write))
        yield
    verify(root, prepared)


def validate_modes(args):
    require(getattr(args,'split','train')=='train','TRAIN only; refuse VAL/TEST before source reads')
    require(getattr(args,'task',None) in TASKS and getattr(args,'method',None) in METHODS,'exact original16/three methods')
    require(type(getattr(args,'check',None)) is bool and type(getattr(args,'execute',None)) is bool
        and args.check != args.execute,'one explicit --check or --execute required')


def require_prior_stop(root,name):
    # Read only Root control metadata; never convert this incomplete cohort to readiness.
    require(name==PRIOR_OWNER+'/controlled_stop_receipt.json','exact Root controlled-stop receipt path required')
    plan=json.loads(raw(root,PRIOR_OWNER+'/plan.json',PRIOR_PLAN_SHA))
    launch=json.loads(raw(root,PRIOR_OWNER+'/launch.json'))
    status=json.loads(raw(root,PRIOR_OWNER+'/status.json'));failure=json.loads(raw(root,PRIOR_OWNER+'/failure.json'))
    stop=json.loads(raw(root,PRIOR_OWNER+'/STOP'));receipt=json.loads(raw(root,name,PRIOR_STOP_SHA))
    def same_ref(value,path,checksum=None):
        return type(value) is dict and set(value)<= {'root','path','sha256'} and value.get('root','project')=='project'            and value.get('path')==path and value.get('sha256')==(checksum if checksum else pointer(root,path)['sha256'])
    require(receipt.get('schema')==PRIOR_STOP_SCHEMA and receipt.get('actual_tool_exit_observed') is True
        and receipt.get('controlled_queue_stop') is True and receipt.get('controlled_stop_completed') is True
        and receipt.get('task_goal_paused') is False and receipt.get('natural_completion') is False
        and receipt.get('cohort_complete') is False and type(receipt.get('returncode')) is int and receipt['returncode']==1
        and receipt.get('qualification_granted') is False and receipt.get('Native_failure_inferred') is False
        and receipt.get('infrastructure_failure_inferred') is False,'Root actual controlled exit1 metadata required')
    require(same_ref(receipt.get('plan'),PRIOR_OWNER+'/plan.json',PRIOR_PLAN_SHA)
        and all(same_ref(receipt.get(key),PRIOR_OWNER+'/'+filename) for key,filename in
                (('status','status.json'),('failure','failure.json'),('stop','STOP'))), 'exact hash-bound controlled owner metadata')
    from .repaired_readiness_v1144 import _process,_not_live
    process=_process(launch.get('process'))
    require(receipt.get('process')==process and _not_live(process),'original owner PID/start must actually be dead')
    failed=status.get('failed');completed=status.get('completed');jobs=plan.get('jobs')
    require(status.get('stage')=='failed' and status.get('active') is None and failure==status
        and type(failed) is list and len(failed)==1 and failed[0].get('active') is None
        and failed[0].get('exception_type')=='ValueError'
        and failed[0].get('message')=='owner1149: STOP before next fixed source'
        and failed[0].get('native_failure_inferred') is False and failed[0].get('automatic_retry') is False,
        'only clean before-next-source queue STOP is accepted; child infra is not controlled completion')
    require(type(jobs) is list and len(jobs)==192 and type(completed) is list and 0<=len(completed)<192
        and type(receipt.get('completed')) is int and receipt['completed']==len(completed)
        and type(receipt.get('expected_cases')) is int and receipt['expected_cases']==192
        and type(status.get('attempted')) is int and status['attempted']==len(completed), 'strict incomplete192 prefix count')
    key=lambda row:(row.get('task'),row.get('raw_episode_id'),row.get('method'))
    require([key(row) for row in completed]==[key(row) for row in jobs[:len(completed)]]
        and all(type(row.get('attempt')) is int and row['attempt']==1 for row in completed), 'all completed cases are original plan prefix without retry')
    require(stop.get('schema')=='r10_root_controlled_queue_stop_for_confirmed_repair_20261024'
        and stop.get('requested_by')=='root' and type(stop.get('reason')) is str and bool(stop['reason'])
        and stop.get('current_child_must_finish') is True and stop.get('task_goal_paused') is False
        and stop.get('Native_outcome_inferred') is False and stop.get('qualification_granted') is False,
        'actual Root controlled stamp preserves the current child and all outcomes')
    return pointer(root,name)
