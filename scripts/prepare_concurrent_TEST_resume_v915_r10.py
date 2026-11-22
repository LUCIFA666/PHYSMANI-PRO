"""Freeze one qualified concurrency continuation; never launch any work.

Only explicit --prepare performs admission and writes the fresh setup. The
original826 suite/source maps keep their exact keys and SHAs. All old clean
native outcomes, including False and zero ticks, are retained. Full original
scientific byte-verification gates are run only by that future explicit call.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, digest, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.frozen_input_edges_v463 import FrozenInputIndex
from our_mani.multitask.common_rml_protocol_v530 import fingerprint
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as owner
from scripts import audit_concurrent_resumed_formal_TEST_v906_r10 as auditor
from scripts import run_batched_formal_driver_v901_r10 as bootstrap
from scripts import run_concurrent_TRAIN_resource_canary_v903_r10 as canary
from scripts import report_concurrent_TEST_halfhour_v909_r10 as reporter
from scripts import audit_environment_resumed_formal_VAL_v824_r10 as environment_auditor

SOURCE = 'scripts/prepare_concurrent_TEST_resume_v915_r10.py'
SETUP = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
CONTROL = SETUP+'/resume_control.json'
LOCK = SETUP+'/resume_inputs.json'
BATCH_CONTROL = SETUP+'/formal_batch_control.json'
BATCH_LOCK = SETUP+'/formal_batch_inputs.json'
ORIGINAL_SETUP = 'artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup'
SUITE = ORIGINAL_SETUP+'/suite.json'
SUITE_LOCK = ORIGINAL_SETUP+'/frozen_inputs.json'
SOURCE_INPUTS = ORIGINAL_SETUP+'/test_source_inputs.json'
VAL_RELEASE = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/VAL_release.json'
VAL_RELEASE_SHA = '03b7ba8677e333512d5396411248784497ac72a82ffc68206865b68e2b0a577b'
PREVIOUS = owner.PREVIOUS
CONTROLLER = 'scripts/continue_concurrent_TEST_v907_r10.py'
QUALIFIER = 'scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py'
QUALIFICATION = 'runtime/framework/concurrent_TRAIN_resource_qualification_v904.json'
QUALIFICATION_LAUNCH = 'runtime/framework/concurrent_TRAIN_resource_audit_v904/launch.json'
STANDING = 'runtime/framework/latest_executor_continuation_v817.json'
AUTHORIZATION = 'runtime/framework/user_concurrency_authorization_v905.json'
ENVIRONMENT_PATHS = dict(environment_wrapper='scripts/run_frozen_system_environment_v820_r10.py',
    environment_manifest='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json',
    environment_namespace_proof='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json')
ENVIRONMENT_EXTRA = (
    'artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/frozen_environment_receipt_v820.json',
    'artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_library_namespace_inventory_v820.json')
PROFILE = deepcopy(canary.PROFILE)
STOP_PATHS = ('runtime/framework/multitask_r10_STOP', owner.RUNTIME+'/STOP',
    'runtime/framework/ordered_concurrent_TEST_v907/STOP', PREVIOUS+'/STOP', owner.PREVIOUS_CONTROLLER+'/STOP',
    'runtime/framework/ordered_resume_VAL_TEST_v775/STOP', 'runtime/framework/formal_VAL_resume_v772/STOP',
    'runtime/framework/formal_VAL_env_resume_v823/STOP')
TESTS = ('tests/test_frozen_lookup_batch_v900.py', 'tests/test_batched_formal_driver_v901.py',
    'tests/test_concurrent_resumed_formal_TEST_owner_v902.py', 'tests/test_concurrent_TRAIN_resource_canary_v903.py',
    'tests/test_concurrent_TRAIN_resource_audit_v904.py', 'tests/test_concurrent_resumed_formal_TEST_audit_v906.py',
    'tests/test_concurrent_TEST_controller_v907.py', 'tests/test_concurrent_TEST_reporter_v909.py',
    'tests/test_prepare_concurrent_TEST_resume_v915.py')
REQUIRED_SOURCES = tuple(sorted({SOURCE, CONTROLLER, QUALIFIER,
    'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py', *TESTS, *auditor.SOURCES,
    *bootstrap.SOURCES, *canary.SOURCES, *reporter.SOURCES}))


def require(value, message):
    if not value:
        raise ValueError('concurrent resume preparation: '+message)


def stopped(root=ROOT):
    return any((Path(root)/name).exists() for name in STOP_PATHS)


def merge_maps(*maps):
    """Preserve each original declaration; reject every differing key SHA."""
    result = {}
    for mapping in maps:
        require(isinstance(mapping, dict) and mapping, 'explicit nonempty original input maps required')
        for name, checksum in mapping.items():
            require(isinstance(name, str) and name and isinstance(checksum, str)
                and len(checksum) == 64 and not set(checksum)-set('0123456789abcdef'), 'strict frozen file/SHA map required')
            require(name not in result or result[name] == checksum, 'conflicting original frozen key: '+name)
            result[name] = checksum
    return result


def add_frozen_file(root, files, path, expected=None):
    root = Path(root).resolve()
    path = root/path
    canonical = path.resolve(strict=True)
    require(canonical.is_relative_to(root) and path == canonical, 'canonical project supplement required')
    checksum = file_sha(path)
    require(expected is None or checksum == expected, 'supplement differs from expected bytes: '+str(path))
    names = (str(canonical), str(canonical.relative_to(root)))
    declared = [name for name in names if name in files]
    require(len(declared) <= 1, 'ambiguous absolute/relative canonical declaration: '+str(path))
    for name in declared:
        require(files[name] == checksum, 'conflicting supplementary SHA: '+name)
    if not declared:
        files[str(canonical)] = checksum
    return dict(path=str(canonical.relative_to(root)), sha256=checksum)


def require_canonical_consistency(graph, files):
    """Original unused aliases may remain; differing canonical SHAs cannot.

    This does not authorize a pointer lookup. Original728/900 still require
    exactly one canonical declaration for every path they actually consume.
    """
    require(all(len({files[name] for name in names}) == 1 for names in graph.by_path.values()),
            'conflicting canonical alias SHAs in merged frozen graph')
    return True


def partition(plan, status, terminal):
    rows = plan.get('jobs', [])
    done = status.get('completed')
    require(isinstance(done, dict), 'original completed map required')
    require(isinstance(rows, list) and all(not any(key in row for key in owner.ROW_PROVENANCE) for row in rows),
            'only the actual original753 plan may be adopted, never another resume tier')
    completed = sorted(done)
    remaining = sorted({row['job_id'] for row in rows}-set(completed))
    control = dict(schema=owner.CONTROL_SCHEMA, split='test', expected_cases=1920, source_count=640,
        completed_job_ids=completed, remaining_job_ids=remaining,
        completed_key_sha256=digest(completed), remaining_key_sha256=digest(remaining),
        policy_sha256=plan.get('policy_sha256'), user_authorized_resume=True,
        standing_authorization_reused=True, new_user_event_invented=False, physical_policy_changed=False)
    jobs = owner.require_partition(control, plan, status, terminal)
    require(plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False
        and status.get('native_slots') == status.get('GPU_slots') == 0 and status.get('automatic_retry') is False,
        'exact original unfiltered one-time plan and drained resources required')
    return dict(jobs=jobs, completed=completed, remaining=remaining, policy_sha256=plan['policy_sha256'])


def old_evidence_paths(row):
    key = row['job_id']
    leaves = [str(Path(PREVIOUS)/'CPU'/key/name) for name in
        ('launch.json', 'stdout.log', 'stderr.log', 'terminal.json', 'CPU_preflight.json')]
    leaves += [str(Path(PREVIOUS)/'native'/key/name) for name in ('launch.json', 'stdout.log', 'stderr.log', 'terminal.json')]
    out = owner.option(row['argv'], '--out')
    return [*leaves, str(Path(out)/'case.json'), str(Path(out)/'admission_driver_receipt.json')]


def formal_bootstrap_control(suite):
    return dict(schema=bootstrap.SCHEMA, execution_enabled=True, entry='formal', scientific_policy_changed=False,
        models_or_training_changed=False, argv_bindings={'--split':'test', '--manifest':suite['source_manifests']['test']['path'],
            '--suite':SUITE, '--suite-lock':SUITE_LOCK, '--source-inputs':SOURCE_INPUTS, '--validation-release':VAL_RELEASE})


def require_previous_exit(previous, is_alive):
    for name in ('launch', 'controller_launch'):
        value = previous[name]
        require(type(value.get('pid')) is int and value['pid'] > 0 and type(value.get('start_ticks')) is int
            and value['start_ticks'] > 0 and not is_alive(value), 'old physical owner/controller must have exited')
    state, failure = previous['controller_status'], previous['controller_failure']
    require(state.get('stage') == 'requires_concrete_repair' and state.get('automatic_retry') is False
        and failure.get('complete') is False and failure.get('user_stop') is False
        and failure.get('automatic_retry') is False, 'genuine827 failure/status after clean DRAIN required; no terminal is invented')
    part = partition(previous['plan'], previous['status'], previous['terminal'])
    return dict(previous_owner_and_controller_exited=True,
        identities=[dict(pid=previous[k]['pid'], start_ticks=previous[k]['start_ticks']) for k in ('launch','controller_launch')],
        completed=len(part['completed']), remaining=len(part['remaining']))


def require_resource_proof(value, control, *, audit_launch):
    """An actual independent904 proof, never a903 summary with renamed flags."""
    owner.require_resource_qualification(value, control)
    require(control['resource_qualifier']['path'] == QUALIFIER
        and value.get('actual_independent_audit', {}).get('path') == QUALIFICATION_LAUNCH
        and owner.valid_pointer(value.get('actual_independent_audit'))
        and audit_launch.get('source_sha256') == control['resource_qualifier']['sha256']
        and type(audit_launch.get('pid')) is int and audit_launch['pid'] > 0
        and type(audit_launch.get('start_ticks')) is int and audit_launch['start_ticks'] > 0,
        'real independently launched904 source/PID proof required')
    for key in ('control', 'lock'):
        require(owner.valid_pointer(value.get(key)) and audit_launch.get(key) == value[key], 'actual904 small control/lock differs')
    argv = audit_launch.get('argv', [])
    require(isinstance(argv, list) and argv and (ROOT/argv[0]).resolve() == (ROOT/QUALIFIER).resolve()
        and (ROOT/owner.option(argv, '--control')).resolve() == (ROOT/value['control']['path']).resolve()
        and (ROOT/owner.option(argv, '--lock')).resolve() == (ROOT/value['lock']['path']).resolve()
        and owner.option(argv, '--lock-sha256') == value['lock']['sha256']
        and ('--out' not in argv or owner.option(argv, '--out') == QUALIFICATION), 'actual independent904 command required')
    env = environment_auditor.require_environment_receipt(audit_launch.get('frozen_environment'), control,
        owner_pid=audit_launch['pid'])
    require(env['observer_start_ticks'] == audit_launch['start_ticks'], 'actual904 namespace start differs')
    overlap = value.get('launcher_overlap', {})
    require(overlap.get('independently_swept_all_child_intervals') is True
        and overlap.get('overlap_verified') is True and overlap.get('observed_native_peak') == 4
        and overlap.get('observed_GPU_peak') == 2 and isinstance(overlap.get('intervals'), dict)
        and len(overlap['intervals']) == 6, 'independently swept all six actual child intervals with4/2 overlap required')
    hardware = value.get('hardware_observation', {})
    require(hardware.get('actual_two_compute_contexts_observed') is True
        and type(hardware.get('valid_samples')) is int and hardware['valid_samples'] > 0
        and hardware.get('required_headroom_mib') == 2048
        and type(hardware.get('minimum_sampled_headroom_mib')) is int and hardware['minimum_sampled_headroom_mib'] >= 2048
        and hardware.get('continuous_hardware_bound_claimed') is False,
        'actual two-case GPU contexts and at least2GiB sampled headroom required')
    witness = hardware.get('witness', {})
    require(isinstance(witness.get('jobs'), dict) and len(witness['jobs']) == 2
        and set(witness['jobs']) <= set(overlap['intervals'])
        and all(key.endswith('_physmani') for key in witness['jobs'])
        and all(type(row.get('pid')) is int and row['pid'] > 0 and type(row.get('start_ticks')) is int
            and row['start_ticks'] > 0 for row in witness['jobs'].values())
        and len({(row['pid'],row['start_ticks']) for row in witness['jobs'].values()}) == 2
        and type(witness.get('headroom_mib')) is int and witness['headroom_mib'] >= 2048,
        'actual distinct original PhysMani main-process overlap witness required')
    require(isinstance(value.get('inputs'), dict) and value['inputs']
        and all(owner.valid_pointer(dict(path=name,sha256=checksum)) and qualification_path_allowed(name)
            for name,checksum in value['inputs'].items())
        and value.get('native_failures_retained') is True and value.get('raw_manifest_model_contents_read') is False
        and value.get('wall_clock_contention_invariance_claimed') is False, 'honest source-bound independent resource evidence required')
    return value


def qualification_path_allowed(name):
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        return False
    runtime = Path(canary.RUNTIME)
    path = Path(name)
    if path.is_relative_to(runtime):
        suffix = path.relative_to(runtime).parts
        return ((len(suffix) == 1 and suffix[0] in ('plan.json', 'CPU_owner_launch.json', 'CPU_owner_terminal.json',
            'native_owner_launch.json', 'native_owner_terminal.json', 'run_summary.json', 'native_resource_events.json',
            'native_resource_samples.jsonl')) or (len(suffix) == 3 and suffix[0] in ('CPU','native')
            and suffix[2] in ('launch.json','terminal.json','stdout.log','stderr.log','CPU_preflight.json')))
    if path.is_relative_to(canary.OUTPUT):
        suffix = path.relative_to(canary.OUTPUT).parts
        return len(suffix) == 4 and suffix[0] == canary.TASK and suffix[1] in canary.RAWS and suffix[2] in canary.METHODS \
            and suffix[3] in ('case.json','batch_driver_receipt.json')
    return False


@contextmanager
def idle_resources(env, previous, root=ROOT):
    require(not stopped(root), 'STOP before resource checks')
    require_previous_exit(previous, owner.alive)
    with ExitStack() as stack:
        for name in canary.LOCKS:
            stream = stack.enter_context((Path(root)/name).open('a+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_previous_exit(previous, owner.alive)
        inventory = owner.process_visibility()
        require(inventory.get('host_visibility_available') is True
            and inventory.get('same_uid_inventory_incomplete') is False
            and inventory.get('live_project_native_processes') == [] and inventory.get('ancestor_chain_complete') is True,
            'complete empty host-visible native/controller inventory required')
        actual = owner.environment_owner.require_frozen_environment(env)
        require(not stopped(root), 'STOP after resource checks')
        yield dict(inventory=inventory, frozen_environment=actual)


def prepare(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT and not stopped(root), 'actual project and live standing continuation required')
    require(not (root/SETUP).exists() and not (root/owner.RUNTIME).exists() and not (root/owner.OUTPUT).exists()
        and not (root/'runtime/framework/ordered_concurrent_TEST_v907').exists(), 'all new namespaces must be fresh; no implicit retry')
    for name in REQUIRED_SOURCES:
        require((root/name).is_file(), 'new source/test is not ready: '+name)
    ledger = auditor.EvidenceLedger(root)
    def pointer(name):
        if not ledger.closed:
            return ledger.pointer(name)
        path = str(root/name)
        require(path in ledger.used, 'cannot add unobserved evidence after ledger close')
        return deepcopy(ledger.used[path]['pointer'])
    def read(name): return ledger.json(name)
    release_pointer = pointer(VAL_RELEASE)
    require(release_pointer['sha256'] == VAL_RELEASE_SHA, 'exact genuine completed824 VAL release required before TEST metadata')
    released = read(VAL_RELEASE)
    owner.require_complete_VAL_release(dict(validation_release=release_pointer,
        policy_sha256=released.get('policy_sha256'), author_outcome_protocol=released.get('author_outcome_protocol')), VAL_RELEASE)
    env = {key:pointer(name) for key,name in ENVIRONMENT_PATHS.items()}
    previous = {name:read(PREVIOUS+'/'+name+'.json') for name in ('launch','plan','status','terminal')}
    previous.update({name:read(owner.PREVIOUS_CONTROLLER+'/'+name.removeprefix('controller_')+'.json')
        for name in ('controller_launch','controller_status','controller_failure')})
    previous_proof = require_previous_exit(previous, owner.alive)
    standing, authorization = read(STANDING), read(AUTHORIZATION)
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and authorization.get('user_authorized_concurrency_change') is True, 'real continuing/concurrency authorization required')
    with idle_resources(env, previous):
        pass
    part = partition(previous['plan'], previous['status'], previous['terminal'])
    suite_pointer, suite_lock_pointer, source_pointer, release_pointer = map(pointer, (SUITE,SUITE_LOCK,SOURCE_INPUTS,VAL_RELEASE))
    suite = read(SUITE)
    seed = dict(policy_sha256=part['policy_sha256'], author_outcome_protocol=suite['author_outcome_protocol'],
        validation_release=release_pointer, resource_profile=PROFILE, resource_qualifier=pointer(QUALIFIER), **env)
    release = owner.require_complete_VAL_release(seed, VAL_RELEASE)
    shared = {name:pointer(name) for name in bootstrap.SOURCES}
    seed['execution_implementation'] = owner.execution_implementation(dict(kind='batched901', pointer=shared[owner.BATCH_ENTRY]), shared)
    qualification = read(QUALIFICATION)
    require(owner.valid_pointer(qualification.get('actual_independent_audit'))
        and qualification['actual_independent_audit']['path'] == QUALIFICATION_LAUNCH,
        'exact independent904 launch pointer required before opening any proof dependency')
    audit_launch = ledger.json(qualification['actual_independent_audit']['path'],
        expected=qualification['actual_independent_audit']['sha256'])
    require_resource_proof(qualification, seed, audit_launch=audit_launch)
    # Only the original TRAIN canary's small source/control map is traversed.
    train_control, train_binding = canary.load_control(qualification['control']['path'], qualification['lock']['path'], qualification['lock']['sha256'])
    require(train_control['execution_implementation'] == seed['execution_implementation']
        and train_control['author_outcome_protocol'] == seed['author_outcome_protocol']
        and train_control['policy_sha256'] == seed['policy_sha256'], 'qualified TRAIN and formal source implementation differ')
    for name, checksum in qualification['inputs'].items():
        require(qualification_path_allowed(name), 'non-canary qualification payload is refused before opening: '+str(name))
        ledger.read(name, expected=checksum)
    for key in ('control','lock'):
        ledger.read(qualification[key]['path'], expected=qualification[key]['sha256'])
    for name, checksum in train_binding['files'].items():
        require(Path(name).is_relative_to(root), 'canary small source/metadata must stay in project')
        ledger.read(name, expected=checksum)
    suite_lock, source_meta = read(SUITE_LOCK), read(SOURCE_INPUTS)
    suite_files, source_files = suite_lock['files'], source_meta['files']
    verify_inputs(root, suite_files)
    manifest, sources, proof = read_formal_source(root, suite, suite_files, 'test', validation_release=release)
    require(proof['policy_sha256'] == seed['policy_sha256'] and suite['author_outcome_protocol'] == outcome_protocol_binding(root),
        'same original frozen physical policy/protocol required')
    require(source_meta.get('schema') == 'r10_full16_post_admission_split_source_lock_v1'
        and source_meta.get('split') == 'test' and source_meta.get('source_count') == len(sources) == 640
        and source_meta.get('policy_sha256') == seed['policy_sha256'] and source_meta.get('TEST_used_for_fitting') is False
        and source_meta.get('source_manifest_sha256') == suite['source_manifests']['test']['sha256'], 'unchanged original640 source lock required')
    verify_inputs(root, source_files)
    files = merge_maps(suite_files, source_files)
    auditor.require_original_plan(previous['plan'], dict(seed, suite=suite_pointer, suite_lock=suite_lock_pointer,
        source_inputs=source_pointer), suite, sources, proof, suite['formal_case_driver'], suite['formal_admission_guard'])
    old_launch = previous['launch']
    protocol = seed['author_outcome_protocol']
    require(old_launch.get('source_sha256') == file_sha(root/'scripts/run_formal_suite16_owner_v753_r10.py')
        and old_launch.get('suite_lock_sha256') == suite_lock_pointer['sha256']
        and old_launch.get('split_source_lock_sha256') == source_pointer['sha256']
        and old_launch.get('source_manifest') == suite['source_manifests']['test']
        and old_launch.get('policy_sha256') == seed['policy_sha256'] and old_launch.get('author_outcome_protocol') == protocol,
        'old753 actually admitted source/suite/protocol differs')
    profile = fingerprint(proof['routes']['common_author_RML_profile'])
    successes = 0
    for key in part['completed']:
        require(not stopped(root), 'STOP during old evidence freeze')
        row = part['jobs'][key]
        for name in old_evidence_paths(row): ledger.read(name)
        cpu_leaf, native_leaf = (str(Path(PREVIOUS)/kind/key) for kind in ('CPU','native'))
        cpu = read(cpu_leaf+'/CPU_preflight.json')
        require(cpu == read(cpu_leaf+'/stdout.log'), 'old actual CPU stdout differs')
        original = deepcopy(row)
        original.update(imported_completed=True, evidence_runtime=PREVIOUS, physical_argv=row['argv'],
            execution_entry=dict(kind='original751', pointer=row['admission_driver']))
        identity = owner.accept_CPU(cpu, original, source_pointer['sha256'], {})
        bindings = {key:row[key] for key in owner.BINDING_KEYS}
        for leaf, check in ((cpu_leaf,True),(native_leaf,False)):
            launch = read(leaf+'/launch.json')
            auditor.require_mixed_launch(launch, original, bindings, check=check, identity=identity,
                stage_owner=old_launch, control=seed)
            auditor.require_clean_terminal(read(leaf+'/terminal.json'))
            require(not owner.alive(dict(pid=launch['launcher_pid'],start_ticks=launch['launcher_start_ticks'])),
                    'every imported original child PID/start must have exited')
        native = read(native_leaf+'/terminal.json')
        status_result = previous['status']['completed'][key]
        auditor.require_clean_terminal(status_result)
        require(all(status_result.get(name) == value for name,value in native.items()), 'old status must preserve actual native terminal')
        out = Path(owner.option(row['argv'], '--out'))
        case, driver = read(out/'case.json'), read(out/'admission_driver_receipt.json')
        cp, dp = pointer(out/'case.json'), pointer(out/'admission_driver_receipt.json')
        require(native.get('case') == status_result.get('case') == cp
            and native.get('admission_driver_receipt') == status_result.get('admission_driver_receipt') == dp,
            'old native/status case or751 receipt byte link differs')
        auditor.original_audit.require_driver_evidence(driver, row['admission_driver'], row['admission_guard'],
            dict(worker=row['case_worker'],consumer=row['native_runtime']), root/out/'case.json', 'test')
        metrics = auditor.require_case_outcome(case, identity, protocol, seed['policy_sha256'], profile)
        require(status_result.get('native_success') is metrics['native_success'] and status_result.get('motion_pass') is metrics['motion_pass'],
                'all old nativeFalse/motionFalse outcomes must stay literal')
        successes += int(metrics['native_success'])
    require(previous['terminal'].get('native_successes') == successes, 'old terminal native total differs')
    for name in REQUIRED_SOURCES: ledger.read(name)
    for name in (*ENVIRONMENT_EXTRA, STANDING, AUTHORIZATION, PREVIOUS+'/DRAIN'): ledger.read(name)
    # Freeze all explicitly consumed project metadata/evidence, with no directory walk.
    for name, value in ledger.used.items(): add_frozen_file(root, files, name, value['pointer']['sha256'])
    for frozen in (suite_files, source_files):
        require(all(files.get(name) == checksum for name,checksum in frozen.items()), 'original input map was amended')
    verify_inputs(root, files)
    base_graph_files = dict(files)
    base_graph = FrozenInputIndex(root, base_graph_files)
    require_canonical_consistency(base_graph, base_graph_files)
    ledger.close()
    with idle_resources(env, previous) as actual:
        require(not (root/SETUP).exists() and not (root/owner.RUNTIME).exists() and not (root/owner.OUTPUT).exists(), 'fresh one-time setup required')
        require(not stopped(root), 'STOP before committing setup')
        require(base_graph.valid(), 'original merged input topology changed before setup creation')
        setup = root/SETUP
        setup.mkdir(parents=True, exist_ok=False)
        batch_control = formal_bootstrap_control(suite)
        atomic_json(root/BATCH_CONTROL, batch_control)
        small_files = {str(root/name):shared[name]['sha256'] for name in bootstrap.SOURCES}
        add_frozen_file(root, small_files, BATCH_CONTROL)
        atomic_json(root/BATCH_LOCK, dict(schema=bootstrap.LOCK_SCHEMA, execution_enabled=True, files=small_files))
        entry = dict(kind='batched901', pointer=shared[owner.BATCH_ENTRY],
            batch_control=add_frozen_file(root,files,BATCH_CONTROL), batch_lock=add_frozen_file(root,files,BATCH_LOCK))
        control = dict(schema=owner.CONTROL_SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), split='test',
            expected_cases=1920, source_count=640, previous_runtime=PREVIOUS, previous_controller_runtime=owner.PREVIOUS_CONTROLLER,
            completed_job_ids=part['completed'], remaining_job_ids=part['remaining'],
            completed_key_sha256=digest(part['completed']), remaining_key_sha256=digest(part['remaining']),
            policy_sha256=seed['policy_sha256'], author_outcome_protocol=protocol,
            suite=suite_pointer, suite_lock=suite_lock_pointer, source_inputs=source_pointer, validation_release=release_pointer,
            resume_owner=pointer('scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'), resume_auditor=pointer(auditor.SOURCE),
            resume_controller=pointer(CONTROLLER), resource_qualification=pointer(QUALIFICATION), resource_qualifier=pointer(QUALIFIER),
            resource_profile=PROFILE, execution_entry=entry, execution_implementation=seed['execution_implementation'],
            standing_continuation_record=pointer(STANDING), concurrency_authorization_record=pointer(AUTHORIZATION),
            user_authorized_resume=True, standing_authorization_reused=True, new_user_event_invented=False,
            physical_policy_changed=False, models_retrained=False, old_evidence_modified=False,
            completed_results_selected_by_success=False, native_started=False, GPU_started=False,
            prepared_frozen_environment=actual['frozen_environment'], preparation_inventory=actual['inventory'],
            previous_process_exit_proof=previous_proof, original_input_maps_preserved=True,
            TEST_metadata_and_completed_case_evidence_read=True, TEST_used_for_fitting=False, **env)
        for key,name in (('previous_launch',PREVIOUS+'/launch.json'),('previous_plan',PREVIOUS+'/plan.json'),
            ('previous_status',PREVIOUS+'/status.json'),('previous_terminal',PREVIOUS+'/terminal.json'),
            ('previous_DRAIN',PREVIOUS+'/DRAIN'),('previous_controller_launch',owner.PREVIOUS_CONTROLLER+'/launch.json'),
            ('previous_controller_status',owner.PREVIOUS_CONTROLLER+'/status.json'),('previous_controller_failure',owner.PREVIOUS_CONTROLLER+'/failure.json')):
            control[key] = pointer(name)
        owner.require_partition(control, previous['plan'], previous['status'], previous['terminal'])
        owner.require_resource_qualification(qualification, control)
        atomic_json(root/CONTROL, control)
        cp = add_frozen_file(root,files,CONTROL)
        graph = FrozenInputIndex(root, files)
        require_canonical_consistency(graph, files)
        verify_inputs(root, files)
        require(graph.valid() and not stopped(root), 'input topology changed or STOP before final lock publication')
        atomic_json(root/LOCK, dict(schema=owner.LOCK_SCHEMA, approved_for_execution=True, files=files, control=cp,
            split='test', policy_sha256=seed['policy_sha256'], native_started=False, GPU_started=False,
            original_suite_map_preserved=True, original_source_map_preserved=True, automatic_retry=False))
        return dict(complete=True, completed=len(part['completed']), remaining=len(part['remaining']), expected=1920,
            policy_sha256=seed['policy_sha256'], resume_control=cp, resume_control_lock=add_frozen_file(root,{},LOCK),
            formal_batch_control=entry['batch_control'], formal_batch_lock=entry['batch_lock'],
            native_started=False, GPU_started=False, jobs_started=False, old_evidence_modified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare:
        print(json.dumps(dict(execution_enabled=False, native_started=False, GPU_started=False,
            message='inactive: explicit --prepare is required; no inputs inspected or setup written')))
        return
    print(json.dumps(prepare(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
