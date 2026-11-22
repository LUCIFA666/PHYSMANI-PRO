"""Unactivated945 audit after a fresh single-graph public-held TRAIN192.

Only existing evidence is read. The original315 VAL sources may be read solely
inside1209's live1213 capability, after reauditing every new TRAIN case. Reusing those
sources with a new policy is not a new independent dataset. No success threshold
is applied and this report never permits TEST. Missing evidence is not readiness.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib

from . import repaired_readiness_v1144 as base
from . import public_held_repaired_admission_v1209 as gate
from . import public_held_repaired_validation_inputs_v1210 as inputs
from . import public_held_repaired_case_v1206 as worker
from .public_held_repaired_readiness_v1208 import validate_parallel_phase, validate_master_session, INHERITED_ENTRY

SOURCE = 'our_mani/multitask/public_held_repaired_validation_audit_v1211.py'
SCHEMA = 'r10_fresh_public_held_repaired_validation_audit_v1211'
COHORT_SCHEMA = 'r10_public_held_validation_execution_cohort_v1211'
PLAN_SCHEMA = 'r10_public_held_validation_execution_plan_v1207'
SOURCE_COUNT, CASE_COUNT = 315, 945
BUNDLE_PREFIX = 'artifacts/framework/multitask_r10/public_held_repaired_validation_inputs_v1210/'
OWNER_PREFIX = 'runtime/framework/public_held_repaired_owner_v1207/validation/'
HASHES, METHODS, TASKS, BLOCKS = base.HASHES, base.METHODS, base.TASKS, base.BLOCKS
require, same, pointer, read_json = base.require, base.same, base.pointer, base.read_json


def _ref_equal(left, right):
    return same(pointer(left), pointer(right))


def _typed(ref, prefix, filename=None):
    """Classify metadata paths before hashing/reading any addressed bytes."""
    ref = pointer(ref); name = ref['path']; path = Path(name)
    require(ref['root'] == 'project' and name.startswith(prefix) and not path.is_absolute()
        and '..' not in path.parts and path.as_posix() == name and '\\' not in name
        and (filename is None or path.name == filename), 'typed project metadata pointer required')
    return ref


def _stamp(value, graph_ref, graph):
    base._stamp(value, graph_ref, graph)



def _worker_declarations(root,graph):
    """The new auditor cannot relabel an old worker/schema as this controller."""
    worker_ref=worker.pointer(root,worker.WORKER)
    preparer_ref=worker.pointer(root,worker.SOURCE)
    consumer_ref=worker.pointer(root,worker.RUNTIME)
    for route in graph['routes']:
        declaration=route['evaluation']
        require(declaration.get('case_schema')==worker.CASE_SCHEMAS['validation']
            and _ref_equal(declaration.get('worker'),worker_ref)
            and _ref_equal(declaration.get('preparer'),preparer_ref)
            and _ref_equal(declaration.get('consumer'),consumer_ref),
            'actual1206 worker/preparer/744 consumer and validation case schema required')


def _bundle(root, graph_ref, bundle, bundle_ref, current):
    """Metadata and source envelope validation, always under the active gate."""
    graph, admission = current['graph'], current['admission']
    require(bundle.get('schema') == inputs.SCHEMA and bundle.get('source_count') == SOURCE_COUNT
        and type(bundle.get('source_count')) is int and bundle.get('Native_started') is False
        and bundle.get('TEST_read') is False, 'actual new1148 validation bundle required')
    _stamp(bundle, graph_ref, graph)
    require(_ref_equal(bundle.get('admission'), current['admission_ref']), 'bundle admission differs')
    directory = str(Path(bundle_ref['path']).parent)+'/'
    typed = {name:_typed(bundle.get(name),directory,name+'.json') for name in ('suite','suite_lock','source_inputs')}
    require(all(ref['path'] == directory+name+'.json' for name,ref in typed.items()),
        'source metadata must be siblings of actual1148 bundle')
    args = SimpleNamespace(**{name:ref['path'] for name,ref in typed.items()})
    chosen = dict(graph=graph, graph_ref=graph_ref)
    metadata = worker._formal_metadata(root, chosen, args, dict(admission=pointer(current['admission_ref'])))
    suite, suite_ref = read_json(root, metadata['suite'])
    envelope, lock_ref = read_json(root, metadata['suite_lock'])
    source_lock, source_ref = read_json(root, metadata['source_inputs'])
    require(all(same(value.get(key),admission.get(key))
        for value in (bundle,suite,source_lock) for key in gate.DISCLOSURE_FIELDS),
        'source envelope must retain the fresh qualification and separately closed history')
    require(all(value.get('mixed_control_lineage') is False for value in (bundle,suite,source_lock)),
        'new physical policy cannot inherit mixed-control acceptance')
    require(_ref_equal(suite.get('readiness'), admission['readiness'])
        and suite.get('formal_TEST_permitted') is False and suite.get('TEST_used_for_fitting') is False
        and suite.get('tasks') == list(TASKS) and suite.get('methods') == list(METHODS),
        'complete original16/three-method validation suite required')
    # The real640-compatible reader rechecks graph/admission before touching the
    # single admitted original VAL manifest. It rejects TEST and other manifests.
    _, rows, source_proof = gate.read_formal_source(root, suite, envelope['files'], 'validation')
    require(len(rows) == SOURCE_COUNT and len({(r['task'],r['raw_episode_id']) for r in rows}) == SOURCE_COUNT
        and {r['task'] for r in rows} == set(TASKS)
        and all(type(r.get('raw_episode_id')) is str and r['raw_episode_id'] for r in rows),
        'all315 unique original validation sources across16 tasks required')
    require(source_proof.get('actual_split') == 'validation'
        and source_proof.get('component_purpose') == 'evaluation'
        and source_proof.get('policy_sha256') == graph['policy_sha256']
        and source_proof.get('formal_TEST_permitted') is False,
        'actual1147 evaluation source capability required')
    manifest_ref = pointer(metadata['manifest'])
    expected_files = {str(root/manifest_ref['path']):manifest_ref['sha256']}
    source_stamps = {}
    for row in rows:
        source = Path(row['path'])
        require(source.is_absolute() and source.name == 'data.h5'
            and 'val_package_compressed' in source.parts, 'only admitted original VAL reset data required')
        require(str(source) not in expected_files, 'one distinct original source file per manifest row required')
        expected_files[str(source)] = source_lock['files'].get(str(source))
    require(set(source_lock['files']) == set(expected_files)
        and source_lock.get('original_VAL_bytes_reused_for_new_policy') is True
        and source_lock.get('new_independent_validation_dataset') is False,
        'exact post-admission original315 source envelope required')
    for name, checksum in expected_files.items():
        require(type(checksum) is str and len(checksum) == 64, 'actual frozen source checksum required')
        # The source reader checked the manifest; HDF5 bytes are rehashed without
        # importing h5py, opening a simulator or invoking any policy/native API.
        if name != str(root/manifest_ref['path']):
            gate._permit(root, current['permit'])
            require(inputs._hash_file(Path(name)) == checksum, 'admitted VAL reset bytes changed')
            source_stamps[name] = inputs._stamp(Path(name).stat())
    return rows, metadata, (suite_ref, lock_ref, source_ref), source_proof, source_stamps


def _argv(root, argv, job, route, graph_ref, graph, metadata, *, check):
    declaration = route['evaluation']; actual = pointer(declaration['worker'])
    require(type(argv) is list and all(type(v) is str for v in argv)
        and sum(v in (actual['path'],str(root/actual['path'])) for v in argv) == 1,
        'actual graph-declared evaluation worker must occur once in argv')
    wrapper = graph['control']['components']['environment_wrapper']
    require(sum(v in (wrapper['path'],str(root/wrapper['path'])) for v in argv) == 1,
        'actual graph-frozen820 wrapper required')
    launcher = 'scripts/run_aligned_gpu.sh' if route['GPU_required'] else 'scripts/run_local_sim.sh'
    require(any(p['root'] == 'project' and p['path'] == launcher for p in graph['control']['sources_map'].values())
        and sum(v in (launcher,str(root/launcher)) for v in argv) == 1,
        'actual method-specific Native launcher must be frozen')
    for flag,value in (('--manifest-sha256',graph['physical']['environment_manifest']['sha256']),
            ('--graph',graph_ref['path']),('--task',job['task']),('--raw-episode-id',job['raw_episode_id']),
            ('--method',job['method']),('--split','validation'),('--out',job['out']),('--runtime',job['runtime']),
            ('--config',route['configuration']['path'])):
        base._option(argv,flag,value)
    for name in ('suite','suite_lock','source_inputs','admission'):
        base._option(argv,'--'+name.replace('_','-'),pointer(metadata[name])['path'])
    base._option(argv,'--frozen-inputs',route['frozen_inputs']['path'],required=False)
    require(argv.count('--check') == int(check), 'actual CPU/native mode differs')
    require('--validation-release' not in argv, 'this validation cohort cannot carry TEST release evidence')


def _phase(root, refs, job, route, graph_ref, graph, witness, metadata, owner_runtime, *, check):
    require(type(refs) is dict and set(refs) == {'launch','terminal','sidecar'},
        'actual child launch/terminal/sidecar pointers required')
    phase = 'CPU_check' if check else 'native'
    base_path = owner_runtime+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/'+phase+'/'
    for key,name in (('launch',base_path+'launch.json'),('terminal',base_path+'terminal.json'),
            ('sidecar',job['runtime']+'/'+phase+'/worker_sidecar.json')):
        require(pointer(refs[key])['path'] == name, 'phase evidence escaped planned typed metadata path')
    launch, lr = read_json(root, refs['launch']); terminal, tr = read_json(root, refs['terminal'])
    sidecar, sr = read_json(root, refs['sidecar'])
    outer, inner = base._process(launch.get('process')), base._process(sidecar.get('process'))
    require(same(base._process(terminal.get('process')),outer)
        and type(terminal.get('returncode')) is int and terminal['returncode'] == 0
        and terminal.get('automatic_retry') is False and launch.get('automatic_retry') is False,
        'original child exit0 without retry required')
    declaration = route['evaluation']
    for value in (launch,sidecar):
        _stamp(value,graph_ref,graph)
        require(_ref_equal(value.get('actual_worker'),declaration['worker'])
            and _ref_equal(value.get('actual_preparer'),declaration['preparer']),
            'actual evaluation worker/preparer source differs')
    _argv(root,launch.get('argv'),job,route,graph_ref,graph,metadata,check=check)
    execution = sidecar.get('execution',{})
    require(sidecar.get('complete') is True and execution.get('complete') is True
        and same(sidecar.get('graph_witness'),witness) and same(execution.get('graph_witness'),witness)
        and execution.get('actual_split') == 'validation' and execution.get('native_started') is (not check),
        'complete genuine validation CPU/native graph witness required')
    require(sidecar.get('automatic_retry') is False,
        'actual parallel child must not automatically retry')
    if check:
        require(execution.get('mode') == 'CPU_check' and execution.get('GPU_started') is False
            and execution.get('physical_factory_constructed') is False,
            'CPU source preflight cannot construct a physical factory or start GPU')
    else:
        restored = execution.get('runtime_scope_restoration',{})
        require(execution.get('mode') == 'graph_bound_native_validation'
            and execution.get('native_false_preserved') is True
            and execution.get('qualification_granted_by_worker') is False
            and restored.get('complete') is True and restored.get('restored') is True
            and type(restored.get('checks')) is dict and bool(restored['checks'])
            and all(v is True for v in restored['checks'].values())
            and restored.get('extra_native_calls') == 0 and type(restored.get('extra_native_calls')) is int,
            'actual native outcome and completed owned scope restoration required')
    ns = sidecar.get('frozen_OS_environment',{})
    require(ns.get('complete') is True
        and all(type(ns.get(k)) is int for k in ('outer_wrapper_pid','outer_wrapper_start_ticks','observer_pid','observer_start_ticks'))
        and (ns['outer_wrapper_pid'],ns['outer_wrapper_start_ticks']) == (outer['pid'],outer['start_ticks'])
        and (ns['observer_pid'],ns['observer_start_ticks']) == (inner['pid'],inner['start_ticks'])
        and _ref_equal(ns.get('wrapper'),graph['control']['components']['environment_wrapper'])
        and _ref_equal(ns.get('manifest'),graph['physical']['environment_manifest'])
        and _ref_equal(ns.get('namespace_qualification'),graph['control']['static_inputs']['environment_namespace_qualification']),
        'actual820 outer/inner identity and environment evidence required')
    require(sr['path'] == job['runtime']+('/CPU_check/worker_sidecar.json' if check else '/native/worker_sidecar.json'),
        'sidecar escaped its planned fresh case namespace')
    require(base._not_live(outer) and base._not_live(inner), 'completed child process must no longer be live')
    resource=validate_parallel_phase(root,sidecar,launch,job,graph_ref,graph,check=check)
    return sidecar,dict(launch=lr,terminal=tr,sidecar=sr),(inner['pid'],inner['start_ticks']),(outer['pid'],outer['start_ticks']),resource


def _owner(root, cohort, graph_ref, graph, plan_ref):
    refs = cohort.get('owner',{})
    require(type(refs) is dict and set(refs) == {'launch','terminal','exec_completion'}, 'completed evaluation owner evidence required')
    directory = str(Path(plan_ref['path']).parent)+'/'
    for key,ref in refs.items():
        _typed(ref,directory)
        require(Path(pointer(ref)['path']).parent == Path(directory), 'owner evidence escaped actual owner directory')
        if key in ('launch','terminal'):require(pointer(ref)['path'] == directory+key+'.json', 'actual owner metadata filename required')
    launch, _ = read_json(root,refs['launch']); terminal, _ = read_json(root,refs['terminal'])
    completion, _ = read_json(root,refs['exec_completion']); process = base._process(launch.get('process'))
    _stamp(launch,graph_ref,graph)
    require(_ref_equal(launch.get('source'),graph['control']['components']['evaluation_owner'])
        and _ref_equal(launch.get('plan'),plan_ref), 'actual graph-frozen evaluation owner/plan required')
    require(terminal.get('complete') is True and type(terminal.get('completed')) is int
        and terminal['completed'] == CASE_COUNT and type(terminal.get('attempted')) is int
        and terminal['attempted'] == CASE_COUNT and type(terminal.get('infrastructure_failed')) is int
        and terminal['infrastructure_failed'] == 0 and terminal.get('active') is None
        and terminal.get('automatic_retry') is False
        and same(base._process(terminal.get('process')),process)
        and _ref_equal(terminal.get('plan'),plan_ref), 'all945 original attempts must naturally complete')
    _stamp(terminal,graph_ref,graph)
    require(same(base._process(completion.get('process')),process)
        and completion.get('actual_tool_exit_observed') is True and completion.get('natural_completion') is True
        and type(completion.get('returncode')) is int and completion['returncode'] == 0
        and _ref_equal(completion.get('plan'),plan_ref)
        and _ref_equal(completion.get('terminal'),refs['terminal']) and base._not_live(process),
        'actual root-observed owner natural exit0 and dead PID/start identity required')
    plan,_=read_json(root,plan_ref)
    return validate_master_session(root,refs,plan_ref,graph_ref,graph,plan['jobs'])


def _models(root, case, cpu, route, graph, metadata):
    """Check the actual CPU loader and recorded model branch, without loading it."""
    proof = cpu['execution'].get('source_and_model_preflight',{})
    config, _ = read_json(root,route['configuration'])
    require(proof.get('GPU_started') is False
        and _ref_equal(proof.get('split_source_input_lock'),metadata['source_inputs']),
        'actual CPU loader must bind the admitted reset source lock without GPU')
    if route['method'] == 'physmani':
        checkpoint = proof.get('original_author_checkpoint',{})
        expected = Path(config.get('baseline_checkpoint',''))
        candidates = [dict(path=str(Path(graph['roots'][p['root']])/p['path']),sha256=p['sha256'])
            for p in route['checkpoints'].values()]
        require(expected.is_absolute() and expected.name=='epoch_99999.pth'
            and config.get('baseline_reference_filter')=='author_direct_setpoint'
            and checkpoint == dict(path=str(expected),sha256=checkpoint.get('sha256'))
            and checkpoint in candidates and proof.get('original_neural_policy_and_online_world_model') is True
            and case.get('trained_component_identity') is None
            and case.get('actor_metrics',{}).get('original_baseline_algorithm_unchanged') is True,
            'actual released PhysMani checkpoint and unchanged baseline branch required')
        return
    require(_ref_equal(case.get('shared_context_checkpoint'),graph['physical']['shared_context_checkpoint'])
        and _ref_equal(config.get('context_checkpoint'),graph['physical']['shared_context_checkpoint']),
        'actual shared context checkpoint differs')
    component = case.get('trained_component_identity',{})
    if route['task'] in BLOCKS:
        binding = proof.get('actual_shared16_block_adapter',{})
        require(binding.get('task') == route['task']
            and binding.get('checkpoint_sha256') == graph['physical']['shared_context_checkpoint']['sha256']
            and same(component.get('actual_shared16_block_adapter'),binding),
            'actual shared block adapter/checkpoint branch differs')
    else:
        identity = proof.get('component_identity',{})
        require(proof.get('actual_model_loader_validation') is True and proof.get('native_started') is False
            and proof.get('old_TRAIN_actor_constructor_executed') is False
            and identity.get('schema') == 'r10_direct_public_RML_component_identity_v643'
            and identity.get('task') == route['task'] and identity.get('method') == route['method'],
            'actual643 CPU model loader identity required')
        unhashed = dict(identity); unhashed.pop('sha256',None)
        expected = {k:v for k,v in config.get('learned_artifacts',{}).get(route['task'],{}).items()
            if k in ('geometry','interaction','rigid_layout','colour')}
        require(all(expected.get(k) is not None for k in ('geometry','interaction','rigid_layout'))
            and same(identity.get('learned_operation_artifacts'),expected)
            and same(component.get('learned_operation_artifacts'),expected)
            and identity.get('sha256') == base.digest(unhashed)
            and component.get('sha256') == identity.get('sha256')
            and same(proof.get('component_domain'),case['component_domain'])
            and _ref_equal(identity.get('shared_context_checkpoint'),graph['physical']['shared_context_checkpoint'])
            and _ref_equal(component.get('shared_context_checkpoint'),graph['physical']['shared_context_checkpoint'])
            and all(any(_ref_equal(ref,p) for p in route['models'].values()) for ref in expected.values() if ref is not None),
            'actual configured operation artifacts must match frozen graph models')


def _case(root, case, case_ref, job, route, source, graph_ref, graph, witness, cpu, native, native_pid, protocol, profile, metadata):
    key = base._key(job); identity = case.get('identity',{}); unhashed = dict(identity); unhashed.pop('sha256',None)
    require(case.get('split') == 'validation' and case.get('task') == key[0] and case.get('method') == key[2]
        and case.get('source_raw_episode_id') == key[1] and type(case.get('process_id')) is int
        and case['process_id'] == native_pid and case.get('automatic_retry') is False
        and case.get('TEST_used_for_fitting') is False and same(case.get('graph_witness'),witness),
        'actual case evaluation source/worker graph differs')
    require(identity.get('version') == 'r10_fresh_case_identity_v1' and identity.get('split') == 'validation'
        and identity.get('task') == key[0] and identity.get('method') == key[2]
        and identity.get('source_raw_episode_id') == key[1] and identity.get('parent_raw_episode_id') == key[1]
        and identity.get('raw_episode_id') == key[1]+'__policy_aug_r0' and type(identity.get('replica')) is int
        and identity['replica'] == 0 and identity.get('exploration_seed') is None and identity.get('demo_only') is False
        and identity.get('development_demo_only',False) is False
        and case.get('demo_only') is False and case.get('development_demo_only') is False
        and case.get('formal_validation_metric') is True
        and case.get('raw_episode_id') == identity['raw_episode_id'] and case.get('parent_raw_episode_id') == key[1]
        and case.get('case_isolation') == 'fresh_simulator_process_per_episode'
        and case.get('formal_test_used') is False and case.get('original_source_motion_replayed') is False
        and case.get('recorded_initial_conditions',{}).get('metric_only') is True
        and case.get('recorded_initial_conditions',{}).get('policy_use_allowed') is False,
        'genuine original validation identity and metric-only reset required')
    ep = case.get('execution_protocol',{})
    require(_ref_equal(ep.get('inherited742_preflight_entry'),INHERITED_ENTRY)
        and ep.get('actual_worker_verified_by_new_control_graph') is True
        and ep.get('preflight_entry_is_actual_outer_worker') is False,
        'actual1206 execution must remain distinct from inherited1167 source preflight')
    require(ep.get('schema') == worker.SCHEMA and identity.get('execution_protocol_schema') == ep['schema']
        and identity.get('execution_protocol_sha256') == base.digest(ep)
        and same(cpu['execution'].get('execution_protocol'),ep) and same(native['execution'].get('execution_protocol'),ep),
        'actual CPU/native/case execution protocol identity differs')
    for field,expected in (('actual_worker',route['evaluation']['worker']),('actual_preparer',route['evaluation']['preparer']),
            ('inherited_consumer',route['evaluation']['consumer']),('source_overlay_lock',graph['physical']['source_overlay_lock'])):
        require(_ref_equal(ep.get(field),expected), 'actual evaluation source differs: '+field)
    _stamp(ep,graph_ref,graph)
    require(ep.get('selected_physical_factory') == route['physical_factory'] and same(ep.get('factory_binding'),witness['factory_binding'])
        and ep.get('execution_branch') == witness['model_branch'] and ep.get('actual_split') == 'validation'
        and ep.get('component_purpose') == 'evaluation' and ep.get('formal_score_contribution') is True
        and ep.get('formal_validation_admitted') is True and same(ep.get('post_admission_metadata'),metadata)
        and ep.get('original_745_execution_acceptance_granted') is False and ep.get('qualification_granted_by_worker') is False
        and ep.get('automatic_retry') is False and ep.get('TEST_supported') is False
        and type(ep.get('additional_native_getter_observe_predict_or_success_calls')) is int
        and ep['additional_native_getter_observe_predict_or_success_calls'] == 0
        and same(ep.get('unchanged_action_outcome_sampling'),protocol), 'genuine evaluation factory/protocol required')
    require(_ref_equal(case.get('source_overlay_lock'),graph['physical']['source_overlay_lock'])
        and case.get('original_745_execution_acceptance_granted') is False
        and case.get('qualification_granted_by_worker') is False and case.get('formal_score_contribution') is True
        and case.get('formal_TEST_metric') is False, 'case cannot claim old qualification or TEST metric')
    require(identity.get('sha256') == base.digest(unhashed) and identity.get('config_sha256') == route['configuration']['sha256']
        and identity.get('frozen_inputs_sha256') == route['frozen_inputs']['sha256']
        and identity.get('source_input_lock_sha256') == metadata['source_inputs']['sha256']
        and identity.get('source_row_sha256') == base.digest(source) and identity.get('graph_sha256') == graph_ref['sha256']
        and all(identity.get(k) == graph[k] for k in HASHES)
        and same(cpu['execution'].get('identity'),identity) and same(native['execution'].get('identity'),identity)
        and same(cpu.get('identity'),identity) and same(native.get('identity'),identity),
        'actual source/config/model-bound graph identity differs')
    domain = case.get('component_domain',{})
    require(domain.get('schema') == 'r10_source_bound_RML_component_purpose_v640' and domain.get('purpose') == 'evaluation'
        and domain.get('split') == 'validation' and domain.get('task') == key[0] and domain.get('method') == key[2]
        and domain.get('raw_episode_id') == key[1] and domain.get('VAL_TEST_relabelled_as_TRAIN') is False
        and domain.get('policy_sha256') == graph['policy_sha256']
        and same(cpu['execution'].get('component_domain'),domain) and same(native['execution'].get('component_domain'),domain),
        'genuine640 evaluation capability required; TRAIN relabelling refused')
    require(_ref_equal(native['execution'].get('case'),case_ref)
        and native['execution'].get('native_success') is case.get('official_task_success')
        and native['execution'].get('native_failure') is case.get('native_terminal_failure'), 'actual sidecar/outcome differs')
    require(_ref_equal(protocol.get('runtime_consumer'),route['evaluation']['consumer']), 'original741 consumer required')
    base.validate_sampling_case(case,protocol,block=key[0] in BLOCKS,case_version=route['evaluation']['case_schema'])
    _models(root,case,cpu,route,graph,metadata)
    base._attestation(case,profile)
    return base._motion(case)


def audit_validation(root, graph_ref, bundle_ref, cohort_ref, *, permit):
    """Return a deterministic proof of ALL945 fresh outcomes; write/start none."""
    root = Path(root).resolve()
    # These are project metadata only. Admission and live1106 must precede any
    # source manifest, reset data or suite source-reader access.
    graph_ref = _typed(graph_ref,'artifacts/framework/multitask_r10/','policy_graph.json')
    cohort_ref = _typed(cohort_ref,OWNER_PREFIX,'cohort.json')
    bundle_ref = pointer(bundle_ref)
    require(bundle_ref['root'] == 'project' and bundle_ref['path'].startswith(BUNDLE_PREFIX)
        and Path(bundle_ref['path']).name == 'bundle.json',
        'explicit new1148 metadata bundle required before any read; source/TEST pointers refused')
    bundle, bundle_ref = read_json(root,bundle_ref)
    _typed(bundle.get('admission'),'artifacts/framework/multitask_r10/','admission.json')
    with gate.admission_scope(root,graph_ref,bundle.get('admission'),permit) as current:
        graph = current['graph']; graph_ref = pointer(current['graph_ref'])
        self_ref = dict(root='project',path=SOURCE,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        require(_ref_equal(graph['control']['components']['evaluation_auditor'],self_ref),
            'actual1211 evaluation auditor must have been frozen before TRAIN192')
        _worker_declarations(root,graph)
        rows, metadata, bundle_inputs, _, source_stamps = _bundle(root,graph_ref,bundle,bundle_ref,current)
        sources = {(row['task'],row['raw_episode_id']):row for row in rows}
        expected = {(task,raw,method) for task,raw in sources for method in METHODS}
        profile, protocol = base._profile(graph,root)
        cohort, cohort_ref = read_json(root,cohort_ref)
        require(cohort.get('schema') == COHORT_SCHEMA and cohort.get('complete') is True
            and cohort.get('split') == 'validation' and cohort.get('automatic_retry') is False
            and type(cohort.get('expected_cases')) is int and cohort['expected_cases'] == CASE_COUNT
            and cohort.get('source_outcome_filtering') is False and cohort.get('reused_cases') == []
            and _ref_equal(cohort.get('validation_bundle'),bundle_ref), 'fresh unfiltered945 validation cohort required')
        _stamp(cohort,graph_ref,graph)
        expected_plan = str(Path(cohort_ref['path']).parent/'plan.json')
        require(pointer(cohort.get('plan'))['path'] == expected_plan, 'cohort must name its actual prelaunch owner plan')
        plan, plan_ref = read_json(root,cohort.get('plan'))
        require(plan.get('schema') == PLAN_SCHEMA and plan.get('split') == 'validation'
            and type(plan.get('expected_cases')) is int and plan['expected_cases'] == CASE_COUNT
            and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False
            and plan.get('reused_cases') == [] and _ref_equal(plan.get('validation_bundle'),bundle_ref),
            'prelaunch original315 bythree new-policy evaluation plan required')
        require(all(all(same(value.get(key),current['admission'].get(key)) for key in gate.DISCLOSURE_FIELDS)
            and value.get('mixed_control_lineage') is False
            and value.get('case_graphs_relabelled') is False for value in (plan,cohort)),
            'validation plan/cohort must retain the fresh TRAIN192 proof and its separate historical closure')
        require(plan.get('owner_runtime') == str(Path(cohort_ref['path']).parent), 'planned owner runtime differs')
        _stamp(plan,graph_ref,graph)
        jobs, records = plan.get('jobs'), cohort.get('records')
        require(type(jobs) is list and len(jobs) == CASE_COUNT and type(records) is list and len(records) == CASE_COUNT,
            'all945 planned/completed attempts required; partial validation is not ready')
        keys, record_keys = [base._key(v) for v in jobs],[base._key(v) for v in records]
        require(len(set(keys)) == len(set(record_keys)) == CASE_COUNT and set(keys) == set(record_keys) == expected,
            'exact admitted315 bythree grid once without replacement required')
        require(len({j.get('out') for j in jobs}) == len({j.get('runtime') for j in jobs}) == CASE_COUNT,
            'fresh unique validation output/runtime namespaces required')
        master_session=_owner(root,cohort,graph_ref,graph,plan_ref)
        routes = {(r['task'],r['method']):r for r in graph['routes']}; by_key = {base._key(v):v for v in records}
        result, process_ids, evidence_paths = [],set(),set()
        resource_refs=[]
        for job in jobs:
            key = base._key(job); record = by_key[key]; route = routes[(key[0],key[2])]; source = sources[key[:2]]
            require(type(record.get('attempt')) is int and record['attempt'] == 1, 'one original attempt per source required')
            require(_ref_equal(job.get('configuration'),route['configuration'])
                and job.get('source_row_sha256') == base.digest(source), 'prelaunch source/config selection differs')
            for field,prefix in (('out',worker.ARTIFACT_PREFIX+'validation/'),('runtime',worker.RUNTIME_PREFIX+'validation/')):
                name = job.get(field)
                require(type(name) is str and name.startswith(prefix) and Path(name).as_posix() == name
                    and '..' not in Path(name).parts and not any('demo_only' in p for p in Path(name).parts),
                    'genuine fresh1206 validation case namespace required')
            witness = base.make_graph_witness(graph_ref,graph,route,source,split='validation')
            cpu, cr, cp, co, cres = _phase(root,record.get('CPU_check'),job,route,graph_ref,graph,witness,metadata,plan['owner_runtime'],check=True)
            native, nr, np, no, nres = _phase(root,record.get('native'),job,route,graph_ref,graph,witness,metadata,plan['owner_runtime'],check=False)
            resource_refs.extend(res[k] for res in (cres,nres) for k in ('session','claim','terminal'))
            require(same(cres['session'],master_session) and same(nres['session'],master_session),
                'all validation cases must descend from this exact master session')
            current_processes = (cp,co,np,no)
            require(len(set(current_processes)) == 4 and not process_ids.intersection(current_processes),
                'fresh actual CPU/native outer and inner worker processes required for every case')
            process_ids.update(current_processes)
            require(pointer(record.get('case'))['path'] == job['out']+'/case.json'
                and pointer(record.get('owned_hooks'))['path'] == job['out']+'/owned_hook_receipt.json',
                'actual case/hook receipt escaped planned output')
            case, case_ref = read_json(root,record.get('case')); hooks, hook_ref = read_json(root,record.get('owned_hooks'))
            for reference in (case_ref,hook_ref,*cr.values(),*nr.values()):
                require(reference['path'] not in evidence_paths, 'actual per-case evidence cannot be reused')
                evidence_paths.add(reference['path'])
            motion_pass = _case(root,case,case_ref,job,route,source,graph_ref,graph,witness,cpu,native,np[0],protocol,profile,metadata)
            require(hooks.get('actual_split') == 'validation' and type(hooks.get('reset')) is list
                and type(hooks.get('paths')) is list and all(v.get('complete') is True for v in hooks['reset'])
                and all(v.get('close_complete') is True for v in hooks['paths'])
                and same(hooks.get('execution_protocol'),case['execution_protocol'])
                and same(hooks.get('graph_witness'),witness), 'original reset/path cleanup and graph hook evidence required')
            require(len(hooks['reset']) == (0 if key[0] in BLOCKS else 1), 'original core/block reset receipt shape required')
            result.append(dict(task=key[0],raw_episode_id=key[1],method=key[2],native_success=case['official_task_success'],
                native_terminal_failure=case['native_terminal_failure'],motion_pass=motion_pass,
                counted_ticks=case['counted_clock']['ticks'],case=case_ref,owned_hooks=hook_ref,CPU_check=cr,native=nr))
        gate._permit(root,permit)
        require(same(gate._graph(root,graph_ref),graph), 'graph changed during validation audit')
        refs = [bundle_ref,cohort_ref,plan_ref,*bundle_inputs,*cohort['owner'].values(),*resource_refs]
        for record in records:
            refs.extend((record['case'],record['owned_hooks']))
            for phase in ('CPU_check','native'):refs.extend(record[phase].values())
        for ref in refs:read_json(root,ref)
        for name,stamp in source_stamps.items():
            require(inputs._stamp(Path(name).stat()) == stamp, 'VAL source changed during result audit')
        result.sort(key=lambda v:(v['task'],v['raw_episode_id'],v['method']))
        return dict(schema=SCHEMA,graph=graph_ref,validation_bundle=bundle_ref,cohort=cohort_ref,
            **{key:current['admission'][key] for key in gate.DISCLOSURE_FIELDS},
            **{key:graph[key] for key in HASHES},complete=True,split='validation',purpose='evaluation',
            source_count=SOURCE_COUNT,case_count=CASE_COUNT,cases=result,
            groups=[dict(task=t,method=m,completed=sum(v['task'] == t and v['method'] == m for v in result),
                native_successes=sum(v['native_success'] for v in result if v['task'] == t and v['method'] == m),
                native_terminal_failures=sum(v['native_terminal_failure'] for v in result if v['task'] == t and v['method'] == m))
                for t in TASKS for m in METHODS],
            methods=[dict(method=m,completed=sum(v['method'] == m for v in result),
                native_successes=sum(v['native_success'] for v in result if v['method'] == m),
                native_terminal_failures=sum(v['native_terminal_failure'] for v in result if v['method'] == m)) for m in METHODS],
            native_success_is_not_an_admission_threshold=True,motion_is_diagnostic_only=True,all_failures_retained=True,
            automatic_retry=False,source_outcome_filtering=False,old_protocol_outputs_reused=False,
            original_VAL_bytes_reused_for_new_policy=True,new_independent_validation_dataset=False,
            formal_TEST_permitted=False,TEST_read=False,Native_started_by_auditor=False,GPU_started_by_auditor=False,
            block_inherited_observe_success_calls_are_not_new_sampler_calls=True,block_only_tick_success_call_claimed=False)
