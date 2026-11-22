"""Six fixed TRAIN development keys for the separately identified1203 factory.

The1154/745/742/744 substrate is inherited. This envelope grants no192 or
VAL/TEST acceptance and never claims the frozen12c96 physical graph identity.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import train_block_preclose_full16_overlay_v1154 as parent
from . import public_held_block_phase_v1203 as candidate

SOURCE = 'our_mani/multitask/train_public_held_block_development_v1204.py'
WORKER = 'scripts/run_TRAIN_public_held_block_development_v1204_r10.py'
TEST_SOURCE = 'tests/test_train_public_held_block_development_v1204.py'
SCHEMA = 'r10_fixed_TRAIN_public_held_block_development_v1204'
LOCK_SCHEMA = SCHEMA+'_source_lock'
CPU_SCHEMA = SCHEMA+'_observed_CPU_evidence'
FACTORY_MODULE = 'our_mani.multitask.public_held_block_phase_v1203'
FACTORY_SOURCE = candidate.SOURCE
FACTORY_SHA = '8f2670d18ffb387f41becb56ba7a16e72111419c562f5af706f3be7d3445c77e'
FACTORY_TEST = 'tests/test_public_held_block_phase_v1203.py'
FACTORY_TEST_SHA = '1a2c1c0dabc38769e30dd9273630cfc212d871dae690fe2c10df50f435065995'
PARENT_LOCK = 'artifacts/framework/multitask_r10/TRAIN_block_preclose_full16_v1154_setup/source_overlay_lock.json'
PARENT_LOCK_SHA = '0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9'
NAME = 'TRAIN_public_held_block_v1204'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/'+NAME+'_setup/'
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/'+NAME+'/development_demo_only/'
RUNTIME_PREFIX = 'runtime/framework/'+NAME+'/cases/'
GUARD = 'our_mani.multitask.post_completed_TEST_resource_guard_v1106'
LEGACY_WORKER = parent.LEGACY_WORKER
LEGACY_RUNTIME = parent.LEGACY_RUNTIME
PAIR_SOURCES = (('pick_moving_target_on_the_table','episode0'),
                ('pick_moving_target_on_the_table_high_speed','episode100'),
                ('pick_moving_target_on_the_table','episode1'))
METHODS = ('cv_geometry','context_geometry')
FIXED_CASES = tuple((task,raw,method) for task,raw in PAIR_SOURCES for method in METHODS)
EXPLICIT_IMPORT_SOURCES = (
    LEGACY_WORKER, LEGACY_RUNTIME,
    'our_mani/multitask/rml_case_preflight_v742.py',
    'our_mani/multitask/frozen_input_edges_v463.py',
    'scripts/evaluate_multitask_case_r10.py',
    'our_mani/multitask/evaluation.py',
    'our_mani/multitask/repaired_readiness_v1144.py',
    'our_mani/multitask/repaired_policy_graph_v1143.py',
    'our_mani/multitask/post_TEST_resource_guard_v924.py',
    'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
    'scripts/run_frozen_system_environment_v820_r10.py',
    'scripts/run_local_sim.sh',
)


def require(value,message):
    if not value:
        raise ValueError('TRAIN1204: '+message)


digest = parent.digest
raw = parent.raw
pointer = parent.pointer
_path = parent._path


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _parent_lock(root):
    raw(root,PARENT_LOCK,PARENT_LOCK_SHA)
    value,ref=parent.read_overlay_lock(root,PARENT_LOCK)
    require(ref==dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA) and value['integration_complete'] is True,
        'actual frozen1154 substrate lock required')
    return value,ref


def source_roster(inherited_files):
    return tuple(dict.fromkeys((*inherited_files,PARENT_LOCK,parent.SOURCE,parent.WORKER,parent.TEST_SOURCE,
        SOURCE,WORKER,TEST_SOURCE,FACTORY_SOURCE,FACTORY_TEST,*candidate.SOURCES,*EXPLICIT_IMPORT_SOURCES)))


def _current_sources(root,inherited):
    names=source_roster(inherited['files'])
    files={name:pointer(root,name)['sha256'] for name in names}
    require(all(files[name]==expected for name,expected in inherited['files'].items()),
        'frozen1154 dependency changed')
    require(files[FACTORY_SOURCE]==FACTORY_SHA and files[FACTORY_TEST]==FACTORY_TEST_SHA,
        'CPU/peer-reviewed1203 source and tests changed; use a new version')
    return names,files


def factory_binding(files):
    return dict(module=FACTORY_MODULE,function='integrated_block_components',source=FACTORY_SOURCE,
        sha256=files[FACTORY_SOURCE],parent1153_source=candidate.PARENT_SOURCE,
        parent1153_sha256=candidate.PARENT_SHA256,new_physical_candidate=True,
        borrowed_frozen_physical_graph_identity=False)


def _cpu_evidence(root,reference,files):
    require(type(reference) is dict and set(reference)=={'path','sha256'},'explicit observed CPU evidence pointer required')
    _path(root,reference['path'],prefix=SETUP_PREFIX)
    require(Path(reference['path']).name=='control_CPU_evidence.json','typed CPU evidence basename required')
    evidence=json.loads(raw(root,reference['path'],reference['sha256']))
    require(evidence.get('schema')==CPU_SCHEMA and evidence.get('complete') is True
        and evidence.get('actual_tool_exit_observed') is True
        and type(evidence.get('returncode')) is int and evidence['returncode']==0
        and type(evidence.get('tests_run')) is int and evidence['tests_run']>0
        and type(evidence.get('skipped')) is int and evidence['skipped']==0
        and evidence.get('native_started') is False and evidence.get('GPU_started') is False
        and evidence.get('source_hashes_before')==files and evidence.get('source_hashes_after')==files
        and evidence.get('parent_source_overlay_lock')==dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA),
        'Root-observed successful CPU-only checks must bind every actual1204 dependency')
    return evidence


def write_source_lock(root,name,*,integration_complete=False,cpu_evidence_ref=None):
    root=Path(root).resolve();file=_path(root,name,prefix=SETUP_PREFIX,must_exist=False)
    require(type(integration_complete) is bool,'explicit integration readiness required')
    inherited,parent_ref=_parent_lock(root);names,files=_current_sources(root,inherited)
    if integration_complete:
        _cpu_evidence(root,cpu_evidence_ref,files)
    else:
        require(cpu_evidence_ref is None,'pending lock cannot claim CPU completion')
    value=dict(schema=LOCK_SCHEMA,required_sources=list(names),files=files,
        parent_source_overlay_lock=parent_ref,factory=factory_binding(files),
        integration_complete=integration_complete,integration_cpu_evidence=cpu_evidence_ref,
        fixed_cases=[list(row) for row in FIXED_CASES],
        original_TRAIN_manifest=dict(path=parent.MANIFEST,sha256=parent.MANIFEST_SHA),
        original_routes=dict(path=parent.ROUTES,sha256=parent.ROUTES_SHA),
        original_source_inputs=dict(path=parent.TRAIN_INPUTS,sha256=parent.TRAIN_INPUTS_SHA),
        shared671_checkpoint=deepcopy(parent.CHECKPOINT),
        development_only=True,formal_TRAIN192_acceptance_granted=False,formal_VAL_TEST_qualification=False,
        borrowed_frozen_physical_graph_identity=False,automatic_retry=False,source_outcome_filtering=False,
        all_six_outcomes_retained=True,TEST_used_for_fitting=False)
    write_json(file,value);return pointer(root,name)


def read_source_lock(root,name):
    _path(root,name,prefix=SETUP_PREFIX)
    value=json.loads(raw(root,name));inherited,parent_ref=_parent_lock(root);names,files=_current_sources(root,inherited)
    require(value.get('schema')==LOCK_SCHEMA and value.get('required_sources')==list(names)
        and value.get('files')==files and value.get('factory')==factory_binding(files)
        and value.get('parent_source_overlay_lock')==parent_ref
        and value.get('fixed_cases')==[list(row) for row in FIXED_CASES]
        and value.get('original_TRAIN_manifest')==dict(path=parent.MANIFEST,sha256=parent.MANIFEST_SHA)
        and value.get('original_routes')==dict(path=parent.ROUTES,sha256=parent.ROUTES_SHA)
        and value.get('original_source_inputs')==dict(path=parent.TRAIN_INPUTS,sha256=parent.TRAIN_INPUTS_SHA)
        and value.get('shared671_checkpoint')==parent.CHECKPOINT
        and value.get('development_only') is True and value.get('formal_TRAIN192_acceptance_granted') is False
        and value.get('formal_VAL_TEST_qualification') is False
        and value.get('borrowed_frozen_physical_graph_identity') is False
        and value.get('automatic_retry') is False and value.get('source_outcome_filtering') is False
        and value.get('all_six_outcomes_retained') is True and value.get('TEST_used_for_fitting') is False,
        'actual separately identified fixed-six TRAIN development source lock required')
    require(type(value.get('integration_complete')) is bool,'explicit integration state required')
    if value['integration_complete']:
        _cpu_evidence(root,value.get('integration_cpu_evidence'),files)
    else:
        require(value.get('integration_cpu_evidence') is None,'pending integration cannot claim CPU completion')
    return value,pointer(root,name)


def selection(root,args):
    require(getattr(args,'split','train')=='train','VAL/TEST cannot use this development entry')
    key=(args.task,args.raw_episode_id,args.method)
    require(key in FIXED_CASES,'only the preregistered paired TRAIN six; no replacement or outcome selection')
    selected=parent.selection(root,args)
    require(selected['original_route']['GPU_required'] is False,'geometry-only original method route required')
    return dict(selected,fixed_development_index=FIXED_CASES.index(key),development_only=True)


def prepare(root,args):
    root=Path(root).resolve();chosen=selection(root,args);locked,ref=read_source_lock(root,args.overlay_lock)
    out=_path(root,args.out,prefix=ARTIFACT_PREFIX,must_exist=False);require(not out.exists(),'fresh development case required; no overwrite/retry')
    route=chosen['original_route']
    forwarded=SimpleNamespace(config=route['configuration']['path'],manifest=parent.MANIFEST,
        frozen_inputs=route['frozen_inputs']['path'],task=args.task,raw_episode_id=args.raw_episode_id,
        method=args.method,split='train',out=args.out,replica=0,exploration_seed=None,
        development_demo=True,record_demo=False,demo_only=False,suite=None,suite_lock=None,
        validation_release=None,source_inputs=None,check=args.check)
    substrate=importlib.import_module('our_mani.multitask.rml_case_preflight_v742')
    prepared=substrate.prepare(root,forwarded,root/LEGACY_WORKER)
    module=importlib.import_module(FACTORY_MODULE)
    actual_binding=module.binding(root,prepared['config']['baseline_repo'])
    require(actual_binding['source_sha256']==locked['files'][FACTORY_SOURCE]
        and all(locked['files'].get(name)==expected for name,expected in actual_binding['sources_sha256'].items()),
        'actual1203 factory binding or inherited source closure changed')
    protocol=dict(schema=SCHEMA,source_overlay_lock=ref,actual_worker=pointer(root,WORKER),
        actual_preparer=pointer(root,SOURCE),inherited_consumer=pointer(root,LEGACY_RUNTIME),
        original_route=route,original_route_sha256=chosen['original_route_sha256'],
        parent_source_overlay_lock=dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA),
        factory=factory_binding(locked['files']),public_phase_component_binding=actual_binding,
        unchanged_action_outcome_sampling=substrate.outcome_protocol_binding(root),
        changed_factor='public_grasp_phase_and_inhand_pose_admission',
        original743_prepare_worker_check='substrate_only_not_new_worker_approval',
        original1153_preclose_and1114_RRT_RML_gripper_preserved=True,
        additional_native_getter_observe_predict_camera_or_success_calls=0,
        native_physics_predicates_clock_budget_and_exceptions_unchanged=True,
        development_only=True,formal_score_contribution=False,formal_TRAIN192_acceptance_granted=False,
        formal_VAL_TEST_qualification=False,borrowed_frozen_physical_graph_identity=False,
        automatic_retry=False,source_outcome_filtering=False,all_six_outcomes_retained=True)
    old=deepcopy(prepared['identity']);require(old.get('physical_policy_sha256') is None,'cannot borrow a formal physical graph identity')
    identity=dict(old);identity.pop('sha256')
    identity.update(execution_protocol_schema=SCHEMA,execution_protocol_sha256=digest(protocol),
        source_overlay_lock_sha256=ref['sha256'],original_substrate_identity_sha256=old['sha256'])
    identity['sha256']=digest(identity)
    prepared.update(identity=identity,source_substrate_identity=old,execution_protocol=protocol,
        source_overlay=locked,source_overlay_pointer=ref,overlay_lock_name=args.overlay_lock,
        overlay_selection=chosen,runtime_scope_restoration=None,root_explicit_execute=getattr(args,'execute',False) is True)
    return prepared


def verify(root,prepared):
    substrate=importlib.import_module('our_mani.multitask.rml_case_preflight_v742');substrate.verify(root,prepared)
    locked,ref=read_source_lock(root,prepared['overlay_lock_name'])
    require(locked==prepared['source_overlay'] and ref==prepared['source_overlay_pointer'],'development source lock changed')
    require(digest(prepared['execution_protocol'])==prepared['identity']['execution_protocol_sha256'],'actual development protocol changed')


def require_native_ready(prepared):
    require(prepared['source_overlay']['integration_complete'] is True,'Native disabled before observed CPU/source lock completion')
    require(prepared.get('root_explicit_execute') is True,'Root must explicitly invoke --execute; CPU/default never starts Native')


def annotated_output(prepared,path,value):
    value=deepcopy(value);require(type(value) is dict,'dictionary artifact required')
    require(not any(name in value for name in ('physical_policy_sha256','control_graph_sha256','graph')),
        'this candidate cannot relabel artifacts as a frozen physical/control graph')
    value.update(execution_protocol=prepared['execution_protocol'],source_overlay_lock=prepared['source_overlay_pointer'],
        development_only=True,formal_score_contribution=False,formal_TRAIN192_acceptance_granted=False,
        formal_VAL_TEST_qualification=False,borrowed_frozen_physical_graph_identity=False,TEST_used_for_fitting=False)
    if Path(path).name=='case.json':
        value['inherited_consumer_record_version']=value.pop('version',None);value['version']=SCHEMA+'_case'
        if 'full16_frozen_execution' in value:value['inherited_TRAIN_source_execution_record']=value.pop('full16_frozen_execution')
    return value


@contextmanager
def runtime_scope(root,prepared,runtime,*,permit):
    require_native_ready(prepared);guard=importlib.import_module(GUARD);guard.require_permit(permit,root)
    verify(root,prepared)
    require(Path(runtime.__file__).resolve()==Path(root)/LEGACY_RUNTIME,'original744 runtime required')
    originals=dict(writer=runtime.atomic_json,geometry=runtime._geometry,baseline=runtime._baseline)
    calls=0;written=set();error=None
    try:
        with ExitStack() as lifetime:
            def geometry(current,env,descriptions):
                nonlocal calls
                require(current is prepared and calls==0,'one actual1203 factory per prepared case required');calls+=1
                module=importlib.import_module(FACTORY_MODULE)
                frontend,policy,executor=lifetime.enter_context(module.integrated_block_components(root,current['config'],current['args'].task,current['args'].method,env))
                proof=dict(actual_shared16_block_adapter=current['model_preflight']['actual_shared16_block_adapter'],
                    component_domain=current['component_domain'].receipt(),new_public_held_block_integration=module.binding(root,current['config']['baseline_repo']),
                    new_execution_protocol_schema=SCHEMA,source_overlay_lock=current['source_overlay_pointer'])
                return None,proof,frontend,policy,executor
            def write(file,value):
                file=Path(file);require(file.parent==prepared['out'] and file.name not in written and not file.exists(),'new development output must not escape or overwrite')
                if file.name=='case.json':verify(root,prepared)
                result=originals['writer'](file,annotated_output(prepared,file,value));written.add(file.name);return result
            lifetime.enter_context(patch.object(runtime,'_geometry',geometry))
            lifetime.enter_context(patch.object(runtime,'atomic_json',write))
            yield
    except BaseException as exc:
        error=exc;raise
    finally:
        checks=dict(writer_restored=runtime.atomic_json==originals['writer'],geometry_restored=runtime._geometry==originals['geometry'],baseline_preserved=runtime._baseline==originals['baseline'])
        prepared['runtime_scope_restoration']=dict(complete=True,restored=all(checks.values()),checks=checks,
            geometry_factory_calls=calls,extra_native_calls=0,original_exception_type=None if error is None else type(error).__name__)
        if error is None:require(all(checks.values()),'owned1203 runtime scope did not restore')
