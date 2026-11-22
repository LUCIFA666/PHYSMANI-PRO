"""One genuine TRAIN/validation case under a pre-frozen repaired policy graph.

The original742 source/model/purpose preflight and744 physical loop remain the
substrate. The actual worker is1167, never a substituted743 identity. Validation
requires1147 before742 may read its source. Graph/proof/split artifacts are data
arguments, so their later creation does not require changing this source.
The frozen1165 overlay supplies physical provenance for core1164/block1153;
this module never calls the development1165 preparer or its stop gate.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import repaired_policy_graph_v1143 as graphs
from . import repaired_readiness_v1144 as readiness

SOURCE = 'our_mani/multitask/repaired_policy_case_v1167.py'
WORKER = 'scripts/run_repaired_policy_case_v1167_r10.py'
TEST_SOURCE = 'tests/test_repaired_policy_case_v1167.py'
SCHEMA = 'r10_graph_bound_repaired_policy_case_v1167'
CASE_SCHEMAS = dict(train=SCHEMA+'_TRAIN_case',validation=SCHEMA+'_validation_case')
RUNTIME = 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
INHERITED_VERSION = 'r10_full16_explicit_author_action_outcome_v744'
PREFLIGHT = 'our_mani/multitask/rml_case_preflight_v742.py'
ADMISSION_MODULE = 'our_mani.multitask.repaired_admission_v1147'
ADMISSION_SOURCE = ADMISSION_MODULE.replace('.','/')+'.py'
SOURCE_INPUTS_SOURCE = 'our_mani/multitask/repaired_validation_inputs_v1148.py'
SOURCE_INPUTS_SCHEMA = 'r10_post_admission_repaired_validation_inputs_v1148'
GUARD_SOURCE = 'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py'
ENV_SOURCE = 'scripts/run_frozen_system_environment_v820_r10.py'
PURPOSE_SOURCE = 'our_mani/multitask/rml_component_domain_v640.py'
TRAIN_MANIFEST = 'artifacts/framework/protocol/train_manifest.json'
BLOCKS = ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/repaired_policy_v1167/'
RUNTIME_PREFIX = 'runtime/framework/repaired_policy_v1167/'
# Immutable physical provenance only. New formal qualification is graph-bound,
# and is never borrowed from the development entry or the interrupted cohort.
PHYSICAL_SOURCE_OVERLAY = dict(root='project',
    path='artifacts/framework/multitask_r10/TRAIN_native_release_full16_v1165_setup/source_overlay_lock.json',
    sha256='7d1b53b6a56bd48012e0850320b859d765b7c4b6d32f94f6aa607a22f8eef929')

FACTORIES = {
    'core_geometry': ('our_mani/multitask/author_native_release_core_v1164.py','integrated_components'),
    'block_geometry': ('our_mani/multitask/retained_preclose_block_integration_v1153.py','integrated_block_components'),
    'physmani': (RUNTIME,'_baseline'),
}
_FORMAL = ContextVar('repaired1167_admission_scope',default=None)


def require(value,message):
    if not value:raise ValueError('case1167: '+message)


def digest(value):return readiness.digest(value)


def path(root,name,*,prefix=None,exists=True):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
        and str(Path(name))==name and '\\' not in name,'canonical project-relative path required')
    if prefix is not None:require(name.startswith(prefix),'explicit new1167 namespace required')
    result=Path(root)/name
    require(result.resolve(strict=exists)==result,'symlink/alias inputs and outputs forbidden')
    if exists:require(result.is_file(),'explicit file required')
    return result


def pointer(root,name):
    file=path(root,name)
    return dict(root='project',path=name,sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def _plain(ref):
    require(ref.get('root','project')=='project','project metadata/source pointer required')
    return dict(path=ref['path'],sha256=ref['sha256'])


def _same(left,right):return readiness.pointer(left)==readiness.pointer(right)


def _actual_source(root,name,declared):
    require(_same(pointer(root,name),declared),'actual source differs from graph declaration: '+name)


def write_json(file,value):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def _branch(task,method):
    return 'physmani' if method=='physmani' else ('block_geometry' if task in BLOCKS else 'core_geometry')


def _factory(root,graph,route):
    branch=_branch(route['task'],route['method']);name,callable_name=FACTORIES[branch]
    definition=graph['physical']['factories'][route['physical_factory']]
    require(definition['callable']==callable_name,'registered physical factory callable required')
    _actual_source(root,name,definition['source'])
    return dict(branch=branch,definition=definition,module=name[:-3].replace('/','.'),function=callable_name)


def selection(root,args):
    require(args.split in ('train','validation'),'only TRAIN and genuine validation are supported; TEST refused')
    require(args.task in graphs.TASKS and args.method in graphs.METHODS,'registered original16 by3 route required')
    require(type(args.raw_episode_id) is str and bool(args.raw_episode_id),'explicit original source ID required')
    graph_ref=pointer(root,args.graph);graph=graphs.read_policy_graph(root,graph_ref)
    route=next(row for row in graph['routes'] if (row['task'],row['method'])==(args.task,args.method))
    declaration=route['train' if args.split=='train' else 'evaluation']
    require(declaration['case_schema']==CASE_SCHEMAS[args.split],'actual new1167 case schema required')
    for name,key in ((WORKER,'worker'),(SOURCE,'preparer'),(RUNTIME,'consumer')):
        _actual_source(root,name,declaration[key])
    require(args.config==_plain(route['configuration'])['path'],'explicit config argv must match graph route')
    for component,name in (('resource_guard',GUARD_SOURCE),('environment_wrapper',ENV_SOURCE),('component_purpose',PURPOSE_SOURCE)):
        _actual_source(root,name,graph['control']['components'][component])
    require(graph['train_source_manifest']['root']=='project' and graph['train_source_manifest']['path']==TRAIN_MANIFEST,
        'original TRAIN manifest pointer required')
    factory=_factory(root,graph,route)
    out=path(root,args.out,prefix=ARTIFACT_PREFIX+args.split+'/',exists=False)
    require(not out.exists(),'fresh output required; no overwrite/retry')
    runtime=path(root,args.runtime,prefix=RUNTIME_PREFIX+args.split+'/',exists=False)
    if args.split=='train':
        require('development_demo_only' in out.parts,'742 TRAIN requires explicit development_demo_only namespace')
        require(all(getattr(args,key,None) is None for key in ('suite','suite_lock','source_inputs','admission')),
            'TRAIN cannot carry formal evidence or source locks')
    else:
        require(not any('demo_only' in part for part in out.parts),'validation cannot use demo namespaces')
        require(all(type(getattr(args,key,None)) is str and getattr(args,key) for key in ('suite','suite_lock','source_inputs','admission')),
            'validation requires explicit suite/suite-lock/source-inputs/admission')
        _actual_source(root,ADMISSION_SOURCE,graph['control']['components']['admission_guard'])
        _actual_source(root,ADMISSION_SOURCE,graph['control']['components']['formal_source_reader'])
    return dict(graph=graph,graph_ref=graph_ref,route=route,declaration=declaration,factory=factory,
        split=args.split,task=args.task,method=args.method,raw_episode_id=args.raw_episode_id,out=out,runtime=runtime)


@contextmanager
def admission_scope(root,chosen,args,permit):
    if args.split=='train':
        yield None;return
    require(_FORMAL.get() is None,'formal admission scopes cannot nest')
    admission_ref=pointer(root,args.admission)
    module=importlib.import_module(ADMISSION_MODULE)
    require(Path(module.__file__).resolve()==Path(root)/ADMISSION_SOURCE,'actual registered1147 module required')
    old=importlib.import_module('our_mani.multitask.formal_suite_admission_v532')
    token=None
    try:
        with module.admission_scope(root,chosen['graph_ref'],admission_ref,permit) as proof, \
             patch.object(old,'read_formal_source',module.read_formal_source):
            token=_FORMAL.set(dict(graph=chosen['graph_ref'],admission=admission_ref))
            yield proof
    finally:
        if token is not None:_FORMAL.reset(token)


def _source_overlay(root,graph):
    require(_same(graph['physical']['source_overlay_lock'],PHYSICAL_SOURCE_OVERLAY),
        'exact frozen1165 source provenance for core1164/block1153 required')
    value,ref=graphs.read_project_json(root,graph['physical']['source_overlay_lock'])
    require(value.get('integration_complete') is True and value.get('unpinned_sources')==[]
        and type(value.get('files')) is dict and type(value.get('required_sources')) is list
        and len(value['required_sources'])==len(set(value['required_sources']))
        and set(value['files'])==set(value['required_sources']), 'complete original physical source overlay required')
    require(value.get('original_745_execution_acceptance_granted') is False
        and value.get('formal_VAL_TEST_qualification') is False,'physical provenance cannot grant formal qualification')
    for name,checksum in value['files'].items():
        require(pointer(root,name)['sha256']==checksum,'physical overlay source changed: '+name)
    cpu,cpu_ref=graphs.read_project_json(root,value['integration_cpu_evidence'])
    require(cpu.get('complete') is True and type(cpu.get('tests_exit_code')) is int and cpu['tests_exit_code']==0
        and cpu.get('native_started') is False and cpu.get('GPU_started') is False
        and cpu.get('source_hashes')==value['files'],'actual source-bound physical integration CPU evidence required')
    # Both selected physical branches must be active declarations in the same
    # checked overlay. Producer metadata may additionally describe the prior
    # comparison, but cannot substitute the registered source/callable/hash.
    for branch,field in (('core_geometry','core_factory'),('block_geometry','block_factory')):
        name,function=FACTORIES[branch];actual=pointer(root,name)['sha256']
        factory=value.get(field)
        expected=dict(source=name,function=function,sha256=actual)
        require(type(factory) is dict
            and {key:factory.get(key) for key in expected}==expected
            and factory.get('module')==name[:-3].replace('/','.')
            and value['files'].get(name)==actual,
            'physical overlay must exactly bind registered '+field+' and include its actual source')
    return dict(pointer=ref,CPU_evidence=cpu_ref,source_count=len(value['files']),qualified_as_source_provenance_only=True)


def _validate_graph_models(root,prepared,chosen):
    from . import rml_case_preflight_v742 as substrate
    graph,route=chosen['graph'],chosen['route']
    for ref in [*route['models'].values(),*route['checkpoints'].values()]:
        alias=ref.get('root','project');file=Path(graph['roots'][alias])/ref['path']
        substrate._checked(root,prepared['files'],file)
    config=prepared['config']
    if route['method']=='physmani':
        file=Path(config['baseline_checkpoint']).resolve()
        require(any(file==Path(graph['roots'][ref['root']])/ref['path'] for ref in route['checkpoints'].values()),
            'actual baseline checkpoint must be graph-declared')
    else:
        require(_same(config['context_checkpoint'],graph['physical']['shared_context_checkpoint']),
            'actual shared context checkpoint must match frozen graph')
        if route['task'] not in BLOCKS:
            artifacts=config.get('learned_artifacts',{}).get(route['task'],{})
            require(type(artifacts) is dict and all(artifacts.get(key) is not None
                for key in ('geometry','interaction','rigid_layout')),'actual learned core operation artifacts required')
            for name,ref in artifacts.items():
                if ref is None:continue
                require(any(_same(ref,declared) for declared in route['models'].values()),
                    'actual configured core operation artifact missing from graph models: '+name)
    profile,_=graphs.read_project_json(root,graph['physical']['common_RML']['profile'])
    protocol,_=graphs.read_project_json(root,graph['physical']['outcome_sampling']['protocol'])
    require(digest(profile)==digest(prepared['common_author_RML_profile']),'graph-frozen complete530 profile differs')
    require(digest(protocol)==digest(substrate.outcome_protocol_binding(root)),'graph-frozen complete741 protocol differs')
    return protocol


def _formal_metadata(root,chosen,args,active):
    """Only post-admission metadata;742 retains all actual VAL HDF5 checks."""
    graph=chosen['graph'];suite_ref=pointer(root,args.suite);lock_ref=pointer(root,args.suite_lock)
    inputs_ref=pointer(root,args.source_inputs)
    suite,_=graphs.read_project_json(root,suite_ref)
    lock,_=graphs.read_project_json(root,lock_ref)
    inputs,_=graphs.read_project_json(root,inputs_ref)
    require(suite.get('schema')=='r10_repaired_validation_suite_v1147'
        and _same(suite.get('graph',{}),chosen['graph_ref'])
        and _same(suite.get('admission',{}),active['admission'])
        and _same(suite.get('source_inputs',{}),inputs_ref)
        and all(suite.get(key)==graph[key] for key in readiness.HASHES),
        'actual suite graph/admission/source-input pointer differs')
    require(lock.get('schema')==SOURCE_INPUTS_SCHEMA+'_suite_evidence_lock'
        and lock.get('split')=='validation' and lock.get('formal_TEST_permitted') is False
        and lock.get('TEST_used_for_fitting') is False
        and _same(lock.get('graph',{}),chosen['graph_ref'])
        and _same(lock.get('admission',{}),active['admission'])
        and lock.get('policy_sha256')==graph['policy_sha256'] and type(lock.get('files')) is dict,
        'actual suite evidence lock differs from admitted graph')
    for ref in (chosen['graph_ref'],active['admission'],suite_ref,inputs_ref):
        require(lock['files'].get(str(Path(root)/ref['path']))==ref['sha256'],
            'suite evidence lock omits an actual metadata pointer')
    manifest=suite.get('source_manifests',{}).get('validation')
    require(type(manifest) is dict,'admitted validation manifest pointer required')
    _plain(manifest)  # A path declaration only: do not open the source manifest here.
    producer=graph['control']['components']['split_source_lock_writer']
    _actual_source(root,SOURCE_INPUTS_SOURCE,producer)
    require(inputs.get('schema')==graph['control']['contracts']['split_source_lock_schema']
        and inputs.get('producer_schema')==SOURCE_INPUTS_SCHEMA
        and _same(inputs.get('producer',{}),producer)
        and inputs.get('split')=='validation' and type(inputs.get('source_count')) is int
        and inputs['source_count']==315 and inputs.get('source_manifest_sha256')==manifest['sha256']
        and _same(inputs.get('graph',{}),chosen['graph_ref'])
        and _same(inputs.get('admission',{}),active['admission'])
        and all(inputs.get(key)==graph[key] for key in readiness.HASHES)
        and inputs.get('TEST_used_for_fitting') is False and inputs.get('source_read_after_admission') is True
        and type(inputs.get('files')) is dict,
        'new graph-bound1148 validation source lock required')
    return dict(suite=suite_ref,suite_lock=lock_ref,source_inputs=inputs_ref,
        admission=active['admission'],manifest=manifest)


def prepare(root,args,*,chosen=None):
    root=Path(root).resolve();chosen=selection(root,args) if chosen is None else chosen
    route,graph=chosen['route'],chosen['graph'];formal=args.split=='validation'
    if formal:
        active=_FORMAL.get()
        require(active is not None and _same(active['graph'],chosen['graph_ref'])
            and _same(active['admission'],pointer(root,args.admission)),'1147 scope must precede all formal source preparation')
        formal_metadata=_formal_metadata(root,chosen,args,active)
        manifest=_plain(formal_metadata['manifest'])['path']
    else:manifest=TRAIN_MANIFEST;formal_metadata=None
    forwarded=SimpleNamespace(config=args.config,manifest=manifest,frozen_inputs=_plain(route['frozen_inputs'])['path'],
        task=args.task,raw_episode_id=args.raw_episode_id,method=args.method,split=args.split,out=args.out,
        replica=0,exploration_seed=None,development_demo=not formal,record_demo=False,demo_only=False,
        suite=getattr(args,'suite',None),suite_lock=getattr(args,'suite_lock',None),validation_release=None,
        source_inputs=getattr(args,'source_inputs',None),check=args.check)
    from . import rml_case_preflight_v742 as substrate
    prepared=substrate.prepare(root,forwarded,root/WORKER)
    require(prepared['formal'] is formal,'actual preflight formal purpose differs')
    domain=prepared['component_domain'].receipt()
    require(domain.get('purpose')==chosen['declaration']['purpose'] and domain.get('split')==args.split
        and domain.get('VAL_TEST_relabelled_as_TRAIN') is False,'genuine640 component domain required')
    if formal:require(prepared['full16_policy_sha256']==graph['policy_sha256'],'admitted policy differs from frozen graph')
    expected_protocol=_validate_graph_models(root,prepared,chosen)
    provenance=_source_overlay(root,graph)
    witness=readiness.make_graph_witness(chosen['graph_ref'],graph,route,prepared['row'],split=args.split)
    protocol=dict(schema=SCHEMA,graph=chosen['graph_ref'],**{key:graph[key] for key in readiness.HASHES},
        actual_worker=chosen['declaration']['worker'],actual_preparer=chosen['declaration']['preparer'],
        inherited_consumer=chosen['declaration']['consumer'],source_overlay_lock=graph['physical']['source_overlay_lock'],
        selected_physical_factory=route['physical_factory'],factory_binding=chosen['factory']['definition'],
        execution_branch=chosen['factory']['branch'],actual_split=args.split,component_purpose=domain['purpose'],
        formal_score_contribution=formal,formal_validation_admitted=formal,
        post_admission_metadata=formal_metadata,
        original_745_execution_acceptance_granted=False,qualification_granted_by_worker=False,
        unchanged_action_outcome_sampling=expected_protocol,
        old742_source_model_and_source_lock_checks_retained=True,actual742_worker_is1167=True,
        pose_action_max_ticks=10,public_hold_max_ticks=1,total_budget_ticks=200,counted_dt=.05,
        native_physics_RML_gripper_position_clipping_and_outcomes_unchanged=True,
        additional_native_getter_observe_predict_or_success_calls=0,automatic_retry=False,TEST_supported=False)
    old_identity=deepcopy(prepared['identity']);identity=deepcopy(old_identity);identity.pop('sha256',None)
    identity.update(execution_protocol_schema=SCHEMA,execution_protocol_sha256=digest(protocol),
        original_substrate_identity_sha256=old_identity['sha256'])
    identity=readiness.bind_graph_identity(identity,witness)
    prepared.update(identity=identity,source_substrate_identity=old_identity,execution_protocol=protocol,
        graph_witness=witness,graph=graph,graph_ref=chosen['graph_ref'],graph_selection=chosen,
        physical_source_provenance=provenance,full16_policy_sha256=graph['policy_sha256'],
        expected_author_outcome_protocol=expected_protocol,runtime_scope_restoration=None,
        formal_metadata=formal_metadata)
    return prepared


def verify(root,prepared):
    from . import rml_case_preflight_v742 as substrate
    substrate.verify(root,prepared)
    graph=graphs.read_policy_graph(root,prepared['graph_ref'])
    require(graph==prepared['graph'],'frozen graph changed during case')
    require(pointer(root,prepared['graph_ref']['path'])==prepared['graph_ref'],'graph artifact changed during case')
    _source_overlay(root,graph)
    if prepared.get('formal_metadata') is not None:
        for key in ('suite','suite_lock','source_inputs','admission'):
            ref=prepared['formal_metadata'][key]
            require(pointer(root,ref['path'])==ref,'post-admission metadata changed during case: '+key)
    require(digest(prepared['execution_protocol'])==prepared['identity']['execution_protocol_sha256'],
        'execution protocol identity changed')


def annotated_output(prepared,file,value):
    require(type(value) is dict,'dictionary consumer artifact required')
    result=deepcopy(value);formal=prepared['args'].split=='validation'
    result.update(execution_protocol=prepared['execution_protocol'],graph_witness=prepared['graph_witness'],
        source_overlay_lock=prepared['graph']['physical']['source_overlay_lock'],
        original_745_execution_acceptance_granted=False,formal_score_contribution=formal,
        qualification_granted_by_worker=False,formal_TEST_metric=False,TEST_used_for_fitting=False)
    if Path(file).name=='case.json':
        require(result.get('version')==INHERITED_VERSION,'actual inherited744 case record version required before first write')
        result['inherited_consumer_record_version']=result.pop('version',None)
        result['version']=prepared['graph_selection']['declaration']['case_schema']
        if 'full16_frozen_execution' in result:
            result['inherited_consumer_source_execution_record']=result.pop('full16_frozen_execution')
        result['owned_factory_scope_restoration']='reported_after_scope_exit_in_worker_sidecar'
    return result


def _import_factory(root,definition):
    module=importlib.import_module(definition['module'])
    require(Path(module.__file__).resolve()==Path(root)/definition['definition']['source']['path'],
        'actual imported physical factory file differs')
    return module


@contextmanager
def runtime_scope(root,prepared,runtime):
    """Hold owned factories through original744 receipts; restore on every exit."""
    verify(root,prepared)
    require(Path(runtime.__file__).resolve()==Path(root)/RUNTIME,'actual original744 consumer required')
    original_write,original_geometry,original_baseline=runtime.atomic_json,runtime._geometry,runtime._baseline
    chosen=prepared['graph_selection'];owned=[];written=set();original_error=None;geometry_calls=0
    try:
        with ExitStack() as lifetime:
            if chosen['factory']['branch']!='physmani':
                def geometry(current,env,descriptions):
                    nonlocal geometry_calls
                    require(current is prepared,'physical factory escaped prepared case')
                    require(geometry_calls==0,'one owned physical geometry construction per case required')
                    geometry_calls+=1
                    module=_import_factory(root,chosen['factory'])
                    if chosen['factory']['branch']=='block_geometry':
                        frontend,policy,executor=lifetime.enter_context(module.integrated_block_components(
                            root,current['config'],current['args'].task,current['args'].method,env))
                        actor=None
                        proof=dict(actual_shared16_block_adapter=current['model_preflight']['actual_shared16_block_adapter'],
                            component_domain=current['component_domain'].receipt(),new_block_integration=module.binding(root,current['config']['baseline_repo']))
                    else:
                        actor_module=importlib.import_module('our_mani.multitask.rml_component_extensions_v708')
                        actor,proof=actor_module.create_actor(root,env,current['config'],current['args'].method,
                            descriptions,domain=current['component_domain'])
                        resolver=actor.role_resolver
                        before=dict(observer=actor.mode_observer,resolver_object=resolver,
                            resolver=None if resolver is None else resolver.resolve)
                        events={}
                        frontend,policy,executor=lifetime.enter_context(module.integrated_components(
                            root,current['config'],current['args'].task,actor,events=events))
                        owned.append(dict(actor=actor,before=before,frontend=frontend,
                            source=vars(policy).get('source_policy',policy),events=events))
                    proof=dict(proof,graph_witness=current['graph_witness'],new_execution_protocol_schema=SCHEMA)
                    return actor,proof,frontend,policy,executor
                lifetime.enter_context(patch.object(runtime,'_geometry',geometry))
            def write(file,value):
                file=Path(file)
                require(file.parent==prepared['out'],'consumer output escaped fresh1167 namespace')
                require(file.name not in written and not file.exists(),'consumer artifact overwrite forbidden')
                if file.name=='case.json':verify(root,prepared)
                result=original_write(file,annotated_output(prepared,file,value));written.add(file.name);return result
            lifetime.enter_context(patch.object(runtime,'atomic_json',write))
            yield
    except BaseException as error:
        original_error=error;raise
    finally:
        checks=dict(writer_restored=runtime.atomic_json==original_write,geometry_restored=runtime._geometry==original_geometry,
            original_baseline_preserved=runtime._baseline==original_baseline)
        receipts=[]
        for item in owned:
            actor,frontend,source=item['actor'],item['frontend'],item['source']
            checks['owned_actor_'+str(len(receipts))]=(actor.mode_observer is item['before']['observer']
                and actor.role_resolver is item['before']['resolver_object']
                and (actor.role_resolver is None or actor.role_resolver.resolve==item['before']['resolver']))
            checks['owned_methods_'+str(len(receipts))]=('observe' not in vars(frontend)
                and all(name not in vars(source) for name in ('select','_done','candidates','receipt')))
            receipts.append(deepcopy(item['events']))
        prepared['runtime_scope_restoration']=dict(complete=True,restored=all(checks.values()),checks=checks,
            owned_factory_receipts=receipts,original_exception_type=None if original_error is None else type(original_error).__name__,
            geometry_factory_calls=geometry_calls,extra_native_calls=0)
        if not all(checks.values()) and original_error is None:raise RuntimeError('case1167: owned runtime scopes did not restore')


def environment_proof(root,chosen):
    graph=chosen['graph'];wrapper=graph['control']['components']['environment_wrapper']
    manifest=graph['physical']['environment_manifest']
    qualification=graph['control']['static_inputs'].get('environment_namespace_qualification')
    require(qualification is not None,'pre-frozen820 namespace qualification pointer required')
    graphs.read_project_json(root,qualification)
    module=importlib.import_module('scripts.run_frozen_system_environment_v820_r10')
    proof=module.capture_namespace_evidence(Path(root)/_plain(manifest)['path'],manifest['sha256'],
        Path(root)/_plain(qualification)['path'],qualification['sha256'])
    import os
    require(proof.get('complete') is True and proof.get('observer_pid')==os.getpid()
        and _same(proof.get('wrapper',{}),wrapper) and _same(proof.get('manifest',{}),manifest)
        and _same(proof.get('namespace_qualification',{}),qualification),'actual same-process820 namespace proof required')
    return proof
