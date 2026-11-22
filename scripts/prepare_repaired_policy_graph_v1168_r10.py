"""Explicit, CPU-only graph preparation. Default invocation writes/starts none.

Inventory first, Root review second, prepare-only third. No VAL/TEST sample,
row or result is parsed or selected. The complete legacy immutable integrity
lock, including historical metadata/result keys, remains verbatim; original
blind integrity-SHA verification is distinct from any model/source selection.
No simulator, model constructor, subprocess or admission capability is invoked.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from our_mani.multitask import repaired_policy_graph_v1143 as graphs

SOURCE='scripts/prepare_repaired_policy_graph_v1168_r10.py'
TEST_SOURCE='tests/test_prepare_repaired_policy_graph_v1168.py'
SCHEMA='r10_repaired_graph_preparation_inventory_v1168'
RECEIPT_SCHEMA='r10_repaired_graph_preparation_receipt_v1168'
CONTROL_CPU_NAME='control_CPU_evidence.json'
CONTROL_CPU_SCHEMA='r10_repaired_control_CPU_evidence_v1168'
BASE_INPUTS='artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
BASE_SHA='d497facb879abb961cc2d2812e9233bccf43f35c927d71e1a1caa6e3f1e9932e'
BASE_COUNT=108285
ROUTES='artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json'
ROUTES_SHA='5ea2190dae2b986dc6f7d1b208c0372e2750da68e610639963b697d055e720c0'
TRAIN_MANIFEST='artifacts/framework/protocol/train_manifest.json'
TRAIN_SHA='c414e70ef4a1daf5ea9b2c8adb247576197a9b2decde97479f9006cfa3ca23a1'
WORKER='scripts/run_repaired_policy_case_v1167_r10.py'
PREPARER='our_mani/multitask/repaired_policy_case_v1167.py'
CONSUMER='scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
ORIGINAL_RUNTIME='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'
WRAPPER='scripts/run_frozen_system_environment_v820_r10.py'
SIM='scripts/run_local_sim.sh'
GPU_SIM='scripts/run_aligned_gpu.sh'
ENV_MANIFEST='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json'
ENV_QUALIFICATION='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json'
ENV_SHA='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
CONTEXT='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt'
CONTEXT_SHA='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef'
PHYSICAL_OVERLAY=dict(root='project',
    path='artifacts/framework/multitask_r10/TRAIN_native_release_full16_v1165_setup/source_overlay_lock.json',
    sha256='7d1b53b6a56bd48012e0850320b859d765b7c4b6d32f94f6aa607a22f8eef929')
FACTORIES=dict(core_geometry=('our_mani/multitask/author_native_release_core_v1164.py','integrated_components'),
    block_geometry=('our_mani/multitask/retained_preclose_block_integration_v1153.py','integrated_block_components'),
    physmani=(CONSUMER,'_baseline'))
# Reviewed transitive local imports absent from the immutable745 map. Neither
# module is changed or invoked here. A declaration is not a static proof that
# every possible Python/dynamic import has been enumerated.
REVIEWED_SOURCE_PINS={
    'our_mani/multitask/three_method_release16.py':'faf77babdd431be41c5765a34dfb1b08a18e7bf8ec1a70b6f26b78f7210477f3',
    'our_mani/multitask/track_dataset.py':'22c89adb80f9e1de758ed157cfde78e61bc1a314764fbef16c0246cf81dcecb8',
}
BLOCKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
COMPONENTS=dict(train_owner='scripts/run_repaired_policy_owner_v1169_r10.py',
    evaluation_owner='scripts/run_repaired_policy_owner_v1169_r10.py',
    train_auditor='our_mani/multitask/repaired_readiness_v1144.py',
    evaluation_auditor='our_mani/multitask/repaired_validation_audit_v1170.py',
    admission_guard='our_mani/multitask/repaired_admission_v1147.py',
    admission_driver='our_mani/multitask/repaired_admission_v1147.py',
    formal_source_reader='our_mani/multitask/repaired_admission_v1147.py',
    split_source_lock_writer='our_mani/multitask/repaired_validation_inputs_v1148.py',
    resource_guard='our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
    component_purpose='our_mani/multitask/rml_component_domain_v640.py',
    environment_wrapper=WRAPPER,graph_preparation_helper=SOURCE)
COMPONENT_CALLABLES=dict(train_owner=('main',),evaluation_owner=('main',),
    train_auditor=('audit_fresh_train192','require_readiness_proof'),evaluation_auditor=('audit_validation',),
    admission_guard=('admission_scope',),admission_driver=('create_admission',),
    formal_source_reader=('read_formal_source',),split_source_lock_writer=('write_validation_inputs',),
    resource_guard=('reserved_resources_after_TEST','require_permit'),
    component_purpose=('for_train','for_evaluation'),environment_wrapper=('main',),graph_preparation_helper=('main',))
PREDECESSOR_VERIFIER='scripts/run_TRAIN_native_release_comparison_owner_v1166_r10.py'
PREDECESSOR_VERIFIER_TEST='tests/test_train_native_release_comparison_owner_v1166.py'
CONTROL_SOURCES=tuple(dict.fromkeys((*COMPONENTS.values(),graphs.SOURCE,WORKER,PREPARER,SIM,GPU_SIM,PREDECESSOR_VERIFIER)))
CONTROL_TESTS=('tests/test_repaired_policy_graph_v1143.py','tests/test_repaired_readiness_v1144.py',
    'tests/test_repaired_policy_case_v1167.py','tests/test_repaired_admission_v1147.py',
    'tests/test_repaired_validation_inputs_v1148.py','tests/test_repaired_policy_owner_v1169.py',
    'tests/test_repaired_validation_audit_v1170.py',PREDECESSOR_VERIFIER_TEST,TEST_SOURCE)
PREFIX='artifacts/framework/multitask_r10/'


def require(value,message):
    if not value:raise ValueError('prepare1168: '+message)


def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def path(root,name):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
        and Path(name).as_posix()==name and '\\' not in name,'canonical relative name required')
    file=Path(root)/name;require(file.resolve(strict=False)==file,'symlink/alias traversal refused')
    return file


def pointer(root,name):
    file=path(root,name);require(file.is_file(),'existing explicit file required: '+name)
    return dict(root='project',path=name,sha256=sha(file))


def metadata(root,reference,*,filename=None):
    name=reference if type(reference) is str else reference.get('path')
    require(type(name) is str and name.startswith(PREFIX)
        and (filename is None or Path(name).name==filename),'typed preparation metadata required before read')
    ref=pointer(root,name) if type(reference) is str else reference
    value,normalized=graphs.read_project_json(root,ref);return value,normalized


def fresh(root,name,value):
    file=path(root,name);file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return pointer(root,name)


def imported(root,name):
    module=importlib.import_module(name[:-3].replace('/','.'))
    require(Path(module.__file__).resolve()==Path(root)/name,'actual registered module file required')
    return module


def selected_physical_overlay(root,reference):
    """Reject an old/unselected source envelope before opening it."""
    name=reference if type(reference) is str else reference.get('path') if type(reference) is dict else None
    require(name==PHYSICAL_OVERLAY['path'],'exact frozen1165 physical source overlay required')
    if type(reference) is dict:
        require(reference.get('root','project')=='project' and reference.get('sha256')==PHYSICAL_OVERLAY['sha256'],
            'exact frozen1165 physical overlay SHA/root required')
    actual=pointer(root,name)
    require(actual==PHYSICAL_OVERLAY,'frozen1165 physical source overlay bytes changed')
    return actual


def callable_metadata(root,source,names):
    """Lexical function declarations only: do not import or run control code."""
    reference=pointer(root,source)
    tree=ast.parse(path(root,source).read_bytes())
    functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    require(type(names) in (tuple,list) and bool(names) and len(names)==len(set(names)),
            'explicit distinct source callable names required')
    require(all(type(name) is str and name in functions for name in names),'registered callable missing from actual source: '+source)
    def detail(name):
        node=functions[name];args=node.args
        return dict(qualified_name=source[:-3].replace('/','.')+'.'+name,
            first_def_line=node.lineno,last_def_line=node.end_lineno,
            async_function=isinstance(node,ast.AsyncFunctionDef),
            positional_only=[a.arg for a in args.posonlyargs],positional=[a.arg for a in args.args],
            keyword_only=[a.arg for a in args.kwonlyargs],
            vararg=None if args.vararg is None else args.vararg.arg,
            kwarg=None if args.kwarg is None else args.kwarg.arg,
            decorators=[ast.unparse(d) for d in node.decorator_list])
    return dict(source=reference,callables={name:detail(name) for name in names},
        metadata_kind='AST_source_declaration_not_runtime_invocation',code_imported_or_executed=False)


def reviewed_source_refs(root):
    result={}
    for name,expected in REVIEWED_SOURCE_PINS.items():
        reference=pointer(root,name)
        require(reference['sha256']==expected,'reviewed transitive source bytes changed: '+name)
        result[name]=reference
    return result


def _substrate(root):
    require(sha(path(root,BASE_INPUTS))==BASE_SHA and sha(path(root,ROUTES))==ROUTES_SHA
        and sha(path(root,TRAIN_MANIFEST))==TRAIN_SHA,'unchanged original745 metadata required')
    base=json.loads(path(root,BASE_INPUTS).read_text());files=base['files']
    require(type(files) is dict and len(files)==BASE_COUNT and all(Path(k).is_absolute() for k in files),
        'complete original absolute-key745 input map required')
    routes=json.loads(path(root,ROUTES).read_text())['routes']
    require(len(routes)==48 and {(r['task'],r['method']) for r in routes}==
        {(t,m) for t in graphs.TASKS for m in graphs.METHODS},'exact original48 routes required')
    configs={}
    for route in routes:
        value,ref=graphs.read_project_json(root,route['configuration']);configs[(route['task'],route['method'])]=value
        require(files.get(str(Path(root)/ref['path']))==ref['sha256'],'original config not in745 map')
    native={v['baseline_repo'] for v in configs.values()}
    require(len(native)==1,'single original Native repository required')
    native=Path(next(iter(native)));require(native.is_absolute() and native.is_dir() and native.resolve()==native,'canonical Native root required')
    profile=next(iter(configs.values()))['common_author_RML_profile']
    require(all(v['common_author_RML_profile']==profile for v in configs.values()),'same complete530 profile for all methods required')
    common=imported(root,graphs.RML_SOURCE);common.fingerprint(profile)
    for name,expected in profile['sources_sha256'].items():
        require(pointer(root,name)['sha256']==expected,'original local530 source changed')
    return base,routes,configs,native,profile


def _model_pointer(root,native,files,value):
    if type(value) is dict:
        require(value.get('root','project')=='project','configured model pointer must remain in project')
        name=value['path'];file=path(root,name);expected=value['sha256'];alias='project'
    else:
        require(type(value) is str,'configured model path required')
        file=Path(value) if Path(value).is_absolute() else path(root,value)
        require(file.resolve()==file and file.is_file(),'canonical configured model required')
        if file.is_relative_to(root):alias='project';name=str(file.relative_to(root))
        else:
            require(file.is_relative_to(native),'only released Native-repository external models supported')
            alias='native';name=str(file.relative_to(native))
        expected=files.get(str(file))
    require(file.is_file() and type(expected) is str and len(expected)==64 and files.get(str(file))==expected
        and file.suffix not in graphs.DATASET_SUFFIXES,'configured model must be declared in original745 map; no sample pointer')
    require(sha(file)==expected,'configured model bytes differ from frozen original745 pointer')
    return dict(root=alias,path=name,sha256=expected)


def _models(root,native,files,route,config):
    task,method=route['task'],route['method'];models={};checkpoints={}
    if method=='physmani':
        checkpoint=_model_pointer(root,native,files,config['baseline_checkpoint'])
        require(Path(checkpoint['path']).name=='epoch_99999.pth','original released baseline checkpoint required')
        models['released_neural_policy']=checkpoint;checkpoints['released_neural_policy']=checkpoint
    else:
        require(config.get('context_checkpoint')==dict(path=CONTEXT,sha256=CONTEXT_SHA),'unchanged671 geometry checkpoint required')
        checkpoint=_model_pointer(root,native,files,config['context_checkpoint'])
        # This is a configured/preflight checkpoint for CV, not a claim that CV
        # performs Context inference. All Context predictions use this same671.
        checkpoints['configured_shared_context']=checkpoint
        if task in BLOCKS:
            models['public_goal_marker_calibration']=_model_pointer(root,native,files,config['goal_calibration'])
            models['configured_shared_block_context']=checkpoint
        else:
            for label,value in config['learned_artifacts'][task].items():
                if value is not None:models['configured_'+label]=_model_pointer(root,native,files,value)
            for label,value in (('ring_affordance',config.get('ring_affordances',{}).get(task)),):
                if value is not None:models[label]=_model_pointer(root,native,files,value)
    require(bool(models),'honest nonempty actual model map required')
    return models,checkpoints


def collect_inventory(root,physical_overlay_ref):
    root=Path(root).resolve();physical_overlay_ref=selected_physical_overlay(root,physical_overlay_ref)
    reviewed=reviewed_source_refs(root)
    base,routes,configs,native,profile=_substrate(root)
    require(set(graphs.COMPONENTS)<=set(COMPONENTS),'every actual1143 control role must be explicitly registered')
    physical,physical_ref=metadata(root,physical_overlay_ref,filename='source_overlay_lock.json')
    require(physical.get('integration_complete') is True and physical.get('unpinned_sources')==[]
        and set(physical.get('files',{}))==set(physical.get('required_sources',[]))
        and physical.get('original_745_execution_acceptance_granted') is False
        and physical.get('formal_VAL_TEST_qualification') is False,'complete physical-only provenance required')
    for name,expected in physical['files'].items():require(pointer(root,name)['sha256']==expected,'physical source changed: '+name)
    cpu,cpu_ref=metadata(root,physical['integration_cpu_evidence'])
    require(cpu.get('complete') is True and type(cpu.get('tests_exit_code')) is int and cpu['tests_exit_code']==0
        and cpu.get('native_started') is False and cpu.get('GPU_started') is False
        and cpu.get('source_hashes')==physical['files'],'actual physical integration CPU provenance required')
    factories={};factory_sources=[]
    for branch,key,callable_name in (('core_geometry','core_factory','integrated_components'),
            ('block_geometry','block_factory','integrated_block_components')):
        entry=physical.get(key,{})
        require(entry.get('function')==callable_name and entry.get('module')==entry.get('source','')[:-3].replace('/','.')
            and physical['files'].get(entry.get('source'))==entry.get('sha256'),'source-bound physical factory required: '+branch)
        source=pointer(root,entry['source']);module=imported(root,entry['source'])
        factory_sources.extend(getattr(module,'SOURCES',(entry['source'],)))
        factories[branch]=dict(source=source,callable=callable_name)
    factories['physmani']=dict(source=pointer(root,CONSUMER),callable='_baseline')
    require(all((entry['source']['path'],entry['callable'])==FACTORIES[branch] for branch,entry in factories.items()),
            'only selected1164/1153/744 physical factories supported')
    worker=imported(root,PREPARER)
    require(worker.PHYSICAL_SOURCE_OVERLAY==physical_ref,'actual1167 worker must pin the selected physical overlay')
    registry_matches=all(worker.FACTORIES[k]==(v['source']['path'],v['callable']) for k,v in factories.items())
    substrate=imported(root,'our_mani/multitask/rml_case_preflight_v742.py')
    physical_names=set(physical['required_sources'])|set(factory_sources)|set(substrate.SOURCES)|set(reviewed)|{
        CONSUMER,ORIGINAL_RUNTIME,graphs.RML_SOURCE,graphs.OUTCOME_SOURCE,*profile['sources_sha256']}
    controls=set(CONTROL_SOURCES)|set(substrate.SOURCES)|set(reviewed)
    tests={p for p in physical_names|controls if p.startswith('tests/')}|set(CONTROL_TESTS)
    def code_map(names):
        return {'project:'+name:pointer(root,name) for name in sorted(names)
            if Path(name).suffix in graphs.SOURCE_SUFFIXES and not name.startswith('tests/')}
    ps=code_map(physical_names);cs=code_map(controls)
    for name,expected in profile['native_sources_sha256'].items():
        file=native/name;require(file.resolve()==file and sha(file)==expected,'original Native530 source changed')
        ps['native:'+name]=dict(root='native',path=name,sha256=expected)
    test_refs={name:pointer(root,name) for name in sorted(tests)}
    required_cpu={str(Path(root)/p['path']):p['sha256'] for p in (*ps.values(),*cs.values(),*test_refs.values()) if p['root']=='project'}
    required_cpu.update({str(native/p['path']):p['sha256'] for p in ps.values() if p['root']=='native'})
    rows=[]
    for route in routes:
        config=configs[(route['task'],route['method'])];models,checkpoints=_models(root,native,base['files'],route,config)
        branch='physmani' if route['method']=='physmani' else ('block_geometry' if route['task'] in BLOCKS else 'core_geometry')
        rows.append(dict(task=route['task'],method=route['method'],configuration=route['configuration'],models=models,
            checkpoints=checkpoints,physical_factory=branch,GPU_required=route['method']=='physmani'))
    require(set(COMPONENT_CALLABLES)==set(COMPONENTS),'every control role needs actual function metadata')
    functions={role:callable_metadata(root,name,COMPONENT_CALLABLES[role]) for role,name in COMPONENTS.items()}
    worker_functions=dict(worker=callable_metadata(root,WORKER,('run_case','main')),
        preparer=callable_metadata(root,PREPARER,('prepare','runtime_scope','admission_scope')))
    physical_functions={branch:callable_metadata(root,entry['source']['path'],(entry['callable'],))
                        for branch,entry in factories.items()}
    dynamic_functions=callable_metadata(root,PREDECESSOR_VERIFIER,('validate','completed_case'))
    return dict(schema=SCHEMA,roots=dict(project=str(root),native=str(native)),physical_sources_map=ps,control_sources_map=cs,
        control_components={k:pointer(root,v) for k,v in COMPONENTS.items()},factories=factories,models_by_route=rows,
        actual_control_callables=dict(admission_driver='our_mani.multitask.repaired_admission_v1147.create_admission',
            split_source_lock_writer='our_mani.multitask.repaired_validation_inputs_v1148.write_validation_inputs'),
        control_function_metadata=functions,worker_function_metadata=worker_functions,
        physical_function_metadata=physical_functions,
        predecessor_verifier_function_metadata=dynamic_functions,
        reviewed_extra_transitive_sources=reviewed,
        test_sources=test_refs,physical_overlay=physical_ref,physical_CPU_evidence=cpu_ref,
        physical_integrity_files={name:pointer(root,name) for name in physical['files']},
        old745_inputs=pointer(root,BASE_INPUTS),old745_key_count=BASE_COUNT,original_routes=pointer(root,ROUTES),
        original_TRAIN_manifest=pointer(root,TRAIN_MANIFEST),common_RML_profile=profile,
        author_outcome_protocol=base['author_outcome_protocol'],
        environment_manifest=pointer(root,ENV_MANIFEST),environment_namespace_qualification=pointer(root,ENV_QUALIFICATION),
        required_control_CPU_source_hashes=required_cpu,
        worker_registry_matches_selected_physical_factories=registry_matches,
        legacy_immutable_integrity_lock_retained=True,legacy_integrity_domain_is_not_TRAIN_only=True,
        no_VAL_TEST_sample_row_result_parsing_or_selection=True,
        model_branch_declarations=dict(physmani='released_epoch99999_and_original_online_world_model_only',
            baseline_unused_geometry_config_fields_not_loaded=['context_checkpoint','learned_artifacts'],
            geometry_configured_shared_checkpoint='671; CV declaration is not Context inference'),
        complete_source_closure_is_Root_reviewed_declaration_not_static_program_proof=True,
        inventory_grants_readiness_or_admission=False,Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False)


def write_inventory(root,physical_overlay_ref,*,namespace):
    root=Path(root).resolve();require(namespace.startswith(PREFIX),'new preparation namespace required')
    inventory=collect_inventory(root,physical_overlay_ref)
    directory=path(root,namespace);directory.mkdir(parents=True,exist_ok=False)
    return fresh(root,namespace+'/source_inventory.json',inventory)


def _merge(root,base_files,additions):
    merged=dict(base_files)
    # Do not normalize/rewrite108285 keys. Avoid a second canonical alias when
    # adding new relative source/metadata names to this absolute-key substrate.
    by_path={}
    for name in base_files:by_path.setdefault(str(Path(name).resolve()),[]).append(name)
    for file,expected in additions:
        file=Path(file);require(file.is_absolute() and file.resolve()==file,'canonical new executable input required')
        aliases=by_path.get(str(file),[])
        if aliases:
            require(len(aliases)==1 and merged[aliases[0]]==expected,'new key conflicts with canonical old input')
        else:merged[str(file)]=expected;by_path[str(file)]=[str(file)]
    require(all(merged.get(k)==v for k,v in base_files.items()),'original745 key/hash mutation refused')
    return merged


def _control_cpu(root,reference,inventory):
    value,reference=metadata(root,reference,filename=CONTROL_CPU_NAME)
    require(value.get('schema')==CONTROL_CPU_SCHEMA and value.get('complete') is True
        and type(value.get('tests_exit_code')) is int and value['tests_exit_code']==0
        and value.get('native_started') is False and value.get('GPU_started') is False
        and type(value.get('source_hashes')) is dict,'Root-observed complete CPU evidence required')
    normalized={}
    for name,checksum in value['source_hashes'].items():
        require(type(name) is str,'plain CPU source path required')
        file=Path(name) if Path(name).is_absolute() else path(root,name)
        require(str(file) not in normalized,'CPU source alias duplication refused')
        normalized[str(file)]=checksum
    # No arbitrary extra path is opened here (especially no source manifest or
    # sample). Root can generate this exact source-only map from the inventory.
    require(normalized==inventory['required_control_CPU_source_hashes'],
        'CPU evidence must bind the entire reviewed physical/control/test source map exactly')
    return reference


def pilot_argv(root,graph_ref,graph,*,label):
    """Three TRAIN --check commands only; this helper never executes them."""
    manifest=json.loads(path(root,TRAIN_MANIFEST).read_text());rows=manifest['episodes']
    selections=(('core_geometry','insert_onto_rotating_peg','cv_geometry'),
        ('block_geometry',BLOCKS[0],'context_geometry'),('physmani','beat_the_rotating_buzz','physmani'))
    result=[]
    for branch,task,method in selections:
        raw=next(row['raw_episode_id'] for row in rows if row['task']==task)
        route=next(row for row in graph['routes'] if (row['task'],row['method'])==(task,method))
        suffix='cpu_pilot_'+label+'/'+task+'/'+raw+'/'+method
        argv=[str(Path(root)/'.conda-env/bin/python'),'-B',WRAPPER,'--manifest-sha256',ENV_SHA,'--','bash',
            GPU_SIM if method=='physmani' else SIM,route['train']['worker']['path'],
            '--graph',graph_ref['path'],'--config',route['configuration']['path'],'--split','train',
            '--task',task,'--raw-episode-id',raw,'--method',method,
            '--out','artifacts/framework/multitask_r10/repaired_policy_v1167/train/development_demo_only/'+suffix,
            '--runtime','runtime/framework/repaired_policy_v1167/train/'+suffix,'--check']
        result.append(dict(branch=branch,task=task,method=method,raw_episode_id=raw,argv=argv,
            command_only=True,Native_started=False,GPU_started=False))
    return result


def prepare_only(root,inventory_ref,control_cpu_ref,*,namespace,inventory_reviewed=False):
    """Root-invoked first-write data preparation; no launch or admission."""
    root=Path(root).resolve()
    require(inventory_reviewed is True,'Root must review the inventory before explicit preparation')
    inventory,inventory_ref=metadata(root,inventory_ref,filename='source_inventory.json')
    require(inventory.get('schema')==SCHEMA,'new preparation inventory schema required')
    actual=collect_inventory(root,inventory['physical_overlay'])
    require(actual==inventory,'source/model/config inventory changed; regenerate and review')
    require(inventory['worker_registry_matches_selected_physical_factories'] is True,
        'final1167 registry must match selected physical core/block before graph preparation')
    cpu_ref=_control_cpu(root,control_cpu_ref,inventory)
    require(type(namespace) is str and namespace.startswith(PREFIX),'new preparation namespace required')
    directory=path(root,namespace);require(not directory.exists(),'fresh preparation namespace required')
    # All safety/source/review checks precede the first write.
    directory.mkdir(parents=True,exist_ok=False)
    profile_ref=fresh(root,namespace+'/common_RML_profile.json',inventory['common_RML_profile'])
    protocol_ref=fresh(root,namespace+'/author_outcome_protocol.json',inventory['author_outcome_protocol'])
    base=json.loads(path(root,BASE_INPUTS).read_text())
    additions=[];roots=inventory['roots']
    for p in (*inventory['physical_sources_map'].values(),*inventory['control_sources_map'].values(),*inventory['test_sources'].values(),
            *inventory.get('physical_integrity_files',{}).values()):
        additions.append((Path(roots[p['root']])/p['path'],p['sha256']))
    refs=(inventory_ref,cpu_ref,inventory['physical_overlay'],inventory['physical_CPU_evidence'],profile_ref,protocol_ref,
        inventory['environment_manifest'],inventory['environment_namespace_qualification'])
    additions.extend((root/p['path'],p['sha256']) for p in refs)
    files=_merge(root,base['files'],additions)
    merged=fresh(root,namespace+'/executable_inputs.json',dict(schema='r10_merged_repaired_executable_inputs_v1168',
        files=files,old745_substrate=inventory['old745_inputs'],old745_keys_preserved=BASE_COUNT,
        original_745_execution_acceptance_granted=False,formal_VAL_TEST_qualification=False,
        author_outcome_protocol=inventory['author_outcome_protocol'],
        formal_split_data_locked_separately_after_admission=True,TEST_used_for_fitting=False,
        legacy_immutable_integrity_lock_retained=True,legacy_integrity_domain_is_not_TRAIN_only=True,
        no_VAL_TEST_sample_row_result_parsing_or_selection=True,
        no_self_graph_cohort_or_future_evidence_in_input_map=True))
    routes=[]
    for original in inventory['models_by_route']:
        row=deepcopy(original);row['frozen_inputs']=merged
        for purpose,key,split in (('train_qualification','train','train'),('evaluation','evaluation','validation')):
            row[key]=dict(purpose=purpose,actual_split=split,worker=pointer(root,WORKER),preparer=pointer(root,PREPARER),
                consumer=pointer(root,CONSUMER),case_schema='r10_graph_bound_repaired_policy_case_v1167_'+
                    ('TRAIN' if split=='train' else 'validation')+'_case',physical_factory=row['physical_factory'],
                factory_control='declared_frozen_factory_only',runtime_policy_override_allowed=False)
        routes.append(row)
    ps=inventory['physical_sources_map'];cs=inventory['control_sources_map']
    spec=dict(schema=graphs.SPEC_SCHEMA,tasks=list(graphs.TASKS),methods=list(graphs.METHODS),
        external_roots={k:v for k,v in roots.items() if k!='project'},train_source_manifest=inventory['original_TRAIN_manifest'],
        physical=dict(factories=inventory['factories'],sources_map=ps,complete_source_closure_declared=True,
            source_overlay_lock=inventory['physical_overlay'],environment_manifest=inventory['environment_manifest'],
            shared_context_checkpoint=dict(root='project',path=CONTEXT,sha256=CONTEXT_SHA),
            common_RML=dict(source=pointer(root,graphs.RML_SOURCE),profile=profile_ref),
            outcome_sampling=dict(source=pointer(root,graphs.OUTCOME_SOURCE),protocol=protocol_ref)),
        control=dict(sources_map=cs,components=inventory['control_components'],complete_source_closure_declared=True,
            contracts=dict(readiness_schema='r10_fresh_TRAIN192_readiness_v1144',
                admission_schema='r10_repaired_validation_admission_v1147',split_source_lock_schema='r10_full16_post_admission_split_source_lock_v1'),
            static_inputs=dict(environment_namespace_qualification=inventory['environment_namespace_qualification'],
                reviewed_source_inventory=inventory_ref,Root_control_CPU_evidence=cpu_ref)),routes=routes)
    spec_ref=fresh(root,namespace+'/policy_spec.json',spec)
    graph_ref=graphs.write_policy_graph(root,spec,namespace=namespace+'/graph')
    graph=graphs.read_policy_graph(root,graph_ref)
    commands=pilot_argv(root,graph_ref,graph,label=directory.name)
    pilot_ref=fresh(root,namespace+'/cpu_pilot_argv.json',dict(commands=commands,Native_started=False,GPU_started=False))
    receipt=fresh(root,namespace+'/preparation_receipt.json',dict(schema=RECEIPT_SCHEMA,
        inventory=inventory_ref,control_CPU_evidence=cpu_ref,physical_overlay=inventory['physical_overlay'],
        merged_executable_inputs=merged,spec=spec_ref,graph=graph_ref,cpu_pilot_argv=pilot_ref,
        physical_policy_sha256=graph['physical_policy_sha256'],control_graph_sha256=graph['control_graph_sha256'],
        policy_sha256=graph['policy_sha256'],old745_key_values_preserved=True,qualification_granted=False,
        legacy_immutable_integrity_lock_retained=True,no_VAL_TEST_sample_row_result_parsing_or_selection=True,
        Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False))
    return dict(receipt=receipt,graph=graph_ref,merged_executable_inputs=merged,cpu_pilot_argv=pilot_ref,
        Native_started=False,GPU_started=False,qualification_granted=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--inventory-only',action='store_true');mode.add_argument('--prepare-only',action='store_true')
    for name in ('namespace','physical-overlay-lock','inventory','control-cpu-evidence'):parser.add_argument('--'+name)
    parser.add_argument('--inventory-reviewed',action='store_true');args=parser.parse_args(argv)
    if args.inventory_only:
        require(args.namespace and args.physical_overlay_lock,'inventory requires explicit namespace and physical overlay')
        result=write_inventory(ROOT,args.physical_overlay_lock,namespace=args.namespace)
    elif args.prepare_only:
        require(args.namespace and args.inventory and args.control_cpu_evidence,'prepare requires reviewed inventory/CPU evidence/namespace')
        result=prepare_only(ROOT,args.inventory,args.control_cpu_evidence,namespace=args.namespace,inventory_reviewed=args.inventory_reviewed)
    else:result=dict(enabled=False,files_written=False,Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False)
    print(json.dumps(result,ensure_ascii=False));return result


if __name__=='__main__':main()
