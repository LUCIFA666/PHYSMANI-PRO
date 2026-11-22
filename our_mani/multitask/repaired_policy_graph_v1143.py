"""Compile an explicit, acyclic source graph; never qualify or run a policy.

Workers, selectors, guards and auditors must already exist before this graph is
written. Later TRAIN192/readiness/admission/split evidence binds its hashes in a
separate artifact. This compiler hashes only explicitly supplied files, does not
select episodes, inspect dataset samples, import a candidate or launch jobs.
Worker/factory and complete-source declarations remain reviewable declarations,
not a claim that static hashing proves the program's runtime behavior.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

SOURCE = 'our_mani/multitask/repaired_policy_graph_v1143.py'
SPEC_SCHEMA = 'r10_repaired_policy_graph_spec_v1143'
SCHEMA = 'r10_frozen_repaired_policy_graph_v1143'
METHODS = ('cv_geometry','context_geometry','physmani')
TASKS = tuple(sorted(family+speed for family in (
    'beat_the_rotating_buzz','insert_onto_rotating_peg','moving_basketball_in_hoop',
    'pick_moving_target_on_the_table','place_cups_on_rotating_frame','push_moving_button',
    'put_rubbish_in_moving_bin','remove_cups_from_rotating_frame') for speed in ('','_high_speed')))
RML_SOURCE = 'our_mani/multitask/common_rml_protocol_v530.py'
OUTCOME_SOURCE = 'our_mani/multitask/native_outcome_sampling_v741.py'
COMPONENTS = ('train_owner','train_auditor','evaluation_owner','evaluation_auditor',
    'admission_guard','admission_driver','formal_source_reader','split_source_lock_writer',
    'resource_guard','component_purpose','environment_wrapper')
SOURCE_SUFFIXES = {'.py','.sh','.lua','.so','.c','.cpp','.h','.hpp'}
DATASET_SUFFIXES = {'.h5','.hdf5','.npz','.npy','.csv','.tsv','.parquet','.png','.jpg','.jpeg','.mp4'}
EVIDENCE = re.compile(r'(^|[/_.:-])(readiness|admission|cohort|results?|case_receipt|launch_receipt|validation_release|split_inputs?|split_source_lock)([/_.:-]|$)',re.I)
LABEL = re.compile(r'^[A-Za-z0-9_./:-]+$')
SHA = re.compile(r'^[0-9a-f]{64}$')


class PolicyGraphError(ValueError):
    pass


def _require(condition,message):
    if not condition:raise PolicyGraphError(message)


def _text(value,label):
    _require(type(value) is str and bool(value.strip()) and value==value.strip()
        and value.lower() not in ('pending','none','null','todo','tbd','placeholder'),label+' must be complete text')
    return value


def _keys(value,required,label,optional=()):
    _require(type(value) is dict and set(required)<=set(value)
        and set(value)<=set(required)|set(optional),label+' has missing or unsupported fields')


def _relative(name):
    _text(name,'relative path')
    path=PurePosixPath(name)
    _require(not path.is_absolute() and path.as_posix()==name and '\\' not in name
        and name!='.' and '..' not in path.parts,'canonical relative path required')
    return name


def _root(value):
    _require(isinstance(value,(str,Path)),'root path required')
    path=Path(value)
    _require(path.is_absolute() and path.is_dir() and str(path)==str(path.resolve()),'existing canonical absolute root required')
    return path


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _roots(root,external):
    _require(type(external) is dict,'explicit external_roots mapping required')
    result={'project':_root(root)}
    _require(all(type(alias) is str for alias in external),'plain root aliases required')
    for alias,value in sorted(external.items()):
        _require(type(alias) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',alias)
            and alias!='project','unique nonproject root alias required')
        result[alias]=_root(value)
    _require(len(set(result.values()))==len(result),'root aliases must identify distinct roots')
    return result


class _Files:
    def __init__(self,roots):self.roots=roots;self.hashes={}

    def pointer(self,value,*,source=False,metadata=False,no_evidence=False):
        _keys(value,('path','sha256'),'pointer',('root',))
        alias=value.get('root','project');_require(type(alias) is str and alias in self.roots,'declared pointer root required')
        name=_relative(value['path']);checksum=value['sha256']
        _require(type(checksum) is str and SHA.fullmatch(checksum),'explicit lowercase SHA256 required')
        path=self.roots[alias]/name
        _require(path.is_file() and path.resolve()==path,'existing regular file without symlink traversal required: '+name)
        if source:_require(path.suffix in SOURCE_SUFFIXES,'source maps contain source files only')
        if metadata:_require(path.suffix=='.json','JSON metadata pointer required')
        _require(path.suffix.lower() not in DATASET_SUFFIXES,'dataset/sample files are outside a source graph')
        if no_evidence:_require(not EVIDENCE.search(name),'postfreeze evidence cannot enter this graph')
        key=(alias,name)
        if key not in self.hashes:self.hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        _require(self.hashes[key]==checksum,'declared file SHA256 differs: '+name)
        return dict(root=alias,path=name,sha256=checksum)

    def revalidate(self):
        for (alias,name),checksum in self.hashes.items():
            path=self.roots[alias]/name
            _require(path.is_file() and path.resolve()==path and hashlib.sha256(path.read_bytes()).hexdigest()==checksum,
                'declared file changed during graph compilation: '+name)

    def mapping(self,value,*,source=False,no_evidence=False,nonempty=True):
        _require(type(value) is dict and (bool(value) or not nonempty),'explicit pointer mapping required')
        result={}
        for label,pointer in sorted(value.items()):
            _require(type(label) is str and LABEL.fullmatch(label),'stable pointer label required')
            if no_evidence:_require(not EVIDENCE.search(label),'postfreeze evidence label forbidden')
            result[label]=self.pointer(pointer,source=source,no_evidence=no_evidence)
        return result


def _contains(mapping,pointer):
    return pointer in mapping.values()


def _worker(files,value,purpose,factory,control_sources,physical_sources):
    _keys(value,('purpose','actual_split','worker','preparer','consumer','case_schema',
        'physical_factory','factory_control','runtime_policy_override_allowed'),'worker declaration')
    _require(value['purpose']==purpose and value['actual_split']==('train' if purpose=='train_qualification' else 'validation'),
        'truthful TRAIN/evaluation purpose and actual split required')
    _require(value['physical_factory']==factory and value['factory_control']=='declared_frozen_factory_only'
        and value['runtime_policy_override_allowed'] is False,'controlled identical physical factory declaration required')
    result={key:value[key] for key in ('purpose','actual_split','physical_factory','factory_control','runtime_policy_override_allowed')}
    result['case_schema']=_text(value['case_schema'],'actual case schema')
    for key in ('worker','preparer','consumer'):
        pointer=files.pointer(value[key],source=True)
        pool=physical_sources if key=='consumer' else control_sources
        _require(_contains(pool,pointer),'actual '+key+' missing from its complete source map')
        result[key]=pointer
    return result


def compile_policy_graph(root,spec):
    """Validate complete declarations and bytes, returning a deterministic graph.

    No future proof pointer belongs in spec. A missing actual evaluation worker
    is an error, even when a proposed TRAIN worker already exists.
    """
    _keys(spec,('schema','tasks','methods','external_roots','train_source_manifest','physical','control','routes'),'graph spec')
    _require(spec['schema']==SPEC_SCHEMA,'new graph spec schema required')
    _require(type(spec['tasks']) is list and len(spec['tasks'])==16 and all(type(x) is str for x in spec['tasks'])
        and set(spec['tasks'])==set(TASKS),'exact original16 task roster required')
    _require(type(spec['methods']) is list and len(spec['methods'])==3 and all(type(x) is str for x in spec['methods'])
        and set(spec['methods'])==set(METHODS),'exact authorized3 method roster required')
    roots=_roots(root,spec['external_roots']);files=_Files(roots)
    manifest=files.pointer(spec['train_source_manifest'],metadata=True,no_evidence=True)
    physical=spec['physical'];_keys(physical,('factories','source_overlay_lock','sources_map',
        'complete_source_closure_declared','common_RML','outcome_sampling','environment_manifest','shared_context_checkpoint'),'physical graph')
    _require(physical['complete_source_closure_declared'] is True,'complete physical source closure must be explicitly declared')
    ps=files.mapping(physical['sources_map'],source=True)
    factories={}
    _require(type(physical['factories']) is dict and physical['factories'],'physical factories must be explicit')
    for name,value in sorted(physical['factories'].items()):
        _require(type(name) is str and LABEL.fullmatch(name),'factory label required')
        _keys(value,('source','callable'),'physical factory')
        pointer=files.pointer(value['source'],source=True)
        _require(_contains(ps,pointer),'physical factory missing from complete physical source map')
        callable_name=_text(value['callable'],'factory callable')
        _require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*',callable_name),'explicit callable name required')
        factories[name]=dict(source=pointer,callable=callable_name)
    normalized_physical=dict(factories=factories,sources_map=ps,complete_source_closure_declared=True,
        source_overlay_lock=files.pointer(physical['source_overlay_lock'],metadata=True,no_evidence=True),
        environment_manifest=files.pointer(physical['environment_manifest'],metadata=True,no_evidence=True),
        shared_context_checkpoint=files.pointer(physical['shared_context_checkpoint'],no_evidence=True))
    for key,source_name,artifact in (('common_RML',RML_SOURCE,'profile'),('outcome_sampling',OUTCOME_SOURCE,'protocol')):
        _keys(physical[key],('source',artifact),key)
        pointer=files.pointer(physical[key]['source'],source=True)
        _require(pointer['root']=='project' and pointer['path']==source_name and _contains(ps,pointer),'explicit original530/741 source binding required')
        normalized_physical[key]=dict(source=pointer,**{artifact:files.pointer(physical[key][artifact],metadata=True,no_evidence=True)})
    control=spec['control'];_keys(control,('sources_map','components','contracts','static_inputs','complete_source_closure_declared'),'control graph')
    _require(control['complete_source_closure_declared'] is True,'complete control source closure must be explicitly declared')
    cs=files.mapping(control['sources_map'],source=True)
    compiler=files.pointer(dict(path=SOURCE,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),source=True)
    _require(_contains(cs,compiler),'actual graph compiler must be in control source map')
    _require(type(control['components']) is dict and set(COMPONENTS)<=set(control['components']),'all actual control components must exist before freeze')
    components=files.mapping(control['components'],source=True)
    _require(all(_contains(cs,pointer) for pointer in components.values()),'control components missing from complete source map')
    _keys(control['contracts'],('readiness_schema','admission_schema','split_source_lock_schema'),'future evidence contracts')
    contracts={name:_text(value,'future evidence schema') for name,value in sorted(control['contracts'].items())}
    _require(contracts['readiness_schema']!='r10_exact16_fixed_four_common_author_RML_readiness_v532',
        'old532 readiness cannot qualify a repaired graph')
    normalized_control=dict(sources_map=cs,components=components,contracts=contracts,
        static_inputs=files.mapping(control['static_inputs'],no_evidence=True,nonempty=False),complete_source_closure_declared=True)
    _require(type(spec['routes']) is list and len(spec['routes'])==48,'exact48 routes required')
    routes=[];seen=set()
    for row in spec['routes']:
        _keys(row,('task','method','configuration','frozen_inputs','models','checkpoints','physical_factory','GPU_required','train','evaluation'),'route')
        task,method=row['task'],row['method'];key=(task,method)
        _require(type(task) is str and type(method) is str and task in TASKS and method in METHODS and key not in seen,'duplicate or unknown task/method route')
        seen.add(key);factory=row['physical_factory']
        _require(type(factory) is str and factory in factories,'declared physical factory required')
        _require(row['GPU_required'] is (method=='physmani'),'GPU declaration must match actual method')
        models=files.mapping(row['models'],no_evidence=True)
        checkpoints=files.mapping(row['checkpoints'],no_evidence=True,nonempty=method!='cv_geometry')
        if method=='context_geometry':_require(_contains(checkpoints,normalized_physical['shared_context_checkpoint']),'all Context routes must bind declared shared checkpoint')
        routes.append(dict(task=task,method=method,configuration=files.pointer(row['configuration'],metadata=True,no_evidence=True),
            frozen_inputs=files.pointer(row['frozen_inputs'],metadata=True,no_evidence=True),models=models,checkpoints=checkpoints,
            physical_factory=factory,GPU_required=row['GPU_required'],
            train=_worker(files,row['train'],'train_qualification',factory,cs,ps),
            evaluation=_worker(files,row['evaluation'],'evaluation',factory,cs,ps)))
    _require(seen=={(task,method) for task in TASKS for method in METHODS},'incomplete original16 by3 route grid')
    files.revalidate()
    routes.sort(key=lambda row:(row['task'],row['method']))
    physical_rows=[{key:value for key,value in row.items() if key not in ('train','evaluation')} for row in routes]
    physical_sha=_digest(dict(schema='r10_physical_policy_fingerprint_v1143',physical=normalized_physical,routes=physical_rows,tasks=list(TASKS),methods=list(METHODS)))
    root_names={name:str(path) for name,path in sorted(roots.items())}
    control_sha=_digest(dict(schema='r10_control_graph_fingerprint_v1143',physical_policy_sha256=physical_sha,
        roots=root_names,control=normalized_control,train_source_manifest=manifest,
        workers=[dict(task=row['task'],method=row['method'],train=row['train'],evaluation=row['evaluation']) for row in routes]))
    policy_sha=_digest(dict(schema='r10_repaired_policy_fingerprint_v1143',physical_policy_sha256=physical_sha,control_graph_sha256=control_sha))
    return dict(schema=SCHEMA,roots=root_names,tasks=list(TASKS),methods=list(METHODS),train_source_manifest=manifest,
        physical=normalized_physical,control=normalized_control,routes=routes,
        physical_policy_sha256=physical_sha,control_graph_sha256=control_sha,policy_sha256=policy_sha,
        source_graph_complete=True,source_closure_and_worker_control_are_declarations_not_execution_proof=True,
        future_evidence_pointers_included=False,readiness_generated=False,Native_started_by_compiler=False,
        qualification_or_admission_granted=False,formal_VAL_TEST_qualified=False)


def read_project_json(root,pointer):
    """Read one explicitly addressed, hash-bound project JSON metadata object."""
    files=_Files({'project':_root(root)})
    normalized=files.pointer(pointer,metadata=True)
    data=(files.roots['project']/normalized['path']).read_bytes()
    _require(hashlib.sha256(data).hexdigest()==normalized['sha256'],'JSON metadata changed while reading')
    value=json.loads(data)
    _require(type(value) is dict,'JSON metadata object required')
    return value,normalized


def read_policy_graph(root,graph_ref):
    graph,_=read_project_json(root,graph_ref)
    _require(graph.get('schema')==SCHEMA,'new repaired source graph schema required')
    roots=graph.get('roots',{})
    _require(type(roots) is dict and roots.get('project')==str(_root(root)),'graph project root differs')
    _require(all(key in graph for key in ('tasks','methods','train_source_manifest','physical','control','routes')),
        'stored graph declarations incomplete')
    spec={key:graph[key] for key in ('tasks','methods','train_source_manifest','physical','control','routes')}
    spec.update(schema=SPEC_SCHEMA,external_roots={name:path for name,path in roots.items() if name!='project'})
    expected=compile_policy_graph(root,spec)
    _require(graph==expected,'stored graph bytes or fingerprints differ from complete declarations')
    return graph


def write_policy_graph(root,spec,*,namespace):
    """First-write one new explicit namespace, after full validation succeeds."""
    graph=compile_policy_graph(root,spec);name=_relative(namespace)
    _require(name.startswith('artifacts/framework/multitask_r10/') and len(PurePosixPath(name).parts)>3,
        'explicit fresh artifacts/framework/multitask_r10 namespace required')
    project=_root(root);directory=project/name
    _require(directory.parent.is_dir() and directory.parent.resolve()==directory.parent,'existing canonical namespace parent required')
    # mkdir is the exclusive first-write claim. No existing directory is reused.
    directory.mkdir(exist_ok=False)
    path=directory/'policy_graph.json'
    data=(json.dumps(graph,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path.relative_to(project)),sha256=hashlib.sha256(data).hexdigest())
