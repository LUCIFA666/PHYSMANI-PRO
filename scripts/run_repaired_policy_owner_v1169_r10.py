"""Explicit graph-bound TRAIN192 or admitted VAL945; inert by default.

Each CPU/native child owns its1106 lease. The parent holds only its owner lock;
its short validation planning capability has exited before any Popen. No TEST,
retry, outcome filtering or self-issued tool-exit/readiness proof is supported.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from our_mani.multitask import repaired_policy_graph_v1143 as graphs
from our_mani.multitask import repaired_readiness_v1144 as readiness

SOURCE='scripts/run_repaired_policy_owner_v1169_r10.py'
TEST_SOURCE='tests/test_repaired_policy_owner_v1169.py'
WORKER='scripts/run_repaired_policy_case_v1167_r10.py'
PREPARER='our_mani/multitask/repaired_policy_case_v1167.py'
CONSUMER='scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
WRAPPER='scripts/run_frozen_system_environment_v820_r10.py'
GUARD='our_mani/multitask/post_completed_TEST_resource_guard_v1106.py'
GATE='our_mani/multitask/repaired_admission_v1147.py'
WRITER='our_mani/multitask/repaired_validation_inputs_v1148.py'
ENV_SHA='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
TRAIN_MANIFEST='artifacts/framework/protocol/train_manifest.json'
BASE='runtime/framework/repaired_policy_owner_v1169'
ART='artifacts/framework/multitask_r10/repaired_policy_v1167'
RUNTIME='runtime/framework/repaired_policy_v1167'
SCHEMA='r10_repaired_graph_bound_serial_owner_v1169'
VAL_PLAN_SCHEMA='r10_fresh_repaired_validation_execution_plan_v1169'
VAL_COHORT_SCHEMA='r10_fresh_repaired_validation_execution_cohort_v1170'
REVIEW_SCHEMA='r10_repaired_graph_owner_technical_review_v1169'
BUNDLE_SCHEMA='r10_post_admission_repaired_validation_inputs_v1148'
BUNDLE_PREFIX='artifacts/framework/multitask_r10/repaired_validation_inputs_v1148/'
CASE_SCHEMAS={s:'r10_graph_bound_repaired_policy_case_v1167_'+('TRAIN' if s=='train' else 'validation')+'_case'
    for s in ('train','validation')}
SMOKE_SOURCE='scripts/run_TRAIN_native_release_comparison_owner_v1166_r10.py'
SMOKE_SOURCE_SHA='4796fe37bf495e7becc93bf0d7d17dbe51a62f9ab1822d2e8a999d4eb825164d'
SMOKE='runtime/framework/TRAIN_native_release_full16_v1165/owner_v1166'
SMOKE_PLAN='artifacts/framework/multitask_r10/TRAIN_native_release_full16_v1165_setup/bounded_plan_v1166.json'
SMOKE_PLAN_SHA='f83204a5661e77866249400b201570bd479ba15bcd4888007c2ee700ad8aeb8d'
SMOKE_EXIT_SCHEMA='r10_actual_natural_native_release_comparison_owner_exit_v1166'
SMOKE_PROCESS=dict(pid=1057833,start_ticks=13094334)
SMOKE_LOCK=dict(path='artifacts/framework/multitask_r10/TRAIN_native_release_full16_v1165_setup/source_overlay_lock.json',
    sha256='7d1b53b6a56bd48012e0850320b859d765b7c4b6d32f94f6aa607a22f8eef929')
SMOKE_STOP=dict(path='runtime/framework/repaired_policy_owner_v1149/train/conservative_v1159_train192/controlled_stop_receipt.json',
    sha256='2b338d68ad079facad2703d0d1393980f86611d2a64095bf75b77d0eddff5bad')
SMOKE_CASES=(('moving_basketball_in_hoop','episode104','cv_geometry'),
    ('moving_basketball_in_hoop','episode104','context_geometry'),
    ('moving_basketball_in_hoop_high_speed','episode0','context_geometry'),
    ('beat_the_rotating_buzz','episode10','cv_geometry'),
    ('beat_the_rotating_buzz_high_speed','episode102','context_geometry'))


def require(value,message):
    if not value:raise ValueError('owner1169: '+message)


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path(root,name):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
        and Path(name).as_posix()==name and '\\' not in name,'canonical project-relative path required')
    result=Path(root)/name
    require(result.resolve(strict=False)==result,'symlink/alias traversal forbidden')
    return result


def pointer(root,name):
    file=path(root,name);require(file.is_file(),'actual file required: '+name)
    return dict(root='project',path=name,sha256=sha(file))


def ref(root,value):return pointer(root,value) if type(value) is str else readiness.pointer(value)
def read(root,reference):return readiness.read_json(root,reference)
def same(a,b):return readiness.same(a,b)


def artifact_ref(root,value,*,prefix,basename=None):
    """Type the external metadata name BEFORE hashing or opening its bytes."""
    declared=dict(root='project',path=value) if type(value) is str else readiness.pointer(value)
    require(declared.get('root')=='project' and type(declared.get('path')) is str
        and declared['path'].startswith(prefix)
        and (basename is None or Path(declared['path']).name==basename),'explicit typed metadata namespace required')
    path(root,declared['path'])  # Reject symlinks/aliases without opening contents.
    return ref(root,value)


def exact_ref(root,value,expected):
    declared=dict(root='project',path=value) if type(value) is str else readiness.pointer(value)
    require(declared.get('root')=='project' and declared.get('path')==expected,'actual typed evidence path differs')
    path(root,expected)
    return ref(root,value)


def fresh(file,value):
    file=Path(file);require(file.resolve(strict=False)==file,'canonical fresh output required')
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def atomic(file,value):
    file=Path(file);temporary=file.with_name(file.name+'.tmp.'+str(os.getpid()))
    fresh(temporary,value);os.replace(temporary,file)


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid
    require(type(pid) is int and pid>0,'actual PID required')
    fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]))


def stamp(graph_ref,graph):return dict(graph=graph_ref,**{k:graph[k] for k in readiness.HASHES})


def split_count(split):
    require(split in ('train','validation'),'only TRAIN or validation supported; TEST refused')
    return 192 if split=='train' else 945


def _source(root,name,expected):
    require(same(pointer(root,name),readiness.pointer(expected)),'actual frozen source differs: '+name)


def graph_for_owner(root,graph_ref,split):
    split_count(split);graph_ref=artifact_ref(root,graph_ref,prefix='artifacts/framework/multitask_r10/',basename='policy_graph.json')
    graph=graphs.read_policy_graph(root,graph_ref)
    role='train_owner' if split=='train' else 'evaluation_owner'
    _source(root,SOURCE,graph['control']['components'][role])
    require(sha(Path(__file__))==pointer(root,SOURCE)['sha256'],'actual imported owner source differs')
    _source(root,WRAPPER,graph['control']['components']['environment_wrapper'])
    _source(root,GUARD,graph['control']['components']['resource_guard'])
    require(graph['physical']['environment_manifest']['sha256']==ENV_SHA,'actual frozen820 manifest bytes required')
    require(graph['train_source_manifest']['root']=='project' and graph['train_source_manifest']['path']==TRAIN_MANIFEST,
        'actual original TRAIN manifest pointer required')
    control=list(graph['control']['sources_map'].values())
    for name in ('scripts/run_aligned_gpu.sh','scripts/run_local_sim.sh'):
        require(pointer(root,name) in control,'both method-specific launchers must be control-frozen')
    for route in graph['routes']:
        declaration=route['train' if split=='train' else 'evaluation']
        for name,key in ((WORKER,'worker'),(PREPARER,'preparer'),(CONSUMER,'consumer')):
            _source(root,name,declaration[key])
        require(declaration['case_schema']==CASE_SCHEMAS[split],'actual1167 case schema required')
        require(route['configuration']['root']==route['frozen_inputs']['root']=='project','actual project configuration/input envelope required')
    return graph,readiness.pointer(graph_ref)


def _prior_module(root):
    """Use the actual frozen development verifier, with its owned root restored."""
    require(pointer(root,SMOKE_SOURCE)['sha256']==SMOKE_SOURCE_SHA,'exact frozen1166 source required')
    return importlib.import_module(SMOKE_SOURCE[:-3].replace('/','.'))


def _prior_phase(root,previous,job,refs,*,check,lock):
    phase='CPU_check' if check else 'native'
    require(type(refs) is dict and set(refs)=={'launch','terminal','sidecar'},'prior six compact phase references required')
    base=SMOKE+'/launches/'+job['job_id']+'/'+phase
    launch,_=read(root,exact_ref(root,refs['launch'],base+'/launch.json'))
    terminal,_=read(root,exact_ref(root,refs['terminal'],base+'/terminal.json'))
    sidecar,_=read(root,exact_ref(root,refs['sidecar'],job['runtime']+'/'+phase+'/worker_sidecar.json'))
    outer=readiness._process(dict(pid=launch.get('pid'),start_ticks=launch.get('start_ticks')))
    inner=readiness._process(sidecar.get('process'));ns=sidecar.get('frozen_OS_environment',{});e=sidecar.get('execution',{})
    require(same(readiness._process(dict(pid=terminal.get('pid'),start_ticks=terminal.get('start_ticks'))),outer)
        and type(terminal.get('returncode')) is int and terminal['returncode']==0 and terminal.get('automatic_retry') is False
        and launch.get('check_only') is check and same(launch.get('argv'),previous.command(job,check)),
        'prior actual child argv/PID/strict exit0 required')
    require(sidecar.get('schema')==previous.CANDIDATE_SCHEMA and sidecar.get('complete') is True and e.get('complete') is True
        and same(readiness.pointer(sidecar.get('source_overlay_lock')),lock)
        and sidecar.get('formal_score_contribution') is False
        and (sidecar.get('selection',{}).get('task'),sidecar.get('selection',{}).get('raw_episode_id'),sidecar.get('selection',{}).get('method'))
            ==(job['task'],job['raw'],job['method'])
        and same(readiness.pointer(sidecar.get('actual_worker')),pointer(root,previous.WORKER))
        and same(readiness.pointer(sidecar.get('actual_preparer')),pointer(root,previous.PREPARER))
        and ns.get('complete') is True
        and all(type(ns.get(k)) is int for k in ('outer_wrapper_pid','outer_wrapper_start_ticks','observer_pid','observer_start_ticks'))
        and ns.get('outer_wrapper_pid')==outer['pid'] and ns.get('outer_wrapper_start_ticks')==outer['start_ticks']
        and ns.get('observer_pid')==inner['pid'] and ns.get('observer_start_ticks')==inner['start_ticks'],
        'prior actual worker/selection/820 binding required')
    require(sidecar.get('prior_controlled_stop')==(None if check else SMOKE_STOP),'prior CPU/Native stop metadata differs')
    if check:
        require(e.get('mode')=='CPU_check' and e.get('native_started') is False and e.get('GPU_started') is False
            and e.get('actual_split')=='train' and e.get('native_execution_ready') is True,'prior complete CPU preflight required')
    else:
        require(e.get('mode')=='new_protocol_native_TRAIN' and e.get('native_started') is True
            and e.get('native_false_preserved') is True,'prior original Native outcomes required')
    require(readiness._not_live(outer) and readiness._not_live(inner),
        'prior actual outer wrapper and inner worker identities must both be dead')


def smoke_complete(root):
    """Require the actual five-case1166 exit; it never qualifies fresh192."""
    previous=_prior_module(root)
    plan,plan_ref=read(root,exact_ref(root,dict(path=SMOKE_PLAN,sha256=SMOKE_PLAN_SHA),SMOKE_PLAN))
    launch,launch_ref=read(root,pointer(root,SMOKE+'/launch.json'))
    status,status_ref=read(root,pointer(root,SMOKE+'/status.json'))
    terminal,terminal_ref=read(root,pointer(root,SMOKE+'/terminal.json'))
    completion,completion_ref=read(root,pointer(root,SMOKE+'/exec_completion.json'))
    process=readiness._process(dict(pid=launch.get('pid'),start_ticks=launch.get('start_ticks')))
    lock=readiness.pointer(SMOKE_LOCK);stop=readiness.pointer(SMOKE_STOP)
    require(plan_ref['sha256']==SMOKE_PLAN_SHA
        and plan.get('schema')=='r10_bounded_five_TRAIN_native_release_comparison_plan_v1166'
        and [(j.get('task'),j.get('raw'),j.get('method')) for j in plan.get('jobs',[])]==list(SMOKE_CASES)
        and same(readiness.pointer(plan.get('source_overlay_lock')),lock)
        and same(readiness.pointer(launch.get('plan')),plan_ref)
        and same(readiness.pointer(launch.get('source_overlay_lock')),lock)
        and launch.get('automatic_retry') is False and launch.get('max_native_slots')==1,
        'exact original fixed five1166 plan/launch required')
    require(same(process,SMOKE_PROCESS) and status.get('schema')=='r10_bounded_five_TRAIN_native_release_comparison_status_v1166'
        and type(status.get('expected')) is int and status['expected']==5 and status.get('stage')=='complete'
        and status.get('active') is None and status.get('failed')==[] and not status.get('error')
        and status.get('automatic_retry') is False
        and [(r.get('task'),r.get('raw'),r.get('method')) for r in status.get('completed',[])]==list(SMOKE_CASES),
        'all five original1166 attempts must naturally complete without active/infra cases')
    require(terminal.get('complete') is True and type(terminal.get('completed')) is int and terminal['completed']==5
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed']==0
        and terminal.get('automatic_retry') is False,'complete actual1166 terminal required')
    require(completion.get('schema')==SMOKE_EXIT_SCHEMA and completion.get('actual_tool_exit_observed') is True
        and completion.get('natural_completion') is True and type(completion.get('returncode')) is int and completion['returncode']==0
        and same(readiness._process(completion.get('process')),process)
        and type(completion.get('owner_pid')) is int and completion['owner_pid']==process['pid']
        and type(completion.get('owner_start_ticks')) is int and completion['owner_start_ticks']==process['start_ticks']
        and type(completion.get('completed')) is int and completion['completed']==5
        and type(completion.get('infrastructure_failed')) is int and completion['infrastructure_failed']==0
        and same(readiness.pointer(completion.get('plan')),plan_ref)
        and same(readiness.pointer(completion.get('terminal')),terminal_ref)
        and same(readiness.pointer(completion.get('source_overlay_lock')),lock)
        and same(readiness.pointer(completion.get('prior_controlled_stop')),stop) and readiness._not_live(process),
        'Root actual natural1166 tool exit0/dead process and provenance required')
    # validate() rechecks the real1165 source lock, original routes/models,
    # controlled-stop provenance and Root review. completed_case() closes all
    # original outcomes without imposing any NativeTrue or motionTrue gate.
    prior_root=previous.ROOT
    try:
        previous.ROOT=Path(root)
        previous.validate(plan)
        for job,row in zip(plan['jobs'],status['completed']):
            for phase,check in (('CPU_check',True),('native',False)):
                _prior_phase(root,previous,job,row[phase],check=check,lock=lock)
            observed=previous.completed_case(job,row['CPU_check'],row['native'],SMOKE_LOCK)
            require(same(observed,row),'prior original case/six references/compact record changed')
    finally:previous.ROOT=prior_root
    return dict(launch=launch_ref,status=status_ref,terminal=terminal_ref,exec_completion=completion_ref,
        source=pointer(root,SMOKE_SOURCE),plan=plan_ref,process=process,natural_completion=True,
        development_only=True,fresh192_qualification_granted=False)


def validation_rows(root,graph_ref,graph,bundle_ref):
    """The only parent VAL read: admitted, short leased, and before spawning."""
    bundle_ref=artifact_ref(root,bundle_ref,prefix=BUNDLE_PREFIX,basename='bundle.json')
    bundle,bundle_ref=read(root,bundle_ref)
    require(bundle.get('schema')==BUNDLE_SCHEMA and bundle.get('source_count')==315
        and bundle.get('Native_started') is False and bundle.get('TEST_read') is False,'actual1148 validation bundle required')
    readiness._stamp(bundle,graph_ref,graph)
    artifact_ref(root,bundle.get('admission'),prefix='artifacts/framework/multitask_r10/',basename='admission.json')
    directory=Path(bundle_ref['path']).parent.as_posix()
    for key,name in (('suite','suite.json'),('suite_lock','suite_lock.json'),('source_inputs','source_inputs.json')):
        exact_ref(root,bundle.get(key),directory+'/'+name)
    for name,key in ((GATE,'admission_guard'),(GATE,'formal_source_reader'),(WRITER,'split_source_lock_writer')):
        _source(root,name,graph['control']['components'][key])
    guard=importlib.import_module(GUARD[:-3].replace('/','.'))
    gate=importlib.import_module(GATE[:-3].replace('/','.'))
    with guard.reserved_resources_after_TEST(root,enabled=True) as permit:
        guard.require_permit(permit,root)
        with gate.admission_scope(root,graph_ref,bundle['admission'],permit):
            suite,_=read(root,bundle['suite']);lock,_=read(root,bundle['suite_lock']);inputs,_=read(root,bundle['source_inputs'])
            require(same(readiness.pointer(suite.get('source_inputs')),readiness.pointer(bundle['source_inputs']))
                and same(readiness.pointer(suite.get('admission')),readiness.pointer(bundle['admission']))
                and same(readiness.pointer(lock.get('admission')),readiness.pointer(bundle['admission']))
                and inputs.get('split')=='validation' and inputs.get('source_count')==315,
                'bundle/suite/source-inputs/admission closure differs')
            for value in (suite,inputs):readiness._stamp(value,graph_ref,graph)
            require(same(readiness.pointer(lock.get('graph')),graph_ref)
                and lock.get('policy_sha256')==graph['policy_sha256'],'suite lock belongs to a different graph')
            for reference in (graph_ref,bundle['admission'],bundle['suite'],bundle['source_inputs']):
                reference=readiness.pointer(reference)
                require(lock.get('files',{}).get(str(Path(root)/reference['path']))==reference['sha256'],
                    'suite lock omits actual bundle evidence')
            _,rows,_=gate.read_formal_source(root,suite,lock['files'],'validation')
            require(type(rows) is list and len(rows)==315,'exact admitted315 validation rows required')
        guard.require_permit(permit,root)
    # No capability or permit escapes this function.
    return rows,bundle_ref,bundle


def build_jobs(graph,sources,split,namespace):
    split_count(split)
    require(type(namespace) is str and re.fullmatch('[A-Za-z][A-Za-z0-9_.-]{0,95}',namespace),'fresh simple cohort namespace required')
    keys=[(r['task'],r['raw_episode_id']) for r in sources]
    require(len(keys)==(64 if split=='train' else 315) and len(set(keys))==len(keys)
        and {k[0] for k in keys}==set(graphs.TASKS),'exact unique original source set required')
    if split=='train':require(all(sum(t==k[0] for k in keys)==4 for t in graphs.TASKS),'four original TRAIN parents per task required')
    routes={(r['task'],r['method']):r for r in graph['routes']};jobs=[]
    for source in sources:
        for method in graphs.METHODS:
            task,raw=source['task'],source['raw_episode_id']
            require(type(raw) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',raw),'original simple source ID required')
            route=routes[(task,method)];suffix=f'cohorts/{namespace}/{task}/{raw}/{method}'
            jobs.append(dict(task=task,raw_episode_id=raw,method=method,source_row_sha256=readiness.digest(source),
                configuration=route['configuration'],out=f'{ART}/{split}/'+('development_demo_only/' if split=='train' else '')+suffix,
                runtime=f'{RUNTIME}/{split}/{suffix}'))
    require(len(jobs)==split_count(split),'exact source-by-three full grid required')
    return jobs


def technical_review(root,reference,graph_ref,graph,split,namespace,predecessor,bundle_ref):
    reference=artifact_ref(root,reference,prefix='artifacts/framework/multitask_r10/')
    review,reference=read(root,reference);readiness._stamp(review,graph_ref,graph)
    require(review.get('schema')==REVIEW_SCHEMA and review.get('reviewer')=='root' and review.get('review_complete') is True
        and review.get('release_fixed_cohort') is True and review.get('split')==split
        and review.get('namespace')==namespace
        and type(review.get('expected_cases')) is int and review['expected_cases']==split_count(split)
        and review.get('automatic_retry') is False and review.get('source_outcome_filtering') is False,
        'Root technical review for this graph and exact fixed cohort required')
    require(same(readiness.pointer(review.get('predecessor_exec_completion')),predecessor['exec_completion']),
        'technical review must bind actual five-case1166 completion')
    if split=='validation':
        require(same(readiness.pointer(review.get('validation_bundle')),bundle_ref), 'technical review must bind admitted validation bundle')
    return reference


def prepare_owner(root,graph_ref,*,split,namespace,review_ref,bundle_ref=None):
    root=Path(root).resolve();split_count(split)
    graph,graph_ref=graph_for_owner(root,graph_ref,split)
    owner=f'{BASE}/{split}/{namespace}'
    require(not path(root,owner).exists(),'fresh never-prepared owner namespace required')
    predecessor=smoke_complete(root)  # Mandatory before either source domain.
    if split=='train':
        require(bundle_ref is None,'TRAIN cannot carry formal source evidence')
        manifest,_=read(root,graph['train_source_manifest'])
        sources=list(readiness._sources(manifest).values());bundle=None
    else:
        sources,bundle_ref,bundle=validation_rows(root,graph_ref,graph,bundle_ref)
    jobs=build_jobs(graph,sources,split,namespace)
    review_ref=technical_review(root,review_ref,graph_ref,graph,split,namespace,predecessor,bundle_ref)
    for job in jobs:
        require(not path(root,job['out']).exists() and not path(root,job['runtime']).exists(),'all case output/runtime namespaces must be fresh')
    plan=dict(schema=readiness.PLAN_SCHEMA if split=='train' else VAL_PLAN_SCHEMA,
        split=split,namespace=namespace,owner_runtime=owner,expected_cases=split_count(split),sources=sources,jobs=jobs,
        automatic_retry=False,source_outcome_filtering=False,reused_cases=[],max_native_slots=1,
        CPU_check_before_every_native_case=True,remaining_queue=[],review=review_ref,predecessor=predecessor,
        validation_bundle=bundle_ref,formal_TEST_permitted=False,TEST_used_for_fitting=False,
        readiness_granted_by_owner=False,created_utc=utc(),**stamp(graph_ref,graph))
    # No graph, model, source or historical artifact is rewritten.
    graph_for_owner(root,graph_ref,split)
    name=owner+'/plan.json';fresh(path(root,name),plan)
    return pointer(root,name)


def _load_plan(root,plan_ref,*,admit_sources):
    plan_ref=artifact_ref(root,plan_ref,prefix=BASE+'/',basename='plan.json')
    plan,plan_ref=read(root,plan_ref);split=plan.get('split');count=split_count(split)
    graph,graph_ref=graph_for_owner(root,plan['graph'],split);readiness._stamp(plan,graph_ref,graph)
    require(plan.get('schema')==(readiness.PLAN_SCHEMA if split=='train' else VAL_PLAN_SCHEMA)
        and type(plan.get('expected_cases')) is int and plan['expected_cases']==count
        and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False
        and plan.get('reused_cases')==[] and type(plan.get('max_native_slots')) is int and plan['max_native_slots']==1
        and plan.get('CPU_check_before_every_native_case') is True and plan.get('remaining_queue')==[]
        and plan.get('formal_TEST_permitted') is False and plan.get('TEST_used_for_fitting') is False
        and plan.get('readiness_granted_by_owner') is False,'exact unfiltered serial no-retry graph-bound plan required')
    expected_owner=f'{BASE}/{split}/{plan["namespace"]}'
    require(plan.get('owner_runtime')==expected_owner and plan_ref['path']==expected_owner+'/plan.json', 'actual plan namespace differs')
    sources=plan['sources']
    predecessor=plan.get('predecessor');bundle_ref=plan.get('validation_bundle');bundle=None
    require(type(predecessor) is dict and predecessor.get('development_only') is True
        and predecessor.get('fresh192_qualification_granted') is False,'honest mandatory1166 prerequisite required')
    if admit_sources:require(same(smoke_complete(root),predecessor),'actual five-case1166 completion changed')
    if split=='train':
        require(bundle_ref is None,'TRAIN cannot reuse formal evidence')
        manifest,_=read(root,graph['train_source_manifest'])
        require(same(sources,list(readiness._sources(manifest).values())),'exact original first-four TRAIN rows required')
    elif admit_sources:
        current,bundle_ref,bundle=validation_rows(root,graph_ref,graph,bundle_ref)
        require(same(sources,current),'actual admitted validation315 source rows changed')
    require(same(plan['jobs'],build_jobs(graph,sources,split,plan['namespace'])),'planned grid/config/source/output paths differ')
    technical_review(root,plan['review'],graph_ref,graph,split,plan['namespace'],predecessor,bundle_ref)
    return plan,plan_ref,graph,bundle


def command(root,plan,graph,job,*,check):
    split=plan['split'];route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    worker=route['train' if split=='train' else 'evaluation']['worker']
    argv=[str(Path(root)/'.conda-env/bin/python'),'-B',WRAPPER,'--manifest-sha256',graph['physical']['environment_manifest']['sha256'],
        '--','bash','scripts/run_aligned_gpu.sh' if route['GPU_required'] else 'scripts/run_local_sim.sh',worker['path'],
        '--graph',plan['graph']['path'],'--config',route['configuration']['path'],'--split',split,
        '--task',job['task'],'--raw-episode-id',job['raw_episode_id'],'--method',job['method'],
        '--out',job['out'],'--runtime',job['runtime']]
    if split=='validation':
        bundle,_=read(root,plan['validation_bundle'])
        for flag,key in (('--suite','suite'),('--suite-lock','suite_lock'),('--source-inputs','source_inputs'),('--admission','admission')):
            argv.extend((flag,bundle[key]['path']))
    return argv+(['--check'] if check else [])


def stop_requested(root,plan):
    return any(path(root,name).exists() for name in (BASE+'/STOP',plan['owner_runtime']+'/STOP',
        RUNTIME+'/'+plan['split']+'/STOP','runtime/framework/multitask_r10_STOP'))


def child_phase(root,plan,graph,job,*,check):
    phase='CPU_check' if check else 'native';base=plan['owner_runtime']+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/'+phase
    directory=path(root,base);directory.mkdir(parents=True,exist_ok=False)
    argv=command(root,plan,graph,job,check=check);child=None;process=None;started=time.monotonic()
    route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    declaration=route['train' if plan['split']=='train' else 'evaluation']
    with (directory/'stdout.log').open('x') as out,(directory/'stderr.log').open('x') as err:
        try:
            child=subprocess.Popen(argv,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            process=process_identity(child.pid)
            fresh(directory/'launch.json',dict(process=process,argv=argv,actual_worker=declaration['worker'],
                actual_preparer=declaration['preparer'],check_only=check,split=plan['split'],
                task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],created_utc=utc(),
                automatic_retry=False,
                **stamp(plan['graph'],graph)))
            code=child.wait()
        except BaseException as error:
            observed=None if child is None else child.poll()
            fresh(directory/'owner_interruption.json',dict(process=process,observed_returncode=observed,
                child_exit_unknown=child is not None and observed is None,exception_type=type(error).__name__,
                normal_child_terminal_claimed=False,native_failure_inferred=False,automatic_retry=False))
            raise
    fresh(directory/'terminal.json',dict(process=process,returncode=code,automatic_retry=False,
        finished_utc=utc(),wall_seconds=time.monotonic()-started))
    require(type(code) is int and code==0,'actual '+phase+' child infrastructure failure; no retry')
    result=dict(launch=pointer(root,base+'/launch.json'),terminal=pointer(root,base+'/terminal.json'),
        sidecar=pointer(root,job['runtime']+'/'+phase+'/worker_sidecar.json'))
    validate_phase(root,plan,graph,job,result,check=check)
    return result


def validate_phase(root,plan,graph,job,refs,*,check):
    require(type(refs) is dict and set(refs)=={'launch','terminal','sidecar'},'actual phase launch/terminal/sidecar required')
    phase='CPU_check' if check else 'native'
    base=plan['owner_runtime']+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/'+phase
    launch,launch_ref=read(root,exact_ref(root,refs['launch'],base+'/launch.json'))
    terminal,terminal_ref=read(root,exact_ref(root,refs['terminal'],base+'/terminal.json'))
    process=readiness._process(launch.get('process'))
    require(launch_ref['path']==base+'/launch.json' and terminal_ref['path']==base+'/terminal.json'
        and same(readiness._process(terminal.get('process')),process)
        and type(terminal.get('returncode')) is int and terminal['returncode']==0
        and terminal.get('automatic_retry') is False
        and same(launch.get('argv'),command(root,plan,graph,job,check=check)),
        'actual original child argv/PID/exit0 binding required')
    readiness._stamp(launch,plan['graph'],graph)
    route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    declaration=route['train' if plan['split']=='train' else 'evaluation']
    require(same(readiness.pointer(launch.get('actual_worker')),declaration['worker'])
        and same(readiness.pointer(launch.get('actual_preparer')),declaration['preparer']),
        'actual launched worker/preparer differs')
    value,reference=read(root,exact_ref(root,refs['sidecar'],job['runtime']+'/'+phase+'/worker_sidecar.json'))
    require(reference['path']==job['runtime']+'/'+phase+'/worker_sidecar.json','actual sidecar escaped planned namespace')
    execution=value.get('execution',{});inner=readiness._process(value.get('process'))
    namespace=value.get('frozen_OS_environment',{})
    readiness._stamp(value,plan['graph'],graph)
    require(value.get('complete') is True and execution.get('complete') is True
        and same(readiness.pointer(value.get('actual_worker')),declaration['worker'])
        and same(readiness.pointer(value.get('actual_preparer')),declaration['preparer'])
        and namespace.get('complete') is True
        and all(type(namespace.get(k)) is int for k in ('outer_wrapper_pid','outer_wrapper_start_ticks','observer_pid','observer_start_ticks'))
        and namespace['outer_wrapper_pid']==process['pid'] and namespace['outer_wrapper_start_ticks']==process['start_ticks']
        and namespace['observer_pid']==inner['pid'] and namespace['observer_start_ticks']==inner['start_ticks'],
        'actual820 outer/inner process and worker/preparer binding required')
    require(execution.get('native_started') is (not check) and execution.get('actual_split')==plan['split'], 'actual child CPU/native purpose differs')
    if check:require(execution.get('mode')=='CPU_check' and execution.get('GPU_started') is False,'actual CPU-only preflight required')
    else:require(execution.get('mode')=='graph_bound_native_'+('TRAIN' if plan['split']=='train' else 'validation')
        and execution.get('native_false_preserved') is True,'actual graph-bound Native sidecar required')
    return value


def completed_case(root,plan,graph,job,cpu,native):
    """Compact closure; the post-run1144/1170 auditor remains authoritative."""
    CPU=validate_phase(root,plan,graph,job,cpu,check=True)
    Native=validate_phase(root,plan,graph,job,native,check=False)
    case,case_ref=read(root,pointer(root,job['out']+'/case.json'))
    split=plan['split'];route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    declaration=route['train' if split=='train' else 'evaluation']
    source=next(r for r in plan['sources'] if (r['task'],r['raw_episode_id'])==(job['task'],job['raw_episode_id']))
    witness=readiness.make_graph_witness(plan['graph'],graph,route,source,split=split)
    identity=case.get('identity',{});unhashed=dict(identity);unhashed.pop('sha256',None)
    inner=readiness._process(Native.get('process'));cpu_inner=readiness._process(CPU.get('process'))
    require(not same(inner,cpu_inner) and case.get('version')==declaration['case_schema'] and case.get('complete') is True
        and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None
        and type(case.get('process_id')) is int and case['process_id']==inner['pid']
        and (case.get('task'),case.get('source_raw_episode_id'),case.get('method'),case.get('split'))==
            (job['task'],job['raw_episode_id'],job['method'],split)
        and same(case.get('graph_witness'),witness) and same(CPU.get('graph_witness'),witness) and same(Native.get('graph_witness'),witness)
        and same(CPU['execution'].get('graph_witness'),witness) and same(Native['execution'].get('graph_witness'),witness)
        and identity.get('sha256')==readiness.digest(unhashed)
        and identity.get('config_sha256')==route['configuration']['sha256']
        and identity.get('frozen_inputs_sha256')==route['frozen_inputs']['sha256']
        and identity.get('source_row_sha256')==readiness.digest(source)
        and identity.get('graph_sha256')==plan['graph']['sha256'] and all(identity.get(k)==graph[k] for k in readiness.HASHES)
        and same(CPU['execution'].get('identity'),identity) and same(Native['execution'].get('identity'),identity),
        'clean actual CPU/native/case graph/config/model/source identity binding required')
    require(same(case.get('execution_protocol'),CPU['execution'].get('execution_protocol'))
        and same(case.get('execution_protocol'),Native['execution'].get('execution_protocol'))
        and identity.get('execution_protocol_schema')==case.get('execution_protocol',{}).get('schema')
        and identity.get('execution_protocol_sha256')==readiness.digest(case.get('execution_protocol'))
        and case.get('original_745_execution_acceptance_granted') is False
        and case.get('qualification_granted_by_worker') is False
        and case.get('formal_score_contribution') is (split=='validation'),'honest actual new protocol boundary required')
    require(same(readiness.pointer(Native['execution'].get('case')),case_ref)
        and Native['execution'].get('native_success') is case.get('official_task_success')
        and Native['execution'].get('native_failure') is case.get('native_terminal_failure')
        and Native['execution'].get('runtime_scope_restoration',{}).get('complete') is True
        and Native['execution'].get('runtime_scope_restoration',{}).get('restored') is True,'actual Native outcomes/restoration differ')
    profile,protocol=readiness._profile(graph,root)
    receipt=readiness.validate_sampling_case(case,protocol,block=job['task'] in readiness.BLOCKS,case_version=declaration['case_schema'])
    readiness._attestation(case,profile);motion_pass=readiness._motion(case)
    hooks,hook_ref=read(root,pointer(root,job['out']+'/owned_hook_receipt.json'))
    require(hooks.get('actual_split')==split and type(hooks.get('reset')) is list and type(hooks.get('paths')) is list
        and all(type(r) is dict and r.get('complete') is True for r in hooks['reset'])
        and all(type(r) is dict and r.get('close_complete') is True for r in hooks['paths']), 'actual owned reset/RML cleanup required')
    if job['task'] in readiness.BLOCKS:require(hooks['reset']==[],'original block reset interface required')
    else:require(len(hooks['reset'])==1,'one actual core reset receipt required')
    return dict(task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],attempt=1,
        native_success=case['official_task_success'],native_terminal_failure=case['native_terminal_failure'],
        counted_ticks=case['counted_clock']['ticks'],counted_seconds=case['counted_clock']['seconds'],motion_pass=motion_pass,
        completed_actions=len(receipt['completed_actions']),case=case_ref,owned_hooks=hook_ref,CPU_check=cpu,native=native)


def _refresh(root,plan_ref,plan,graph):
    current,_=read(root,plan_ref);require(same(current,plan),'immutable prepared plan changed')
    current_graph,_=graph_for_owner(root,plan['graph'],plan['split']);require(same(current_graph,graph),'frozen execution graph changed')
    read(root,plan['review'])
    if plan['split']=='validation':read(root,plan['validation_bundle'])
    predecessor=plan['predecessor']
    for key in ('launch','status','terminal','exec_completion','plan'):read(root,predecessor[key])
    _source(root,SMOKE_SOURCE,predecessor['source'])
    require(readiness._not_live(predecessor['process']),'prior1166 owner identity became live')


def execute_owner(root,plan_ref,plan_sha256):
    root=Path(root).resolve();plan_ref=artifact_ref(root,plan_ref,prefix=BASE+'/',basename='plan.json')
    require(type(plan_sha256) is str and plan_ref['sha256']==plan_sha256,'explicit exact prepared plan SHA required')
    global_lock=path(root,BASE+'/owner.lock');global_lock.parent.mkdir(parents=True,exist_ok=True)
    with global_lock.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        header,_=read(root,plan_ref)
        # Refuse duplicate owners before any admission/manifest planning work.
        split_count(header.get('split'));base=path(root,header['owner_runtime'])
        require(not (base/'launch.json').exists(),'already launched; no resume or retry')
        # Any short parent admission has exited before child_phase is reachable.
        plan,plan_ref,graph,_=_load_plan(root,plan_ref,admit_sources=True)
        for job in plan['jobs']:
            require(not path(root,job['out']).exists() and not path(root,job['runtime']).exists(),'all native/CPU case namespaces must still be fresh')
        process=process_identity()
        fresh(base/'launch.json',dict(schema=SCHEMA,process=process,source=pointer(root,SOURCE),plan=plan_ref,
            argv=list(sys.argv),split=plan['split'],automatic_retry=False,created_utc=utc(),**stamp(plan['graph'],graph)))
        status=dict(schema=SCHEMA,stage='running',expected_cases=split_count(plan['split']),attempted=0,
            completed=[],failed=[],active=None,split=plan['split'],plan=plan_ref,
            automatic_retry=False,source_outcome_filtering=False,readiness_granted_by_owner=False,updated_utc=utc())
        atomic(base/'status.json',status)
        try:
            for index,job in enumerate(plan['jobs']):
                require(not stop_requested(root,plan),'STOP before next fixed source')
                status.update(attempted=index+1,active=dict(task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],phase='CPU_check'),updated_utc=utc())
                atomic(base/'status.json',status);_refresh(root,plan_ref,plan,graph)
                cpu=child_phase(root,plan,graph,job,check=True)
                require(not stop_requested(root,plan),'STOP after CPU preflight')
                status['active']['phase']='native';atomic(base/'status.json',status);_refresh(root,plan_ref,plan,graph)
                native=child_phase(root,plan,graph,job,check=False)
                status['active']['phase']='result_closure';atomic(base/'status.json',status)
                result=completed_case(root,plan,graph,job,cpu,native)
                status['completed'].append(result);status.update(active=None,updated_utc=utc());atomic(base/'status.json',status)
                print(json.dumps(dict(completed=len(status['completed']),expected=split_count(plan['split']),
                    task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],native_success=result['native_success']),ensure_ascii=False),flush=True)
            _refresh(root,plan_ref,plan,graph)
            require(len(status['completed'])==split_count(plan['split']),'all fixed cases required')
            status.update(stage='complete',active=None,updated_utc=utc());atomic(base/'status.json',status)
            fresh(base/'terminal.json',dict(schema=SCHEMA,complete=True,process=process,attempted=status['attempted'],
                completed=len(status['completed']),infrastructure_failed=0,active=None,automatic_retry=False,
                split=plan['split'],plan=plan_ref,finished_utc=utc(),actual_tool_exit_observed_by_owner=False,
                readiness_granted_by_owner=False,**stamp(plan['graph'],graph)))
        except BaseException as error:
            status['failed'].append(dict(active=status.get('active'),exception_type=type(error).__name__,
                message=str(error)[:1500],native_failure_inferred=False,automatic_retry=False))
            status.update(stage='failed',error=dict(type=type(error).__name__,message=str(error)[:1500]),updated_utc=utc())
            fresh(base/'failure.json',status);raise
        finally:atomic(base/'status.json',status)
    return dict(complete=True,completed=len(status['completed']),terminal=pointer(root,plan['owner_runtime']+'/terminal.json'),
        actual_tool_exit_observed_by_owner=False,readiness_granted_by_owner=False)


def _assemble(root,plan_ref,actual_exec_completion_ref,*,split,schema):
    root=Path(root).resolve();plan,plan_ref,graph,_=_load_plan(root,plan_ref,admit_sources=False)
    require(plan['split']==split,'TRAIN192 and validation945 cohorts are distinct')
    base=plan['owner_runtime'];launch,launch_ref=read(root,pointer(root,base+'/launch.json'))
    status,_=read(root,pointer(root,base+'/status.json'));terminal,terminal_ref=read(root,pointer(root,base+'/terminal.json'))
    actual_ref=artifact_ref(root,actual_exec_completion_ref,prefix=base+'/')
    actual,actual_ref=read(root,actual_ref);process=readiness._process(launch.get('process'))
    count=split_count(split)
    readiness._stamp(launch,plan['graph'],graph);readiness._stamp(terminal,plan['graph'],graph)
    require(same(readiness.pointer(launch.get('source')),pointer(root,SOURCE))
        and same(readiness.pointer(launch.get('plan')),plan_ref) and status.get('stage')=='complete'
        and status.get('failed')==[] and status.get('active') is None and type(status.get('attempted')) is int
        and status['attempted']==count and len(status.get('completed',[]))==count,
        'all fixed attempts must actually finish before cohort assembly')
    require(terminal.get('complete') is True and type(terminal.get('completed')) is int and terminal['completed']==count
        and type(terminal.get('attempted')) is int and terminal['attempted']==count
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed']==0
        and terminal.get('automatic_retry') is False and terminal.get('active') is None
        and same(readiness._process(terminal.get('process')),process)
        and same(readiness.pointer(terminal.get('plan')),plan_ref),
        'actual complete owner terminal without infra failure required')
    require(same(readiness._process(actual.get('process')),process)
        and actual.get('actual_tool_exit_observed') is True and actual.get('natural_completion') is True
        and type(actual.get('returncode')) is int and actual['returncode']==0
        and same(readiness.pointer(actual.get('plan')),plan_ref)
        and same(readiness.pointer(actual.get('terminal')),terminal_ref) and readiness._not_live(process),
        'only Root can supply actual tool exit0/dead identity after natural completion')
    records=status['completed'];keys=[readiness._key(r) for r in records]
    require(keys==[readiness._key(j) for j in plan['jobs']] and len(set(keys))==count,'all fixed cases exactly once in original order required')
    for job,row in zip(plan['jobs'],records):
        require(type(row.get('attempt')) is int and row['attempt']==1,'one original attempt per case required')
        for phase in ('CPU_check','native'):
            require(set(row[phase])=={'launch','terminal','sidecar'},'actual per-phase compact pointers required')
        observed=completed_case(root,plan,graph,job,row['CPU_check'],row['native'])
        require(same(observed,row),'compact status differs from actual completed source case')
    _refresh(root,plan_ref,plan,graph)
    value=dict(schema=schema,complete=True,split=split,expected_cases=count,records=records,plan=plan_ref,
        owner=dict(launch=launch_ref,terminal=terminal_ref,exec_completion=actual_ref),
        automatic_retry=False,source_outcome_filtering=False,reused_cases=[],
        validation_bundle=plan['validation_bundle'],readiness_granted_by_owner=False,**stamp(plan['graph'],graph))
    name=base+'/cohort.json';fresh(path(root,name),value)
    return pointer(root,name)


def assemble_cohort(root,plan_ref,actual_exec_completion_ref):
    """TRAIN-only, post-exit collection;1144 must still audit every receipt."""
    return _assemble(root,plan_ref,actual_exec_completion_ref,split='train',schema=readiness.COHORT_SCHEMA)


def assemble_validation_cohort(root,plan_ref,actual_exec_completion_ref):
    """Distinct VAL945 collection; never issues TRAIN192 or readiness."""
    return _assemble(root,plan_ref,actual_exec_completion_ref,split='validation',schema=VAL_COHORT_SCHEMA)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('prepare','execute','assemble-cohort','assemble-validation-cohort'):mode.add_argument('--'+flag,action='store_true')
    for name in ('graph','split','namespace','review','validation-bundle','plan','plan-sha256','actual-exec-completion'):parser.add_argument('--'+name)
    args=parser.parse_args(argv)
    if args.prepare:
        require(all(getattr(args,k) for k in ('graph','split','namespace','review')),'prepare requires graph/split/namespace/review')
        result=prepare_owner(ROOT,args.graph,split=args.split,namespace=args.namespace,review_ref=args.review,bundle_ref=args.validation_bundle)
    elif args.execute:
        require(args.plan and args.plan_sha256,'execute requires actual plan and explicit SHA')
        result=execute_owner(ROOT,args.plan,args.plan_sha256)
    elif args.assemble_cohort or args.assemble_validation_cohort:
        require(args.plan and args.actual_exec_completion,'post-exit collection requires plan and Root actual exit receipt')
        function=assemble_validation_cohort if args.assemble_validation_cohort else assemble_cohort
        result=function(ROOT,args.plan,args.actual_exec_completion)
    else:result=dict(enabled=False,Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False)
    print(json.dumps(result,ensure_ascii=False),flush=True);return result


if __name__=='__main__':main()
