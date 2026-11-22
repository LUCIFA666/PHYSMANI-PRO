"""Three fixed matched TRAIN block preclose candidates; default is inert.

Two original first-four parents, CPU then Native serial without retry/filtering.
Each frozen1154 child owns1106; the parent holds only its owner lock.
This development comparison cannot grant formal TRAIN192/VAL/TEST readiness.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_TRAIN_block_preclose_comparison_v1155_r10.py'
TEST_SOURCE = 'tests/test_TRAIN_block_preclose_comparison_v1155.py'
WORKER = 'scripts/run_TRAIN_block_preclose_full16_overlay_v1154_r10.py'
PREPARER = 'our_mani/multitask/train_block_preclose_full16_overlay_v1154.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
SIM = 'scripts/run_local_sim.sh'
SAMPLER_VERIFIER = 'our_mani/multitask/author_outcome_receipt_v749.py'
NAME = 'TRAIN_block_preclose_full16_v1154'
SETUP = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
ART = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/matched_v1155'
CASE_RUNTIME = 'runtime/framework/' + NAME + '/cases/matched_v1155'
RUNTIME = 'runtime/framework/' + NAME + '/owner_v1155'
PLAN = SETUP + 'bounded_plan_v1155.json'
SOURCE_LOCK = SETUP + 'source_overlay_lock.json'
SOURCE_LOCK_SHA = '0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9'

ROOT_REVIEW = SETUP + 'root_review_v1155.json'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA = 'r10_bounded_three_TRAIN_block_preclose_comparison_plan_v1155'
STATUS_SCHEMA = 'r10_bounded_three_TRAIN_block_preclose_comparison_status_v1155'
REVIEW_SCHEMA = 'r10_fixed_three_block_preclose_comparison_review_v1155'
EXIT_SCHEMA = 'r10_actual_natural_high_ring_diagnostic_owner_exit_v1152'
PREDECESSOR = 'runtime/framework/TRAIN_high_ring_diagnostic_v1151/owner_v1152'
PREDECESSOR_PLAN = 'artifacts/framework/multitask_r10/TRAIN_high_ring_diagnostic_v1151_setup/bounded_plan_v1152.json'
PREDECESSOR_PLAN_SHA = 'aca6a814b7e1614b8074b048696b53a777f694b7cce569aa184a1652d359a443'
PREDECESSOR_SOURCE = 'scripts/run_TRAIN_high_ring_diagnostic_owner_v1152_r10.py'
PREDECESSOR_SOURCE_SHA = '8e1809670602819e4be6147b73ab2aae484a50b4534b95fdb512789d69f0ef29'
PREDECESSOR_COMPLETION = PREDECESSOR + '/exec_completion.json'
PREDECESSOR_EXEC_SESSION = 47395
PREDECESSOR_CASES = tuple(('insert_onto_rotating_peg_high_speed','episode102',method)
    for method in ('cv_geometry','context_geometry'))
BLOCKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
METHODS = ('cv_geometry', 'context_geometry')
CANARIES = (
    ('pick_moving_target_on_the_table', 'episode0', 'cv_geometry'),
    ('pick_moving_target_on_the_table', 'episode0', 'context_geometry'),
    ('pick_moving_target_on_the_table_high_speed', 'episode100', 'context_geometry'),
)
FIXED_SOURCES = [dict(task=t, raw_episode_id=r, method=m) for t,r,m in CANARIES]
CANDIDATE_SCHEMA = 'r10_full16_fixed_TRAIN_block_preclose_overlay_v1154'
BLOCK_MODULE = 'our_mani.multitask.retained_preclose_block_integration_v1153'
BLOCK_SOURCE = BLOCK_MODULE.replace('.','/') + '.py'
CHANGED_FACTOR = 'retained_original_block_first_preclose_witness'
COMPARISON_PARENT_LOCK = dict(path='artifacts/framework/multitask_r10/TRAIN_author_release_full16_v1141_setup/source_overlay_lock.json',
    sha256='cbb0f900d27270b2c41974827224ce105dff0f48d7714e5f65837b98dc3a26c8')
FROZEN_PINS = {'our_mani/multitask/train_block_preclose_full16_overlay_v1154.py': '8a4b8a2019a979b4483efce9b39f5ebd7671450ed94771dd897fab04d29ca9c0', 'scripts/run_TRAIN_block_preclose_full16_overlay_v1154_r10.py': '7e64cb2b7fa66017aeeeca3a7f6d4a684910ee6c1d4093844f46aedd7d2b4d2f', 'tests/test_train_block_preclose_full16_overlay_v1154.py': '1981d432b3ac106adbd38f4682aa742ef2e3f9374324c11fd307c50d114a76a0', 'our_mani/multitask/retained_preclose_block_integration_v1153.py': 'ef646686984cf2de785e00ae39ec460bb68ac2fae9b3b4c1b9ca8e5a303a1b92', 'tests/test_retained_preclose_block_integration_v1153.py': 'dec9c1acc091d687bf75028bb089a082775192aba319638b8de64627f0b28cb4'}


def require(value, message):
    if not value: raise ValueError('TRAIN1155: ' + message)


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pointer(name): return dict(path=name, sha256=sha(ROOT/name))


def fresh(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def atomic(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    fresh(temporary,value);os.replace(temporary,path)


def checked_reference(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str
        and not Path(ref['path']).is_absolute() and '..' not in Path(ref['path']).parts
        and str(Path(ref['path']))==ref['path'], 'canonical project evidence pointer required')
    require(sha(ROOT/ref['path'])==ref['sha256'],'evidence changed: '+ref['path'])
    return ref


def overlay_module():
    require(type(SOURCE_LOCK_SHA) is str and len(SOURCE_LOCK_SHA)==64 and all(type(v) is str and len(v)==64 for v in FROZEN_PINS.values()),'pending1153/1154 pins cannot prepare or execute')
    for name,expected in FROZEN_PINS.items(): require(sha(ROOT/name)==expected,'frozen source changed: '+name)
    require(sha(ROOT/SOURCE_LOCK)==SOURCE_LOCK_SHA,'exact frozen1154 overlay lock required')
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from our_mani.multitask import train_block_preclose_full16_overlay_v1154
    return train_block_preclose_full16_overlay_v1154


def kind(task):
    require(task in BLOCKS,'only the exact two block task families are allowed')
    return 'block'


def build_jobs(roster, routes):
    require(roster.get('split')=='train','actual TRAIN manifest required')
    route_rows=routes.get('routes',[])
    require(len(route_rows)==48 and len({(r['task'],r['method']) for r in route_rows})==48,
        'complete unique original48 routes required')
    rows=[]
    for task,raw,method in CANARIES:
        parents=[r['raw_episode_id'] for r in roster['episodes'] if r['task']==task][:4]
        require(len(parents)==4 and len(set(parents))==4 and raw==parents[0],
            'fixed original first parent within unique first-four TRAIN sources required')
        selected=[r for r in route_rows if (r['task'],r['method'])==(task,method)]
        require(len(selected)==1 and method in METHODS and selected[0].get('GPU_required') is False,
            'exact original CPU geometry method route required')
        config=selected[0]['configuration'];require(type(config) is dict and set(config)=={'path','sha256'},'original config pointer required')
        job='_'.join((kind(task),task,raw,method))
        rows.append(dict(task=task,raw=raw,method=method,kind=kind(task),config=config['path'],
            config_sha256=config['sha256'],job_id=job,out=f'{ART}/{task}/{raw}/{method}',runtime=f'{CASE_RUNTIME}/{job}'))
    require(len(rows)==len({r['job_id'] for r in rows})==3,'exactly three unique source-method cases required')
    return rows


def process_not_live(pid,ticks):
    require(type(pid) is int and pid>0 and type(ticks) is int and ticks>0,'actual predecessor PID/start required')
    try:fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return True
    return int(fields[19])!=ticks or fields[0] in ('Z','X')


def predecessor_complete():
    require(sha(ROOT/PREDECESSOR_PLAN)==PREDECESSOR_PLAN_SHA and sha(ROOT/PREDECESSOR_SOURCE)==PREDECESSOR_SOURCE_SHA,
        'exact1152 source/plan required')
    names={k:PREDECESSOR+'/'+k+'.json' for k in ('launch','status','terminal')}
    launch,status,terminal=(json.loads((ROOT/names[k]).read_text()) for k in ('launch','status','terminal'))
    expected_plan=dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA)
    require(launch.get('plan')==expected_plan and status.get('stage')=='complete' and status.get('active') is None
        and status.get('failed')==[] and not status.get('error')
        and [(x['task'],x['raw'],x['method']) for x in status.get('completed',[])]==list(PREDECESSOR_CASES),
        '1152 must naturally complete exactly both fixed diagnostic cases without infrastructure failure')
    require(terminal.get('complete') is True and type(terminal.get('completed')) is int and terminal['completed']==2
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed']==0
        and terminal.get('automatic_retry') is False,'actual1152 terminal required')
    pid,ticks=launch.get('pid'),launch.get('start_ticks');require(process_not_live(pid,ticks),'1152 owner still live')
    completion=json.loads((ROOT/PREDECESSOR_COMPLETION).read_text())
    require(completion.get('schema')==EXIT_SCHEMA and completion.get('actual_tool_exit_observed') is True
        and type(completion.get('returncode')) is int and completion['returncode']==0
        and type(completion.get('unified_exec_session')) is int and completion['unified_exec_session']==PREDECESSOR_EXEC_SESSION
        and completion.get('owner_pid')==pid and completion.get('owner_start_ticks')==ticks
        and completion.get('owner_alive') is False and completion.get('natural_completion') is True
        and type(completion.get('native_cases_complete')) is int and completion['native_cases_complete']==2
        and type(completion.get('completed_case_infrastructure_failures')) is int and completion['completed_case_infrastructure_failures']==0
        and completion.get('plan')==expected_plan and completion.get('terminal')==pointer(names['terminal']),
        'actual root session47395 tool-exit0/dead identity/natural1152 completion receipt required')
    predecessor_plan=json.loads((ROOT/PREDECESSOR_PLAN).read_text())
    jobs=predecessor_plan.get('jobs',[])
    require([(j['task'],j['raw'],j['method']) for j in jobs]==list(PREDECESSOR_CASES),
        'exact two source/method diagnostic jobs required')
    for row in status['completed']:
        checked_reference(row['case']);checked_reference(row['owned_hooks'])
        for stage in ('CPU_check','native'):
            for field in ('launch','terminal','sidecar'):checked_reference(row[stage][field])
        predecessor_record_quality(row,jobs,predecessor_plan['source_overlay_lock'])
    return dict(launch=pointer(names['launch']),status=pointer(names['status']),terminal=pointer(names['terminal']),
        actual_exec_completion=pointer(PREDECESSOR_COMPLETION),original_pid=pid,original_start_ticks=ticks,
        process_identity_not_live=True,natural_completion=True,prior_native_outcomes_relabelled=False)


def predecessor_record_quality(row,jobs,lock):
    """Recheck frozen1152's completed-case JSON only, without starting a job."""
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from scripts import run_TRAIN_high_ring_diagnostic_owner_v1152_r10 as previous
    require(Path(previous.__file__).resolve()==ROOT/PREDECESSOR_SOURCE
        and previous.ROOT==ROOT,'actual pinned1152 read-only verifier required')
    job=next(j for j in jobs if (j['task'],j['raw'],j['method'])==(row['task'],row['raw'],row['method']))
    actual=previous.completed_case(job,row['CPU_check'],row['native'],lock)
    require(actual==row,'previous diagnostic case/quality receipt changed')


def root_review(lock, predecessor):
    value=json.loads((ROOT/ROOT_REVIEW).read_text())
    require(value.get('schema')==REVIEW_SCHEMA and value.get('review_complete') is True
        and value.get('release_bounded_three') is True and value.get('reviewer')=='root'
        and value.get('fixed_TRAIN_sources')==FIXED_SOURCES and value.get('source_overlay_lock')==lock
        and value.get('exec_completion')==predecessor['actual_exec_completion']
        and value.get('development_regression_only') is True and value.get('formal192_readiness') is False
        and value.get('changed_factor')==CHANGED_FACTOR,
        'root technical review for only three matched development cases required')
    return pointer(ROOT_REVIEW)


def source_names(overlay, locked, jobs):
    names={SOURCE,TEST_SOURCE,WORKER,PREPARER,WRAPPER,SIM,SAMPLER_VERIFIER,SOURCE_LOCK,ROOT_REVIEW,
        PREDECESSOR_PLAN,PREDECESSOR_SOURCE,PREDECESSOR_COMPLETION,
        *(PREDECESSOR+'/'+x+'.json' for x in ('launch','status','terminal')),
        overlay.MANIFEST,overlay.ROUTES,overlay.TRAIN_INPUTS,*overlay.OVERLAY_SOURCES,*overlay.FROZEN_PINS,*FROZEN_PINS,
        locked['integration_cpu_evidence']['path'],*(row['config'] for row in jobs)}
    from scripts import run_TRAIN_block_preclose_full16_overlay_v1154_r10 as worker
    names.update(entry['path'] for entry in worker.ENVIRONMENT.values())
    return names


def _jobs(overlay):
    roster=json.loads(overlay.raw(ROOT,overlay.MANIFEST,overlay.MANIFEST_SHA))
    routes=json.loads(overlay.raw(ROOT,overlay.ROUTES,overlay.ROUTES_SHA))
    rows=build_jobs(roster,routes)
    for row in rows:
        overlay.selection(ROOT,argparse.Namespace(task=row['task'],method=row['method'],raw_episode_id=row['raw'],config=row['config']))
        require(sha(ROOT/row['config'])==row['config_sha256'],'original route config changed')
    return rows


def validate(plan):
    require(plan.get('schema')==PLAN_SCHEMA and plan.get('expected_cases')==3 and plan.get('source_parent_count')==2
        and plan.get('fixed_TRAIN_sources')==FIXED_SOURCES and plan.get('split')=='train'
        and plan.get('methods')==list(METHODS) and plan.get('max_native_slots')==1
        and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False
        and plan.get('CPU_check_before_every_native_case') is True and plan.get('remaining_queue')==[]
        and plan.get('development_regression_only') is True and plan.get('formal192_readiness') is False
        and plan.get('formal_VAL_TEST') is False and plan.get('changed_factor')==CHANGED_FACTOR and plan.get('predecessor_plan')==dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA),
        'fixed three-case block preclose comparison protocol required')
    for name,checksum in plan['inputs_sha256'].items():require(sha(ROOT/name)==checksum,'pinned input changed: '+name)
    overlay=overlay_module();locked,lock=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(lock==dict(path=SOURCE_LOCK,sha256=SOURCE_LOCK_SHA)==plan['source_overlay_lock']
        and locked['integration_complete'] is True and not locked['unpinned_sources'],'complete exact1154 source/CPU lock required')
    require(plan['jobs']==_jobs(overlay),'exact three fixed sources/routes/output paths required')
    predecessor=predecessor_complete()
    require(plan.get('predecessor')==predecessor and plan.get('root_review')==root_review(lock,predecessor),'predecessor/root review changed')
    require(source_names(overlay,locked,plan['jobs'])<=set(plan['inputs_sha256']),'complete frozen input graph required')
    return locked


def prepare():
    require(not (ROOT/PLAN).exists() and not (ROOT/RUNTIME).exists(),'fresh unprepared owner namespace required')
    predecessor=predecessor_complete();overlay=overlay_module();locked,lock=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(locked['integration_complete'] is True and not locked['unpinned_sources'],'complete CPU-qualified1154 graph required')
    review=root_review(lock,predecessor);jobs=_jobs(overlay)
    for row in jobs:require(not (ROOT/row['out']).exists() and not (ROOT/row['runtime']).exists(),'fresh case namespaces required')
    plan=dict(schema=PLAN_SCHEMA,created_utc=utc(),expected_cases=3,source_parent_count=2,jobs=jobs,
        fixed_TRAIN_sources=FIXED_SOURCES,methods=list(METHODS),split='train',max_native_slots=1,
        automatic_retry=False,source_outcome_filtering=False,CPU_check_before_every_native_case=True,
        remaining_queue=[],development_regression_only=True,formal192_readiness=False,formal_VAL_TEST=False,changed_factor=CHANGED_FACTOR,
        source_overlay_lock=lock,predecessor_plan=dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA),
        predecessor=predecessor,root_review=review,
        inputs_sha256={name:sha(ROOT/name) for name in sorted(source_names(overlay,locked,jobs))})
    validate(plan);fresh(ROOT/PLAN,plan)
    result=dict(plan=PLAN,sha256=sha(ROOT/PLAN),cases=3,Native_started=False);print(json.dumps(result),flush=True);return result


def command(row,check=False):
    args=[str(ROOT/'.conda-env/bin/python'),'-B',WRAPPER,'--manifest-sha256',ENV_SHA,'--','bash',SIM,WORKER,
        '--config',row['config'],'--task',row['task'],'--raw-episode-id',row['raw'],'--method',row['method'],
        '--out',row['out'],'--runtime',row['runtime'],'--overlay-lock',SOURCE_LOCK]
    return args+(['--check'] if check else [])


def _stop_requested():
    return any((ROOT/name).exists() for name in (RUNTIME+'/STOP','runtime/framework/'+NAME+'/STOP','runtime/framework/multitask_r10_STOP'))


def status_write(status):
    atomic(ROOT/RUNTIME/'status.json',status)


def validate_protocol(row, protocol, lock):
    require(row['task'] in BLOCKS and protocol.get('schema')==CANDIDATE_SCHEMA
        and protocol.get('source_overlay_lock')==lock
        and protocol.get('actual_worker')==pointer(WORKER) and protocol.get('actual_preparer')==pointer(PREPARER)
        and protocol.get('execution_branch')=='block1153'
        and protocol.get('comparison_parent_source_overlay_lock')==COMPARISON_PARENT_LOCK
        and protocol.get('core_changed_relative_to1141') is False and protocol.get('baseline_changed_relative_to1141') is False
        and protocol.get('formal_score_contribution') is False and protocol.get('formal_VAL_TEST_qualification') is False
        and protocol.get('changed_factor')==CHANGED_FACTOR
        and protocol.get('block_factory',{}).get('module')==BLOCK_MODULE
        and protocol['block_factory'].get('function')=='integrated_block_components'
        and protocol['block_factory'].get('source')==BLOCK_SOURCE
        and protocol['block_factory'].get('sha256')==FROZEN_PINS[BLOCK_SOURCE],
        'actual source-bound retained-first-preclose block1153 route required')
    component=protocol.get('public_phase_component_binding',{})
    require(type(component) is dict and set(component)=={'block_integration'}
        and type(FROZEN_PINS[BLOCK_SOURCE]) is str and len(FROZEN_PINS[BLOCK_SOURCE])==64
        and component['block_integration'].get('source_sha256')==FROZEN_PINS[BLOCK_SOURCE]
        and component['block_integration'].get('schema')=='r10_retained_original_block_preclose_integration_v1153'
        and component['block_integration'].get('changed_factor')==CHANGED_FACTOR,
        'actual1153 public block binding required')


def child_phase(row,check,status):
    phase='CPU_check' if check else 'native';leaf=ROOT/RUNTIME/'launches'/row['job_id']/phase
    leaf.mkdir(parents=True,exist_ok=False);argv=command(row,check);child=None;ticks=None;started=time.monotonic()
    status.update(active=dict(job_id=row['job_id'],stage=phase),updated_utc=utc());status_write(status)
    with (leaf/'stdout.log').open('x') as out,(leaf/'stderr.log').open('x') as err:
        try:
            child=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
            fresh(leaf/'launch.json',dict(created_utc=utc(),pid=child.pid,start_ticks=ticks,argv=argv,check_only=check,parent_pid=os.getpid()))
            code=child.wait()
        except BaseException as error:
            observed=None if child is None else child.poll()
            fresh(leaf/'owner_interruption.json',dict(observed_utc=utc(),child_pid=None if child is None else child.pid,
                child_start_ticks=ticks,observed_returncode=observed,child_exit_unknown=child is not None and observed is None,
                exception_type=type(error).__name__,normal_child_terminal_claimed=False,automatic_retry=False));raise
    fresh(leaf/'terminal.json',dict(finished_utc=utc(),returncode=code,pid=child.pid,start_ticks=ticks,wall_seconds=time.monotonic()-started,automatic_retry=False))
    require(type(code) is int and code==0,phase+' infrastructure failure; no retry')
    name=row['runtime']+'/'+phase+'/worker_sidecar.json';sidecar=json.loads((ROOT/name).read_text());execution=sidecar.get('execution',{});process=sidecar.get('process',{});namespace=sidecar.get('frozen_OS_environment',{})
    require(sidecar.get('schema')==CANDIDATE_SCHEMA and sidecar.get('complete') is True and execution.get('complete') is True
        and sidecar.get('source_overlay_lock')==status['source_overlay_lock'] and sidecar.get('formal_score_contribution') is False
        and (sidecar.get('selection',{}).get('task'),sidecar.get('selection',{}).get('raw_episode_id'),sidecar.get('selection',{}).get('method'))==(row['task'],row['raw'],row['method'])
        and namespace.get('complete') is True
        and namespace.get('outer_wrapper_pid')==child.pid and namespace.get('outer_wrapper_start_ticks')==ticks
        and namespace.get('observer_pid')==process.get('pid') and namespace.get('observer_start_ticks')==process.get('start_ticks')
        and type(process.get('pid')) is int and process['pid']>0 and type(process.get('start_ticks')) is int and process['start_ticks']>0,
        'actual worker identity/source/selection/namespace receipt required')
    if check:
        require(execution.get('mode')=='CPU_check' and execution.get('native_started') is False and execution.get('GPU_started') is False
            and execution.get('actual_split')=='train' and execution.get('native_execution_ready') is True,'actual CPU-only complete preflight required')
        validate_protocol(row,execution.get('execution_protocol',{}),status['source_overlay_lock'])
    else:
        require(execution.get('mode')=='new_protocol_native_TRAIN' and execution.get('native_started') is True
            and execution.get('native_false_preserved') is True,'actual Native sidecar required')
    # Never embed execution/source graphs in owner status; they are in sidecar.
    return dict(launch=pointer(str((leaf/'launch.json').relative_to(ROOT))),terminal=pointer(str((leaf/'terminal.json').relative_to(ROOT))),sidecar=pointer(name))


def validate_clock_and_sampler(case):
    require(case.get('task') in BLOCKS,'only original block sampling is supported by this matched owner')
    clock=case.get('counted_clock',{});ticks=clock.get('ticks');alignment=case.get('alignment_receipt',{});owned=alignment.get('clock',{})
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt')==.05 and type(clock.get('seconds')) in (int,float)
        and abs(clock['seconds']-ticks*.05)<=1e-8 and type(owned.get('ticks')) is int and owned['ticks']==ticks and owned.get('dt')==.05
        and owned.get('simulation_seconds')==clock['seconds'],'original200tick/.05s clock accounting required')
    require(case.get('version')==CANDIDATE_SCHEMA+'_case'
        and case.get('inherited_consumer_record_version')=='r10_full16_explicit_author_action_outcome_v744',
        'honest new1154 and inherited744 version boundary required')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case
    # Read-only projection for a verifier whose literal header is744. The new
    # header was checked above; neither the case object nor its file is edited.
    projected=dict(case,version=case['inherited_consumer_record_version'])
    receipt=require_sampling_case(projected,case['author_outcome_protocol'],block=True)
    return dict(counted_clock=clock,completed_actions=len(receipt['completed_actions']),
        sampler_schema=receipt['schema'],original_clock_and_action_sampling_preserved=True)


def completed_case(row,cpu,native,lock):
    checked_reference(cpu['sidecar']);checked_reference(native['sidecar'])
    CPU=json.loads((ROOT/cpu['sidecar']['path']).read_text())
    Native=json.loads((ROOT/native['sidecar']['path']).read_text())
    ce,ne=CPU['execution'],Native['execution'];name=row['out']+'/case.json';case=json.loads((ROOT/name).read_text());case_ref=pointer(name)
    require(case.get('version')==CANDIDATE_SCHEMA+'_case' and case.get('complete') is True and case.get('cleanup_completed') is True
        and case.get('split')=='train' and (case.get('task'),case.get('source_raw_episode_id'),case.get('method'))==(row['task'],row['raw'],row['method'])
        and case.get('process_id')==Native['process']['pid'] and case.get('source_overlay_lock')==lock
        and case.get('infrastructure_error') is None and case.get('formal_score_contribution') is False
        and case.get('original_745_execution_acceptance_granted') is False
        and ce['identity']==ne['identity']==case.get('identity') and case.get('execution_protocol')==ce.get('execution_protocol')
        and type(case.get('official_task_success')) is bool and type(case.get('native_terminal_failure')) is bool
        and ne.get('case')==case_ref and ne.get('native_success') is case['official_task_success']
        and ne.get('native_failure') is case['native_terminal_failure'],'clean complete actual case/result/CPU/Native identity binding required')
    require(case.get('author_outcome_protocol')==case['execution_protocol'].get('original_route',{}).get('author_outcome_protocol'),
        'case sampler must bind the original frozen route protocol')
    validate_protocol(row,case['execution_protocol'],lock);clock=validate_clock_and_sampler(case)
    hook=row['out']+'/owned_hook_receipt.json';hooks=json.loads((ROOT/hook).read_text());block=row['task'] in BLOCKS
    require(hooks.get('actual_split')=='train' and all(x.get('close_complete') is True for x in hooks.get('paths',[])), 'actual original cleanup hooks required')
    require(block and hooks.get('reset')==[] and hooks.get('shared_development_hooks_applied') is False,
        'original block reset interface required')
    return dict(job_id=row['job_id'],task=row['task'],raw=row['raw'],method=row['method'],
        native_success=case['official_task_success'],native_terminal_failure=case['native_terminal_failure'],
        clock_quality=clock,case=case_ref,owned_hooks=pointer(hook),CPU_check=cpu,native=native)


def execute(checksum):
    require(bool(checksum) and sha(ROOT/PLAN)==checksum,'explicit actual plan SHA required')
    plan=json.loads((ROOT/PLAN).read_text());validate(plan);predecessor=predecessor_complete();base=ROOT/RUNTIME;base.mkdir(parents=True,exist_ok=True)
    with (base/'owner.lock').open('a+') as owner_lock:
        fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);require(not (base/'launch.json').exists(),'already launched; no automatic resume/retry')
        for row in plan['jobs']:require(not (ROOT/row['out']).exists() and not (ROOT/row['runtime']).exists(),'all new case namespaces must be fresh')
        fresh(base/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
            argv=list(sys.argv),plan=dict(path=PLAN,sha256=checksum),source_overlay_lock=plan['source_overlay_lock'],predecessor=predecessor,automatic_retry=False,max_native_slots=1))
        status=dict(schema=STATUS_SCHEMA,expected=3,completed=[],failed=[],active=None,stage='running',updated_utc=utc(),
            source_overlay_lock=plan['source_overlay_lock'],automatic_retry=False,development_regression_only=True,formal192_readiness=False)
        status_write(status)
        try:
            for row in plan['jobs']:
                if _stop_requested():raise InterruptedError('actual STOP before case')
                validate(plan);predecessor_complete();cpu=child_phase(row,True,status)
                if _stop_requested():raise InterruptedError('actual STOP after CPU check')
                validate(plan);predecessor_complete();native=child_phase(row,False,status)
                status['active']['stage']='result_review';result=completed_case(row,cpu,native,plan['source_overlay_lock'])
                status['completed'].append(result);status.update(active=None,updated_utc=utc());status_write(status)
                print(json.dumps(dict(completed=len(status['completed']),expected=3,last=row['job_id'],native_success=result['native_success']),ensure_ascii=False),flush=True)
            validate(plan);status.update(stage='complete',active=None,updated_utc=utc())
            fresh(base/'terminal.json',dict(complete=True,completed=3,infrastructure_failed=0,automatic_retry=False,finished_utc=utc(),development_regression_only=True,formal192_readiness=False))
        except BaseException as error:
            if status.get('active'):status['failed'].append(dict(**status['active'],exception_type=type(error).__name__,native_failure_inferred=False,automatic_retry=False))
            status.update(stage='failed',updated_utc=utc(),error=dict(type=type(error).__name__,message=str(error)));fresh(base/'failure.json',status);raise
        finally:status_write(status)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');p.add_argument('--plan-sha256');args=p.parse_args(argv)
    if args.prepare:return prepare()
    if args.execute:return execute(args.plan_sha256)
    print(json.dumps(dict(enabled=False,Native_started=False,GPU_started=False)))


if __name__=='__main__':main()
