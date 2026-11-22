"""Eight fixed TRAIN development regressions, inert unless explicitly invoked.

Five original first-four parents across method/speed/core/block branches. The
frozen1141 worker owns each child's1106 lease; this parent owns only its lock.
CPU then Native, serial, no retry, no success filtering, no formal192 admission.
Status contains compact pointers; full worker evidence stays in each sidecar.
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
SOURCE = 'scripts/run_TRAIN_author_release_regression_canaries_v1145_r10.py'
TEST_SOURCE = 'tests/test_TRAIN_author_release_regression_canaries_v1145.py'
WORKER = 'scripts/run_TRAIN_author_release_full16_overlay_v1141_r10.py'
PREPARER = 'our_mani/multitask/train_author_release_full16_overlay_v1141.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
SIM = 'scripts/run_local_sim.sh'
SAMPLER_VERIFIER = 'our_mani/multitask/author_outcome_receipt_v749.py'
NAME = 'TRAIN_author_release_full16_v1141'
SETUP = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
ART = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/regression_v1145'
CASE_RUNTIME = 'runtime/framework/' + NAME + '/cases/regression_v1145'
RUNTIME = 'runtime/framework/' + NAME + '/owner_v1145'
PLAN = SETUP + 'bounded_plan_v1145.json'
SOURCE_LOCK = SETUP + 'source_overlay_lock.json'
SOURCE_LOCK_SHA = 'cbb0f900d27270b2c41974827224ce105dff0f48d7714e5f65837b98dc3a26c8'
ROOT_REVIEW = SETUP + 'root_review_v1145.json'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA = 'r10_bounded_eight_TRAIN_author_release_regression_plan_v1145'
STATUS_SCHEMA = 'r10_bounded_eight_TRAIN_author_release_regression_status_v1145'
REVIEW_SCHEMA = 'r10_fixed_eight_author_release_regression_review_v1145'
EXIT_SCHEMA = 'r10_actual_natural_author_release_canary_owner_exit_v1142'
PREDECESSOR = 'runtime/framework/TRAIN_author_release_full16_v1141/owner_v1142'
PREDECESSOR_PLAN = SETUP + 'bounded_plan_v1142.json'
PREDECESSOR_PLAN_SHA = '1bb6b814a7e403f98cd0f5d874363b6a70462c3c97a37815c21d4faca40f387f'
PREDECESSOR_SOURCE = 'scripts/run_TRAIN_author_release_canary_v1142_r10.py'
PREDECESSOR_SOURCE_SHA = 'cabc8d08518741f6c7c6210388640022546bd36fa9cb22da289fecbaf73313b3'
PREDECESSOR_COMPLETION = PREDECESSOR + '/exec_completion.json'
PREDECESSOR_CASES = (('insert_onto_rotating_peg', 'episode101', 'cv_geometry'),)
BLOCKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
METHODS = ('cv_geometry', 'context_geometry')
CANARIES = (
    ('insert_onto_rotating_peg', 'episode101', 'context_geometry'),
    ('insert_onto_rotating_peg_high_speed', 'episode102', 'cv_geometry'),
    ('insert_onto_rotating_peg_high_speed', 'episode102', 'context_geometry'),
    ('beat_the_rotating_buzz_high_speed', 'episode102', 'cv_geometry'),
    ('beat_the_rotating_buzz_high_speed', 'episode102', 'context_geometry'),
    ('pick_moving_target_on_the_table', 'episode0', 'cv_geometry'),
    ('pick_moving_target_on_the_table', 'episode0', 'context_geometry'),
    ('pick_moving_target_on_the_table_high_speed', 'episode100', 'context_geometry'),
)
FIXED_SOURCES = [dict(task=t, raw_episode_id=r, method=m) for t,r,m in CANARIES]
CANDIDATE_SCHEMA = 'r10_full16_fixed_TRAIN_author_release_overlay_v1141'
FROZEN_PINS = {
    PREPARER: 'ec76994e889f9649a4702887cd01aac66e315bf8c2084beb0e3b4c6866c7bfd4',
    WORKER: '7ce09b69c1e5051062cc570065d7b9b24e19cff7ec2e30728948ef6d3def1baf',
    'tests/test_train_author_release_full16_overlay_v1141.py': '7efd2e5abb6ff7f2e794341d661223b687295ce4fe1a456ef639febbf6d39964',
    'our_mani/multitask/author_ring_release_integration_v1140.py': '1ece8e678c59363ff4d5085ddf2659a7ba5dce3810dcc2ee771974ce09a30486',
    'tests/test_author_ring_release_integration_v1140.py': '3f4bc1df051a534d802d668667faa91df11912fe6b8542d9ebae0798b05243af',
    'our_mani/multitask/committed_public_role_integration_v1137.py': '4cde7bb1bbb2869968f54160ee5fa9ad1c8ec0770e81bd9e064121161d845e6a',
    'our_mani/multitask/author_aligned_block_integration_v1120.py': 'aac6bddf10e9360e7606718d357b1acd9c5241bf92ab684d3a54a591759cd147',
    'our_mani/multitask/public_buzz_approach_curve_v1130.py': '2f71d16fd045e03fedf9633cd52960290b60e2981ae1cd05d0f620247ffcdbc4',
}


def require(value, message):
    if not value: raise ValueError('TRAIN1145: ' + message)


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
    for name,expected in FROZEN_PINS.items(): require(sha(ROOT/name)==expected,'frozen source changed: '+name)
    require(sha(ROOT/SOURCE_LOCK)==SOURCE_LOCK_SHA,'exact frozen1141 overlay lock required')
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from our_mani.multitask import train_author_release_full16_overlay_v1141
    return train_author_release_full16_overlay_v1141


def kind(task):
    if task in BLOCKS:return 'block'
    if task.startswith('insert_onto_rotating_peg'):return 'ring'
    if task=='beat_the_rotating_buzz_high_speed':return 'buzz'
    raise ValueError('TRAIN1145: task outside exact development canaries')


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
    require(len(rows)==len({r['job_id'] for r in rows})==8,'exactly eight unique source-method cases required')
    return rows


def process_not_live(pid,ticks):
    require(type(pid) is int and pid>0 and type(ticks) is int and ticks>0,'actual predecessor PID/start required')
    try:fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return True
    return int(fields[19])!=ticks or fields[0] in ('Z','X')


def predecessor_complete():
    require(sha(ROOT/PREDECESSOR_PLAN)==PREDECESSOR_PLAN_SHA and sha(ROOT/PREDECESSOR_SOURCE)==PREDECESSOR_SOURCE_SHA,
        'exact1142 source/plan required')
    names={k:PREDECESSOR+'/'+k+'.json' for k in ('launch','status','terminal')}
    launch,status,terminal=(json.loads((ROOT/names[k]).read_text()) for k in ('launch','status','terminal'))
    expected_plan=dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA)
    require(launch.get('plan')==expected_plan and status.get('stage')=='complete' and status.get('active') is None
        and status.get('failed')==[] and not status.get('error')
        and [(x['task'],x['raw'],x['method']) for x in status.get('completed',[])]==list(PREDECESSOR_CASES),
        '1142 must naturally complete exactly one case with no infrastructure failure')
    require(terminal.get('complete') is True and terminal.get('completed')==1
        and terminal.get('infrastructure_failed')==0 and terminal.get('automatic_retry') is False,'actual1142 terminal required')
    pid,ticks=launch.get('pid'),launch.get('start_ticks');require(process_not_live(pid,ticks),'1142 owner still live')
    completion=json.loads((ROOT/PREDECESSOR_COMPLETION).read_text())
    require(completion.get('schema')==EXIT_SCHEMA and completion.get('actual_tool_exit_observed') is True
        and type(completion.get('returncode')) is int and completion['returncode']==0
        and completion.get('owner_pid')==pid and completion.get('owner_start_ticks')==ticks
        and completion.get('owner_alive') is False and completion.get('natural_completion') is True
        and completion.get('native_cases_complete')==1 and completion.get('completed_case_infrastructure_failures')==0
        and completion.get('plan')==expected_plan and completion.get('terminal')==pointer(names['terminal']),
        'actual root tool-exit0/dead identity/natural1142 completion receipt required')
    for row in status['completed']:
        checked_reference(row['case']);checked_reference(row['owned_hooks'])
        for stage in ('CPU_check','native'):
            for field in ('launch','terminal','sidecar'):checked_reference(row[stage][field])
    return dict(launch=pointer(names['launch']),status=pointer(names['status']),terminal=pointer(names['terminal']),
        actual_exec_completion=pointer(PREDECESSOR_COMPLETION),original_pid=pid,original_start_ticks=ticks,
        process_identity_not_live=True,natural_completion=True,prior_native_outcomes_relabelled=False)


def root_review(lock, predecessor):
    value=json.loads((ROOT/ROOT_REVIEW).read_text())
    require(value.get('schema')==REVIEW_SCHEMA and value.get('review_complete') is True
        and value.get('release_bounded_eight') is True and value.get('reviewer')=='root'
        and value.get('fixed_TRAIN_sources')==FIXED_SOURCES and value.get('source_overlay_lock')==lock
        and value.get('exec_completion')==predecessor['actual_exec_completion']
        and value.get('development_regression_only') is True and value.get('formal192_readiness') is False,
        'root technical review for only eight development cases required')
    return pointer(ROOT_REVIEW)


def source_names(overlay, locked, jobs):
    names={SOURCE,TEST_SOURCE,WORKER,PREPARER,WRAPPER,SIM,SAMPLER_VERIFIER,SOURCE_LOCK,ROOT_REVIEW,
        PREDECESSOR_PLAN,PREDECESSOR_SOURCE,PREDECESSOR_COMPLETION,
        *(PREDECESSOR+'/'+x+'.json' for x in ('launch','status','terminal')),
        overlay.MANIFEST,overlay.ROUTES,overlay.TRAIN_INPUTS,*overlay.OVERLAY_SOURCES,*overlay.FROZEN_PINS,*FROZEN_PINS,
        locked['integration_cpu_evidence']['path'],*(row['config'] for row in jobs)}
    from scripts import run_TRAIN_author_release_full16_overlay_v1141_r10 as worker
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
    require(plan.get('schema')==PLAN_SCHEMA and plan.get('expected_cases')==8 and plan.get('source_parent_count')==5
        and plan.get('fixed_TRAIN_sources')==FIXED_SOURCES and plan.get('split')=='train'
        and plan.get('methods')==list(METHODS) and plan.get('max_native_slots')==1
        and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False
        and plan.get('CPU_check_before_every_native_case') is True and plan.get('remaining_queue')==[]
        and plan.get('development_regression_only') is True and plan.get('formal192_readiness') is False
        and plan.get('formal_VAL_TEST') is False and plan.get('predecessor_plan')==dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA),
        'fixed eight-case development regression protocol required')
    for name,checksum in plan['inputs_sha256'].items():require(sha(ROOT/name)==checksum,'pinned input changed: '+name)
    overlay=overlay_module();locked,lock=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(lock==dict(path=SOURCE_LOCK,sha256=SOURCE_LOCK_SHA)==plan['source_overlay_lock']
        and locked['integration_complete'] is True and not locked['unpinned_sources'],'complete exact1141 source/CPU lock required')
    require(plan['jobs']==_jobs(overlay),'exact eight fixed sources/routes/output paths required')
    predecessor=predecessor_complete()
    require(plan.get('predecessor')==predecessor and plan.get('root_review')==root_review(lock,predecessor),'predecessor/root review changed')
    require(source_names(overlay,locked,plan['jobs'])<=set(plan['inputs_sha256']),'complete frozen input graph required')
    return locked


def prepare():
    require(not (ROOT/PLAN).exists() and not (ROOT/RUNTIME).exists(),'fresh unprepared owner namespace required')
    predecessor=predecessor_complete();overlay=overlay_module();locked,lock=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(locked['integration_complete'] is True and not locked['unpinned_sources'],'complete CPU-qualified1141 graph required')
    review=root_review(lock,predecessor);jobs=_jobs(overlay)
    for row in jobs:require(not (ROOT/row['out']).exists() and not (ROOT/row['runtime']).exists(),'fresh case namespaces required')
    plan=dict(schema=PLAN_SCHEMA,created_utc=utc(),expected_cases=8,source_parent_count=5,jobs=jobs,
        fixed_TRAIN_sources=FIXED_SOURCES,methods=list(METHODS),split='train',max_native_slots=1,
        automatic_retry=False,source_outcome_filtering=False,CPU_check_before_every_native_case=True,
        remaining_queue=[],development_regression_only=True,formal192_readiness=False,formal_VAL_TEST=False,
        source_overlay_lock=lock,predecessor_plan=dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA),
        predecessor=predecessor,root_review=review,
        inputs_sha256={name:sha(ROOT/name) for name in sorted(source_names(overlay,locked,jobs))})
    validate(plan);fresh(ROOT/PLAN,plan)
    result=dict(plan=PLAN,sha256=sha(ROOT/PLAN),cases=8,Native_started=False);print(json.dumps(result),flush=True);return result


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
    block=row['task'] in BLOCKS
    require(protocol.get('schema')==CANDIDATE_SCHEMA and protocol.get('source_overlay_lock')==lock
        and protocol.get('actual_worker')==pointer(WORKER) and protocol.get('actual_preparer')==pointer(PREPARER)
        and protocol.get('execution_branch')==('block1120' if block else 'core_selected_integration')
        and protocol.get('formal_score_contribution') is False and protocol.get('formal_VAL_TEST_qualification') is False,
        'actual frozen1141 worker/preparer/branch protocol required')
    if block:
        require(protocol.get('block_factory',{}).get('module')=='our_mani.multitask.author_aligned_block_integration_v1120'
            and protocol.get('changed_factor')=='no_core_policy_variant_applied'
            and protocol.get('public_phase_component_binding') is None,'original1120 block branch required')
    else:
        factor='author_wp4_release_with_confirmed_semantic_commitment' if row['kind']=='ring' else 'frozen_TRAIN_approach_curve_continuation'
        require(protocol.get('changed_factor')==factor and protocol.get('core_factory',{}).get('module')=='our_mani.multitask.author_ring_release_integration_v1140',
            'actual per-task author-release/Buzz1130 core required')
        component=protocol.get('public_phase_component_binding',{});rule=component.get('task_rule',{})
        source='our_mani/multitask/'+('author_ring_release_integration_v1140.py' if row['kind']=='ring' else 'public_buzz_approach_curve_v1130.py')
        require(component.get('integration',{}).get('schema')=='r10_author_ring_release_and_committed_role_integration_v1140'
            and rule.get('task')==row['task'] and rule.get('source_sha256')==FROZEN_PINS[source],
            'actual source-bound per-task rule required')


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
    clock=case.get('counted_clock',{});ticks=clock.get('ticks');alignment=case.get('alignment_receipt',{});owned=alignment.get('clock',{})
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt')==.05 and type(clock.get('seconds')) in (int,float)
        and abs(clock['seconds']-ticks*.05)<=1e-8 and owned.get('ticks')==ticks and owned.get('dt')==.05
        and owned.get('simulation_seconds')==clock['seconds'],'original200tick/.05s clock accounting required')
    require(case.get('version')==CANDIDATE_SCHEMA+'_case'
        and case.get('inherited_consumer_record_version')=='r10_full16_explicit_author_action_outcome_v744',
        'honest new1141 and inherited744 version boundary required')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case
    # Read-only projection for a verifier whose literal header is744. The new
    # header was checked above; neither the case object nor its file is edited.
    projected=dict(case,version=case['inherited_consumer_record_version'])
    receipt=require_sampling_case(projected,case['author_outcome_protocol'],block=case['task'] in BLOCKS)
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
    if block:
        require(hooks.get('reset')==[] and hooks.get('shared_development_hooks_applied') is False,'original block reset interface required')
    else:
        require(len(hooks.get('reset',[]))==1 and hooks['reset'][0].get('complete') is True
            and hooks.get('shared_development_hooks_applied') is True,'original core reset interface required')
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
        status=dict(schema=STATUS_SCHEMA,expected=8,completed=[],failed=[],active=None,stage='running',updated_utc=utc(),
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
                print(json.dumps(dict(completed=len(status['completed']),expected=8,last=row['job_id'],native_success=result['native_success']),ensure_ascii=False),flush=True)
            validate(plan);status.update(stage='complete',active=None,updated_utc=utc())
            fresh(base/'terminal.json',dict(complete=True,completed=8,infrastructure_failed=0,automatic_retry=False,finished_utc=utc(),development_regression_only=True,formal192_readiness=False))
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
