"""Five fixed matched TRAIN author first-release cases; default is inert.

CPU and Native execute serially once per fixed source/method. Only children own
the1106 lease; the parent holds its owner flock. The Root-controlled51-case stop
is an incomplete predecessor, never a natural192 completion or qualification.
No observed outcome filters sources; NativeFalse/fatal/motionFalse are retained.
"""
import argparse
from datetime import datetime,timezone
import fcntl,hashlib,json,os
from pathlib import Path
import subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
SOURCE='scripts/run_TRAIN_native_release_comparison_owner_v1166_r10.py'
TEST_SOURCE='tests/test_train_native_release_comparison_owner_v1166.py'
WORKER='scripts/run_TRAIN_native_release_full16_overlay_v1165_r10.py'
PREPARER='our_mani/multitask/train_native_release_full16_overlay_v1165.py'
WRAPPER='scripts/run_frozen_system_environment_v820_r10.py'
SIM='scripts/run_local_sim.sh'
SAMPLER_VERIFIER='our_mani/multitask/author_outcome_receipt_v749.py'
NAME='TRAIN_native_release_full16_v1165'
SETUP='artifacts/framework/multitask_r10/'+NAME+'_setup/'
ART='artifacts/framework/multitask_r10/'+NAME+'/development_demo_only/matched_v1166'
CASE_RUNTIME='runtime/framework/'+NAME+'/cases/matched_v1166'
RUNTIME='runtime/framework/'+NAME+'/owner_v1166'
PLAN=SETUP+'bounded_plan_v1166.json'
ROOT_REVIEW=SETUP+'root_review_v1166.json'
SOURCE_LOCK=SETUP+'source_overlay_lock.json'
SOURCE_LOCK_SHA='7d1b53b6a56bd48012e0850320b859d765b7c4b6d32f94f6aa607a22f8eef929'
ENV_SHA='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA='r10_bounded_five_TRAIN_native_release_comparison_plan_v1166'
STATUS_SCHEMA='r10_bounded_five_TRAIN_native_release_comparison_status_v1166'
REVIEW_SCHEMA='r10_fixed_five_native_release_comparison_review_v1166'
PRIOR_OWNER='runtime/framework/repaired_policy_owner_v1149/train/conservative_v1159_train192'
PRIOR_STOP=PRIOR_OWNER+'/controlled_stop_receipt.json'
PRIOR_STOP_SHA='2b338d68ad079facad2703d0d1393980f86611d2a64095bf75b77d0eddff5bad'
BASKETS=('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
BUZZ_TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
TASKS=(*BASKETS,*BUZZ_TASKS)
METHODS=('cv_geometry','context_geometry')
CANARIES=(
 ('moving_basketball_in_hoop','episode104','cv_geometry'),
 ('moving_basketball_in_hoop','episode104','context_geometry'),
 ('moving_basketball_in_hoop_high_speed','episode0','context_geometry'),
 ('beat_the_rotating_buzz','episode10','cv_geometry'),
 ('beat_the_rotating_buzz_high_speed','episode102','context_geometry'),
)
FIXED_SOURCES=[dict(task=t,raw_episode_id=r,method=m)for t,r,m in CANARIES]
CANDIDATE_SCHEMA='r10_full16_fixed_TRAIN_native_release_overlay_v1165'
CORE_MODULE='our_mani.multitask.author_native_release_core_v1164'
CORE_SOURCE=CORE_MODULE.replace('.','/')+'.py'
CORE_SCHEMA='r10_retained_author_native_release_core_integration_v1164'
CHANGED_FACTOR='retained_original_author_first_release_qualification'
COMPARISON_PARENT_LOCK=dict(path='artifacts/framework/multitask_r10/TRAIN_block_preclose_full16_v1154_setup/source_overlay_lock.json',
 sha256='0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9')
FROZEN_PINS={
 PREPARER:'04c39e55792376bac9a869f743f3702ba934e1d481fa42aa1d77c25a2ca21e93',
 WORKER:'88d216ddeecbf1c4d61062bb1699a54d4eeac1e6c53a1c0df602e580f6707724',
 'tests/test_train_native_release_full16_overlay_v1165.py':'042d53362fca820481fdc2d5ddf4d6bda04677023d202d22c0a8f979050d0199',
 CORE_SOURCE:'14f005b3c5b6b96a63095a8560ec9192624cd3566a59f6e13b36b2a43116ffbd',
 'tests/test_author_native_release_core_v1164.py':'5ec13e57a092297ab888c47a2f976860a51209d257008b833539509f84f23c41',
}


def require(value, message):
    if not value: raise ValueError('TRAIN1166: ' + message)


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
    require(type(SOURCE_LOCK_SHA)is str and len(SOURCE_LOCK_SHA)==64
        and all(type(v)is str and len(v)==64 for v in FROZEN_PINS.values()),'pending real1165 lock/source pins cannot prepare or execute')
    for name,expected in FROZEN_PINS.items():require(sha(ROOT/name)==expected,'frozen source changed: '+name)
    require(sha(ROOT/SOURCE_LOCK)==SOURCE_LOCK_SHA,'exact frozen1165 overlay lock required')
    if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
    from our_mani.multitask import train_native_release_full16_overlay_v1165
    return train_native_release_full16_overlay_v1165


def kind(task):
    require(task in TASKS,'only exact four author first-release tasks')
    return 'basket' if task in BASKETS else 'buzz' 


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
    require(len(rows)==len({r['job_id'] for r in rows})==5,'exactly five unique source-method cases required')
    return rows


def predecessor_complete():
    overlay=overlay_module()
    proof=overlay.require_prior_stop(ROOT,PRIOR_STOP)
    require(proof==dict(path=PRIOR_STOP,sha256=PRIOR_STOP_SHA),'actual hash-pinned controlled51-case stop required')
    return dict(prior_controlled_stop=proof,incomplete_prior_cohort=True,
        prior192_readiness_granted=False,prior_outcomes_preserved=True)


def root_review(lock,predecessor):
    value=json.loads((ROOT/ROOT_REVIEW).read_text())
    require(value.get('schema')==REVIEW_SCHEMA and value.get('review_complete')is True
        and value.get('release_bounded_five')is True and value.get('reviewer')=='root'
        and value.get('fixed_TRAIN_sources')==FIXED_SOURCES and value.get('source_overlay_lock')==lock
        and value.get('prior_controlled_stop')==predecessor['prior_controlled_stop']
        and value.get('development_regression_only')is True and value.get('formal192_readiness')is False
        and value.get('changed_factor')==CHANGED_FACTOR,'Root technical review for only five fixed matched cases required')
    return pointer(ROOT_REVIEW)


def source_names(overlay,locked,jobs):
    names={SOURCE,TEST_SOURCE,WORKER,PREPARER,WRAPPER,SIM,SAMPLER_VERIFIER,SOURCE_LOCK,ROOT_REVIEW,
        *(PRIOR_OWNER+'/'+x for x in ('plan.json','launch.json','status.json','failure.json','STOP','controlled_stop_receipt.json')),
        overlay.MANIFEST,overlay.ROUTES,overlay.TRAIN_INPUTS,*overlay.OVERLAY_SOURCES,*overlay.FROZEN_PINS,*FROZEN_PINS,
        locked['integration_cpu_evidence']['path'],*(row['config']for row in jobs)}
    from scripts import run_TRAIN_native_release_full16_overlay_v1165_r10 as worker
    names.update(entry['path']for entry in worker.ENVIRONMENT.values())
    return names


def _jobs(overlay):
    roster=json.loads(overlay.raw(ROOT,overlay.MANIFEST,overlay.MANIFEST_SHA))
    routes=json.loads(overlay.raw(ROOT,overlay.ROUTES,overlay.ROUTES_SHA))
    rows=build_jobs(roster,routes)
    for row in rows:
        overlay.selection(ROOT,argparse.Namespace(task=row['task'],method=row['method'],raw_episode_id=row['raw'],config=row['config'],split='train',check=True,execute=False,frozen_inputs=None))
        require(sha(ROOT/row['config'])==row['config_sha256'],'original route config changed')
    return rows


def validate(plan):
    require(plan.get('schema')==PLAN_SCHEMA and plan.get('expected_cases')==5 and plan.get('source_parent_count')==4
        and plan.get('fixed_TRAIN_sources')==FIXED_SOURCES and plan.get('split')=='train'
        and plan.get('methods')==list(METHODS) and plan.get('max_native_slots')==1
        and plan.get('automatic_retry')is False and plan.get('source_outcome_filtering')is False
        and plan.get('CPU_check_before_every_native_case')is True and plan.get('remaining_queue')==[]
        and plan.get('development_regression_only')is True and plan.get('formal192_readiness')is False
        and plan.get('formal_VAL_TEST')is False and plan.get('changed_factor')==CHANGED_FACTOR
        and plan.get('Native_RRT_physics_thresholds_targets_weights_changed')is False,
        'fixed five-case author first-release comparison protocol required')
    for name,checksum in plan['inputs_sha256'].items():require(sha(ROOT/name)==checksum,'pinned input changed: '+name)
    overlay=overlay_module();locked,lock=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(lock==dict(path=SOURCE_LOCK,sha256=SOURCE_LOCK_SHA)==plan['source_overlay_lock']
        and locked['integration_complete']is True and not locked['unpinned_sources'],'complete actual1165 source/CPU lock required')
    require(plan['jobs']==_jobs(overlay),'exact fixed sources/routes/output paths required')
    predecessor=predecessor_complete()
    require(plan.get('predecessor')==predecessor and plan.get('root_review')==root_review(lock,predecessor),'prior-stop/Root review changed')
    require(source_names(overlay,locked,plan['jobs'])<=set(plan['inputs_sha256']),'complete frozen input graph required')
    return locked


def prepare():
    require(not(ROOT/PLAN).exists()and not(ROOT/RUNTIME).exists(),'fresh unprepared owner namespace required')
    predecessor=predecessor_complete();overlay=overlay_module();locked,lock=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(locked['integration_complete']is True and not locked['unpinned_sources'],'complete CPU-qualified1165 graph required')
    review=root_review(lock,predecessor);jobs=_jobs(overlay)
    for row in jobs:require(not(ROOT/row['out']).exists()and not(ROOT/row['runtime']).exists(),'fresh case namespaces required')
    plan=dict(schema=PLAN_SCHEMA,created_utc=utc(),expected_cases=5,source_parent_count=4,jobs=jobs,
        fixed_TRAIN_sources=FIXED_SOURCES,methods=list(METHODS),split='train',max_native_slots=1,
        automatic_retry=False,source_outcome_filtering=False,CPU_check_before_every_native_case=True,
        remaining_queue=[],development_regression_only=True,formal192_readiness=False,formal_VAL_TEST=False,
        changed_factor=CHANGED_FACTOR,Native_RRT_physics_thresholds_targets_weights_changed=False,
        source_overlay_lock=lock,predecessor=predecessor,root_review=review,
        inputs_sha256={name:sha(ROOT/name)for name in sorted(source_names(overlay,locked,jobs))})
    validate(plan);fresh(ROOT/PLAN,plan)
    result=dict(plan=PLAN,sha256=sha(ROOT/PLAN),cases=5,Native_started=False);print(json.dumps(result),flush=True);return result


def command(row,check=False):
    argv=[str(ROOT/'.conda-env/bin/python'),'-B',WRAPPER,'--manifest-sha256',ENV_SHA,'--','bash',SIM,WORKER,
        '--config',row['config'],'--task',row['task'],'--raw-episode-id',row['raw'],'--method',row['method'],
        '--out',row['out'],'--runtime',row['runtime'],'--overlay-lock',SOURCE_LOCK,'--split','train']
    return argv+(['--check']if check else ['--execute','--prior-stop',PRIOR_STOP])


def _stop_requested():
    return any((ROOT/name).exists() for name in (RUNTIME+'/STOP','runtime/framework/'+NAME+'/STOP','runtime/framework/multitask_r10_STOP'))


def status_write(status):
    atomic(ROOT/RUNTIME/'status.json',status)


def validate_protocol(row,protocol,lock):
    require(row['task']in TASKS and protocol.get('schema')==CANDIDATE_SCHEMA
        and protocol.get('source_overlay_lock')==lock
        and protocol.get('actual_worker')==pointer(WORKER)and protocol.get('actual_preparer')==pointer(PREPARER)
        and protocol.get('execution_branch')=='core_selected_integration'
        and protocol.get('comparison_parent_source_overlay_lock')==COMPARISON_PARENT_LOCK
        and protocol.get('core_changed_relative_to1154')is True
        and protocol.get('block_changed_relative_to1154')is False and protocol.get('baseline_changed_relative_to1154')is False
        and protocol.get('formal_score_contribution')is False and protocol.get('formal_VAL_TEST_qualification')is False
        and protocol.get('changed_factor')==CHANGED_FACTOR
        and protocol.get('core_factory',{}).get('module')==CORE_MODULE
        and protocol['core_factory'].get('function')=='integrated_components'
        and protocol['core_factory'].get('source')==CORE_SOURCE
        and protocol['core_factory'].get('sha256')==FROZEN_PINS[CORE_SOURCE],
        'actual source-bound1164 retained first-release route required')
    component=protocol.get('public_phase_component_binding',{});keys={'integration','task_rule'}
    if row['task']in BUZZ_TASKS:keys.add('unchanged_Buzz_approach_rule')
    require(type(component)is dict and set(component)==keys
        and component['integration'].get('source_sha256')==FROZEN_PINS[CORE_SOURCE]
        and component['integration'].get('schema')==CORE_SCHEMA
        and component['task_rule'].get('task')==row['task']
        and component['task_rule'].get('source_sha256')==FROZEN_PINS[CORE_SOURCE]
        and component['task_rule'].get('changed_factor')==CHANGED_FACTOR
        and component['task_rule'].get('controlled_phase')==('deposit'if row['task']in BASKETS else'release'),
        'actual1164 per-task first-release binding required')
    if row['task']in BUZZ_TASKS:
        require(component['unchanged_Buzz_approach_rule'].get('task')==row['task'], 'unchanged1130 task binding required')


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
    require(sidecar.get('actual_worker')==pointer(WORKER) and sidecar.get('actual_preparer')==pointer(PREPARER),'actual new worker/preparer header required')
    require(sidecar.get('prior_controlled_stop')==(None if check else dict(path=PRIOR_STOP,sha256=PRIOR_STOP_SHA)),'honest CPU/Native prior-stop binding required')
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
    require(case.get('task') in TASKS,'only original core741 sampling is supported by this matched owner')
    clock=case.get('counted_clock',{});ticks=clock.get('ticks');alignment=case.get('alignment_receipt',{});owned=alignment.get('clock',{})
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt')==.05 and type(clock.get('seconds')) in (int,float)
        and abs(clock['seconds']-ticks*.05)<=1e-8 and type(owned.get('ticks')) is int and owned['ticks']==ticks and owned.get('dt')==.05
        and owned.get('simulation_seconds')==clock['seconds'],'original200tick/.05s clock accounting required')
    require(case.get('version')==CANDIDATE_SCHEMA+'_case'
        and case.get('inherited_consumer_record_version')=='r10_full16_explicit_author_action_outcome_v744',
        'honest new1165 and inherited744 version boundary required')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case
    # Read-only projection for a verifier whose literal header is744. The new
    # header was checked above; neither the case object nor its file is edited.
    projected=dict(case,version=case['inherited_consumer_record_version'])
    receipt=require_sampling_case(projected,case['author_outcome_protocol'],block=False)
    return dict(counted_clock=clock,completed_actions=len(receipt['completed_actions']),
        sampler_schema=receipt['schema'],original_clock_and_action_sampling_preserved=True)


def completed_case(row,cpu,native,lock):
    for phase in (cpu,native):
        require(type(phase) is dict and set(phase)=={'launch','terminal','sidecar'},
            'actual compact launch/terminal/sidecar references required')
        for name in ('launch','terminal','sidecar'):checked_reference(phase[name])
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
    require(type(case.get('all_stage_counted_joint_motion',{}).get('all_stage_actual_window_checks_passed'))is bool,'original measured motion diagnostic must be retained')
    hook=row['out']+'/owned_hook_receipt.json';hooks=json.loads((ROOT/hook).read_text())
    require(hooks.get('actual_split')=='train' and all(x.get('close_complete') is True for x in hooks.get('paths',[])), 'actual original cleanup hooks required')
    require(type(hooks.get('reset')) is list and len(hooks['reset'])==1
        and hooks['reset'][0].get('complete') is True and hooks.get('shared_development_hooks_applied') is True,
        'one original captured core reset interface required')
    return dict(job_id=row['job_id'],task=row['task'],raw=row['raw'],method=row['method'],
        native_success=case['official_task_success'],native_terminal_failure=case['native_terminal_failure'],
        motion_pass=case['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed'],clock_quality=clock,case=case_ref,owned_hooks=pointer(hook),CPU_check=cpu,native=native)


def execute(checksum):
    require(bool(checksum) and sha(ROOT/PLAN)==checksum,'explicit actual plan SHA required')
    plan=json.loads((ROOT/PLAN).read_text());validate(plan);predecessor=predecessor_complete();base=ROOT/RUNTIME;base.mkdir(parents=True,exist_ok=True)
    with (base/'owner.lock').open('a+') as owner_lock:
        fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);require(not (base/'launch.json').exists(),'already launched; no automatic resume/retry')
        for row in plan['jobs']:require(not (ROOT/row['out']).exists() and not (ROOT/row['runtime']).exists(),'all new case namespaces must be fresh')
        fresh(base/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
            argv=list(sys.argv),plan=dict(path=PLAN,sha256=checksum),source_overlay_lock=plan['source_overlay_lock'],predecessor=predecessor,automatic_retry=False,max_native_slots=1))
        status=dict(schema=STATUS_SCHEMA,expected=5,completed=[],failed=[],active=None,stage='running',updated_utc=utc(),
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
                print(json.dumps(dict(completed=len(status['completed']),expected=5,last=row['job_id'],native_success=result['native_success']),ensure_ascii=False),flush=True)
            validate(plan);status.update(stage='complete',active=None,updated_utc=utc())
            fresh(base/'terminal.json',dict(complete=True,completed=5,infrastructure_failed=0,automatic_retry=False,finished_utc=utc(),development_regression_only=True,formal192_readiness=False))
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
