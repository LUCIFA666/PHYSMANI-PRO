"""New author-outcome TRAIN qualification, complete VAL, then complete TEST.

Reuse accepted1600-parent training. Old tick-fatal VAL cases are quarantined.
Each new stage launches once; no outcome filtering or automatic retries.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
BASE=ROOT/'runtime/framework/ordered_author_outcome_v758'
ART=ROOT/'artifacts/framework/multitask_r10'
OWNER=ROOT/'runtime/framework/native_resource_owner_v748'
CHILDREN=[]


def utc():return datetime.now(timezone.utc).isoformat()
def start_ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def stopped():return (BASE/'STOP').exists() or (ROOT/'runtime/framework/multitask_r10_STOP').exists()
def pulse(stage,**progress):
    atomic_json(BASE/'status.json',dict(updated_utc=utc(),stage=stage,**progress,automatic_retry=False),overwrite=True)
def pause(stage,**progress):
    if stopped():raise InterruptedError('user-controlled project stop')
    pulse(stage,**progress);time.sleep(5.)


def launch(stage,script,*args):
    if stopped():raise InterruptedError('user-controlled project stop')
    leaf=BASE/'stages'/stage;leaf.mkdir(parents=True,exist_ok=False)
    argv=[str(ROOT/'.conda-env/bin/python'),script,*args]
    log=(leaf/'stdout.log').open('xb');err=(leaf/'stderr.log').open('xb')
    child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=err,start_new_session=True)
    CHILDREN.append((child,log,err))
    atomic_json(leaf/'launch.json',dict(created_utc=utc(),argv=argv,pid=child.pid,start_ticks=start_ticks(child.pid),
        source_sha256=file_sha(ROOT/script),automatic_retry=False))
    return child,leaf


def finish(stage,launched):
    child,leaf=launched
    while child.poll() is None:pause(stage,child_pid=child.pid)
    code=child.wait();atomic_json(leaf/'terminal.json',dict(complete=code==0,returncode=code,finished_utc=utc(),automatic_retry=False))
    if code:raise RuntimeError(f'{stage} failed with exit{code}; preserve logs and repair before any new stage')


def run(stage,script,*args):finish(stage,launch(stage,script,*args))


def run_formal(stage,script,*args):
    launched=launch(stage,script,*args);child,leaf=launched
    runtime=ROOT/args[args.index('--runtime')+1]
    detected=None
    while child.poll() is None:
        status_path=runtime/'status.json'
        if status_path.exists():
            state=json.loads(status_path.read_text())
            if state.get('failed') and detected is None:
                detected={k:v for k,v in list(state['failed'].items())[:3]}
                stop=runtime/'DRAIN'
                if not stop.exists():atomic_json(stop,dict(reason='concrete infrastructure failure; stop new scheduling and naturally drain owned children',created_utc=utc(),failures=detected))
        pause(stage,child_pid=child.pid,infrastructure_failure_detected=detected)
    finish(stage,launched)
    terminal=json.loads((runtime/'terminal.json').read_text())
    if detected or terminal.get('complete') is not True:
        raise RuntimeError('formal infrastructure failed; no retry: '+json.dumps(detected or terminal))


def live_owned(launch_path):
    if not launch_path.exists():return False
    launch_record=json.loads(launch_path.read_text());proc=Path(f'/proc/{launch_record["pid"]}/stat')
    if not proc.exists():return False
    stat=proc.read_text().rsplit(')',1)[1].split()
    return int(stat[19])==launch_record['start_ticks'] and stat[0]!='Z'


def request_drain(runtime,reason):
    if live_owned(runtime/'launch.json') and not (runtime/'DRAIN').exists():
        atomic_json(runtime/'DRAIN',dict(reason=reason,created_utc=utc(),user_stop_received=False))


def wait_qualification():
    setup=ART/'author_outcome_qualification_v745_setup'
    while True:
        state=json.loads((OWNER/'status.json').read_text())
        failed={name:r for name,r in state['completed'].items() if r.get('complete') is not True}
        if failed or state.get('rejected'):
            request_drain(OWNER,'new TRAIN infrastructure failure; naturally finish active cases')
            raise RuntimeError('new TRAIN qualification infrastructure error: '+json.dumps(failed or state['rejected']))
        if len(state['completed'])==192:
            if state['active_jobs']:raise ValueError('192 cases cannot leave additional active work')
            return
        pause('new_author_outcome_TRAIN_192',expected=192,completed=len(state['completed']),
              active=state['active_jobs'],GPU_slots=state['allocated_GPU_slots'])


def set_reporter_source():
    path=ROOT/'runtime/framework/formal_VAL_report_v750/source.json'
    old=json.loads(path.read_text())
    old.update(runtime='runtime/framework/formal_VAL_v753',controller=str(BASE.relative_to(ROOT)),
        validation_release='artifacts/framework/multitask_r10/formal_suite16_v752/VAL_release.json',
        protocol_status='formal_VAL')
    atomic_json(path,old,overwrite=True)


def main():
    lock=(ROOT/'runtime/framework/ordered_author_outcome_v758.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if BASE.exists():raise FileExistsError('preserve the one-time new-protocol controller')
    handoff=ROOT/'runtime/framework/outcome_sampling_handoff_v742/scheduler_superseded.json'
    record=json.loads(handoff.read_text())
    if (record.get('complete') is not True or record.get('user_stop_received') is not False
        or record.get('native_or_GPU_interrupted') is not False or record.get('CPU_interrupted') is not False
        or record.get('old_VAL_cohort_superseded') is not True):
        raise ValueError('old-cohort noninterrupting supersession evidence required')
    for runtime in ('ordered_continuation_v735','formal_VAL_v731','native_resource_owner_v401'):
        if live_owned(ROOT/'runtime/framework'/runtime/'launch.json'):
            raise RuntimeError('previous scheduling owner is still alive: '+runtime)
    trained=json.loads((ROOT/'runtime/framework/joint16_after_executor_training_v671/terminal.json').read_text())
    if (trained.get('complete') is not True or trained['TRAIN_parents']!=1600 or trained['epochs']!=30
        or trained['checkpoint']['sha256']!=record['new_training_checkpoint_sha256']):
        raise ValueError('retain the accepted completed joint16 training')
    BASE.mkdir();atomic_json(BASE/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=start_ticks(os.getpid()),
        source_sha256=file_sha(__file__),superseded_cohort_evidence=dict(path=str(handoff.relative_to(ROOT)),sha256=file_sha(handoff)),
        accepted_training=trained['checkpoint'],new_training_repeated=False,new_TRAIN_runtime_cases=192,
        exactly_three_methods=['cv_geometry','context_geometry','physmani'],Value_NGFF_FASTER_enabled=False,
        automatic_retry=False,user_order='correct author executor protocol, unified trained model, complete VAL then TEST until user stop'))
    try:
        train_owner=launch('new_TRAIN_native_owner','scripts/run_native_resource_owner_v748_r10.py')
        while not (OWNER/'status.json').exists():
            if train_owner[0].poll() is not None:raise RuntimeError('new TRAIN owner failed before status')
            pause('starting_new_TRAIN_owner')
        preparation=launch('new_TRAIN_actual_CPU_checks','scripts/prepare_author_outcome_qualification_v745_r10.py')
        queue=launch('queue_new_TRAIN_cases_once','scripts/queue_author_outcome_qualification_v747_r10.py')
        setup=ART/'author_outcome_qualification_v745_setup'
        while preparation[0].poll() is None or queue[0].poll() is None:
            for stage,launched in (('new_TRAIN_actual_CPU_checks',preparation),('queue_new_TRAIN_cases_once',queue)):
                code=launched[0].poll()
                if code is not None and code!=0:
                    raise RuntimeError(stage+f' failed with exit{code}; preserve stage logs')
            if train_owner[0].poll() is not None:
                raise RuntimeError('new TRAIN owner exited before the qualification completed')
            if preparation[0].poll()==0:
                acceptance=setup/'CPU_acceptance.json'
                if not acceptance.exists() or json.loads(acceptance.read_text()).get('complete') is not True:
                    raise RuntimeError('CPU preparation ended without complete192-case acceptance')
            if list(setup.glob('*_CPU_failed.json')):
                request_drain(OWNER,'actual CPU qualification failed; do not start further native cases')
                raise RuntimeError('new TRAIN actual-argv CPU failure; preserve evidence before any repair')
            if (OWNER/'status.json').exists():
                state=json.loads((OWNER/'status.json').read_text())
                if state.get('rejected') or any(r.get('complete') is not True for r in state['completed'].values()):
                    request_drain(OWNER,'new TRAIN infrastructure failure; naturally drain existing cases')
                    raise RuntimeError('new TRAIN native infrastructure failure')
            pause('new_TRAIN_CPU_checks_and_runtime',CPU_acceptance_exists=(setup/'CPU_acceptance.json').exists())
        finish('new_TRAIN_actual_CPU_checks',preparation);finish('queue_new_TRAIN_cases_once',queue)
        accepted=json.loads((setup/'CPU_acceptance.json').read_text())
        if accepted.get('complete') is not True or len(accepted['actual_argv_preflights'])!=192:
            raise ValueError('all192 actual new-runtime CPU checks required')
        wait_qualification()
        run('independent_new_TRAIN_readiness','scripts/audit_author_outcome_readiness_v746_r10.py','--freeze')
        request_drain(OWNER,'all192 TRAIN cases independently accepted; release ownership for VAL')
        finish('new_TRAIN_native_owner',train_owner)
        run('freeze_new_VAL_suite','scripts/prepare_formal_VAL_author_outcome_v752_r10.py')
        vs='artifacts/framework/multitask_r10/formal_suite16_v752_setup'
        run('freeze_new_VAL_sources','scripts/prepare_formal_split_source_lock_v756_r10.py','--suite',vs+'/suite.json',
            '--suite-lock',vs+'/frozen_inputs.json','--split','validation','--out',vs+'/validation_source_inputs.json')
        run('new_VAL_actual_CPU_probes','scripts/check_formal_admission_v757_r10.py','--suite',vs+'/suite.json',
            '--suite-lock',vs+'/frozen_inputs.json','--source-inputs',vs+'/validation_source_inputs.json',
            '--out','runtime/framework/formal_admission_preflight_v757')
        common=['--suite',vs+'/suite.json','--suite-lock',vs+'/frozen_inputs.json','--source-inputs',vs+'/validation_source_inputs.json',
            '--split','validation','--runtime','runtime/framework/formal_VAL_v753',
            '--out','artifacts/framework/multitask_r10/formal_suite16_v752/validation']
        set_reporter_source()
        run_formal('formal_VAL_945','scripts/run_formal_suite16_owner_v753_r10.py',*common)
        release='artifacts/framework/multitask_r10/formal_suite16_v752/VAL_release.json'
        run('independent_VAL_release','scripts/audit_formal_suite16_results_v754_r10.py',*common[:-2],'--out',release)
        run('freeze_TEST_after_complete_VAL','scripts/prepare_formal_TEST_author_outcome_v755_r10.py')
        ts='artifacts/framework/multitask_r10/formal_TEST_suite16_v755_setup'
        run('freeze_TEST_sources_after_complete_VAL','scripts/prepare_formal_split_source_lock_v756_r10.py',
            '--suite',ts+'/suite.json','--suite-lock',ts+'/frozen_inputs.json','--split','test',
            '--validation-release',release,'--out',ts+'/test_source_inputs.json')
        common=['--suite',ts+'/suite.json','--suite-lock',ts+'/frozen_inputs.json','--source-inputs',ts+'/test_source_inputs.json',
            '--split','test','--validation-release',release,'--runtime','runtime/framework/formal_TEST_v753',
            '--out','artifacts/framework/multitask_r10/formal_suite16_v752/test']
        run_formal('formal_TEST_1920','scripts/run_formal_suite16_owner_v753_r10.py',*common)
        result='artifacts/framework/multitask_r10/formal_suite16_v752/TEST_results.json'
        run('independent_TEST_results','scripts/audit_formal_suite16_results_v754_r10.py',*common[:-2],'--out',result)
        atomic_json(BASE/'terminal.json',dict(complete=True,finished_utc=utc(),TEST_results=result,automatic_retry=False))
        pulse('complete',TEST_results=result)
    except BaseException as error:
        user_stop=isinstance(error,InterruptedError)
        if not user_stop:
            runtimes=[OWNER,ROOT/'runtime/framework/formal_VAL_v753',ROOT/'runtime/framework/formal_TEST_v753']
            for runtime in runtimes:request_drain(runtime,'concrete repair needed; preserve and naturally drain existing cases')
            while any(live_owned(r/'launch.json') for r in runtimes):
                pulse('draining_for_concrete_repair',error=repr(error));time.sleep(5.)
                if stopped():user_stop=True;break
        for child,_,_ in CHILDREN:
            if child.poll() is None:interrupt_owned_tree(child)
        failure=dict(complete=False,finished_utc=utc(),error=repr(error),traceback=traceback.format_exc(),
            user_stop=user_stop,automatic_retry=False,previous_native_results_preserved=True)
        atomic_json(BASE/'failure.json',failure)
        pulse('user_stopped' if user_stop else 'requires_concrete_repair',error=repr(error))
        raise
    finally:
        for _,log,err in CHILDREN:log.close();err.close()


if __name__=='__main__':main()


