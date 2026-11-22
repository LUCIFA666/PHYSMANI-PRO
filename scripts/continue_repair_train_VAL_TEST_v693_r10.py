"""User-authorized ordered continuation, one launch per stage and no retries.

Adopt the existing677/679/401 repair smoke. Require its complete runtime proof
before new training, then exact192 TRAIN qualification, full VAL, then TEST.
Any infrastructure failure leaves evidence and requires a concrete repair.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
BASE=ROOT/'runtime/framework/ordered_continuation_v693'
ART=ROOT/'artifacts/framework/multitask_r10'
OWNER=ROOT/'runtime/framework/native_resource_owner_v401'
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


def wait_cases(stage,setup):
    while True:
        if stopped():raise InterruptedError('user-controlled project stop')
        acceptance=setup/'CPU_acceptance.json'
        failed=list(setup.glob('*_CPU_failed.json'))
        if failed:raise RuntimeError(f'{stage}: CPU failures retained; do not start new stages: '+','.join(p.name for p in failed[:3]))
        if acceptance.exists():
            d=json.loads(acceptance.read_text())
            if d.get('complete') is not True:raise RuntimeError(stage+': incomplete CPU admission')
            jobs=d['actual_argv_preflights'];complete=0;successes=0
            for job in jobs:
                folder=ROOT/job['argv'][job['argv'].index('--out')+1]
                if (folder/'infrastructure_error.json').exists():raise RuntimeError(stage+': native infrastructure error at '+str(folder))
                if (folder/'case.json').exists():
                    case=json.loads((folder/'case.json').read_text())
                    if case.get('complete') is not True or case.get('cleanup_completed') is not True:
                        raise RuntimeError(stage+': native case is incomplete at '+str(folder))
                    complete+=1;successes+=bool(case['official_task_success'])
            if complete==len(jobs):return
            pause(stage,expected=len(jobs),native_completed=complete,native_successes=successes)
        else:pause(stage,waiting_for_actual_CPU_acceptance=True)


def drain_owner(stage,*,close=False):
    launch=json.loads((OWNER/'launch.json').read_text())
    while True:
        if stopped():raise InterruptedError('user-controlled project stop')
        state=json.loads((OWNER/'status.json').read_text())
        pending=[]
        for p in (OWNER/'requests').glob('*.json'):
            if p.name in state.get('rejected',{}):continue
            r=json.loads(p.read_text())
            if r['job_id'] not in state['completed'] and r['job_id'] not in state['active_jobs']:pending.append(r['job_id'])
        if not state['active_jobs'] and not pending:break
        pause(stage,active_jobs=state['active_jobs'],pending=len(pending))
    if start_ticks(launch['pid'])!=launch['start_ticks']:raise ValueError('TRAIN owner identity changed')
    if close:
        if (OWNER/'STOP').exists():raise ValueError('inspect an existing TRAIN owner STOP before transition')
        atomic_json(OWNER/'STOP',dict(reason='all TRAIN qualification complete; hand native ownership to admitted VAL stage',created_utc=utc()))
        while Path(f'/proc/{launch["pid"]}/stat').exists():
            stat=Path(f'/proc/{launch["pid"]}/stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':break
            pause(stage,TRAIN_owner_draining=True)


def main():
    lock=(ROOT/'runtime/framework/ordered_continuation_v693.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if BASE.exists():raise FileExistsError('preserve existing ordered continuation; no duplicate supervisor')
    BASE.mkdir();atomic_json(BASE/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=start_ticks(os.getpid()),
        source_sha256=file_sha(__file__),user_order='repair then new joint16 training then VAL then TEST, until user stop',
        exactly_three_methods=['cv_geometry','context_geometry','physmani'],Value_NGFF_FASTER_enabled=False,automatic_retry=False))
    try:
        wait_cases('wait_executor_repair_smokes',ART/'rml_suite16_smoke_v677_setup')
        drain_owner('wait_TRAIN_owner_before_new_training')
        run('freeze_executors','scripts/audit_executor_freeze_v670_r10.py','--freeze')
        run('new_joint16_training','scripts/run_joint16_after_executor_freeze_v671_r10.py')
        run('independent_new_training_acceptance','scripts/audit_joint16_after_executor_training_v672_r10.py')
        preparation=launch('prepare_final_fixed_four','scripts/prepare_post_training_qualification_v680_r10.py')
        queue=launch('queue_verified_final_fixed_four','scripts/queue_verified_post_training_v687_r10.py')
        finish('prepare_final_fixed_four',preparation);finish('queue_verified_final_fixed_four',queue)
        wait_cases('wait_final_fixed_four_native',ART/'post_training_qualification_v680_setup')
        run('final_fixed_four_readiness','scripts/audit_post_training_readiness_v685_r10.py','--freeze')
        run('freeze_VAL_suite','scripts/prepare_formal_VAL_suite16_v688_r10.py')
        vs='artifacts/framework/multitask_r10/formal_suite16_v688_setup'
        run('freeze_VAL_sources','scripts/prepare_formal_split_source_lock_v686_r10.py','--suite',vs+'/suite.json',
            '--suite-lock',vs+'/frozen_inputs.json','--split','validation','--out',vs+'/validation_source_inputs.json')
        drain_owner('graceful_TRAIN_to_VAL_owner_transition',close=True)
        common=['--suite',vs+'/suite.json','--suite-lock',vs+'/frozen_inputs.json','--source-inputs',vs+'/validation_source_inputs.json',
                '--split','validation','--runtime','runtime/framework/formal_VAL_v689','--out','artifacts/framework/multitask_r10/formal_suite16_v688/validation']
        run('formal_VAL_945','scripts/run_formal_suite16_owner_v689_r10.py',*common)
        release='artifacts/framework/multitask_r10/formal_suite16_v688/VAL_release.json'
        run('independent_VAL_release','scripts/audit_formal_suite16_results_v690_r10.py',*common[:-2], '--out',release)
        run('freeze_TEST_suite_after_complete_VAL','scripts/prepare_formal_TEST_suite16_v691_r10.py')
        ts='artifacts/framework/multitask_r10/formal_TEST_suite16_v691_setup'
        run('freeze_TEST_sources_after_complete_VAL','scripts/prepare_formal_split_source_lock_v686_r10.py','--suite',ts+'/suite.json',
            '--suite-lock',ts+'/frozen_inputs.json','--split','test','--validation-release',release,'--out',ts+'/test_source_inputs.json')
        common=['--suite',ts+'/suite.json','--suite-lock',ts+'/frozen_inputs.json','--source-inputs',ts+'/test_source_inputs.json',
                '--split','test','--validation-release',release,'--runtime','runtime/framework/formal_TEST_v689',
                '--out','artifacts/framework/multitask_r10/formal_suite16_v688/test']
        run('formal_TEST_1920','scripts/run_formal_suite16_owner_v689_r10.py',*common)
        result='artifacts/framework/multitask_r10/formal_suite16_v688/TEST_results.json'
        run('independent_TEST_results','scripts/audit_formal_suite16_results_v690_r10.py',*common[:-2],'--out',result)
        atomic_json(BASE/'terminal.json',dict(complete=True,finished_utc=utc(),TEST_results=result,automatic_retry=False))
        pulse('complete',TEST_results=result)
    except BaseException as error:
        for child,_,_ in CHILDREN:
            if child.poll() is None:interrupt_owned_tree(child)
        failure=dict(complete=False,finished_utc=utc(),error=repr(error),traceback=traceback.format_exc(),
            user_stop=isinstance(error,InterruptedError),automatic_retry=False,previous_native_results_preserved=True)
        atomic_json(BASE/'failure.json',failure);pulse('user_stopped' if failure['user_stop'] else 'requires_concrete_repair',error=repr(error))
        raise
    finally:
        for _,log,err in CHILDREN:log.close();err.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
