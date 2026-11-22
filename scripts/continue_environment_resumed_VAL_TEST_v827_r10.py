"""Continue the same VAL grid after exact private frozen-OS environment restoration.

Preserve591 completed outcomes; run354 remaining cases once in new outputs.
The physical policy, models and source identities stay frozen. No TEST data
is selected before the complete independent resumed945-case VAL release.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
from scripts.run_environment_resumed_formal_VAL_owner_v823_r10 import load_resume_control,pointer
from scripts import run_environment_resumed_formal_VAL_owner_v823_r10 as owner

BASE=ROOT/'runtime/framework/ordered_env_resume_VAL_TEST_v827'
SETUP='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup'
ART='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823'
VAL_RUNTIME='runtime/framework/formal_VAL_env_resume_v823'
TEST_RUNTIME='runtime/framework/formal_TEST_after_environment_resume_v753'
CHILDREN=[]
ENVIRONMENT_RECEIPT=None


def utc():return datetime.now(timezone.utc).isoformat()
def stopped():return ((BASE/'STOP').exists() or (ROOT/'runtime/framework/multitask_r10_STOP').exists()
    or (ROOT/'runtime/framework/ordered_resume_VAL_TEST_v775/STOP').exists()
    or (ROOT/'runtime/framework/formal_VAL_resume_v772/STOP').exists()
    or (ROOT/VAL_RUNTIME/'STOP').exists() or (ROOT/TEST_RUNTIME/'STOP').exists())
def pulse(stage,**kwargs):atomic_json(BASE/'status.json',dict(updated_utc=utc(),stage=stage,automatic_retry=False,
    frozen_environment=ENVIRONMENT_RECEIPT,**kwargs),overwrite=True)
def guard():
    if stopped():raise InterruptedError('explicit user stop')
def alive(record):
    path=Path('/proc')/str(record['pid'])/'stat'
    if not path.exists():return False
    stat=path.read_text().rsplit(')',1)[1].split()
    return stat[0]!='Z' and int(stat[19])==record['start_ticks']


def launch(stage,script,*args):
    guard()
    if not ENVIRONMENT_RECEIPT or os.readlink('/proc/self/ns/mnt')!=ENVIRONMENT_RECEIPT['current_mount_namespace']:
        raise ValueError('every child stage must inherit the actual820 namespace')
    leaf=BASE/'stages'/stage;leaf.mkdir(parents=True,exist_ok=False)
    argv=[str(ROOT/'.conda-env/bin/python'),'-u',script,*args]
    log=(leaf/'stdout.log').open('xb');err=(leaf/'stderr.log').open('xb')
    child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=err,start_new_session=True)
    CHILDREN.append((child,log,err))
    ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    atomic_json(leaf/'launch.json',dict(created_utc=utc(),argv=argv,pid=child.pid,start_ticks=ticks,source_sha256=file_sha(ROOT/script),
        inherited_frozen_environment=ENVIRONMENT_RECEIPT,current_mount_namespace=os.readlink('/proc/self/ns/mnt'),automatic_retry=False))
    return child,leaf


def run(stage,script,*args,formal_runtime=None):
    child,leaf=launch(stage,script,*args);detected=None
    while child.poll() is None:
        guard()
        if formal_runtime:
            status=ROOT/formal_runtime/'status.json'
            if status.exists():
                state=json.loads(status.read_text())
                if state.get('failed') and detected is None:
                    detected=state['failed'];drain=ROOT/formal_runtime/'DRAIN'
                    if not drain.exists():atomic_json(drain,dict(created_utc=utc(),reason='concrete new infrastructure failure; naturally drain active children'))
        pulse(stage,child_pid=child.pid,infrastructure_failure_detected=detected);time.sleep(2.)
    code=child.wait();atomic_json(leaf/'terminal.json',dict(complete=code==0,returncode=code,finished_utc=utc(),automatic_retry=False))
    if code or detected:raise RuntimeError(f'{stage} did not complete cleanly: exit{code}, failure{detected}')
    if formal_runtime and json.loads((ROOT/formal_runtime/'terminal.json').read_text()).get('complete') is not True:
        raise RuntimeError('formal owner lacks complete full grid')


def await_initial_status(child,runtime):
    while not (ROOT/runtime/'status.json').exists():
        guard()
        if child.poll() is not None:raise RuntimeError('environment-resumed owner exited before first status')
        pulse('await_environment_VAL_status',child_pid=child.pid);time.sleep(2.)


def main():
    global ENVIRONMENT_RECEIPT
    guard()
    # Share the previous controller's lock as well as this new one-time lock.
    original_lock=(ROOT/'runtime/framework/ordered_resume_VAL_TEST_v775.lock').open('a+')
    fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(ROOT/'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
    if BASE.exists():raise FileExistsError('preserve this one-time explicit resume controller')
    original='artifacts/framework/multitask_r10/formal_suite16_v752_setup'
    control_path=SETUP+'/resume_control.json';resume_lock=SETUP+'/resume_inputs.json'
    control,files=load_resume_control(control_path,resume_lock,original+'/suite.json',original+'/frozen_inputs.json',original+'/validation_source_inputs.json')
    ENVIRONMENT_RECEIPT=owner.ENVIRONMENT_RECEIPT
    if files.get(str(Path(__file__).resolve()))!=file_sha(__file__):raise ValueError('freeze actual resumed controller before execution')
    previous=json.loads((ROOT/control['previous_launch']['path']).read_text())
    if alive(previous):raise RuntimeError('previous owner has not exited; no overlapping scheduling')
    inventory=owner.process_visibility()
    if not inventory['host_visibility_available'] or inventory['live_project_native_processes'] or inventory['same_uid_inventory_incomplete']:
        raise RuntimeError('complete host native/controller inventory is required before new scheduling')
    BASE.mkdir();atomic_json(BASE/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        source_sha256=file_sha(__file__),resume_control=pointer(ROOT/control_path),resume_control_lock=pointer(ROOT/resume_lock),
        frozen_environment=ENVIRONMENT_RECEIPT,user_authorized_resume=True,inventory=inventory))
    try:
        common=['--suite',original+'/suite.json','--suite-lock',original+'/frozen_inputs.json','--source-inputs',original+'/validation_source_inputs.json',
                '--split','validation','--runtime',VAL_RUNTIME,'--resume-control',control_path,'--resume-lock',resume_lock]
        # Reporter822 is independently owned by root and is never launched or killed here.
        native,leaf=launch('environment_resume_VAL_945','scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py',*common,'--out',ART+'/validation')
        await_initial_status(native,VAL_RUNTIME)
        detected=None
        while native.poll() is None:
            guard();state=json.loads((ROOT/VAL_RUNTIME/'status.json').read_text())
            if state.get('failed') and detected is None:
                detected=state['failed'];drain=ROOT/VAL_RUNTIME/'DRAIN'
                if not drain.exists():atomic_json(drain,dict(created_utc=utc(),reason='concrete resumed VAL infrastructure failure'))
            pulse('formal_VAL_945',child_pid=native.pid,infrastructure_failure_detected=detected);time.sleep(2.)
        code=native.wait();atomic_json(leaf/'terminal.json',dict(complete=code==0,returncode=code,finished_utc=utc(),automatic_retry=False))
        terminal=json.loads((ROOT/VAL_RUNTIME/'terminal.json').read_text())
        if code or detected or terminal.get('complete') is not True:raise RuntimeError('resumed VAL requires concrete repair; never retry automatically')
        release=ART+'/VAL_release.json'
        run('independent_resumed_VAL_release','scripts/audit_environment_resumed_formal_VAL_v824_r10.py',*common,'--out',release)
        run('prepare_TEST_after_complete_VAL','scripts/prepare_formal_TEST_after_environment_resume_v826_r10.py','--validation-release',release)
        test='artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup'
        run('lock_original_TEST_sources','scripts/prepare_formal_split_source_lock_v756_r10.py','--suite',test+'/suite.json','--suite-lock',test+'/frozen_inputs.json',
            '--split','test','--validation-release',release,'--out',test+'/test_source_inputs.json')
        common=['--suite',test+'/suite.json','--suite-lock',test+'/frozen_inputs.json','--source-inputs',test+'/test_source_inputs.json',
                '--split','test','--validation-release',release,'--runtime',TEST_RUNTIME]
        run('formal_TEST_1920','scripts/run_formal_suite16_owner_v753_r10.py',*common,'--out',ART+'/test',formal_runtime=TEST_RUNTIME)
        run('independent_TEST_results','scripts/audit_formal_suite16_results_v754_r10.py',*common,'--out',ART+'/TEST_results.json')
        atomic_json(BASE/'terminal.json',dict(complete=True,finished_utc=utc(),TEST_results=ART+'/TEST_results.json',automatic_retry=False));pulse('complete')
    except BaseException as error:
        user_stop=isinstance(error,InterruptedError)
        if not user_stop:
            for name in (VAL_RUNTIME,TEST_RUNTIME):
                runtime=ROOT/name
                if (runtime/'launch.json').exists() and alive(json.loads((runtime/'launch.json').read_text())):
                    drain=runtime/'DRAIN'
                    if not drain.exists():atomic_json(drain,dict(created_utc=utc(),reason='concrete resume failure; no new work; naturally drain'))
            while any((ROOT/name/'launch.json').exists() and alive(json.loads((ROOT/name/'launch.json').read_text())) for name in (VAL_RUNTIME,TEST_RUNTIME)):
                if stopped():user_stop=True;break
                pulse('draining_for_concrete_repair',error=repr(error));time.sleep(2.)
        for child,_,_ in CHILDREN:
            if child.poll() is None:interrupt_owned_tree(child)
        atomic_json(BASE/'failure.json',dict(complete=False,user_stop=user_stop,finished_utc=utc(),error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        pulse('user_stopped' if user_stop else 'requires_concrete_repair',error=repr(error));raise
    finally:
        for _,log,err in CHILDREN:log.close();err.close()


if __name__=='__main__':main()
