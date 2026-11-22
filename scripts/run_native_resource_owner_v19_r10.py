"""Single owner for two native TRAIN slots and one GPU slot.

Append-only requests are emitted by the primary agent after source preflight.
This owner never retries a request and never promotes a task to VAL or TEST.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair,process_visibility

BASE=ROOT/'runtime/framework/native_resource_owner_v19'
REQUESTS=BASE/'requests'
STOP=BASE/'STOP'
GLOBAL_STOP=ROOT/'runtime/framework/multitask_r10_STOP'
METHODS=('cv_geometry','context_geometry','physmani')


def utc():return datetime.now(timezone.utc).isoformat()


def validate_request(value):
    if set(value)!={'schema','job_id','kind','argv','use_gpu','timeout_seconds','source_sha256'}:
        raise ValueError('Exact source-bound TRAIN request fields required')
    if value['schema']!='r10_owned_native_TRAIN_request_v19' or not value['job_id'].replace('_','').isalnum():
        raise ValueError('Invalid TRAIN request identity')
    argv=value['argv']
    if not isinstance(argv,list) or not all(isinstance(x,str) for x in argv):raise ValueError('Explicit argv required')
    if value['kind']=='high_cup_collection':
        if len(argv)<3 or Path(argv[1]).name!='collect_TRAIN_high_cup_periodic_replan_r10.py' or '--run-queue' not in argv:
            raise ValueError('Only the reviewed high-cup TRAIN collector is allowed')
        if '--workers' not in argv or argv[argv.index('--workers')+1]!='1' or value['use_gpu'] is not False:
            raise ValueError('Each collection request owns exactly one CPU native slot')
        script=ROOT/argv[1]
    elif value['kind']=='executor_smoke':
        if len(argv)<4 or argv[0]!='bash' or argv[1] not in ('scripts/run_local_sim.sh','scripts/run_aligned_gpu.sh'):
            raise ValueError('Reviewed native launcher required')
        script=ROOT/argv[2]
        if not script.name.startswith(('run_geometry_','run_task_executor_smoke_','run_bin_baseline_TRAIN_')):
            raise ValueError('Only TRAIN executor smoke workers are allowed')
        if '--split' not in argv or argv[argv.index('--split')+1]!='train' or '--development-demo' not in argv:
            raise ValueError('VAL/TEST cannot enter the TRAIN resource owner')
        if '--method' not in argv or argv[argv.index('--method')+1] not in METHODS:
            raise ValueError('Only the three authorized methods are allowed')
        gpu=argv[argv.index('--method')+1]=='physmani'
        if value['use_gpu'] is not gpu or (argv[1]=='scripts/run_aligned_gpu.sh') is not gpu:
            raise ValueError('GPU allocation must match the actual method/launcher')
    else:raise ValueError('Unknown TRAIN request kind')
    if not script.resolve().is_relative_to(ROOT) or file_sha(script)!=value['source_sha256']:
        raise ValueError('Actual requested worker source changed')
    if type(value['timeout_seconds']) not in (int,float) or not 0<value['timeout_seconds']<=14400:
        raise ValueError('Explicit bounded wall timeout required')
    return value


def process_tree(parent):
    rows={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(p.name)]=(int(stat[1]),int(stat[19]))
        except (OSError,ValueError,IndexError):pass
    descendants={parent};changed=True
    while changed:
        before=len(descendants)
        descendants.update(pid for pid,(ppid,_) in rows.items() if ppid in descendants)
        changed=len(descendants)!=before
    return {pid:rows[pid][1] for pid in descendants if pid in rows}


def interrupt_owned_tree(process):
    owned=process_tree(process.pid)
    for sig,delay in ((signal.SIGINT,5),(signal.SIGTERM,5),(signal.SIGKILL,0)):
        for pid,ticks in reversed(list(owned.items())):
            try:
                current=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
                if current==ticks:os.kill(pid,sig)
            except (OSError,ValueError,IndexError):pass
        if process.poll() is not None:break
        if delay:
            try:process.wait(timeout=delay)
            except subprocess.TimeoutExpired:pass


def main():
    BASE.mkdir(exist_ok=True);REQUESTS.mkdir(exist_ok=True)
    if (BASE/'launch.json').exists():raise RuntimeError('Preserve previous resource owner; no automatic restart')
    inventory=process_visibility()
    if not inventory['host_visibility_available'] or inventory['live_project_native_processes'] or inventory['same_uid_inventory_incomplete']:
        raise RuntimeError('Existing native workers or incomplete host process view')
    cancelled=threading.Event();active={};finished={};seen=set()

    def one(path,value):
        leaf=BASE/'jobs'/value['job_id'];leaf.mkdir(parents=True,exist_ok=False)
        started=time.monotonic();process=None;interrupted=False;timeout=False
        try:
            validate_request(value)
            if cancelled.is_set() or STOP.exists() or GLOBAL_STOP.exists():raise RuntimeError('Controlled stop before launch')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            if not value['use_gpu']:env['CUDA_VISIBLE_DEVICES']=''
            with (leaf/'stdout.log').open('xb') as log:
                process=subprocess.Popen(value['argv'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
                atomic_json(leaf/'launch.json',dict(created_utc=utc(),request=value,request_sha256=file_sha(path),
                    launcher_pid=process.pid,launcher_start_ticks=ticks,parent_pid=os.getpid(),native_slots=1,GPU_slots=int(value['use_gpu'])))
                while process.poll() is None:
                    timeout=time.monotonic()-started>value['timeout_seconds']
                    if cancelled.is_set() or STOP.exists() or GLOBAL_STOP.exists() or timeout:
                        interrupted=not timeout;interrupt_owned_tree(process);break
                    cancelled.wait(.5)
                code=process.wait()
            row=dict(complete=code==0 and not interrupted and not timeout,returncode=code,
                interrupted=interrupted,timed_out=timeout,automatic_retry=False,policy_failure_label_fabricated=False,
                wall_seconds=time.monotonic()-started,finished_utc=utc(),stdout_sha256=file_sha(leaf/'stdout.log'))
        except BaseException as error:
            if process is not None and process.poll() is None:interrupt_owned_tree(process)
            row=dict(complete=False,automatic_retry=False,error=repr(error),traceback=traceback.format_exc(),
                policy_failure_label_fabricated=False,finished_utc=utc())
        atomic_json(leaf/'terminal.json',row)
        return row

    with exclusive_pair():
        atomic_json(BASE/'launch.json',dict(created_utc=utc(),pid=os.getpid(),source_sha256=file_sha(__file__),
            max_native_slots=2,max_GPU_slots=1,automatic_retry=False,formal_VAL_TEST_enabled=False,inventory=inventory))
        with ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                if STOP.exists() or GLOBAL_STOP.exists():cancelled.set()
                for job_id,(future,value) in list(active.items()):
                    if future.done():
                        finished[job_id]=future.result();del active[job_id]
                if not cancelled.is_set():
                    for path in sorted(REQUESTS.glob('*.json')):
                        if path.name in seen or len(active)>=2:continue
                        value=validate_request(json.loads(path.read_text()))
                        if value['use_gpu'] and any(v['use_gpu'] for _,v in active.values()):continue
                        if value['job_id'] in active or value['job_id'] in finished:raise RuntimeError('Duplicate owned job id')
                        seen.add(path.name);active[value['job_id']]=(pool.submit(one,path,value),value)
                atomic_json(BASE/'status.json',dict(updated_utc=utc(),active_jobs=list(active),completed=finished,
                    stop_requested=cancelled.is_set(),max_native_slots=2,allocated_native_slots=len(active),
                    allocated_GPU_slots=sum(v['use_gpu'] for _,v in active.values()),automatic_retry=False),overwrite=True)
                if cancelled.is_set() and not active:break
                cancelled.wait(2.)


if __name__=='__main__':main()
