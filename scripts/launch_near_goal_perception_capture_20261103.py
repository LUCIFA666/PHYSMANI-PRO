"""Fresh, serial host owner for a perception-only Native capture parent.

Never resumes an old run. The shell owns Native locks; this process owns its
process group and records the actual wait result, including administrative
stops. Default command parsing does not start Native without all arguments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT/'runtime/framework/near_goal_perception_20261103'
SHELL = ROOT/'scripts/run_near_goal_perception_capture_20261103.sh'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--parent-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collector-script', choices=('collect_near_goal_perception_20261103.py',
        'collect_near_goal_depth_owner_20261103.py'),default='collect_near_goal_perception_20261103.py')
    parser.add_argument('--timeout-seconds', type=float, default=180.)
    parser.add_argument('--stop-grace-seconds', type=float, default=20.)
    args=parser.parse_args(argv)
    if (not 0<args.timeout_seconds<=1800 or not 0<args.stop_grace_seconds<=60):
        parser.error('bounded positive timeout and stop grace required')
    output=args.output.resolve();plan=args.plan.resolve()
    if FAMILY.resolve() not in output.parents or ROOT not in plan.parents or not plan.is_file():
        parser.error('fresh perception runtime case and existing plan required')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        parser.error('fresh case only; no resumption permitted')
    prefix=output.parent/(output.name+'.owner')
    log_path=Path(str(prefix)+'.log');input_path=Path(str(prefix)+'.json')
    exit_path=Path(str(prefix)+'.actual_exit.json')
    if any(p.exists() for p in (log_path,input_path,exit_path)):
        parser.error('fresh owner identity required')
    stop_paths=(output/'STOP',output.parent/(output.name+'.STOP'),FAMILY/'STOP')
    if any(p.exists() for p in stop_paths):
        parser.error('administrative STOP is preserved')
    command=['bash',str(SHELL),'--plan',str(plan),'--parent-id',args.parent_id,'--output',str(output)]
    worker_script=ROOT/'scripts'/args.collector_script
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),SHELL,worker_script,plan)}
    capture_environment=os.environ.copy()
    capture_environment['OUR_MANI_CAPTURE_WORKER']=str(worker_script.relative_to(ROOT))
    request={'signal':None};handlers={}
    def stop(signum,frame):
        request['signal']=signum
    for signum in (signal.SIGINT,signal.SIGTERM):
        handlers[signum]=signal.signal(signum,stop)
    started=utc();stop_reason=None;stop_at=None;term_at=None;killed=False
    process=None;owner_error=None
    try:
        with log_path.open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=capture_environment)
            write_new(input_path,dict(started_utc=started,child_pid=process.pid,process_group=process.pid,
                parent_id=args.parent_id,argv=command,source_sha256=source_hashes,
                collector_script=str(worker_script.relative_to(ROOT)),
                perception_only=True,benchmark_episode=False,old_runs_resumed=False))
            print(json.dumps(dict(event='started',child_pid=process.pid,log=str(log_path))),flush=True)
            begin=time.monotonic()
            while process.poll() is None:
                now=time.monotonic()
                if stop_reason is None:
                    if request['signal'] is not None:stop_reason='owner_signal_'+str(request['signal'])
                    elif any(p.exists() for p in stop_paths):stop_reason='administrative_STOP'
                    elif now-begin>args.timeout_seconds:stop_reason='parent_wall_timeout'
                    if stop_reason is not None:
                        stop_at=now
                        marker=stop_paths[1]
                        if not marker.exists():marker.write_text(stop_reason+'\n',encoding='utf-8')
                if stop_at is not None and term_at is None and now-stop_at>=args.stop_grace_seconds:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    term_at=now
                if term_at is not None and not killed and now-term_at>=5.:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    killed=True
                time.sleep(.2)
            code=process.wait()
        unchanged=all(sha(ROOT/p)==digest for p,digest in source_hashes.items())
        write_new(exit_path,dict(started_utc=started,finished_utc=utc(),actual_process_returncode=code,
            output=str(output),parent_id=args.parent_id,stop_reason=stop_reason,
            SIGTERM_sent=term_at is not None,SIGKILL_sent=killed,source_unchanged=unchanged,
            actual_wait_completed=True,benchmark_success_claimed=False))
        print(json.dumps(dict(event='finished',actual_process_returncode=code,output=str(output))),flush=True)
        return code if code else (2 if stop_reason or not unchanged else 0)
    except BaseException as error:
        owner_error=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
        if process is not None and not exit_path.exists():
            code=process.wait()
            write_new(exit_path,dict(started_utc=started,finished_utc=utc(),
                actual_process_returncode=code,output=str(output),parent_id=args.parent_id,
                stop_reason=stop_reason,owner_error=owner_error,
                terminated_during_owner_cleanup=True,source_unchanged=False,
                actual_wait_completed=True,benchmark_success_claimed=False,
                data_admitted_for_training=False))
        for signum,handler in handlers.items():signal.signal(signum,handler)


if __name__=='__main__':raise SystemExit(main())
