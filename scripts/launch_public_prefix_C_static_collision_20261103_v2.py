"""Independent PUBLIC-prefix static-collision replay owner; no Native without explicit complete admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILY = ROOT/'runtime/framework/local_mpc_20261102'
SHELL = ROOT/'scripts/run_public_prefix_C_static_collision_20261103_v2.sh'
WORKER = ROOT/'scripts/replay_public_prefix_C_static_collision_20261103_v2.py'


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--anchor-id',required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source-freeze',type=Path)
    parser.add_argument('--source-freeze-sha256')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout-seconds',type=float,default=600.)
    parser.add_argument('--stop-grace-seconds',type=float,default=20.)
    args=parser.parse_args(argv)
    if not 0<args.timeout_seconds<=1800 or not 0<args.stop_grace_seconds<=60:
        parser.error('bounded positive timeout and stop grace required')
    from our_mani.local_mpc_20261103.public_prefix_static_collision_replay_20261103_v2 import check
    admission=check(args.request,args.anchor_id,source_freeze=args.source_freeze,expected_source_freeze_sha256=args.source_freeze_sha256)
    if not args.execute:
        print(json.dumps(admission,allow_nan=False),flush=True);return 0
    if args.output is None:parser.error('--execute requires --output')
    output=args.output.resolve();request_path=args.request.resolve()
    if not admission['ready']:parser.error('Exact replay source freeze is required')
    if FAMILY.resolve() not in output.parents or ROOT not in request_path.parents:
        parser.error('fresh local MPC runtime case and existing ROOT config required')
    if output.exists():parser.error('fresh case only; no resumption')
    prefix=output.parent/(output.name+'.owner')
    log_path=Path(str(prefix)+'.log');input_path=Path(str(prefix)+'.json')
    exit_path=Path(str(prefix)+'.actual_exit.json')
    if any(p.exists() for p in (log_path,input_path,exit_path)):
        parser.error('fresh owner identity required')
    stop_paths=(output/'STOP',output.parent/(output.name+'.STOP'),output.parent/'STOP',FAMILY/'STOP')
    if any(p.exists() for p in stop_paths):parser.error('administrative STOP is preserved')
    command=['bash',str(SHELL),'--request',str(request_path),'--anchor-id',args.anchor_id,'--output',str(output),
             '--source-freeze',str(args.source_freeze.resolve()),
             '--source-freeze-sha256',args.source_freeze_sha256,'--execute']
    paths=(Path(__file__).resolve(),SHELL,WORKER,request_path,args.source_freeze.resolve())
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    # Everything above, including the real certificate, is checked before any
    # mkdir, owner creation, Popen or Native lock acquisition.
    output.parent.mkdir(parents=True,exist_ok=True)
    request={'signal':None};handlers={}
    def stop(signum,frame):request['signal']=signum
    for signum in (signal.SIGINT,signal.SIGTERM):handlers[signum]=signal.signal(signum,stop)
    started=utc();stop_reason=stop_at=term_at=None;killed=False
    process=None;owner_error=None
    try:
        with log_path.open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_new(input_path,dict(started_utc=started,child_pid=process.pid,process_group=process.pid,
                anchor_id=args.anchor_id,argv=command,source_sha256=source_hashes,
                source_freeze=admission['source_freeze'],C_command_executed=False,
                collision_qualified_before_query=False,old_runs_resumed=False,
                development_only=True,TEST_read=False))
            print(json.dumps(dict(event='started',child_pid=process.pid,log=str(log_path))),flush=True)
            begin=time.monotonic()
            while process.poll() is None:
                now=time.monotonic()
                if stop_reason is None:
                    if request['signal'] is not None:stop_reason='owner_signal_'+str(request['signal'])
                    elif any(p.exists() for p in stop_paths):stop_reason='administrative_STOP'
                    elif now-begin>args.timeout_seconds:stop_reason='case_wall_timeout'
                    if stop_reason is not None:
                        stop_at=now
                        if not stop_paths[1].exists():stop_paths[1].write_text(stop_reason+'\n')
                if stop_at is not None and term_at is None and now-stop_at>=args.stop_grace_seconds:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    term_at=now
                if stop_at is not None and output.is_dir() and not (output/'STOP').exists():
                    (output/'STOP').write_text(stop_reason+'\n')
                if term_at is not None and not killed and now-term_at>=5.:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    killed=True
                time.sleep(.2)
            code=process.wait()
        unchanged=all(sha(ROOT/p)==digest for p,digest in source_hashes.items())
        write_new(exit_path,dict(started_utc=started,finished_utc=utc(),actual_process_returncode=code,
            output=str(output),anchor_id=args.anchor_id,stop_reason=stop_reason,
            SIGTERM_sent=term_at is not None,SIGKILL_sent=killed,source_unchanged=unchanged,
            actual_wait_completed=True,benchmark_success_claimed=False))
        print(json.dumps(dict(event='finished',actual_process_returncode=code,output=str(output))),flush=True)
        return code if code else (2 if stop_reason or not unchanged else 0)
    except BaseException as error:
        owner_error=dict(type=type(error).__name__,message=str(error));raise
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
            write_new(exit_path,dict(started_utc=started,finished_utc=utc(),actual_process_returncode=code,
                output=str(output),anchor_id=args.anchor_id,stop_reason=stop_reason,owner_error=owner_error,
                terminated_during_owner_cleanup=True,source_unchanged=False,actual_wait_completed=True,
                benchmark_success_claimed=False))
        for signum,handler in handlers.items():signal.signal(signum,handler)


if __name__=='__main__':raise SystemExit(main())
