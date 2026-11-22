"""Wait for the closed Native collection receipt, then own the complete pipeline."""
import subprocess
import time
from .common import ROOT,OUT,NATIVE_PYTHON,plan,read,write,status,stopped,utc


def main():
    p=plan();status('TRAIN_collection',complete=False,planned_episodes=16)
    write(OUT/'continuation_request.json',dict(started_utc=utc(),wait_for='collection_v1/actual_exit.json',old_queues_resumed=False))
    start=time.monotonic()
    while not(OUT/'collection_v1/actual_exit.json').exists():
        stopped(p)
        if time.monotonic()-start>14400:raise TimeoutError('Native collection receipt unavailable; no training started')
        time.sleep(1.)
    receipt=read(OUT/'collection_v1/actual_exit.json')
    if not receipt['complete']:raise RuntimeError('Native collection failed; no training started')
    stopped(p)
    command=[NATIVE_PYTHON,'-B','-m','our_mani.cube_training_cycle_20261111_v1.pipeline','--execute']
    with(OUT/'pipeline_stdout.log').open('x')as f:
        process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        code=process.wait()
    write(OUT/'continuation_actual_exit.json',dict(complete=code==0,actual_process_returncode=code,actual_wait_completed=True,finished_utc=utc()))
    return code

if __name__=='__main__':raise SystemExit(main())
