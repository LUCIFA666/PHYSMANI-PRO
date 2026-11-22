"""Two-worker isolated TRAIN orientation pilot; no policy metrics or retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import pilot_orientation_registration_r10 as pilot
from audit_train_resets_r10 import require_completed_delivery

SOURCE_SHA=pilot.sha(__file__)


def write(path,value):
    pilot.write(path,value)


def child(plan_path,task,out):
    plan=pilot.verify_plan(plan_path);gate=require_completed_delivery()
    out=Path(out).resolve();receipts=out.parent/'process_receipts'
    started=dict(task=task,native_child_pid=os.getpid(),plan_sha256=plan['plan_sha256'],
        wrapper_sha256=SOURCE_SHA,pilot_source_sha256=pilot.sha(pilot.__file__),
        GPU_disabled=os.environ.get('CUDA_VISIBLE_DEVICES')=='',teacher_used=False,
        formal_metrics_included=False,argv=sys.argv,release_gate=gate)
    pilot.require(started['GPU_disabled'],'native pilot requires disabled CUDA')
    write(receipts/(task+'.started.json'),started)
    try:
        result=pilot.run_case(plan_path,task,out)
        write(receipts/(task+'.native_receipt.json'),dict(started,completed=True,
            report_sha256=pilot.sha(out/'report.json'),sensor_observations_sha256=pilot.sha(out/'sensor_observations.npz'),
            actual_frames=result['actual_frames'],counted_hold_ticks=result['counted_hold_ticks']))
    except Exception as error:
        write(receipts/(task+'.error.json'),dict(started,completed=False,infrastructure_error=repr(error),
            traceback=traceback.format_exc(),counted_as_policy_failure=False))
        raise


def run_one(plan_path,row,out):
    pilot.verify_plan(plan_path);require_completed_delivery()
    pilot.require(pilot.sha(__file__)==SOURCE_SHA,'running wrapper source changed')
    pilot.require(not (ROOT/'runtime/framework/multitask_r10_STOP').exists(),'controlled R10 stop before case')
    task=row['task'];case=out/task;log=out/'logs'/(task+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    pilot.require(not case.exists() and not log.exists(),'preserve prior/partial pilot evidence; choose new run output')
    command=['bash','scripts/run_local_sim.sh',str(Path(__file__).resolve().relative_to(ROOT)),
        '--child','--plan',str(plan_path),'--task',task,'--out',str(case)]
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='0',PYTHONFAULTHANDLER='1'),
            stdout=stream,stderr=subprocess.STDOUT)
        code=process.wait()
    started_path=out/'process_receipts'/(task+'.started.json')
    started=json.loads(started_path.read_text()) if started_path.exists() else None
    report_path=case/'report.json';receipt_path=out/'process_receipts'/(task+'.native_receipt.json')
    success=code==0 and report_path.is_file() and receipt_path.is_file()
    receipt=dict(task=task,completed=success,exit_code=code,command=command,launcher_pid=process.pid,
        native_child_pid=started.get('native_child_pid') if started else None,log_sha256=pilot.sha(log),
        wrapper_sha256=SOURCE_SHA,pilot_source_sha256=pilot.sha(pilot.__file__),
        plan_file_sha256=pilot.sha(plan_path),formal_metrics_included=False,counted_as_policy_failure=False)
    if success:
        result=json.loads(report_path.read_text());native=json.loads(receipt_path.read_text())
        pilot.require(result['task']==task and result['plan_sha256']==started['plan_sha256']
            and native['report_sha256']==pilot.sha(report_path)
            and native['sensor_observations_sha256']==pilot.sha(case/'sensor_observations.npz'),
            'native pilot output/identity changed')
        receipt.update(report=str(report_path.relative_to(ROOT)),report_sha256=pilot.sha(report_path),
            actual_frames=result['actual_frames'],counted_hold_ticks=result['counted_hold_ticks'])
    else:receipt['infrastructure_error']='native child failed or complete report/receipt missing; no retry and no policy label'
    write(out/'process_receipts'/(task+'.parent_receipt.json'),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--child',action='store_true')
    parser.add_argument('--task');args=parser.parse_args()
    plan_path=(ROOT/args.plan).resolve();out=(ROOT/args.out).resolve()
    pilot.require(out.is_relative_to(pilot.BASE.resolve()),'pilot output must stay on desktop R10 project')
    if args.child:
        pilot.require(args.task,'exact native child task required');child(plan_path,args.task,out);return
    pilot.require(args.workers in (1,2),'pilot concurrency at most two isolated native children')
    plan=pilot.verify_plan(plan_path);gate=require_completed_delivery()
    pilot.require(not out.exists(),'new pilot run output required');out.mkdir(parents=True)
    write(out/'execution_inputs.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_file_sha256=pilot.sha(plan_path),plan_sha256=plan['plan_sha256'],wrapper_sha256=SOURCE_SHA,
        pilot_source_sha256=pilot.sha(pilot.__file__),workers=args.workers,planned_cases=14,
        native_process_per_case=True,GPU_disabled=True,teacher_used=False,release_gate=gate,
        source_inputs_sha256=plan['inputs_sha256'],formal_metrics_included=False,automatic_annotation_promotion=False))
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,plan_path,row,out) for row in plan['cases']]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps(dict(task=result['task'],completed=result['completed'],finished=len(results),expected=14,
                exit_code=result['exit_code']),ensure_ascii=False),flush=True)
    pilot.verify_plan(plan_path);require_completed_delivery()
    write(out/'queue_report.json',dict(complete=True,expected=14,completed=sum(r['completed'] for r in results),
        infrastructure_cases=sum(not r['completed'] for r in results),results=results,
        distinct_native_processes=len({r['native_child_pid'] for r in results if r['native_child_pid']}),
        formal_metrics_included=False,thresholds_changed=False,annotation_changes_written=False,
        automatic_annotation_promotion=False,source_unchanged=True))


if __name__=='__main__':main()
