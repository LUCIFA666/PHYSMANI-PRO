"""Select whole predeclared tasks from a full275 plan, then exec old queue.

This process does not replace the original projected backend or raw collector.
The old queue's worker argv, process receipts, JIT/import/physical gates remain
the actual execution path. Selection never reads results or policy outcomes.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import digest,file_sha,atomic_json


def selected_command(plan_path,tasks,workers,*,python_executable=None):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    assert plan_path.is_relative_to(ROOT)
    assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
    assert plan['split']=='validation' and plan['methods']==['physmani'] and plan['replicas']==1
    assert plan['expected_cases']==plan['source_episodes']==len(plan['jobs'])==275
    assert plan['official_TEST_used_for_fitting'] is False
    assert tasks and len(tasks)==len(set(tasks)) and type(workers) is int and workers>=1
    known={job['row']['task'] for job in plan['jobs']}
    assert set(tasks)<=known
    jobs=[job for job in plan['jobs'] if job['row']['task'] in set(tasks)]
    assert jobs and all(job['replica']==0 and job['record_demo'] is False and job['exploration_seed'] is None for job in jobs)
    argv=[str(Path(python_executable or sys.executable).resolve()),
        str(ROOT/'scripts/evaluate_physmani_projected_VAL_r10.py'),'--run-queue','--plan',str(plan_path),
        '--workers',str(workers)]
    for job in jobs:argv+=['--queue-job-id',job['job_id']]
    return argv,jobs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--task',action='append',required=True)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--selection-receipt',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();path=(ROOT/args.plan).resolve()
    argv,jobs=selected_command(path,args.task,args.workers)
    receipt=(ROOT/args.selection_receipt).resolve();assert receipt.is_relative_to(ROOT) and not receipt.exists()
    atomic_json(receipt,dict(schema='r10_original_full275_PhysMani_predeclared_task_selection_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),source=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)),
        actual_full275_plan=dict(path=str(path),sha256=file_sha(path)),tasks=args.task,expected275=275,
        selected_expected=len(jobs),selected_job_ids=[job['job_id'] for job in jobs],
        selected_original_source_rows=[job['row'] for job in jobs],actual_exec_argv=argv,
        outcomes_read_for_selection=False,old12_reused_or_relabelled=False,TEST_started=False,
        metadata_check_only=args.check))
    if args.check:
        print(json.dumps(dict(selected_expected=len(jobs),tasks=args.task,argv=argv)));return
    os.execve(argv[0],argv,os.environ.copy())


if __name__=='__main__':main()
