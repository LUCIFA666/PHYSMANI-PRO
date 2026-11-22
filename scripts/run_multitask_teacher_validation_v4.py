"""Authorized fixed three-source TRAIN check before the full immutable v4 queue."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_multitask_training_r10 import (verify_plan,require_r09_release,case_directory,
    validate_case_result,write_json,sha)

BASE=ROOT/'artifacts/framework/multitask_r10'
PLAN=BASE/'TRAIN_smoke_collection_plan_v4.json'
OUT=BASE/'TRAIN_smoke_collection_v4'
TASKS=('place_cups_on_rotating_frame','insert_onto_rotating_peg','put_rubbish_in_moving_bin')


def main():
    plan=verify_plan(PLAN);gate=require_r09_release(ROOT/'artifacts/framework/aligned_r09/final_report.json')
    index=json.loads((ROOT/'artifacts/framework/aligned_r09/demo_index.json').read_text())
    assert index.get('complete') is True and index.get('expected')==index.get('completed')==10 and len(index.get('entries',[]))==10
    cases=[next(case for case in plan['cases'] if case['task']==task) for task in TASKS]
    launcher=ROOT/'scripts/run_local_sim.sh';source_sha=sha(__file__)
    def one(case):
        verify_plan(PLAN)
        canonical=case_directory(OUT,case)/'case.json'
        if canonical.exists():
            record=validate_case_result(canonical,plan,case)
            return dict(task=case['task'],case_id=case['case_id'],status='resumed_verified',process_id=record['process_id'])
        attempts=OUT/'infrastructure_attempts'/case['case_id'];number=0
        while (attempts/str(number)).exists():number+=1
        attempt=attempts/str(number);attempt.mkdir(parents=True)
        child_out=attempt/'output';log=attempt/'native.log'
        command=['bash',str(launcher),str(ROOT/'scripts/collect_multitask_training_r10.py'),
            '--collect-case','--plan',str(PLAN),'--case-id',case['case_id'],'--output',str(child_out),
            '--guard-report',str(ROOT/'artifacts/framework/aligned_r09/final_report.json')]
        started=datetime.now(timezone.utc).isoformat()
        with log.open('xb') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
            try:code=process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=None
        raw=case_directory(child_out,case)/'case.json';status='infrastructure_error';detail=''
        if code==0 and raw.exists():
            record=validate_case_result(raw,plan,case);verify_plan(PLAN)
            record['parent_process_receipt']=dict(launcher=str(launcher.relative_to(ROOT)),launcher_sha256=sha(launcher),
                process_id=process.pid,command=command,child_case_path=str(raw.relative_to(ROOT)),child_case_sha256=sha(raw),
                parent_source=str(Path(__file__).relative_to(ROOT)),parent_source_sha256=source_sha,
                parent_process_id=os.getpid(),subset_fixed_tasks=list(TASKS),plan_sha256=plan['plan_sha256'])
            write_json(canonical,record);status='completed'
        else:detail='fresh TRAIN process failed; no policy-failure label and no automatic retry'
        receipt=dict(task=case['task'],case_id=case['case_id'],status=status,returncode=code,detail=detail,
            process_id=process.pid,command=command,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
            parent_source_sha256=source_sha,plan_sha256=plan['plan_sha256'],policy_failure_label_assigned=False,gate=gate)
        write_json(attempt/'parent_receipt.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False),flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(one,cases))
    report=dict(version='r10_fixed_three_TRAIN_teacher_validation_v4',plan_sha256=plan['plan_sha256'],
        tasks=list(TASKS),expected=3,completed=sum(row['status'] in ('completed','resumed_verified') for row in receipts),
        complete=all(row['status'] in ('completed','resumed_verified') for row in receipts),receipts=receipts,
        parent_process_id=os.getpid(),parent_source_sha256=source_sha,GPU_used=False,
        success_used_for_retry=False,automatic_retries=False)
    write_json(OUT/'three_case_validation_report.json',report)
    if not report['complete']:raise SystemExit(2)


if __name__=='__main__':main()
