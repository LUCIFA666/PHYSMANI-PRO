"""Run fixed cases from a full TRAIN plan, then resume that unchanged plan.

Only infrastructure acceptance controls expansion. Native task success never
selects a source, replaces a failure, or grants a model/TEST release.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import collect_multitask_training_r10 as collection


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(plan_path, selection_path, output):
    plan_path, selection_path = Path(plan_path).resolve(), Path(selection_path).resolve()
    output = collection.destination(output)
    plan = collection.verify_plan(plan_path)
    selection = json.loads(selection_path.read_text())
    ids = selection['case_ids']
    known = {case['case_id'] for case in plan['cases']}
    if (len(plan['cases']) != 2800 or len(ids) != 12 or len(set(ids)) != 12 or set(ids)-known
            or plan['split'] != 'train'):
        raise ValueError('fixed twelve-case subset of the full 2800-case TRAIN plan required')
    identities = dict(plan=dict(path=str(plan_path),sha256=sha(plan_path),internal_sha256=plan['plan_sha256']),
        selection=dict(path=str(selection_path),sha256=sha(selection_path)),
        launcher=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    output.mkdir(parents=True,exist_ok=True)
    def status(stage, **fields):
        collection.write_json(output/'launch_status.json',dict(stage=stage,updated_utc=utc(),
            **identities,**fields),overwrite=True)
        p = collection.BASE/'status.json'
        value = json.loads(p.read_text())
        value.update(stage=stage,active_full_TRAIN_directory=str(output.relative_to(ROOT)),
            active_full_TRAIN_plan=dict(path=str(plan_path.relative_to(ROOT)),sha256=identities['plan']['sha256']),
            full_TRAIN_native_running=stage in ('full_source_TRAIN_fixed12_running','full_source_TRAIN_parallel8_running'),
            updated_utc=utc())
        value['TRAIN_v5_behavior_design']['actual_full_collection_started']=True
        value['TRAIN_v5_behavior_design']['actual_full_collection_version']=output.name
        temporary=p.with_suffix('.launch.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)
    base = [sys.executable,'-B',str(ROOT/'scripts/collect_multitask_training_r10.py'),
        '--run-queue','--plan',str(plan_path),'--output',str(output),'--workers','8']
    status('full_source_TRAIN_fixed12_running',expected_cases=2800,selected_cases=12)
    initial = base + [item for identity in ids for item in ('--queue-case-id',identity)]
    with (output/'initial12_queue.log').open('ab') as log:
        result = subprocess.run(initial,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status('full_source_TRAIN_fixed12_process_error',returncode=result.returncode)
        return 2
    report = json.loads((output/'queue_report.json').read_text())
    if (report.get('selected_complete') is not True or report.get('selected_expected')!=12
            or report.get('expected')!=2800 or set(r.get('case_id') for r in report['receipts'])!=set(ids)):
        status('full_source_TRAIN_fixed12_infrastructure_repair',automatic_retry=False)
        return 2
    collection.verify_plan(plan_path)
    for case in plan['cases']:
        if case['case_id'] in ids:
            collection.validate_case_result(collection.case_directory(output,case)/'case.json',plan,case)
    if collection.COLLECTION_STOP.exists():
        status('full_source_TRAIN_explicit_stop',automatic_retry=False)
        return 0
    status('full_source_TRAIN_parallel8_running',expected_cases=2800,fixed12_accepted=True,
        native_task_success_was_not_a_gate=True)
    with (output/'full_queue.log').open('ab') as log:
        result = subprocess.run(base,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    manifest = json.loads((output/'manifest.json').read_text()) if (output/'manifest.json').exists() else {}
    complete = result.returncode==0 and manifest.get('complete') is True and manifest.get('completed')==2800
    status('full_source_TRAIN_acquisition_complete' if complete else 'full_source_TRAIN_acquisition_pending',
        completed=manifest.get('completed',0),expected=2800,returncode=result.returncode,
        model_training_completed=False,official_TEST_evaluated=False,automatic_retry=False)
    return 0 if complete else 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.plan,args.selection,args.output))
