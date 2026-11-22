"""Capture the fixed eight V10 calibration parents, serially and once.

Default performs preparation checks only. Execution requires a final plan,
its external SHA, a pinned method profile and fresh output paths. No model or
calibration quality evaluation is run by this queue.
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
FAMILY = ROOT/'runtime/framework/near_goal_perception_20261103'
HOST = ROOT/'scripts/launch_near_goal_position_v10_calibration8_20261103.py'
EXPECTED = tuple('near_goal_perception_position_v10_seed'+str(i) for i in range(261103656,261103664))
SHOTS = tuple(f'shot{i:02d}' for i in range(10))


def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def verify_plan(path, digest=None):
    if digest is not None and sha(path)!=digest: raise ValueError('Fixed V10 plan changed')
    plan=load(path)
    if (plan.get('scope')!='position_calibration8_v10'
            or tuple(p['parent_id'] for p in plan['parents'])!=EXPECTED
            or any(tuple(s['shot_id'] for s in p['shots'])!=SHOTS for p in plan['parents'])):
        raise ValueError('Complete fixed eight-parent/80-shot assignment required')
    for name,value in plan['source_sha256'].items():
        if sha(ROOT/name)!=value: raise ValueError('Frozen capture source changed: '+name)
    for name,value in plan['external_source_sha256'].items():
        if sha(name)!=value: raise ValueError('Frozen external capture source changed: '+name)
    profile=plan['future_position_profile'];p=Path(profile['path'])
    p=p if p.is_absolute() else ROOT/p
    if sha(p)!=profile['sha256']: raise ValueError('Predeclared V10 method profile changed')
    return plan


def stopped(root, ident=None):
    paths=[root/'STOP',FAMILY/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP']
    if ident: paths.extend((root/ident/'STOP',root/(ident+'.STOP')))
    return any(p.exists() for p in paths)


def audit_closed(root, parent, plan, plan_sha):
    ident=parent['parent_id'];case=root/ident
    owner=load(root/(ident+'.owner.actual_exit.json'))
    result=load(case/'result.json');run=load(case/'run_inputs.json')
    if (type(owner.get('actual_process_returncode')) is not int or owner['actual_process_returncode']!=0
            or owner.get('actual_wait_completed') is not True or owner.get('stop_reason') is not None
            or owner.get('source_unchanged') is not True or owner.get('parent_id')!=ident
            or Path(owner.get('output','')).resolve()!=case.resolve()
            or any(result.get(k) is not True for k in ('complete','native_closed','source_unchanged'))
            or result.get('parent_id')!=ident or type(result.get('controlled_physics_ticks')) is not int
            or result['controlled_physics_ticks']!=0 or run.get('parent_id')!=ident
            or run.get('plan_sha256')!=plan_sha or run.get('source_sha256')!=plan['source_sha256']
            or run.get('external_source_sha256')!=plan['external_source_sha256']):
        raise ValueError('Actual clean closed0/zero-tick/source/parent identity required')
    frames=result.get('shots',[])
    if tuple(f.get('shot_id') for f in frames)!=SHOTS:
        raise ValueError('All ten ordered shots required; never deduplicate')
    for actual,planned in zip(frames,parent['shots']):
        if (actual.get('scene_id')!=ident+'/'+planned['shot_id']
                or actual.get('condition')!=planned.get('condition')
                or actual.get('condition_detail')!=planned.get('condition_detail')
                or actual.get('position_band')!=planned.get('position_band')
                or actual.get('independent_scene') is not True
                or type(actual.get('physics_ticks')) is not int or actual['physics_ticks']!=0):
            raise ValueError('Exact planned scene/band/zero-tick metadata required')
    return dict(parent_id=ident,actual_process_returncode=0,native_closed=True,shots=10,
        source_unchanged=True,model_forward_calls=0,no_source_or_snapshot_replacement=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    plan_path=args.plan.resolve(strict=True);plan=load(plan_path)
    if args.execute and plan.get('preparation_only') is not False:
        parser.error('Preparation-only plan cannot create output or start Native')
    if args.execute and not args.plan_sha256: parser.error('Execution requires external fixed plan SHA')
    plan=verify_plan(plan_path,args.plan_sha256)
    from our_mani.local_mpc_20261103.depth_owner_position_v10_calibration8 import check_capture_plan
    checked=check_capture_plan(ROOT,plan_path)
    root=args.output_root.resolve()
    if FAMILY.resolve() not in root.parents: parser.error('New perception runtime output root required')
    if root.exists(): raise FileExistsError('Fresh whole cohort only; no resumption or retry')
    if stopped(root): raise ValueError('Administrative STOP is preserved')
    if not args.execute:
        print(json.dumps(dict(complete=True,Native_started=False,model_forward_calls=0,
            parent_count=8,shots=80,preparation=checked,output_root=str(root)),allow_nan=False));return 0
    if checked.get('capture_ready') is not True: parser.error('Complete final capture admission required')
    root.mkdir(parents=True,exist_ok=False);ledger=root/'serial_queue_v1.jsonl';plan_sha=sha(plan_path)
    append(ledger,dict(event='start',utc=now(),fixed_order=list(EXPECTED),plan_sha256=plan_sha,
        queue_source_sha256=sha(__file__),model_forward_calls=0))
    try:
        for parent in plan['parents']:
            ident=parent['parent_id']
            if stopped(root,ident): raise ValueError('Administrative STOP prevents next parent')
            verify_plan(plan_path,plan_sha)
            command=[sys.executable,'-B',str(HOST),'--plan',str(plan_path),'--parent-id',ident,
                '--output',str(root/ident),'--execute']
            completed=subprocess.run(command,cwd=ROOT,check=False)
            if completed.returncode!=0: raise RuntimeError('Capture infrastructure failure; no retry: '+ident)
            row=audit_closed(root,parent,plan,plan_sha);verify_plan(plan_path,plan_sha)
            append(ledger,dict(event='actual_closed',utc=now(),**row))
        append(ledger,dict(event='complete',utc=now(),all8_actual_closed=True,snapshots=80,
            model_forward_calls=0,no_source_or_snapshot_replacement=True));return 0
    except BaseException as error:
        append(ledger,dict(event='infrastructure_stop',utc=now(),error_type=type(error).__name__,
            error=str(error),no_retry=True));raise


if __name__=='__main__': raise SystemExit(main())
