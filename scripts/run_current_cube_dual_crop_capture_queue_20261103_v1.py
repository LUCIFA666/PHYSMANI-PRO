"""Capture the fixed forty current-RGBD CF parents, serially and once.

Default performs preparation checks only. Execution requires a final plan,
its external SHA, the fixed metadata/source plan and fresh output paths. A fixed
closed first-fit prefix may be audited once before continuing the remaining39.
No model training or quality evaluation is run by this queue.
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
HOST = ROOT/'scripts/launch_current_cube_dual_crop_20261103_v2.py'
PREPARATION = ROOT/'artifacts/framework/local_mpc_20261102/current_RGBD_centre_TRAIN_data_audit_v1/dual_crop_capture_preparation_manifest_v2.json'
PREPARATION_SHA256 = 'befefa20fd2c0d26635729f38138ba4ba8b459c6c31ab3083f753f01cbf5e32c'
EXPECTED = tuple(json.loads(PREPARATION.read_text())['minimal_pilot_capture']['parent_ids'])
SHOTS = tuple(f'shot{i:02d}' for i in range(13))


def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def verify_plan(path, digest=None):
    if digest is not None and sha(path)!=digest: raise ValueError('Fixed dual-crop CF plan changed')
    plan=load(path)
    if (plan.get('scope')!='current_cube_dual_crop_CF_pilot40'
            or tuple(p['parent_id'] for p in plan['parents'])!=EXPECTED
            or any(tuple(s['shot_id'] for s in p['shots'])!=SHOTS for p in plan['parents'])):
        raise ValueError('Complete fixed forty-parent/520-shot assignment required')
    for name,value in plan['source_sha256'].items():
        if sha(ROOT/name)!=value: raise ValueError('Frozen capture source changed: '+name)
    for name,value in plan['external_source_sha256'].items():
        if sha(name)!=value: raise ValueError('Frozen external capture source changed: '+name)
    if sha(PREPARATION)!=PREPARATION_SHA256:
        raise ValueError('Original metadata-first pilot40 changed')
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
            or result.get('bootstrap_legal_Native_baseline_restored') is not True
            or run.get('plan_sha256')!=plan_sha or run.get('source_sha256')!=plan['source_sha256']
            or run.get('external_source_sha256')!=plan['external_source_sha256']
            or result.get('Native_sources_sha256_after')!=run.get('Native_sources_sha256')):
        raise ValueError('Actual clean closed0/zero-tick/source/parent identity required')
    frames=result.get('shots',[])
    if tuple(f.get('shot_id') for f in frames)!=SHOTS:
        raise ValueError('All thirteen ordered shots required; never deduplicate')
    for actual,planned in zip(frames,parent['shots']):
        if (actual.get('scene_id')!=ident+'/'+planned['shot_id']
                or actual.get('condition')!=planned.get('condition')
                or actual.get('condition_detail')!=planned.get('condition_detail')
                or actual.get('position_band')!=planned.get('position_band')
                or actual.get('independent_scene') is not True
                or type(actual.get('physics_ticks')) is not int or actual['physics_ticks']!=0):
            raise ValueError('Exact planned scene/band/zero-tick metadata required')
    archive=case/'source_snapshot';manifest=load(archive/'manifest.json');seen={}
    for item in manifest['files']:
        path=(archive/item['archive']).resolve(strict=True)
        if archive.resolve() not in path.parents or item['original'] in seen or sha(path)!=item['sha256']:
            raise ValueError('Original source byte archive missing/changed/duplicated')
        seen[item['original']]=item['sha256']
    expected={str((ROOT/name).resolve()):value for name,value in plan['source_sha256'].items()}
    expected.update(plan['external_source_sha256'])
    if any(seen.get(name)!=value for name,value in expected.items()) or sha(archive/'plan.json')!=plan_sha:
        raise ValueError('Whole fixed source graph and archived plan required')
    return dict(parent_id=ident,actual_process_returncode=0,native_closed=True,shots=13,
        source_unchanged=True,model_forward_calls=0,no_source_or_snapshot_replacement=True)


def validate_prefix(root,plan,plan_sha,start_index):
    if type(start_index) is not int or start_index not in (0,1):
        raise ValueError('Only fresh wholecohort or the fixed first-fit prefix is supported')
    if start_index==0:
        if root.exists(): raise FileExistsError('Fresh whole cohort only; no resumption or retry')
        return None
    if not root.is_dir() or (root/'serial_queue_v1.jsonl').exists():
        raise ValueError('Exactly one standalone first-fit capture must precede remaining39')
    first=plan['parents'][0]
    if first['parent_id']!=EXPECTED[0] or first['role']!='fit':
        raise ValueError('Only the preassigned first-fit parent may form the prefix')
    receipt=audit_closed(root,first,plan,plan_sha)
    for parent in plan['parents'][1:]:
        case=root/parent['parent_id'];prefix=case.with_name(case.name+'.owner')
        if case.exists() or any(Path(str(prefix)+suffix).exists() for suffix in ('.json','.log','.actual_exit.json')):
            raise FileExistsError('Remaining39 must all be fresh; no source replacement or retry')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--start-index',type=int,choices=(0,1),default=0,
        help='0 captures only first-fit13 for QA; 1 audits it then captures remaining39, never reruns it')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    plan_path=args.plan.resolve(strict=True);plan=load(plan_path)
    if args.execute and (plan.get('preparation_only') is not False or plan.get('execution_authorized') is not True):
        parser.error('Preparation-only plan cannot create output or start Native')
    if args.execute and not args.plan_sha256: parser.error('Execution requires external fixed plan SHA')
    plan=verify_plan(plan_path,args.plan_sha256)
    from our_mani.local_mpc_20261103.current_cube_dual_crop_capture_v2 import check_capture_plan
    checked=check_capture_plan(ROOT,plan_path)
    root=args.output_root.resolve()
    if FAMILY.resolve() not in root.parents: parser.error('New perception runtime output root required')
    if stopped(root): raise ValueError('Administrative STOP is preserved')
    plan_sha=sha(plan_path)
    prefix_receipt=validate_prefix(root,plan,plan_sha,args.start_index)
    if not args.execute:
        print(json.dumps(dict(complete=True,Native_started=False,model_forward_calls=0,
            parent_count=40,shots=520,preparation=checked,output_root=str(root),
            closed_prefix_count=args.start_index,stage_parent_count=1 if args.start_index==0 else 39,
            automatic_following_stage=False),allow_nan=False));return 0
    if checked.get('capture_ready') is not True: parser.error('Complete final capture admission required')
    if args.start_index==0:root.mkdir(parents=True,exist_ok=False)
    ledger=root/('first_fit_preview_queue_v1.jsonl' if args.start_index==0 else 'serial_queue_v1.jsonl')
    append(ledger,dict(event='start',utc=now(),fixed_order=list(EXPECTED),plan_sha256=plan_sha,
        queue_source_sha256=sha(__file__),model_forward_calls=0,closed_prefix_count=args.start_index,
        prefix_receipt=prefix_receipt,no_first_parent_reexecution=True))
    try:
        parents=plan['parents'][:1] if args.start_index==0 else plan['parents'][1:]
        for parent in parents:
            ident=parent['parent_id']
            if stopped(root,ident): raise ValueError('Administrative STOP prevents next parent')
            verify_plan(plan_path,plan_sha)
            command=[sys.executable,'-B',str(HOST),'--plan',str(plan_path),'--parent-id',ident,
                '--output',str(root/ident),'--execute']
            completed=subprocess.run(command,cwd=ROOT,check=False)
            if completed.returncode!=0: raise RuntimeError('Capture infrastructure failure; no retry: '+ident)
            row=audit_closed(root,parent,plan,plan_sha);verify_plan(plan_path,plan_sha)
            append(ledger,dict(event='actual_closed',utc=now(),**row))
        if args.start_index==0:
            append(ledger,dict(event='first_fit_preview_closed',utc=now(),all40_actual_closed=False,
                snapshots=13,remaining_parents_not_started=39,automatic_continuation=False,
                model_forward_calls=0,no_source_or_snapshot_replacement=True))
        else:
            audit_closed(root,plan['parents'][0],plan,plan_sha)
            append(ledger,dict(event='complete',utc=now(),all40_actual_closed=True,snapshots=520,
                model_forward_calls=0,no_source_or_snapshot_replacement=True))
        return 0
    except BaseException as error:
        append(ledger,dict(event='infrastructure_stop',utc=now(),error_type=type(error).__name__,
            error=str(error),no_retry=True));raise


if __name__=='__main__': raise SystemExit(main())
