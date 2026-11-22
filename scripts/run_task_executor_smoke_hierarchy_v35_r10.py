"""TRAIN reset-only waypoint hierarchy evidence, never an online actor input."""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from scripts import evaluate_multitask_case_r10 as evaluator
from scripts import collect_multitask_recovery_r10 as recovery

TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')


def chain(obj):
    result=[];seen=set()
    while obj is not None:
        handle=obj.get_handle()
        if handle in seen:raise ValueError('cycle in actual native hierarchy')
        seen.add(handle);result.append(dict(name=obj.get_name(),pose_world=list(map(float,obj.get_pose()))));obj=obj.get_parent()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--frozen-inputs',required=True)
    p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--raw-episode-id',required=True);p.add_argument('--split',choices=['train'],required=True)
    p.add_argument('--method',choices=['cv_geometry'],required=True);p.add_argument('--development-demo',action='store_true',required=True);p.add_argument('--out',required=True);p.add_argument('--check',action='store_true');args=p.parse_args();args.replica=0
    config=json.loads((ROOT/args.config).read_text());lock=json.loads((ROOT/args.frozen_inputs).read_text());verify_inputs(ROOT,lock['files'])
    for path in (Path(__file__),ROOT/args.config,ROOT/args.manifest):evaluator.locked_digest(path,lock['files'])
    manifest=json.loads((ROOT/args.manifest).read_text());(row,_),values=evaluator.select_source(manifest,'train',args.task,args.raw_episode_id)
    first4=[r['raw_episode_id'] for r in values if r['task']==args.task][:4]
    if args.raw_episode_id not in first4:raise ValueError('fixed first-four original TRAIN sources only')
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10/remove_hierarchy_TRAIN_v35') or out.exists():raise ValueError('fresh diagnostic output namespace required')
    plan=dict(protocol=config,inputs=lock['files']);captures=dict(simulations=[],paths=[])
    if args.check:print(json.dumps(dict(complete=True,native_started=False,teacher_used=False,policy_used=False)));return
    env=None
    try:
        with recovery.scoped_owned_hooks(plan,captures):
            env,_=evaluator.NativeBackend().environment(dict(args=args,config=config,row=row))
            before=(env.time,env.ledger.ticks,env.physics_time(),env.task.t)
            task=env.task;frame=task._frame_base
            waypoints={}
            for i in range(6):
                obj=getattr(task,'w'+str(i));waypoints['waypoint'+str(i)]=dict(parent_chain=chain(obj),
                    frame_local_xyzw=list(map(float,obj.get_pose(relative_to=frame))),
                    cup_local_xyzw={c.get_name():list(map(float,obj.get_pose(relative_to=c))) for c in task.cups},
                    spoke_local_xyzw={s.get_name():list(map(float,obj.get_pose(relative_to=s))) for s in task.spokes})
            after=(env.time,env.ledger.ticks,env.physics_time(),env.task.t)
            if before!=after or before[0]!=0 or before[1]!=0:raise ValueError('hierarchy observer advanced episode physics')
            receipt=env.alignment_receipt();reset=env.reset_recovery_receipt
            env.close();env=None
        verify_inputs(ROOT,lock['files'])
        out.mkdir(parents=True)
        record=dict(schema='r10_fixed_TRAIN_reset_only_waypoint_hierarchy_v35',complete=True,task=args.task,source=row,
            waypoints=waypoints,clock_before=before,clock_after=after,clock_unchanged=True,
            actual_reset_receipt=reset,alignment_receipt=receipt,process_id=os.getpid(),
            frozen_input_lock=dict(path=args.frozen_inputs,sha256=file_sha(ROOT/args.frozen_inputs)),
            teacher_used=False,policy_used=False,GPU_used=False,counted_actions=0,native_labels_are_offline_evidence_only=True,
            runtime_actor_input=False,TEST_read=False)
        atomic_json(out/'hierarchy.json',record);print(json.dumps(dict(complete=True,task=args.task,source=args.raw_episode_id,counted_actions=0)),flush=True)
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
