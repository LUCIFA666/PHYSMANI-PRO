"""One fixed TRAIN full-collector diagnostic with read-only probe logging.

Original collection, tracker, CAD ghosts, teacher and control inputs are used.
Only this fresh process wraps AlignedRobotModel._probe in memory; its original
exception is re-raised unchanged. Diagnostic outputs are never training cases.
"""
import argparse
from datetime import datetime,timezone
import faulthandler
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
CASE='2193509ecffd78f5'
TASK='remove_cups_from_rotating_frame_high_speed'


def clean(value):
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return dict(nonfinite=repr(value))
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    return value


def pose_difference(before,after):
    a,b=np.asarray(before,float),np.asarray(after,float)
    result=dict(before_xyzw=a.tolist(),after_xyzw=b.tolist())
    if a.shape!=b.shape or a.shape!=(7,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(result,valid=False,reason='nonfinite or malformed raw world pose')
    norms=[float(np.linalg.norm(value[3:])) for value in (a,b)]
    result.update(valid=all(value>0 for value in norms),position_delta_m=(b[:3]-a[:3]).tolist(),
        position_component_max_abs_m=float(np.max(np.abs(b[:3]-a[:3]))),
        quaternion_norm_before=norms[0],quaternion_norm_after=norms[1],
        quaternion_raw_max_abs_difference=float(np.max(np.abs(b[3:]-a[3:]))))
    if result['valid']:
        result['physical_rotation_difference_rad']=float((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude())
    return result


def differences(before,after):
    result={}
    for name in ('q','servo_targets'):
        a,b=np.asarray(before[name],float),np.asarray(after[name],float)
        result[name]=dict(delta=(b-a).tolist(),max_abs_difference=float(np.max(np.abs(b-a))),
            frozen_1e_6_check_passed=bool(np.allclose(a,b,rtol=0,atol=1e-6)))
    ik=pose_difference(before['ik_world_xyzw'],after['ik_world_xyzw']);result['ik_world_SE3']=ik
    conditions=dict(q=result['q']['frozen_1e_6_check_passed'],servo_targets=result['servo_targets']['frozen_1e_6_check_passed'],
        ik_shape_finite=ik.get('valid',False),
        ik_quaternion_norm_before=bool(ik.get('valid') and abs(ik['quaternion_norm_before']-1.)<=1e-6),
        ik_quaternion_norm_after=bool(ik.get('valid') and abs(ik['quaternion_norm_after']-1.)<=1e-6),
        ik_position=bool(ik.get('valid') and ik['position_component_max_abs_m']<=1e-6),
        ik_physical_orientation=bool(ik.get('valid') and ik['physical_rotation_difference_rad']<=1e-6))
    result.update(frozen_probe_checks_from_wrapper_snapshots=conditions,
        failed_check_names=[name for name,valid in conditions.items() if not valid],
        threshold_changed=False,thresholds=dict(q_rad=1e-6,servo_rad=1e-6,ik_position_component_m=1e-6,
            quaternion_norm_absolute=1e-6,ik_physical_orientation_rad=1e-6),
        snapshots_are_extra_read_only_reads_not_the_original_probe_local_variables=True)
    result['world_poses']={name:pose_difference(value,after['world_poses'][name])
        for name,value in before.get('world_poses',{}).items() if name in after.get('world_poses',{})}
    return clean(result)


def callback_metadata(fn):
    from our_mani.contracts import Pose
    values={};closure=getattr(fn,'__closure__',None) or ()
    for name,cell in zip(getattr(getattr(fn,'__code__',None),'co_freevars',()),closure):
        try: value=cell.cell_contents
        except ValueError: continue
        if isinstance(value,Pose): values[name]=value.xyzw().tolist()
    return dict(name=getattr(fn,'__qualname__',repr(fn)),pure_pose_closure_values=values)


def snapshot(model):
    env=model.service.env;arm=model.arm
    from our_mani.multitask.native_tasks import resolve_native_roles
    clock_before=model.service._clock()
    poses=dict(robot_base=arm.get_pose(),robot_tip=arm.get_tip().get_pose())
    roles=resolve_native_roles(env.task,env.spec,scene_object_resolver=env.resolve_scene_object)
    for role in env.spec.roles:
        if role.kind=='body': poses['metric_only_native_body:'+role.name]=roles[role.name].get_pose()
    for role,body in model.service.estimated_bodies.items(): poses['CAD_collision_ghost:'+role]=body.ghost.get_pose()
    value=dict(q=arm.get_joint_positions(),servo_targets=arm.get_joint_target_positions(),
        ik_world_xyzw=arm._ik_target.get_pose(),world_poses=poses,clock=model.service._clock(),
        counted_tick=int(env.ledger.ticks),task_time=float(env.task.t),physics_time=float(env.physics_time()),
        estimated_collision_body_ids=list(model.service.estimated_bodies),
        estimated_attachment_ids=list(model.service.estimated_attachments),
        metric_only_native_world_poses_are_not_policy_inputs=True)
    value['snapshot_reads_changed_clock']=clock_before!=model.service._clock()
    return clean(value)


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--case-id',default=CASE,choices=(CASE,))
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    from our_mani.multitask.evaluation import atomic_json,file_sha
    import collect_multitask_training_r10 as collector
    plan_path=(ROOT/args.plan).resolve();plan=collector.verify_plan(plan_path)
    collector.require_r09_release(ROOT/'artifacts/framework/aligned_r09/final_report.json')
    demos=json.loads((ROOT/'artifacts/framework/aligned_r09/demo_index.json').read_text())
    if demos.get('complete') is not True or demos.get('completed')!=10: raise ValueError('R09 ten-demo release required')
    selected=[case for case in plan['cases'] if case['case_id']==CASE]
    if len(selected)!=1 or selected[0]['task']!=TASK or selected[0]['split']!='train' or selected[0]['replica']!=0:
        raise ValueError('only the originally failing fixed TRAIN task/source/replica is authorized')
    out=(ROOT/args.out).resolve();base=ROOT/'artifacts/framework/multitask_r10/diagnostics'
    if not out.is_relative_to(base) or out.exists(): raise ValueError('use a new explicit diagnostic directory')
    out.mkdir(parents=True)
    from our_mani.multitask.maps import AlignedRobotModel
    original=AlignedRobotModel._probe;probe_log=out/'probe_records.jsonl';count=0;first_failure=None;logging_errors=[]
    source_files={str(path.relative_to(ROOT)):file_sha(path) for path in (
        Path(__file__),ROOT/'our_mani/multitask/maps.py',ROOT/'our_mani/multitask/simulation.py',
        ROOT/'our_mani/reachability.py',ROOT/'scripts/collect_multitask_training_r10.py')}
    header=dict(diagnostic_only=True,eligible_for_training=False,formal_metrics_included=False,
        original_collector_path_used=True,tracker_and_CAD_ghosts_created_by_original_collector=True,
        plan=dict(path=str(plan_path),sha256=file_sha(plan_path),plan_sha256=plan['plan_sha256']),case=selected[0],
        process_id=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),GPU_used=False,
        instrumentation_source_sha256=source_files,
        original_probe_source_sha256=file_sha(ROOT/'our_mani/multitask/maps.py'),
        original_probe_function_source=inspect.getsource(original),
        instrumentation='fresh-process class wrapper; read-only before/after; callback/control/input/threshold unchanged',
        native_body_pose_reads_are_diagnostic_metrics_only=True,
        created_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'inputs.json',header)
    def wrapped(model,joints,fn):
        nonlocal count,first_failure
        index=count;count+=1;row=dict(probe_index=index,requested_joints=np.asarray(joints).tolist(),callback=callback_metadata(fn))
        try: row['before']=snapshot(model)
        except Exception as error: row['before_read_error']=repr(error)
        error=None
        try:
            result=original(model,joints,fn);row['original_exception']=None;return result
        except BaseException as caught:
            error=caught;row['original_exception']=repr(caught);row['original_exception_type']=type(caught).__name__
            row['original_traceback']=traceback.format_exc()
            raise
        finally:
            try:
                row['after']=snapshot(model)
                if 'before' in row: row['differences']=differences(row['before'],row['after'])
            except Exception as diagnostic_error: row['after_or_difference_read_error']=repr(diagnostic_error)
            try:
                with probe_log.open('a') as stream: stream.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                if error is not None and first_failure is None:
                    first_failure=row;atomic_json(out/'first_original_probe_failure.json',clean(row))
                print(json.dumps(dict(probe_index=index,tick=row.get('before',{}).get('counted_tick'),
                    error=row.get('original_exception'),failed_checks=row.get('differences',{}).get('failed_check_names')),ensure_ascii=False),flush=True)
            except Exception as diagnostic_error:
                logging_errors.append(repr(diagnostic_error))
    AlignedRobotModel._probe=wrapped
    status='running';exception=None;original_record=None
    try:
        # Original collector creates its complete native path in this isolated
        # diagnostic subtree. It is never merged into canonical acquisition.
        original_record=collector.collect_native_case(plan_path,CASE,out/'collector_execution',
            ROOT/'artifacts/framework/aligned_r09/final_report.json')
        status='completed_without_original_probe_failure'
        raw=collector.case_directory(out/'collector_execution',selected[0])/'case.json'
        if raw.exists():
            # Preserve the original bytes, but remove the ordinary case name and
            # explicitly quarantine its receipt as diagnostic-only after cleanup.
            record=json.loads(raw.read_text());record.update(status='diagnostic_only_completed',
                diagnostic_only=True,eligible_for_training=False,formal_metrics_included=False,
                instrumentation_source_sha256=source_files)
            atomic_json(raw.with_name('diagnostic_only_collector_receipt.json'),record);raw.unlink()
    except BaseException as error:
        status='original_collector_failed';exception=dict(type=type(error).__name__,repr=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        AlignedRobotModel._probe=original
        unchanged={name:file_sha(ROOT/name)==checksum for name,checksum in source_files.items()}
        report=dict(**header,status=status,original_exception=exception,original_probe_calls=count,
            first_original_probe_failure_captured=first_failure is not None,
            first_failure_check_names=first_failure.get('differences',{}).get('failed_check_names') if first_failure else None,
            logging_errors=logging_errors,frozen_sources_unchanged=unchanged,
            first_failure_path=str(out/'first_original_probe_failure.json') if first_failure else None,
            probe_records_path=str(probe_log),native_execution_repeated=False,
            simplified_diagnostic_success_is_not_resolution_evidence=True,
            diagnostic_completed_ticks=(original_record.get('alignment',{}).get('clock',{}).get('ticks') if original_record else
                first_failure.get('before',{}).get('counted_tick') if first_failure else None))
        atomic_json(out/'diagnostic_report.json',clean(report))


if __name__=='__main__': main()
