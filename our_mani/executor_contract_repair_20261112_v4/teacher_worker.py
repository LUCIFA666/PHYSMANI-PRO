"""TRAIN command snapshots as explicit executor inputs; no policy inference."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from .common import OUT,read,write,append,pointer,sha,sources,verify,stopped


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stopped();case=read(args.case)
    if case['role']!='TRAIN_executor_teacher_goal_diagnostic':raise ValueError('Explicit TRAIN teacher-goal diagnostic required')
    calibration=str(Path(case['calibration_episode']).resolve(strict=True))
    if read(OUT/'calibration_lock.json').get(calibration)!=sha(calibration):raise ValueError('TRAIN source changed')
    with h5py.File(calibration)as source:
        commands=source['low_dim_obs/expert_trajectory'];indices=[0,min(50,len(commands)-1)]
        targets=np.asarray(commands[indices])[:,0]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);frozen=sources()
    frozen[calibration]=sha(calibration)
    write(output/'request.json',dict(case=case,case_file=pointer(args.case),source_sha256=frozen,
        original_TRAIN_source=pointer(calibration),command_frame_indices=indices,command_snapshots=targets,
        archived_episode_replayed=False,NN_forward_calls=0,learned_policy_task_ability_evaluated=False))
    from .backend import TaskExecutionBackend
    from .pose_executor import PoseGoalExecutor,WorldPoseGoal
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor
    from ..local_mpc_20261102.executor import JointLimits
    from ..executor_contract_repair_20261112_v2.audit import audit_execution
    from scipy.spatial.transform import Rotation
    backend=None
    try:
        backend=TaskExecutionBackend(case['task'],calibration,seed=case['seed'],variation=0,
                    task_time_limit_seconds=60,execute_native=True,dataset_use='train',renderer_profile='official_all_GL')
        backend.on_hardware_contact=lambda r:append(output/'hardware_table_contacts.jsonl',r)
        native=backend.joint_limits();limits=JointLimits(native['lower'],native['upper'],native['velocity'],acceleration=12.,jerk=1000.)
        core=PhysManiExecutor(backend,limits,initial_gripper_opening=backend.requested_opening)
        dispatcher=PoseGoalExecutor(backend,core);phases=[];reason='teacher_snapshots_completed'
        for index,target in enumerate(targets):
            if backend.terminal():break
            receipts=[];start=backend.clock();rejected=None
            for chunk in range(20):
                stopped()
                if backend.terminal():break
                r=dispatcher.execute(WorldPoseGoal(backend.clock(),target[:3],target[3:7],float(target[7]>=.5)))
                append(output/'decisions.jsonl',dict(goal_index=index,chunk=chunk,**{k:v for k,v in r.items()if k!='execution'}))
                for row in r['execution']:append(output/'execution.jsonl',row)
                receipts.append(r['status'])
                if r['status']not in ('chunk_complete','truncated'):
                    rejected=r['status'];reason=rejected;break
            actual=np.asarray(backend.arm.get_tip().get_pose())
            phases.append(dict(goal_index=index,source_frame_index=indices[index],target=target.tolist(),
                actual_pose=actual.tolist(),position_error_m=float(np.linalg.norm(actual[:3]-target[:3])),
                rotation_error_rad=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(target[3:7])).magnitude()),
                actual_seconds=backend.clock()-start,gripper_done=core.gripper_done,chunk_statuses=receipts,rejected=rejected))
            if rejected:break
        judge=backend.native_task_judge_receipt()
        if judge['native_task_success_latched']:reason='original_Native_task_success'
        audit=audit_execution(output/'execution.jsonl');verify(frozen)
        result=dict(complete=True,task=case['task'],seed=case['seed'],phases=phases,stop_reason=reason,
            original_native_success=judge['native_task_success_latched'],native_task_judge=judge,
            actual_ticks=backend._ticks,actual_seconds=backend.clock(),issued_reference_audit=audit,
            hardware_contact_audit=backend.hardware_contact_audit(),source_unchanged=True,
            NN_forward_calls=0,optimizer_steps=0,learned_policy_task_success_claimed=False,
            original_TRAIN_source=pointer(calibration),archived_episode_replayed=False)
        backend.close();result['native_closed']=True;write(output/'result.json',result)
        print({k:v for k,v in result.items()if k not in ('native_task_judge','phases')},flush=True);return 0
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if backend is not None:backend.close()


if __name__=='__main__':raise SystemExit(main())
