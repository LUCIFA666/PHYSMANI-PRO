"""Robot-only counted open/close probe; no task-object pose used."""
import argparse
from pathlib import Path
import numpy as np
from .common import OUT,read,write,append,sources,verify,stopped,sha

def chain(obj):
    result=[]
    while obj is not None:
        result.append(dict(name=obj.get_name(),type=type(obj).__name__))
        obj=obj.get_parent()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stopped();case=read(args.case)
    calibration=str(Path(case['calibration_episode']).resolve(strict=True))
    if read(OUT/'calibration_lock.json').get(calibration)!=sha(calibration):raise ValueError('Calibration changed')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);frozen=sources();backend=None
    from .backend import TaskExecutionBackend
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor
    from ..execution_interface_20261111_v1.commands import JointTrajectory
    from ..local_mpc_20261102.executor import JointLimits
    try:
        backend=TaskExecutionBackend(case['task'],calibration,seed=case['seed'],variation=0,task_time_limit_seconds=60,
                                    execute_native=True,dataset_use='train',renderer_profile='official_all_GL')
        write(output/'request.json',dict(case=case,source_sha256=frozen,robot_only=True,task_object_pose_read=False,
            tip_chain=chain(backend.arm.get_tip()),attachment_chain=chain(backend.gripper._attach_point)))
        native=backend.joint_limits();limits=JointLimits(native['lower'],native['upper'],native['velocity'],acceleration=12.,jerk=1000.)
        core=PhysManiExecutor(backend,limits,initial_gripper_opening=backend.requested_opening)
        q=np.asarray(backend.get_joint_positions());transforms=[]
        for opening in (0.,1.):
            for _ in range(60):
                stopped()
                if backend.terminal():break
                receipt=core.execute_command(JointTrajectory(start_time=backend.clock(),positions=[q],times_from_start=[.05],opening=opening))
                append(output/'execution.jsonl',receipt.to_dict())
                tip=np.asarray(backend.arm.get_tip().get_matrix());attach=np.asarray(backend.gripper._attach_point.get_matrix())
                relative=np.linalg.solve(tip,attach);transforms.append(relative)
                append(output/'robot_geometry.jsonl',dict(time=backend.clock(),opening=opening,open_amount=backend.gripper.get_open_amount(),
                    tip_to_attachment_matrix=relative,gripper_done=core.gripper_done))
        from ..executor_contract_repair_20261112_v2.audit import audit_execution
        audit=audit_execution(output/'execution.jsonl');verify(frozen)
        result=dict(complete=True,native_closed=True,source_unchanged=True,robot_only=True,actual_ticks=backend._ticks,
            actual_seconds=backend.clock(),tip_attachment_max_matrix_change=float(np.max(abs(np.asarray(transforms)-transforms[0]))),
            issued_reference_audit=audit,hardware_contact_audit=backend.hardware_contact_audit(),NN_forward_calls=0)
        backend.close();write(output/'result.json',result);print(result,flush=True);return 0
    finally:
        if backend is not None:backend.close()

if __name__=='__main__':raise SystemExit(main())
