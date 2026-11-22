"""Frozen-model TRAIN check of the repaired executor's actual consumer path."""
import argparse
from pathlib import Path
from .common import OUT, read, write, append,pointer, stopped, sources, verify, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    stopped()
    case=read(args.case)
    if case['role']!='TRAIN_executor_diagnostic':raise ValueError('TRAIN executor diagnostic only')
    calibration=str(Path(case['calibration_episode']).resolve(strict=True))
    if read(OUT/'calibration_lock.json').get(calibration)!=sha(calibration):raise ValueError('Calibration changed')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    frozen=sources()
    checkpoint=Path(case['checkpoint']).resolve(strict=True)
    if sha(checkpoint)!=case['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs')
    frozen[str(checkpoint)]=case['checkpoint_sha256']
    write(output/'request.json',dict(case=case,case_file=pointer(args.case),source_sha256=frozen))
    from .backend import TaskExecutionBackend
    from ..multitask_executor_cycle_20261112_v6.runtime import PublicContinuousHistoryPolicy
    from .bank_runtime import FrozenCommandBankPolicy
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor
    from ..local_mpc_20261102.executor import JointLimits
    from .controller import run
    backend=None
    try:
        backend=TaskExecutionBackend(case['task'],calibration,seed=case['seed'],variation=0,
                    task_time_limit_seconds=60,execute_native=True,dataset_use='train',renderer_profile='official_all_GL')
        backend.on_hardware_contact=lambda value:append(output/'hardware_table_contacts.jsonl',value)
        native=backend.joint_limits()
        limits=JointLimits(lower=native['lower'],upper=native['upper'],velocity=native['velocity'],
                           acceleration=12.,jerk=1000.,provenance=native.get('provenance',{}))
        executor=PhysManiExecutor(backend,limits,initial_gripper_opening=backend.requested_opening)
        policy_class=FrozenCommandBankPolicy if case.get('proposal_selection')=='feasible_bank'else PublicContinuousHistoryPolicy
        policy=policy_class(str(checkpoint),case['checkpoint_sha256'])
        backend.on_public_frame=policy.observe
        backend.public_sample()
        result=run(backend,executor,policy,output,max_chunks=case['max_chunks'])
        result['hardware_contact_audit']=backend.hardware_contact_audit()
        from ..executor_contract_repair_20261112_v2.audit import audit_execution
        result['issued_reference_audit']=audit_execution(output/'execution.jsonl')
        if not result['issued_reference_audit']['all_reference_differences_within_limits']:
            raise RuntimeError('Issued reference finite differences exceed constraints')
        result.update(task=case['task'],seed=case['seed'],role=case['role'],
                      checkpoint_sha256=policy.checkpoint_sha256,source_unchanged=True)
        verify(frozen)
        backend.close();result['native_closed']=True
        write(output/'result.json',result)
        print({k:v for k,v in result.items()if k not in ('chunks','native_task_judge')},flush=True)
        return 0
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error)))
        if hasattr(error,'receipt'):write(output/'failed_execution_receipt.json',error.receipt.to_dict())
        raise
    finally:
        if backend is not None:backend.close()


if __name__=='__main__':raise SystemExit(main())
