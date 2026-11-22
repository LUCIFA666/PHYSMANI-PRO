"""Current-runtime paired Cube worker; default check starts no Native or NN."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import traceback

from .contracts import ROOT,SCHEMA,validate_plan,case_config,sources_for_plan,native_sources_for_plan,check_freeze,sha
from ..local_mpc_20261102 import worker as original
from ..local_mpc_20261102.records import write_new,utc,archive_sources,verify_sources


def check(plan_path,case_id,profile,*,source_freeze=None,source_freeze_sha256=None,strict=False):
    plan=validate_plan(plan_path)
    case=next((c for c in plan['cases']if c['id']==case_id),None)
    if case is None:raise ValueError('Exactly a predeclared paired case required')
    config=case_config(plan,case,profile)
    cp=ROOT/config['candidate_transformer_checkpoint']
    missing=[]
    if not cp.is_file()or sha(cp)!=config['candidate_transformer_checkpoint_sha256']:
        missing.append('actual_frozen_candidate_checkpoint')
    vision=ROOT/config['position_refiner_checkpoint']
    if not vision.is_file()or sha(vision)!=config['position_refiner_checkpoint_sha256']:
        missing.append('unchanged_actual_V10_position_checkpoint')
    if source_freeze is None or source_freeze_sha256 is None:missing.append('final_source_freeze')
    if plan.get('preparation_only')is not False or plan.get('execution_authorized')is not True:
        missing.append('Root_final_plan')
    sources=None
    if not missing:
        sources=check_freeze(plan_path,source_freeze,source_freeze_sha256)
        if strict:
            from .candidate_runtime import candidate_loader
            # Actual model/spec/SHA checks precede mkdir/Native, zero forward.
            candidate_loader(cp,config['candidate_transformer_checkpoint_sha256'],profile)
    if strict and missing:raise ValueError('Execution prerequisites missing: '+', '.join(missing))
    return dict(schema=SCHEMA,case_id=case_id,profile=profile,task=case['task'],seed=case['seed'],
        ready=not missing,missing=missing,Native_started=False,NN_forward_calls=0,
        physics_dt=.05,decision_ticks=2,budget_ticks=160,source_sha256=sources)


def execute(plan_path,case_id,profile,output,*,source_freeze,source_freeze_sha256):
    admission=check(plan_path,case_id,profile,source_freeze=source_freeze,
        source_freeze_sha256=source_freeze_sha256,strict=True)
    plan=validate_plan(plan_path);case=next(c for c in plan['cases']if c['id']==case_id)
    config=case_config(plan,case,profile);output=Path(output).resolve()
    family=ROOT/'runtime/framework/cube_v72_live_20261104_v1'
    if family not in output.parents or output.exists():raise ValueError('Fresh new-family output only')
    stops=(output/'STOP',output.parent/'STOP',output.with_name(output.name+'.STOP'),family/'STOP',
           ROOT/'runtime/framework/local_mpc_20261102/STOP')
    if any(p.exists()for p in stops):raise ValueError('Administrative STOP preserved')
    from scripts.feedback_current_environment_20261104_v2 import current_environment
    environment=current_environment()  # explicit current profile, never v820 proof
    native_sources=original.native_source_snapshot(config)
    sources=admission['source_sha256'];output.mkdir(parents=True,exist_ok=False)
    write_new(output/'run_inputs.json',dict(started_utc=utc(),config=config,scene='native_generated',
        source_sha256=sources,native_sources_sha256=native_sources,frozen_environment=environment,
        current_runtime_profile=True,old_OS_equivalence_claimed=False,TEST_read=False,
        Native_labels_used_by_policy=False,paired_case_id=case_id,candidate_profile=profile))
    backend=recorder=executor=None;code=1
    try:
        archive_sources(output,sources)
        from .backend import CubeTaskNativeBackendV1
        from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
        from ..local_mpc_20261102.recording import FrameRecorder
        from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
        from ..local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Config
        from .producer import ObservedCubeV72Backend,run_cube_v72_live
        backend=CubeTaskNativeBackendV1(case['task'],case['calibration_episode'],seed=case['seed'],
            variation=case['variation'],budget_ticks=160,gripper_velocity=config['gripper_velocity'],execute_native=True)
        write_new(output/'initial_snapshot.json',backend.initial_snapshot)
        backend=CompletedGripperBackend(backend)
        recorder=FrameRecorder()
        backend=ObservedCubeV72Backend(backend,output/'public_shadow',candidate_mode=profile,
            checkpoint=ROOT/config['candidate_transformer_checkpoint'],
            checkpoint_sha256=config['candidate_transformer_checkpoint_sha256'],
            refiner_checkpoint=ROOT/config['position_refiner_checkpoint'],
            refiner_checkpoint_sha256=config['position_refiner_checkpoint_sha256'],
            goal_sphere_filter=True,postconfirmed_maintenance=True,
            active_config=ActiveVisualConfig(**config['active_visual_settings']),
            grasp_config=ActiveVisualGraspV41Config(**config['public_grasp_settings']),
            recorder=recorder,save_frames=True,require_public_goal_completion=True)
        backend.sample();executor=original.make_executor(backend,config)
        result=run_cube_v72_live(backend,executor,output,config,recorder=recorder)
        original.append_ticks(backend,output)
        recorder.close();recorder=None
        if original.native_source_snapshot(config)!=native_sources:raise RuntimeError('Original Native task source changed')
        verify_sources(sources);check_freeze(plan_path,source_freeze,source_freeze_sha256)
        result.update(started_from_original_V72=True,native_task=case['task'],seed=case['seed'],
            candidate_profile=profile,native_generated_motion=True,official_VAL_TEST=False,
            source_unchanged=True,gripper_completion_protocol=backend.gripper_completion_receipt(),
            formal_task_qualification=False,finished_utc=utc())
        write_new(output/'result.json',result);code=0
    except BaseException as error:
        if backend is not None:original.append_ticks(backend,output)
        if executor is not None and executor.last_receipt is not None:
            write_new(output/'last_execution_receipt.json',executor.last_receipt)
        write_new(output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            task=case['task'],profile=profile,source_sha256=sources,Native_task_outcome_is_not_infrastructure=True))
        traceback.print_exc()
    finally:
        closed=False
        try:
            if recorder is not None:recorder.close()
        finally:
            try:
                if backend is not None:backend.close();closed=True
            finally:
                write_new(output/'worker_exit.json',dict(exit_code=code,native_closed=closed,
                    result_recorded=(output/'result.json').is_file(),finished_utc=utc()))
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--case-id',required=True)
    p.add_argument('--profile',choices=('legacy_pilot','cube_candidate_v1'),required=True)
    p.add_argument('--source-freeze',type=Path);p.add_argument('--source-freeze-sha256')
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--print-source-freeze',action='store_true');a=p.parse_args(argv)
    if a.print_source_freeze:
        if a.execute:p.error('Cannot execute while generating source content')
        print(json.dumps(dict(schema='Cube_V72_current_runtime_execution_freeze_20261104_v1',
            files_sha256=sources_for_plan(a.plan),native_files_sha256=native_sources_for_plan(a.plan)),allow_nan=False));return 0
    if not a.execute:
        print(json.dumps(check(a.plan,a.case_id,a.profile,source_freeze=a.source_freeze,
            source_freeze_sha256=a.source_freeze_sha256),allow_nan=False));return 0
    if a.output is None or a.source_freeze is None or not a.source_freeze_sha256:
        p.error('Execute requires fresh output and external final source freeze/SHA')
    return execute(a.plan,a.case_id,a.profile,a.output,source_freeze=a.source_freeze,
        source_freeze_sha256=a.source_freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
