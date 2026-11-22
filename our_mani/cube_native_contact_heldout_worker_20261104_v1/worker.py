"""Current-runtime corrected Cube worker. Default is no-model/no-Native check."""
import argparse
from pathlib import Path
import json
import traceback

from .contracts import (ROOT,SCHEMA,FREEZE_SCHEMA,PROFILES,validate_plan,case_config,
    sources_for_plan,code_sources_for_plan,native_sources_for_plan,check_freeze,sha)
from ..local_mpc_20261102 import worker as original
from ..local_mpc_20261102.records import write_new,utc,archive_sources,verify_sources


def check(plan_path,case_id,profile,*,source_freeze=None,source_freeze_sha256=None,strict=False):
    plan=validate_plan(plan_path)
    case=next((c for c in plan['cases']if c['id']==case_id),None)
    if case is None or profile not in PROFILES:raise ValueError('Fixed implemented case/profile required')
    config=case_config(plan,case,profile);missing=[]
    cp=config['candidate_transformer_checkpoint'];cp_sha=config['candidate_transformer_checkpoint_sha256']
    if cp is None or cp_sha is None:missing.append('actual_frozen_candidate_checkpoint')
    elif not (ROOT/cp).is_file()or sha(ROOT/cp)!=cp_sha:missing.append('actual_frozen_candidate_checkpoint')
    vision=ROOT/config['position_refiner_checkpoint']
    if not vision.is_file()or sha(vision)!=config['position_refiner_checkpoint_sha256']:
        missing.append('unchanged_actual_V10_position_checkpoint')
    # A freeze is for the complete fixed cohort, not a partially available profile.
    if any(p['checkpoint']is None or p['sha256']is None for p in plan['candidate_profiles'].values()):
        missing.append('complete_fixed_profile_checkpoint_set')
    if source_freeze is None or source_freeze_sha256 is None:missing.append('final_source_freeze')
    if plan['preparation_only']is not False or plan['execution_authorized']is not True:missing.append('Root_final_plan')
    sources=None
    if not missing:
        sources=check_freeze(plan_path,source_freeze,source_freeze_sha256)
        if strict:
            from ..cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
            forecast_sha=config['corrected_forecast_checkpoint_sha256']
            CubeResidualMotionDynamics(ROOT/config['corrected_forecast_checkpoint'],expected_sha256=forecast_sha)
            if profile=='native_contact_legacy':
                from ..cube_native_contact_feedback_live_20261104_v1.candidate_runtime import NativeContactPilotInferenceV1 as CorrectedFrozenPilotInference
                CorrectedFrozenPilotInference(ROOT/cp,expected_sha256=cp_sha,forecast_sha256=forecast_sha)
            else:
                from ..cube_native_contact_feedback_live_20261104_v1.candidate_runtime import NativeContactMatchedInferenceV1 as MatchedCorrectedCandidateInference
                MatchedCorrectedCandidateInference(ROOT/cp,expected_sha256=cp_sha,forecast_sha256=forecast_sha)
    if strict and missing:raise ValueError('Execution prerequisites missing: '+', '.join(missing))
    return dict(schema=SCHEMA,case_id=case_id,profile=profile,task=case['task'],seed=case['seed'],
        ready=not missing,missing=missing,Native_started=False,NN_forward_calls=0,
        strict_model_constructors_checked=bool(strict and not missing),physics_dt=.05,
        decision_ticks=2,budget_ticks=160,source_sha256=sources,
        repository_source_names_relative=True,Native_sources_separate=True,reobserve_enabled=False)


def execute(plan_path,case_id,profile,output,*,source_freeze,source_freeze_sha256):
    admission=check(plan_path,case_id,profile,source_freeze=source_freeze,
        source_freeze_sha256=source_freeze_sha256,strict=True)
    plan=validate_plan(plan_path);case=next(c for c in plan['cases']if c['id']==case_id)
    config=case_config(plan,case,profile);output=Path(output).resolve()
    family=ROOT/'runtime/framework/cube_native_contact_heldout_worker_20261104_v1'
    if family not in output.parents or output.exists():raise ValueError('Fresh new-family output only')
    stops=(output/'STOP',output.parent/'STOP',output.with_name(output.name+'.STOP'),family/'STOP',
        ROOT/'runtime/framework/local_mpc_20261102/STOP')
    if any(p.exists()for p in stops):raise ValueError('Administrative STOP preserved')
    from scripts.feedback_current_environment_20261104_v2 import current_environment
    environment=current_environment()
    native_sources=original.native_source_snapshot(config)
    native_seven=native_sources_for_plan(plan_path)
    sources=admission['source_sha256'];output.mkdir(parents=True,exist_ok=False)
    write_new(output/'run_inputs.json',dict(started_utc=utc(),config=config,scene='native_generated',
        source_sha256=sources,native_sources_sha256=native_sources,native_files_sha256=native_seven,
        native_sources_separate_from_repository_archive=True,frozen_environment=environment,
        current_runtime_profile=True,old_OS_equivalence_claimed=False,TEST_read=False,
        Native_labels_used_by_policy=True,Native_contact_feedback_used=True,Native_target_pose_used_by_policy=False,paired_case_id=case_id,candidate_profile=profile))
    backend=recorder=executor=None;code=1;native_constructor_entered=False
    try:
        archive_sources(output,sources)
        from ..cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1 as CubeTaskNativeBackendV1
        from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
        from ..local_mpc_20261102.recording import FrameRecorder
        if profile=='native_contact_legacy':
            from ..cube_native_contact_feedback_live_20261104_v1.producer import ObservedNativeContactCubeBackendV1 as Observer,run_native_contact_cube_v1 as run
            mode='legacy_pilot_with_corrected_forecast_development'
        else:
            from ..cube_native_contact_feedback_live_20261104_v1.producer import ObservedNativeContactCubeBackendV1 as Observer,run_native_contact_cube_v1 as run
            mode='matched_candidate_with_corrected_forecast'
        # Retain a partial object for shutdown if its constructor raises.
        backend=CubeTaskNativeBackendV1.__new__(CubeTaskNativeBackendV1)
        native_constructor_entered=True
        CubeTaskNativeBackendV1.__init__(backend,case['task'],case['calibration_episode'],seed=case['seed'],
            variation=case['variation'],budget_ticks=160,gripper_velocity=config['gripper_velocity'],execute_native=True)
        write_new(output/'initial_snapshot.json',backend.initial_snapshot)
        backend=CompletedGripperBackend(backend);recorder=FrameRecorder()
        backend=Observer(backend,output/'public_shadow',candidate_mode=mode,
            checkpoint=ROOT/config['candidate_transformer_checkpoint'],checkpoint_sha256=config['candidate_transformer_checkpoint_sha256'],
            forecast_checkpoint=ROOT/config['corrected_forecast_checkpoint'],forecast_checkpoint_sha256=config['corrected_forecast_checkpoint_sha256'],
            refiner_checkpoint=ROOT/config['position_refiner_checkpoint'],refiner_checkpoint_sha256=config['position_refiner_checkpoint_sha256'],
            recorder=recorder,save_frames=True)
        backend.sample();executor=original.make_executor(backend,config)
        result=run(backend,executor,output,config,recorder=recorder)
        original.append_ticks(backend,output);recorder.close();recorder=None
        if original.native_source_snapshot(config)!=native_sources:raise RuntimeError('Original actual task source changed')
        verify_sources(sources);check_freeze(plan_path,source_freeze,source_freeze_sha256)
        result.update(started_from_original_V72=False,original_V72_free_motion_search_reused=True,native_task=case['task'],seed=case['seed'],
            candidate_profile=profile,native_generated_motion=True,official_VAL_TEST=False,
            source_unchanged=True,gripper_completion_protocol=backend.gripper_completion_receipt(),
            reobserve_enabled=False,formal_task_qualification=False,finished_utc=utc())
        write_new(output/'result.json',result);code=0
    except BaseException as error:
        tick_error=None
        if backend is not None:
            try:original.append_ticks(backend,output)
            except BaseException as nested:tick_error=type(nested).__name__+': '+str(nested)
        if executor is not None and executor.last_receipt is not None:
            write_new(output/'last_execution_receipt.json',executor.last_receipt)
        write_new(output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            task=case['task'],profile=profile,source_sha256=sources,partial_tick_record_error=tick_error,
            infrastructure_failure=True,task_outcome_not_scored_from_this_error=True))
        traceback.print_exc()
    finally:
        closed=False;close_errors=[]
        if recorder is not None:
            try:recorder.close()
            except BaseException as error:close_errors.append('recorder:'+repr(error))
        if backend is not None:
            try:backend.close();closed=True
            except BaseException as error:close_errors.append('backend:'+repr(error))
        if close_errors:code=1
        write_new(output/'worker_exit.json',dict(exit_code=code,native_closed=closed,
            native_constructor_entered=native_constructor_entered,close_errors=close_errors,
            result_recorded=(output/'result.json').is_file(),finished_utc=utc()))
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--case-id')
    p.add_argument('--profile',choices=PROFILES);p.add_argument('--source-freeze',type=Path)
    p.add_argument('--source-freeze-sha256');p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--strict-check',action='store_true')
    p.add_argument('--print-source-freeze',action='store_true');a=p.parse_args(argv)
    if a.print_source_freeze:
        if a.execute:p.error('Source generation cannot execute a worker')
        print(json.dumps(dict(schema=FREEZE_SCHEMA,files_sha256=sources_for_plan(a.plan),
            native_files_sha256=native_sources_for_plan(a.plan)),allow_nan=False));return 0
    if a.case_id is None or a.profile is None:p.error('Worker check/execute requires a fixed case-id/profile')
    if not a.execute:
        print(json.dumps(check(a.plan,a.case_id,a.profile,source_freeze=a.source_freeze,
            source_freeze_sha256=a.source_freeze_sha256,strict=a.strict_check),allow_nan=False));return 0
    if a.output is None or a.source_freeze is None or not a.source_freeze_sha256:
        p.error('Execute requires fresh output and final external freeze/SHA')
    return execute(a.plan,a.case_id,a.profile,a.output,source_freeze=a.source_freeze,
        source_freeze_sha256=a.source_freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
