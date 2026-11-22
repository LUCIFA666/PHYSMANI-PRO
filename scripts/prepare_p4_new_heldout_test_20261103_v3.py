"""V59 V3 materializer, closed-devVAL+ROOT-admission gated before any TEST mkdir.

Default command only checks readiness. Explicit --materialize is required and
cannot change the preassigned partition or bypass full source/actual-exit/model
identity. No simulator, fitting, official TEST, or heldout outcome reads occur.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
RUNNER="scripts/run_p4_serial_heldout_evaluation_20261103_v3.py"
SELF="scripts/prepare_p4_new_heldout_test_20261103_v3.py"
PARTITION="config/framework/P4_new_heldout_TEST_partition_v59_20261103_v1.json"
TRAIN="config/framework/P4_teacher_train_20261103_v1/plan.json"
DEVVAL="config/framework/P4_teacher_devVAL_20261103_v1/plan.json"
MODEL="artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/model_freeze_before_newdevVAL.json"


def module(root):
    spec=importlib.util.spec_from_file_location("newheldout_serial_evaluation",root/RUNNER)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def fixed_jobs():
    api=module(ROOT)
    path=ROOT/PARTITION
    if api.sha(path)!=api.PARTITION_SHA:raise ValueError("fixed V59 partition SHA required")
    return deepcopy(api.fixed_partition(ROOT)["jobs"])


def prepare(directory,*,student_plan,devval_owner,devval_runner,partition_path,admission_path,materialize=False):
    root=ROOT; api=module(root)
    for name,expected in api.DERIVED_TOOLS.items():
        if api.sha(root/name)!=expected:raise RuntimeError("unchanged reviewed V2 parent tools required")
    directory=Path(directory).resolve()
    if root/"config/framework" not in directory.parents or directory.exists():
        raise ValueError("fresh config/framework namespace required")
    partition_path=Path(partition_path).resolve()
    if partition_path!=root/api.PARTITION_PATH or api.sha(partition_path)!=api.PARTITION_SHA:
        raise ValueError("only the new preassigned V59 TEST16 partition is permitted")
    partition=api.fixed_partition(root)
    pointers={role:dict(path=name,sha256=api.sha(root/name))for role,name in api.PARTITION_PATHS.items()}
    if {k:{"path":v["path"],"sha256":v["sha256"]}for k,v in partition["preassigned_partitions"].items()}!=pointers:
        raise RuntimeError("all four preassigned source family identities must remain fixed")
    prior_parents,prior_seeds,prior_xy,_=api.partitions(root,dict(preassigned_partitions=pointers))
    train,devval=(api.load(root/name) for name in (TRAIN,DEVVAL))
    student_path=Path(student_plan).resolve();student_relative=str(student_path.relative_to(root))
    student=api.load(student_path)
    if len(train["jobs"])!=16 or len(devval["jobs"])!=8 or len(student["jobs"])!=8:
        raise ValueError("fixed TRAIN16, devVAL8 assignments required")
    owner=str(Path(devval_owner).resolve().relative_to(root))
    owner_inputs=api.load(root/owner/"inputs.json")
    declared=Path(owner_inputs.get("plan",""));declared=declared if declared.is_absolute() else root/declared
    if owner_inputs.get("plan_sha256")!=api.sha(student_path) or declared.resolve()!=student_path:
        raise RuntimeError("student devVAL parent owner plan identity failed")
    first=student["jobs"][0]
    base_path=root/first["config"];base=api.load(base_path)
    if any(base.get(key)!=value for key,value in api.PROTOCOL.items()):
        raise RuntimeError("exact frozen candidate weights/pi/budget base required")
    # This reads only the actual source identity envelope, never initial state,
    # native_ticks, result success, public success, or task evaluation records.
    val_run=api.load(root/first["path"]/"run_inputs.json")
    native=val_run["native_sources_sha256"]
    api.check_native_sources(base,native)
    historical_common={name:value for name,value in val_run["source_sha256"].items() if name!=first["config"]}
    if len(historical_common)<89:
        raise RuntimeError("full reviewed devVAL controller inventory including original 90 sources required")
    api.check_map(root,historical_common)
    sys.path.insert(0,str(root))
    from our_mani.local_mpc_20261102.records import freeze_sources
    api.check_map(root,api.load(root/MODEL)["files_sha256"])
    runner_path=Path(devval_runner).resolve()
    if api.sha(runner_path)!=owner_inputs["runner_sha256"]:
        raise ValueError("historical actual devVAL owner runner SHA mismatch")
    val_evidence=dict(plan=dict(path=student_relative,sha256=api.sha(student_path)),plan_schema=student["schema"],owner=owner,
        owner_inputs=dict(path=owner+"/inputs.json",sha256=api.sha(root/owner/"inputs.json")),
        runner_sha256=owner_inputs["runner_sha256"],
        admission="identity gate only; separate mandatory ROOT quality admission is checked before TEST materialization")
    # Complete identity checks precede creating any TEST configs. An external
    # historical runner is hash-checked above and archived, never executed.
    reference_gate=api.verify_devval(root,dict(preassigned_partitions=pointers),dict(
        development_validation=val_evidence,common_controller_files_sha256=historical_common,
        native_sources_sha256=native,behavior_configuration=api.behavior_configuration(base),
        expected_worker_information_setting=val_run["information_setting"]))
    admission_path=Path(admission_path).resolve()
    if admission_path!=root/api.ADMISSION_PATH:
        raise ValueError("fixed ROOT V59 admission path required")
    admission_pointer=dict(path=str(admission_path.relative_to(root)),sha256=api.sha(admission_path))
    admission_evidence=api.verify_admission(root,admission_pointer,devval_plan=val_evidence["plan"],
        heldout_partition=dict(path=str(partition_path.relative_to(root)),sha256=api.sha(partition_path)))
    if not materialize:
        return dict(preflight_passed=True,materialization_ready=True,TEST_configs_materialized=False,
            Native_started=False,complete_devVAL_actual_exit_gate=reference_gate,root_quality_admission=admission_evidence,
            requires_explicit_materialize=True)
    existing_parents=set(prior_parents);existing_seeds=set(prior_seeds);existing_xy=set(prior_xy)
    jobs=[];configs=[];name=directory.name
    # Assign the complete TEST split in memory before writing any new scene.
    for assigned in partition["jobs"]:
        job=deepcopy(assigned);scene=job["scene"];point=tuple(job["initial_xy"])
        velocity=job["initial_velocity_m_s"][:2];changed=job["changed_velocity_m_s"][:2]
        tick=job["speed_change_at_tick"];seed=job["seed"];parent=job["parent_id"];episode=job["id"]
        if point in existing_xy or seed in existing_seeds or parent in existing_parents:
            raise ValueError("preassigned heldout identity overlaps any of48 prior assignments")
        existing_xy.add(point);existing_seeds.add(seed);existing_parents.add(parent)
        job.update(config=str((directory/(episode+".json")).relative_to(root)),
            path=f"{api.FAMILY}/{name}/{episode}",evaluation=f"{api.ARTIFACTS}/{name}/{episode}")
        cfg=deepcopy(base)
        cfg.update(schema="P4_new_generated_heldout_TEST_V59_public_frozen_controller_20261103_v3",
            seed=seed,dataset_use="test",initial_target_position=[*point,.775],
            uniform_velocity=[*velocity,0.],speed_change_at_tick=tick,changed_velocity=[*changed,0.],
            simulation_seconds=8.,evaluation_role="new_heldout_TEST_under_modified_control_protocol_not_official_PhysMani_TEST",
            information_setting="public_RGBD_goalaware_red100_robot_causal_GRU_frozen_candidate_pi_V59",
            official_PhysMani_TEST=False,official_VAL_or_TEST_source=False,
            new_generated_heldout_TEST=True,multi_task_or_rotation_generalization_claim=False,
            TEST_labels_used_for_fit_or_selection=False,boundary_bounce_preserved=True,
            initial_uniform_is_not_a_no_bounce_guarantee=True)
        for stale in ("paired_teacher_case","paired_reference","paired_student_v51_case","preassigned_parent_split_preserved"):
            cfg.pop(stale,None)
        cfg["P4_collection_metadata"]=dict(parent_id=parent,split="test",episode_id=episode,
            variant_id=name,parent_assigned_before_episode_generation=True,
            all_splits_assigned_before_TEST_generation=True,randomized_initialization=False,
            official_VAL_or_TEST_source=False,initial_z_is_placeholder_replaced_by_Native_default=True,
            single_axis_aligned_cube_size_m=.05,no_outcome_dependent_scenario_selection=True,
            old_NN_finger_response_is_not_requalified=True)
        jobs.append(job);configs.append(cfg)
    external_runtime=api.RUNTIME_SH
    calibration=base["calibration_train_episode"]
    if "/train_package_compressed/" not in calibration:
        raise ValueError("shared original TRAIN calibration only")
    external={name:api.sha(name) for name in (external_runtime,calibration)}
    # Recheck both immutable admission documents and current controller bytes
    # immediately before the first prospective TEST write.
    api.check_map(root,{admission_pointer["path"]:admission_pointer["sha256"],
        admission_evidence["entry_gate_protocol"]["path"]:admission_evidence["entry_gate_protocol"]["sha256"]})
    api.check_map(root,historical_common)
    # Output directories are not created here. Only the new config namespace.
    directory.mkdir(parents=True,exist_ok=False)
    common=None
    for job,cfg in zip(jobs,configs):
        path=root/job["config"];api.write_new(path,cfg);job["config_sha256"]=api.sha(path)
        current={k:v for k,v in freeze_sources(path,"p1").items() if k!=job["config"]}
        if current!=historical_common or (common is not None and current!=common):
            raise RuntimeError("TEST must use the exact existing devVAL controller, model and parameters")
        common=current
    plan=dict(schema=api.PLAN_SCHEMA,split="test",prepared_only=True,Native_started=False,training_started=False,
        scope="fresh V59 heldout TEST under modified100ms protocol; single known5cm cube; neither official benchmark nor multi-task qualification",
        official_PhysMani_TEST=False,official_VAL_data_read=False,official_TEST_data_read=False,
        multi_task_or_rotation_generalization_claim=False,all_splits_assigned_before_TEST_generation=True,
        selection_from_TEST_results=False,TEST_task_failure_does_not_stop_or_replace_remaining_parents=True,
        no_training_or_model_selection_after_TEST_open=True,source_equality_required_between_devVAL_and_TEST=True,
        semantic_tasks=1,initial_object_yaw_forced_or_randomized=False,
        maximum_physical_budget_seconds=8.,early_public_controller_termination_is_unchanged=True,
        original_Native_XY_bounds=dict(x=[.05,.35],y=[-.45,.45]),boundary_bounce_preserved=True,
        uniform_definition="initial velocity only; original Native boundary reflection may reverse velocity within 8s",
        preassigned_partitions=pointers,
        heldout_partition=dict(path=str(partition_path.relative_to(root)),sha256=api.sha(partition_path)),
        shared_TRAIN_calibration=dict(path=calibration,sha256=external[calibration],use="shared calibration only; not heldout labels"),
        model_checkpoint_sha256=api.CHECKPOINT_SHA,candidate_pi_symmetry_representative=True,
        TRAIN_only_model_freeze=dict(path=MODEL,sha256=api.sha(root/MODEL)),
        complete_devVAL_reference_checked_before_config_materialization=True,
        explicit_ROOT_quality_admission=admission_pointer,
        cross_split_behavior_annotation_keys=sorted(api.SCENARIO_AND_ANNOTATION_KEYS),
        collection_metadata_annotation_keys=sorted(api.METADATA_ANNOTATION_KEYS),
        assignment_behavior_annotations_are_not_policy_inputs=True,
        runner=dict(path=RUNNER,sha256=api.sha(root/RUNNER)),
        preparer=dict(path=SELF,sha256=api.sha(root/SELF)),
        controller_freeze=str((directory/"controller_freeze.json").relative_to(root)),
        batch_output_parent=f"{api.FAMILY}/{name}",evaluation_output_parent=f"{api.ARTIFACTS}/{name}",
        suggested_owner=f"{api.FAMILY}/{name}_owner",estimated_native_workers=1,
        infra_case_wall_timeout_seconds=900.,STOP_grace_seconds=60.,
        STOP_contract="new owner/batch STOP only; actual child wait always captured; no old STOP read or removal",
        failure_contract="task failures retained; infrastructure or administrative STOP pauses remaining assigned cases, no retry/resume",
        control={key:api.PROTOCOL[key] for key in ("physics_dt","control_dt","decision_ticks","horizon_steps",
            "num_candidates","cem_sequences_total","cem_iteration_sequences","simulation_seconds")},jobs=jobs)
    plan_path=directory/"plan.json";api.write_new(plan_path,plan)
    archive=directory/"historical_devVAL_owner_runner.py"
    with archive.open("xb") as stream:stream.write(runner_path.read_bytes())
    val_evidence["runner"]=dict(path=str(archive.relative_to(root)),sha256=api.sha(archive))
    files={}
    extra=[RUNNER,SELF,*api.DERIVED_TOOLS,api.METRIC_MODULE,"scripts/run_frozen_system_environment_v820_r10.py",
        "artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json",
        TRAIN,DEVVAL,student_relative,str((root/owner/"inputs.json").relative_to(root)),str(plan_path.relative_to(root)),
        str(partition_path.relative_to(root))]
    extra += [j["config"]for source in (train,devval,student,api.load(root/api.PARTITION_PATHS["onpolicy_train"]))for j in source["jobs"]]
    extra += [api.PARTITION_PATHS["onpolicy_train"],api.PARTITION_PATHS["prior_test"],api.PARTITION_EVIDENCE_PATH]
    extra += [admission_pointer["path"],admission_evidence["entry_gate_protocol"]["path"]]
    extra += [j["config"] for j in jobs]
    if val_evidence.get("runner"):extra.append(val_evidence["runner"]["path"])
    for name in extra:files[name]=api.sha(root/name)
    freeze=dict(schema=api.FREEZE_SCHEMA,prepared_only=True,
        plan=dict(path=str(plan_path.relative_to(root)),sha256=api.sha(plan_path)),
        checkpoint_sha256=api.CHECKPOINT_SHA,candidate_pi_symmetry_representative=True,
        no_TEST_fitting_or_selection=True,all_splits_preassigned=True,
        per_case_source_count=len(common)+1,common_controller_files_sha256=common,
        behavior_configuration=api.behavior_configuration(base),
        evaluation_and_assignment_files_sha256=files,external_files_sha256=external,
        native_sources_sha256=native,development_validation=val_evidence,development_admission=admission_pointer,
        expected_worker_information_setting=val_run["information_setting"],
        complete_devVAL_model_identity=reference_gate,
        qualification="explicit identity freeze; not an accuracy, grasp, or official benchmark qualification",
        devVAL_results_opened_only_after_each_actual_exit_for_identity=True,root_quality_criteria_not_reinterpreted=True,TEST_results_read=False,Native_started=False)
    freeze_path=directory/"controller_freeze.json";api.write_new(freeze_path,freeze)
    api.validate_freeze(root,plan_path,freeze_path)
    return dict(plan=str(plan_path.relative_to(root)),plan_sha256=api.sha(plan_path),
        controller_freeze=str(freeze_path.relative_to(root)),controller_freeze_sha256=api.sha(freeze_path),
        runner_sha256=api.sha(root/RUNNER),heldout_parents=16,Native_started=False,
        runtime_case_outputs_created=False,TEST_configs_materialized=True,official_VAL_or_TEST_data_read=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition",type=Path,default=ROOT/PARTITION)
    parser.add_argument("--config-dir",type=Path,required=True)
    parser.add_argument("--admission",type=Path,default=ROOT/"artifacts/framework/local_mpc_20261102/P4_v59_development_admission_to_fresh_TEST_v1/admission.json")
    parser.add_argument("--materialize",action="store_true")
    parser.add_argument("--student-devval-plan",type=Path,required=True)
    parser.add_argument("--devval-owner",type=Path,required=True)
    parser.add_argument("--devval-runner",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=prepare(args.config_dir,student_plan=args.student_devval_plan,devval_owner=args.devval_owner,
                       devval_runner=args.devval_runner,partition_path=args.partition,admission_path=args.admission,materialize=args.materialize)
    except FileNotFoundError as error:
        if args.materialize:raise
        result=dict(preflight_passed=False,materialization_ready=False,TEST_configs_materialized=False,
            Native_started=False,missing=str(error.filename),reason="complete same-source V59 devVAL8 and ROOT quality admission required")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
