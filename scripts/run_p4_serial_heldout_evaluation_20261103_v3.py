"""V59 fresh-preassigned TEST16 V3, default preflight only.

All test materialization/execution requires the complete same-source eight-case
V59 devVAL exit+model chain AND the explicit ROOT predeclared quality-admission
record. Assignment never depends on TEST outcomes; no retry or replacement.
Serial process supervision/evaluation mechanisms retain the V2 implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "P4_new_generated_heldout_TEST_V59_20261103_v3"
FREEZE_SCHEMA = "P4_modified_control_heldout_TEST_V59_controller_freeze_20261103_v3"
CHECKPOINT_SHA = "61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60"
METRIC_MODULE = "our_mani/local_mpc_control_20261103/native_task_evaluation.py"
FAMILY = "runtime/framework/local_mpc_20261102"
ARTIFACTS = "artifacts/framework/local_mpc_20261102"
PARTITION_PATHS = dict(train="config/framework/P4_teacher_train_20261103_v1/plan.json",
    onpolicy_train="config/framework/P2_onpolicy_probe_TRAIN8_20261103_v1/plan.json",
    devval="config/framework/P4_teacher_devVAL_20261103_v1/plan.json",
    prior_test="config/framework/P4_new_heldout_TEST_partition_20261103_v1.json")
PARTITION_COUNTS=dict(train=16,onpolicy_train=8,devval=8,prior_test=16)
PARTITION_EVIDENCE_PATH="artifacts/framework/local_mpc_20261102/P4_new_heldout_partition_v59_preparation_v1/assignment_evidence.json"
PARTITION_EVIDENCE_SHA="1a33ed3f44cea08f9b3daa8085af77e88836f32c4ddeb4c0b70eee0fe90a0f1d"
RUNTIME_SH = "/home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh"
PARTITION_PATH = "config/framework/P4_new_heldout_TEST_partition_v59_20261103_v1.json"
PARTITION_SHA = "c917d28f7889258b8961a3e462bf511dd9819d1ee01df9f63951998b15c59234"
PROTOCOL = dict(physics_dt=.05, control_dt=.1, decision_ticks=2,
    horizon_seconds=.8, horizon_steps=8, num_candidates=8,
    cem_sequences_total=256, cem_iteration_sequences=[88,88,80],
    simulation_seconds=8., preserve_native_height=True, terminal_goal_cost=True,
    response_factory="coupled_joint_20261103_v1", response_controller="coupled",
    execution_model="coupled", action_space="joint", use_joint_goal_planner=True,
    public_grasp_candidate_transformer_v5=True, candidate_pi_symmetry_representative=True,
    candidate_transformer_checkpoint_sha256=CHECKPOINT_SHA,
    gripper_completion_latch=True, Native_labels_used_by_policy=False,
    geometry_fallback_used=False, Transformer_training_complete=True,
    candidate_darkred_target_v59=True,candidate_goal_sphere_filter_v56=True,
    postconfirmed_position_maintenance_v56=True,finger_response_fastpath=True,
    require_public_goal_completion=True,cache_fixed_probe_nominal_ik=True,
    probe_terminal_at_fixed_deadline=True,probe_onpolicy_joint_response=True)
SCENARIO_AND_ANNOTATION_KEYS = frozenset(("schema","seed","dataset_use","initial_target_position",
    "uniform_velocity","speed_change_at_tick","changed_velocity","evaluation_role","information_setting",
    "P4_collection_metadata","paired_reference","paired_teacher_case","paired_student_v51_case","preassigned_parent_split_preserved",
    "official_PhysMani_TEST","official_VAL_or_TEST_source","new_generated_heldout_TEST",
    "multi_task_or_rotation_generalization_claim","TEST_labels_used_for_fit_or_selection",
    "boundary_bounce_preserved","initial_uniform_is_not_a_no_bounce_guarantee"))

EXPECTED_INFORMATION = "public_candidate_darkred_v59_development"
V59_DEVVAL_SCHEMA = "P4_v59_conditional_full_reused_devVAL8_preassignment"
ADMISSION_PATH = "artifacts/framework/local_mpc_20261102/P4_v59_development_admission_to_fresh_TEST_v1/admission.json"
ENTRY_GATE_PROTOCOL_PATH = "artifacts/framework/local_mpc_20261102/P4_v59_development_admission_to_fresh_TEST_v1/entry_gate_protocol.json"
ENTRY_GATE_PROTOCOL_SHA = "1c06e839ae1b3300426e8239da22fb4887b96dcde0af078035448753beb8cb5e"
DERIVED_TOOLS = {
    "scripts/run_p4_serial_heldout_evaluation_20261103_v2.py":"e9047072fb735be7dc619e7860f27d68df41cf3b762205fb09186300d10b378d",
    "scripts/prepare_p4_new_heldout_test_20261103_v2.py":"bcada7424e630f7f88ab6c31bc435f63d1fa124e482cc671f2262dff84472738"}
V3_TOOLS = ("scripts/run_p4_serial_heldout_evaluation_20261103_v3.py",
            "scripts/prepare_p4_new_heldout_test_20261103_v3.py")
BEHAVIOR_ANNOTATION_KEYS = frozenset(("initial_speed_m_s","changed_speed_m_s","initial_direction_kind",
    "initial_speed_seen_in_training_assignments","speed_change_sign_relation"))
# Explicit annotation names only. Unknown keys remain behavior and cannot be
# silently ignored. No prefix wildcard or arbitrary nested dictionary removal.
METADATA_ANNOTATION_KEYS = frozenset(("parent_id","split","episode_id","variant_id",
    "parent_assigned_before_episode_generation","all_splits_assigned_before_TEST_generation",
    "randomized_initialization","official_VAL_or_TEST_source","initial_z_is_placeholder_replaced_by_Native_default",
    "single_axis_aligned_cube_size_m","no_outcome_dependent_scenario_selection",
    "old_NN_finger_response_is_not_requalified","new_development_VAL_only","tau_definition"))
SCENARIO_AND_ANNOTATION_KEYS = SCENARIO_AND_ANNOTATION_KEYS | BEHAVIOR_ANNOTATION_KEYS


def behavior_configuration(config):
    """Only scenario identity and explicit annotations may differ across splits."""
    metadata=config.get("P4_collection_metadata",{})
    if not isinstance(metadata,dict) or set(metadata)-METADATA_ANNOTATION_KEYS:
        raise ValueError("only the explicitly named collection annotations may differ")
    return {key:value for key,value in config.items() if key not in SCENARIO_AND_ANNOTATION_KEYS}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required: " + str(path))
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def path_in(root, value, family=None):
    """Canonical project-relative inputs; never follow an output alias outside."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != value:
        raise ValueError("canonical project-relative path required: " + str(value))
    root = Path(root).resolve()
    path = (root / relative).resolve()
    required = (root / family).resolve() if family else root
    if required not in path.parents:
        raise ValueError("path outside required project family: " + str(value))
    return path


def checked_pointer(root, pointer):
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError("exact path/SHA pointer required")
    path = path_in(root, pointer["path"])
    if sha(path) != pointer["sha256"]:
        raise RuntimeError("pointer SHA changed: " + str(path))
    return path


def check_map(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("nonempty exact file identity map required")
    for name, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64 or sha(path_in(root, name)) != expected:
            raise RuntimeError("frozen file changed: " + str(name))


def xy(config):
    p = config.get("initial_target_position")
    if not isinstance(p, list) or len(p) != 3 or not all(type(x) in (int,float) and math.isfinite(x) for x in p):
        raise ValueError("finite initial target XYZ required")
    return tuple(p[:2])


def assignment_motion_key(value):
    position=value.get("initial_xy",value.get("initial_target_position"))
    v0=value.get("initial_velocity_m_s",value.get("uniform_velocity"))
    v1=value.get("changed_velocity_m_s",value.get("changed_velocity"))
    if (not isinstance(position,list) or len(position)not in(2,3) or not isinstance(v0,list) or len(v0)!=3
            or not isinstance(v1,list) or len(v1)!=3 or type(value.get("speed_change_at_tick"))is not int
            or any(type(x)not in(int,float)or not math.isfinite(x)for x in position+v0+v1)):
        raise ValueError("finite preassigned XY/velocity/tick motion required")
    return tuple(round(float(x),12)for x in position[:2]+v0+[float(value["speed_change_at_tick"])]+v1)


def canonical_motion_sha256(value):
    return hashlib.sha256(json.dumps(assignment_motion_key(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def fixed_partition(root):
    path=path_in(root,PARTITION_PATH)
    if sha(path)!=PARTITION_SHA:raise RuntimeError("preassigned fresh V59 partition changed")
    value=load(path)
    if (value.get("schema")!="P4_new_heldout_TEST_partition_v59_20261103_v1" or value.get("split")!="test"
            or value.get("execution_status")!="conditional_not_materialized"
            or value.get("policy_independent_fixed_partition")is not True
            or value.get("model_inputs_unchanged_no_taskID_or_source_speed_or_behavior_annotation")is not True
            or set(value.get("behavior_annotation_fields",[]))!=BEHAVIOR_ANNOTATION_KEYS
            or len(value.get("jobs",[]))!=16):
        raise RuntimeError("fixed policy-independent V59 partition declaration required")
    evidence=value.get("assignment_evidence")
    if evidence!={"path":PARTITION_EVIDENCE_PATH,"sha256":PARTITION_EVIDENCE_SHA}:
        raise RuntimeError("original 48-assignment disjointness evidence required")
    checked_pointer(root,evidence)
    for index,job in enumerate(value["jobs"]):
        if (job.get("seed")!=261103500+index or job.get("split")!="test"
                or job.get("behavior_annotations_are_not_policy_inputs")is not True
                or set(job.get("behavior_annotations",{}))!=BEHAVIOR_ANNOTATION_KEYS
                or job.get("canonical_motion_sha256")!=canonical_motion_sha256(job)):
            raise RuntimeError("all16 ordered fixed motion identities and explicit non-policy annotations required")
    return value


def partitions(root, plan):
    """All four assignment families only; no prior TEST trajectory/result reads."""
    if set(plan.get("preassigned_partitions",{}))!=set(PARTITION_PATHS):
        raise ValueError("all TRAIN16/onpolicy8/devVAL8/priorTEST16 assignment pointers required")
    seen_parents,seen_seeds,seen_xy=set(),set(),set();evidence={}
    for role,count in PARTITION_COUNTS.items():
        reference=plan["preassigned_partitions"][role]
        if reference.get("path")!=PARTITION_PATHS[role]:raise ValueError("exact preassigned family path required")
        source=load(checked_pointer(root,reference))
        if len(source.get("jobs",[]))!=count:raise ValueError("fixed48 prior assignments required")
        for job in source["jobs"]:
            expected="val"if role=="devval"else "test"if role=="prior_test"else "train"
            if job.get("split")!=expected:raise ValueError("assignment role mismatch")
            if role=="prior_test":
                config=job;point=tuple(config["initial_xy"])
            else:
                config_path=path_in(root,job["config"])
                if sha(config_path)!=job["config_sha256"]:raise RuntimeError("prior assigned config changed")
                config=load(config_path);point=xy(config)
            assignment_motion_key(config)
            if job["parent_id"]in seen_parents or job["seed"]in seen_seeds:
                raise ValueError("all48 prior parent and seed identities must be disjoint")
            seen_parents.add(job["parent_id"]);seen_seeds.add(job["seed"]);seen_xy.add(point)
        evidence[role]=reference
    if len(seen_parents)!=48:raise ValueError("complete48 prior family parent roster required")
    return seen_parents,seen_seeds,seen_xy,evidence


def validate_plan(root, plan_path):
    plan_path = Path(plan_path).resolve(); plan = load(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("split") != "test":
        raise ValueError("explicit newly generated held-out TEST plan required")
    if any(plan.get(k) is not False for k in ("official_VAL_data_read", "official_TEST_data_read",
                                               "official_PhysMani_TEST", "multi_task_or_rotation_generalization_claim")):
        raise ValueError("new generated single-cube modified protocol scope required")
    if plan.get("all_splits_assigned_before_TEST_generation") is not True or plan.get("selection_from_TEST_results") is not False:
        raise ValueError("fixed preassigned partition with no TEST selection required")
    parents,seeds,points,_ = partitions(root,plan)
    jobs = plan.get("jobs",[])
    if len(jobs) != 16 or Counter(j.get("scene") for j in jobs) != dict(static=2,uniform=8,speed_change=6):
        raise ValueError("all sixteen fixed parents (2/8/6) required")
    batch = path_in(root,plan["batch_output_parent"],FAMILY)
    artifact = path_in(root,plan["evaluation_output_parent"],ARTIFACTS)
    configs=[]; identities=set(); out_paths=set()
    for index,job in enumerate(jobs):
        if (job.get("split") != "test" or type(job.get("seed")) is not int or job["seed"] != 261103500+index
                or job["parent_id"] in parents or job["seed"] in seeds or job["id"] in identities):
            raise ValueError("fixed ordered TEST parent/seed/ID contract failed")
        cfg_path=path_in(root,job["config"],"config/framework")
        if sha(cfg_path) != job["config_sha256"]:
            raise RuntimeError("held-out config changed")
        cfg=load(cfg_path); point=xy(cfg); meta=cfg.get("P4_collection_metadata",{})
        if point in points:
            raise ValueError("TEST XY must be unique and disjoint from TRAIN/devVAL")
        if not .05 <= point[0] <= .35 or not -.45 <= point[1] <= .45:
            raise ValueError("Native task area bounds required")
        if (cfg.get("dataset_use") != "test" or cfg.get("seed") != job["seed"] or
                any(meta.get(k) != v for k,v in dict(parent_id=job["parent_id"],split="test",episode_id=job["id"],
                                                    parent_assigned_before_episode_generation=True).items())):
            raise ValueError("exact parent/split/config identity required")
        if (list(point)!=job.get("initial_xy") or cfg.get("uniform_velocity")!=job.get("initial_velocity_m_s")
                or cfg.get("changed_velocity")!=job.get("changed_velocity_m_s")
                or cfg.get("speed_change_at_tick")!=job.get("speed_change_at_tick")):
            raise ValueError("config cannot change the policy-independent preassigned scene")
        if any(cfg.get(k) != v for k,v in PROTOCOL.items()):
            raise ValueError("frozen model/control budget/pi protocol changed")
        if any(cfg.get(k,False) for k in ("diagnostic_no_proxy_attachment","counterfactual_no_proxy",
                "public_grasp_prototype","public_grasp_active_visual_v4","public_grasp_active_visual_v41",
                "public_grasp_active_visual_v42","public_grasp_cold_probe_v43","public_motion_shadow","public_pregrasp_control")):
            raise ValueError("one explicit frozen public candidate controller and original Native proxy protocol only")
        if (cfg.get("task") != "pick_moving_target_on_the_table" or
                cfg.get("calibration_train_episode") != plan["shared_TRAIN_calibration"]["path"] or
                "/train_package_compressed/" not in cfg.get("calibration_train_episode","")):
            raise ValueError("shared original TRAIN calibration only")
        for key in ("uniform_velocity","changed_velocity"):
            v=cfg.get(key)
            if not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) or v[2]!=0:
                raise ValueError("finite planar velocity required")
        if type(cfg.get("speed_change_at_tick")) is not int or not 1 <= cfg["speed_change_at_tick"] < 160:
            raise ValueError("predeclared physical-tick change time required")
        if job["scene"] == "static" and any(cfg["uniform_velocity"]+cfg["changed_velocity"]):
            raise ValueError("static scenario must remain static")
        output=path_in(root,job["path"],FAMILY); evaluation=path_in(root,job["evaluation"],ARTIFACTS)
        if output.parent != batch or evaluation.parent != artifact or output in out_paths:
            raise ValueError("one unique case under exact fresh TEST batch required")
        parents.add(job["parent_id"]);seeds.add(job["seed"]);points.add(point);identities.add(job["id"])
        out_paths.add(output);configs.append(cfg)
    if plan.get("control") != {k:PROTOCOL[k] for k in ("physics_dt","control_dt","decision_ticks","horizon_steps",
            "num_candidates","cem_sequences_total","cem_iteration_sequences","simulation_seconds")}:
        raise ValueError("explicit complete unchanged compute/time budget required")
    if plan.get("heldout_partition")!={"path":PARTITION_PATH,"sha256":PARTITION_SHA}:
        raise ValueError("the previously frozen unchanged TEST16 partition is required")
    assignment=fixed_partition(root)
    if (assignment.get("schema")!="P4_new_heldout_TEST_partition_v59_20261103_v1" or
            assignment.get("policy_independent_fixed_partition") is not True or
            [assignment_identity(j) for j in jobs] != assignment.get("jobs")
            or {k:{"path":v["path"],"sha256":v["sha256"]}for k,v in assignment["preassigned_partitions"].items()}!=plan["preassigned_partitions"]):
        raise ValueError("TEST scene identities must equal the previously fixed policy-independent partition")
    return plan,configs


def assignment_identity(job):
    keys=("id","parent_id","split","seed","scene","motion_kind","initial_xy",
        "initial_velocity_m_s","speed_change_at_tick","changed_velocity_m_s",
        "boundary_bounce_preserved","initial_uniform_is_not_a_no_bounce_guarantee",
        "behavior_annotations","behavior_annotations_are_not_policy_inputs","canonical_motion_sha256")
    return {key:job[key]for key in keys}


def verify_admission(root, pointer, *, devval_plan, heldout_partition):
    if not isinstance(pointer,dict) or pointer.get("path")!=ADMISSION_PATH:
        raise ValueError("the ROOT V59 quality-admission record is mandatory")
    path=checked_pointer(root,pointer);record=load(path)
    if (record.get("schema")!="P4_V59_devVAL_to_newheldout_admission_v1"
            or record.get("allow_materialize_and_execute") is not True
            or record.get("devval_plan")!=devval_plan or record.get("heldout_partition")!=heldout_partition):
        raise RuntimeError("quality admission must bind exact V59 devVAL and fresh preassigned partition")
    protocol=record.get("entry_gate_protocol")
    if not isinstance(protocol,dict) or protocol.get("path")!=ENTRY_GATE_PROTOCOL_PATH:
        raise ValueError("predeclared ROOT entry-gate protocol pointer required")
    if protocol.get("sha256")!=ENTRY_GATE_PROTOCOL_SHA:
        raise RuntimeError("predeclared entry-gate protocol identity changed")
    protocol_path=checked_pointer(root,protocol);declaration=load(protocol_path)
    if (declaration.get("schema")!="P4_V59_devVAL_to_newheldout_entry_gate_protocol_v1"
            or declaration.get("full_devVAL_plan")!=devval_plan["path"] or declaration.get("new_TEST_partition")!=heldout_partition["path"]):
        raise RuntimeError("predeclared gate must bind the same full8 plan and new16 assignment")
    # Criteria are ROOT-owned. This tool must not invent thresholds, change
    # final-current to everTrue, or select parents using any TEST outcome.
    return dict(path=pointer["path"],sha256=pointer["sha256"],entry_gate_protocol=protocol,
        exact_binding_verified=True,criteria_reinterpreted_by_tool=False,
        protocol_sha256=sha(protocol_path))


def check_native_sources(config,native):
    if not isinstance(native,dict)or len(native)!=5:
        raise ValueError("full unchanged five-file Native source inventory required")
    native_root=Path(config["native_repo"])
    for name,expected in native.items():
        if Path(name).is_absolute()or ".."in Path(name).parts or sha(native_root/name)!=expected:
            raise RuntimeError("original Native task/environment source changed")


def validate_freeze(root, plan_path, freeze_path):
    plan,configs=validate_plan(root,plan_path); freeze=load(freeze_path)
    if freeze.get("expected_worker_information_setting")!=EXPECTED_INFORMATION:
        raise ValueError("exact V59 public worker information-setting required")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("explicit same model/controller TEST freeze required")
    if freeze.get("candidate_pi_symmetry_representative") is not True or freeze.get("no_TEST_fitting_or_selection") is not True:
        raise ValueError("pi representative and no TEST optimization contract required")
    if freeze.get("plan") != dict(path=str(Path(plan_path).resolve().relative_to(root)),sha256=sha(plan_path)):
        raise ValueError("freeze must bind exact TEST plan SHA")
    count=freeze.get("per_case_source_count")
    if type(count) is not int or count<90 or len(freeze.get("common_controller_files_sha256",{})) != count-1:
        raise ValueError("explicit full controller inventory, including the original 90 sources, required")
    check_map(root,freeze["common_controller_files_sha256"])
    check_map(root,freeze["evaluation_and_assignment_files_sha256"])
    if freeze["evaluation_and_assignment_files_sha256"].get(str(Path(__file__).resolve().relative_to(root))) != sha(__file__):
        raise ValueError("actual evaluation runner must be explicitly frozen")
    if freeze["evaluation_and_assignment_files_sha256"].get(METRIC_MODULE) != sha(Path(root)/METRIC_MODULE):
        raise ValueError("post-exit pure metric implementation must be frozen")
    for name,expected in DERIVED_TOOLS.items():
        if sha(path_in(root,name))!=expected or freeze["evaluation_and_assignment_files_sha256"].get(name)!=expected:
            raise RuntimeError("reviewed unchanged V2 derivation tool identity required")
    for name in (*V3_TOOLS,PARTITION_EVIDENCE_PATH):
        if freeze["evaluation_and_assignment_files_sha256"].get(name)!=sha(path_in(root,name)):
            raise RuntimeError("both new V3 tools must be explicitly frozen")
    admission=verify_admission(root,freeze.get("development_admission"),
        devval_plan=freeze["development_validation"]["plan"],heldout_partition=plan["heldout_partition"])
    for pointer in (freeze["development_admission"],admission["entry_gate_protocol"]):
        if freeze["evaluation_and_assignment_files_sha256"].get(pointer["path"])!=pointer["sha256"]:
            raise RuntimeError("quality-admission and predeclared-gate bytes must join the outer freeze")
    # Calibration is hashed as an opaque original TRAIN file, never parsed here.
    if set(freeze.get("external_files_sha256",{}))!={RUNTIME_SH,plan["shared_TRAIN_calibration"]["path"]}:
        raise ValueError("only explicit runtime.sh and original TRAIN calibration external files permitted")
    for name,expected in freeze.get("external_files_sha256",{}).items():
        if not Path(name).is_absolute() or sha(name) != expected:
            raise RuntimeError("explicit external runtime/TRAIN calibration changed")
    if not freeze.get("external_files_sha256") or freeze["external_files_sha256"].get(plan["shared_TRAIN_calibration"]["path"]) != plan["shared_TRAIN_calibration"]["sha256"]:
        raise ValueError("exact original TRAIN calibration identity required")
    # Recompute the current worker's own inventory; no model or Native import.
    sys.path.insert(0,str(root))
    from our_mani.local_mpc_20261102.records import freeze_sources
    for job in plan["jobs"]:
        expected=dict(freeze["common_controller_files_sha256"],**{job["config"]:job["config_sha256"]})
        if freeze_sources(path_in(root,job["config"]),"p1") != expected:
            raise RuntimeError("worker source inventory differs from explicit TEST freeze")
        if freeze["evaluation_and_assignment_files_sha256"].get(job["config"]) != job["config_sha256"]:
            raise ValueError("all sixteen configs must be frozen before execution")
    if any(behavior_configuration(cfg)!=freeze.get("behavior_configuration") for cfg in configs):
        raise ValueError("TEST controller settings differ from the exact frozen devVAL behavior")
    check_native_sources(configs[0],freeze.get("native_sources_sha256"))
    return plan,configs,freeze


def parse_time(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timezone-aware actual exit timestamps required")
    return result


def audit_exit(root,job,config,common_sources,native_sources,*,expected_information_setting="public_candidate_transformer_v5_development"):
    """No GT reads. A completed case is auditable irrespective of task success."""
    case=path_in(root,job["path"],FAMILY)
    paths=dict(owner=case.parent/(case.name+".owner.json"),
               actual=case.parent/(case.name+".owner.actual_exit.json"),
               worker=case/"worker_exit.json",result=case/"result.json",inputs=case/"run_inputs.json")
    # Inspect exit-only documents before opening a result that could contain
    # Native labels. A failed/missing actual exit must not unlock GT reads.
    owner,actual,worker=(load(paths[key]) for key in ("owner","actual","worker"))
    argv=["--mode","p1","--scene",job["scene"],"--config",job["config"],"--output",job["path"]]
    if (owner.get("argv") != argv or type(owner.get("child_pid")) is not int or owner["child_pid"]<=0
            or actual.get("output")!=str(case) or actual.get("started_utc")!=owner.get("started_utc")
            or type(actual.get("actual_process_returncode")) is not int or actual["actual_process_returncode"]!=0):
        raise RuntimeError("exact launcher assignment and actual zero returncode required")
    if parse_time(actual["finished_utc"]) < parse_time(owner["started_utc"]):
        raise RuntimeError("actual exit precedes spawn")
    if job.get("split")=="test":
        identity=actual.get("spawned_process",{})
        command=["bash",str(Path(root)/"scripts/run_local_mpc_20261102.sh"),*argv]
        if (actual.get("actual_process_exit_observed") is not True or actual.get("actual_wait_returncode")!=0
                or actual.get("timed_out") is not False or actual.get("administrative_stop") is not False
                or actual.get("supervision_error") is not None or actual.get("spawned_process")!=owner.get("spawned_process")
                or type(identity.get("start_ticks")) is not int or identity.get("pid")!=owner["child_pid"]
                or not isinstance(identity.get("boot_id"),str) or not identity["boot_id"]
                or actual.get("exact_command")!=command or owner.get("exact_command")!=command):
            raise RuntimeError("actual observed new TEST child exit identity required")
    if (type(worker.get("exit_code")) is not int or worker["exit_code"]!=0 or
            worker.get("native_closed") is not True or worker.get("result_recorded") is not True):
        raise RuntimeError("actual worker exit and Native close required")
    if parse_time(worker["finished_utc"]) > parse_time(actual["finished_utc"]):
        raise RuntimeError("worker terminal must precede actual parent wait exit")
    result,inputs=load(paths["result"]),load(paths["inputs"])
    if result.get("complete") is not True or result.get("source_unchanged") is not True:
        raise RuntimeError("unchanged completed result required")
    expected=dict(common_sources,**{job["config"]:sha(path_in(root,job["config"]))})
    if (inputs.get("mode")!="p1" or inputs.get("scene")!=job["scene"] or inputs.get("config")!=config
            or inputs.get("source_sha256")!=expected or inputs.get("native_sources_sha256")!=native_sources
            or inputs.get("TEST_read") is not False or inputs.get("old_experiments_resumed") is not False
            or inputs.get("information_setting")!=expected_information_setting):
        raise RuntimeError("exact actual public-controller/config/source receipt required")
    archives={}
    for name,expected_sha in expected.items():
        original=path_in(root,name)
        if case/"source_snapshot" not in (case/"source_snapshot"/name).resolve().parents:
            raise ValueError("source snapshot traversal")
        snapshot=case/"source_snapshot"/name
        if sha(snapshot)!=expected_sha:
            raise RuntimeError("archived source differs from actual run_inputs: "+name)
        archives[name]=expected_sha
    return dict(actual_exit_verified=True,Native_closed=True,
        receipts={key:dict(path=str(path.relative_to(root)),sha256=sha(path)) for key,path in paths.items()},
        archived_sources_sha256=archives,task_outcome_used_for_admission=False)


def verify_case_checkpoint_chain(root,job,config,common_sources,exit_evidence):
    """Strict all-case transitive model identity; never repair old metadata.

    Exit/source auditing must have completed before opening these archived
    training-model documents. Hashes of archived bytes, not current working
    model/source files, establish what each actual child consumed.
    """
    if exit_evidence.get("actual_exit_verified") is not True or exit_evidence.get("Native_closed") is not True:
        raise ValueError("actual exit and Native close audit must precede checkpoint identity")
    case=path_in(root,job["path"],FAMILY)
    checkpoint=config.get("candidate_transformer_checkpoint")
    manifest=config.get("candidate_model_freeze_manifest")
    if (config.get("candidate_transformer_checkpoint_sha256")!=CHECKPOINT_SHA or
            config.get("Transformer_training_complete") is not True or
            not isinstance(checkpoint,str) or not isinstance(manifest,str) or
            common_sources.get(checkpoint)!=CHECKPOINT_SHA or manifest not in common_sources):
        raise RuntimeError("each config and actual source map must bind the exact completed checkpoint61")
    # path_in rejects parent traversal and aliases outside the project, while
    # only archived case bytes are opened for the model proof below.
    path_in(root,checkpoint);path_in(root,manifest)
    snapshot=case/"source_snapshot"
    if sha(snapshot/checkpoint)!=CHECKPOINT_SHA or sha(snapshot/manifest)!=common_sources[manifest]:
        raise RuntimeError("archived checkpoint or model-freeze identity changed")
    inputs=load(case/"run_inputs.json")
    if (inputs.get("config")!=config or inputs.get("source_sha256",{}).get(checkpoint)!=CHECKPOINT_SHA
            or inputs.get("source_sha256",{}).get(manifest)!=common_sources[manifest]):
        raise RuntimeError("actual run_inputs must independently bind checkpoint61 and the model freeze")
    model=load(snapshot/manifest)
    if (model.get("schema")!="P4_model_freeze_before_newdevVAL_v1" or model.get("training_complete") is not True
            or model.get("fit_split")!="train" or model.get("new_devVAL_used_for_fit") is not False
            or model.get("formal_TEST_used") is not False or model.get("checkpoint")!=checkpoint
            or not isinstance(model.get("files_sha256"),dict) or model["files_sha256"].get(checkpoint)!=CHECKPOINT_SHA):
        raise RuntimeError("archived completed TRAIN-only model freeze must bind exact checkpoint61")
    for name,digest in model["files_sha256"].items():
        path_in(root,name)
        if common_sources.get(name)!=digest or inputs["source_sha256"].get(name)!=digest or sha(snapshot/name)!=digest:
            raise RuntimeError("complete archived model/source/training-evidence identity chain required: "+name)
    for pointer in exit_evidence["receipts"].values():
        if sha(path_in(root,pointer["path"]))!=pointer["sha256"]:
            raise RuntimeError("case metadata changed during checkpoint proof")
    return dict(checkpoint_path=checkpoint,checkpoint_sha256=CHECKPOINT_SHA,
        archived_model_freeze=dict(path=str((snapshot/manifest).relative_to(root)),sha256=common_sources[manifest]),
        model_source_and_training_evidence_files_checked=len(model["files_sha256"]),
        config_run_inputs_archive_and_TRAIN_model_freeze_agree=True,historical_metadata_written=False)


def normalized_v59_devval_jobs(root,val,assigned):
    if (val.get("schema")!=V59_DEVVAL_SCHEMA or val.get("formal_TEST_used") is not False
            or val.get("training_started") is not False or val.get("new_independent_validation") is not False
            or val.get("parents_same_as_prior_devVAL") is not True
            or val.get("all8_kept_regardless_of_outcome_once_gate_passes") is not True
            or len(val.get("jobs",[]))!=8 or len(assigned.get("jobs",[]))!=8):
        raise RuntimeError("exact generic V59 eight-parent assignment declaration required")
    rows=[]
    for raw,original in zip(val["jobs"],assigned["jobs"]):
        cfg=load(path_in(root,raw["config"]));meta=cfg.get("P4_collection_metadata",{})
        episode=original.get("episode_id",original.get("id"))
        if (raw.get("id")!=episode or meta.get("episode_id")!=episode or meta.get("parent_id")!=original["parent_id"]
                or meta.get("split")!="val" or cfg.get("dataset_use")!="val"
                or cfg.get("seed")!=original["seed"] or raw.get("scene")!=original["scene"]
                or meta.get("parent_assigned_before_episode_generation") is not True):
            raise RuntimeError("V59 private role normalization must retain exact preassigned devVAL identity")
        job=dict(raw,parent_id=original["parent_id"],seed=original["seed"],split="val")
        rows.append(job)
    return rows


def verify_devval(root,plan,freeze):
    """Gate on all eight actual exits; never gate on their Native success rates."""
    ref=freeze["development_validation"]
    val=load(checked_pointer(root,ref["plan"]))
    assigned=load(checked_pointer(root,plan["preassigned_partitions"]["devval"]))
    if "checkpoint_sha256" in val and val["checkpoint_sha256"]!=CHECKPOINT_SHA:
        raise RuntimeError("declared top-level checkpoint SHA must be exact61; missing alone requires all8 transitive proof")
    if (ref.get("plan_schema")!=V59_DEVVAL_SCHEMA or val.get("schema")!=ref["plan_schema"]
            or freeze.get("expected_worker_information_setting")!=EXPECTED_INFORMATION):
        raise RuntimeError("explicit same-source V59 generic devVAL plan required")
    normalized_jobs=normalized_v59_devval_jobs(root,val,assigned)
    owner=path_in(root,ref["owner"],FAMILY)
    inputs_path=checked_pointer(root,ref["owner_inputs"]); inputs=load(inputs_path)
    declared=Path(inputs.get("plan",""));declared=declared if declared.is_absolute() else Path(root)/declared
    if (inputs.get("plan_sha256")!=ref["plan"]["sha256"] or declared.resolve()!=path_in(root,ref["plan"]["path"]) or
            inputs.get("runner_sha256")!=ref["runner_sha256"] or inputs_path.parent!=owner):
        raise RuntimeError("student devVAL owner identity chain failed")
    if ref.get("runner"):
        checked_pointer(root,ref["runner"])
        if ref["runner"]["sha256"]!=ref["runner_sha256"]:
            raise RuntimeError("devVAL actual runner identity mismatch")
    rows=[]
    for job,assigned_job in zip(normalized_jobs,assigned["jobs"]):
        cfg=load(path_in(root,job["config"]))
        if (any(cfg.get(k)!=v for k,v in PROTOCOL.items()) or
                behavior_configuration(cfg)!=freeze.get("behavior_configuration")):
            raise RuntimeError("devVAL used a different model/control protocol")
        original=load(path_in(root,assigned_job["config"]))
        if any(cfg.get(key)!=original.get(key) for key in ("seed","initial_target_position","uniform_velocity",
                "speed_change_at_tick","changed_velocity")):
            raise RuntimeError("student devVAL must retain all eight preassigned scene identities")
        evidence=audit_exit(root,job,cfg,freeze["common_controller_files_sha256"],freeze["native_sources_sha256"],
            expected_information_setting=freeze.get("expected_worker_information_setting","public_candidate_transformer_v5_development"))
        model=verify_case_checkpoint_chain(root,job,cfg,freeze["common_controller_files_sha256"],evidence)
        rows.append(dict(id=job["id"],parent_id=job["parent_id"],split="val",evidence=evidence,model_identity=model))
    return dict(complete=True,actual_children_reaped=8,success_rate_admission_gate=False,
                same_controller_and_checkpoint=True,top_level_checkpoint_SHA_present="checkpoint_sha256" in val,
                model_identity_basis="all8 exact config/run_inputs/archive/completed TRAIN-model-freeze chains",
                historical_plan_or_config_written=False,private_generic_V59_role_normalization=True,rows=rows)


def preflight(root,plan_path,freeze_path):
    plan,configs,freeze=validate_freeze(root,plan_path,freeze_path)
    try:
        validation=verify_devval(root,plan,freeze)
    except FileNotFoundError as error:
        validation=dict(complete=False,reason="required devVAL actual-exit or model/source identity evidence is missing",missing=str(error.filename))
    return dict(preflight_passed=True,execution_ready=validation["complete"],Native_started=False,
        training_started=False,new_heldout_TEST_parents=16,split="test",official_TEST_data_read=False,
        official_VAL_data_read=False,original_PhysMani_official_TEST=False,
        common_controller_source_count=len(freeze["common_controller_files_sha256"]),
        per_case_source_count=freeze["per_case_source_count"],
        TEST_plan_sha256=sha(plan_path),controller_freeze_sha256=sha(freeze_path),
        runner_sha256=sha(__file__),development_validation=validation,
        execution_requires_explicit_execute_and_exact_plan_and_freeze_SHA=True)


def process_identity(pid):
    try:
        line=Path("/proc",str(pid),"stat").read_text()
        ticks=int(line[line.rfind(")")+2:].split()[19])
    except (FileNotFoundError,IndexError,ValueError):
        ticks=None
    return dict(pid=pid,start_ticks=ticks,boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def requested_stop(owner,batch):
    # Deliberately do not read old owner/global STOP files.
    return (Path(owner)/"STOP").exists() or (Path(batch)/"STOP").exists()


def propagate_stop(owner,batch,reason):
    for path in (Path(owner)/"STOP",Path(batch)/"STOP"):
        try:
            with path.open("x",encoding="utf-8") as stream:
                stream.write(reason+"\n")
        except FileExistsError:
            pass


def supervise_child(command,*,cwd,owner,batch,case,argv,timeout_seconds,stop_grace_seconds,signal_stop=None):
    """Actually wait/reap once. STOP remains administrative even for rc != 0."""
    case=Path(case);prefix=case.parent/(case.name+".owner")
    started=utc(); wall=time.monotonic(); timed_out=False; administrative=False; escalations=[]
    timeout_marker="heldout infrastructure wall timeout; wait/reap actual child\n"
    supervisor_marker="heldout infrastructure supervisor failure; group termination and actual wait\n"
    supervision_error=None
    def administrative_requested():
        if (Path(owner)/"STOP").exists():return True
        path=Path(batch)/"STOP"
        return path.exists() and path.read_text() not in (timeout_marker,supervisor_marker)
    with Path(str(prefix)+".log").open("x",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        spawned=dict(pid=process.pid,start_ticks=None,boot_id=None)
        stop_started=None
        try:
            spawned=process_identity(process.pid)
            write_new(Path(str(prefix)+".json"),dict(started_utc=started,child_pid=process.pid,
                argv=argv,exact_command=command,spawned_process=spawned,actual_wait_required=True))
            while True:
                if administrative_requested() or (signal_stop is not None and signal_stop()):
                    administrative=True
                    propagate_stop(owner,batch,"heldout owner administrative STOP; wait/reap actual child")
                if not timed_out and time.monotonic()-wall>=timeout_seconds:
                    timed_out=True
                    try:
                        with (Path(batch)/"STOP").open("x",encoding="utf-8") as stream:stream.write(timeout_marker)
                    except FileExistsError:
                        pass
                stopping=administrative or timed_out
                if stopping and stop_started is None:
                    stop_started=time.monotonic()
                if stopping and time.monotonic()-stop_started>=stop_grace_seconds and "SIGTERM" not in escalations:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    escalations.append("SIGTERM")
                if stopping and time.monotonic()-stop_started>=stop_grace_seconds+5 and "SIGKILL" not in escalations:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    escalations.append("SIGKILL")
                try:
                    code=process.wait(timeout=.1);break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            supervision_error=dict(type=type(error).__name__,message=str(error))
            try:
                with (Path(batch)/"STOP").open("x",encoding="utf-8") as stream:stream.write(supervisor_marker)
            except FileExistsError:
                pass
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:code=process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                code=process.wait()
    # STOP arriving exactly as wait completes still has direct administrative evidence.
    administrative=administrative or administrative_requested() or bool(signal_stop and signal_stop())
    receipt=dict(started_utc=started,finished_utc=utc(),actual_process_returncode=code,output=str(case),
        actual_process_exit_observed=True,actual_wait_returncode=code,spawned_process=spawned,
        wall_seconds=time.monotonic()-wall,timed_out=timed_out,administrative_stop=administrative,
        termination_escalations=escalations,exact_command=command,no_retry=True,supervision_error=supervision_error)
    write_new(Path(str(prefix)+".actual_exit.json"),receipt)
    return receipt


def evaluate_after_exit(root,job,config,freeze):
    """Fresh explicit heldout role. Do not weaken the old development evaluator."""
    evidence=audit_exit(root,job,config,freeze["common_controller_files_sha256"],freeze["native_sources_sha256"],
        expected_information_setting=freeze.get("expected_worker_information_setting","public_candidate_transformer_v5_development"))
    if job.get("split")!="test" or config.get("dataset_use")!="test":
        raise ValueError("explicit new heldout TEST evaluation role required")
    # This line is intentionally after the actual-exit/source audit: the only GT reads.
    case=path_in(root,job["path"],FAMILY)
    raw={name:(case/name).read_bytes() for name in ("initial_snapshot.json","native_ticks.jsonl")}
    initial=json.loads(raw["initial_snapshot.json"])
    if (initial.get("native_height_preserved") is not True or
            abs(initial["state"]["object_position"][2]-initial["native_default_target_pose_before_override"][2])>1e-8):
        raise RuntimeError("original Native reset height was not preserved")
    ticks=[json.loads(line) for line in raw["native_ticks.jsonl"].decode().splitlines() if line.strip()]
    sys.path.insert(0,str(root))
    from our_mani.local_mpc_control_20261103.native_task_evaluation import evaluate_states
    metrics,ledger=evaluate_states(initial["state"],ticks)
    if metrics["final"]["time"]>config["simulation_seconds"]+1e-8:
        raise RuntimeError("recorded case exceeded the predeclared 8s physical budget")
    if any((case/name).read_bytes()!=data for name,data in raw.items()):
        raise RuntimeError("post-exit labels changed during evaluation")
    output=path_in(root,job["evaluation"],ARTIFACTS);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema="P4_newheldout_V59_modified_protocol_postexit_metrics_20261103_v3",
        id=job["id"],parent_id=job["parent_id"],split="test",scene=job["scene"],metrics=metrics,
        original_PhysMani_official_TEST=False,multi_task_or_rotation_generalization_claim=False,
        GT_used_only_after_actual_exit=True,GT_used_by_controller=False,task_failures_retained=True,
        original_Native_task_flag_reported_separately=True,strict_current_hold_reported_separately=True,
        controller_public_success_does_not_replace_Native_success=True,formal_benchmark_qualification=False,
        exit_and_source_evidence=evidence,
        labels_sha256={name:hashlib.sha256(data).hexdigest() for name,data in raw.items()})
    write_new(output/"report.json",report)
    with (output/"tick_metrics.jsonl").open("x",encoding="utf-8") as stream:
        for row in ledger:stream.write(json.dumps(row,allow_nan=False)+"\n")
    return report


def summary(rows,planned,complete,termination,actual_exits=None):
    actual_exits=actual_exits or []
    started={r["parent_id"] for r in actual_exits}
    return dict(schema="P4_newheldout_V59_TEST_serial_summary_20261103_v3",complete=complete,
        split="test",termination=termination,planned_parents=len(planned),completed_evaluated_parents=len(rows),
        pending_parent_ids=[j["parent_id"] for j in planned if j["parent_id"] not in {r["parent_id"] for r in rows}],
        actual_exit_outcomes=actual_exits,
        unstarted_parent_ids=[j["parent_id"] for j in planned if j["parent_id"] not in started],
        original_PhysMani_official_TEST=False,modified_control_protocol=True,
        Native_task_success_count=sum(r["metrics"]["native_success_any"] for r in rows),
        strict_current_lifted_hold_count=sum(r["metrics"]["strict_current_joint_condition"]["any"] for r in rows),
        Native_task_success_rate=(sum(r["metrics"]["native_success_any"] for r in rows)/len(planned) if complete else None),
        strict_current_lifted_hold_rate=(sum(r["metrics"]["strict_current_joint_condition"]["any"] for r in rows)/len(planned) if complete else None),
        task_failures_not_retried_or_replaced=True,results=rows)


def execute(root,plan_path,freeze_path,owner_path,*,plan_sha256,freeze_sha256):
    if sha(plan_path)!=plan_sha256 or sha(freeze_path)!=freeze_sha256:
        raise ValueError("explicit exact preflight plan and controller freeze SHA required")
    plan,configs,freeze=validate_freeze(root,plan_path,freeze_path)
    gate=verify_devval(root,plan,freeze)  # required complete eight; no task-success test
    owner=Path(owner_path).resolve();batch=path_in(root,plan["batch_output_parent"],FAMILY)
    if owner!=path_in(root,plan["suggested_owner"],FAMILY) or owner.exists() or batch.exists():
        raise FileExistsError("fresh exact preassigned owner and case namespace required")
    for job in plan["jobs"]:
        case=path_in(root,job["path"],FAMILY)
        if (case.exists() or path_in(root,job["evaluation"],ARTIFACTS).exists() or
                any(case.parent.glob(case.name+".owner*"))):
            raise FileExistsError("existing cases/receipts/artifacts may not be overwritten or resumed")
    batch.mkdir(parents=True,exist_ok=False);owner.mkdir(parents=True,exist_ok=False)
    stopped=[False];previous_handlers={}
    def mark_stop(signum,frame):
        stopped[0]=True
    for signum in (signal.SIGTERM,signal.SIGINT):
        previous_handlers[signum]=signal.signal(signum,mark_stop)
    rows=[];actual_exits=[];termination="InfrastructureFailure";returncode=1
    with (owner/"owner.lock").open("x") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_new(owner/"inputs.json",dict(started_utc=utc(),owner_process=process_identity(os.getpid()),
            plan_path=str(Path(plan_path).resolve()),plan_sha256=plan_sha256,
            controller_freeze_path=str(Path(freeze_path).resolve()),controller_freeze_sha256=freeze_sha256,
            runner_sha256=sha(__file__),complete_devVAL_actual_exit_gate=gate,
            all_TEST_scenarios_preassigned=True,estimated_native_workers=1,no_automatic_retry=True,
            physical_budget_seconds_per_case=8.,runtime_python="resolved by frozen runtime.sh PATH; no training sys.executable override"))
        try:
            for job,cfg in zip(plan["jobs"],configs):
                if stopped[0] or requested_stop(owner,batch):
                    propagate_stop(owner,batch,"new heldout owner STOP before next case")
                    termination="AdministrativeSTOP";returncode=2;break
                if sha(plan_path)!=plan_sha256 or sha(freeze_path)!=freeze_sha256:
                    raise RuntimeError("TEST assignment/freeze changed")
                validate_freeze(root,plan_path,freeze_path)
                if stopped[0] or requested_stop(owner,batch):
                    propagate_stop(owner,batch,"new heldout owner STOP during pre-spawn source audit")
                    termination="AdministrativeSTOP";returncode=2;break
                argv=["--mode","p1","--scene",job["scene"],"--config",job["config"],"--output",job["path"]]
                command=["bash",str(Path(root)/"scripts/run_local_mpc_20261102.sh"),*argv]
                with (owner/"journal.jsonl").open("a",encoding="utf-8") as journal:
                    journal.write(json.dumps(dict(event="spawn_requested",time=utc(),id=job["id"],exact_command=command))+"\n")
                actual=supervise_child(command,cwd=root,owner=owner,batch=batch,
                    case=path_in(root,job["path"]),argv=argv,timeout_seconds=plan["infra_case_wall_timeout_seconds"],
                    stop_grace_seconds=plan["STOP_grace_seconds"],signal_stop=lambda:stopped[0])
                with (owner/"journal.jsonl").open("a",encoding="utf-8") as journal:
                    journal.write(json.dumps(dict(event="actual_wait_complete",time=utc(),id=job["id"],actual=actual))+"\n")
                actual_exits.append(dict(id=job["id"],parent_id=job["parent_id"],actual_exit=actual,
                    task_failure_inferred_from_nonzero_exit=False))
                if actual["administrative_stop"]:
                    termination="AdministrativeSTOP";returncode=2;break
                if actual["timed_out"] or actual.get("supervision_error") or actual["actual_process_returncode"]!=0:
                    # Worker may honor an existing legacy global STOP. Inspect only
                    # this newly produced terminal, never read or clear the old STOP.
                    failure_path=path_in(root,job["path"])/"failure.json"
                    failure=load(failure_path) if failure_path.is_file() else {}
                    if failure.get("type")=="InterruptedError" and "STOP" in str(failure.get("message","")):
                        termination="AdministrativeSTOP";returncode=2
                    else:termination="InfrastructureFailure";returncode=1
                    break
                validate_freeze(root,plan_path,freeze_path)
                if stopped[0] or requested_stop(owner,batch):
                    propagate_stop(owner,batch,"new heldout owner STOP before post-exit evaluation")
                    termination="AdministrativeSTOP";returncode=2;break
                row=evaluate_after_exit(root,job,cfg,freeze)
                write_new(owner/(job["id"]+".json"),row);rows.append(row)
                print(json.dumps(dict(event="evaluated_after_actual_exit",id=job["id"],
                    Native_task_success=row["metrics"]["native_success_any"],
                    strict_current_lifted_hold=row["metrics"]["strict_current_joint_condition"]["any"])),flush=True)
            else:termination="Complete";returncode=0
        except BaseException as error:
            if stopped[0] or requested_stop(owner,batch):
                termination="AdministrativeSTOP";returncode=2
            write_new(owner/"owner_failure.json",dict(type=type(error).__name__,message=str(error),
                termination=termination,finished_utc=utc()))
        finally:
            write_new(owner/"summary.json",summary(rows,plan["jobs"],termination=="Complete",termination,actual_exits))
            write_new(owner/"owner_terminal.json",dict(finished_utc=utc(),returncode=returncode,
                termination=termination,all_spawned_children_actually_waited=True,owner_process=process_identity(os.getpid())))
            for signum,handler in previous_handlers.items():signal.signal(signum,handler)
    return returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--controller-freeze",type=Path,required=True)
    parser.add_argument("--owner",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--plan-sha256");parser.add_argument("--freeze-sha256")
    args=parser.parse_args()
    if args.execute:
        if not all((args.owner,args.plan_sha256,args.freeze_sha256)):
            parser.error("--execute requires --owner, --plan-sha256 and --freeze-sha256")
        return execute(ROOT,args.plan,args.controller_freeze,args.owner,
                       plan_sha256=args.plan_sha256,freeze_sha256=args.freeze_sha256)
    print(json.dumps(preflight(ROOT,args.plan,args.controller_freeze),ensure_ascii=False,indent=2));return 0


if __name__=="__main__":
    raise SystemExit(main())
