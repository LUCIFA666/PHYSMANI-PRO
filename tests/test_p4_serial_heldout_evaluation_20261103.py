"""CPU contracts for new-heldout partitioning, real exits and serial admission."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


runner=imported("heldout_runner_tests",ROOT/"scripts/run_p4_serial_heldout_evaluation_20261103.py")
prep=imported("heldout_preparer_tests",ROOT/"scripts/prepare_p4_new_heldout_test_20261103.py")


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        runner.write_new(path,value);return path

    def pointer(self,path):
        return dict(path=str(path.relative_to(self.root)),sha256=runner.sha(path))

    def case(self,index=0,*,split="val"):
        cfg=dict(runner.PROTOCOL,seed=100+index,dataset_use=split)
        cfg_path=self.put(f"config/framework/val/{index}.json",cfg)
        source=self.root/"source.py"
        if not source.exists():source.write_text("source = 1\n")
        common={"source.py":runner.sha(source)}
        job=dict(id=str(index),parent_id=f"parent{index}",split=split,seed=100+index,scene="static",
            config=str(cfg_path.relative_to(self.root)),path=f"{runner.FAMILY}/val/{index}")
        case=self.root/job["path"];case.mkdir(parents=True)
        start="2026-11-03T00:00:00+00:00";finish="2026-11-03T00:00:02+00:00"
        argv=["--mode","p1","--scene","static","--config",job["config"],"--output",job["path"]]
        identity=dict(pid=123,start_ticks=123456,boot_id="cpu-fixture")
        command=["bash",str(self.root/"scripts/run_local_mpc_20261102.sh"),*argv]
        self.put(str(case.relative_to(self.root))+".owner.json",dict(started_utc=start,child_pid=123,argv=argv,spawned_process=identity,exact_command=command))
        self.put(str(case.relative_to(self.root))+".owner.actual_exit.json",dict(started_utc=start,finished_utc=finish,
            actual_process_returncode=0,output=str(case),actual_process_exit_observed=True,actual_wait_returncode=0,
            timed_out=False,administrative_stop=False,supervision_error=None,spawned_process=identity,exact_command=command))
        self.put(str(case.relative_to(self.root))+"/worker_exit.json",dict(finished_utc="2026-11-03T00:00:01+00:00",
            exit_code=0,result_recorded=True,native_closed=True))
        self.put(str(case.relative_to(self.root))+"/result.json",dict(complete=True,source_unchanged=True,
            milestone_passed=False,Native_task_success_claimed=False))
        sources=dict(common,**{job["config"]:runner.sha(cfg_path)})
        self.put(str(case.relative_to(self.root))+"/run_inputs.json",dict(mode="p1",scene="static",config=cfg,
            source_sha256=sources,native_sources_sha256={"native.py":"a"*64},TEST_read=False,
            old_experiments_resumed=False,information_setting="public_candidate_transformer_v5_development"))
        for name in sources:
            target=case/"source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((self.root/name).read_bytes())
        return job,cfg,common,{"native.py":"a"*64}

    def replace(self,path,**updates):
        value=runner.load(path);value.update(updates);path.write_text(json.dumps(value));return value

    def test_preassigned_16_schedule_disjoint_without_any_results(self):
        self.assertEqual(len(prep.SCHEDULE),16)
        self.assertEqual([x[0] for x in prep.SCHEDULE].count("static"),2)
        self.assertEqual([x[0] for x in prep.SCHEDULE].count("uniform"),8)
        self.assertEqual([x[0] for x in prep.SCHEDULE].count("speed_change"),6)
        points=[x[1] for x in prep.SCHEDULE];self.assertEqual(len(set(points)),16)
        existing=set()
        for name in (prep.TRAIN,prep.DEVVAL):
            for job in runner.load(ROOT/name)["jobs"]:
                existing.add(runner.xy(runner.load(ROOT/job["config"])))
        self.assertFalse(set(points)&existing)
        self.assertTrue(all(.05<=x<=.35 and -.45<=y<=.45 for x,y in points))

    def generated_plan_fixture(self):
        jobs=copy.deepcopy(prep.fixed_jobs())
        calibration="/opaque/train_package_compressed/calibration/data.h5"
        for job in jobs:
            config=dict(runner.PROTOCOL,seed=job["seed"],dataset_use="test",task="pick_moving_target_on_the_table",
                calibration_train_episode=calibration,initial_target_position=[*job["initial_xy"],.775],
                uniform_velocity=job["initial_velocity_m_s"],changed_velocity=job["changed_velocity_m_s"],
                speed_change_at_tick=job["speed_change_at_tick"],
                P4_collection_metadata=dict(parent_id=job["parent_id"],split="test",episode_id=job["id"],
                    parent_assigned_before_episode_generation=True))
            path=self.put(f"config/framework/new/{job['id']}.json",config)
            job.update(config=str(path.relative_to(self.root)),config_sha256=runner.sha(path),
                path=f"{runner.FAMILY}/new/{job['id']}",evaluation=f"{runner.ARTIFACTS}/new/{job['id']}")
        partition=self.put("config/framework/partition.json",dict(schema="P4_new_heldout_TEST_partition_20261103_v1",
            policy_independent_fixed_partition=True,jobs=prep.fixed_jobs()))
        path=self.put("config/framework/new/plan.json",dict(schema=runner.PLAN_SCHEMA,split="test",
            official_VAL_data_read=False,official_TEST_data_read=False,official_PhysMani_TEST=False,
            multi_task_or_rotation_generalization_claim=False,all_splits_assigned_before_TEST_generation=True,
            selection_from_TEST_results=False,preassigned_partitions={},batch_output_parent=f"{runner.FAMILY}/new",
            evaluation_output_parent=f"{runner.ARTIFACTS}/new",shared_TRAIN_calibration=dict(path=calibration),
            heldout_partition=self.pointer(partition),
            control={k:runner.PROTOCOL[k] for k in ("physics_dt","control_dt","decision_ticks","horizon_steps",
                "num_candidates","cem_sequences_total","cem_iteration_sequences","simulation_seconds")},jobs=jobs))
        return path,jobs

    def test_config_cannot_mutate_fixed_partition_when_policy_changes(self):
        path,jobs=self.generated_plan_fixture()
        with mock.patch.object(runner,"partitions",side_effect=lambda *args:(set(),set(),set(),{})):
            plan,configs=runner.validate_plan(self.root,path)
            self.assertEqual(len(configs),16)
            config_path=self.root/jobs[0]["config"]
            self.replace(config_path,initial_target_position=[.18,-.18,.775])
            plan["jobs"][0]["config_sha256"]=runner.sha(config_path)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,"policy-independent preassigned scene"):
                runner.validate_plan(self.root,path)

    def test_policy_revision_changes_behavior_not_partition_identity(self):
        base=dict(runner.PROTOCOL,active_visual_settings=dict(probe_seconds=.8,attached_absolute_orientation_gate=True))
        revised=copy.deepcopy(base);revised["active_visual_settings"]["attached_absolute_orientation_gate"]=False
        self.assertNotEqual(runner.behavior_configuration(base),runner.behavior_configuration(revised))
        self.assertEqual(prep.fixed_jobs(),runner.load(ROOT/prep.PARTITION)["jobs"])

    def test_task_failure_is_valid_exit_and_no_GT_opened(self):
        job,cfg,common,native=self.case()
        # No initial_snapshot/native_ticks exist. Exit admission cannot need GT.
        evidence=runner.audit_exit(self.root,job,cfg,common,native)
        self.assertTrue(evidence["actual_exit_verified"])
        self.assertFalse(evidence["task_outcome_used_for_admission"])

    def test_snapshot_tamper_fails_even_when_current_source_matches(self):
        job,cfg,common,native=self.case();case=self.root/job["path"]
        (case/"source_snapshot/source.py").write_text("source = 2\n")
        with self.assertRaisesRegex(RuntimeError,"archived source"):
            runner.audit_exit(self.root,job,cfg,common,native)

    def test_native_close_or_actual_returncode_required(self):
        job,cfg,common,native=self.case();case=self.root/job["path"]
        self.replace(case/"worker_exit.json",native_closed=False)
        with self.assertRaises(RuntimeError):runner.audit_exit(self.root,job,cfg,common,native)
        self.replace(case/"worker_exit.json",native_closed=True)
        self.replace(Path(str(case)+".owner.actual_exit.json"),actual_process_returncode=True)
        with self.assertRaises(RuntimeError):runner.audit_exit(self.root,job,cfg,common,native)

    def test_wrong_case_argv_and_unobserved_TEST_exit_fail(self):
        job,cfg,common,native=self.case(split="test");case=self.root/job["path"]
        path=Path(str(case)+".owner.actual_exit.json")
        self.replace(path,actual_process_exit_observed=False)
        with self.assertRaisesRegex(RuntimeError,"observed new TEST"):
            runner.audit_exit(self.root,job,cfg,common,native)
        self.replace(path,actual_process_exit_observed=True)
        self.replace(Path(str(case)+".owner.json"),argv=["--mode","p0"])
        with self.assertRaises(RuntimeError):runner.audit_exit(self.root,job,cfg,common,native)

    def test_missing_exit_cannot_open_truth(self):
        job,cfg,common,native=self.case(split="test");case=self.root/job["path"]
        Path(str(case)+".owner.actual_exit.json").unlink()
        with mock.patch.object(Path,"read_bytes",side_effect=AssertionError("GT read before exit")):
            with self.assertRaises(FileNotFoundError):
                runner.evaluate_after_exit(self.root,job,cfg,dict(common_controller_files_sha256=common,native_sources_sha256=native))

    def test_nonzero_exit_cannot_even_open_result_labels(self):
        job,cfg,common,native=self.case();case=self.root/job["path"]
        self.replace(Path(str(case)+".owner.actual_exit.json"),actual_process_returncode=1)
        original=runner.load
        def safe_load(path):
            if Path(path).name=="result.json":raise AssertionError("result labels unlocked before actual rc check")
            return original(path)
        with mock.patch.object(runner,"load",side_effect=safe_load):
            with self.assertRaises(RuntimeError):runner.audit_exit(self.root,job,cfg,common,native)

    def test_completed_heldout_metrics_keep_native_and_strict_hold_separate(self):
        job,cfg,common,native=self.case(split="test");case=self.root/job["path"]
        job["evaluation"]=f"{runner.ARTIFACTS}/heldout/0"
        initial=dict(time=0.,counted_tick=0,object_position=[0.,0.,.775],tcp_position=[0.,0.,1.],
            tcp_quaternion=[0.,0.,0.,1.],held=False,native_success=False)
        self.put(str(case.relative_to(self.root))+"/initial_snapshot.json",dict(state=initial,
            native_height_preserved=True,native_default_target_pose_before_override=[0.,0.,.775]))
        tick=dict(initial,time=.05,counted_tick=1,native_success=True)
        (case/"native_ticks.jsonl").write_text(json.dumps(tick)+"\n")
        report=runner.evaluate_after_exit(self.root,job,cfg,dict(common_controller_files_sha256=common,native_sources_sha256=native))
        self.assertTrue(report["metrics"]["native_success_any"])
        self.assertFalse(report["metrics"]["strict_current_joint_condition"]["any"])
        self.assertTrue(report["GT_used_only_after_actual_exit"])
        self.assertFalse(report["formal_benchmark_qualification"])

    def devval_fixture(self):
        fixtures=[self.case(i) for i in range(8)];jobs=[x[0] for x in fixtures]
        val=self.put("config/framework/student_plan.json",dict(schema="P4_student_public_closedloop_newdevVAL_v51_preassigned_parents",
            split="val",checkpoint_sha256=runner.CHECKPOINT_SHA,weights_frozen=True,jobs=jobs))
        assigned=self.put("config/framework/assigned.json",dict(jobs=jobs))
        owner=f"{runner.FAMILY}/val_owner"
        inp=self.put(owner+"/inputs.json",dict(plan=str(val.relative_to(self.root)),plan_sha256=runner.sha(val),runner_sha256="b"*64))
        plan=dict(preassigned_partitions=dict(devval=self.pointer(assigned)))
        freeze=dict(development_validation=dict(plan=self.pointer(val),plan_schema="P4_student_public_closedloop_newdevVAL_v51_preassigned_parents",
            owner=owner,owner_inputs=self.pointer(inp),runner_sha256="b"*64),
            common_controller_files_sha256=fixtures[0][2],native_sources_sha256=fixtures[0][3],
            behavior_configuration=runner.behavior_configuration(fixtures[0][1]))
        return plan,freeze,jobs

    def test_devval_requires_all_eight_actual_exits_no_success_gate(self):
        plan,freeze,jobs=self.devval_fixture()
        gate=runner.verify_devval(self.root,plan,freeze)
        self.assertEqual(gate["actual_children_reaped"],8)
        self.assertFalse(gate["success_rate_admission_gate"])
        Path(str(self.root/jobs[-1]["path"])+".owner.actual_exit.json").unlink()
        with self.assertRaises(FileNotFoundError):runner.verify_devval(self.root,plan,freeze)

    def test_devval_model_or_threshold_change_rejected(self):
        plan,freeze,jobs=self.devval_fixture()
        path=self.root/jobs[3]["config"]
        self.replace(path,gripper_velocity=.3)
        with self.assertRaisesRegex(RuntimeError,"different model/control"):
            runner.verify_devval(self.root,plan,freeze)

    def test_default_preflight_never_spawns_and_reports_missing_gate(self):
        planfile=self.put("config/framework/plan.json",{})
        freezefile=self.put("config/framework/freeze.json",{})
        with mock.patch.object(runner,"validate_freeze",return_value=({},[],dict(common_controller_files_sha256={},per_case_source_count=90))), \
                mock.patch.object(runner,"verify_devval",side_effect=FileNotFoundError(2,"missing","actual_exit")), \
                mock.patch.object(subprocess,"Popen",side_effect=AssertionError("preflight launched")):
            result=runner.preflight(self.root,planfile,freezefile)
        self.assertFalse(result["execution_ready"]);self.assertFalse(result["Native_started"])
        self.assertFalse((self.root/runner.FAMILY).exists())

    def test_STOP_actual_nonzero_is_administrative_and_reaped(self):
        owner=self.root/"owner";batch=self.root/"batch";owner.mkdir();batch.mkdir()
        case=batch/"case"
        timer=threading.Timer(.1,lambda:(owner/"STOP").write_text("user stop\n"))
        timer.start();self.addCleanup(timer.join)
        actual=runner.supervise_child([sys.executable,"-B","-c","import time;time.sleep(.3);raise SystemExit(7)"],
            cwd=self.root,owner=owner,batch=batch,case=case,argv=[],timeout_seconds=10,stop_grace_seconds=1)
        self.assertTrue(actual["actual_process_exit_observed"])
        self.assertTrue(actual["administrative_stop"]);self.assertFalse(actual["timed_out"])
        self.assertEqual(actual["actual_wait_returncode"],7)
        self.assertTrue((batch/"STOP").exists())
        self.assertFalse(Path("/proc",str(actual["spawned_process"]["pid"])).exists())

    def test_timeout_STOP_does_not_misclassify_as_administrative(self):
        owner=self.root/"owner";batch=self.root/"batch";owner.mkdir();batch.mkdir()
        actual=runner.supervise_child([sys.executable,"-B","-c","import time;time.sleep(.25)"],
            cwd=self.root,owner=owner,batch=batch,case=batch/"case",argv=[],timeout_seconds=.05,stop_grace_seconds=1)
        self.assertTrue(actual["timed_out"]);self.assertFalse(actual["administrative_stop"])
        self.assertTrue(actual["actual_process_exit_observed"])

    def test_unstopped_exception_is_not_native_task_false(self):
        owner=self.root/"owner";batch=self.root/"batch";owner.mkdir();batch.mkdir()
        actual=runner.supervise_child([sys.executable,"-B","-c","raise RuntimeError('cpu synthetic infrastructure')"],
            cwd=self.root,owner=owner,batch=batch,case=batch/"case",argv=[],timeout_seconds=10,stop_grace_seconds=1)
        self.assertFalse(actual["administrative_stop"]);self.assertNotEqual(actual["actual_wait_returncode"],0)

    def test_supervisor_error_still_reaps_and_records_actual_child_rc(self):
        owner=self.root/"owner";batch=self.root/"batch";owner.mkdir();batch.mkdir()
        real_write=runner.write_new
        def failing_once(path,value):
            if str(path).endswith(".owner.json"):raise OSError("synthetic spawn receipt failure")
            return real_write(path,value)
        with mock.patch.object(runner,"write_new",side_effect=failing_once):
            actual=runner.supervise_child([sys.executable,"-B","-c","import time;time.sleep(5)"],
                cwd=self.root,owner=owner,batch=batch,case=batch/"case",argv=[],timeout_seconds=10,stop_grace_seconds=1)
        self.assertTrue(actual["actual_process_exit_observed"])
        self.assertFalse(actual["administrative_stop"])
        self.assertEqual(actual["supervision_error"]["type"],"OSError")
        self.assertIsInstance(actual["actual_process_returncode"],int)
        self.assertFalse(Path("/proc",str(actual["spawned_process"]["pid"])).exists())

    def test_all_sixteen_task_failures_still_run_fixed_full_partition(self):
        batch=f"{runner.FAMILY}/test_batch";owner=self.root/(batch+"_owner")
        jobs=[dict(id=str(i),parent_id=f"p{i}",scene="static",config=f"config/{i}.json",path=f"{batch}/{i}",
                   evaluation=f"{runner.ARTIFACTS}/test/{i}") for i in range(16)]
        plan=dict(batch_output_parent=batch,suggested_owner=batch+"_owner",jobs=jobs,infra_case_wall_timeout_seconds=5,STOP_grace_seconds=1)
        planpath=self.put("config/framework/plan.json",{});freezepath=self.put("config/framework/freeze.json",{})
        called=[]
        def child(*args,**kwargs):
            called.append(kwargs["case"])
            return dict(administrative_stop=False,timed_out=False,actual_process_returncode=0)
        def metric(root,job,*args):
            return dict(id=job["id"],parent_id=job["parent_id"],metrics=dict(native_success_any=False,strict_current_joint_condition=dict(any=False)))
        with mock.patch.object(runner,"validate_freeze",return_value=(plan,[{}]*16,{})), \
                mock.patch.object(runner,"verify_devval",return_value=dict(complete=True)), \
                mock.patch.object(runner,"supervise_child",side_effect=child), \
                mock.patch.object(runner,"evaluate_after_exit",side_effect=metric):
            code=runner.execute(self.root,planpath,freezepath,owner,
                plan_sha256=runner.sha(planpath),freeze_sha256=runner.sha(freezepath))
        result=runner.load(owner/"summary.json")
        self.assertEqual(code,0);self.assertEqual(len(called),16);self.assertTrue(result["complete"])
        self.assertEqual(result["Native_task_success_rate"],0.)
        self.assertEqual(result["planned_parents"],16);self.assertFalse(result["pending_parent_ids"])

    def test_partial_summary_has_no_misleading_fixed_denominator_rate(self):
        rows=[dict(parent_id="a",metrics=dict(native_success_any=False,strict_current_joint_condition=dict(any=False)))]
        result=runner.summary(rows,[dict(parent_id="a"),dict(parent_id="b")],False,"AdministrativeSTOP")
        self.assertIsNone(result["Native_task_success_rate"])
        self.assertEqual(result["pending_parent_ids"],["b"])

    def test_paths_cannot_target_old_owner_or_alias_outside_project(self):
        with self.assertRaises(ValueError):runner.path_in(self.root,"../official_TEST")
        with self.assertRaises(ValueError):runner.path_in(self.root,"/absolute/old_STOP")
        (self.root/"alias").symlink_to("/tmp")
        with self.assertRaises(ValueError):runner.path_in(self.root,"alias/outside")


if __name__=="__main__":unittest.main()
