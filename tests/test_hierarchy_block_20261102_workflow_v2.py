import json
from pathlib import Path
import tempfile
import unittest
import sys
import time
import threading
from unittest.mock import patch

from our_mani.hierarchy_block_20261102.workflow_v2 import (
    ARTIFACT_FAMILY, RUNTIME_FAMILY, TASKS, Workflow, WorkflowFailed,
    atomic_json, pointer, process_identity, qualify_parallel, validate_training_exit, WorkflowBlocked,
)


class FakeBackend:
    def __init__(self, root, *, incomplete_val=False, stop_first=False):
        self.root = root
        self.incomplete_val, self.stop_first = incomplete_val, stop_first
        self.calls, self.plans, self.freeze_ref = [], {}, None
        (root / "source.py").write_text("# fixed mock dependency\n")

    def source_files(self):
        return ("source.py",)

    def dependencies(self):
        return dict(ready=True, missing=[])

    def training_command(self, mode, runtime):
        self.calls.append("training_" + mode)
        return ["unit-test-only", mode]

    def build_freeze(self):
        self.calls.append("freeze")
        path = self.root / "freeze.json"
        atomic_json(path, {"frozen": True, "precision_qualified": False})
        self.freeze_ref = pointer(self.root, path)
        return path

    def assert_freeze(self, path):
        return pointer(self.root, path)

    def prepare(self, split, freeze_path, prior_release):
        self.calls.append("prepare_" + split)
        if split == "test":
            self.calls.append("TEST_source_reader")
            if prior_release is None or not json.loads(prior_release.read_text())["complete"]:
                raise AssertionError("TEST source read before complete VAL release")
        counts = [4] * 2 if split == "train" else [20, 20] if split == "validation" else [40] * 2
        cases = [dict(case_key=f"{task}__episode{i}", task=task, raw_episode_id=f"episode{i}")
                 for task, count in zip(TASKS, counts) for i in range(count)]
        plan = dict(expected_cases=sum(counts), cases=cases, freeze=self.freeze_ref, native_execution_ready=True)
        if split == "test":
            plan.update(complete_Native_VAL_observed_before_TEST_read=True,
                        actual_VAL_exits_revalidated=40, TEST_used_for_fitting_or_selection=False)
        path = self.root / (split + "_plan.json")
        atomic_json(path, plan); self.plans[split] = plan
        return plan, path

    def preflight(self, split, *unused):
        self.calls.append("preflight_" + split)
        return dict(native_started=False)

    def launch(self, split, plan_path, case_key, *unused, **resource):
        self.calls.append(("case", split, case_key))
        if self.stop_first:
            return dict(category="AdministrativeSTOP", actual_process_exit_observed=True,
                        case_complete=False, returncode=2, timed_out=False)
        return dict(category="NativeFalse", actual_process_exit_observed=True,
                    case_complete=True, returncode=0, timed_out=False, case_key=case_key,
                    resource_slot=resource.get("resource_slot", 0), parallel=resource.get("parallel", False))

    def case_resource_evidence(self, receipt):
        key = receipt.get("case_key", "missing")
        return dict(schema="hierarchy_block_native_isolation_20261102_v1", complete=True, restored=True,
                    errors=[], parallel=receipt.get("parallel", False), slot=receipt.get("resource_slot",0),
                    display=":"+key, network_namespace="net-"+key, case_runtime="runtime/"+key,
                    case_output="output/"+key, private_paths={name:name+key for name in ("TMPDIR","XDG_CACHE_HOME","XDG_RUNTIME_DIR")},
                    process={"pid":key,"start_ticks":1,"boot_id":"test"},
                    native_interval={"start_monotonic": 1.,"end_monotonic":2.},
                    listener_audit_complete=True, netnamespace_isolated=True, all_threads_within_assignment=True,
                    assigned_cpus=list(range(receipt.get("resource_slot",0)*8,(receipt.get("resource_slot",0)+1)*8)))

    def native_exit_evidence(self, receipt):
        return dict(command=["unit-test-only", "Native"], actual_returncode=receipt["returncode"])

    def audit(self, split, *unused):
        self.calls.append("audit_" + split)
        plan = self.plans[split]
        release = dict(complete=not (split == "validation" and self.incomplete_val),
                       freeze=self.freeze_ref, expected_cases=plan["expected_cases"],
                       actual_process_exits_audited=plan["expected_cases"], cases=plan["cases"])
        path = self.root / (split + "_release.json")
        atomic_json(path, release)
        return release, path

    def request_stop(self, split, reason):
        self.calls.append(("stop", split))


class WorkflowTests(unittest.TestCase):
    def build(self, root, **kwargs):
        backend = FakeBackend(root, **kwargs)
        owner = Workflow(root, backend=backend, command_runner=lambda *args, **unused: dict(
            actual_process_exit_observed=True, actual_returncode=0, administrative_stop=False, timed_out=False))
        return owner, backend

    def test_stage_order_and_nativefalse_continue_without_retry_or_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root)
            result = owner.run()
            self.assertTrue(result["complete"])
            self.assertLess(backend.calls.index("training_CPU"), backend.calls.index("training_GPU"))
            self.assertLess(backend.calls.index("training_GPU"), backend.calls.index("freeze"))
            self.assertLess(backend.calls.index("audit_train"), backend.calls.index("prepare_validation"))
            self.assertLess(backend.calls.index("audit_validation"), backend.calls.index("TEST_source_reader"))
            cases = [row for row in backend.calls if isinstance(row, tuple) and row[0] == "case"]
            self.assertEqual(len(cases), 8 + 40 + 80)
            self.assertEqual(len(set(cases)), len(cases))
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v2/summary.json").read_text())
            self.assertEqual(summary["estimated_native_workers"], 1)
            self.assertEqual(summary["splits"]["validation"]["outcomes"]["NativeFalse"], 40)
            self.assertEqual(summary["splits"]["test"]["outcomes"]["NativeFalse"], 80)
            self.assertFalse(summary["automatic_retry"])
            self.assertFalse(summary["precision_qualified"])
            self.assertFalse(summary["native_success_is_precision_qualification"])
            self.assertTrue(all(value["native_success_rate"] == 0.
                for value in summary["splits"]["test"]["per_task"].values()))

    def test_incomplete_val_release_never_calls_test_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend = self.build(Path(temporary), incomplete_val=True)
            result = owner.run()
            self.assertEqual(result["stage"], "blocked")
            self.assertNotIn("TEST_source_reader", backend.calls)
            self.assertNotIn("prepare_test", backend.calls)

    def test_administrative_stop_is_not_nativefalse_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root, stop_first=True)
            result = owner.run()
            self.assertEqual(result["stage"], "administrative_stop")
            cases = [row for row in backend.calls if isinstance(row, tuple) and row[0] == "case"]
            self.assertEqual(len(cases), 1)
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v2/summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["outcomes"]["AdministrativeSTOP"], 1)
            self.assertEqual(summary["splits"]["train"]["outcomes"]["NativeFalse"], 0)
            self.assertNotIn("TEST_source_reader", backend.calls)

    def test_namespace_stop_prevents_following_gpu_and_native_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root)
            def runner(command, **unused):
                owner.stop_path.write_text("explicit test administrative stop\n")
                return dict(actual_process_exit_observed=True, actual_returncode=0,
                            administrative_stop=False, timed_out=False)
            owner.command_runner = runner
            result = owner.run()
            self.assertEqual(result["stage"], "administrative_stop")
            self.assertNotIn("training_GPU", backend.calls)
            self.assertNotIn("prepare_train", backend.calls)
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v2/summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["attempted"], 0)
            self.assertEqual(summary["splits"]["train"]["outcomes"]["NativeFalse"], 0)

    def test_actual_cpu_subprocess_exit_and_command_are_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner = Workflow(root)
            owner.runtime.mkdir(parents=True)
            owner.status = {"process": process_identity()}
            command = [sys.executable, "-c", "import sys; sys.exit(7)"]
            result = owner.run_subprocess(command, stage="CPU_receipt_test")
            self.assertEqual(result["actual_returncode"], 7)
            self.assertTrue(result["actual_process_exit_observed"])
            self.assertEqual(result["command"], command)
            self.assertGreaterEqual(result["wall_seconds"], 0.)
            self.assertFalse(result["administrative_stop"])

    def test_missing_dependency_blocks_before_wait_or_any_child_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend = self.build(Path(temporary))
            backend.dependencies = lambda: dict(ready=False, missing=["required_TEST_extension.py"])
            result = owner.run()
            self.assertEqual(result["stage"], "blocked")
            self.assertEqual(backend.calls, [])

    def test_failed_training_or_missing_checkpoint_cannot_start_native(self):
        for failure in ("training", "freeze"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                owner, backend = self.build(Path(temporary))
                if failure == "training":
                    owner.command_runner = lambda *args, **unused: dict(actual_process_exit_observed=True,
                        actual_returncode=1, administrative_stop=False, timed_out=False)
                else:
                    from our_mani.hierarchy_block_20261102.workflow_v2 import WorkflowBlocked
                    backend.build_freeze = lambda: (_ for _ in ()).throw(WorkflowBlocked("no trained checkpoint"))
                result = owner.run()
                self.assertFalse(result["complete"])
                self.assertNotIn("prepare_train", backend.calls)
                self.assertFalse(any(isinstance(row, tuple) and row[0] == "case" for row in backend.calls))

    def test_nonzero_actual_case_exit_is_infrastructure_not_nativefalse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, backend = self.build(root)
            backend.launch = lambda *args, **kwargs: dict(category="NativeFalse",
                actual_process_exit_observed=True, case_complete=False, returncode=1, timed_out=False)
            result = owner.run()
            self.assertEqual(result["stage"], "failed")
            summary = json.loads((root / ARTIFACT_FAMILY / "workflow_v2/summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["outcomes"]["InfrastructureFailure"], 1)
            self.assertEqual(summary["splits"]["train"]["outcomes"]["NativeFalse"], 0)

    def test_parallel_two_then_four_qualifies_without_using_native_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); owner, backend = self.build(root)
            owner.requested_native_workers = 4
            launch = backend.launch
            guard, active = threading.Lock(), [0,0]
            def measured(*args, **kwargs):
                with guard:
                    active[0] += 1; active[1] = max(active[1],active[0])
                try:
                    time.sleep(.003)
                    return launch(*args,**kwargs)
                finally:
                    with guard: active[0] -= 1
            backend.launch = measured
            result = owner.run()
            self.assertTrue(result["complete"])
            self.assertEqual(active[1], 4)
            self.assertEqual(owner.proven_native_workers, 4)
            records = [r for r in backend.calls if isinstance(r,tuple) and r[0]=="case"]
            self.assertEqual(len(records),128)
            self.assertEqual(len(set(records)),128)
            self.assertEqual(len(owner.parallel_qualifications),2)
            for ref in owner.parallel_qualifications:
                self.assertFalse(json.loads((root/ref["path"]).read_text())["task_success_rate_used"])

    def test_no_four_way_physical_overlap_keeps_proven_two_without_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend = self.build(Path(temporary)); owner.requested_native_workers=4
            original = backend.case_resource_evidence
            def no_four(receipt):
                evidence = original(receipt)
                if receipt.get("resource_slot",0) >= 2:
                    evidence["native_interval"]={"start_monotonic":3.,"end_monotonic":4.}
                return evidence
            backend.case_resource_evidence=no_four
            self.assertTrue(owner.run()["complete"])
            self.assertEqual(owner.proven_native_workers,2)
            cases=[r for r in backend.calls if isinstance(r,tuple) and r[0]=="case"]
            self.assertEqual(len(cases),len(set(cases)))

    def test_parallel_infrastructure_closes_admission_after_reaping_entire_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend=self.build(Path(temporary)); owner.requested_native_workers=4
            original=backend.launch
            def failed(split, plan, case, *args, **kwargs):
                result=original(split,plan,case,*args,**kwargs)
                if split=="train" and case==TASKS[0]+"__episode1":
                    result.update(returncode=1,case_complete=False,category="InfrastructureFailure")
                return result
            backend.launch=failed
            self.assertEqual(owner.run()["stage"],"failed")
            cases=[r for r in backend.calls if isinstance(r,tuple) and r[0]=="case"]
            self.assertEqual(len(cases),6)
            self.assertEqual(len(owner.case_outcomes["train"]),6)
            self.assertNotIn("prepare_validation",backend.calls)

    def test_adoption_skips_gpu_but_keeps_cpu_check_and_full_native_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, backend=self.build(Path(temporary)); owner.adopt_training_exit=Path("explicit_exit.json")
            backend.adopt_training=lambda path: backend.calls.append("adopt_actual_exit") or dict(optimization_repeated=False)
            self.assertTrue(owner.run()["complete"])
            self.assertIn("training_CPU",backend.calls)
            self.assertNotIn("training_GPU",backend.calls)
            self.assertLess(backend.calls.index("adopt_actual_exit"),backend.calls.index("freeze"))

    def test_resource_aliases_never_qualify_even_if_every_task_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend=FakeBackend(Path(temporary))
            rows=[]
            for i in range(2):
                evidence=backend.case_resource_evidence(dict(case_key=str(i),parallel=True,resource_slot=i))
                evidence["display"]=":shared"
                rows.append(dict(case_key=str(i),outcome_category="NativeTrue",resources=evidence))
            with self.assertRaisesRegex(WorkflowFailed,"share a process"):
                qualify_parallel(rows,2)


class TrainingAdoptionTests(unittest.TestCase):
    def fixture(self, root):
        import torch
        from our_mani.hierarchy_block_20261102.workflow_v2 import sha256, DEFAULT_TRAINING_CLI
        source=root/"our_mani/hierarchy_block_20261102/training.py"; source.parent.mkdir(parents=True)
        source.write_text("# explicit CPU fixture\n")
        launcher=root/"scripts/run_hierarchy_training_process_block_20261102.py"; launcher.parent.mkdir(parents=True)
        launcher.write_text("# explicit CPU fixture launcher\n")
        identity={"code_sha256":{str(source.relative_to(root)):sha256(source)}}
        artifact=root/ARTIFACT_FAMILY/"training_v1"; artifact.mkdir(parents=True)
        process={"pid":99999999,"start_ticks":1,"boot_id":"synthetic-dead-process"}
        checkpoint=artifact/"best.pt"
        torch.save(dict(version="hierarchy_block_training_20261102_v1", model_state_dict={"explicit_fixture":torch.zeros(1)},
            metadata=dict(training_complete=True,completed_epochs=120,TEST_read=False,identity=identity)),checkpoint)
        status=artifact/"status.json"
        atomic_json(status,dict(version="hierarchy_block_training_20261102_v1",complete=True,completed_epochs=120,
            TEST_read=False,identity=identity,process=process,best_checkpoint_sha256=sha256(checkpoint),precision_qualified=False))
        directory=root/RUNTIME_FAMILY/"training_process_v1";directory.mkdir(parents=True)
        command=[sys.executable,"-B","-u",DEFAULT_TRAINING_CLI,"--device","cuda:0","--workers","4"]
        atomic_json(directory/"launch.json",dict(command=command,launcher_source_sha256=sha256(launcher)))
        atomic_json(directory/"spawned.json",dict(process=process))
        (directory/"training.log").write_text("explicit CPU fixture; no training launched\n")
        receipt=dict(schema="hierarchy_block_external_training_actual_exit_20261102_v1",command=command,
            actual_process_exit_observed=True,returncode=0,actual_returncode=0,timed_out=False,administrative_stop=False,
            training_status=pointer(root,status),checkpoint=pointer(root,checkpoint),launch=pointer(root,directory/"launch.json"),
            spawned_process=pointer(root,directory/"spawned.json"),log=pointer(root,directory/"training.log"))
        path=directory/"exit.json";atomic_json(path,receipt)
        return path,receipt,source

    def test_exact_transitive_exit_chain_accepts_only_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,receipt,source=self.fixture(root)
            self.assertFalse(validate_training_exit(root,path)["optimization_repeated"])
            source.write_text("changed fitting source\n")
            with self.assertRaises((ValueError,WorkflowBlocked)):
                validate_training_exit(root,path)

    def test_unobserved_or_nonzero_exit_is_never_adopted(self):
        for field,value in (("actual_process_exit_observed",False),("returncode",1),("timed_out",True),("administrative_stop",True)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);path,receipt,_=self.fixture(root)
                receipt[field]=value;atomic_json(path,receipt)
                with self.assertRaisesRegex(WorkflowBlocked,"observed actual exit"):
                    validate_training_exit(root,path)


if __name__ == "__main__":
    unittest.main()
