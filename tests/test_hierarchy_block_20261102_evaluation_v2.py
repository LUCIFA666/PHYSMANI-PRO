"""Meaningful CPU contracts; no native runtime, model substitutes or TEST files."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import json
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from our_mani.hierarchy_block_20261102 import evaluation_v2 as e
from our_mani.aligned.protocol import TickLedger
from our_mani.contracts import Pose, RobotState


class FakeEnvironment:
    def __init__(self, success_tick=None, failure_tick=None):
        self.ledger = TickLedger()
        self._native_success_latched = self._native_failure_latched = self._native_terminal_latched = False
        self.success_tick, self.failure_tick = success_tick, failure_tick
        self.task = SimpleNamespace(_fail_conditions=[])
        self.actions = []

    def observe(self):
        robot = RobotState(self.ledger.time, Pose(np.zeros(3)), np.zeros(3), np.zeros(3),
                           np.zeros(7), np.zeros(7), True, False, True)
        return {"timestamp": self.ledger.time, "observation/images": {}, "observation/depths": {},
                "observation/calibrations": {}, "observation/instance_masks": {"oracle": 999},
                "expert_debug": "never expose", "observation/state": "never expose"}, robot

    @contextmanager
    def native_action_scope(self, **kwargs):
        self.actions.append(kwargs)
        yield

    def _before_action(self):
        self.ledger.before()

    def hold_tick(self):
        self.ledger.after(.05, .05, "hold")


class FakeActor:
    def __init__(self):
        self.seen = []
        self.counter = 0

    def observe(self, observation, robot):
        assert "observation/instance_masks" not in observation and "expert_debug" not in observation
        assert "observation/state" not in observation and robot.task_success is False
        self.seen.append(observation["timestamp"])

    def act(self, observation, robot, remaining_seconds):
        assert robot.task_success is False
        self.counter += 1
        # The action is the network interface's result, not a production fake.
        latent = f"{self.counter:064x}"
        return {"action": np.array([.1, .2, .3, 0, 0, 0, 1, 1.]),
                "receipt": {"public_tracks_valid": True, "real_forecast_consumed": True,
                            "event": "candidate_handoff", "forecast_generated_at": robot.timestamp,
                            "old_stage_rule_calls": 0, "task_metadata_used": False,
                            "shared_skill_latent_sha256": latent,
                            "candidate_skill_latent_sha256": latent,
                            "policy_skill_latent_sha256": latent}}


class FakeExecutor:
    skill_policy = None

    def __init__(self):
        self.actions = []

    def execute(self, env, action, observer):
        self.actions.append(action.copy())
        for _ in range(min(10, int(round(env.ledger.remaining / env.ledger.dt)))):
            env.ledger.after(.05, .05, "mock_native")
            if env.success_tick is not None and env.ledger.ticks >= env.success_tick:
                env._native_success_latched = env._native_terminal_latched = True
            if env.failure_tick is not None and env.ledger.ticks >= env.failure_tick:
                env._native_failure_latched = env._native_terminal_latched = True
            observer(*env.observe())
            if env._native_terminal_latched:
                break
        return {"ticks": env.ledger.ticks}


class EvaluationTests(unittest.TestCase):
    def test_actual_cli_uses_runtime_native_python_and_imports_cffi_without_native(self):
        project=Path(__file__).resolve().parents[1]
        command=["bash","-c",'source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh; exec python -B "$1" --check-native-python',
                 "native-abi-CPU-test",str(project/"scripts/evaluate_hierarchy_case_block_20261102_v2.py")]
        result=subprocess.run(command,cwd=project,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        proof=json.loads(result.stdout)
        self.assertEqual(proof["version"][:2],[3,10])
        self.assertEqual(Path(proof["resolved_executable"]),e.NATIVE_PYTHON.resolve())
        self.assertIn("_cffi_backend",proof["cffi_backend"])
        self.assertFalse(proof["native_started"])
        with patch.object(e.sys,"executable","/unverified/training/python"):
            with self.assertRaisesRegex(e.EvaluationBlocked,"training Python"):
                e.native_interpreter_contract()
    def test_train8_and_original_val40_rosters(self):
        train = {"split": "train", "episodes": [{"task": t, "raw_episode_id": f"episode{i}"} for t in e.TASKS for i in range(7)]}
        selected = e.select_sources(train, "train")
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(row["raw_episode_id"] in {"episode0", "episode1", "episode2", "episode3"} for row in selected))
        val = {"split": "validation", "episodes": [{"task": t, "raw_episode_id": f"episode{i}"} for t, n in zip(e.TASKS, (20, 20)) for i in range(n)]}
        self.assertEqual(len(e.select_sources(val, "validation")), 40)
        val["episodes"].pop()
        with self.assertRaises(e.EvaluationBlocked):
            e.select_sources(val, "validation")

    def test_test_rejected_before_any_file_read(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("TEST read")):
            with self.assertRaisesRegex(e.EvaluationBlocked, "TEST remains sealed"):
                e.prepare_plan(Path("/does/not/exist"), "test")

    def test_missing_or_untrained_model_dependency_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(e.EvaluationBlocked, "freeze"):
                e.validate_freeze(root, {})
            partial = {"schema": e.FREEZE_SCHEMA, "frozen": True, "tasks": list(e.TASKS),
                       "readiness": {key: True for key in e.READY_FIELDS}}
            partial["readiness"]["candidate_trained"] = False
            with self.assertRaisesRegex(e.EvaluationBlocked, "candidate_trained"):
                e.validate_freeze(root, partial)

    def test_freeze_requires_completed_candidate_checkpoint_evidence_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            components = {}
            for name in e.COMPONENTS:
                weight = root / (name + ".pt")
                weight.write_bytes(("fixture_" + name).encode())
                proof = root / (name + ".json")
                proof.write_text(json.dumps({"complete": True, "trained_components": [name],
                    "component_artifact_sha256": {name: e.sha(weight)}, "TEST_used": False}))
                components[name] = {"artifact": e.pointer(root, weight), "evidence": e.pointer(root, proof), "trained": True}
            actor = root / "our_mani/hierarchy_block_20261102/actor_v2.py"
            actor.parent.mkdir(parents=True)
            actor.write_text("# This CPU fixture is never imported.\n")
            freeze = {"schema": e.FREEZE_SCHEMA, "frozen": True, "tasks": list(e.TASKS),
                      "readiness": {key: True for key in e.READY_FIELDS}, "components": components,
                      "actor": {"module": "our_mani.hierarchy_block_20261102.actor_v2", "factory": "create", "source": e.pointer(root, actor)},
                      "protocol": {"dt": .05, "simulation_seconds": 10., "gripper_velocity": .2,
                                   "baseline_decision_ticks": 10, "warmup_ticks": 4,
                                   "baseline_repo": "/home/ubuntu/桌面/phsimani/PhysMani",
                                   "inference_device": "cpu", "current_instance_masks": True, "information_setting": "native sensors filtered before actor"},
                      "precision_qualified": False, "policy_inputs": "causal_RGBD_calibration_robot_only",
                      "old_stage_policy_allowed": False, "forecast_fallback_allowed": False}
            e.validate_freeze(root, freeze)
            readiness = root / "cpu_readiness.json"
            readiness.write_text(json.dumps({"complete": True, "device": "cpu"}))
            freeze["cpu_actor_readiness"] = e.pointer(root, readiness)
            report = root / "offline_report.json"; report.write_text("{}")
            freeze["offline_precision_reports"] = [e.pointer(root, report)]
            freeze["training_status"] = e.pointer(root, report)
            e.validate_freeze(root, freeze)
            report.write_text('{"changed":true}')
            with self.assertRaisesRegex(e.EvaluationBlocked, "artifact changed"):
                e.validate_freeze(root, freeze)
            report.write_text("{}")
            readiness.write_text(json.dumps({"complete": False, "device": "cpu"}))
            freeze["cpu_actor_readiness"] = e.pointer(root, readiness)
            with self.assertRaisesRegex(e.EvaluationBlocked, "CPU actor readiness"):
                e.validate_freeze(root, freeze)
            readiness.write_text(json.dumps({"complete": True, "device": "cpu"}))
            freeze["cpu_actor_readiness"] = e.pointer(root, readiness)
            proof = root / "candidate_network.json"
            record = json.loads(proof.read_text()); record["complete"] = False
            proof.write_text(json.dumps(record))
            components["candidate_network"]["evidence"] = e.pointer(root, proof)
            with self.assertRaisesRegex(e.EvaluationBlocked, "checkpoint-bound TRAIN evidence"):
                e.validate_freeze(root, freeze)
            record["complete"] = True; proof.write_text(json.dumps(record))
            components["candidate_network"]["evidence"] = e.pointer(root, proof)
            (root / "candidate_network.pt").write_bytes(b"changed_checkpoint")
            with self.assertRaisesRegex(e.EvaluationBlocked, "artifact changed"):
                e.validate_freeze(root, freeze)

    def test_incomplete_actual_native_val_exit_cannot_issue_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"case_key": "first"}
            plan = {"split": "validation", "expected_cases": 40, "cases": [case]}
            runtime = root / e.RUNTIME / "validation/cases/first"
            runtime.mkdir(parents=True)
            (runtime / "parent_exit.json").write_text(json.dumps({"actual_process_exit_observed": True,
                "case_complete": True, "returncode": 0, "timed_out": True}))
            with patch.object(e, "verify_plan", return_value=(plan, {})):
                with self.assertRaisesRegex(e.EvaluationBlocked, "natural, exit-audited"):
                    e.audit_validation(root, root / "plan.json")
            self.assertFalse((root / e.ARTIFACTS / "validation/Native_VAL_release.json").exists())

    def test_parent_records_actual_wait_exit_and_never_silently_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"; plan_path.write_text("{}")
            plan = {"split": "train", "cases": [{"case_key": "first"}], "environment": {"sha256": "0" * 64}}
            def spawned(*args, **kwargs):
                command=args[0]
                self.assertEqual(command[command.index("scripts/evaluate_hierarchy_case_block_20261102_v2.py")-2],"python")
                self.assertNotIn(str(root/".conda-env/bin/python"),command)
                output = root / e.RUNTIME / "train/cases/first/terminal.json"
                output.write_text(json.dumps({"complete": True, "natural_completion": True,
                    "category": "NativeFalse", "administrative_stop": False}))
                return SimpleNamespace(pid=os.getpid(), wait=lambda **kwargs: 0)
            with patch.object(e, "verify_plan", return_value=(plan, {})), patch.object(e.subprocess, "Popen", side_effect=spawned) as launch:
                record = e.launch_case(root, plan_path, "first")
                self.assertTrue(record["actual_process_exit_observed"])
                self.assertTrue(record["case_complete"])
                self.assertEqual(record["category"], "NativeFalse")
                self.assertEqual(record["returncode"], 0)
                with self.assertRaisesRegex(e.EvaluationBlocked, "no silent retry"):
                    e.launch_case(root, plan_path, "first")
                self.assertEqual(launch.call_count, 1)

    def test_nonzero_process_exit_is_infrastructure_even_with_provisional_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"; plan_path.write_text("{}")
            plan = {"split": "train", "cases": [{"case_key": "first"}], "environment": {"sha256": "0" * 64}}
            def spawned(*args, **kwargs):
                output = root / e.RUNTIME / "train/cases/first/terminal.json"
                output.write_text(json.dumps({"complete": True, "natural_completion": True,
                    "category": "NativeTrue", "administrative_stop": False}))
                return SimpleNamespace(pid=os.getpid(), wait=lambda **kwargs: 1)
            with patch.object(e, "verify_plan", return_value=(plan, {})), patch.object(e.subprocess, "Popen", side_effect=spawned):
                record = e.launch_case(root, plan_path, "first")
                self.assertFalse(record["case_complete"])
                self.assertEqual(record["category"], "InfrastructureFailure")
                self.assertEqual(record["worker_reported_category"], "NativeTrue")

    def test_initialization_stop_preserves_actual_exit_and_missing_terminal_as_administrative(self):
        for requested in (True, False):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_path = root / "plan.json"; plan_path.write_text("{}")
                plan = {"split": "train", "cases": [{"case_key": "first"}], "environment": {"sha256": "0"*64}}
                owner_stop = root / "runtime/framework/hierarchy_block_20261102/workflow_v2/STOP"
                def spawned(*args, **kwargs):
                    if requested:
                        owner_stop.parent.mkdir(parents=True)
                        owner_stop.write_text("explicit administrative request\n")
                    return SimpleNamespace(pid=os.getpid(), wait=lambda **kwargs: 2)
                with patch.object(e, "verify_plan", return_value=(plan, {})), patch.object(e.subprocess, "Popen", side_effect=spawned):
                    record = e.launch_case(root, plan_path, "first", administrative_stop_path=owner_stop)
                self.assertEqual(record["category"], "AdministrativeSTOP" if requested else "InfrastructureFailure")
                self.assertTrue(record["actual_process_exit_observed"])
                self.assertEqual(record["returncode"], 2)
                self.assertIsNone(record["terminal"])
                self.assertFalse(record["case_complete"])
                self.assertEqual(len(record["administrative_stop_evidence"]), int(requested))

    def test_published_freeze_cannot_omit_cpu_attestation_and_old_stop_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / e.FAMILY_ROOT / "freeze.json"; path.parent.mkdir(parents=True)
            path.write_text("{}")
            with self.assertRaisesRegex(e.EvaluationBlocked, "actual CPU actor"):
                e.load_freeze(root, path)
            with self.assertRaisesRegex(e.EvaluationBlocked, "new block namespace"):
                e.stop_evidence(root, root / "runtime/framework/hierarchy_20261030/evaluation_v1/STOP")

    def test_final_resource_failure_keeps_worker_infrastructure_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); key="synthetic_case"
            plan_path=root/"plan.json";plan_path.write_text("{}")
            case={"case_key":key,"task":e.TASKS[0]}
            runtime=root/e.RUNTIME/"train/cases"/key;runtime.mkdir(parents=True)
            (runtime/"launch.json").write_text(json.dumps({"token":"test-token","parent":e.process_identity()}))
            actor_source=root/"synthetic_actor.py";actor_source.write_text("# never imported test fixture\n")
            closed=[]
            fixture_actor=SimpleNamespace(observe=lambda *_:None,act=lambda *_:None,close=lambda:closed.append("actor"))
            fixture_module=SimpleNamespace(__file__=str(actor_source),create=lambda **_:fixture_actor)
            fixture_env=SimpleNamespace(close=lambda:closed.append("env"))
            freeze={"actor":{"module":"synthetic_actor","factory":"create","source":e.pointer(root,actor_source)},"protocol":{}}
            plan={"cases":[case],"split":"train","freeze":{"path":"synthetic","sha256":"0"*64},"environment":{"sha256":"0"*64}}
            resources={"errors":[],"complete":False,"restored":False,"native_interval":{}}
            @contextmanager
            def final_failure(*args,**kwargs):
                yield resources
                resources["errors"].append({"message":"simulated cleanup audit failure"})
                raise RuntimeError("simulated final resource audit failure")
            def interval(receipt,phase):
                receipt["native_interval"][phase+"_monotonic"]=1. if phase=="start" else 2.
            with patch.object(e,"verify_plan",return_value=(plan,freeze)), patch.object(e,"evaluation_locks",final_failure), \
                 patch.object(e.importlib,"import_module",return_value=fixture_module), patch.object(e,"create_native_environment",return_value=fixture_env), \
                 patch.object(e,"run_loop",return_value=dict(complete=True,natural_completion=True,category="NativeFalse")), \
                 patch("scripts.run_frozen_system_environment_v820_r10.namespace_context"), \
                 patch("our_mani.hierarchy_block_20261102.isolation.record_native_interval",side_effect=interval), \
                 patch("our_mani.multitask.author_aligned_geometry_executor_v1114.AuthorAlignedGeometryExecutor",return_value=FakeExecutor()), \
                 patch.dict(os.environ,DISPLAY=":explicit_cpu_fixture",HIERARCHY_EVALUATION_PARENT_TOKEN="test-token"):
                record=e.worker_case(root,plan_path,key)
            self.assertEqual(record["category"],"InfrastructureFailure")
            self.assertFalse(record["complete"])
            self.assertEqual(record["resources"]["errors"][0]["message"],"simulated cleanup audit failure")
            self.assertEqual(set(closed),{"actor","env"})
            self.assertTrue((runtime/"terminal.json").is_file())

    def test_dataflow_native_budget_shared_latent_and_no_old_rules(self):
        env, actor, executor = FakeEnvironment(), FakeActor(), FakeExecutor()
        result = e.run_loop(env, actor, executor, block=False)
        self.assertEqual(result["category"], "NativeFalse")
        self.assertEqual(result["counted_ticks"], 200)
        self.assertEqual(len(executor.actions), 20)
        self.assertEqual(len(env.actions), 24)
        self.assertEqual(len(result["holds"]), 4)
        self.assertTrue(all(np.array_equal(a, [.1, .2, .3, 0, 0, 0, 1, 1]) for a in executor.actions))
        self.assertFalse(result["actual_process_exit_observed"])
        self.assertEqual(len(actor.seen), 201)
        self.assertEqual(len(actor.seen), len(set(actor.seen)))

    def test_success_failure_and_administrative_stop_separate(self):
        success = e.run_loop(FakeEnvironment(success_tick=7), FakeActor(), FakeExecutor(), block=False)
        failure = e.run_loop(FakeEnvironment(failure_tick=5), FakeActor(), FakeExecutor(), block=False)
        stop = e.run_loop(FakeEnvironment(), FakeActor(), FakeExecutor(), block=False, stop_requested=lambda: True)
        self.assertEqual(success["category"], "NativeTrue")
        self.assertEqual(success["counted_ticks"], 7)
        self.assertEqual(failure["category"], "NativeTerminalFailure")
        self.assertTrue(failure["native_terminal_failure"])
        self.assertEqual(stop["category"], "AdministrativeSTOP")
        self.assertIsNone(stop["native_success"])
        self.assertFalse(stop["complete"])

    def test_no_public_tracks_means_no_executor_action_or_fake_forecast(self):
        env, actor, executor = FakeEnvironment(), FakeActor(), FakeExecutor()
        original = actor.act
        def unavailable(*args, **kwargs):
            decision = original(*args, **kwargs)
            decision["receipt"]["public_tracks_valid"] = False
            return decision
        actor.act = unavailable
        with self.assertRaisesRegex(e.ActorContractError, "public tracking"):
            e.run_loop(env, actor, executor, block=False)
        self.assertEqual(len(executor.actions), 0)
        self.assertEqual(env.ledger.ticks, 4)

    def test_runtime_occlusion_is_counted_unknown_and_not_infrastructure(self):
        env, actor, executor = FakeEnvironment(), FakeActor(), FakeExecutor()
        actor.act = lambda *args, **kwargs: {"action": None, "receipt": {
            "event": "perception_unknown", "reason": "ring_occluded", "real_forecast_consumed": False,
            "old_stage_rule_calls": 0, "task_metadata_used": False}}
        result = e.run_loop(env, actor, executor, block=False)
        self.assertEqual(result["category"], "NativeFalse")
        self.assertEqual(result["perception_unknown_holds"], 196)
        self.assertFalse(result["infrastructure_failed"])
        self.assertTrue(result["complete"])
        self.assertEqual(len(executor.actions), 0)

    def test_learned_skill_continuation_does_not_claim_cached_forecast(self):
        actor = FakeActor()
        observation, robot = FakeEnvironment().observe()
        decision = actor.act(e.public_observation(observation), e.public_robot(robot), 10.)
        decision["receipt"].update(event="skill_execution", real_forecast_consumed=False,
                                   candidate_skill_latent_sha256=None, public_tracks_valid=False)
        action, receipt = e.validate_decision(decision, timestamp=robot.timestamp)
        self.assertEqual(action.shape, (8,))
        decision["receipt"]["real_forecast_consumed"] = True
        with self.assertRaisesRegex(e.ActorContractError, "no new forecast"):
            e.validate_decision(decision, timestamp=robot.timestamp)

    def test_distinct_latents_old_stage_gate_and_nonunit_action_refused(self):
        actor = FakeActor()
        observation, robot = FakeEnvironment().observe()
        decision = actor.act(e.public_observation(observation), e.public_robot(robot), 10.)
        decision["receipt"]["policy_skill_latent_sha256"] = "b" * 64
        with self.assertRaisesRegex(e.ActorContractError, "shared skill latent"):
            e.validate_decision(decision, timestamp=robot.timestamp)
        decision["receipt"]["policy_skill_latent_sha256"] = decision["receipt"]["shared_skill_latent_sha256"]
        decision["receipt"]["old_stage_rule_calls"] = 1
        with self.assertRaisesRegex(e.ActorContractError, "old stage"):
            e.validate_decision(decision, timestamp=robot.timestamp)
        executor = FakeExecutor()
        executor.skill_policy = object()
        with self.assertRaisesRegex(e.ActorContractError, "must be absent"):
            e.run_loop(FakeEnvironment(), FakeActor(), executor, block=False)
        decision["receipt"]["old_stage_rule_calls"] = 0
        decision["action"][3:7] = 0
        with self.assertRaisesRegex(e.ActorContractError, "unit-xyzw"):
            e.validate_decision(decision, timestamp=robot.timestamp)

    def test_legacy_block_receipt_does_not_call_success_again(self):
        env = FakeEnvironment(success_tick=7)
        env.task.success = lambda: (_ for _ in ()).throw(AssertionError("extra task success call"))
        result = e.run_loop(env, FakeActor(), FakeExecutor(), block=True)
        self.assertEqual(result["category"], "NativeTrue")
        self.assertEqual(len(env.actions), 0)


if __name__ == "__main__":
    unittest.main()
