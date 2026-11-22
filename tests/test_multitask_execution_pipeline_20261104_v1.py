"""CPU metadata/lifecycle fixtures; no model, episode payload, Native or real fit."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask_execution_pipeline_20261104_v1 import runner as r
from our_mani.multitask_execution_pipeline_20261104_v1.registry import TASKS, METHOD, task_family
from our_mani.multitask_execution_pipeline_20261104_v1.native_adapter import NativeOutcomeAdapter


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = self.root / 'new_current_network_hook.py'
        self.entry.write_text('# synthetic completion writer fixture\n')
        self.model = self.root / 'synthetic_weights.bin'
        self.model.write_bytes(b'not a real checkpoint; no runtime is constructed')
        self.split_path = self.root / 'split.json'
        self.rows = [dict(task=t, split=s, role='fit' if s == 'train' else s,
            source_parent_group=s + ':' + t, raw_episode_id='episode0',
            eligible_for_pipeline=s != 'test', offline_eligible=s != 'test',
            native_eval_included=s in ('validation', 'test'),
            source_path='/never-opened/' + s + '/' + t)
            for t in TASKS for s in ('train', 'validation', 'test')]
        self.save_split()
        self.plan = dict(schema=r.PLAN_SCHEMA, method=METHOD, tasks=list(TASKS),
            preparation_only=True, execution_authorized=False,
            native_success_authority='task.success()', split_manifest=r.pointer(self.split_path),
            source_sha256={str(self.entry): r.sha(self.entry)}, hooks={},
            task_adapter_readiness={t: dict(public_observation=False,
                task_phase_controller=False, native_reset=False) for t in TASKS})
        self.plan_path = self.root / 'plan.json'
        self.state = self.root / 'state'
        self.save_plan()

    def save_split(self):
        self.split_path.write_text(json.dumps(dict(schema=r.SPLIT_SCHEMA, episodes=self.rows)))

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def hook(self, stage):
        return dict(method=METHOD, stage=stage, resource_lock_owner='pipeline',
            resource='native' if stage in ('validation', 'test') else 'cpu',
            argv=['python', str(self.entry), '--output', '{output}'],
            entrypoint=r.pointer(self.entry), source_sha256={str(self.entry): r.sha(self.entry)},
            timeout_seconds=30, all_tasks=list(TASKS),
            native_success_authority='task.success()',
            policy_input_contract='current_RGBD_public_K_T_proprioception_only',
            Native_GT_contact_or_instance_masks_as_policy_inputs=False)

    def previous(self, stage):
        d = self.state / stage
        d.mkdir(parents=True)
        r.write_new(d / 'receipt.json', dict(stage=stage, complete=True,
            plan_sha256=r.sha(self.plan_path), actual_process_returncode=0,
            completion_files=[], selected_model=r.pointer(self.model),
            fitted_component='synthetic_whole_network_fixture', current_network_end_to_end_ready=True))

    def test_exact_sixteen_single_method_and_no_process_default(self):
        self.assertEqual(len(TASKS), 16)
        self.assertEqual(len({task_family(t) for t in TASKS}), 8)
        with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            result = r.dry_run(self.root, self.plan_path)
        self.assertEqual(result['subprocesses_started'], 0)
        self.assertTrue(all(v['status'] == 'BLOCKED' for v in result['stages']))
        self.assertFalse(self.state.exists())

    def test_split_parent_leak_rejected_and_reserved_never_fit(self):
        reserved = dict(self.rows[0], raw_episode_id='reserved', role='reserved',
            eligible_for_pipeline=False, offline_eligible=False,
            offline_quarantined=True, source_parent_group='reserved:unique')
        self.rows.append(reserved)
        self.save_split()
        _, selected = r.validate_split(self.root, r.pointer(self.split_path))
        self.assertEqual(len(selected), 48)  # includes all16 TEST metadata despite false decode eligibility
        # Quarantine is retained metadata, never a fake split-prefixed group.
        reserved['source_parent_group'] = self.rows[2]['source_parent_group']
        self.save_split()
        r.validate_split(self.root, r.pointer(self.split_path))
        # A quarantined VAL source is forbidden to the decoder but remains in
        # the exact Native VAL denominator, even if metadata overlaps TEST.
        self.rows[1].update(source_parent_group=self.rows[2]['source_parent_group'],
            eligible_for_pipeline=False, offline_eligible=False, offline_quarantined=True)
        self.save_split()
        _, retained = r.validate_split(self.root, r.pointer(self.split_path))
        self.assertEqual(sum(v['split'] == 'validation' for v in retained), 16)
        self.rows[1]['source_parent_group'] = self.rows[0]['source_parent_group']
        self.rows[1]['eligible_for_pipeline'] = True
        self.rows[1]['offline_eligible'] = True
        self.save_split()
        with self.assertRaisesRegex(r.PipelineBlocked, 'leakage'):
            r.validate_split(self.root, r.pointer(self.split_path))

    def test_no_eval_before_actual_fit_and_no_old_method(self):
        self.plan['hooks']['validation'] = self.hook('validation')
        self.save_plan()
        with self.assertRaises(FileNotFoundError):
            r.stage_admission(self.root, self.plan_path, 'validation', self.state)
        self.plan['method'] = 'context_geometry'
        self.save_plan()
        with self.assertRaisesRegex(r.PipelineBlocked, 'single-current-network'):
            r.validate_plan(self.root, self.plan_path)

    def test_preparation_cannot_mkdir_or_launch(self):
        self.plan['hooks']['data_check'] = self.hook('data_check')
        self.save_plan()
        with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(r.PipelineBlocked, 'Preparation-only'):
                r.run_stage(self.root, self.plan_path, 'data_check', self.state, r.sha(self.plan_path))
        self.assertFalse(self.state.exists())

    def test_component_only_fit_never_admits_native_validation(self):
        self.plan['hooks']['validation'] = self.hook('validation')
        self.plan['task_adapter_readiness'] = {t: dict(public_observation=True,
            task_phase_controller=True, native_reset=True) for t in TASKS}
        self.save_plan()
        self.previous('fit')
        path = self.state / 'fit/receipt.json'
        value = r.load(path)
        value.update(fitted_component='semantic_task_e8_only', current_network_end_to_end_ready=False)
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(r.PipelineBlocked, 'component-only'):
            r.stage_admission(self.root, self.plan_path, 'validation', self.state)

    def test_actual_task_false_retains_all_fixed_cases_and_freeze_same_model(self):
        self.plan.update(preparation_only=False, execution_authorized=True)
        self.plan['hooks']['validation'] = self.hook('validation')
        self.plan['task_adapter_readiness'] = {t: dict(public_observation=True,
            task_phase_controller=True, native_reset=True) for t in TASKS}
        self.save_plan()
        self.previous('fit')
        by_index = [row for row in self.rows if row['split'] == 'validation']
        calls = []
        def synthetic_closed(root, argv, output, timeout, stops):
            row = by_index[len(calls)]
            calls.append(row['task'])
            artifact = Path(argv[argv.index('--output') + 1])
            self.assertFalse(artifact.exists())
            artifact.mkdir()
            r.write_new(artifact / 'completion.json', dict(complete=True, method=METHOD,
                stage='validation', source_unchanged=True, task=row['task'],
                source_row_sha256=r.digest(row), native_closed=True,
                native_success_source='task.success()', native_ever_success=False,
                native_final_success=False, model=r.pointer(self.model)))
            return 0, None
        with patch.object(r, 'resource_locks', return_value=nullcontext()), patch.object(r, '_run_command', side_effect=synthetic_closed):
            receipt = r.run_stage(self.root, self.plan_path, 'validation', self.state, r.sha(self.plan_path))
            self.assertEqual(calls, list(TASKS))
            self.assertEqual(receipt['cases'], 16)
            frozen = r.run_stage(self.root, self.plan_path, 'freeze', self.state, r.sha(self.plan_path))
        self.assertEqual(frozen['selected_model'], r.pointer(self.model))
        self.assertFalse(frozen['numeric_task_success_gate_invented'])
        self.model.write_bytes(b'changed')
        with self.assertRaises(r.PipelineBlocked):
            r.stage_admission(self.root, self.plan_path, 'freeze', self.state)

    def test_host_owner_write_failure_still_terminates_and_reaps(self):
        output = self.root / 'lifecycle'
        output.mkdir()
        class Process:
            pid = 424242
            done = False
            waits = 0
            def poll(self): return -15 if self.done else None
            def wait(self, timeout=None):
                self.waits += 1
                self.done = True
                return -15
        process = Process()
        original = r.write_new
        def fail_owner(path, value):
            if Path(path).name == 'owner.json':
                raise RuntimeError('synthetic owner serialization failure after Popen')
            return original(path, value)
        with patch.object(r.subprocess, 'Popen', return_value=process), patch.object(r.os, 'killpg') as kill, patch.object(r, 'write_new', side_effect=fail_owner):
            with self.assertRaisesRegex(RuntimeError, 'after Popen'):
                r._run_command(self.root, ['never executed'], output, 30, [])
        kill.assert_called_once()
        self.assertGreaterEqual(process.waits, 1)
        self.assertTrue(r.load(output / 'owner.actual_exit.json')['actual_wait_completed'])

    def test_native_adapter_rejects_non_native_and_keeps_first_terminal(self):
        with self.assertRaises(TypeError):
            NativeOutcomeAdapter(object(), dict(task=TASKS[0], native_module='rlbench.tasks.' + TASKS[0], native_class='RealNative'))
        # Test the reducer only, without constructing any passing native adapter.
        class ScriptedOutcome:
            def success(self): return self.value
        obj = object.__new__(NativeOutcomeAdapter)
        obj._task = ScriptedOutcome()
        obj._success_function = ScriptedOutcome.success
        obj._terminal = obj._current = None
        obj._ever_success = False
        obj._task.value = (False, True)
        obj.observe_after_physics(.05)
        obj._task.value = (True, True)
        obj.observe_after_physics(.10)
        out = obj.metric_receipt()
        self.assertTrue(out['native_first_terminal']['terminal_failure'])
        self.assertTrue(out['native_current']['success'])
        self.assertFalse(out['policy_use_allowed'])
        with self.assertRaises(ValueError): obj.observe_after_physics(.10)


if __name__ == '__main__':
    unittest.main()
