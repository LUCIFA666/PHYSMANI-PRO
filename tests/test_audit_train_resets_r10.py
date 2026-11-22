"""Reset-repeat preparation and release guards; no native process launched."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_train_resets_r10.py'
module = importlib.util.spec_from_file_location('reset_repeat_r10_tested', SCRIPT)
audit = importlib.util.module_from_spec(module);module.loader.exec_module(audit)
from our_mani.multitask.reset_parity import digest

LEGACY = dict(robot_position_m=.001, robot_orientation_rad=.002, robot_joints_rad=.0001,
              target_position_m=.0001, fixture_pose_component=.0001, camera_calibration_component=.00001)


class RepeatTests(unittest.TestCase):
    def test_preregistered_numeric_thresholds_remain_pending_without_automatic_fit(self):
        profile = audit.preregistered_profile(LEGACY)
        assert profile['thresholds']['body_orientation_rad'] == 1e-4
        assert profile['thresholds']['body_linear_velocity_m_s'] == 1e-4
        assert profile['thresholds']['native_joint_position_units'] == 1e-8
        assert profile['thresholds']['schedule_numeric_absolute'] == 1e-8
        assert profile['status'] != 'frozen_after_TRAIN_repeat_validation'
        assert profile['TRAIN_evidence'] is None and profile['threshold_relaxation_automatic'] is False

    def test_plan_has_exact_14x4_fixed_source_seed_and_rejects_seed_or_threshold_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);base = root / 'artifacts/framework/multitask_r10';base.mkdir(parents=True)
            legacy = root / 'artifacts/framework/aligned_r09/acceptance_tolerances.json'
            legacy.parent.mkdir(parents=True);legacy.write_text(json.dumps(LEGACY))
            rows = []
            for task, spec in audit.TASK_SPECS.items():
                source = root / 'train_package_compressed' / task / 'episode9/data.h5'
                source.parent.mkdir(parents=True);source.write_bytes(b'mock HDF5 provenance only')
                cache = base / (task + '.npz');cache.write_bytes(b'mock small cache')
                rows.append(dict(split='train', task=task, source_episode='episode9', source_path=str(source),
                    source_file_stat=audit.source_stat(source), npz_path=str(cache.relative_to(root)),
                    npz_sha256=audit.sha(cache), variation=spec.supported_variations[0]))
            manifest = base / 'manifest.json';manifest.write_text(json.dumps(dict(
                version='multitask_r10_grouped_small_demonstrations_v1', complete=True,
                TEST_used_for_training_calibration_or_selection=False, episodes=rows)))
            protocol = base / 'protocol.json';protocol.write_text(json.dumps(dict(
                dt=.05, simulation_seconds=10., gripper_velocity=.2, current_instance_masks=True,
                information_setting='CPU mock source identity, no native run', baseline_repo=str(root / 'native'), seed=0)))
            paths = [root / 'scripts/run_local_sim.sh', root / 'our_mani/sensor_alignment/author_rgbd.py',
                     *[root / 'our_mani/multitask' / name for name in
                       ('reset_parity.py', 'simulation.py', 'maps.py', 'native_tasks.py', 'specs.py', 'protocol.py')]]
            for path in paths: path.parent.mkdir(parents=True, exist_ok=True);path.write_text('# mock source')
            output = base / 'plan.json'
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'BASE', base), \
                    patch.object(audit, 'native_inputs', return_value={}):
                plan = audit.build_plan(manifest, protocol, output)
                assert len(plan['cases']) == 56 and plan['counted_actions_per_case'] == 0
                assert plan['native_started'] is False and plan['teacher_used'] is False
                assert audit.verify_plan(output)['plan_sha256'] == plan['plan_sha256']
                for task in audit.TASK_SPECS:
                    group = [row for row in plan['cases'] if row['task'] == task]
                    assert [row['repeat'] for row in group] == [0, 1, 2, 3]
                    assert [row['pair'] for row in group] == ['fit', 'fit', 'independent_validation', 'independent_validation']
                    assert len({json.dumps(row['source'], sort_keys=True) for row in group}) == 1
                changed = deepcopy(plan);changed['cases'][0]['seed'] += 1
                changed['plan_sha256'] = digest({k: v for k, v in changed.items() if k != 'plan_sha256'})
                output.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'source/seed'): audit.verify_plan(output)
                changed = deepcopy(plan)
                for row in changed['cases']: row['seed'] = 5
                changed['plan_sha256'] = digest({k: v for k, v in changed.items() if k != 'plan_sha256'})
                output.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'source/seed'): audit.verify_plan(output)
                changed = deepcopy(plan);changed['profile']['thresholds']['body_orientation_rad'] = 1.
                changed['plan_sha256'] = digest({k: v for k, v in changed.items() if k != 'plan_sha256'})
                output.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'preregistered thresholds'): audit.verify_plan(output)

    def test_missing_demo_or_incomplete_formal_gate_blocks_before_native_launch(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'ROOT', Path(folder)), \
                patch.object(audit.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'canonical ten demos'): audit.require_completed_delivery()
            r09 = Path(folder) / 'artifacts/framework/aligned_r09';r09.mkdir(parents=True)
            (r09 / 'final_report.json').write_text(json.dumps({'complete': False}))
            (r09 / 'demo_index.json').write_text(json.dumps({'complete': True}))
            with self.assertRaisesRegex(ValueError, 'formal400 gate'): audit.require_completed_delivery()
            launch.assert_not_called()

    def test_complete_delivery_requires_fixed_ten_inventory_and_unchanged_decoded_video_evidence(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'ROOT', Path(folder)):
            root = Path(folder);r09 = root / 'artifacts/framework/aligned_r09';r09.mkdir(parents=True)
            report = r09 / 'final_report.json';report.write_text(json.dumps(dict(complete=True,
                protocol_acceptance_passed=True, total_test_runs=400, expected_test_runs=400,
                acceptance_errors=[], fixed_candidate_evaluation_release={'acknowledged': True})))
            (r09 / 'status.json').write_text(json.dumps(dict(stage='complete')))
            entries = []
            for method in audit.DEMO_METHODS:
                for task in audit.DEMO_TASKS:
                    video = r09 / (method + '__' + task + '.mp4');video.write_bytes(b'mock video identity only')
                    entries.append(dict(method=method, task=task, raw_episode_id=audit.DEMO_FIRST_RAW[task],
                        video=str(video.relative_to(root)), video_sha256=audit.sha(video),
                        video_validation=dict(decoded_frame_count=10)))
            index_path = r09 / 'demo_index.json'
            index = dict(version='r09_canonical_ten_demo_index_v1', complete=True, completed=10, expected=10,
                formal_final_report_sha256=audit.sha(report), original_formal_statistics_replaced=False,
                replay_results_in_formal_statistics=False, entries=entries)
            index_path.write_text(json.dumps(index))
            guard = audit.require_completed_delivery()
            assert len(guard['video_sha256']) == 10 and guard['success_outcomes_used_for_thresholds'] is False
            index['entries'][0]['raw_episode_id'] = 'episode999'
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'inventory'): audit.require_completed_delivery()
            index['entries'][0]['raw_episode_id'] = audit.DEMO_FIRST_RAW[index['entries'][0]['task']]
            index['entries'][0]['video_validation']['decoded_frame_count'] = 0
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'decoded demo'): audit.require_completed_delivery()

    def test_cached_case_requires_bound_process_receipt_and_log(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder);directory = output / 'cases/task__repeat0';directory.mkdir(parents=True)
            record = directory / 'case.json';record.write_text(json.dumps(dict(process_id=123)))
            (directory / 'reset.log').write_text('mock native process log')
            (directory / 'process_receipt.json').write_text(json.dumps(dict(
                child_record_sha256=audit.sha(record), exit_code=0)))
            with patch.object(audit, 'verify_plan', return_value={'plan_sha256': 'fixed'}), \
                    patch.object(audit, 'verify_execution', return_value={'sha256': 'fixed execution'}), \
                    patch.object(audit, 'validate_case_identity'), patch.object(audit.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'cached native reset'):
                    audit.run_case('plan', 'execution', {'case_id': 'task__repeat0'}, output,
                                   __import__('threading').Event())
                launch.assert_not_called()

    def test_child_release_failure_occurs_before_runtime_import_or_native_execution(self):
        with patch.object(audit, 'verify_plan', return_value={}), \
                patch.object(audit, 'verify_execution', side_effect=ValueError('demo gate pending')), \
                patch.object(audit.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'demo gate pending'):
                audit.child('mock plan', 'mock execution', 'fixed case', 'mock output')
            launch.assert_not_called()

    def test_cpu_worker_budget_cannot_be_expanded(self):
        with patch.object(audit, 'verify_plan') as verify:
            with self.assertRaisesRegex(ValueError, 'four'): audit.run_queue('mock', 'mock', workers=5)
            verify.assert_not_called()


if __name__ == '__main__': unittest.main()
