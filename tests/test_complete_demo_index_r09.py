"""Fixed-case demo orchestration contracts without native/GPU child processes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/complete_demo_index_r09.py'
spec = importlib.util.spec_from_file_location('r09_demo_index_tested', SCRIPT)
demo = importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    def test_exact_ten_entries_eight_replays_and_resource_bounds_routing(self):
        rows = [dict(task=task, raw_episode_id=demo.FIRST_RAW[task]) for task in demo.TASKS]
        rows += [dict(task=task, raw_episode_id='episode999') for task in demo.TASKS]
        specs = demo.fixed_specs(rows)
        assert len(specs) == 10 and sum(item['reuse_formal_video'] for item in specs) == 2
        missing = [item for item in specs if not item['reuse_formal_video']]
        assert sum(item['queue'] == 'cpu' for item in missing) == 6
        assert sum(item['queue'] == 'gpu' for item in missing) == 2
        for item in missing:
            child = demo.DEMO_BASE / item['method'] / item['name'] / 'attempt_fixture'
            command = demo.command_for(item, child)
            assert command[command.index('--raw-episode-id') + 1] == demo.FIRST_RAW[item['task']]
            assert command[command.index('--out') + 1].startswith('artifacts/framework/aligned_r09/demo_only/')
            if item['queue'] == 'gpu':
                assert demo.BASELINE_HARNESS in command and '--video-out' in command
            else:
                assert demo.OURS_EVALUATOR in command and command[-2:] == ['--videos', '1']
        reversed_rows = rows[2:] + rows[:2]
        with self.assertRaises(ValueError): demo.fixed_specs(reversed_rows)

    def test_formal400_missing_or_incomplete_never_launches_native(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(demo, 'BASE', Path(directory)), \
                patch.object(demo.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'not available'): demo.prepare()
            for name, data in [('final_report.json', {'complete': False}),
                               ('fixed_candidate_evaluation_release.json', {}), ('success_acceptance.json', {})]:
                (Path(directory) / name).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'complete accepted formal400'): demo.prepare()
            launch.assert_not_called()

    def test_video_full_decode_count_must_match_each_method_frame_convention(self):
        video = Path('mock.mp4')
        record = {'steps': [1, 2, 3], 'alignment_receipt': {'clock': {'ticks': 3}}}
        decoded = dict(decoded_frame_count=3, size=[512, 512], fps=20.)
        with patch.object(demo.delivery, 'decode_video', return_value=decoded):
            assert demo.validated_video(video, record, 'cv_geometry')['decoded_frame_count'] == 3
            with self.assertRaisesRegex(ValueError, 'length differs'):
                demo.validated_video(video, record, 'physmani')
        decoded = dict(decoded, decoded_frame_count=4)
        with patch.object(demo.delivery, 'decode_video', return_value=decoded):
            assert demo.validated_video(video, record, 'physmani')['decoded_frame_count'] == 4
        decoded = dict(decoded, fps=10.)
        with patch.object(demo.delivery, 'decode_video', return_value=decoded):
            with self.assertRaisesRegex(ValueError, '20Hz'):
                demo.validated_video(video, record, 'physmani')

    def test_failed_task_demo_is_retained_separately_and_cached_without_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            root, base = Path(directory), Path(directory) / 'artifacts/framework/aligned_r09'
            out = base / 'demo_only/supplement';out.mkdir(parents=True)
            (out / 'demo_execution_inputs.json').write_text('{"frozen":true}')
            formal_path = root / 'formal.json'
            formal = dict(official_task_success=True, failure=None, simulation_seconds=3.,
                case_process_receipt={'child_record_path': 'formal_child.json', 'child_record_sha256': 'f' * 64})
            formal_path.write_text(json.dumps(formal))
            method = 'physmani'
            item = dict(method=method, task=demo.TASKS[0], raw_episode_id='episode2',
                        name=demo.TASKS[0] + '__episode2', queue='gpu', reuse_formal_video=False)
            class FakeChild:
                pid = 12345
                def __init__(self, command, **kwargs):
                    child = root / command[command.index('--out') + 1]
                    assert child.is_dir() and not any(child.iterdir()), 'baseline child --out must start empty'
                    record = dict(official_task_success=False, policy_failure='fixed first case failed',
                                  evaluation_process_pid=12346, simulation_seconds=10.)
                    (child / (item['name'] + '.json')).write_text(json.dumps(record))
                    (child / (item['task'] + '_inputs.json')).write_text('{"sha256":"same"}')
                    (child / (item['name'] + '.mp4')).write_bytes(b'mock encoded video')
                    (child / 'demo_receipt.json').write_text('{"demo_only":true}')
                def wait(self): return 0
            def checked(item, child, original, original_inputs):
                path = child / (item['name'] + '.json')
                return (path, json.loads(path.read_text()), child / (item['task'] + '_inputs.json'),
                    child / (item['name'] + '.mp4'), {'decoded_frame_count': 201},
                    {'accepted': True, 'problems': []})
            with patch.object(demo, 'ROOT', root), patch.object(demo, 'BASE', base), \
                    patch.object(demo, 'DEMO_BASE', out), patch.object(demo, 'verify_snapshot'), \
                    patch.object(demo, 'formal_source', return_value=(formal_path, formal, {'sha256': 'same'})), \
                    patch.object(demo, 'validate_replay', side_effect=checked), \
                    patch.object(demo.subprocess, 'Popen', side_effect=FakeChild) as launch:
                first = demo.run_replay(item, {}, threading.Event())
                assert first['formal_result']['official_task_success'] is True
                assert first['replay_result']['official_task_success'] is False
                assert first['replay_included_in_formal_statistics'] is False
                assert first['source'] == 'independent_demo_only_replay'
                second = demo.run_replay(item, {}, threading.Event())
                assert second['video_sha256'] == first['video_sha256']
                assert launch.call_count == 1
                (out / 'demo_execution_inputs.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'cached demo recording identity differs'):
                    demo.run_replay(item, {}, threading.Event())
                assert launch.call_count == 1

    def test_cancelled_pending_demo_does_not_launch(self):
        event = threading.Event();event.set()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_path = root / 'formal.json';formal_path.write_text('{}')
            item = dict(method='cv_geometry', task=demo.TASKS[0], raw_episode_id='episode2', name='fixedcase')
            with patch.object(demo, 'DEMO_BASE', root / 'demo'), patch.object(demo, 'verify_snapshot'), \
                    patch.object(demo, 'formal_source', return_value=(formal_path, {}, {})), \
                    patch.object(demo.subprocess, 'Popen') as launch:
                with self.assertRaises(InterruptedError): demo.run_replay(item, {}, event)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
