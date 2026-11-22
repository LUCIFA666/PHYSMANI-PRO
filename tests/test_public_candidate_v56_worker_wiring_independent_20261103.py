"""Independent worker/archival CPU routing; every Native boundary is mocked."""
from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import our_mani.local_mpc_20261102.worker as worker
from our_mani.local_mpc_20261102.records import freeze_sources, sha
import our_mani.local_mpc_control_20261103.public_candidate_goal_maintenance_v56 as v56
import our_mani.local_mpc_control_20261103.public_grasp_candidates_v5 as v5
import our_mani.local_mpc_control_20261103.public_grasp_v4 as v4
import our_mani.local_mpc_control_20261103.public_grasp_v42 as v42

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT/'config/framework/P4_student_closedloop_devVAL_v55_v1/00_val_static.json'
EXTRA = {'our_mani/local_mpc_control_20261103/'+name for name in (
    'public_candidate_goal_maintenance_v56.py', 'postconfirmed_maintenance_candidate_20261103.py',
    'public_grasp_v42.py', 'goal_aware_binding_v42.py', 'goal_aware_perception.py')}


class FakeRaw:
    initial_snapshot = {'scope': 'CPU_fake_no_Native'}
    def __init__(self): self.closes = 0
    def close(self): self.closes += 1


class FakeObserver:
    def __init__(self, raw): self.raw = raw; self.samples = 0
    def sample(self): self.samples += 1
    def close(self): self.raw.close()


class V56WorkerWiringTests(unittest.TestCase):
    def config(self, **overrides):
        # No weights or recorded case data are opened by the mock worker.
        result = dict(public_grasp_candidate_transformer_v5=True,
            candidate_transformer_checkpoint='CPU_fake_checkpoint_not_opened.pt',
            candidate_transformer_checkpoint_sha256='0'*64)
        result.update(overrides)
        return result

    def mocked_worker(self, config, mode='p1'):
        raw = FakeRaw(); constructors = []; runners = []
        def constructor(name):
            def build(backend, *args, **kwargs):
                observer = FakeObserver(backend)
                constructors.append((name, kwargs, observer))
                return observer
            return build
        def run(name):
            def call(backend, executor, output, cfg, **kwargs):
                runners.append((name, backend))
                return dict(complete=True, mocked_CPU_only=True)
            return call
        with TemporaryDirectory(prefix='v56_worker_CPU_') as directory, ExitStack() as stack:
            root = Path(directory); config_path = root/'config.json'; config_path.write_text(json.dumps(config))
            output = root/'runtime/framework/local_mpc_20261102/explicit_CPU_mock_case'
            stack.enter_context(patch.object(worker, 'ROOT', root))
            stack.enter_context(patch.object(worker, 'freeze_sources', return_value={}))
            stack.enter_context(patch.object(worker, 'archive_sources'))
            stack.enter_context(patch.object(worker, 'native_source_snapshot', return_value={'CPU': 'unchanged'}))
            stack.enter_context(patch.object(worker, 'dependency_receipt', return_value={'CPU': True}))
            stack.enter_context(patch.object(worker, 'environment_proof', return_value={'CPU': True}))
            native = stack.enter_context(patch.object(worker, 'make_backend', return_value=raw))
            stack.enter_context(patch.object(worker, 'make_executor', return_value=SimpleNamespace(last_receipt=None)))
            stack.enter_context(patch.object(worker, 'append_ticks'))
            stack.enter_context(patch.object(worker, 'verify_sources'))
            stack.enter_context(patch.object(worker, 'FrameRecorder', return_value=SimpleNamespace(close=lambda: None)))
            stack.enter_context(patch.object(v4, 'ActiveVisualPublicBridge', return_value=SimpleNamespace(CPU=True)))
            stack.enter_context(patch.object(v5, 'ObservedPublicCandidateV5Backend', side_effect=constructor('v5')))
            stack.enter_context(patch.object(v56, 'ObservedPublicCandidateV56Backend', side_effect=constructor('v56')))
            old42 = stack.enter_context(patch.object(v42, 'ObservedPublicGraspV42Backend', side_effect=AssertionError('old v42 route forbidden')))
            stack.enter_context(patch.object(v5, 'run_public_grasp_candidates_v5', side_effect=run('v5')))
            stack.enter_context(patch.object(v56, 'run_public_candidate_goal_maintenance_v56', side_effect=run('v56')))
            old42run = stack.enter_context(patch.object(v42, 'run_public_grasp_v42', side_effect=AssertionError('old v42 runner forbidden')))
            with redirect_stderr(io.StringIO()):
                rc = worker.main(['--mode', mode, '--scene', 'static', '--output', str(output), '--config', str(config_path)])
            report = dict(rc=rc, constructors=constructors, runners=runners, closes=raw.closes,
                native_calls=native.call_count, old42_calls=old42.call_count+old42run.call_count,
                exit=json.loads((output/'worker_exit.json').read_text()))
            if (output/'run_inputs.json').exists(): report['inputs'] = json.loads((output/'run_inputs.json').read_text())
            if (output/'failure.json').exists(): report['failure'] = json.loads((output/'failure.json').read_text())
            return report

    def test_default_and_explicit_false_construct_and_dispatch_only_original_v5(self):
        for options in ({}, dict(candidate_goal_sphere_filter_v56=False, postconfirmed_position_maintenance_v56=False)):
            with self.subTest(options=options):
                report = self.mocked_worker(self.config(**options))
                self.assertEqual(report['rc'], 0)
                self.assertEqual([name for name, _, _ in report['constructors']], ['v5'])
                self.assertEqual([name for name, _ in report['runners']], ['v5'])
                self.assertEqual(report['inputs']['information_setting'], 'public_candidate_transformer_v5_development')
                self.assertEqual(report['closes'], 1); self.assertTrue(report['exit']['native_closed'])
                self.assertEqual(report['old42_calls'], 0)

    def test_each_modifier_selects_v56_and_passes_exact_flags_to_observer(self):
        for goal, maintenance in ((True, False), (False, True), (True, True)):
            with self.subTest(goal=goal, maintenance=maintenance):
                report = self.mocked_worker(self.config(candidate_goal_sphere_filter_v56=goal,
                    postconfirmed_position_maintenance_v56=maintenance))
                self.assertEqual(report['rc'], 0)
                self.assertEqual([name for name, _, _ in report['constructors']], ['v56'])
                self.assertEqual([name for name, _ in report['runners']], ['v56'])
                name, options, observer = report['constructors'][0]
                self.assertIs(options['goal_sphere_filter'], goal)
                self.assertIs(options['postconfirmed_maintenance'], maintenance)
                self.assertIs(report['runners'][0][1], observer)
                self.assertEqual(observer.samples, 1)
                self.assertEqual(report['old42_calls'], 0)
                self.assertEqual(report['inputs']['information_setting'], 'public_candidate_transformer_v56_development')
                self.assertEqual(report['closes'], 1)

    def test_bad_types_non_candidate_non_p1_and_v42_combination_reject_before_backend(self):
        cases = [
            (self.config(candidate_goal_sphere_filter_v56=1), 'p1'),
            (self.config(postconfirmed_position_maintenance_v56=None), 'p1'),
            (self.config(candidate_goal_sphere_filter_v56=True, public_grasp_candidate_transformer_v5=False), 'p1'),
            (self.config(candidate_goal_sphere_filter_v56=True, public_grasp_candidate_transformer_v5=1), 'p1'),
            (self.config(postconfirmed_position_maintenance_v56=True), 'p0'),
            (self.config(candidate_goal_sphere_filter_v56=True, public_grasp_active_visual_v42=True), 'p1'),
        ]
        for config, mode in cases:
            with self.subTest(config=config, mode=mode):
                report = self.mocked_worker(config, mode)
                self.assertEqual(report['rc'], 1)
                self.assertEqual(report['failure']['type'], 'ValueError')
                self.assertEqual(report['native_calls'], 0)
                self.assertEqual(report['constructors'], [])
                self.assertEqual(report['runners'], [])
                self.assertFalse(report['exit']['native_closed'])

    def test_real_inventory_adds_exact_same_five_for_each_true_combination(self):
        config = json.loads(BASE_CONFIG.read_text())
        with TemporaryDirectory(prefix='v56_source_CPU_', dir=ROOT/'config/framework') as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(config)); original = freeze_sources(path, 'p1')
            explicit = deepcopy(config); explicit.update(candidate_goal_sphere_filter_v56=False,
                postconfirmed_position_maintenance_v56=False)
            path.write_text(json.dumps(explicit)); disabled = freeze_sources(path, 'p1')
            self.assertEqual(set(disabled), set(original))
            own_config = str(path.relative_to(ROOT))
            self.assertEqual({k:v for k,v in disabled.items() if k != own_config},
                {k:v for k,v in original.items() if k != own_config})
            for goal, maintenance in ((True, False), (False, True), (True, True)):
                config.update(candidate_goal_sphere_filter_v56=goal, postconfirmed_position_maintenance_v56=maintenance)
                path.write_text(json.dumps(config)); enabled = freeze_sources(path, 'p1')
                self.assertEqual(set(enabled)-set(original), EXTRA)
                self.assertEqual(set(original)-set(enabled), set())
                for name in EXTRA:
                    self.assertEqual(enabled[name], sha(ROOT/name))
            bad = deepcopy(config); bad['public_grasp_candidate_transformer_v5'] = False
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'candidate route'):
                freeze_sources(path, 'p1')
            bad = deepcopy(config); bad['postconfirmed_position_maintenance_v56'] = 'true'
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'boolean'):
                freeze_sources(path, 'p1')


if __name__ == '__main__':
    unittest.main()
