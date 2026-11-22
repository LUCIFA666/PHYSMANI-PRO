"""CPU boundaries for the new geometry-only backend and TRAIN entrypoint."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import servo_response_actor_backend as backend
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.interaction import (InteractionLibrary, CandidateConfig, MPCConfig,
    CandidateGenerator, SE3InteractionMPC, InteractionController, GripperTimingConfig)
from our_mani.multitask.online_servo_response import servo_response_binding
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.reachability import TimedReachability
from scripts import run_geometry_servo_response_smoke_r10 as smoke
from tests.test_feedback_interaction import GateModel

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'artifacts/framework/multitask_r10/geometry_feedback_smoke_slack2_v1_setup'
TASK = 'moving_basketball_in_hoop'


def prepared(method='cv_geometry'):
    configuration = json.loads((SETUP / 'configuration.json').read_text())
    original_feedback = configuration['policy_overlay']['configuration']
    configuration['policy_overlay'] = backend.servo_policy_overlay(ROOT, original_feedback, {})
    library = InteractionLibrary.load(backend.factory.artifact_path(ROOT,
        configuration['learned_artifacts'][TASK]['interaction']))
    identity = backend.servo_continuation_identity(ROOT, configuration, TASK, method.split('_')[0],
        library, CandidateConfig(**configuration['candidate_config']), MPCConfig(**configuration['mpc_config']))
    configuration['continuation_identities'][TASK][method.split('_')[0]] = identity['sha256']
    files = json.loads((SETUP / 'frozen_inputs.json').read_text())['files']
    for source in configuration['policy_overlay']['sources'].values():
        files[str(ROOT / source['path'])] = source['sha256']
    return dict(args=SimpleNamespace(method=method, task=TASK), config=configuration, files=files), library, identity


class BackendTests(unittest.TestCase):
    def test_overlay_binds_both_normalized_configurations_and_three_actual_sources(self):
        value, library, identity = prepared()
        protocol = value['config']; overlay = protocol['policy_overlay']
        self.assertEqual(set(overlay['sources']), {'actor_backend', 'feedback_controller', 'servo_response'})
        self.assertEqual(overlay['configurations']['feedback']['phase_tracking_slack_seconds'], 2.)
        self.assertEqual(overlay['configurations']['feedback']['acceleration_fraction'], .5)
        self.assertEqual(overlay['configurations']['servo_response'], servo_response_binding({})['normalized_config'])
        for source in overlay['sources'].values():
            self.assertEqual(file_sha(ROOT / source['path']), source['sha256'])
        self.assertEqual(backend.verify_servo_policy_overlay(ROOT, protocol), overlay)
        changed = deepcopy(protocol)
        changed['policy_overlay'] = backend.servo_policy_overlay(ROOT,
            dict(overlay['configurations']['feedback'], phase_tracking_slack_seconds=0.), {})
        changed_identity = backend.servo_continuation_identity(ROOT, changed, TASK, 'cv', library,
            CandidateConfig(**changed['candidate_config']), MPCConfig(**changed['mpc_config']))
        self.assertNotEqual(identity['sha256'], changed_identity['sha256'])
        for key in overlay['sources']:
            altered = deepcopy(protocol); altered['policy_overlay']['sources'][key]['sha256'] = '0' * 64
            with self.assertRaises(MultitaskProtocolError): backend.verify_servo_policy_overlay(ROOT, altered)
        altered = deepcopy(protocol); altered['simulation_seconds'] = 12.
        with self.assertRaises(MultitaskProtocolError): backend.verify_servo_policy_overlay(ROOT, altered)

    def test_actual_cv_and_context_cpu_loaders_require_new_identity_and_locked_sources(self):
        for method in backend.METHODS:
            value, _, identity = prepared(method)
            result = backend.servo_actor_model_preflight(ROOT, value)
            self.assertEqual(result['continuation_identity'], identity)
            self.assertTrue(result['all_loaders_CPU_only']); self.assertFalse(result['native_started'])
            self.assertEqual(result['context_checkpoint_sha256'] is not None, method == 'context_geometry')
            altered = deepcopy(value)
            altered['config']['continuation_identities'][TASK][method.split('_')[0]] = identity['original_continuation']['sha256']
            with self.assertRaises(MultitaskProtocolError): backend.servo_actor_model_preflight(ROOT, altered)
            altered = deepcopy(value)
            source = value['config']['policy_overlay']['sources']['servo_response']['path']
            del altered['files'][str(ROOT / source)]
            with self.assertRaises(MultitaskProtocolError): backend.servo_actor_model_preflight(ROOT, altered)

    def test_real_installers_run_in_required_order_before_first_act(self):
        value, library, identity = prepared()
        protocol = value['config']; model = GateModel(); calls = []
        def original_actor(root, env, base_protocol, method, descriptions, **kwargs):
            self.assertEqual(method, 'cv_geometry')
            self.assertEqual(base_protocol['continuation_identities'][TASK]['cv'], identity['original_continuation']['sha256'])
            candidate = CandidateConfig(**protocol['candidate_config'])
            collision = lambda **kw: dict(feasible=True, evidence={})
            collision.bind_candidate = lambda **kw: None
            generator = CandidateGenerator(library, TimedReachability(model), collision, candidate)
            controller = InteractionController(library, generator, SE3InteractionMPC(MPCConfig(**protocol['mpc_config'])),
                predictor='cv', ranking='geometry', continuation_sha256=identity['original_continuation']['sha256'],
                gripper_timing_config=GripperTimingConfig(**protocol['gripper_timing_config']))
            actor = SimpleNamespace(controller=controller, robot_model=model,
                act=lambda *args: dict(command='hold'), metrics_report=lambda: dict(original_metrics=True))
            calls.append(actor)
            return actor, identity['original_continuation']
        with patch.object(backend.factory, 'create_actor', side_effect=original_actor):
            actor, extended = backend.create_servo_response_actor(ROOT,
                SimpleNamespace(spec=SimpleNamespace(task_name=TASK)), protocol, 'cv_geometry', [])
        self.assertEqual(len(calls), 1)
        self.assertIs(type(actor.controller), ObservedFeedbackController)
        self.assertEqual(actor.controller.continuation_sha256, extended['sha256'])
        metrics = actor.metrics_report()
        self.assertTrue(metrics['original_metrics'])
        self.assertEqual(metrics['observed_feedback_controller']['step_calls'], 0)
        self.assertEqual(metrics['online_servo_response']['normalized_config'],
            protocol['policy_overlay']['configurations']['servo_response'])
        self.assertEqual(metrics['online_servo_response']['command_events'], [])
        self.assertFalse(metrics['online_servo_response']['actual_tracking_guaranteed'])

    def test_backend_rejects_removed_methods_before_loading_models_or_creating_actor(self):
        value, _, _ = prepared()
        with patch.object(backend.factory, 'create_actor') as create:
            for method in ('cv_value', 'context_value', 'physmani'):
                value['args'].method = method
                with self.assertRaises(MultitaskProtocolError): backend.servo_actor_model_preflight(ROOT, value)
                with self.assertRaises(MultitaskProtocolError):
                    backend.create_servo_response_actor(ROOT, None, value['config'], method, [])
            create.assert_not_called()

    def test_wrapper_cli_and_preflight_cannot_enter_formal_or_fitting_scope(self):
        required = ['--config', 'x', '--manifest', 'x', '--frozen-inputs', 'x', '--task', TASK,
            '--raw-episode-id', '104', '--out', 'x', '--split', 'train', '--method', 'cv_geometry', '--development-demo']
        args = smoke.parser().parse_args(required)
        for split in ('validation', 'test'):
            arguments = required.copy(); arguments[arguments.index('--split') + 1] = split
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): smoke.parser().parse_args(arguments)
        for method in ('cv_value', 'context_value', 'physmani'):
            arguments = required.copy(); arguments[arguments.index('--method') + 1] = method
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): smoke.parser().parse_args(arguments)
        for updates in ({'split':'test'}, {'development_demo':False}, {'replica':1}, {'exploration_seed':0}):
            altered = deepcopy(args)
            for key, value in updates.items(): setattr(altered, key, value)
            with patch.object(smoke.evaluator, 'preflight') as original, self.assertRaises(MultitaskProtocolError):
                smoke.preflight(altered)
            original.assert_not_called()


if __name__ == '__main__':
    unittest.main()
