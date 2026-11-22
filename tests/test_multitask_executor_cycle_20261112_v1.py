"""Mechanical sequencing / bounded retiming; no Native/weights required."""
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.multitask_executor_cycle_20261112_v1.sequencing import (
    SequenceConfig, arrival_error, requested_opening_after_arrival, retime_reference_path)
from our_mani.multitask_executor_cycle_20261112_v1.backend import TaskExecutionBackend
from our_mani.execution_interface_20261111_v1.executor import PhysManiExecutor
from our_mani.local_mpc_20261102.executor import JointLimits


class MechanicalSequencing(unittest.TestCase):
    def reference(self, opening=0.):
        return SimpleNamespace(position=np.zeros(3), quaternion=np.array([0., 0., 0., 1.]), opening_intent=opening)

    def test_neither_path_completion_nor_elapsed_time_grants_gripper_intent(self):
        opening, audit = requested_opening_after_arrival(self.reference(), [.1, 0, 0, 0, 0, 0, 1], 1., SequenceConfig())
        self.assertEqual(opening, 1.)
        self.assertFalse(audit['measured_arm_arrived'])
        self.assertFalse(audit['task_success_inferred'])

    def test_actual_arrival_authorizes_both_close_and_release(self):
        for desired in (0., 1.):
            opening, audit = requested_opening_after_arrival(self.reference(desired), [0, 0, 0, 0, 0, 0, 1], 1.-desired, SequenceConfig())
            self.assertEqual(opening, desired)
            self.assertTrue(audit['measured_arm_arrived'])
            self.assertFalse(audit['object_alignment_proven'])

    def test_orientation_must_arrive_and_quaternion_sign_is_equivalent(self):
        opening, audit = requested_opening_after_arrival(self.reference(), [0, 0, 0, 1, 0, 0, 0], 1., SequenceConfig())
        self.assertEqual(opening, 1.)
        self.assertFalse(audit['measured_arm_arrived'])
        self.assertEqual(arrival_error(self.reference(), [0, 0, 0, 0, 0, 0, -1]), (0., 0.))

    def test_first_true_success_is_latched_even_without_terminate(self):
        value = SimpleNamespace(native_task_success_latched=False, native_success_first_time=None)
        current = {'success': True, 'physical_timestamp': .05}
        value.metric_receipt = lambda: {'native_current': current}
        TaskExecutionBackend._latch(value)
        current.update(success=False, physical_timestamp=.1)
        TaskExecutionBackend._latch(value)
        self.assertTrue(value.native_task_success_latched)
        self.assertEqual(value.native_success_first_time, .05)

    def test_retiming_preserves_polyline_and_enforces_unchanged_discrete_limits(self):
        executor = PhysManiExecutor.__new__(PhysManiExecutor)
        executor._last_direct = None
        executor.config = SimpleNamespace(physics_dt=.05, clock_tolerance=2e-5)
        executor.limits = JointLimits(lower=[-3.]*7, upper=[3.]*7, velocity=[2.]*7, acceleration=12., jerk=1000.)
        raw = [np.full(7, .001), np.full(7, .04), np.full(7, .08), np.full(7, .081)]
        points, receipt = retime_reference_path(raw, executor, np.zeros(7), np.zeros(7), 0.)
        self.assertGreater(receipt['time_dilation'], 1.)
        self.assertFalse(receipt['limits_relaxed'])
        self.assertTrue(np.array_equal(points[-1], raw[-1]))
        self.assertTrue(all(np.allclose(x, x[0]) for x in points))
        references = executor._trajectory_references(SimpleNamespace(positions=points), np.zeros(7), np.zeros(7), 0.)
        self.assertTrue(np.array_equal(references[-1][1], np.zeros(7)))
        self.assertTrue(np.array_equal(references[-1][2], np.zeros(7)))

    def test_unbounded_or_nonfinite_inputs_do_not_generate_actions(self):
        for value in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                SequenceConfig(arrival_position_m=value)
        with self.assertRaises(ValueError):
            arrival_error(self.reference(), [0, 0, 0, 0, 0, 0, 0])

    def test_new_small_plan_separates_all_cached_parents_and_native_reset_roles(self):
        from our_mani.multitask_executor_cycle_20261112_v1.common import PLAN, read
        plan = read(PLAN)
        self.assertEqual(len(plan['tasks']), 14)
        parents = plan['cached_TRAIN']
        self.assertEqual(len(parents), 56)
        self.assertEqual(len({p['source_parent_group'] for p in parents}), 56)
        self.assertTrue(all(p['source_split'] == 'train' for p in parents))
        self.assertEqual(len({c['seed'] for c in plan['cases']}), 84)
        self.assertTrue(all(c['budget_ticks'] == 1200 for c in plan['cases']))

    def test_head_cache_decoder_matches_full_world_model_math_and_trains_only_heads(self):
        import torch
        from our_mani.world_tcp_command_warmstart_20261104_v3.model import WorldTCPCommandTransformerV3, CandidateConfig
        from our_mani.multitask_executor_cycle_20261112_v1.fit import HeadDecoder
        torch.set_num_threads(2)
        torch.manual_seed(112)
        model = WorldTCPCommandTransformerV3(CandidateConfig(width=24, layers=1, heads=4, dropout=0.)).eval()
        with torch.no_grad():
            for head in (model.position_head, model.rotation_head, model.opening_head, model.demo_likelihood_head):
                head.weight.normal_(std=.01)
        robot = torch.zeros(1, 14)
        robot[:, -1] = 1.
        calibration = torch.zeros(1, 4, 23)
        calibration[:, :, :9] = torch.eye(3).flatten()
        calibration[:, :, 9:21] = torch.eye(4)[:3].flatten()
        calibration[:, :, 21:] = torch.tensor([.01, 4.])
        inputs = dict(visual_memory=torch.randn(1, 86, 1024), visual_memory_mask=torch.ones(1, 86, dtype=torch.bool),
                      current_robot=robot, camera_calibration=calibration, task_e=torch.full((1, 16), .125),
                      fine_visual=torch.randn(1, 512, 512), fine_geometry=torch.zeros(1, 512, 6),
                      fine_token_mask=torch.ones(1, 512, dtype=torch.bool))
        captured = []
        handle = model.output_norm.register_forward_hook(lambda m, args, output: captured.append(output.detach()))
        with torch.no_grad():
            expected = model(**inputs)
        handle.remove()
        model.requires_grad_(False)
        decoder = HeadDecoder(model).requires_grad_(True)
        actual = decoder(captured[0], robot)
        for key in actual:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        actual['world_tcp_positions'].sum().backward()
        self.assertIsNotNone(model.position_head.weight.grad)
        self.assertIsNone(model.transformer.layers[0].linear1.weight.grad)


if __name__ == '__main__':
    unittest.main()
