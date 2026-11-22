"""No Native/model/raw calls: transparent original admission and branch trees."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('diagnostic1103_cpu', 'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py')
tap = load('tap934_cpu_for1103', 'our_mani/multitask/train_public_collision_tap_v934.py')


class TransparentAdmission(unittest.TestCase):
    def test_exact_once_return_identity_and_restore(self):
        calls = []
        result = (False, {'reason': 'original'})
        class Executor:
            def admit_joint_target(self, q):
                calls.append(q)
                return result
        executor = Executor()
        original = executor.admit_joint_target
        observer = tap.TrainPublicCollisionTap(enabled=False)
        observer._thread = __import__('threading').get_ident()
        observer._snapshot = lambda: {}
        observer._patch(executor, 'admit_joint_target', original,
                        'executor.admit_joint_target', {'kind': 'existing_callback'})
        q = object()
        self.assertIs(executor.admit_joint_target(q), result)
        self.assertEqual(calls, [q])
        self.assertEqual(observer.events[0]['original_calls_performed'], 1)
        self.assertEqual(observer.events[0]['extra_queries_performed'], 0)
        observer._restore()
        self.assertNotIn('admit_joint_target', executor.__dict__)

    def test_original_exception_identity_not_replaced(self):
        error = RuntimeError('original')
        class Executor:
            def admit_joint_target(self, q):
                raise error
        executor = Executor()
        observer = tap.TrainPublicCollisionTap(enabled=False)
        observer._thread = __import__('threading').get_ident()
        observer._snapshot = lambda: {}
        observer._patch(executor, 'admit_joint_target', executor.admit_joint_target,
                        'executor.admit_joint_target', {})
        with self.assertRaises(RuntimeError) as caught:
            executor.admit_joint_target([])
        self.assertIs(caught.exception, error)
        observer._restore()


def event(i, parent, op, result, **kw):
    return dict(call_id=i, parent_call_id=parent, operation=op, result=result, returned=True, **kw)


class BranchTrees(unittest.TestCase):
    def summary(self, inner, *, dropped=0):
        proof = {'reason': 'public_forecast_collision_on_RML_target_sweep', 'failed_fraction': 0.}
        root = event(1, None, 'executor.admit_joint_target', [False, proof])
        return m.summarize_collision(dict(events=[root] + inner, dropped_events=dropped, error_count=0))

    def test_interval_and_unavailable_are_separate(self):
        self.assertEqual(self.summary([event(2, 1, 'model.collision', True)])['branch_counts'],
                         {'model_current_interval_precondition': 1})
        self.assertEqual(self.summary([event(2, 1, 'model.collision', True),
                                      event(3, 2, 'model.collision_query', True)])['branch_counts'],
                         {'public_prediction_or_geometry_unavailable_fail_closed': 1})

    def test_arm_and_held_world_are_separate(self):
        common = [event(2, 1, 'model.collision', True), event(3, 2, 'model.collision_query', True),
                  event(4, 3, 'map.check', True, arguments={'kwargs': {'geometry_modes': {'ring': 'primary'}}}),
                  event(5, 4, 'map.public_scene', {'status': 'unknown'})]
        arm = self.summary(common + [event(6, 4, 'arm.check_arm_collision', True)])
        self.assertEqual(arm['branch_counts'], {'arm_collision_collection': 1})
        held = self.summary(common + [event(6, 4, 'arm.check_arm_collision', False),
                     event(7, 4, 'registered_ghost.check_collision', True,
                           registered_receiver={'registered_roles': [{'role': 'ring'}]}, arguments={'args': []})])
        self.assertEqual(held['branch_counts'], {'held_world_registered_ghost_collision': 1})
        self.assertFalse(held['admissions'][0]['physical_collision_claimed'])

    def test_incomplete_tree_stays_unknown(self):
        s = self.summary([event(2, 1, 'model.collision', True)], dropped=1)
        self.assertEqual(s['branch_counts'], {'unknown_incomplete_call_tree': 1})

    def test_phase_output_reuse_does_not_allow_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107/cases/'
            p = Path(d) / (prefix + 'case0')
            (p / 'CPU_check').mkdir(parents=True)
            self.assertEqual(m._fresh_path(d, prefix + 'case0', prefix, allow_existing_dir=True), p)
            with self.assertRaises(ValueError):
                m._fresh_path(d, prefix + 'case0', prefix)


class CanonicalResourceLease(unittest.TestCase):
    def test_real924_registry_lifetime_and_exclusive_lock(self):
        guard = load('guard1106_cpu', 'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py')
        from our_mani.multitask import post_TEST_resource_guard_v924 as legacy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'runtime/framework').mkdir(parents=True)
            proof = dict(complete=True, known_processes=[], CPU_test_only=True)
            inventory = dict(host_visibility_available=True, ancestor_chain_complete=True,
                             same_uid_inventory_incomplete=False, live_project_native_processes=[])
            # All completion/inventory reads are replaced with synthetic CPU
            # evidence; native observers are never imported or entered here.
            with patch.object(guard, '_legacy', return_value=legacy), \
                 patch.object(guard, 'require_complete_TEST_metadata', return_value=proof), \
                 patch.object(guard, '_inventory', return_value=inventory):
                with guard.reserved_resources_after_TEST(root, enabled=True) as permit:
                    self.assertTrue(legacy.require_permit(permit, root)['complete'])
                    with self.assertRaises(RuntimeError):
                        with guard._reserve(root):
                            self.fail('exclusive resource lock was not held')
                with self.assertRaises(ValueError):
                    legacy.require_permit(permit, root)


if __name__ == '__main__':
    unittest.main()
