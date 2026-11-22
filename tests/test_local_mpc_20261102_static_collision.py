"""CPU-only failure-injection verification; no Native / GPU imports."""
from types import SimpleNamespace
import unittest

from our_mani.local_mpc_20261102.static_collision import (
    StaticCollisionClockError, StaticCollisionRestoreError, check_static_arm_collision,
)


class FakeShape:
    def __init__(self, flag):
        self.flag = flag
        self.calls = []
        self.fail_once_after_set = False
        self.fail_restore_value = None

    def is_collidable(self):
        return self.flag

    def set_collidable(self, value):
        self.calls.append(value)
        if self.fail_restore_value is not None and value == self.fail_restore_value:
            raise RuntimeError("injected restoration failure")
        self.flag = value
        if self.fail_once_after_set:
            self.fail_once_after_set = False
            raise RuntimeError("injected setup failure after mutation")

    def get_pose(self, *args, **kwargs):
        raise AssertionError("collision query must never read a target world pose")

    def set_pose(self, *args, **kwargs):
        raise AssertionError("collision query must never write a target world pose")

    def set_position(self, *args, **kwargs):
        raise AssertionError("collision query must never write target position")

    def set_quaternion(self, *args, **kwargs):
        raise AssertionError("collision query must never write target orientation")


class FakeArm:
    def __init__(self, callback):
        self.callback = callback
        self.queries = 0

    def check_arm_collision(self):
        self.queries += 1
        return self.callback()

    def set_joint_positions(self, *args, **kwargs):
        raise AssertionError("flag helper must never write robot joints")


def fixture(callback=None):
    clock = [0.0, 0.0, 0, 0]
    service = SimpleNamespace(static=[FakeShape(True), FakeShape(False)],
                              ghosts=[FakeShape(False), FakeShape(True)],
                              target=FakeShape(True), queries=0)

    def query():
        assert all(not shape.flag for shape in service.static)
        assert all(shape.flag for shape in service.ghosts)
        assert service.target.flag is False
        return callback(clock) if callback else True

    service.env = SimpleNamespace(arm=FakeArm(query))
    service.check = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy check is forbidden"))
    service.env.pyrep = SimpleNamespace(step=lambda: (_ for _ in ()).throw(AssertionError("physics step forbidden")))
    shapes = service.static + service.ghosts + [service.target]
    return service, shapes, lambda: tuple(clock)


class StaticCollisionQueryTests(unittest.TestCase):
    def test_success_uses_only_flags_and_restores_original_ghost_values(self):
        service, shapes, clock = fixture()
        before = [shape.flag for shape in shapes]
        receipt = {}
        self.assertTrue(check_static_arm_collision(service, clock=clock, receipt=receipt))
        self.assertEqual([shape.flag for shape in shapes], before)
        self.assertEqual(service.queries, 1)
        self.assertEqual(service.env.arm.queries, 1)
        self.assertTrue(receipt["flags_restored"])
        self.assertTrue(receipt["clock_unchanged"])
        self.assertEqual(receipt["physical_steps"], 0)
        self.assertFalse(receipt["target_pose_read"])
        self.assertFalse(receipt["target_pose_setter_called"])
        self.assertFalse(receipt["physics_step_called"])

    def test_native_query_exception_restores_every_flag_and_propagates(self):
        def fail(clock):
            raise RuntimeError("injected native collision failure")
        service, shapes, clock = fixture(fail)
        before = [shape.flag for shape in shapes]
        receipt = {}
        with self.assertRaisesRegex(RuntimeError, "injected native collision failure"):
            check_static_arm_collision(service, clock=clock, receipt=receipt)
        self.assertEqual([shape.flag for shape in shapes], before)
        self.assertTrue(receipt["flags_restored"])
        self.assertTrue(receipt["clock_unchanged"])
        self.assertEqual(service.queries, 1)

    def test_setup_exception_after_mutation_still_restores_without_native_query(self):
        service, shapes, clock = fixture()
        before = [shape.flag for shape in shapes]
        service.ghosts[0].fail_once_after_set = True
        receipt = {}
        with self.assertRaisesRegex(RuntimeError, "injected setup failure"):
            check_static_arm_collision(service, clock=clock, receipt=receipt)
        self.assertEqual([shape.flag for shape in shapes], before)
        self.assertTrue(receipt["flags_restored"])
        self.assertEqual(service.env.arm.queries, 0)
        self.assertEqual(service.queries, 0)

    def test_restore_failure_does_not_skip_remaining_restoration(self):
        service, shapes, clock = fixture()
        before = [shape.flag for shape in shapes]
        service.target.fail_restore_value = True
        receipt = {}
        with self.assertRaises(StaticCollisionRestoreError) as error:
            check_static_arm_collision(service, clock=clock, receipt=receipt)
        self.assertEqual([shape.flag for shape in shapes[:-1]], before[:-1])
        self.assertFalse(service.target.flag)
        self.assertFalse(receipt["flags_restored"])
        self.assertTrue(error.exception.clock_unchanged)
        self.assertGreaterEqual(len(error.exception.failures), 1)

    def test_unexpected_clock_advance_is_detected_after_restoring_flags(self):
        def advance(clock):
            clock[0] += .05
            clock[1] += .05
            clock[2] += 1
            clock[3] += 1
            return False
        service, shapes, clock = fixture(advance)
        before = [shape.flag for shape in shapes]
        receipt = {}
        with self.assertRaises(StaticCollisionClockError):
            check_static_arm_collision(service, clock=clock, receipt=receipt)
        self.assertEqual([shape.flag for shape in shapes], before)
        self.assertTrue(receipt["flags_restored"])
        self.assertFalse(receipt["clock_unchanged"])
        self.assertIsNone(receipt["physical_steps"])

    def test_aliasing_shape_roles_is_rejected_before_any_write(self):
        service, shapes, clock = fixture()
        service.target = service.static[0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            check_static_arm_collision(service, clock=clock)
        self.assertFalse(any(shape.calls for shape in shapes))
        self.assertEqual(service.env.arm.queries, 0)


if __name__ == "__main__":
    unittest.main()
