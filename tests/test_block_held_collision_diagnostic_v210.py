from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from our_mani.multitask.block_held_collision_diagnostic_v210 import SmoothBlockEnvironment,OriginalBoundary


class Tests(unittest.TestCase):
    def fixture(self, result):
        calls=[]
        def original(offset):calls.append(offset);return result
        boundary=SmoothBlockEnvironment.__new__(SmoothBlockEnvironment)
        boundary.env=NS(robot_model=NS(collision_query=original),ledger=NS(ticks=3),time=.15)
        boundary.public_robot=NS(contact=True)
        boundary.held_collision_diagnostics=[];boundary.held_collision_diagnostic_candidates=0
        boundary._diagnose=lambda offset:dict(private_metric='never-returned-to-policy')
        return boundary,calls,original

    @staticmethod
    @contextmanager
    def base_scope(instance):yield

    def test_original_collision_bool_and_call_count_unchanged(self):
        for result in (True,False):
            b,calls,original=self.fixture(result)
            with patch.object(OriginalBoundary,'_collision_scope',self.base_scope):
                with b._collision_scope():self.assertIs(b.env.robot_model.collision_query(.2),result)
            self.assertEqual(calls,[.2])
            self.assertIs(b.env.robot_model.collision_query,original)

    def test_diagnostic_errors_do_not_change_original_collision_result(self):
        b,calls,original=self.fixture(True)
        def fail(offset):raise ValueError('metric unavailable')
        b._diagnose=fail
        with patch.object(OriginalBoundary,'_collision_scope',self.base_scope):
            with b._collision_scope():self.assertTrue(b.env.robot_model.collision_query(.2))
        self.assertEqual(calls,[.2])
        self.assertIn('diagnostic_unavailable',b.held_collision_diagnostics[0])

    def test_bounded_diagnostic_log_does_not_stop_original_queries(self):
        b,calls,original=self.fixture(True)
        with patch.object(OriginalBoundary,'_collision_scope',self.base_scope):
            with b._collision_scope():
                for i in range(40):self.assertTrue(b.env.robot_model.collision_query(i*.01))
        self.assertEqual((len(calls),len(b.held_collision_diagnostics)),(40,16))


if __name__=='__main__':unittest.main()
