import unittest
import sys
from types import ModuleType,SimpleNamespace
from our_mani.executor_grasp_frame_20261113_v1.oracle import instrument

class ForecastTraceTests(unittest.TestCase):
    def test_function_local_import_does_not_hide_forecast_time(self):
        module=ModuleType('executor_forecast_test_helper')
        exec(compile('def compute_target_pose(frame_base,waypoint,t_delay,yaw_speed):\n return [0]*7\n','/tmp/oracle_tasks/helper.py','exec'),module.__dict__)
        sys.modules[module.__name__]=module
        scope={}
        exec(compile('def expert(task,bool_return_path=False):\n from executor_forecast_test_helper import compute_target_pose\n compute_target_pose(None,None,.5,1.)\n return {"stage":"wp1"}\n','/tmp/oracle_tasks/task.py','exec'),scope)
        task=SimpleNamespace(begin=lambda:None,scene=SimpleNamespace(verify=lambda:dict(Native_setter_calls=0)))
        before=sys.getprofile()
        try:
            observed=instrument(scope['expert']);observed(task)
            self.assertEqual(observed.last_timing['lookahead_seconds'],.5)
            self.assertTrue(observed.last_timing['observes_local_import_helpers'])
            self.assertIs(sys.getprofile(),before)
        finally:sys.modules.pop(module.__name__,None)
