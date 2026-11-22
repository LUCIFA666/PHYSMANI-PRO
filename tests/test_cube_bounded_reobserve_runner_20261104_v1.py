"""Small CPU boundary checks: actual new sample wrapper and worker admission.

No model, Native object, real data, or privileged measurement is constructed.
"""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment
from our_mani.cube_corrected_live_20261104_v1.matched_producer import ObservedMatchedCorrectedCubeV72Backend
from our_mani.cube_bounded_reobserve_live_20261104_v1.producer import ObservedBoundedReobserveCubeV72Backend
from our_mani.cube_bounded_reobserve_live_20261104_v1.supervision import ReobservePublicLifecycle
from our_mani.cube_bounded_reobserve_live_20261104_v1 import contracts,worker
ROOT=contracts.ROOT
PLAN=ROOT/'config/framework/cube_bounded_reobserve_live_20261104_v1/plan.json'

class RunnerBoundaries(unittest.TestCase):
    def test_real_new_sample_wraps_original_measurement_once_before_supervision(self):
        order=[]
        with tempfile.TemporaryDirectory() as directory:
            observer=ObservedBoundedReobserveCubeV72Backend.__new__(ObservedBoundedReobserveCubeV72Backend)
            observer.backend=SimpleNamespace(requested_opening=1.)
            observer.output=Path(directory)
            observer.bridge=SimpleNamespace(active=ActiveVisualAttachment(),latest_attachment=None)
            observer.reobserve_lifecycle=ReobservePublicLifecycle()
            robot=PublicRobotSensors(0.,[0,0,1],[0,0,0,1],np.zeros(7),np.zeros(7),[1,1])
            obj=ObjectState('target',0.,Pose([0,0,.8]),np.zeros((3,3)),np.zeros(3),np.zeros(3))
            result=dict(timestamp=0.,observed_target_available=True,object_observation=obj,
                free_forecast_available=False,Native_inputs_used=False,future_observations_used=False,
                observations_fabricated=False,goal_pose=Pose([.1,0,.8]))
            original_goal=result['goal_pose']
            original_observe=ReobservePublicLifecycle.observe
            def measurement(instance,state=None):
                order.append('original_measurement');instance.latest_robot=robot;return result
            def supervision(instance,*args,**kwargs):
                order.append('supervisor');return original_observe(instance,*args,**kwargs)
            with patch.object(ObservedMatchedCorrectedCubeV72Backend,'sample',measurement),patch.object(ReobservePublicLifecycle,'observe',supervision):
                returned=observer.sample()
            self.assertIs(returned,result);self.assertIs(returned['goal_pose'],original_goal)
            self.assertEqual(order,['original_measurement','supervisor'])
            self.assertEqual(observer.reobserve_lifecycle.latest['fresh_real50ms_frames'],1)
            self.assertIsNone(observer.bridge.active.attempt_started)
            self.assertEqual(len((Path(directory)/'bounded_reobserve_public_trace.jsonl').read_text().splitlines()),1)

    def test_pending_checkpoint_default_checks_and_execute_reject_before_mkdir(self):
        plan=contracts.validate_plan(PLAN)
        self.assertEqual(len(plan['cases']),4)
        for case in plan['cases']:
            report=worker.check(PLAN,case['id'],'corrected_matched_reobserve')
            self.assertFalse(report['ready']);self.assertEqual(report['NN_forward_calls'],0)
            self.assertFalse(report['Native_started'])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'never_created'
            with self.assertRaisesRegex(ValueError,'Execution prerequisites missing'):
                worker.execute(PLAN,'ordinary_00','corrected_matched_reobserve',out,
                    source_freeze=None,source_freeze_sha256=None)
            self.assertFalse(out.exists())

    def test_relative_code_archive_separate_native7_and_exact_source_bytes(self):
        mapping=contracts.code_sources_for_plan(PLAN)
        self.assertTrue(mapping)
        for name,digest in mapping.items():
            self.assertFalse(Path(name).is_absolute());self.assertNotIn('..',Path(name).parts)
            self.assertEqual(contracts.sha(ROOT/name),digest)
        self.assertEqual(len(contracts.native_sources_for_plan(PLAN)),7)

    def test_runner_has_one_single_dispatch_hook_original_selected_check_and_coordinator(self):
        path=ROOT/'our_mani/cube_bounded_reobserve_live_20261104_v1/producer.py'
        tree=ast.parse(path.read_text())
        runner=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_bounded_reobserve_cube_v72')
        calls=[n for n in ast.walk(runner)if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Name)and n.func.id=='execute_selected_or_reobserve'for n in calls),1)
        self.assertEqual(sum(isinstance(n.func,ast.Attribute)and n.func.attr=='execute_joint_target'for n in calls),0)
        qualify=[n for n in calls if isinstance(n.func,ast.Attribute)and n.func.attr=='qualify_action']
        self.assertEqual(len(qualify),1)
        self.assertEqual([k.value.value for k in qualify[0].keywords if k.arg=='require_predicted_joint_goal'],[True])
        coordinator=[n for n in calls if isinstance(n.func,ast.Name)and n.func.id=='MatchedCorrectedCubeV72Coordinator']
        self.assertEqual(len(coordinator),1)

if __name__=='__main__':unittest.main()
