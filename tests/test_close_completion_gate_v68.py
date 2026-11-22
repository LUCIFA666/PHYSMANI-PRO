"""Check the actual close function and stopped-capture decision agree."""
from pathlib import Path
from types import SimpleNamespace
import json,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.reachability import TimedReachability
from our_mani.multitask.interaction import (InteractionLibrary,CandidateGenerator,CandidateConfig,InteractionController,
    SE3InteractionMPC,MPCConfig,GripperTimingConfig,PolicyState,_world_target)
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.task_reference_v37 import ContinuousReferenceManager
from our_mani.multitask.close_completion_gate_v68 import binding,install

ROOT=Path(__file__).resolve().parents[1]


class Scene:
    def bind_candidate(self,**kwargs):pass
    def __call__(self,**kwargs):return dict(feasible=True)


class Tests(unittest.TestCase):
    def fixture(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/task_executor_smoke_v65_setup/configuration.json').read_text())
        task='put_rubbish_in_moving_bin';library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        # No trajectory is requested in this test. The actual controller,
        # generator's observed-reference verifier and capture method are used.
        generator=CandidateGenerator(library,TimedReachability(SimpleNamespace()),Scene(),CandidateConfig(**c['candidate_config']))
        base=InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**c['mpc_config'])),predictor='cv',ranking='geometry',
            continuation_sha256='CPU_close_gate_fixture',gripper_timing_config=GripperTimingConfig(**c['gripper_timing_config']))
        controller=ObservedFeedbackController.from_controller(base,{})
        manager=ContinuousReferenceManager(controller)
        actor=SimpleNamespace(controller=controller,continuous_reference_manager=manager,metrics_report=lambda:{})
        c['close_completion_gate_v68']={task:binding(ROOT,c,task)};install(actor,c,ROOT,task)
        template=library.for_phase('acquire')[0]
        controller.goal=SimpleNamespace(phase='acquire',template=template,candidate_id='CPU_fixture_acquire',selected_at=0.,reference_id='rubbish',manipulated_ids=('rubbish',))
        body=ObjectState('rubbish',2.,Pose([.2,.2,.8]),np.tile([.2,.2,.8],(3,1)),np.zeros(3),np.zeros(3))
        target=_world_target(body.pose,np.zeros(3),np.zeros(3),template,1.)[0]
        return controller,manager,body,target

    def state(self,body,target,position_error,rotation_error):
        pose=Pose(target.position+np.array([position_error,0.,0.]),Rotation.from_rotvec([rotation_error,0.,0.]).as_matrix()@target.rotation)
        robot=RobotState(2.,pose,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),gripper_open=True,contact=False)
        return PolicyState(robot,{'rubbish':body},{'bodies':{'rubbish':dict(directly_visible=True,registration_valid=True,orientation_observable=False,motion_mode='external')}},{'rubbish':('rubbish',)})

    def test_eight_mm_cannot_freeze_an_uncompletable_grasp_but_four_mm_can_close(self):
        controller,manager,body,target=self.fixture()
        for distance,expected in ((.008,1.),(.004,0.)):
            state=self.state(body,target,distance,0.);manager.current_state=state
            self.assertEqual(manager.capture_at_current_grasp(state.robot),expected==0.)
            self.assertEqual(controller._gripper_opening(state,0.,final_target=target),expected)
        self.assertEqual(controller._close_edge_issued_at,2.)
        self.assertLessEqual(controller.gripper_timing_config.close_position_tolerance_m,.02)

    def test_angular_completion_tolerance_also_applies_to_both_capture_and_close(self):
        controller,manager,body,target=self.fixture()
        state=self.state(body,target,0.,.04);manager.current_state=state
        self.assertFalse(manager.capture_at_current_grasp(state.robot))
        self.assertEqual(controller._gripper_opening(state,0.,final_target=target),1.)
        self.assertIsNone(controller._close_edge_issued_at)


if __name__=='__main__':unittest.main()
