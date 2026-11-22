from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose,RobotState
from our_mani.multitask.interaction import InteractionController,InteractionLibrary,GripperTimingConfig
from our_mani.multitask.physical_command_capture import install,binding
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self):
        task='beat_the_rotating_buzz';config=json.loads((ROOT/'artifacts/framework/multitask_r10/buzz_command_v412_setup'/
            (task+'_configuration.json')).read_text())
        library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        t=library.for_phase('acquire')[0];ref=Pose([.1,.2,.9]);local=Pose.from_xyzw(t.completion_geometry_xyzw)
        target=Pose(ref.position+ref.rotation@local.position,ref.rotation@local.rotation)
        actual=Pose(target.position+[.009,0.,0.],Rotation.from_rotvec([0.,0.,.06]).as_matrix()@target.rotation)
        robot=RobotState(2.,actual,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),True,False)
        state=NS(robot=robot,bodies={},role_bindings={'wand':('wand',)},perception_metadata={'bodies':{'wand':dict(
            motion_mode='attached',mode_provenance='caller_controller_touch_action_proprioception')}})
        goal=NS(phase='acquire',candidate_id='actual-buzz-close',reference_id='wand',manipulated_ids=('wand',),selected_at=1.,template=t)
        c=NS(goal=None,feedback_step_calls=0,phase_index=1,origin=ref,gripper_timing_config=GripperTimingConfig(.02,.1),
            _close_edge_issued_at=2.,generator=NS(observed_reference=lambda *a:NS(pose=ref)))
        c._public_acquire_contact=MethodType(InteractionController._public_acquire_contact,c)
        c._phase_done=MethodType(InteractionController._phase_done,c)
        m=NS(controller=c,confirmations=[],current_state=state,last_confirmed={'opening':1.},capture_stopped=lambda r:True,
            qualification_failed=False,confirm_actual_issued=lambda **kw:None)
        actor=NS(controller=c,continuous_reference_manager=m,metrics_report=lambda:{})
        config['physical_command_capture']={task:binding(ROOT,config,task)}
        witnesses,events=install(actor,config,ROOT,task);c.goal=goal
        return actor,state,target,witnesses

    def dispatch(self,actor,state,passed=True):
        now=state.robot.timestamp
        actor.continuous_reference_manager.confirm_actual_issued(time=now+.05,command={'opening':0.},execution=dict(
            actual_issued_at=now,actual_window_velocity_within_limits=passed,actual_window_acceleration_within_limits=passed))
        state.robot=replace(state.robot,timestamp=now+.05,contact=True)

    def test_actual_dispatch_canonical_geometry_and_unique_contact_required(self):
        actor,state,target,witnesses=self.fixture();c=actor.controller
        self.assertFalse(c._phase_done(state,target,public_geometry=True))
        self.dispatch(actor,state)
        self.assertTrue(witnesses)
        self.assertTrue(c._phase_done(state,target,public_geometry=True))
        self.assertFalse(c._phase_done(state,target,public_geometry=False))
        self.assertFalse(InteractionController._phase_done(c,state,target,public_geometry=True))
        state.perception_metadata['bodies']['frame']=dict(motion_mode='attached')
        self.assertFalse(c._phase_done(state,target,public_geometry=True))

    def test_failed_motor_receipt_or_unstopped_close_cannot_complete(self):
        for stopped,qualified in ((True,False),(False,True)):
            actor,state,target,witnesses=self.fixture()
            actor.continuous_reference_manager.capture_stopped=lambda r:stopped
            self.dispatch(actor,state,qualified)
            self.assertFalse(witnesses)
            self.assertFalse(actor.controller._phase_done(state,target,public_geometry=True))

    def test_original_physical_geometry_not_relaxed(self):
        actor,state,target,witnesses=self.fixture()
        state.robot=replace(state.robot,pose=Pose(target.position+[.021,0.,0.],target.rotation))
        self.dispatch(actor,state)
        self.assertFalse(witnesses)


if __name__=='__main__':unittest.main()
