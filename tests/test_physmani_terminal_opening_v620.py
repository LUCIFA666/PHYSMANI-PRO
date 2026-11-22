from dataclasses import replace
import unittest
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_terminal_opening_v620 import OriginalTerminalOpening
import test_physmani_ring_release_v590 as ring_fixture
import test_physmani_buzz_curve_v602 as buzz_fixture


class Tests(unittest.TestCase):
    def policies(self):
        p,f=ring_fixture.Tests().fixture()
        ring_fixture.Tests().move(f,p._completion_target(p.selected));p.select()
        for p,f in [(p,f),buzz_fixture.Tests().fixture('release')]:
            p.__class__=type('QualifiedTerminal',(OriginalTerminalOpening,p.__class__),{})
            target=p._completion_target(p.selected)
            f.state=PolicyState(replace(f.state.robot,pose=target),f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
            yield p,f,target

    def test_geometry_alone_cannot_open_earlier_than_original_final_RML_gripper_tick(self):
        for p,f,target in self.policies():
            self.assertEqual(p.qualify_opening(0.,0.)[0],0.)
            self.assertEqual(p.qualify_opening(1.,0.)[0],1.)

    def test_current_pose_drift_cannot_reclose_after_a_dispatched_terminal_open_request(self):
        for p,f,target in self.policies():
            f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[.1,0.,0.],target.rotation)),
                f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
            self.assertEqual(p.qualify_opening(1.,0.)[0],0.)
            self.assertEqual(p.qualify_opening(1.,1.)[0],1.)
            self.assertEqual(p.qualify_opening(0.,1.)[0],1.)


if __name__=='__main__':unittest.main()
