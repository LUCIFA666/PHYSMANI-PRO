from copy import deepcopy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState,Pose,Forecast
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.public_state import PublicStateAssembler


def sensors(t,p=(0.,0.,.8),rotation=None):
    q=np.array([0.,0.,0.,1.]) if rotation is None else Rotation.from_matrix(rotation).as_quat()
    return PublicRobotSensors(float(t),p,q,np.zeros(7),np.zeros(7),[.7,.8])


def actuator(t):
    return dict(reference_valid=True,reference_positions=(.1,)*7,reference_velocities=(.2,)*7,
                reference_accelerations=(.3,)*7,target_joints=(.4,)*7,reference_clock=float(t),
                gripper_target=0.,gripper_done=False,reference_is_measured_state=False)


def bridge_result(t,mode='FREE'):
    pose=Pose([.2,-.1,.775]);points=pose.position+np.array([[0.,0.,0.],[.01,0,0.],[0,.01,0.]])
    obj=ObjectState('target',float(t),pose,points,np.array([0.,.02,0.]),np.zeros(3),1.)
    times=t+np.arange(17)*.05
    forecast=Forecast(('target',),times,np.broadcast_to(pose.position,(17,1,3)).copy(),
        np.broadcast_to(np.eye(3),(17,1,3,3)).copy(),np.zeros((17,1,3)),np.zeros((17,1,3)))
    return dict(timestamp=float(t),mode=mode,available=mode in ('FREE','ATTACHED_PUBLIC'),
        free_forecast_available=mode=='FREE',controlled_mode_available=mode=='ATTACHED_PUBLIC',
        object_observation=obj,goal_pose=Pose([.3,.1,1.]),forecast=forecast if mode=='FREE' else None,
        rotation_observable=False,Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
        forecaster_is_injected_test_double=False,tracker_dependency_injected=False,goal_marker_dependency_injected=False,
        controlled=dict(estimated_world_pose=Pose([.1,0.,.85]),tcp_relative_pose=Pose([.1,0.,.05]),
                        evidence_timestamp=float(t),evidence_sources=('rgbd_relative_pose_tracking','public_robot_pose'),
                        orientation_observable=False,caller_asserted_public_evidence_not_Native_verified=True,free_body_GRU_used=False))


class PublicStateAssemblerTests(unittest.TestCase):
    def prepare(self,robot=None):
        assembler=PublicStateAssembler();assembler.observe_robot(sensors(0.))
        current=sensors(.05,(.01,0.,.8)) if robot is None else robot
        self.assertTrue(assembler.observe_robot(current))
        return assembler,current

    def test_FIRST_frame_has_no_invented_velocity_and_real50ms_FD_is_world_twist(self):
        assembler=PublicStateAssembler();r=sensors(0.)
        self.assertFalse(assembler.observe_robot(r));self.assertIsNone(assembler.build(bridge_result(0.),r,actuator(0.)))
        rotated=Rotation.from_euler('z',.01).as_matrix();r=sensors(.05,(.01,0.,.8),rotated)
        self.assertTrue(assembler.observe_robot(r));state=assembler.build(bridge_result(.05),r,actuator(.05))
        np.testing.assert_allclose(state['tcp_velocity'],[.2,0.,0.],atol=1e-12)
        np.testing.assert_allclose(state['tcp_angular_velocity'],[0.,0.,.2],atol=1e-12)
        self.assertFalse(state['held']);self.assertEqual(state['requested_opening'],0.)
        self.assertEqual(state['opening'],.75);np.testing.assert_array_equal(state['finger_open_amounts'],[.7,.8])
        self.assertFalse(state['Native_held_used']);self.assertFalse(state['Native_success_used'])

    def test_gap_requires_new_real_robot_pair_and_quaternion_sign_is_not_rotation(self):
        assembler=PublicStateAssembler();assembler.observe_robot(sensors(0.));r=sensors(.1)
        self.assertFalse(assembler.observe_robot(r));self.assertIsNone(assembler.build(bridge_result(.1),r,actuator(.1)))
        r=PublicRobotSensors(.15,[.01,0,.8],[0.,0.,0.,-1.],np.zeros(7),np.zeros(7),[.7,.8])
        self.assertTrue(assembler.observe_robot(r));state=assembler.build(bridge_result(.15),r,actuator(.15))
        np.testing.assert_allclose(state['tcp_angular_velocity'],np.zeros(3),atol=1e-12)
        with self.assertRaises(ValueError):assembler.observe_robot(r)

    def test_UNRESOLVED_and_STALE_never_become_policy_state(self):
        assembler,r=self.prepare()
        for mode in ('UNRESOLVED','STALE'):
            self.assertIsNone(assembler.build(bridge_result(.05,mode),r,actuator(.05)))

    def test_public_attachment_uses_controlled_offset_kinematics_not_close_command(self):
        assembler,r=self.prepare();result=bridge_result(.05,'ATTACHED_PUBLIC')
        state=assembler.build(result,r,actuator(.05))
        self.assertTrue(state['held']);self.assertIn('public_attachment',state['held_source'])
        np.testing.assert_array_equal(state['object_position'],result['controlled']['estimated_world_pose'].position)
        np.testing.assert_allclose(state['object_linear_velocity'],[.2,0.,0.],atol=1e-12)
        self.assertFalse(state['object_orientation_observable'])
        self.assertFalse(assembler.build(bridge_result(.05),r,actuator(.05))['held'])

    def test_actuator_state_is_not_actual_joint_state_and_is_copied(self):
        assembler,r=self.prepare();h=actuator(.05);state=assembler.build(bridge_result(.05),r,h)
        np.testing.assert_array_equal(state['q'],np.zeros(7));self.assertEqual(state['actuator_state']['reference_positions'],(.1,)*7)
        h['target_joints']=(.9,)*7;self.assertEqual(state['actuator_state']['target_joints'],(.4,)*7)
        bad=actuator(.05);bad['Native_success']=True
        with self.assertRaises(ValueError):assembler.build(bridge_result(.05),r,bad)

    def test_default_rejects_mock_dependencies_and_Native_or_future_flags(self):
        assembler,r=self.prepare()
        for key in ('forecaster_is_injected_test_double','tracker_dependency_injected','goal_marker_dependency_injected'):
            result=bridge_result(.05);result[key]=True
            with self.assertRaises(ValueError):assembler.build(result,r,actuator(.05))
        for key in ('Native_inputs_used','future_observations_used'):
            result=bridge_result(.05);result[key]=True
            with self.assertRaises(ValueError):assembler.build(result,r,actuator(.05))

    def test_stale_or_nonfinite_executor_clock_and_bad_reference_shapes_rejected(self):
        assembler,r=self.prepare()
        for clock in (0.,float('nan')):
            bad=actuator(clock)
            with self.assertRaises(ValueError):assembler.build(bridge_result(.05),r,bad)
        for key in ('reference_positions','reference_velocities','reference_accelerations','target_joints'):
            bad=actuator(.05);bad[key]=[0.]*6
            with self.assertRaises(ValueError):assembler.build(bridge_result(.05),r,bad)

    def test_build_cannot_swap_different_robot_values_at_already_observed_timestamp(self):
        assembler,r=self.prepare();different=sensors(.05,(.2,0.,.8))
        with self.assertRaises(ValueError):assembler.build(bridge_result(.05),different,actuator(.05))

    def test_Native_phase_memory_and_fabricated_observations_rejected(self):
        assembler,r=self.prepare()
        for memory in ({'phase':'approach','Native_held':True},{'native_success':True},{'object_position':[1.,2.,3.]}):
            with self.assertRaises(ValueError):assembler.build(bridge_result(.05),r,actuator(.05),phase_state=memory)
        result=bridge_result(.05);result['observations_fabricated']=True
        with self.assertRaises(ValueError):assembler.build(result,r,actuator(.05))
    def test_phase_memory_boolean_quaternion_and_all_times_are_causal_typed(self):
        assembler,r=self.prepare()
        for memory in ({'phase':'close','close_entered_at':.1},
                       {'phase':'hold','reopening':{'Native_success':True}},
                       {'phase':'hold','hold_tcp_quaternion':[0.,0.,0.,2.]}):
            with self.assertRaises(ValueError):assembler.build(bridge_result(.05),r,actuator(.05),phase_state=memory)
        bad=actuator(.05);bad['gripper_done']={'Native_held':True}
        with self.assertRaises(ValueError):assembler.build(bridge_result(.05),r,bad)

    def test_valid_public_phase_goal_is_tied_to_current_public_goal_and_held_is_estimated(self):
        assembler,r=self.prepare();result=bridge_result(.05)
        phase=dict(phase='hold',dwell=.01,lost_hold=0.,updated_at=.05,held_is_measured=True,
                   lift_goal_position=result['goal_pose'].position.tolist(),hold_tcp_quaternion=[0,0,0,1],reopening=False)
        state=assembler.build(result,r,actuator(.05),phase_state=phase)
        self.assertFalse(state['phase_state']['held_is_measured'])
        self.assertTrue(phase['held_is_measured'])
        phase['lift_goal_position']=[3.,3.,3.]
        with self.assertRaises(ValueError):assembler.build(result,r,actuator(.05),phase_state=phase)



if __name__=='__main__':unittest.main()
