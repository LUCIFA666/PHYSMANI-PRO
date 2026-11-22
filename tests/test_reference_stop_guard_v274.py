from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.contracts import Pose,RobotState,Reachability
from our_mani.reachability import TimedReachability
from our_mani.multitask.reference_stop_guard_v274 import NextReferenceStopGuard,OffsetModel


class Model:
    name='analytic_original_six_unit_test_model'
    lower=np.array([-2.]);upper=np.array([2.]);velocity_limits=np.array([2.]);acceleration_limits=np.array([6.])
    def __init__(self,obstacle=None):self.obstacle=obstacle;self.calls=[]
    def collision(self,q,offset):
        self.calls.append((float(q[0]),float(offset)))
        return self.obstacle is not None and q[0]>=self.obstacle


class Tests(unittest.TestCase):
    def fixture(self,obstacle=None,velocity=.4):
        model=Model(obstacle);gate=TimedReachability(model)
        robot=RobotState(.2,Pose([0.,0.,.8]),np.zeros(3),np.zeros(3),np.array([0.]),np.array([velocity]))
        controller=NS(goal=NS(phase='approach',reference_id='target',manipulated_ids=('target',)),
            phase_index=0,library=NS(phase_order=('approach',)),generator=NS(bind_candidate_scene=lambda *args:None))
        m=NS(native_gate=gate,config=NS(dt=.05),current_forecast=NS(times=np.linspace(.2,1.3,23)),
            current_state=NS(robot=robot),controller=controller,retired_brake_goal=None,plan_kind='tracking',
            pending=None,events=[],qualification_failed=False,confirm_actual_issued=lambda **kwargs:None,
            continuity_state=lambda r:(r.joints.copy(),r.joint_velocities.copy(),'actual fixture reference'),
            prototype=dict(reference_to_joints=[velocity*.05],reference_to_velocity=[velocity],proposal_id='original-one-step'))
        def original_check(r,duration):
            m.pending=deepcopy(m.prototype)
            return Reachability(True,'original_one_step_gate_passed',joints=np.array(m.prototype['reference_to_joints']),
                joint_velocities=np.array(m.prototype['reference_to_velocity']))
        m.check_stream=original_check
        guard=NextReferenceStopGuard(m);guard.install_stream();guard.install_confirmation()
        return m,guard,robot,model

    def test_next_point_safe_but_no_complete_stop_is_rejected_before_dispatch(self):
        m,g,robot,model=self.fixture(.13,1.)
        result=m.check_stream(robot,.05)
        self.assertFalse(result.feasible);self.assertIsNone(m.pending)
        self.assertEqual(g.rejections,1);self.assertIsNone(g.confirmed_stop)
        self.assertTrue(g.make_brake(robot,9.8,'reject next reference before stopping region disappears'))
        self.assertLess(m.plan.q1[0],.13)

    def test_dense_grid_finds_stop_missing_between_original_point15_and_point25(self):
        m,g,robot,model=self.fixture(.12,1.)
        original=[]
        for duration in (.05,.15,.25,.4,.6,.9,1.1):
            end=robot.joints+.5*robot.joint_velocities*duration
            original.append(m.native_gate.check_joints(robot,end,np.zeros(1),duration).feasible)
        self.assertFalse(any(original))
        self.assertTrue(g.make_brake(robot,9.8,'fresh dense duration'))
        self.assertLess(m.plan.end-m.plan.start,.25)
        self.assertEqual(m.plan_kind,'braking')

    def test_safe_next_stop_retains_original_proposal_and_both_full_proofs(self):
        m,g,robot,model=self.fixture()
        result=m.check_stream(robot,.05)
        self.assertTrue(result.feasible);self.assertEqual(m.pending['proposal_id'],'original-one-step')
        proof=m.pending['next_reference_stop_guard_v274']
        self.assertTrue(proof['complete_nominal_offset_stop'])
        self.assertTrue(proof['complete_current_actual_Hermite_to_same_stop'])
        self.assertEqual(proof['proof']['reference_collision_time_offset'],.05)
        self.assertIsNone(g.confirmed_stop)

    def test_stop_cache_requires_matching_actual_qualified_dispatch(self):
        m,g,robot,model=self.fixture();m.check_stream(robot,.05)
        command={'feedforward_execution':{'proposal_id':'wrong-unissued-proposal'}}
        execution={'actual_window_velocity_within_limits':True,'actual_window_acceleration_within_limits':True}
        m.confirm_actual_issued(time=.25,command=command,execution=execution)
        self.assertIsNone(g.confirmed_stop)
        m.check_stream(robot,.05);command['feedforward_execution']['proposal_id']='original-one-step'
        m.confirm_actual_issued(time=.25,command=command,execution=execution)
        self.assertIsNotNone(g.confirmed_stop)

    def test_actual_motion_failure_does_not_admit_cached_stop(self):
        m,g,robot,model=self.fixture();m.check_stream(robot,.05)
        m.confirm_actual_issued(time=.25,command={'feedforward_execution':{'proposal_id':'original-one-step'}},
            execution={'actual_window_velocity_within_limits':True,'actual_window_acceleration_within_limits':False})
        self.assertIsNone(g.confirmed_stop)

    def test_cached_plan_never_bypasses_fresh_obstacle_check(self):
        m,g,robot,model=self.fixture();m.check_stream(robot,.05)
        m.confirm_actual_issued(time=.25,command={'feedforward_execution':{'proposal_id':'original-one-step'}},
            execution={'actual_window_velocity_within_limits':True,'actual_window_acceleration_within_limits':True})
        now=replace(robot,timestamp=.25,joints=np.array([.02]))
        model.obstacle=.02
        self.assertFalse(g.make_brake(now,9.75,'new current collision invalidates cached certificate'))

    def test_short_original_forecast_rejects_moving_reference(self):
        m,g,robot,model=self.fixture();m.current_forecast.times=np.array([.2,.25])
        self.assertFalse(m.check_stream(robot,.05).feasible);self.assertIsNone(m.pending)

    def test_zero_nominal_derivative_needs_no_extra_physics_after_final_tick(self):
        m,g,robot,model=self.fixture(velocity=0.);m.current_forecast.times=np.array([.2,.25])
        self.assertTrue(m.check_stream(robot,.05).feasible)
        self.assertTrue(m.pending['next_reference_stop_guard_v274']['actual_stopped_not_claimed'])

    def test_collision_time_offsets_include_next_reference_tick(self):
        model=Model();shifted=OffsetModel(model,.05);shifted.collision(np.array([.1]),.17)
        self.assertAlmostEqual(model.calls[-1][1],.22)


if __name__=='__main__':unittest.main()
