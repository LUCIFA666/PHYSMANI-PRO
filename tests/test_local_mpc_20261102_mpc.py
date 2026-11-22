"""CPU geometric/clock/budget contracts; no Native, data fitting or TEST files."""
from copy import deepcopy
import unittest

import numpy as np

from our_mani.local_mpc_20261102.mpc import (
    MPCConfig,MPCPlanner,SimpleExecutionModel,contact_target,predict_object,
    quaternion,quat_exp,quat_matrix,stage_cost,update_phase,validate_candidate,validate_state,
    _opening_command,
)


def state(**values):
    data=dict(tcp_position=np.array([0.,0.,.15]),tcp_quaternion=np.array([0.,0.,0.,1.]),
        tcp_velocity=np.zeros(3),object_position=np.zeros(3),object_quaternion=np.array([0.,0.,0.,1.]),
        object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),opening=1.,held=False,time=0.)
    data.update(values)
    return data


def candidates():
    return [dict(id=f"yaw{i//2}-time{i%2}",object_tcp_position=np.zeros(3),
        object_tcp_quaternion=quat_exp(np.array([0.,0.,(i//2)*.15])),deadline=.5+.3*(i%2),
        approach_direction=np.array([0.,0.,1.]),pregrasp_offset=.06,opening=0.,required_opening=1.,
        allowed_relative_speed=.1) for i in range(8)]


class RecordingModel:
    def __init__(self):
        self.model=SimpleExecutionModel();self.calls=0
    def step(self,*args):
        self.calls+=1
        return self.model.step(*args)


class MPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=RecordingModel()
        cls.planner=MPCPlanner(execution_model=cls.model)
        cls.source=state();cls.options=candidates()
        cls.result=cls.planner.plan(cls.source,cls.options)

    def test_exact_total_budget_complete_horizon_and_endpoint_clocks(self):
        result=self.result
        self.assertEqual(result["budget"]["sequences"],256)
        self.assertEqual(result["budget"]["iteration_sequences"],[88,88,80])
        self.assertEqual(self.model.calls,256*8)
        self.assertEqual(result["budget"]["virtual_50ms_substeps"],4096)
        self.assertEqual(result["budget"]["Native_steps"],0)
        self.assertEqual(len(result["sequence"]),8)
        np.testing.assert_allclose([s["time"] for s in result["predicted_states"]],np.arange(9)*.1,atol=1e-12)
        self.assertFalse(result["diagnostics"]["value_network_used"])
        self.assertFalse(result["diagnostics"]["real_time_guaranteed"])
        self.assertGreater(result["diagnostics"]["wall_seconds"],0.)

    def test_rotating_point_velocity_includes_world_lever_arm(self):
        s=validate_state(state(object_linear_velocity=np.array([.2,0.,0.]),object_angular_velocity=np.array([0.,0.,2.])))
        candidate=validate_candidate(dict(candidates()[0],object_tcp_position=np.array([.1,0.,0.])))
        target=contact_target(s,candidate,.5)
        lever=np.array([.1*np.cos(1.),.1*np.sin(1.),0.])
        np.testing.assert_allclose(target["position"],np.array([.1,0.,0.])+lever,atol=1e-12)
        np.testing.assert_allclose(target["linear_velocity"],np.array([.2,0.,0.])+np.cross([0.,0.,2.],lever),atol=1e-12)

    def test_same_time_cost_has_zero_geometry_error(self):
        s=validate_state(state(object_linear_velocity=np.array([.3,0.,0.])))
        candidate=validate_candidate(candidates()[0]);p,q=predict_object(s,.1)
        end=validate_state(dict(s,tcp_position=p,tcp_quaternion=q,tcp_velocity=np.array([.3,0.,0.]),
                                object_position=p,object_quaternion=q,time=.1))
        action=dict(tcp_position=p,tcp_quaternion=q,opening=1.)
        _,parts=stage_cost(s,action,end,candidate,{"phase":"align"})
        self.assertAlmostEqual(parts["position"],0.)
        self.assertAlmostEqual(parts["relative_speed"],0.)

    def test_closed_command_never_invents_held_and_open_releases_immediately(self):
        model=SimpleExecutionModel()
        s=validate_state(state(tcp_position=np.zeros(3)))
        close=dict(tcp_position=np.zeros(3),tcp_quaternion=np.array([0.,0.,0.,1.]),opening=0.)
        end,h,_=model.step(s,close,{},.1)
        self.assertFalse(end["held"])
        self.assertLess(end["opening"],1.)
        self.assertAlmostEqual(end["opening"],.6)
        attached=validate_state(state(tcp_position=np.zeros(3),opening=.8,held=True))
        end,h,_=model.step(attached,close,{},.1)
        self.assertTrue(end["held"])
        opened,_,_=model.step(end,dict(close,opening=1.),h,.1)
        self.assertFalse(opened["held"])

    def test_velocity_reference_is_not_fictional_actuator_feedforward(self):
        model=SimpleExecutionModel();s=validate_state(state())
        action=dict(tcp_position=s["tcp_position"].copy(),tcp_quaternion=s["tcp_quaternion"].copy(),opening=1.)
        plain,_,_=model.step(s,action,{},.1)
        diagnostic_only,_,_=model.step(s,dict(action,tcp_velocity_reference=np.ones(3)*100,
                                            angular_velocity_reference=np.ones(3)*100),{},.1)
        np.testing.assert_array_equal(plain["tcp_position"],diagnostic_only["tcp_position"])
        np.testing.assert_array_equal(plain["tcp_quaternion"],diagnostic_only["tcp_quaternion"])

    def test_geometric_hysteresis_measured_phase_once_and_missed_grasp_reopens(self):
        c=MPCConfig();candidate=validate_candidate(candidates()[0])
        s=validate_state(state(tcp_position=np.zeros(3),time=.3))
        prior=dict(phase="align",dwell=.1,lost_hold=0.,updated_at=.2)
        close=update_phase(s,candidate,prior,c)
        self.assertEqual(close["phase"],"close")
        self.assertEqual(update_phase(s,candidate,close,c)["dwell"],close["dwell"])
        failed=validate_state(dict(s,time=.4,opening=.8,requested_opening=0.))
        recover=update_phase(failed,candidate,close,c)
        self.assertEqual(recover["phase"],"align")
        self.assertTrue(recover["reopening"])
        self.assertFalse(failed["held"])
        self.assertEqual(prior["updated_at"],.2)

    def test_hold_tracks_lift_goal_and_preserves_actual_attachment(self):
        s=state(tcp_position=np.zeros(3),opening=.2,held=True,lift_goal_position=np.array([0.,0.,.12]))
        result=MPCPlanner().plan(s,candidates())
        self.assertEqual(result["phase_state"]["phase"],"hold")
        self.assertGreater(result["predicted_states"][-1]["object_position"][2],0.)
        self.assertEqual(result["action"]["opening"],0.)
        self.assertTrue(result["predicted_states"][-1]["held"])

    def test_warm_shift_counts_every_evaluation_and_deadline_cannot_move(self):
        first=self.result
        feedback=deepcopy(first["predicted_states"][1])
        model=RecordingModel();second=MPCPlanner(execution_model=model).plan(feedback,candidates(),warm_start=first["warm_start"])
        self.assertEqual(model.calls,2048)
        self.assertEqual(second["diagnostics"]["warm_start_evaluations_counted"],3)
        self.assertEqual(second["budget"]["sequences"],256)
        self.assertEqual(second["diagnostics"]["phase_real_updates"],1)
        shifted=candidates()
        next(row for row in shifted if row["id"]==first["selected_candidate_id"])["deadline"]+=.1
        with self.assertRaisesRegex(ValueError,"pushed forward"):
            MPCPlanner().plan(feedback,shifted,warm_start=first["warm_start"])
        future=deepcopy(first["warm_start"]);future["time"]=1.
        with self.assertRaisesRegex(ValueError,"future feedback"):
            MPCPlanner().plan(feedback,candidates(),warm_start=future)

    def test_causality_and_input_immutability(self):
        original=deepcopy(self.source)
        augmented=deepcopy(original);augmented["future_expert_actions"]=np.ones((99,8))*999
        augmented["task_success"]=True
        replay=MPCPlanner().plan(augmented,deepcopy(self.options))
        np.testing.assert_array_equal(replay["action"]["tcp_position"],self.result["action"]["tcp_position"])
        self.assertFalse(self.source["held"])
        np.testing.assert_array_equal(self.source["tcp_position"],original["tcp_position"])
        self.assertEqual(self.result["phase_state"]["updated_at"],0.)

    def test_constraints_all_infeasible_still_consume_the_declared_budget(self):
        result=MPCPlanner(constraint_cost=lambda *_:np.inf).plan(state(),candidates())
        self.assertIsNone(result["action"])
        self.assertEqual(result["budget"]["sequences"],256)
        self.assertEqual(result["diagnostics"]["reason"],"all_sequences_infeasible")

    def test_candidate_opening_clearance_and_closing_are_unambiguous(self):
        with self.assertRaisesRegex(ValueError,"required_opening"):
            validate_candidate(dict(candidates()[0],opening=1.))
        with self.assertRaises(ValueError):MPCConfig(iteration_samples=(256,256,256))
        with self.assertRaises(ValueError):MPCConfig(dt=.5)
        candidate=validate_candidate(candidates()[0])
        far=validate_state(state(tcp_position=np.array([.04,0.,0.]),requested_opening=1.))
        near=validate_state(state(tcp_position=np.zeros(3),requested_opening=1.))
        self.assertEqual(_opening_command(far,candidate,{"phase":"close"},MPCConfig()),1.)
        self.assertEqual(_opening_command(near,candidate,{"phase":"close"},MPCConfig()),0.)

    def test_warm_predicted_hidden_is_never_promoted_to_actual_feedback(self):
        warm=deepcopy(self.result["warm_start"])
        warm["execution_h"]={"requested_opening":1.,"object_in_tcp":np.array([999.,999.,999.])}
        current=state(time=.1,tcp_position=np.array([.02,0.,.1]),object_position=np.array([.02,0.,.08]),
                      held=True,opening=.8,requested_opening=0.,actuator_state={"requested_opening":0.,"actual_joint_reference":np.arange(7.)})
        seen=[]
        class Recorder(SimpleExecutionModel):
            def step(self,s,action,h=None,dt=.1):
                if not seen:seen.append(deepcopy(h))
                return super().step(s,action,h,dt)
        result=MPCPlanner(execution_model=Recorder()).plan(current,candidates(),warm_start=warm)
        self.assertEqual(seen[0]["requested_opening"],0.)
        np.testing.assert_allclose(seen[0]["object_in_tcp"],[0.,0.,-.02],atol=1e-12)
        np.testing.assert_array_equal(seen[0]["actual_joint_reference"],np.arange(7.))
        self.assertNotIn("execution_h",result["warm_start"])
        self.assertFalse(result["diagnostics"]["warm_prediction_used_as_feedback"])
        self.assertEqual(current["actuator_state"]["requested_opening"],0.)

    def test_explicit_real_phase_survives_rejected_action_or_cleared_warm_state(self):
        current=state(time=.2,tcp_position=np.zeros(3),requested_opening=1.,
                      phase_state={"phase":"align","dwell":0.,"lost_hold":0.,"updated_at":.1})
        self.assertEqual(validate_state(current)["phase_state"]["phase"],"align")
        for warm in (None,deepcopy(self.result["warm_start"])):
            result=MPCPlanner().plan(current,candidates(),warm_start=warm)
            self.assertEqual(result["phase_state"]["phase"],"align")
            self.assertAlmostEqual(result["phase_state"]["updated_at"],.2)
        self.assertEqual(current["phase_state"]["updated_at"],.1)


if __name__=="__main__":unittest.main()
