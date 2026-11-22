from copy import deepcopy
import unittest
import numpy as np

from our_mani.local_mpc_20261102 import mpc as original
from our_mani.local_mpc_20261102.batch_mpc import BatchMPCPlanner as OriginalBatch,BatchBounds
from our_mani.local_mpc_20261102.terminal_mpc import (TerminalCostConfig,terminal_cost,TerminalScalarMPCPlanner,TerminalBatchMPCPlanner)


def state(**kw):
    value=dict(tcp_position=np.array([0.,0.,.97]),tcp_quaternion=original.quat_exp([0.,.21,0.]),tcp_velocity=np.zeros(3),
        object_position=np.array([0.,0.,.9]),object_quaternion=np.array([0.,0.,0.,1.]),
        object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),opening=1.,held=False,time=0.,requested_opening=1.,
        phase_state=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=-.1))
    value.update(kw);return original.validate_state(value)


def candidate(**kw):
    value=dict(id='mode',object_tcp_position=np.zeros(3),object_tcp_quaternion=np.array([0.,0.,0.,1.]),deadline=.8,
        approach_direction=np.array([0.,0.,1.]),pregrasp_offset=.07,opening=0.,required_opening=1.,allowed_relative_speed=.14)
    value.update(kw);return original.validate_candidate(value)


def endpoint(initial,**kw):
    value=deepcopy(initial);value['time']=initial['time']+.8;value.update(kw);return value


def fixed_rollout(planner,initial,mode,rot):
    controls=np.zeros((8,6));controls[:,4]=rot
    return planner._rollout(initial,mode,controls,initial['phase_state'],{'requested_opening':1.},None)


class TerminalTests(unittest.TestCase):
    def test_old_cost_has_downward_orientation_threshold_reward(self):
        mode=candidate();old=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=-.1)
        values=[]
        for angle in (.2+1e-7,.2-1e-7):
            initial=state(tcp_quaternion=original.quat_exp([0.,angle,0.]))
            phase=original.update_phase(initial,mode,old)
            nxt=endpoint(initial,time=.1)
            action=original._nominal_action(initial,mode,phase,np.zeros(6),original.MPCConfig())
            value,parts=original.stage_cost(initial,action,nxt,mode,phase)
            values.append((phase['phase'],value,parts['position']))
        self.assertEqual([x[0] for x in values],['approach','align'])
        self.assertAlmostEqual(values[1][2]-values[0][2],(.07/.03)**2)
        self.assertGreater(values[1][1],values[0][1]+5.)

    def test_same_simple_model_prefers_stalling_until_explicit_endpoint_cost(self):
        initial=state();mode=candidate()
        hover=fixed_rollout(original.MPCPlanner(),initial,mode,.21)
        progress=fixed_rollout(original.MPCPlanner(),initial,mode,0.)
        self.assertLess(hover['cost'],progress['cost'])
        self.assertLess(original._geometry(progress['states'][-1],mode)['distance'],1e-5)
        new_hover=fixed_rollout(TerminalScalarMPCPlanner(),initial,mode,.21)
        new_progress=fixed_rollout(TerminalScalarMPCPlanner(),initial,mode,0.)
        self.assertLess(new_progress['cost'],new_hover['cost'])
        self.assertFalse(any(x['held'] for x in new_progress['states']))
        self.assertEqual(hover['sequence'][0]['opening'],1.)

    def test_full_fixed_budget_cem_selects_stall_when_it_is_a_budgeted_warm_sequence(self):
        initial=state();modes=[candidate(id=f'm{i}') for i in range(8)]
        hover=fixed_rollout(original.MPCPlanner(),initial,modes[0],.21)
        warm=dict(time=0.,selected_candidate_id='m0',deadline=.8,phase_state=initial['phase_state'],
                  sequence=hover['sequence'],candidate_ids=[c['id'] for c in modes])
        old=OriginalBatch().plan(initial,modes,warm_start=warm)
        new=TerminalBatchMPCPlanner().plan(initial,modes,warm_start=warm)
        self.assertAlmostEqual(old['cost'],hover['cost'],places=12)
        self.assertAlmostEqual(original._geometry(old['predicted_states'][-1],modes[0])['distance'],.07)
        mode=next(c for c in modes if c['id']==new['selected_candidate_id'])
        self.assertLess(original._geometry(new['predicted_states'][-1],mode)['distance'],.02)
        self.assertEqual(old['budget'],new['budget'])
        self.assertEqual(new['budget']['sequences'],256)
        self.assertEqual(new['diagnostics']['warm_start_evaluations_counted'],3)

    def test_phi_is_phase_independent_and_grades_partial_progress(self):
        initial=state();mode=candidate();values=[]
        for distance in (.07,.04,.025):
            nxt=endpoint(initial,tcp_position=np.array([0.,0.,.9+distance]),tcp_quaternion=np.array([0.,0.,0.,1.]))
            a,_=terminal_cost(initial,nxt,mode,dict(phase='approach'))
            b,_=terminal_cost(initial,nxt,mode,dict(phase='align'))
            c,_=terminal_cost(initial,nxt,mode,dict(phase='hold'))
            self.assertEqual(a,b);self.assertEqual(a,c);values.append(a)
        self.assertGreater(values[0],values[1]);self.assertGreater(values[1],values[2]);self.assertGreater(values[2],0.)

    def test_terminal_uses_same_future_pose_and_rotation_point_speed(self):
        initial=state(object_linear_velocity=np.array([.1,0.,0.]),object_angular_velocity=np.array([0.,0.,1.]))
        mode=candidate(object_tcp_position=np.array([.03,0.,0.]))
        object_p,object_q=original.predict_object(initial,.8)
        target=original.contact_target(initial,mode,.8)
        nxt=endpoint(initial,object_position=object_p,object_quaternion=object_q,
            tcp_position=target['position'],tcp_quaternion=target['quaternion'],tcp_velocity=target['linear_velocity'])
        value,_=terminal_cost(initial,nxt,mode,dict(phase='approach'))
        self.assertLess(value,1e-20)
        nxt['tcp_velocity']=initial['object_linear_velocity']
        value,parts=terminal_cost(initial,nxt,mode,dict(phase='approach'))
        self.assertGreater(parts['grasp_relative_speed'],0.)
        self.assertGreater(value,0.)
        nxt['time']=.7
        with self.assertRaises(ValueError):terminal_cost(initial,nxt,mode,{})

    def test_held_lift_objective_uses_anchor_goal_and_release_has_cost(self):
        initial=state(held=True,opening=.1,requested_opening=0.,tcp_quaternion=np.array([0.,0.,0.,1.]),
                      lift_goal_position=np.array([0.,0.,1.02]))
        nxt=endpoint(initial,object_position=np.array([0.,0.,1.02]),opening=0.)
        value,_=terminal_cost(initial,nxt,candidate(),dict(phase='approach'))
        self.assertEqual(value,0.)
        nxt['held']=False
        value,parts=terminal_cost(initial,nxt,candidate(),dict(phase='hold'))
        self.assertEqual(parts['lost_hold'],25.)
        self.assertGreater(value,0.)

    def test_disabled_exact_legacy_and_enabled_scalar_batch_fixed_budget(self):
        initial=state();modes=[candidate(id=f'm{i}') for i in range(8)]
        old=OriginalBatch().plan(initial,modes)
        disabled=TerminalBatchMPCPlanner(terminal_config=TerminalCostConfig(enabled=False)).plan(initial,modes)
        self.assertEqual(old['cost'],disabled['cost'])
        np.testing.assert_array_equal(old['action']['tcp_position'],disabled['action']['tcp_position'])
        a=TerminalScalarMPCPlanner().plan(initial,modes)
        b=TerminalBatchMPCPlanner().plan(initial,modes)
        self.assertEqual(a['budget'],b['budget']);self.assertEqual(a['budget']['sequences'],256)
        self.assertEqual(a['selected_candidate_id'],b['selected_candidate_id'])
        self.assertAlmostEqual(a['cost'],b['cost'],places=10)
        self.assertAlmostEqual(a['cost_terms']['terminal_phi'],b['cost_terms']['terminal_phi'],places=10)
        np.testing.assert_allclose(a['action']['tcp_position'],b['action']['tcp_position'],atol=1e-12)
        invalid=[dict(x,native_pose_qualified=False) for x in modes]
        result=TerminalBatchMPCPlanner(constraint_cost=BatchBounds()).plan(initial,invalid)
        self.assertIsNone(result['action']);self.assertEqual(result['budget']['sequences'],256)

    def test_nonfinite_and_fabricated_contact_fail_closed(self):
        initial=state();mode=candidate();nxt=endpoint(initial,held=True)
        with self.assertRaisesRegex(ValueError,'invent'):terminal_cost(initial,nxt,mode,{})
        nxt=endpoint(initial,tcp_position=np.array([np.nan,0.,0.]))
        with self.assertRaises(ValueError):terminal_cost(initial,nxt,mode,{})


if __name__=='__main__':unittest.main()
