"""CPU-only scalar/batch equivalence, budget and contact-state contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np

from our_mani.local_mpc_20261102.batch_mpc import BatchMPCPlanner,BatchBounds,qmul,qexp,qlog,rotate
from our_mani.local_mpc_20261102 import mpc as scalar


def state(**kw):
    value=dict(tcp_position=np.array([0.,0.,1.05]),tcp_quaternion=np.array([0.,0.,0.,1.]),tcp_velocity=np.zeros(3),
        object_position=np.array([0.,0.,.9]),object_quaternion=np.array([0.,0.,0.,1.]),
        object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),opening=1.,held=False,time=0.)
    value.update(kw);return value
def candidates():
    return [dict(id=f'm{i}',object_tcp_position=np.zeros(3),object_tcp_quaternion=scalar.quat_exp(np.array([0.,0.,i//2*.15])),
        deadline=.5+.3*(i%2),approach_direction=np.array([0.,0.,1.]),pregrasp_offset=.06,opening=0.,required_opening=1.,allowed_relative_speed=.1) for i in range(8)]


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=state();cls.c=candidates()
        cls.a=scalar.MPCPlanner().plan(cls.s,cls.c)
        cls.b=BatchMPCPlanner().plan(cls.s,cls.c)

    def equal_plan(self,a,b):
        self.assertEqual(a['budget'],b['budget'])
        self.assertEqual(a['selected_candidate_id'],b['selected_candidate_id'])
        if a['action'] is None:
            self.assertIsNone(b['action']);return
        self.assertEqual(a['phase_state']['phase'],b['phase_state']['phase'])
        self.assertAlmostEqual(a['cost'],b['cost'],places=9)
        for x,y in zip(a['sequence'],b['sequence']):
            self.assertEqual(x['opening'],y['opening'])
            for key in ('tcp_position','tcp_quaternion','tcp_velocity_reference','angular_velocity_reference'):
                np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-10)
        for x,y in zip(a['predicted_states'],b['predicted_states']):
            self.assertEqual(x['held'],y['held']);self.assertEqual(x['time'],y['time'])
            for key in ('tcp_position','tcp_quaternion','tcp_velocity','tcp_angular_velocity','object_position','object_quaternion','object_linear_velocity','object_angular_velocity','opening'):
                np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-10)
        for key in ('knot_means','knot_stds'):
            np.testing.assert_allclose(a['warm_start'][key],b['warm_start'][key],rtol=0,atol=1e-10)

    def test_full_budget_and_static_scalar_equivalence(self):
        self.equal_plan(self.a,self.b)
        self.assertEqual(self.b['budget']['iteration_sequences'],[88,88,80])
        self.assertEqual(self.b['budget']['virtual_macrosteps'],2048)
        self.assertEqual(self.b['budget']['virtual_50ms_substeps'],4096)
        self.assertEqual(self.b['budget']['Native_steps'],0)
        self.assertEqual(self.b['diagnostics']['phase_real_updates'],1)

    def test_batch_quaternion_math_matches_scalar_world_frame(self):
        rng=np.random.default_rng(17);v=rng.normal(size=(100,3));p=rng.normal(size=(100,3))
        q=qexp(v);right=qexp(v*.3)
        np.testing.assert_allclose(q,np.stack([scalar.quat_exp(row) for row in v]),rtol=0,atol=1e-14)
        np.testing.assert_allclose(qmul(q,right),np.stack([scalar.quat_multiply(a,b) for a,b in zip(q,right)]),rtol=0,atol=1e-14)
        np.testing.assert_allclose(rotate(q,p),np.stack([scalar.quat_matrix(a)@b for a,b in zip(q,p)]),rtol=0,atol=1e-14)
        np.testing.assert_allclose(qlog(q),np.stack([scalar.quat_log(a) for a in q]),rtol=0,atol=1e-14)

    def test_world_rotation_point_speed_and_constraints(self):
        s=state(object_linear_velocity=np.array([.13,0.,0.]),object_angular_velocity=np.array([.2,.1,1.3]))
        cs=[dict(row,object_tcp_position=np.array([.03,0.,0.]),native_pose_qualified=i%2==0) for i,row in enumerate(candidates())]
        bounds=BatchBounds()
        self.equal_plan(scalar.MPCPlanner(constraint_cost=bounds).plan(s,cs),BatchMPCPlanner(constraint_cost=bounds).plan(s,cs))

    def test_held_high_aperture_is_retained_without_fake_grasp(self):
        s=state(tcp_position=np.array([0.,0.,.92]),held=True,opening=.8,requested_opening=0.,lift_goal_position=np.array([0.,0.,1.05]))
        a=scalar.MPCPlanner().plan(s,candidates());b=BatchMPCPlanner().plan(s,candidates())
        self.equal_plan(a,b)
        self.assertTrue(all(row['held'] for row in b['predicted_states']))
        self.assertTrue(all(not row['held'] for row in self.b['predicted_states']))

    def test_duplicate_time_and_missed_close_hysteresis(self):
        for s in (state(tcp_position=np.array([0.,0.,.9]),time=.4,opening=.8,requested_opening=0.,
                        phase_state={'phase':'close','dwell':0.,'lost_hold':0.,'updated_at':.3,'close_entered_at':.3}),
                  state(tcp_position=np.array([0.,0.,.92]),time=.2,held=True,opening=.8,requested_opening=0.,
                        phase_state={'phase':'hold','dwell':0.,'lost_hold':0.,'updated_at':.2})):
            self.equal_plan(scalar.MPCPlanner().plan(s,candidates()),BatchMPCPlanner().plan(s,candidates()))

    def test_shifted_warm_budget_and_poisoned_execution_hidden(self):
        current=deepcopy(self.a['predicted_states'][1]);current['phase_state']=deepcopy(self.a['phase_state'])
        current['actuator_state']={'requested_opening':self.a['action']['opening']}
        wa=deepcopy(self.a['warm_start']);wb=deepcopy(self.b['warm_start'])
        wa['execution_h']=wb['execution_h']={'object_in_tcp':np.ones(3)*999,'requested_opening':0.}
        a=scalar.MPCPlanner().plan(current,candidates(),warm_start=wa)
        b=BatchMPCPlanner().plan(current,candidates(),warm_start=wb)
        self.equal_plan(a,b)
        self.assertEqual(b['diagnostics']['warm_start_evaluations_counted'],3)
        self.assertNotIn('execution_h',b['warm_start'])

    def test_all_infeasible_even_when_constraint_weight_zero(self):
        cs=[dict(row,native_pose_qualified=False) for row in candidates()]
        config=scalar.MPCConfig();config.weights['constraint']=0.
        bounds=BatchBounds()
        self.equal_plan(scalar.MPCPlanner(config,constraint_cost=bounds).plan(state(),cs),
                        BatchMPCPlanner(config,constraint_cost=bounds).plan(state(),cs))

    def test_knots3_and_delayed_response_explicit_config(self):
        config=scalar.MPCConfig(knots=3,opening_command_delay=.05)
        s=state(tcp_position=np.array([0.,0.,.9]),time=.3,phase_state={'phase':'align','dwell':.1,'lost_hold':0.,'updated_at':.2})
        self.equal_plan(scalar.MPCPlanner(config).plan(s,candidates()),BatchMPCPlanner(config).plan(s,candidates()))

    def test_invalid_callback_and_new_deadline_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'batch callback'):
            BatchMPCPlanner(constraint_cost=lambda *_:0.)
        changed=candidates();next(row for row in changed if row['id']==self.b['selected_candidate_id'])['deadline']+=.1
        with self.assertRaisesRegex(ValueError,'deadline'):
            BatchMPCPlanner().plan(state(time=.1),changed,warm_start=self.b['warm_start'])
        bad=state(time=.1,phase_state={'phase':'align','updated_at':.2,'dwell':0.,'lost_hold':0.})
        with self.assertRaisesRegex(ValueError,'backwards'):
            BatchMPCPlanner().plan(bad,candidates())


if __name__=='__main__':unittest.main()
