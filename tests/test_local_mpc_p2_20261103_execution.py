"""P2 API preparation: pure CPU, no trained network/Native success claims."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.local_mpc_p2_20261103.execution_models import (
    initialize_execution_batch, validate_feedback_state, validate_step_batch,
    SimpleBatchExecutionModel, JointResidualFKExecutionModel)
from our_mani.local_mpc_p2_20261103.batch_mpc_pluggable import BatchMPCPlanner
from our_mani.local_mpc_p2_20261103 import reference as ref
from our_mani.local_mpc_20261102.batch_mpc import BatchMPCPlanner as InlinePlanner
from our_mani.local_mpc_20261102 import mpc as reference

def make_state(**kw):
    state=dict(tcp_position=np.array([0.,0.,1.05]),tcp_quaternion=np.array([0.,0.,0.,1.]),tcp_velocity=np.zeros(3),
               object_position=np.array([0.,0.,.9]),object_quaternion=np.array([0.,0.,0.,1.]),
               object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),opening=1.,held=False,time=0.)
    state.update(kw);return state
def options():
    return [dict(id=f'm{i}',object_tcp_position=np.zeros(3),object_tcp_quaternion=reference.quat_exp(np.array([0.,0.,(i//2)*.15])),
                 deadline=.5+.3*(i%2),approach_direction=np.array([0.,0.,1.]),pregrasp_offset=.06,opening=0.,required_opening=1.,allowed_relative_speed=.1)
            for i in range(8)]

def parity(a,b):
    assert a['budget']==b['budget'],(a['budget'],b['budget'])
    assert a['selected_candidate_id']==b['selected_candidate_id'],(a.get('cost'),b.get('cost'),a['selected_candidate_id'],b['selected_candidate_id'])
    if a['action'] is None:
        assert b['action'] is None
        return {'max_numeric_error':0.,'cost_error':0.,'all_infeasible':True}
    assert a['phase_state']['phase']==b['phase_state']['phase']
    error=abs(a['cost']-b['cost'])
    differences=[error]
    for old,new in zip(a['sequence'],b['sequence']):
        assert old['opening']==new['opening']
        for key in ('tcp_position','tcp_quaternion','tcp_velocity_reference','angular_velocity_reference'):
            delta=np.asarray(old[key])-new[key]
            differences.append(float(np.max(np.abs(delta))))
    for old,new in zip(a['predicted_states'],b['predicted_states']):
        assert old['held']==new['held']
        assert old['time']==new['time']
        for key in ('tcp_position','tcp_quaternion','tcp_velocity','tcp_angular_velocity','object_position','object_quaternion',
                    'object_linear_velocity','object_angular_velocity','opening'):
            differences.append(float(np.max(np.abs(np.asarray(old[key])-new[key]))))
    differences.extend([float(np.max(np.abs(a['warm_start'][key]-b['warm_start'][key]))) for key in ('knot_means','knot_stds')])
    assert max(differences)<1e-9,max(differences)
    return {'max_numeric_error':max(differences),'cost_error':error,'phase':a['phase_state']['phase']}


def joint_state(**kwargs):
    s=make_state(tcp_position=np.array([0.,0.,.9]),q=np.zeros(7),dq=np.zeros(7),requested_opening=1.,
        actuator_state=dict(reference_valid=False,reference_positions=None,reference_velocities=None,
                            reference_accelerations=None,target_joints=None,gripper_target=1.))
    s.update(kwargs);return s


def command(n,joint_goal=None,opening=1.):
    u=dict(tcp_position=np.tile([0.,0.,.9],(n,1)),tcp_quaternion=np.tile([0.,0.,0.,1.],(n,1)),
           opening=np.full(n,opening),tcp_velocity_reference=np.zeros((n,3)),angular_velocity_reference=np.zeros((n,3)))
    if joint_goal is not None:u['joint_goal']=np.broadcast_to(joint_goal,(n,7)).copy()
    return u


def fk(q):
    return dict(tcp_position=q[:,:3]+np.array([0.,0.,.9]),tcp_quaternion=np.tile([0.,0.,0.,1.],(len(q),1)))


def lag_transition(q,dq,q_ref,dq_ref,ddq_ref,latent,dt):
    nxt=q+.4*(q_ref-q)
    return dict(q=nxt,dq=(nxt-q)/dt,latent=latent,feasible=np.ones(len(q),bool))


class PluggableTests(unittest.TestCase):
    def test_actual_measurements_and_actual_controller_references_are_distinct(self):
        q=np.arange(7.)*.01;dq=np.ones(7)*.02
        actuator=dict(reference_valid=True,reference_positions=q+.1,reference_velocities=dq+.03,
                      reference_accelerations=np.ones(7)*.4,target_joints=q+.2,gripper_target=0.)
        s=joint_state(q=q,dq=dq,requested_opening=0.,actuator_state=actuator)
        x,h=initialize_execution_batch(s,88,requires_joint_state=True)
        self.assertEqual(x['q'].shape,(88,7));self.assertEqual(h['q_ref'].shape,(88,7))
        np.testing.assert_array_equal(x['q'][0],q)
        np.testing.assert_array_equal(h['q_ref'][0],q+.1)
        np.testing.assert_array_equal(h['goal_q'][0],q+.2)
        self.assertTrue(h['valid'].all());self.assertTrue(h['initialized'].all())
        self.assertFalse(h['reference_is_measured_state']);self.assertFalse(h['predicted'])
        self.assertTrue((h['requested_opening']==0.).all())

    def test_invalid_reference_preserves_actual_previous_acceleration_without_fake_validity(self):
        s=joint_state(actuator_state=dict(reference_valid=False,reference_positions=np.ones(7)*.1,
            reference_velocities=np.ones(7)*.2,reference_accelerations=np.ones(7)*.3,target_joints=None))
        x,h=initialize_execution_batch(s,3,requires_joint_state=True)
        self.assertFalse(h['valid'].any());self.assertTrue(h['initialized'].all())
        np.testing.assert_array_equal(h['ddq_ref'],np.ones((3,7))*.3)
        np.testing.assert_array_equal(x['q'],np.zeros((3,7)))
        with self.assertRaisesRegex(ValueError,'actual q/dq'):
            initialize_execution_batch(make_state(),2,requires_joint_state=True)

    def test_simple_model_functional_inputs_no_feedforward_and_no_fake_attachment(self):
        model=SimpleBatchExecutionModel()
        actual=joint_state(held=True,opening=.8,requested_opening=0.)
        x,h=model.initialize_batch(actual,2);before_x=deepcopy(x);before_h=deepcopy(h)
        nxt,h_next,info=model.step_batch(x,command(2,opening=0.),h,.1)
        validate_step_batch(x,command(2,opening=0.),nxt,h_next,info,.1)
        self.assertTrue(nxt['held'].all())
        for key in x:np.testing.assert_array_equal(x[key],before_x[key])
        self.assertFalse(h['predicted']);self.assertTrue(h_next['predicted'])
        np.testing.assert_array_equal(h['q_ref'],before_h['q_ref'])
        opened,_,_=model.step_batch(nxt,command(2,opening=1.),h_next,.1)
        self.assertFalse(opened['held'].any())
        fresh,hidden=model.initialize_batch(joint_state(),2)
        closed,_,_=model.step_batch(fresh,command(2,opening=0.),hidden,.1)
        self.assertFalse(closed['held'].any())

    def test_default_model_exact_inline_parity_warm_budget_and_feedback_reset(self):
        s=make_state();cs=options()
        a=InlinePlanner().plan(s,cs);model=SimpleBatchExecutionModel()
        calls=[]
        init=model.initialize_batch
        def observed(actual,n):
            calls.append((n,actual['time'],deepcopy(actual.get('actuator_state'))))
            return init(actual,n)
        model.initialize_batch=observed
        b=BatchMPCPlanner(execution_model=model).plan(s,cs)
        parity(a,b)
        self.assertEqual([row[0] for row in calls],[88,88,80])
        self.assertEqual(b['diagnostics']['execution_model_calls'],24)
        warm=deepcopy(b['warm_start']);warm['execution_h']={'requested_opening':0.,'q_ref':np.ones((88,7))*999}
        feedback=deepcopy(b['predicted_states'][1]);feedback['requested_opening']=1.
        result=BatchMPCPlanner().plan(feedback,cs,warm_start=warm)
        self.assertNotIn('execution_h',result['warm_start'])
        self.assertFalse(result['diagnostics']['virtual_h_used_as_next_real_feedback'])
        self.assertEqual(result['budget']['sequences'],256)

    def test_joint_model_uses_predicted_actual_between_two_reference_steps_then_fk(self):
        limits=ref.ReferenceLimits(np.ones(7)*-2,np.ones(7)*2,np.ones(7)*2)
        seen=[]
        def reference_step(q,dq,goal,h,limits,**kwargs):
            out=ref.reference_step(q,dq,goal,h,limits,**kwargs)
            seen.append((q.copy(),out['q_ref'].copy()))
            return out
        model=JointResidualFKExecutionModel(reference_step,limits,lag_transition,fk)
        x,h=model.initialize_batch(joint_state(),4)
        nxt,hn,info=model.step_batch(x,command(4,joint_goal=np.ones(7)*.2),h,.1)
        validate_step_batch(x,command(4,joint_goal=np.ones(7)*.2),nxt,hn,info,.1)
        self.assertEqual(len(seen),2)
        np.testing.assert_array_equal(seen[1][0],info['predicted_actual_q'][:,0])
        self.assertGreater(float(np.max(np.abs(seen[1][0]-seen[0][1]))),0.)
        np.testing.assert_array_equal(nxt['tcp_position'],fk(nxt['q'])['tcp_position'])
        self.assertFalse(nxt['held'].any());self.assertTrue(hn['predicted'])
        self.assertFalse(h['predicted'])

    def test_joint_goal_adapter_is_explicit_and_missing_goal_never_falls_back(self):
        limits=ref.ReferenceLimits(np.ones(7)*-2,np.ones(7)*2,np.ones(7)*2)
        model=JointResidualFKExecutionModel(ref.reference_step,limits,lag_transition,fk)
        with self.assertRaisesRegex(ValueError,'joint_goal'):
            BatchMPCPlanner(execution_model=model).plan(joint_state(),options())
        def mapper(x,u,h,context):
            goal=x['q'].copy();goal[:,:3]+=u['tcp_position']-x['tcp_position']
            return {'joint_goal':goal}
        result=BatchMPCPlanner(execution_model=model,action_adapter=mapper).plan(joint_state(),options())
        self.assertIn('joint_goal',result['action'])
        self.assertIn('q',result['predicted_states'][1])
        self.assertEqual(result['diagnostics']['execution_model_calls'],24)
        self.assertEqual(result['budget']['virtual_50ms_substeps'],4096)

    def test_clock_and_implicit_contact_model_contract_fail_closed(self):
        class BadClock(SimpleBatchExecutionModel):
            def step_batch(self,x,u,h,dt):
                nxt,hn,info=super().step_batch(x,u,h,dt);nxt['time']+=.05
                return nxt,hn,info
        with self.assertRaisesRegex(ValueError,'clock'):
            BatchMPCPlanner(execution_model=BadClock()).plan(make_state(),options())
        class FakeHeld(SimpleBatchExecutionModel):
            def step_batch(self,x,u,h,dt):
                nxt,hn,info=super().step_batch(x,u,h,dt);nxt['held'][:]=True
                return nxt,hn,info
        with self.assertRaisesRegex(ValueError,'attachment'):
            BatchMPCPlanner(execution_model=FakeHeld()).plan(make_state(),options())

    def test_joint_transition_must_disclose_feasibility_and_failed_rows_are_not_elites(self):
        limits=ref.ReferenceLimits(np.ones(7)*-2,np.ones(7)*2,np.ones(7)*2)
        def missing_mask(q,dq,q_ref,dq_ref,ddq_ref,latent,dt):
            return dict(q=q,dq=dq)
        model=JointResidualFKExecutionModel(ref.reference_step,limits,missing_mask,fk)
        x,h=model.initialize_batch(joint_state(),2)
        with self.assertRaisesRegex(ValueError,'disclose feasible'):
            model.step_batch(x,command(2,joint_goal=np.zeros(7)),h,.1)
        class AllInfeasible(SimpleBatchExecutionModel):
            def step_batch(self,x,u,h,dt):
                nxt,hn,info=super().step_batch(x,u,h,dt)
                info['feasible'][:]=False
                return nxt,hn,info
        result=BatchMPCPlanner(execution_model=AllInfeasible()).plan(make_state(),options())
        self.assertIsNone(result['action'])
        self.assertEqual(result['budget']['sequences'],256)

    def test_pose_joint_adapter_infeasible_rows_cannot_be_selected(self):
        def unreachable(x,u,h,context):
            return dict(joint_goal=x['q'].copy(),joint_goal_feasible=np.zeros(len(x['q']),dtype=bool))
        result=BatchMPCPlanner(action_adapter=unreachable).plan(joint_state(),options())
        self.assertIsNone(result['action'])
        self.assertEqual(result['budget']['sequences'],256)


if __name__=='__main__':unittest.main()
