from dataclasses import replace
import unittest
import numpy as np
import test_remove_retarget_v66 as ports
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.button_contact_corridor import prove


class Tests(unittest.TestCase):
    def fixture(self,end_x=.045):
        q0=np.array([.1,0.,.88,0.,0.,0.,0.]);q1=q0.copy();q1[:3]=[end_x,0.,.75]
        plan=JointTrajectory(0.,.5,q0,np.zeros(7),q1,np.zeros(7))
        robot=RobotState(0.,Pose(q0[:3]),np.zeros(3),np.zeros(3),q0,np.zeros(7),True,False)
        def body(name,z):return ObjectState(name,0.,Pose([0.,0.,z]),np.tile([0.,0.,z],(3,1)),np.array([.1,0.,0.]),np.zeros(3))
        bodies={b.object_id:b for b in (body('button',.76),body('plate',.775))}
        state=PolicyState(robot,bodies,{}, {'button':('button',)})
        forecast=ConstantVelocityDynamics().predict(list(bodies.values()),1.1,.05)
        forecast.metadata['observed_until']=0.
        asset={'bodies':{'plate':{'collision_vertices_body':[[x,y,z] for x in (-.03,.03) for y in (-.03,.03) for z in (-.003,.003)]}}}
        return ports.Cartesian(),plan,state,forecast,asset

    def test_diagonal_approach_can_descend_when_complete_predicted_contact_path_is_aligned(self):
        model,plan,state,forecast,asset=self.fixture()
        accepted,proof=prove(model,plan,state,forecast,'button',asset,.005)
        self.assertTrue(accepted,proof)
        self.assertGreater(proof['contact_samples'],0)
        self.assertGreater(np.linalg.norm(state.robot.pose.position[:2]),proof['contact_radius_m'])

    def test_low_sideways_crossing_is_rejected_before_dispatch(self):
        model,plan,state,forecast,asset=self.fixture(.15)
        accepted,proof=prove(model,plan,state,forecast,'button',asset,.005)
        self.assertFalse(accepted)
        self.assertEqual(proof['reason'],'reference_crosses_plate_height_outside_contact_disk')

    def test_stale_missing_or_no_contact_plane_is_not_a_descent_certificate(self):
        for condition in ('stale','missing','high'):
            model,plan,state,forecast,asset=self.fixture()
            if condition=='stale':forecast.metadata['observed_until']=-.2
            if condition=='missing':state.bodies.pop('plate')
            if condition=='high':plan=replace(plan,q1=plan.q0.copy())
            accepted,proof=prove(model,plan,state,forecast,'button',asset,.005)
            self.assertFalse(accepted,proof)

    def test_actual_button_factory_admits_full_dual_proof_and_keeps_native_completion(self):
        from pathlib import Path
        from types import MethodType
        import json
        import test_button_author_v359 as factory
        from our_mani.multitask.moving_reference import install_moving_reference
        from our_mani.multitask.button_contact_corridor import install
        from our_mani.multitask.reference_constraint_authority import install as install_authority
        from our_mani.multitask.interaction import InteractionCandidate,_world_target
        helper=factory.Tests();helper.setUp()
        try:
            actor=helper.actor();c,m=actor.controller,actor.continuous_reference_manager
            install_moving_reference(actor,phases=('press',));install_authority(actor)
            root=Path(__file__).resolve().parents[1]
            config=json.loads((root/'artifacts/framework/multitask_r10/button_reference_v389_setup/configuration.json').read_text())
            asset=json.loads((root/config['learned_artifacts']['push_moving_button']['geometry']['path']).read_text())
            c.step=MethodType(lambda self,*args:{'command':'hold'},c)
            events=install(actor,asset);c.phase_index=1
            template=c.library.for_phase('press')[0]
            body=ObjectState('button',1.,Pose([.2,0.,.8]),np.tile([.2,0.,.8],(3,1)),np.zeros(3),np.zeros(3))
            plate=replace(body,object_id='plate',pose=Pose([.2,0.,.814]))
            target=_world_target(body.pose,np.zeros(3),np.zeros(3),template,1.)[0]
            pose=Pose(target.position+[0.,0.,.085],target.rotation)
            q=m.native_gate.model.inverse(pose,np.zeros(7))
            robot=RobotState(1.,pose,np.zeros(3),np.zeros(3),q,np.zeros(7),True,False)
            state=PolicyState(robot,{'button':body,'plate':plate},{'bodies':{'button':dict(
                directly_visible=True,registration_valid=True,motion_mode='external')}},{'button':('button',)})
            forecast=ConstantVelocityDynamics().predict([body,plate],1.1,.05);forecast.metadata['observed_until']=1.
            m.last_confirmed=dict(time=1.,joints=q.tolist(),nominal_velocity=np.zeros(7).tolist(),command='joint_trajectory')
            c.generator.collision_check=lambda **kw:dict(feasible=True,evidence={})
            c.generator.bind_candidate_scene=lambda *a,**kw:None
            checked=m.native_gate.check_joints(robot,q,np.zeros(7),.5)
            c.goal=InteractionCandidate('CPU_corridor_press','press',1,'button',(),template,.5,1.5,
                target,np.zeros(3),np.zeros(3),body.pose,np.zeros(3),np.zeros(3),body,checked,0.,
                template.duration_seconds,9.5,0.,True,{})
            c.step(state,forecast,9.)
            admitted=[e for e in events if e.get('accepted')]
            self.assertTrue(admitted,events)
            self.assertTrue(m.check_stream(robot,.05).feasible)
            self.assertFalse(c._phase_done(state,target,public_geometry=True))
            self.assertTrue(admitted[-1]['original_full_reference_and_actual_gates_passed'])
        finally:
            helper.doCleanups()


if __name__=='__main__':unittest.main()
