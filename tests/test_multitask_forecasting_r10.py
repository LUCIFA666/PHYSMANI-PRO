"""CPU-only public forecast bridge: no simulator, GPU or GRU fitting."""
import threading
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Forecast,ObjectState,Pose,RobotState
from our_mani.multitask.forecasting import (ForecastConfig,ForecastingError,MultiBodyForecastBridge,
                                          ObservedAttachment)
from our_mani.multitask.perception import PriorProvenance,RigidChildConstraint


def robot(time,contact=False):
    return RobotState(time,Pose(np.array([0.,0.,1.])),np.zeros(3),np.zeros(3),
                      np.zeros(7),np.zeros(7),contact=contact)


def body(name,time,*,visible=True,rotation=None,angular=(0.,0.,1.)):
    rotation=Rotation.from_rotvec(np.array([0.,0.,time])).as_matrix() if rotation is None else rotation
    return ObjectState(name,time,Pose(np.array([time,0.,.8]),rotation),np.ones((3,3)),
                       np.array([1.,0.,0.]),np.asarray(angular),visible=visible)


def metadata(states,*,declared='full',full=True,edges=(),root='frame'):
    return dict(bodies={s.object_id:dict(visible=s.visible,registration_valid=s.visible,
            orientation_observable=full,orientation_observability=declared) for s in states},
        template_priors={s.object_id:dict(symmetry_axis=[0.,0.,1.]) for s in states},
        rigid_groups=[dict(root=root,members=['frame','peg0'],registration_valid=True,
            rigidity_consistency_tested=all(s.visible for s in states),
            TRAIN_constraints=[dict(parent=e.parent_role,child=e.child_role,evidence_sha256=e.provenance.evidence_sha256)
                               for e in edges])] if edges else [])


def edge(parent='frame',child='peg0'):
    source=PriorProvenance('train_calibrated_rigid_layout','CPU synthetic TRAIN layout',
        'child-local coordinates in parent frame',True,'train',('train:CPU_fixture:parent',),'a'*64)
    return RigidChildConstraint(parent,child,Pose(np.array([.1,0.,0.])),source)


class ZeroGRUContext:
    """Explicit test engine; exercises adapter calls without a neural network."""
    def __init__(self):
        self.metadata=dict(training_complete=True,TEST_used_for_fitting=False,state_schema='CPU_stub_only')
        self.checkpoint_sha256='b'*64
        self.encoder_sha256='c'*64
        self.maximum_validated_horizon=1.1
        self.lock=threading.RLock()
        self.history={}
        self.prediction_calls=0
        self.observation_flags=[]

    def reset(self):
        self.history.clear()

    def observe(self,objects,*,modes,orientation_observability):
        if set(orientation_observability) != {o.object_id for o in objects}:
            raise AssertionError('missing explicit orientation flag')
        for obj in objects:
            self.history.setdefault(obj.object_id,[]).append(obj)
            self.observation_flags.append((obj.object_id,orientation_observability[obj.object_id]))
            if modes[obj.object_id]=='attached':
                raise AssertionError('attached body entered external context')

    def predict(self,objects,horizon,dt):
        self.prediction_calls+=1
        elapsed=np.arange(int(round(horizon/dt))+1)*dt
        p=np.stack([o.pose.position+elapsed[:,None]*o.linear_velocity for o in objects],axis=1)
        r=np.stack([Rotation.from_rotvec(elapsed[:,None]*o.angular_velocity).as_matrix()@o.pose.rotation for o in objects],axis=1)
        v=np.tile(np.stack([o.linear_velocity for o in objects]),(len(elapsed),1,1))
        w=np.tile(np.stack([o.angular_velocity for o in objects]),(len(elapsed),1,1))
        return Forecast(tuple(o.object_id for o in objects),objects[0].timestamp+elapsed,p,r,v,w,
            metadata=dict(contexts={o.object_id:[.125]*16 for o in objects}))


class ForecastBridgeChecks(unittest.TestCase):
    def feed(self,bridge,name='ball',*,count=5,start=0.,full=True,declared='full',angular=(0.,0.,1.)):
        for k in range(count):
            now=start+k*.05
            state=body(name,now,angular=angular)
            bridge.observe([state],metadata([state],full=full,declared=declared),robot(now),modes={name:'external'})
        return robot(now)

    def test_CV_and_context_share_five_frame_warmup_without_GRU(self):
        cv=MultiBodyForecastBridge(('ball',))
        engine=ZeroGRUContext();context=MultiBodyForecastBridge(('ball',),method='context',context=engine)
        r=self.feed(cv,count=4);self.feed(context,count=4)
        self.assertFalse(cv.predict(r).object_ids)
        self.assertFalse(context.predict(r).object_ids)
        self.assertEqual(engine.prediction_calls,0)
        r=self.feed(cv,count=1,start=.2);self.feed(context,count=1,start=.2)
        a,b=cv.predict(r),context.predict(r)
        self.assertEqual(a.object_ids,('ball',))
        self.assertEqual(a.positions.shape,(23,1,3))
        np.testing.assert_allclose(a.positions,b.positions)
        np.testing.assert_allclose(a.rotations,b.rotations)
        self.assertEqual(engine.prediction_calls,1)
        self.assertEqual(engine.observation_flags,[('ball',True)]*5)

    def test_verified_rigid_child_composes_parent_twist_and_copies_context(self):
        constraint=edge();engine=ZeroGRUContext()
        bridge=MultiBodyForecastBridge(('frame','peg0'),method='context',context=engine,rigid_children=(constraint,))
        for k in range(5):
            time=k*.05;parent=body('frame',time)
            pose=Pose(parent.pose.apply(constraint.child_to_parent.position),parent.pose.rotation)
            child=ObjectState('peg0',time,pose,np.ones((3,3)),np.zeros(3),np.zeros(3),visible=False)
            states=[parent,child]
            bridge.observe(states,metadata(states,edges=(constraint,)),robot(time),modes={'frame':'external','peg0':'external'})
        result=bridge.predict(robot(.2))
        self.assertEqual(result.object_ids,('frame','peg0'))
        offset=result.positions[:,1]-result.positions[:,0]
        np.testing.assert_allclose(np.linalg.norm(offset,axis=1),.1)
        np.testing.assert_allclose(result.linear_velocities[:,1],result.linear_velocities[:,0]+np.cross(result.angular_velocities[:,0],offset))
        np.testing.assert_allclose(result.rotations[:,0],result.rotations[:,1])
        self.assertEqual(result.metadata['contexts']['peg0'],result.metadata['contexts']['frame'])
        self.assertFalse(result.metadata['directly_visible']['peg0'])
        self.assertFalse(child.visible)
        self.assertNotIn('peg0',engine.history)

    def test_external_root_occlusion_rejects_virtual_group_measurement_and_rewarms(self):
        constraint=edge();bridge=MultiBodyForecastBridge(('frame','peg0'),rigid_children=(constraint,))
        for k in range(5):
            time=k*.05;states=[body('frame',time),body('peg0',time)]
            bridge.observe(states,metadata(states,edges=(constraint,)),robot(time),modes={'frame':'external','peg0':'external'})
        states=[body('frame',.25,visible=False),body('peg0',.25)]
        bridge.observe(states,metadata(states,edges=(constraint,)),robot(.25),modes={'frame':'external','peg0':'external'})
        result=bridge.predict(robot(.25))
        self.assertFalse(result.object_ids)
        self.assertIn('frame',result.metadata['missing_body_ids'])
        self.assertIn('peg0',result.metadata['missing_body_ids'])
        states=[body('frame',.3),body('peg0',.3)]
        bridge.observe(states,metadata(states,edges=(constraint,)),robot(.3),modes={'frame':'external','peg0':'external'})
        self.assertFalse(bridge.predict(robot(.3)).object_ids)
        self.assertEqual(len(bridge.history['frame']),1)

    def test_attachment_survives_visual_occlusion_without_external_rollout(self):
        engine=ZeroGRUContext();bridge=MultiBodyForecastBridge(('ball',),method='context',context=engine)
        self.feed(bridge)
        state=body('ball',.25,visible=False)
        attachment=ObservedAttachment('ball',Pose(np.array([0.,0.,-.1])),.2)
        bridge.observe([state],metadata([state]),robot(.25,contact=True),modes={'ball':'attached'},attachments={'ball':attachment})
        forecast=bridge.predict(robot(.25,contact=True))
        self.assertFalse(forecast.object_ids)
        self.assertEqual(forecast.metadata['attached_body_ids'],['ball'])
        self.assertIn('ball',bridge.attachment_updates())
        self.assertFalse(state.visible)
        self.assertFalse(engine.history)
        state=body('ball',.3)
        bridge.observe([state],metadata([state]),robot(.3),modes={'ball':'released'})
        self.assertFalse(bridge.attachment_updates())
        self.assertEqual(len(bridge.history['ball']),1)
        self.assertFalse(bridge.predict(robot(.3)).object_ids)

    def test_unknown_mode_gap_and_public_press_do_not_enter_context(self):
        engine=ZeroGRUContext();bridge=MultiBodyForecastBridge(('plate',),method='context',context=engine)
        self.feed(bridge,'plate')
        state=body('plate',.4)
        bridge.observe([state],metadata([state]),robot(.4),modes={'plate':'external'})
        self.assertEqual(len(bridge.history['plate']),1)
        state=body('plate',.45)
        bridge.observe([state],metadata([state]),robot(.45),modes={'plate':'unknown'})
        self.assertFalse(engine.history)
        state=body('plate',.5)
        bridge.observe([state],metadata([state]),robot(.5),modes={'plate':'external'},non_exogenous_ids={'plate'})
        self.assertFalse(engine.history)
        self.assertIn('plate',bridge.predict(robot(.5)).metadata['missing_body_ids'])

    def test_none_rotation_has_no_spin_axis_preserves_only_observed_axis_twist(self):
        bridge=MultiBodyForecastBridge(('ball',))
        r=self.feed(bridge,full=False,declared='none',angular=(0.,0.,99.))
        result=bridge.predict(r)
        np.testing.assert_allclose(result.angular_velocities,0.)
        self.assertFalse(result.metadata['full_orientation_observable']['ball'])
        axis=MultiBodyForecastBridge(('ball',))
        for k in range(5):
            time=k*.05;state=body('ball',time,rotation=np.eye(3),angular=(0.,1.,4.))
            axis.observe([state],metadata([state],declared='axis',full=False),robot(time),modes={'ball':'external'})
        result=axis.predict(robot(.2))
        np.testing.assert_allclose(result.angular_velocities[:,0],np.tile([0.,1.,0.],(23,1)))
        self.assertFalse(result.metadata['full_orientation_observable']['ball'])

    def test_unknown_root_orientation_and_conflicting_TRAIN_evidence_block_child(self):
        constraint=edge();bridge=MultiBodyForecastBridge(('frame','peg0'),rigid_children=(constraint,))
        for k in range(5):
            time=k*.05;states=[body('frame',time),body('peg0',time)]
            bridge.observe(states,metadata(states,edges=(constraint,),declared='none',full=False),robot(time),modes={'frame':'external','peg0':'external'})
        result=bridge.predict(robot(.2))
        self.assertEqual(result.object_ids,('frame',))
        self.assertEqual(result.metadata['missing_body_reasons']['peg0'],'rigid_root_full_orientation_unobservable')
        with self.assertRaises(ForecastingError):
            MultiBodyForecastBridge(('frame','cup0'),rigid_children=(edge(child='cup0'),))

    def test_missing_body_stays_missing_and_collision_fails_closed(self):
        from our_mani.multitask.maps import KnownMapForecastCollision
        bridge=MultiBodyForecastBridge(('ball','hoop'))
        for k in range(5):
            time=k*.05;state=body('ball',time)
            bridge.observe([state],metadata([state]),robot(time),modes={'ball':'external','hoop':'external'})
        forecast=bridge.predict(robot(.2))
        self.assertEqual(forecast.object_ids,('ball',))
        self.assertEqual(forecast.metadata['missing_body_ids'],['hoop'])

        class MapPort:
            required_dynamic_role_ids=('ball','hoop')
            estimated_bodies={'ball':object(),'hoop':object()}
            estimated_attachments={}
            def check(self,*args,**kwargs):
                raise AssertionError('collision probe should not run with a missing body')

        collision=KnownMapForecastCollision(MapPort())
        collision.bind(forecast,.2)
        self.assertTrue(collision.check(.1))


if __name__ == '__main__':
    unittest.main()
