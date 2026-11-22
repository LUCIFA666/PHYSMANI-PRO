from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.physmani_public_rrt_v561 import PublicRRTSkillExecutor,OriginalSkillExecutor


class Tests(unittest.TestCase):
    def fixture(self,phase='acquire',attached=()):
        captured=[]
        @contextmanager
        def scene(poses,**kw):
            captured.append(('scene',poses,kw));yield
        port=NS(_poses=lambda t:({'rubbish':Pose(np.zeros(3)),'tomato0':Pose(np.ones(3))},[]),
            _geometry_modes=lambda:({'rubbish':'primary','tomato0':'primary'},{}),
            bound={'allowed_robot_contact_ids':('rubbish',)})
        c=NS(generator=NS(bind_candidate_scene=lambda *a:None))
        f=NS(actor=NS(controller=c,collision_port=port,bound_attachments=set(attached)),
            state=NS(robot=NS(timestamp=1.)),forecast=NS(metadata={'observed_until':1.}))
        policy=NS(frontend=f,selected=dict(phase=phase,reference_id='rubbish',manipulated_ids=('rubbish',)))
        service=NS(plan_path=lambda *a,**kw:captured.append(('original_plan',kw)),path_queries=0,public_scene=scene)
        arm=NS(get_path=lambda *a,**kw:captured.append(('native_RRT',kw)) or object(),max_acceleration=12.)
        env=NS(map_service=service,arm=arm)
        executor=PublicRRTSkillExecutor(dict(dt=.05,baseline_decision_ticks=10),skill_policy=policy,
            observe_public_frame=lambda *a:None,admit_pose_action=lambda a:(True,{}),admit_joint_target=lambda q:(True,{}))
        return env,executor,captured

    def test_original_native_RRT_uses_public_obstacles_and_keeps_original_contact_permission(self):
        env,executor,seen=self.fixture();original=env.map_service.plan_path
        def run(instance,owned,action,observer=None):
            return owned.map_service.plan_path([0,0,0],[0,0,0,1],ignore_collisions=True,trials=100,algorithm='original')
        with patch.object(OriginalSkillExecutor,'execute',run):executor.execute(env,np.zeros(8))
        self.assertIs(env.map_service.plan_path,original)
        self.assertEqual(set(seen[0][1]),{'tomato0'})
        self.assertEqual(seen[1][1]['ignore_collisions'],False)
        self.assertEqual(seen[1][1]['trials'],100);self.assertEqual(seen[1][1]['algorithm'],'original')
        self.assertEqual(env.arm.max_acceleration,12.);self.assertEqual(env.map_service.path_queries,1)

    def test_nonacquire_and_public_attached_cases_preserve_original_planner(self):
        for phase,held in (('lift',()),('acquire',('rubbish',))):
            env,executor,seen=self.fixture(phase,held)
            def run(instance,owned,action,observer=None):return owned.map_service.plan_path([],[],ignore_collisions=True)
            with patch.object(OriginalSkillExecutor,'execute',run):executor.execute(env,np.zeros(8))
            self.assertEqual(seen,[('original_plan',{'ignore_collisions':True})])

    def test_owned_plan_binding_restores_when_native_RRT_fails(self):
        env,executor,seen=self.fixture();original=env.map_service.plan_path
        def fail(*a,**kw):raise RuntimeError('native path unavailable')
        env.arm.get_path=fail
        def run(instance,owned,action,observer=None):return owned.map_service.plan_path([],[])
        with patch.object(OriginalSkillExecutor,'execute',run):
            with self.assertRaisesRegex(RuntimeError,'native path unavailable'):executor.execute(env,np.zeros(8))
        self.assertIs(env.map_service.plan_path,original)
        self.assertFalse(executor.public_rrt_events[0]['path_found'])


if __name__=='__main__':unittest.main()
