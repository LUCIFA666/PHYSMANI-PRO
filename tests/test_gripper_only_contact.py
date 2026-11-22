from contextlib import contextmanager
from types import SimpleNamespace as NS
import unittest
from our_mani.multitask.gripper_only_contact import install


class Shape:
    def __init__(self,handle,name,colliding=()):self.handle=handle;self.name=name;self.colliding=set(colliding)
    def get_handle(self):return self.handle
    def get_name(self):return self.name
    def is_collidable(self):return True
    def check_collision(self,other):return other.handle in self.colliding


class Tests(unittest.TestCase):
    def fixture(self,colliding=()):
        gripper=Shape(1,'gripper_base');finger=Shape(2,'finger');arm=Shape(3,'panda_link')
        target=Shape(4,'public_cup',colliding);calls=[];scene=[False]
        @contextmanager
        def public_scene(*a,**kw):
            scene[0]=True
            try:yield {'cup0':target}
            finally:scene[0]=False
        service=NS(check=lambda *a,**kw:calls.append(kw) or False,public_scene=public_scene,estimated_attachments={})
        env=NS(gripper=NS(get_objects_in_tree=lambda **kw:[gripper,finger]),
            arm=NS(get_objects_in_tree=lambda **kw:[arm,gripper,finger]))
        actor=NS(controller=NS(feedback_step_calls=0),collision_port=NS(map=service),metrics_report=lambda:{})
        counts,events=install(actor,env,shape_type='CPU_shape')
        return actor,service,target,calls,scene,counts,events

    def test_gripper_contact_remains_allowed_but_arm_contact_is_rejected(self):
        for collision,result in (((1,2),False),((3,),True)):
            actor,service,_,calls,scene,counts,events=self.fixture(collision)
            self.assertEqual(service.check({},allowed_robot_contact_ids=('cup0',)),result)
            self.assertFalse(scene[0]);self.assertEqual(len(calls),1)
            self.assertEqual(counts['pairs'],1)
            self.assertEqual(bool(events),result)

    def test_held_target_is_also_checked_against_non_gripper_arm(self):
        _,service,_,_,scene,_,_=self.fixture((3,));service.estimated_attachments={'cup0':object()}
        self.assertTrue(service.check({}));self.assertFalse(scene[0])

    def test_original_collisions_and_pair_errors_are_not_suppressed(self):
        _,service,target,_,scene,_,_=self.fixture()
        target.check_collision=lambda _:(_ for _ in ()).throw(RuntimeError('native collision infrastructure'))
        with self.assertRaises(RuntimeError):service.check({},allowed_robot_contact_ids=('cup0',))
        self.assertFalse(scene[0])


if __name__=='__main__':unittest.main()
