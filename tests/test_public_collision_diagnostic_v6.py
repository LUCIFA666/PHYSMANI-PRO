from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.multitask.public_collision_diagnostic_v6 import CollisionObserver


class Ghost:
    def __init__(self, collide=False):
        self.collide = collide
        self.calls = 0
    def is_collidable(self): return True
    def get_bounding_box(self): return [-.1,.1,-.1,.1,-.1,.1]
    def get_matrix(self): return np.eye(4)[:3]
    def check_collision(self, other=None):
        self.calls += 1
        return self.collide


def fixture(arm_collision=False, phase='lift'):
    table, ball = Ghost(), Ghost(True)
    count = []
    def arm(other=None):
        count.append(other)
        return arm_collision
    env = SimpleNamespace(arm=SimpleNamespace(check_arm_collision=arm))
    service = SimpleNamespace(static_ghosts=[table], map=[{'name':'table'}],
        estimated_bodies={'ball':SimpleNamespace(ghost=ball,fallback_ghost=None,ghosts=lambda:[ball])},
        estimated_attachments={'ball':object()}, _clock=lambda:(0.,0.),
        _clock_check=lambda previous,label:None)
    calls = []
    def check(*args,**kwargs):
        calls.append((args,kwargs))
        return env.arm.check_arm_collision() or ball.check_collision(table)
    service.check=check
    class Model:
        def collision(self,q,offset):return service.check({},geometry_modes={'ball':'primary'})
    model=Model();actor=SimpleNamespace(robot_model=model,collision_port=SimpleNamespace(
        bound={'phase':phase},frame=(SimpleNamespace(robot=SimpleNamespace(timestamp=4.)),None)))
    env.map_service=service
    observer=CollisionObserver(actor,env)
    return actor,observer,ball,table,count,calls


class Tests(unittest.TestCase):
    def test_held_pair_result_and_original_query_count_are_preserved(self):
        actor,observer,ball,table,arm,calls=fixture()
        self.assertTrue(actor.robot_model.collision(np.zeros(7),.2))
        self.assertEqual(len(calls),1);self.assertEqual(ball.calls,1);self.assertEqual(len(arm),1)
        row=observer.receipt()['events'][0]
        self.assertEqual(row['sample']['future_offset'],.2)
        self.assertEqual(row['held_world'][0]['held']['label'],'estimated:ball:primary')
        self.assertEqual(row['held_world'][0]['obstacle']['label'],'static:table')
        self.assertEqual(row['attachments'],['ball'])

    def test_arm_partner_queries_are_metric_only_and_preserve_original_boolean(self):
        actor,observer,ball,table,arm,calls=fixture(True)
        self.assertTrue(actor.robot_model.collision(np.zeros(7),.2))
        self.assertEqual(len(calls),1);self.assertEqual(ball.calls,0)
        self.assertEqual(len(arm),3)
        self.assertEqual(observer.extra_pair_queries,2)
        self.assertEqual(len(observer.events[0]['arm']['colliding_ghosts']),2)
        self.assertEqual(observer.receipt()['added_physics_ticks'],0)

    def test_nonlift_policy_is_unchanged_and_uninstrumented(self):
        actor,observer,ball,table,arm,calls=fixture(True,'acquire')
        self.assertTrue(actor.robot_model.collision(np.zeros(7),.2))
        self.assertEqual(observer.events,[]);self.assertEqual(len(arm),1)

    def test_record_cap_preserves_future_original_queries(self):
        actor,observer,ball,table,arm,calls=fixture(True)
        observer.events=[{} for _ in range(128)]
        self.assertTrue(actor.robot_model.collision(np.zeros(7),.2))
        self.assertEqual(len(observer.events),128);self.assertEqual(observer.extra_pair_queries,0)
        self.assertEqual(len(arm),1)

    def test_metric_partner_error_does_not_replace_original_result(self):
        actor,observer,ball,table,arm,calls=fixture(True)
        # The original bound arm function is instrumented at construction;
        # simulate an unavailable geometry read, which must remain diagnostic.
        ball.get_matrix=lambda:(_ for _ in ()).throw(RuntimeError('unsupported'))
        self.assertTrue(actor.robot_model.collision(np.zeros(7),.2))
        self.assertEqual(observer.events[0]['arm']['colliding_ghosts'][1]['geometry_unavailable'],'RuntimeError')


if __name__=='__main__':unittest.main()
