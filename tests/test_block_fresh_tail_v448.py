from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.block_fresh_tail_v448 import SmoothBlockEnvironment,finish_counted_budget
import test_block_clip_brake_v315 as old


class Tests(unittest.TestCase):
    def fixture(self):
        env,source=old.Tests().fixture();b=SmoothBlockEnvironment(env)
        for k in ('reference_q','reference_v','public_robot','public_body','local_dynamics'):
            setattr(b,k,getattr(source,k))
        return env,b

    def test_rejected_short_cached_stop_can_use_a_fresh_slower_full_actual_reference_proof(self):
        env,b=self.fixture();b.backup=JointTrajectory(0.,.1,np.zeros(7),np.ones(7),np.full(7,.01),np.zeros(7))
        with b._collision_scope():b._brake(1.,'timeout')
        self.assertEqual(env.actions,1);self.assertTrue(np.all(env.target>0))
        self.assertEqual(b.dense_brakes[-1]['kind'],'fresh_dense_stop')
        self.assertFalse(b.dense_brakes[-1]['trials'][0]['actual']=='feasible_sampled_trajectory')
        self.assertEqual(b.reference_events[-1]['evidence']['reference_gate'],'feasible_sampled_trajectory')
        self.assertTrue(b.reference_events[-1]['evidence']['exact_next_tick_original_collision_gate_passed'])

    def test_collision_still_rejects_every_new_stop_and_stale_actual_state_is_not_used(self):
        env,b=self.fixture();env.robot_model.block=True
        with b._collision_scope():b._brake(1.,'blocked')
        self.assertEqual(env.actions,1);self.assertFalse(b.dense_brakes[-1]['admitted'])
        env,b=self.fixture();b.public_robot=replace(b.public_robot,timestamp=-.05)
        b._brake(1.,'stale')
        self.assertEqual(b.reference_events[-1]['reason'],'fresh_actual_state_required_for_dense_block_brake')

    def test_tail_refreshes_one_current_frame_per_unobserved_counted_tick_without_retrying_task(self):
        env=NS(time=9.85);seen=[];b=NS(env=env,ledger=NS(ticks=197,dt=.05),public_robot=NS(timestamp=9.8),
            public_body=None,tail_frames=[],local_dynamics=NS(observe=lambda bodies:seen.append(bodies[0].timestamp)))
        b.tail_tracker=NS(observe=lambda observation:(NS(timestamp=observation['timestamp']),{}))
        def observe():
            robot=NS(timestamp=env.time);b.public_robot=robot
            return dict(timestamp=env.time),robot
        b.observe=observe
        def hold():b.ledger.ticks+=1;env.time=b.ledger.ticks*.05
        b.hold_tick=hold;b.task=NS(success=lambda:(False,False))
        record=finish_counted_budget(b,dict(official_task_success=False,failure='original_timeout'))
        self.assertEqual(b.ledger.ticks,200);self.assertEqual(len(seen),3)
        self.assertEqual(record['failure'],'original_timeout')
        self.assertEqual(record['block_full_budget_v176']['policy_failed_retries'],0)
        self.assertEqual([e['counted_tick'] for e in b.tail_frames],[197,198,199])


if __name__=='__main__':unittest.main()
