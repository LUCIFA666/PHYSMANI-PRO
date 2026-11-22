"""CPU mechanics contracts with the real executor/latch/qualifier; no Native."""
import copy
import json
import unittest
import numpy as np

from our_mani.physmani_feedback_interface_20261104_v1 import FeedbackFacadeV1, GripperRequestV1
from our_mani.physmani_feedback_interface_20261104_v1.observation import pack_feedback_snapshot


class Backend:
    def __init__(self, budget=100):
        self.q = np.zeros(7); self.dq = np.zeros(7); self.target = self.q.copy()
        self.t = 0.; self.ticks = 0; self.budget = budget; self.requested_opening = 1.
        self.grips = []; self.probes = []; self.ik_calls = 0; self.fail_ik = False
        self.collision = False; self.stop_after = None; self.bad_probe = False

    def clock(self): return self.t
    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.dq.copy()
    def joint_limits(self): return dict(lower=np.full(7, -2.), upper=np.full(7, 2.), velocity=np.full(7, 3.))
    def solve_ik(self, position, quaternion):
        self.ik_calls += 1
        if self.fail_ik: raise RuntimeError('synthetic IK infeasible')
        return np.asarray(position)[0] * np.ones(7)
    def probe_joint_collision(self, q):
        self.probes.append(np.asarray(q).copy())
        if self.bad_probe: self.t += .05
        return self.collision
    def remaining_ticks(self): return max(0, self.budget - self.ticks)
    def terminal(self): return self.stop_after is not None and self.ticks >= self.stop_after
    def set_joint_target_positions(self, q): self.target = np.asarray(q).copy()
    def actuate_gripper(self, opening):
        self.requested_opening = opening; self.grips.append(opening)
        return True
    def tick(self, phase):
        self.dq = (self.target-self.q)/.05; self.q = self.target.copy()
        self.t += .05; self.ticks += 1
    def metric_receipt(self): return {'native_success': False, 'private_native_metric': 'evaluation_only'}
    def read_snapshot(self):
        return pack_feedback_snapshot(physics_timestamp=10.+self.t,task_timestamp=20.+self.t,
            physics_time_origin=10.,task_time_origin=20.,rgb=np.zeros((4,256,256,3), np.uint8),
            depth_m=np.ones((4,256,256),np.float32),K=np.tile(np.eye(3),(4,1,1)),
            T=np.tile(np.eye(4),(4,1,1)),camera_clips=np.tile([.01,4.],(4,1)),
            render_modes=['OPENGL3','OPENGL','OPENGL','OPENGL'],q=self.q,dq=self.dq,
            tcp_pose=[0,0,0,0,0,0,1],finger_joint_positions=[.04,.04],
            finger_joint_velocities=[0.,0.],finger_open_amounts=[1.,1.])


class FeedbackContracts(unittest.TestCase):
    def test_zero_step_latest_and_stale_only_two_ticks(self):
        b = Backend(); f = FeedbackFacadeV1(b)
        first = f.submit_joint_target(np.zeros(7))
        second = f.submit_joint_target(np.zeros(7))
        self.assertEqual(b.ticks, 0)
        self.assertEqual(f.advance(first)['status'], 'stale_plan')
        for wrong in (1, 3, True, 2.):
            with self.assertRaises(ValueError): f.advance(second, max_ticks=wrong)
        r = f.advance(second)
        self.assertEqual((r['status'], r['executed_ticks'], b.ticks), ('cycle_completed', 2, 2))
        self.assertEqual(len(b.probes), 8)
        self.assertEqual([p['alpha'] for p in r['qualification']['probes']], [1.,.125,.25,.375,.5,.625,.75,.875])
        self.assertEqual(f.advance(second)['status'], 'stale_plan')
        self.assertEqual(b.ticks, 2)
        json.dumps(r, allow_nan=False)

    def test_keep_close_open_latch_and_existing_executor_not_reset(self):
        b = Backend(); f = FeedbackFacadeV1(b)
        def cycle(request):
            return f.advance(f.submit_joint_target(np.zeros(7), gripper_request=request))
        cycle('KEEP'); self.assertEqual(b.grips, [1.])
        cycle(GripperRequestV1.CLOSE); self.assertEqual(b.grips, [1.,0.])
        old_state = copy.deepcopy(f.executor.state())
        continued = FeedbackFacadeV1(b, f.executor)
        self.assertEqual(continued.executor.state(), old_state)
        continued.advance(continued.submit_joint_target(np.zeros(7)))
        self.assertEqual(b.grips, [1.,0.])
        cycle('OPEN'); cycle('KEEP'); self.assertEqual(b.grips, [1.,0.,1.])
        self.assertGreater(f.completed_gripper.gripper_completion_receipt()['completed_same_target_ticks_not_restarted'], 0)
        with self.assertRaises(ValueError): f.submit_joint_target(np.zeros(7),gripper_request=.9)

    def test_rejection_no_close_and_failed_ik_no_motion(self):
        b = Backend(); f = FeedbackFacadeV1(b); b.collision = True
        r = f.advance(f.submit_joint_target(np.ones(7)*.01, gripper_request='CLOSE'))
        self.assertEqual(r['status'], 'qualification_rejected')
        self.assertEqual((b.grips, b.ticks, f.executor.gripper_target), ([],0,1.))
        b.collision = False; b.fail_ik = True
        r = f.advance(f.submit_target([.1,0,0],[0,0,0,1],gripper_request='CLOSE'))
        self.assertEqual(r['status'], 'planning_failed')
        self.assertEqual((b.grips,b.ticks,b.ik_calls), ([],0,1))
        r = f.advance(f.submit_joint_target(np.ones(7)*3.,gripper_request='CLOSE'))
        self.assertEqual(r['qualification']['reason'], 'outside_native_joint_limits')
        self.assertEqual((b.grips,b.ticks), ([],0))

    def test_fresh_q_dq_h_clock_and_query_clock_fault(self):
        for mutate in (lambda b,f: setattr(b,'t',.05), lambda b,f: b.q.__setitem__(0,.01),
                       lambda b,f: b.dq.__setitem__(0,.01), lambda b,f: setattr(f.executor,'gripper_target',0.)):
            b = Backend(); f = FeedbackFacadeV1(b); p=f.submit_joint_target(np.zeros(7))
            mutate(b,f)
            self.assertEqual(f.advance(p)['status'], 'state_changed')
            self.assertEqual((b.grips,len(b.probes)), ([],0))
        b=Backend(); f=FeedbackFacadeV1(b); b.bad_probe=True
        r=f.advance(f.submit_joint_target(np.zeros(7),gripper_request='CLOSE'))
        self.assertEqual(r['status'], 'qualification_error')
        self.assertEqual((b.grips,b.ticks), ([],0))
        with self.assertRaises(RuntimeError): f.submit_joint_target(np.zeros(7))

    def test_terminal_and_budget_truncate_without_hidden_step_or_close(self):
        for stop in ('terminal','budget'):
            b=Backend(budget=1 if stop=='budget' else 100)
            if stop=='terminal': b.stop_after=1
            f=FeedbackFacadeV1(b)
            r=f.advance(f.submit_joint_target(np.zeros(7)))
            self.assertEqual((r['status'],r['executed_ticks'],b.ticks), ('truncated',1,1))
            previous=list(b.grips)
            r=f.advance(f.submit_joint_target(np.zeros(7),gripper_request='CLOSE'))
            self.assertEqual((r['status'],r['executed_ticks'],b.grips,b.ticks), ('truncated',0,previous,1))

    def test_bundle_exact_goal_and_reference_persistence(self):
        b=Backend(); f=FeedbackFacadeV1(b)
        q=np.full(7,.01)
        payload=dict(joint_goal=q.copy(),opening=0.,rank=3,cost=.7,
                     selected_bundle={'sequence':np.tile(q,(8,1)), 'diagnostic':float('inf')})
        p=f.submit_joint_target(payload)
        payload['joint_goal'][:]=1.;payload['selected_bundle']['sequence'][:]=1.
        r=f.advance(p)
        self.assertEqual(r['submitted_action']['rank'],3)
        np.testing.assert_array_equal(r['execution']['target_joints'],q)
        np.testing.assert_array_equal(r['submitted_action']['selected_bundle']['sequence'],np.tile(q,(8,1)))
        self.assertEqual((b.ik_calls,b.grips), (0,[1.]))  # no implicit opening from bundle
        acceleration=f.executor.state()['reference_accelerations']
        q2=q+.005
        p=f.submit_joint_target(q2)
        self.assertEqual(f.executor.state()['reference_accelerations'],acceleration)
        measured_velocity=b.dq.copy()
        r=f.advance(p)
        self.assertTrue(r['execution']['reference_reseeded'])
        np.testing.assert_array_equal(r['execution']['initial_measured_velocities'],measured_velocity)
        self.assertFalse(r['upstream_policy_phase_alignment_or_task_qualification_replaced'])

    def test_full_public_snapshot_metric_isolation_and_projection(self):
        b=Backend(); f=FeedbackFacadeV1(b); snap=f.read_snapshot()
        for key in ('dq','finger_joint_positions','finger_joint_velocities','finger_open_amounts','camera_clips'):
            self.assertIn(key,snap)
        self.assertNotIn('native_success',snap)
        self.assertIn('native_success',f.metric_receipt())
        legacy=f.legacy_policy_packet(snap)
        self.assertEqual(legacy['robot15'][-1],0.)
        self.assertFalse(legacy['robot15_observed_mask'][-1])
        self.assertEqual(b.ticks,0)
        original=b.read_snapshot
        def stale_content():
            packet=original(); packet['q']=packet['q'].copy();packet['q'][0]+=.01
            return packet
        b.read_snapshot=stale_content
        with self.assertRaisesRegex(ValueError,'same mechanical frame'): f.read_snapshot()
        self.assertEqual(b.ticks,0)


if __name__ == '__main__': unittest.main()
