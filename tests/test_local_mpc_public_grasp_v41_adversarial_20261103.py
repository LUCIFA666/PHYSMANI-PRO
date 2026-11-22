"""Independent v4.1 recovery tests; synthetic public sensors, CPU only.

History permits a closed-request hold, never a current held or lift assertion.
No Native labels, hidden attachment state, fitting, or physical execution.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualGraspCoordinator
from our_mani.local_mpc_control_20261103.public_grasp_v41 import (
    ConfirmedAttemptRecovery, ActiveVisualGraspV41Coordinator)
from tests.test_local_mpc_active_visual_attachment_adversarial_20261103 import (
    admitted, frame, actuator)


def public_result(attachment, target):
    mode = attachment['mode']
    return dict(timestamp=attachment['timestamp'], mode=mode,
                controlled_mode_available=mode == 'ATTACHED_PUBLIC',
                available=mode == 'ATTACHED_PUBLIC', object_observation=target,
                attachment_estimate=attachment)


def confirmed():
    active, spec = admitted();memory = ConfirmedAttemptRecovery()
    memory.observe(public_result(active.latest, active.last_target), active.last_robot)
    for i in range(1, 6):
        target, robot = frame(i, tcp_delta=(0., 0., .012 * i / 5), body_delta=(0., 0., .012 * i / 5))
        a = active.observe(target, robot, close_requested=True)
        result = public_result(a, target);memory.observe(result, robot)
    return active, memory, result, robot, spec


class Assembler:
    """Only isolate the decision guard; production input checks are unchanged."""
    robot_twist = (np.zeros(3), np.zeros(3), True)
    def __init__(self, current_attached=False):
        self.current_attached=current_attached;self.validated=[]
    def build(self, *args, **kwargs):
        return dict(grasp_motion_mode='ATTACHED_PUBLIC') if self.current_attached else None
    def _inputs(self, result, robot, h, phase):
        self.validated.append((result, robot, h, phase))


def coordinator(active, memory, assembler=None):
    return ActiveVisualGraspV41Coordinator(SimpleNamespace(), object(),
        assembler or Assembler(), active, recovery=memory)


class PublicGraspV41IndependentTests(unittest.TestCase):
    def test_confirmed_then_missing_stays_closed_unknown_after_original_deadline(self):
        active, memory, _, _, spec = confirmed();assembly=Assembler()
        control=coordinator(active, memory, assembly);control.warm={'old_predicted_lift': True}
        control.pending_started=0.
        for i in range(6, 23):
            _, robot=frame(i, tcp_delta=(0., 0., .012), body_delta=(0., 0., .012))
            result=public_result(active.observe(None, robot, close_requested=True), None)
            memory.observe(result, robot)
        self.assertGreater(robot.timestamp, spec['deadline'])
        with patch.object(ActiveVisualGraspCoordinator, 'decide', side_effect=AssertionError('old timeout path was called')):
            decision=control.decide(object(), result, robot, actuator(robot))
        self.assertEqual(decision['mode'], 'POST_CONFIRMED_UNKNOWN_HOLD')
        self.assertIsNone(decision['state']);self.assertIsNone(decision['action'])
        self.assertFalse(decision['current_held_asserted']);self.assertFalse(decision['continue_lift'])
        self.assertFalse(decision['automatic_reopen']);self.assertFalse(decision['GRU_used'])
        self.assertEqual(decision['existing_gripper_request'], 0.)
        self.assertIsNone(control.warm);self.assertEqual(len(assembly.validated), 1)

    def test_same_name_high_confidence_far_jump_is_not_release_or_permission_to_reopen(self):
        active, memory, _, _, _=confirmed();control=coordinator(active, memory)
        for i in range(6, 23):
            target, robot=frame(i, tcp_delta=(0., 0., .012), body_delta=(.20, 0., .090))
            result=public_result(active.observe(target, robot, close_requested=True), target)
            memory.observe(result, robot)
        self.assertEqual(target.object_id, 'target');self.assertEqual(target.confidence, 1.)
        self.assertGreater(np.linalg.norm(target.pose.position-robot.tcp_position), .12)
        with patch.object(ActiveVisualGraspCoordinator, 'decide', side_effect=AssertionError('far-target reopen path was called')):
            decision=control.decide(object(), result, robot, actuator(robot))
        self.assertEqual(decision['mode'], 'POST_CONFIRMED_UNKNOWN_HOLD')
        self.assertFalse(decision['automatic_reopen'])
        self.assertTrue(memory.preserves_closed_attempt(active.attempt_id))

    def test_midmacro_confirmation_is_preserved_even_if_no_decision_saw_attached(self):
        active, memory, _, _, _=confirmed()
        target, robot=frame(6, tcp_delta=(0., 0., .012), body_delta=(.08, 0., .012))
        result=public_result(active.observe(target, robot, close_requested=True), target)
        memory.observe(result, robot)
        # There has been no previous call to coordinator.decide in this test.
        control=coordinator(active, memory)
        with patch.object(ActiveVisualGraspCoordinator, 'decide', side_effect=AssertionError('missing 50ms event ledger')):
            decision=control.decide(object(), result, robot, actuator(robot))
        self.assertEqual(decision['mode'], 'POST_CONFIRMED_UNKNOWN_HOLD')

    def test_history_never_overrides_a_current_explicit_open_request(self):
        active, memory, _, _, _=confirmed()
        target, robot=frame(6, fingers=(.20,.65), tcp_delta=(0.,0.,.012), body_delta=(0.,0.,.012))
        result=public_result(active.observe(target, robot, close_requested=False), target)
        memory.observe(result, robot)
        control=coordinator(active, memory)
        decision=control.decide(object(), result, robot, actuator(robot, gripper_target=1.))
        self.assertEqual(decision['mode'], 'POST_CONFIRMED_EXPLICIT_OPEN_REQUEST_HOLD')
        self.assertEqual(decision['existing_gripper_request'], 1.)
        self.assertIsNone(decision['action']);self.assertFalse(decision['current_held_asserted'])

    def test_genuine_same_attempt_public_release_clears_recovery_but_not_before_window(self):
        active, memory, _, _, _=confirmed();attempt=active.attempt_id
        for i in range(6, 10):
            target, robot=frame(i, fingers=(1.,1.), tcp_delta=(.09,0.,.012), body_delta=(0.,0.,.012))
            result=public_result(active.observe(target, robot, close_requested=False), target)
            memory.observe(result, robot)
            self.assertEqual(memory.preserves_closed_attempt(attempt), i<9)
        self.assertEqual(result['attachment_estimate']['evidence'].event, 'release_confirmed')
        self.assertEqual(result['mode'], 'FREE')
        with patch.object(ActiveVisualGraspCoordinator, 'decide', return_value={'mode':'ORIGINAL_INPUT_HANDLING'}) as base:
            decision=coordinator(active,memory).decide(object(),result,robot,actuator(robot,gripper_target=1.))
        self.assertEqual(decision['mode'],'ORIGINAL_INPUT_HANDLING');base.assert_called_once()

    def test_current_legal_attached_state_delegates_lift_without_fabricating_a_plan(self):
        active, memory, result, robot, _=confirmed()
        marker={'mode':'ATTACHED_PUBLIC','action':{'selected_q_from_model':True}}
        with patch.object(ActiveVisualGraspCoordinator,'decide',return_value=deepcopy(marker)) as base:
            decision=coordinator(active,memory,Assembler(current_attached=True)).decide(object(),result,robot,actuator(robot))
        base.assert_called_once();self.assertEqual(decision['action'],marker['action'])
        self.assertFalse(decision['recovery']['historical_confirmation_is_current_held'])

    def test_unconfirmed_attempt_still_uses_original_pending_timeout_handling(self):
        active,_=admitted();target,robot=frame(1)
        result=public_result(active.observe(target,robot,close_requested=True),target)
        memory=ConfirmedAttemptRecovery();memory.observe(result,robot)
        with patch.object(ActiveVisualGraspCoordinator,'decide',return_value={'mode':'ORIGINAL_PRECONFIRMED_PATH'}) as base:
            decision=coordinator(active,memory).decide(object(),result,robot,actuator(robot))
        self.assertEqual(decision['mode'],'ORIGINAL_PRECONFIRMED_PATH');base.assert_called_once()

    def test_legacy_stale_or_serialized_positive_is_not_a_current_typed_confirmation(self):
        _,_,positive,_,_=confirmed()
        for variant in ('legacy','stale_attachment_time','stale_evidence','serialized'):
            with self.subTest(variant=variant):
                memory=ConfirmedAttemptRecovery();result=deepcopy(positive)
                _,robot=frame(6)
                result['timestamp']=robot.timestamp
                a=result['attachment_estimate']
                if variant=='legacy':a['protocol']='legacy';a['timestamp']=robot.timestamp
                elif variant=='stale_attachment_time':pass
                elif variant=='stale_evidence':a['timestamp']=robot.timestamp
                else:
                    a['timestamp']=robot.timestamp;a['evidence']={'event':'attachment_confirmed','timestamp':robot.timestamp}
                memory.observe(result,robot)
                self.assertFalse(memory.preserves_closed_attempt(a['attempt_id']))

    def test_nonfinite_result_timestamp_is_never_fresh_confirmation(self):
        _,_,result,robot,_=confirmed();memory=ConfirmedAttemptRecovery()
        result=deepcopy(result);result['timestamp']=float('nan')
        with self.assertRaises(ValueError):memory.observe(result,robot)
        self.assertFalse(memory.preserves_closed_attempt(result['attachment_estimate']['attempt_id']))

    def test_mismatched_release_attempt_or_closed_fingers_cannot_clear_ledger(self):
        for variant in ('wrong_attempt','fingers_not_open'):
            with self.subTest(variant=variant):
                active,memory,_,_,_=confirmed();attempt=active.attempt_id
                for i in range(6,10):
                    target,robot=frame(i,fingers=(1.,1.),tcp_delta=(.09,0.,.012),body_delta=(0.,0.,.012))
                    result=public_result(active.observe(target,robot,close_requested=False),target)
                if variant=='wrong_attempt':result['attachment_estimate']['attempt_id']=attempt+1
                else:
                    _,robot=frame(9,fingers=(.2,.2),tcp_delta=(.09,0.,.012),body_delta=(0.,0.,.012))
                memory.observe(result,robot)
                self.assertTrue(memory.preserves_closed_attempt(attempt))


if __name__=='__main__':unittest.main()
