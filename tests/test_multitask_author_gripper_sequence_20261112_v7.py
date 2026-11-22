from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.multitask_executor_cycle_20261112_v7.sequencing import author_sequence_gripper_request
from our_mani.multitask_executor_cycle_20261112_v1.sequencing import SequenceConfig
from our_mani.multitask_executor_cycle_20261112_v7.common import ROOT

class AuthorSequenceContracts(unittest.TestCase):
    def ref(self,opening):return SimpleNamespace(position=np.zeros(3),quaternion=np.array([0,0,0,1.]),opening_intent=opening)
    def test_contact_pose_blocked_by_surface_still_permits_original_gripper_sequence(self):
        opening,r=author_sequence_gripper_request(self.ref(0),[0,0,.025,0,0,0,1],1.,
            arm_sequence_completed=True,terminal=False,config=SequenceConfig())
        self.assertEqual(opening,0.)
        self.assertFalse(r['measured_arm_arrived'])
        self.assertFalse(r['task_success_inferred'])
        self.assertFalse(r['object_alignment_proven'])
    def test_no_early_gripper_before_arm_sequence_or_after_terminal(self):
        for complete,terminal in ((False,False),(True,True),(False,True)):
            opening,r=author_sequence_gripper_request(self.ref(0),[0,0,0,0,0,0,1],1.,
                arm_sequence_completed=complete,terminal=terminal,config=SequenceConfig())
            self.assertEqual(opening,1.)
            self.assertIsNone(r['gripper_intent_authorized_by'])
    def test_completed_arm_sequence_can_issue_release_without_claiming_success(self):
        opening,r=author_sequence_gripper_request(self.ref(1),[0,0,.02,0,0,0,1],0.,
            arm_sequence_completed=True,terminal=False,config=SequenceConfig())
        self.assertEqual(opening,1.)
        self.assertFalse(r['task_success_inferred'])
    def test_current_native_launcher_selects_author_sequence_worker(self):
        owner=(ROOT/'our_mani/multitask_executor_cycle_20261112_v7/owner.py').read_text()
        self.assertIn('run_multitask_executor_cycle_native_20261112_v7.sh',owner)
        script=(ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v7.sh').read_text()
        self.assertIn('multitask_executor_cycle_20261112_v7.native_worker',script)

if __name__=='__main__':unittest.main()
