import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from our_mani.executor_continuous_posture_20261113_v22.posture import ContinuousRobotPosturePrior,diverse_bank_seeds

class IdentityFK:
    def __call__(self,q):return np.broadcast_to(np.eye(4),(*np.asarray(q).shape[:-1],4,4)).copy()

class ContinuousPostureTest(unittest.TestCase):
    def test_near_duplicate_TRAIN_frames_do_not_fill_all_seeds(self):
        bank=np.array([[0.]*7,[.01]*7,[1.]*7,[-1.]*7])
        seeds=diverse_bank_seeds(IdentityFK(),bank,np.zeros(3),[0,0,0,1])
        self.assertEqual(len(seeds),3)

    def test_failed_local_seeds_trigger_internal_global_search(self):
        prior=ContinuousRobotPosturePrior(np.zeros((4,7)));fk=IdentityFK()
        limits=SimpleNamespace(lower=np.full(7,-2.),upper=np.full(7,2.))
        def solve(fk,seed,p,q,lo,hi):
            valid=bool(seed[0]>.5)
            return seed.copy(),dict(after_position_m=0. if valid else 1.,after_rotation_rad=0.)
        with patch('our_mani.executor_continuous_posture_20261113_v22.posture.refine',side_effect=solve):
            selected,receipt=prior.update(SimpleNamespace(fk=fk,safe=lambda q:[True]),limits,np.zeros(7),np.zeros(3),[0,0,0,1],0.)
        self.assertIsNotNone(selected);self.assertTrue(receipt['generic_fallback_used'])
        self.assertFalse(receipt['Native_setters_called'])

    def test_unchanged_goal_keeps_the_solution_without_reoptimizing(self):
        prior=ContinuousRobotPosturePrior(np.zeros((4,7)));prior.selected=np.ones(7)*.2
        prior.last_pose=(np.zeros(3),np.array([0,0,0,1]));prior.last_time=0.;prior.receipt=dict(available=True)
        with patch('our_mani.executor_continuous_posture_20261113_v22.posture.refine')as solve:
            selected,receipt=prior.update(SimpleNamespace(fk=IdentityFK(),safe=lambda q:[True]),None,np.zeros(7),np.zeros(3),[0,0,0,1],1.)
        solve.assert_not_called();np.testing.assert_array_equal(selected,np.ones(7)*.2)
        self.assertTrue(receipt['unchanged_goal_branch_retained'])

if __name__=='__main__':unittest.main()
