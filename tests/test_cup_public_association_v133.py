from pathlib import Path
from types import SimpleNamespace
import json,unittest
import numpy as np
from our_mani.multitask.mode_observer import PublicModeObserver

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self,distances=(.018,.139,.16),visible=(True,True,True)):
        report=json.loads((ROOT/'artifacts/framework/multitask_r10/cup_public_association_radius_TRAIN_v133/report.json').read_text())
        ids=('cup0','cup1','cup2');observer=PublicModeObserver(ids,ids,report['association_radius_m'])
        states=[SimpleNamespace(object_id=n,visible=v,pose=SimpleNamespace(position=np.array([d,0.,0.]))) for n,d,v in zip(ids,distances,visible)]
        robot=SimpleNamespace(contact=True,pose=SimpleNamespace(position=np.zeros(3)))
        return observer,states,robot

    def test_neighbor_cup_outside_fitted_radius_does_not_block_unique_contact(self):
        observer,states,robot=self.fixture();modes,invalid=observer.observe(states,robot)
        self.assertEqual(observer.held,'cup0');self.assertFalse(invalid);self.assertEqual(modes['cup1'],'external')

    def test_two_plausible_cups_still_invalidate_association(self):
        observer,states,robot=self.fixture((.018,.025,.16));_,invalid=observer.observe(states,robot)
        self.assertIsNone(observer.held);self.assertEqual(invalid,{'cup0','cup1','cup2'})

    def test_target_occlusion_does_not_authorize_an_unobserved_grasp(self):
        observer,states,robot=self.fixture(visible=(False,True,True));_,invalid=observer.observe(states,robot)
        self.assertIsNone(observer.held);self.assertTrue(invalid)

    def test_contact_loss_preserves_original_release_semantics(self):
        observer,states,robot=self.fixture();observer.observe(states,robot);robot.contact=False
        modes,invalid=observer.observe(states,robot)
        self.assertIsNone(observer.held);self.assertFalse(invalid);self.assertEqual(modes['cup0'],'released')


if __name__=='__main__':unittest.main()
