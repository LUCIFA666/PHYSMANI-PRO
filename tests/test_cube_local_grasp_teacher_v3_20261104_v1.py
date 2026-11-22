"""Private-label acceptance delta only; no data/model/Native execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from our_mani.cube_candidate_teacher_20261104_v3.labels import actual_local_grasp_contacts
from our_mani.cube_candidate_teacher_20261104_v3.data import compile_contact_episode
from our_mani.cube_candidate_matched_20261104_v2.data import actual_successful_contacts
spec=importlib.util.spec_from_file_location('v2fixture',Path(__file__).with_name('test_cube_candidate_matched_supervision_20261104_v2.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class TeacherV3(unittest.TestCase):
    def test_without_goal_is_new_local_positive_not_fulltask(self):
        public,initial,ticks=f.fixture()
        for r in ticks:r['native_success']=False
        self.assertEqual(actual_successful_contacts(initial,ticks,episode_id=initial['episode_id']),[])
        labels=actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id'])
        self.assertEqual(len(labels),1);self.assertEqual(labels[0]['acceptance_class'],'lift_hold_without_goal');self.assertFalse(labels[0]['full_task_success_claimed'])
        arrays,report=compile_contact_episode(public,initial,ticks,parent_id=f.PARENT,episode_id=initial['episode_id'],expected_forecast_sha256=f.SHA)
        self.assertEqual(report['acceptance_counts'],{'lift_hold_without_goal':1});self.assertGreater(report['pose_only_far_anchors'],0)

    def test_with_goal_class_and_original_thresholds(self):
        _,initial,ticks=f.fixture();labels=actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id'])
        self.assertEqual(labels[0]['acceptance_class'],'lift_hold_with_goal')
        for r in ticks:
            if r['held']:r['object_position'][2]=.039
        self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])

    def test_no_contact_and_brief_or_drifting_hold_are_not_positive(self):
        _,initial,ticks=f.fixture(False);self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])
        _,initial,ticks=f.fixture();ticks=ticks[:34]
        self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])
        _,initial,ticks=f.fixture()
        for r in ticks:
            if r['held']:r['tcp_position'][0]=r['time']
        self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])

if __name__=='__main__':unittest.main()
