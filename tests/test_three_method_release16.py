from copy import deepcopy
import unittest
from our_mani.multitask.evaluation import digest
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,VAL_COUNTS
from our_mani.multitask.three_method_release16 import policy_fingerprint,validation_release,require_test_release


class Tests(unittest.TestCase):
    def fixture(self):
        pointer=dict(path='CPU_fixture',sha256='a'*64)
        routes=dict(shared_context_checkpoint_sha256='b'*64,routes=[dict(task=t,method=m,
            GPU_required=m=='physmani',configuration=pointer,frozen_inputs=pointer,worker=pointer,consumer=pointer)
            for t in TASKS for m in METHODS])
        policy=policy_fingerprint(routes);manifest=dict(split='validation',episodes=[dict(task=t,
            raw_episode_id='source'+str(i),path='CPU_fixture',split='validation') for t in TASKS for i in range(VAL_COUNTS[t])])
        cases=[dict(task=r['task'],source_raw_episode_id=r['raw_episode_id'],method=m,split='validation',
            complete=True,cleanup_completed=True,infrastructure_error=None,official_task_success=False,
            TEST_used_for_fitting=False,all_stage_counted_joint_motion=dict(all_stage_actual_window_checks_passed=m!='physmani'),
            full16_frozen_execution=dict(policy_sha256=policy,source_row_sha256=digest(r)))
            for r in manifest['episodes'] for m in METHODS]
        return routes,manifest,cases

    def test_complete_three_method_grid_can_release_with_native_failures_and_unchanged_baseline_motion_violations(self):
        routes,manifest,cases=self.fixture();release=validation_release(cases,manifest,routes)
        self.assertEqual(release['method_cases'],945);self.assertTrue(require_test_release(release,routes))
        self.assertEqual(sum(release['native_successes_by_method'].values()),0)
        self.assertEqual(release['measured_motion_violations_by_method']['physmani'],315)

    def test_missing_duplicate_old_five_method_and_changed_policy_are_rejected(self):
        routes,manifest,cases=self.fixture()
        for bad in (cases[:-1],cases[:-1]+[cases[0]],cases+[dict(cases[0],method='context_value')]):
            with self.assertRaises(ValueError):validation_release(bad,manifest,routes)
        release=validation_release(cases,manifest,routes);changed=deepcopy(routes)
        changed['routes'][0]['worker']['sha256']='c'*64
        with self.assertRaises(ValueError):require_test_release(release,changed)

    def test_infrastructure_demo_missing_motion_and_ours_motion_violation_do_not_unlock_TEST(self):
        routes,manifest,cases=self.fixture()
        for patch in (dict(infrastructure_error='native crash'),dict(demo_only=True),dict(all_stage_counted_joint_motion={})):
            altered=deepcopy(cases);altered[0].update(patch)
            with self.assertRaises(ValueError):validation_release(altered,manifest,routes)
        cases[0]['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed']=False
        release=validation_release(cases,manifest,routes);self.assertFalse(release['formal_test_permitted'])
        with self.assertRaises(ValueError):require_test_release(release,routes)


if __name__=='__main__':unittest.main()
