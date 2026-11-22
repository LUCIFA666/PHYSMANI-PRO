"""Parent identities and fixed demo sampling are independent of outcomes."""
import unittest

from scripts.run_multitask_fresh_queue_r10 import build_jobs


class QueuePlanTests(unittest.TestCase):
    def rows(self):
        return [dict(task=task, raw_episode_id=raw, path='/readonly/' + task + '/' + raw)
                for task in ('fixture_normal', 'fixture_high') for raw in ('episode2', 'episode0')]

    def test_each_method_task_records_only_first_manifest_source(self):
        methods = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value', 'physmani')
        jobs = build_jobs(self.rows(), 'test', methods, 'a' * 64, 'b' * 64, record_demos=True)
        self.assertEqual(len(jobs), 20)
        selected = [job for job in jobs if job['record_demo']]
        self.assertEqual(len(selected), 10)
        self.assertEqual({job['source_raw_episode_id'] for job in selected}, {'episode2'})
        self.assertEqual(len({job['job_id'] for job in jobs}), len(jobs))

    def test_replica_identity_keeps_parent_and_exploration_is_train_only(self):
        jobs = build_jobs(self.rows()[:1], 'train', ('cv_geometry', 'context_geometry'),
                          'a' * 64, 'b' * 64, replicas=2, explore_first=True)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job['identity']['parent_raw_episode_id'] for job in jobs}, {'episode2'})
        self.assertEqual(len({job['job_id'] for job in jobs}), 4)
        self.assertTrue(all(job['exploration_seed'] is not None for job in jobs))
        with self.assertRaises(ValueError):
            build_jobs(self.rows(), 'test', ('cv_geometry',), 'a' * 64, 'b' * 64, replicas=2)
        with self.assertRaises(ValueError):
            build_jobs(self.rows(), 'validation', ('cv_geometry',), 'a' * 64, 'b' * 64, explore_first=True)


if __name__ == '__main__':
    unittest.main()
