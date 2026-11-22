"""PhysMani VAL excludes an unused TEST checker; TEST keeps its source gate."""
import unittest

from scripts import evaluate_multitask_case_r10 as child


class BaselineSourceScopeTests(unittest.TestCase):
    def names(self, method, split=None):
        return {path.name for path in child.required_local_sources(method, split=split)}

    def test_baseline_development_keeps_shared_physics_and_omits_unused_TEST_checker(self):
        for split in ('train', 'validation'):
            names = self.names('physmani', split)
            self.assertNotIn('release_preflight.py', names)
            self.assertTrue({'simulation.py', 'maps.py', 'actuation.py', 'reset_parity.py',
                'native_tasks.py', 'physmani_executor_v7.py', 'physmani.py'} <= names)

    def test_baseline_TEST_still_requires_the_full_coverage_checker(self):
        self.assertIn('release_preflight.py', self.names('physmani', 'test'))

    def test_unspecified_split_remains_conservative(self):
        self.assertIn('release_preflight.py', self.names('physmani'))

    def test_all_ours_methods_keep_the_coverage_checker_in_all_splits(self):
        for method in ('cv_geometry', 'cv_value', 'context_geometry', 'context_value'):
            for split in ('train', 'validation', 'test'):
                self.assertIn('release_preflight.py', self.names(method, split))


if __name__ == '__main__':
    unittest.main()
