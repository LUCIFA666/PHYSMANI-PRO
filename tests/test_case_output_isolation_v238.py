from pathlib import Path
from types import SimpleNamespace
import unittest
from our_mani.multitask.case_output_isolation_v238 import require_development_output

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_original_cross_method_collision_rejected(self):
        with self.assertRaises(ValueError):
            require_development_output(ROOT,
                'artifacts/framework/multitask_r10/run/development_demo_only/bin/episode105/context_geometry',
                task='bin', raw='episode105', method='cv_geometry')

    def test_wrong_source_and_external_destination_rejected(self):
        for output in ('/tmp/development_demo_only/bin/episode105/cv_geometry',
                'artifacts/framework/multitask_r10/run/development_demo_only/bin/episode106/cv_geometry'):
            with self.assertRaises(ValueError):
                require_development_output(ROOT, output, task='bin', raw='episode105', method='cv_geometry')

    def test_actual_eight_prepared_argv_have_distinct_correct_destinations(self):
        from scripts.prepare_bin_physical_smoke_v240_r10 import command
        outputs = []
        for raw in ('episode105', 'episode106', 'episode119', 'episode123'):
            for method in ('cv_geometry', 'context_geometry'):
                argv = command('put_rubbish_in_moving_bin', raw, method)
                output = argv[argv.index('--out')+1]
                outputs.append(require_development_output(ROOT, output,
                    task='put_rubbish_in_moving_bin', raw=raw, method=method))
        self.assertEqual(len(set(outputs)), 8)


if __name__ == '__main__': unittest.main()
