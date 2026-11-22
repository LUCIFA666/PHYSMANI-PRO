import unittest
import tempfile
from pathlib import Path
from our_mani.benchmark_protocol import file_sha256,validate_manifest,verify_lock,wilson


class BenchmarkProtocolTests(unittest.TestCase):
    def test_manifest_requires_all_variations_and_no_duplicate_ids(self):
        rows=[{'task':task,'variation':v,'official_episode_index':i,'raw_episode_id':f'episode{20*v+i}'}
            for task in ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
            for v in (0,1) for i in range(20)]
        validate_manifest(rows)
        rows[-1]=rows[-2].copy()
        with self.assertRaises(ValueError):validate_manifest(rows)

    def test_wilson_extreme_small_samples(self):
        self.assertIsNone(wilson(0,0))
        lo,hi=wilson(0,40)
        self.assertAlmostEqual(lo,0.)
        self.assertGreater(hi,.08)
        self.assertLess(hi,.1)
        lo,hi=wilson(40,40)
        self.assertGreater(lo,.9)
        self.assertAlmostEqual(hi,1.)

    def test_resume_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pt';path.write_bytes(b'frozen')
            lock={'files':{'weights.pt':file_sha256(path)}}
            verify_lock(directory,lock)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                verify_lock(directory,lock)


if __name__=='__main__':unittest.main()
