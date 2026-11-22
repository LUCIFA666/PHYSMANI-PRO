from pathlib import Path
import tempfile
import unittest
from our_mani.multitask.evaluation import atomic_json
from our_mani.multitask.lossless_case_archive import run_archived_case
from our_mani.multitask.archived_case_reader import read_case


class Tests(unittest.TestCase):
    def test_full_raw_trace_is_restored_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'fresh'
            def runner(*a,**kw):
                record=dict(complete=True,cleanup_completed=True,official_task_success=False,
                    identity={'source':'fixed original TRAIN'},steps=[dict(time=i) for i in range(20)])
                atomic_json(out/'case.json',record);return record
            summary=run_archived_case(dict(out=out),runner=runner)
            self.assertEqual(len(read_case(out/'case.json',full=True)['steps']),20)
            self.assertTrue(read_case(out/'case.json')['steps']['archived_list'])
            archive=Path(summary['full_trace_archive']['path'])
            with archive.open('ab') as stream:stream.write(b'corrupted')
            with self.assertRaises(ValueError):read_case(out/'case.json',full=True)


if __name__=='__main__':unittest.main()
