from pathlib import Path
import gzip
import json
import tempfile
import unittest
import numpy as np
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.lossless_case_archive import run_archived_case,uncompressed_sha
from our_mani.multitask.three_method_release16 import measured_motion


class Tests(unittest.TestCase):
    def test_new_native_failure_trace_round_trips_and_motion_evidence_stays_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'fresh';original={}
            def runner(prepared,backend=None):
                record=dict(complete=True,cleanup_completed=True,infrastructure_error=None,
                    official_task_success=False,identity=dict(sha256='a'*64),
                    actor_metrics=dict(continuous_reference_feedforward=dict(
                        observed_motion_qualification_failed=False,observed_events=[dict(time=.05,violations=[])]),
                        detailed_trial_history=[dict(proof='same full collision and trajectory proof'*100) for _ in range(1000)]))
                atomic_json(out/'case.json',record);original.update(record=record,sha=file_sha(out/'case.json'))
                return record
            compact=run_archived_case(dict(out=out),runner=runner);pointer=compact['full_trace_archive']
            self.assertEqual(uncompressed_sha(pointer['path']),original['sha'])
            with gzip.open(pointer['path'],'rt') as stream:self.assertEqual(json.load(stream),original['record'])
            self.assertFalse(compact['official_task_success']);self.assertTrue(measured_motion(compact))
            self.assertTrue(compact['actor_metrics']['detailed_trial_history']['archived_list'])
            self.assertLess((out/'case.json').stat().st_size,pointer['uncompressed_bytes']/10)
            self.assertFalse((out/'full_evaluator_case.uncompressed.tmp').exists())
            with self.assertRaises(FileExistsError):run_archived_case(dict(out=out),runner=runner)

    def test_no_packaging_before_clean_native_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_archived_case(dict(out=Path(directory)),runner=lambda *a,**kw:dict(complete=False,cleanup_completed=False))

    def test_original_evaluator_in_memory_numpy_values_are_normalised_without_touching_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'fresh'
            def runner(*a,**kw):
                value=dict(complete=True,cleanup_completed=True,joints=np.arange(7.),
                    actor_metrics=dict(continuous_reference_feedforward=dict(observed_events=[dict(velocity=np.zeros(7))])))
                from our_mani.multitask.lossless_case_archive import plain
                atomic_json(out/'case.json',plain(value));return value
            result=run_archived_case(dict(out=out),runner=runner)
            self.assertEqual(result['joints'],list(np.arange(7.)))
            self.assertEqual(result['actor_metrics']['continuous_reference_feedforward']['observed_events'][0]['velocity'],[0.]*7)


if __name__=='__main__':unittest.main()
