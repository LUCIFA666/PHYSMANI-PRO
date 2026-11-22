"""Public synthetic windows and pre-label freeze guards; zero model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.context_physical_eval_20261103_v1.data import compile_public_parent
from our_mani.context_physical_eval_20261103_v1.evaluation import score, window_manifest


def public_rows(n=21):
    rows=[]
    for index in range(n):
        time=index*.05
        rows.append(dict(index=index, public_robot=dict(timestamp=time), close_requested=False, rgbd='synthetic.npz',
            result=dict(timestamp=time,mode='FREE',rotation_observable=False,
                Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
                forecaster_is_injected_test_double=False, object_observation=dict(object_id='target',timestamp=time,
                    pose=dict(position=[time,0.,.8],rotation=np.eye(3).tolist()),linear_velocity=[1.,0.,0.],
                    angular_velocity=[0.,0.,0.],confidence=1.,visible=True))))
    return rows


class PhysicalEvaluationContracts(unittest.TestCase):
    def test_current_history_and_future_are_separate_without_anchor_filter(self):
        windows,ledger=compile_public_parent('synthetic_TRAIN',public_rows())
        self.assertEqual(len(ledger),21)
        self.assertEqual(len(windows),17)
        self.assertEqual(sum(w.future_valid[1:].all() for w in windows),1)
        self.assertEqual(sum(w.future_valid[16] for w in windows),1)
        self.assertTrue(all(w.times[-1] <= w.future_positions[np.flatnonzero(w.future_valid)[0],0]
                            for w in windows if w.future_valid.any()))
        first=window_manifest(windows)
        self.assertEqual(first,window_manifest(windows))
        for window,record in zip(windows,first):
            for mode in ('isolated','burst'):
                keep=record['dropout'][mode]['retained_indices']
                self.assertEqual(keep[-1],len(window.times)-1)
                self.assertGreaterEqual(len(keep),5)

    def test_missing_and_close_boundary_keep_all_frames_and_stop_future_mask(self):
        rows=public_rows()
        rows[8]['result']['object_observation']=None
        rows[8]['result']['mode']='STALE'
        rows[16]['close_requested']=True
        rows[16]['result']['mode']='UNRESOLVED'
        windows,ledger=compile_public_parent('synthetic_TRAIN',rows)
        self.assertEqual(len(ledger),21)
        self.assertFalse(ledger[8]['eligible_free_current'])
        self.assertFalse(ledger[16]['eligible_free_current'])
        self.assertTrue(all(.4 not in w.times and .8 not in w.times for w in windows))
        before=next(w for w in windows if w.anchor==7)
        self.assertFalse(before.future_valid.any())
        self.assertTrue(any(not w.future_valid[16] for w in windows))

    def test_clock_and_nonpublic_provenance_reject(self):
        for change in ('gap','duplicate','robot_mismatch','Native'):
            rows=public_rows()
            if change=='gap':rows[5]['result']['timestamp']+=.01
            if change=='duplicate':rows[5]['result']['timestamp']=rows[4]['result']['timestamp']
            if change=='robot_mismatch':rows[5]['public_robot']['timestamp']+=.01
            if change=='Native':rows[5]['result']['Native_inputs_used']=True
            with self.assertRaises(ValueError):
                compile_public_parent('synthetic_TRAIN',rows)

    def test_incomplete_public_freeze_rejected_before_label_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'prediction_freeze.json').write_text(json.dumps(dict(complete=False)))
            with patch('our_mani.context_physical_eval_20261103_v1.evaluation.read_rows', side_effect=AssertionError('labels must not open')) as reader:
                with self.assertRaises(ValueError):
                    score(path,path/'must_not_exist.json',include_native_labels=True)
                self.assertEqual(reader.call_count,0)
            self.assertFalse((path/'must_not_exist.json').exists())


if __name__=='__main__':
    unittest.main()
