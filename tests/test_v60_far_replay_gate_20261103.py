import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts import score_v60_far_consistency_20261103 as scoring


class ReplayFreezeGateTests(unittest.TestCase):
    def fixture(self,root):
        pred=root/'predictions';pred.mkdir();cases={};rows={};public={}
        source={str(scoring.ROOT/n):scoring.sha(scoring.ROOT/n) for n in scoring.SOURCES}
        settings=dict(cases=list(scoring.CASES),NN_weights_loaded=False,minimum_red=100,
            complete_geometry=False,half_side_m=.025,point_axis_development_allowance_m=.006,
            near_scope_m=.08,tolerance_or_threshold_search=False,source_sha256=source)
        (pred/'settings.json').write_text(json.dumps(settings));outputs={'settings.json':scoring.sha(pred/'settings.json')}
        for name in scoring.CASES:
            case=root/name;case.mkdir();frame=case/'0000.npz';np.savez(frame,timestamp=np.array(0.))
            cal=case/'camera_calibration.json';cal.write_text('{}')
            cases[name]=dict(frames=[frame],camera_calibration=cal)
            public[str(cal)]=scoring.sha(cal);public[str(frame)]=scoring.sha(frame)
            row=dict(case=name,index=0,timestamp=0.,input_path=str(frame),input_sha256=scoring.sha(frame),
                legacy=dict(raw_available=False),conditional=dict(raw_available=False))
            f=pred/(name+'.predictions.jsonl');f.write_text(json.dumps(row)+'\n');outputs[f.name]=scoring.sha(f)
            f=pred/(name+'.current_components.npz');np.savez(f,legacy_points=np.empty((0,3)),
                conditional_points=np.empty((0,3)),legacy_offsets=np.zeros(2,np.int64),conditional_offsets=np.zeros(2,np.int64));outputs[f.name]=scoring.sha(f)
            rows[name]=[dict(index=0,timestamp=0.,input_sha256=scoring.sha(frame))]
        freeze=dict(complete=True,cases=list(scoring.CASES),case_frames={n:1 for n in scoring.CASES},
            frame_rosters=rows,prediction_and_config_files_sha256=outputs,source_sha256=source,
            public_input_sources_sha256=public,GT_labels_opened_during_prediction=False)
        (pred/'public_prediction_freeze.json').write_text(json.dumps(freeze))
        return pred,cases,freeze

    def test_valid_whole_public_fixture_needs_no_labels(self):
        with tempfile.TemporaryDirectory() as t:
            pred,cases,_=self.fixture(Path(t))
            with patch.object(scoring,'roster',return_value=cases):
                _,_,rows=scoring.validate_freeze(pred)
            self.assertEqual(len(rows),4)

    def test_missing_case_refuses_before_posthoc_directory(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);pred,cases,f=self.fixture(root);f['case_frames'].pop(scoring.CASES[-1])
            (pred/'public_prediction_freeze.json').write_text(json.dumps(f))
            with patch.object(scoring,'roster',return_value=cases),self.assertRaises(ValueError):
                scoring.score(pred,root/'posthoc')
            self.assertFalse((root/'posthoc').exists())

    def test_missing_last_calibration_hash_refuses_before_GT(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);pred,cases,f=self.fixture(root)
            f['public_input_sources_sha256'].pop(str(cases[scoring.CASES[-1]]['camera_calibration']))
            (pred/'public_prediction_freeze.json').write_text(json.dumps(f))
            with patch.object(scoring,'roster',return_value=cases),self.assertRaises(ValueError):
                scoring.score(pred,root/'posthoc')
            self.assertFalse((root/'posthoc').exists())

    def test_last_case_bad_frame_even_with_matching_file_SHA_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);pred,cases,f=self.fixture(root);name=scoring.CASES[-1]
            p=pred/(name+'.predictions.jsonl');row=json.loads(p.read_text());row['index']=1;p.write_text(json.dumps(row)+'\n')
            f['prediction_and_config_files_sha256'][p.name]=scoring.sha(p)
            (pred/'public_prediction_freeze.json').write_text(json.dumps(f))
            with patch.object(scoring,'roster',return_value=cases),self.assertRaises(ValueError):
                scoring.score(pred,root/'posthoc')
            self.assertFalse((root/'posthoc').exists())


if __name__=='__main__':unittest.main()
