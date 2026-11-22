"""Compiler projection/role contracts; no Native, model or captured payloads."""
import importlib.util,json,tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('motion_compiler',ROOT/'scripts/compile_cube_trueclock_motion_dataset_20261104_v1.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def records():
    parent={'parent_id':'test','episode_id':'test_motionpass_v2','role':'fit'};public=[];private=[]
    for i in range(24):
        t=i*.05
        public.append(dict(schema='Cube_trueclock_motion_public_frame_v1',parent_id='test',episode_id=parent['episode_id'],physical_tick=i,index=i,timestamp=t,
            physics_timestamp=4.+t,physics_time_origin=4.,task_timestamp=2.+t,task_time_origin=2.,target=dict(available=i!=7,position_world=[.01*t,0.,.8]if i!=7 else None,confidence=1. if i!=7 else None),Native_labels_in_input=False))
        private.append(dict(parent_id='test',episode_id=parent['episode_id'],role='fit',reset_boundary_id=parent['episode_id'],physics_tick=i,
            actual_relative_physics_timestamp=t,physics_time_absolute=4.+t,physics_time_origin=4.,task_time_absolute=2.+t,task_time_origin=2.,
            held=False,requested_opening=1.,close_requested=False,target_position_world=[99.,88.,77.]))
    return parent,public,private


class CompilerContracts(unittest.TestCase):
    def test_public_unknown_not_filled_and_arch_windows_dtype(self):
        parent,p,l=records();public,labels=c.project_episode(p,l,parent)
        self.assertIsNone(public[7]['position_world']);self.assertEqual(public[1]['position_world'],[.0005,0.,.8])
        from our_mani.cube_motion_residual_20261104_v1.data import compile_episode
        windows,report=compile_episode(public,labels,parent_id='test')
        self.assertEqual(report['rows'],24);self.assertGreater(len(windows),0)
        self.assertEqual(windows[0]['inputs']['positions'].shape,(20,3));self.assertEqual(windows[0]['targets']['target_valid'].dtype,np.dtype(bool))
        self.assertTrue(np.all(windows[0]['inputs']['positions'][:windows[0]['inputs']['lengths'],0]<1.))
        self.assertTrue(np.isnan(windows[0]['targets']['target_positions'][0]).all())

    def test_actual_time_mismatch_and_cross_pass_rejected(self):
        parent,p,l=records();l[2]['actual_relative_physics_timestamp']+=.001
        with self.assertRaises(ValueError):c.project_episode(p,l,parent)
        parent,p,l=records();l[2]['episode_id']='other'
        with self.assertRaises(ValueError):c.project_episode(p,l,parent)

    def test_no_close_crossing_and_heldout_before_payload_gate(self):
        parent,p,l=records();l[8]['requested_opening']=0.;l[8]['close_requested']=True
        public,labels=c.project_episode(p,l,parent)
        self.assertFalse(labels[8]['free_motion']);self.assertNotEqual(labels[7]['segment_id'],labels[9]['segment_id'])
        with patch.object(c,'metadata',return_value=({},[])),patch.object(c,'capture_gate',side_effect=AssertionError('must not open any case')):
            with self.assertRaises(ValueError):c.compile_dataset('missing','missing','heldout','uncreated')

    def test_recorded_old_predictions_exact_anchor_not_new_forward(self):
        parent,p,l=records();t=p[4]['timestamp']
        f=dict(times=(t+np.arange(17)*.05).tolist(),positions=np.zeros((17,1,3)).tolist(),metadata={'checkpoint_sha256':c.OLD_GRU_SHA})
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'outputs.jsonl';path.write_text(json.dumps(dict(result=dict(timestamp=t,mode='FREE',free_forecast_available=True,forecast=f)))+'\n')
            rows,report=c.old_forecasts(path,p);self.assertEqual(list(rows),[t]);self.assertEqual(rows[t][0].shape,(17,3));self.assertFalse(report['input_to_new_model'])


if __name__=='__main__':unittest.main()
