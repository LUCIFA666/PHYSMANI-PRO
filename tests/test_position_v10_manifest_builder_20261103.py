"""V10 cohort closure and ten-shot metadata contracts; no image/model access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v10_manifest_builder', ROOT/'scripts/prepare_cube_position_calibration8_manifest_20261103_v10.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def save(path, value):
    path.write_text(json.dumps(value))


class V10ManifestContracts(unittest.TestCase):
    def test_band_mapping_matches_dataset_contract(self):
        from our_mani.cube_position_measurement_20261103_v10.data import POSITION_BANDS
        self.assertEqual(builder.POSITION_BANDS, POSITION_BANDS)
        self.assertEqual(tuple(builder.SHOT_IDS), tuple(f'shot{i:02d}' for i in range(10)))
        self.assertEqual(len(builder.PARTITION_IDS), 8)

    def test_all_eight_clean_owners_required_without_reading_images(self):
        with tempfile.TemporaryDirectory(prefix='v10_manifest_') as temp:
            root = Path(temp)
            parents = [{'parent_id': p} for p in builder.PARTITION_IDS]
            for parent in parents:
                ident = parent['parent_id']; case = root/ident; case.mkdir()
                save(root/(ident+'.owner.actual_exit.json'), dict(actual_process_returncode=0,
                    actual_wait_completed=True,source_unchanged=True,stop_reason=None,parent_id=ident,
                    output=str(case.resolve()),started_utc='2026-11-03T00:00:00+00:00',
                    finished_utc='2026-11-03T00:01:00+00:00'))
                save(case/'result.json',dict(complete=True,native_closed=True,source_unchanged=True,
                    parent_id=ident,controlled_physics_ticks=0))
            with patch.object(builder,'PositionCalibration8V10Dataset',side_effect=AssertionError('No dataset before gate')):
                self.assertEqual(len(builder._closed_cohort(root,parents)[1]),8)
                last = root/(parents[-1]['parent_id']+'.owner.actual_exit.json')
                owner = json.loads(last.read_text()); owner['actual_process_returncode'] = False
                save(last,owner)
                with self.assertRaises(builder.ReadinessError): builder._closed_cohort(root,parents)
                last.unlink()
                with self.assertRaises(builder.ReadinessError): builder._closed_cohort(root,parents)

    def test_ten_ordered_frames_export_band_and_reject_duplicates_or_wrong_band(self):
        with tempfile.TemporaryDirectory(prefix='v10_manifest_') as temp:
            root = Path(temp); case = root/'parent'; case.mkdir()
            for name in ('inputs','labels','initialization','source_snapshot'): (case/name).mkdir()
            for name in ('initialization/public_rgbd.npz','public_initial_sphere.json',
                         'result.json','owner.actual_exit.json','plan.json','partition.json',
                         'run_inputs.json','source_snapshot/manifest.json'):
                (case/name).write_text('{}')
            shots=[]; frames=[]
            for shot in builder.SHOT_IDS:
                band = builder.POSITION_BANDS[shot]
                condition = 'sphere_only' if shot in ('shot00','shot09') else 'cube_far' if shot=='shot01' else 'near_goal'
                planned=dict(shot_id=shot,condition=condition,condition_detail='fixed',position_band=band)
                frame=dict(planned,scene_id='parent/'+shot,independent_scene=True,physics_ticks=0)
                for key,folder in (('input','inputs'),('labels','labels'),('annotation_metadata','labels')):
                    file=case/folder/(shot+'_'+key+'.json');file.write_text('{}')
                    frame[key]=builder._pointer(file)
                shots.append(planned);frames.append(frame)
            parent=dict(parent_id='parent',seed=261103656,role='calib',shots=shots)
            result=dict(shots=frames)
            closed=[(parent,case,case/'owner.actual_exit.json',{},case/'result.json',result)]
            args=(root,closed,case/'plan.json',{},case/'partition.json')
            with patch.object(builder,'_owner_launch',return_value={}):
                manifest,_=builder._candidate(*args)
                self.assertEqual([s['position_band'] for s in manifest['parents'][0]['shots']],
                    [builder.POSITION_BANDS[s] for s in builder.SHOT_IDS])
                frames[-1]['shot_id']='shot08'
                with self.assertRaises(builder.ReadinessError):builder._candidate(*args)
                frames[-1]['shot_id']='shot09';frames[-1]['position_band']='annulus_10_12'
                with self.assertRaises(builder.ReadinessError):builder._candidate(*args)


if __name__ == '__main__': unittest.main()
