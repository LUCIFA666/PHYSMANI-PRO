"""Independent TRAIN-contract adversaries; synthetic files, no fit/Native."""
import hashlib
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from our_mani.scene_task_latent_20261103 import data
from our_mani.scene_task_latent_20261103.model import SceneTaskEncoder, TASKS


def inventory():
    rows=[]
    for family in TASKS:
        # Real numeric identities deliberately differ from list ordinals.
        for number in (0,7,19,101,777):
            for suffix in ('','_high_speed'):
                task=family+suffix
                rows.append(dict(task=task,raw_episode_id=f'episode{number}',
                    public_input_sha256=hashlib.sha256(f'raw:{task}:{number}'.encode()).hexdigest(),
                    cached_input_sha256=hashlib.sha256(f'cached:{task}:{number}'.encode()).hexdigest()))
        for suffix,start in (('',200),('_high_speed',300)):
            for number in range(start,start+5):
                task=family+suffix
                rows.append(dict(task=task,raw_episode_id=f'episode{number}',
                    public_input_sha256=hashlib.sha256(f'raw:{task}:{number}'.encode()).hexdigest(),
                    cached_input_sha256=hashlib.sha256(f'cached:{task}:{number}'.encode()).hexdigest()))
    return rows


def synthetic_hdf(root):
    parent=Path(root)/'episode17';parent.mkdir()
    path=parent/'data.h5'
    pixels=np.zeros((256,256,3),np.uint8)
    pixels[:128,128:]=255
    pixels[128:,:128]=[128,0,0]
    pixels[128:,128:]=[0,128,0]
    with h5py.File(path,'w') as source:
        source.attrs['task']=TASKS[0];source.attrs['episode']=17
        source.attrs['variation']=999
        for index,camera in enumerate(data.CAMERAS):
            rgb=np.full((2,256,256,3),255,np.uint8)
            rgb[0]=32*(index+1)
            depth=np.stack((pixels,np.full_like(pixels,13)))
            source[f'observations/{camera}/rgb']=rgb
            source[f'observations/{camera}/depth']=depth
            source[f'cam_params/{camera}/near']=.1
            source[f'cam_params/{camera}/far']=10.
            source[f'cam_params/{camera}/intrinsics']=np.eye(3)
            source[f'cam_params/{camera}/extrinsics']=np.eye(4)
        source['low_dim_obs/expert_trajectory']=np.full((2,1,8),np.nan)
        source['low_dim_obs/expert_trajectory_stage']=np.array([b'PRIVATE_CLASS_0',b'PRIVATE_CLASS_1'])
        source['low_dim_obs/joint_positions']=np.full((2,7),np.inf)
        source['wm_predictions/future_object_pose']=np.full((2,7),np.nan)
    return path


class SceneTaskLatentAdversarialTests(unittest.TestCase):
    def test_split_groups_real_episode_numbers_across_speed_and_not_unrelated_semantics(self):
        rows=inventory()
        assigned,summary=data.assign_splits(rows)
        self.assertEqual(summary['source_count'],40)
        self.assertEqual(summary['parent_count'],30)
        pairs={}
        for row in assigned:
            key=(row['semantic_task'],row['raw_episode_id'])
            pairs.setdefault(key,set()).add((row['parent_group'],row['split']))
            self.assertEqual(row['source_frame_index'],0)
        self.assertTrue(all(len(group)==1 for group in pairs.values()))
        pick=next(row for row in assigned if row['task']==TASKS[0] and row['raw_episode_id']=='episode777')
        insert=next(row for row in assigned if row['task']==TASKS[1] and row['raw_episode_id']=='episode777')
        self.assertNotEqual(pick['parent_group'],insert['parent_group'])
        self.assertTrue(pick['parent_group'].endswith(':777'))
        fit={row['parent_group'] for row in assigned if row['split']=='fit'}
        internal={row['parent_group'] for row in assigned if row['split']=='internal_val'}
        self.assertFalse(fit&internal)
        reversed_rows,reversed_summary=data.assign_splits(list(reversed(rows)))
        key=lambda row:(row['task'],row['raw_episode_id'])
        values=lambda source:{key(row):(row['parent_group'],row['split']) for row in source}
        self.assertEqual(values(assigned),values(reversed_rows))
        self.assertEqual(summary,reversed_summary)
        self.assertTrue(all('split' not in row for row in rows))

    def test_raw_and_cached_duplicates_form_transitive_parent_group_before_split(self):
        rows=inventory()
        find=lambda task,number:next(row for row in rows if row['task']==task and row['raw_episode_id']==f'episode{number}')
        a=find(TASKS[0],0);b=find(TASKS[0],7);c=find(TASKS[0]+'_high_speed',19)
        b['public_input_sha256']=a['public_input_sha256']
        c['cached_input_sha256']=b['cached_input_sha256']
        assigned,summary=data.assign_splits(rows)
        chosen=[row for row in assigned if row['semantic_task']==TASKS[0]
                and row['raw_episode_id'] in ('episode0','episode7','episode19')]
        self.assertEqual(len(chosen),6)
        self.assertEqual(len({row['parent_group'] for row in chosen}),1)
        self.assertEqual(len({row['split'] for row in chosen}),1)
        self.assertEqual(summary['parent_count'],28)

    def test_identical_public_or_cached_input_with_conflicting_label_is_not_silently_dropped(self):
        for field in ('public_input_sha256','cached_input_sha256'):
            with self.subTest(field=field):
                rows=inventory()
                a=next(row for row in rows if row['task']==TASKS[0])
                b=next(row for row in rows if row['task']==TASKS[1])
                b[field]=a[field]
                with self.assertRaisesRegex(ValueError,'contradictory semantic labels'):
                    data.assign_splits(rows)

    def test_duplicate_source_identity_and_malformed_episode_identity_rejected(self):
        rows=inventory()
        with self.assertRaisesRegex(ValueError,'duplicate task/source identity'):
            data.assign_splits(rows+[dict(rows[0])])
        rows[0]['raw_episode_id']='sorted_index_0'
        with self.assertRaisesRegex(ValueError,'real raw episodeN'):
            data.assign_splits(rows)

    def test_hdf_reads_only_first_actual_sensor_frames_and_no_expert_robot_or_future(self):
        with TemporaryDirectory() as directory:
            path=synthetic_hdf(directory)
            original_file=h5py.File;accesses=[]
            allowed={f'observations/{cam}/{kind}' for cam in data.CAMERAS for kind in ('rgb','depth')}
            allowed|={f'cam_params/{cam}/{kind}' for cam in data.CAMERAS for kind in ('near','far','intrinsics','extrinsics')}
            class GuardedDataset:
                def __init__(self,stream,key):self.stream=stream;self.key=key
                @property
                def shape(self):return self.stream.shape
                @property
                def dtype(self):return self.stream.dtype
                def __getitem__(self,index):
                    expected=0 if self.key.startswith('observations/') else ()
                    if index!=expected:raise AssertionError('future/all-frame read: '+self.key)
                    accesses.append((self.key,index));return self.stream[index]
            class GuardedFile:
                def __init__(self,*args,**kwargs):self.source=original_file(*args,**kwargs)
                @property
                def attrs(self):return self.source.attrs
                def __enter__(self):return self
                def __exit__(self,*args):self.source.close()
                def __getitem__(self,key):
                    if key not in allowed:raise AssertionError('nonpublic dataset opened: '+key)
                    return GuardedDataset(self.source[key],key)
            with patch.object(data.h5py,'File',GuardedFile):
                result=data.read_first_public_rgbd(path,TASKS[0])
            self.assertEqual(len(accesses),24)
            self.assertEqual({key for key,index in accesses},allowed)
            self.assertFalse(result['audit']['future_frames_read'])
            self.assertFalse(result['audit']['expert_or_robot_datasets_read'])
            self.assertFalse(result['audit']['camera_parameters_are_forward_features'])
            self.assertEqual(set(result),{'rgb','depth','audit'})
            self.assertEqual(result['rgb'].shape,(4,64,64,3))
            for index in range(4):np.testing.assert_array_equal(result['rgb'][index],np.full((64,64,3),32*(index+1)))

    def test_RGB24_decode_precedes_resize_and_camera_matrix_cannot_be_direct_feature(self):
        with TemporaryDirectory() as directory:
            path=synthetic_hdf(directory)
            a=data.read_first_public_rgbd(path,TASKS[0])
            expected=[.1,10.,.1+((128<<16)/(2**24-1))*9.9,.1+((128<<8)/(2**24-1))*9.9]
            np.testing.assert_allclose(a['depth'][0,[0,0,63,63],[0,63,0,63]],expected,rtol=1e-6)
            with h5py.File(path,'r+') as source:
                source.attrs['variation']=-1000
                for camera in data.CAMERAS:
                    source[f'cam_params/{camera}/intrinsics'][...]=np.diag([90.,91.,1.])
                    changed=np.eye(4);changed[:3,3]=[100.,200.,300.]
                    source[f'cam_params/{camera}/extrinsics'][...]=changed
                    source[f'observations/{camera}/rgb'][1]=0
                    source[f'observations/{camera}/depth'][1]=255
            b=data.read_first_public_rgbd(path,TASKS[0])
            np.testing.assert_array_equal(a['rgb'],b['rgb'])
            np.testing.assert_array_equal(a['depth'],b['depth'])
            self.assertEqual(a['audit']['public_input_sha256'],b['audit']['public_input_sha256'])
            self.assertEqual(a['audit']['cached_input_sha256'],b['audit']['cached_input_sha256'])
            self.assertNotEqual(a['audit']['calibration_sha256'],b['audit']['calibration_sha256'])

    def test_normalization_excludes_metadata_expert_and_robot_and_rejects_nonfinite_sensor(self):
        class Poison:
            def __array__(self,*args,**kwargs):raise AssertionError('nonpublic feature read')
        arrays=dict(rgb=np.full((2,4,64,64,3),128,np.uint8),depth=np.ones((2,4,64,64),np.float32))
        clean=data.normalized_rgbd(arrays,np.array([1]))
        enriched=dict(arrays,robot=Poison(),task_labels=Poison(),expert_actions=Poison(),camera_intrinsics=Poison())
        np.testing.assert_array_equal(clean,data.normalized_rgbd(enriched,np.array([1])))
        self.assertEqual(clean.shape,(1,4,4,64,64))
        np.testing.assert_allclose(clean[0,:,3],np.log(2.),rtol=1e-6)
        arrays['depth'][1,0,0,0]=np.inf
        with self.assertRaisesRegex(ValueError,'finite sensor-only'):
            data.normalized_rgbd(arrays,np.array([1]))

    def test_scene_model_simnorm_two_simplexes_camera_set_invariance_and_no_metadata_API(self):
        previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            torch.manual_seed(47103)
            model=SceneTaskEncoder().eval()
            pixels=torch.randn(2,4,4,64,64)
            with torch.no_grad():
                out=model(pixels);permuted=model(pixels[:,[2,0,3,1]])
                for views in (1,2,3,4):
                    result=model(pixels[:,:views])
                    self.assertEqual(result['task_e'].shape,(2,16))
                    self.assertEqual(result['task_logits'].shape,(2,2))
                    self.assertEqual(result['task_probabilities'].shape,(2,2))
                    torch.testing.assert_close(result['task_e'].reshape(2,2,8).sum(-1),torch.ones(2,2))
                    torch.testing.assert_close(result['task_probabilities'].sum(-1),torch.ones(2))
                    self.assertTrue(bool((result['task_e']>=0).all()))
                for key in out:torch.testing.assert_close(out[key],permuted[key],rtol=1e-6,atol=1e-7)
            with self.assertRaises(TypeError):model(pixels,task_id=torch.ones(2))
            with self.assertRaises(TypeError):model(pixels,robot=torch.zeros(2,15))
            for bad in [torch.full_like(pixels,float('nan')),torch.zeros(1,5,4,64,64),pixels.to(torch.int32)]:
                with self.assertRaises(ValueError):model(bad)
            self.assertFalse(model.spec()['frozen_uniform_e_P4_direct_substitution_allowed'])
        finally:torch.set_num_threads(previous_threads)

    def test_fit_weights_give_equal_family_and_parent_mass_without_dev_statistics(self):
        from our_mani.scene_task_latent_20261103.training import parent_weights
        rows,_=data.assign_splits(inventory())
        indices=np.array([i for i,row in enumerate(rows) if row['split']=='fit'])
        weights=parent_weights(rows,indices)
        self.assertAlmostEqual(float(weights.mean()),1.)
        masses={}
        for index,weight in zip(indices,weights):
            row=rows[index]
            masses[row['parent_group']]=masses.get(row['parent_group'],0.)+weight
        for label in (0,1):
            chosen=[i for i,index in enumerate(indices) if rows[index]['semantic_label']==label]
            self.assertAlmostEqual(float(weights[chosen].sum()),len(indices)/2)
            groups={rows[index]['parent_group'] for index in indices if rows[index]['semantic_label']==label}
            np.testing.assert_allclose([masses[group] for group in groups],len(indices)/(2*len(groups)))
        contaminated=[dict(row) for row in rows]
        for row in contaminated:
            if row['split']=='internal_val':
                row.update(parent_group='DEV_METADATA_POISON',semantic_label=99)
        np.testing.assert_array_equal(parent_weights(contaminated,indices),weights)

    def test_training_CLI_defaults_to_check_without_optimizer_or_fit(self):
        path=Path(__file__).resolve().parents[1]/'scripts/train_scene_task_latent_20261103.py'
        spec=importlib.util.spec_from_file_location('independent_scene_task_cli_check',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(module,'fit',side_effect=AssertionError('implicit fit')), \
             patch.object(module,'check_prepared',return_value={'parameter_updates':0}) as check, \
             patch('torch.optim.AdamW',side_effect=AssertionError('optimizer during default check')), \
             patch('sys.argv',[str(path),'--dataset','/tmp/CPU_TEST_STUB_NO_DATA_READ']), \
             patch('sys.stdout',new_callable=io.StringIO):
            module.main()
        self.assertEqual(check.call_count,1)
        self.assertEqual(str(check.call_args.args[0]),'/tmp/CPU_TEST_STUB_NO_DATA_READ')
        self.assertEqual(check.call_args.kwargs,{'cpu_threads':2})


if __name__=='__main__':unittest.main()
