"""CPU contracts only: no fitting/optimizerstep, Native or real label reads."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,os
import unittest
import numpy as np
import torch
from our_mani.near_goal_segmentation_20261103 import data,model,evaluation,training

torch.set_num_threads(2)


def inputs():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.full((4,256,256),1.025,np.float32),
        K=np.tile(np.array([[-200.,0.,128.],[0.,-200.,128.],[0.,0.,1.]])[None],(4,1,1)),
        T=np.tile(np.eye(4)[None],(4,1,1)),public_sphere_center=np.array([0.,0.,1.]),
        camera_names=np.array(data.CAMERAS),timestamp=np.array(0.))


def sample(condition='overlap',*,far=False,invisible=False,gt_offset=0.):
    public=inputs();crops=data.make_near_goal_crops(public);semantic=np.zeros((4,64,64),np.uint8)
    if not invisible:semantic[:,28:37,(44 if far else 28):(53 if far else 37)]=1
    semantic[:,2:8,2:8]=2
    centre=np.array([-.082 if far else 0.,0.,1.+gt_offset])
    return dict(features=crops['features'],semantic=semantic,inputs=public,crops=crops,
        cube_center_world=centre,cube_present=not invisible,parent_id='synthetic',scene_id=condition,
        role='calib',condition=condition)


def probabilities(semantic):
    result=np.full((4,3,64,64),.0005,np.float32)
    for k in range(3):result[:,k][semantic==k]=.999
    return result


class Samples(list):
    scope='full32'
    def __init__(self,values):super().__init__(values);self.records=[dict(role=v['role']) for v in values]


class NearGoalTests(unittest.TestCase):
    def test_public_crop_signedK_native_pixels_relative_geometry_raw_unchanged(self):
        public=inputs();public['rgb'][0,128,128]=[128,20,4];before={k:v.copy() for k,v in public.items()}
        crop=data.make_near_goal_crops(public)
        self.assertEqual(crop['features'].shape,(4,9,64,64));np.testing.assert_array_equal(crop['origins_xy'],[[96,96]]*4)
        np.testing.assert_array_equal(crop['rgb'][0,32,32],public['rgb'][0,128,128])
        self.assertAlmostEqual(crop['features'][0,4,32,33],-1.025/200/.1,places=6)
        for k in public:np.testing.assert_array_equal(public[k],before[k])
        moved=deepcopy(public);delta=np.array([.1,-.2,.3]);moved['T'][:,:3,3]+=delta;moved['public_sphere_center']+=delta
        np.testing.assert_allclose(data.make_near_goal_crops(moved)['features'],crop['features'],atol=1e-6,rtol=0)
        for key in ('TCP','object_GT','semantic_role_ids','GRU'):
            bad=dict(public);bad[key]=0
            with self.assertRaises(ValueError):data.make_near_goal_crops(bad)

    def test_crop_padding_invalid_depth_and_behind_camera_not_fabricated(self):
        public=inputs();public['public_sphere_center'][0]=.63
        crop=data.make_near_goal_crops(public);self.assertFalse(crop['valid_crop'].all())
        labels=data.crop_semantics(np.zeros((4,256,256),np.uint8),crop)
        self.assertTrue(np.all(labels[~crop['valid_crop']]==255))
        public=inputs();public['depth_m'][:]=np.nan;crop=data.make_near_goal_crops(public)
        self.assertFalse(crop['valid_depth'].any());self.assertTrue(np.isfinite(crop['features']).all())
        obs=data.current_cube_measurement(public,crop,np.ones((4,64,64),bool));self.assertFalse(obs['refiner_available'])
        public=inputs();public['public_sphere_center'][2]=-1.
        crop=data.make_near_goal_crops(public);self.assertFalse(crop['projection_available'].any());self.assertFalse(crop['features'].any())

    def test_current_geometry_exact_old_algebra_scope_border_and_visibility(self):
        s=sample();mask=s['semantic']==1;measurement=data.current_cube_measurement(s['inputs'],s['crops'],mask)
        self.assertTrue(measurement['refiner_available']);self.assertEqual(measurement['geometry_confidence'],1.)
        np.testing.assert_allclose(measurement['centre'],[0.,0.,1.],atol=1e-6)
        far=sample('cube_far',far=True);self.assertTrue((far['semantic']==1).any())
        measured=data.current_cube_measurement(far['inputs'],far['crops'],far['semantic']==1)
        self.assertFalse(measured['refiner_available']);self.assertIn('outside_predeclared_near_scope_0p08m',measured['reasons'])
        one=mask.copy();one[1:]=False
        self.assertFalse(data.current_cube_measurement(s['inputs'],s['crops'],one)['refiner_available'])
        edge=mask.copy();edge[0,0,32]=True
        self.assertIn('cube_touches_crop_boundary',data.current_cube_measurement(s['inputs'],s['crops'],edge)['reasons'])
        invisible=sample('sphere_only',invisible=True)
        m=data.current_cube_measurement(invisible['inputs'],invisible['crops'],invisible['semantic']==1)
        self.assertFalse(m['refiner_available']);self.assertIsNone(m['centre'])

    def test_shared_UNet_loss_finite_gradients_and_input_only_tensor(self):
        net=model.TinyNearGoalUNet();x=torch.tensor(data.make_near_goal_crops(inputs())['features'])
        y=torch.tensor(sample()['semantic'],dtype=torch.long);logits=net(x)
        self.assertEqual(tuple(logits.shape),(4,3,64,64))
        np.testing.assert_allclose(logits[0].detach(),logits[1].detach(),atol=1e-6)
        loss=model.segmentation_loss(logits,y);self.assertEqual(tuple(loss.shape),(4,));loss.mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        with self.assertRaises(ValueError):net(dict(features=x,labels=y))
        with self.assertRaises(ValueError):net(torch.zeros(4,10,64,64))
        bad=x.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):net(bad)

    def test_oracle_pre_fit_geometry_bottleneck_not_hidden_by_GT_fill(self):
        good=Samples([sample(),sample('sphere_only',invisible=True),sample('cube_far',far=True)])
        report=evaluation.oracle_diagnostic(good);self.assertTrue(report['oracle_centre_gate_passed'])
        self.assertEqual(report['aggregate']['fully_invisible_falseavailable'],0)
        bad=Samples([sample(gt_offset=.02),sample('sphere_only',invisible=True)])
        report=evaluation.oracle_diagnostic(bad);self.assertFalse(report['ready_for_segmentation_fit'])
        self.assertGreater(report['aggregate']['qualified_visible_centre_error']['p95_m'],.005)
        with TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';manifest.write_text('{}')
            with patch.object(training,'CaptureDataset',return_value=bad),patch.object(training,'TinyNearGoalUNet',side_effect=AssertionError('must never construct NN before oracle gate')):
                with self.assertRaisesRegex(ValueError,'oracle visiblemask'):training.train_capture(manifest,root/'blocked')
            status=json.loads((root/'blocked/status.json').read_text());self.assertFalse(status['training_started'])

    def test_fixed_calibration_grid_lowest_pass_or_unqualified_and_far_semantics(self):
        samples=[sample(),sample('sphere_only',invisible=True),sample('cube_far',far=True)]
        probs=[probabilities(s['semantic']) for s in samples]
        reports=[evaluation.evaluate_probabilities(samples,probs,t) for t in evaluation.THRESHOLD_GRID]
        chosen=evaluation.choose_calibrated_threshold(reports);self.assertEqual(chosen['selected_threshold'],.5)
        self.assertEqual(reports[0]['pixels']['cube']['FP'],0)
        self.assertEqual(reports[0]['positions']['negatives']['cube_far']['falseavailable'],0)
        poisoned=probabilities(samples[1]['semantic']);poisoned[:,:,28:37,28:37]=.0005;poisoned[:,1,28:37,28:37]=.999
        bad=[probs[0],poisoned,probs[2]]
        reports=[evaluation.evaluate_probabilities(samples,bad,t) for t in evaluation.THRESHOLD_GRID]
        chosen=evaluation.choose_calibrated_threshold(reports);self.assertFalse(chosen['calibration_passed']);self.assertIsNone(chosen['selected_threshold'])
        with self.assertRaises(ValueError):evaluation.choose_calibrated_threshold(reports[:-1])

    def test_calibration_rejects_abstain_most_or_zero_oracle_eligible(self):
        positives=[sample() for _ in range(10)]
        for i,value in enumerate(positives):value['scene_id']=f'positive{i}'
        samples=positives+[sample('sphere_only',invisible=True),sample('cube_far',far=True)]
        probs=[probabilities(s['semantic']) for s in samples]
        for i in range(1,10):
            mask=positives[i]['semantic']==1
            probs[i][:,1][mask]=.0005;probs[i][:,0][mask]=.999
        reports=[evaluation.evaluate_probabilities(samples,probs,t) for t in evaluation.THRESHOLD_GRID]
        self.assertEqual(reports[0]['pixels']['cube']['precision'],1.)
        self.assertEqual(reports[0]['positions']['oracle_geometry_eligible'],10)
        self.assertEqual(reports[0]['positions']['oracle_eligible_coverage'],.1)
        self.assertLess(reports[0]['positions']['qualified_visible_centre_error']['p95_m'],.005)
        self.assertFalse(evaluation.choose_calibrated_threshold(reports)['calibration_passed'])
        empty=[sample('sphere_only',invisible=True),sample('cube_far',far=True)]
        reports=[evaluation.evaluate_probabilities(empty,[probabilities(s['semantic']) for s in empty],t) for t in evaluation.THRESHOLD_GRID]
        self.assertEqual(reports[0]['positions']['oracle_geometry_eligible'],0)
        self.assertIsNone(evaluation.choose_calibrated_threshold(reports)['selected_threshold'])

    def test_manifest_parent_splits_input_label_separation_and_immutable_sphere(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);public=inputs();semantic=np.zeros((4,256,256),np.uint8)
            semantic[:,124:133,124:133]=1;np.savez_compressed(root/'public_template.npz',**public)
            np.savez_compressed(root/'label_template.npz',semantic_role_ids=semantic,cube_center_world=np.array([0.,0.,1.]),cube_present=np.array(True))
            parents=[]
            partition=dict(parents=[dict(parent_id=parent_id,role='fit' if i<24 else ('selection' if i<28 else 'calibration'),
                pilot=parent_id in data.PILOT_IDS,shot_ids=list(data.SHOT_IDS)) for i,parent_id in enumerate(data.PARTITION_IDS)])
            partition_path=root/'partition.json';partition_path.write_text(json.dumps(partition))
            for p in range(8):
                role='fit' if p<6 else ('selection' if p==6 else 'calibration');shots=[]
                for i in range(8):
                    idir=root/f'p{p}'/'inputs';ldir=root/f'p{p}'/'labels';idir.mkdir(parents=True,exist_ok=True);ldir.mkdir(exist_ok=True)
                    ip=idir/f'{i}.npz';lp=ldir/f'{i}.npz';os.link(root/'public_template.npz',ip);os.link(root/'label_template.npz',lp)
                    shots.append(dict(shot_id=data.SHOT_IDS[i],scene_id=f'{p}/{i}',independent_scene=True,physics_ticks=0,condition='overlap',
                        inputs=dict(path=str(ip.relative_to(root)),sha256=data.sha256(ip)),labels=dict(path=str(lp.relative_to(root)),sha256=data.sha256(lp))))
                parents.append(dict(parent_id=data.PILOT_IDS[p],role=role,shots=shots,public_sphere_fit=dict(source_kind='initial_public_RGBD_RedGoalMarker',GT_pose_used=False,
                    source_frame_id=f'{p}/bootstrap',source_rgbd_sha256='a'*64,calibration_sha256=data.calibration_digest(public),fit_metadata_sha256='b'*64,center_world=[0.,0.,1.])))
            manifest=dict(schema=data.SCHEMA,dataset_split='TRAIN',scope='pilot_smoke',complete=True,
                all_native_closed=True,all_process_exits_observed=True,source_unchanged=True,parents=parents,
                partition=dict(path='partition.json',sha256=data.sha256(partition_path)))
            path=root/'manifest.json';path.write_text(json.dumps(manifest));dataset=data.CaptureDataset(path,'calibration')
            self.assertEqual(len(dataset),8);self.assertEqual(dataset.records[0]['declared_role'],'calibration')
            s=dataset[0];self.assertEqual(s['features'].shape,(4,9,64,64));self.assertTrue((s['semantic']==1).any())
            manifest['parents'][7]['public_sphere_fit']['center_world'][0]=.01;path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'sphere/calibration'):data.CaptureDataset(path,'calib')[0]
            manifest['parents'][7]['public_sphere_fit']['center_world'][0]=0.
            manifest['parents'][0]['role']='calibration';manifest['parents'][7]['role']='fit';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'role differs'):data.CaptureDataset(path)
            manifest['parents'][0]['role']='fit';manifest['parents'][7]['role']='calibration'
            manifest['parents'].pop();path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'preassigned parent cohort'):data.CaptureDataset(path)

    def test_checkpoint_identity_is_JSON_canonical_and_status_is_required(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';manifest.write_text('{}')
            identity=training.source_identity(manifest)
            self.assertEqual(identity,json.loads(json.dumps(identity)))
            checkpoint=root/'unqualified_fixture.pt';checkpoint.write_bytes(b'not_a_real_trained_checkpoint')
            meta=dict(training_complete=True,epochs_completed=40,dataset_split='TRAIN',TEST_read=False,
                quality_protocol=evaluation.QUALITY_PROTOCOL,current_visible_pixels_only=True,completion=False,
                TCP_GRU_GT_input=False,absolute_worldXYZ_input=False,feature_names=data.FEATURE_NAMES,
                class_names=data.CLASS_NAMES,scope='pilot_smoke',model_qualified=False)
            with patch.object(torch,'load',return_value=dict(version=training.VERSION,metadata=meta)):
                with self.assertRaisesRegex(ValueError,'status must accompany'):
                    model.CurrentNearGoalSegmenter(checkpoint,data.sha256(checkpoint),allow_unqualified_diagnostic=True)

    def test_incomplete_checkpoint_rejected_no_random_ready_fallback(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'untrained.pt';torch.save(dict(version=training.VERSION,model_state_dict=model.TinyNearGoalUNet().state_dict(),metadata=dict(training_complete=False)),path)
            with self.assertRaisesRegex(ValueError,'completed exact TRAIN'):model.CurrentNearGoalSegmenter(path,data.sha256(path))

if __name__=='__main__':unittest.main()
