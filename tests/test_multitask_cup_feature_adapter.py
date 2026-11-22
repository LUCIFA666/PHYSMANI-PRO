"""Actual production adapter with saved TRAIN RGBD, no simulator or GT read."""
from dataclasses import replace
import json
import unittest

import numpy as np
from scipy.spatial import cKDTree

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.cup_features import (ROOT,CUP_ROLES,
    load_frozen_cup_features,estimate_cup_feature_pose)
from our_mani.multitask.perception import (BodyTemplate,MultiBodySE3Tracker,RegistrationConfig,
    decode_instance_mask)
from our_mani.multitask.training_geometry import body_templates_from_asset


class ActualCupFeatureAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'artifacts/framework/multitask_r10'
        plan=json.loads((base/'TRAIN_cup_feature_replay_plan_v1.json').read_text())
        cls.case=next(case for case in plan['cases'] if case['case_id']=='orientation_pilot_v2__place_cups_on_rotating_frame')
        cls.task=cls.case['task']
        permissions=json.loads((base/'cup_feature_permissions_v1/cup_feature_priors.json').read_text())
        permission=permissions[cls.task]
        cls.asset=json.loads((ROOT/permission['TRAIN_CAD']['path']).read_text())
        cls.models=load_frozen_cup_features(permission,cls.asset)
        old_asset=json.loads((ROOT/cls.case['geometry']['path']).read_text())
        # Saved mask IDs are scoped to that TRAIN sensor process. Rebind by
        # declared component names, never by native handles, GT or role order.
        bindings={}
        for role,body in cls.asset['bodies'].items():
            previous={component['name']:component for component in old_asset['bodies'][role]['components']}
            bindings[role]={component['name']:tuple(previous[component['name']]['TRAIN_instance_ids'])
                for component in body['components']}
        all_templates=body_templates_from_asset(cls.asset,bindings,
            orientation_observability={role:'full' for role in cls.asset['bodies']},
            cup_feature_models=cls.models)
        cls.templates=tuple(template for template in all_templates if template.role_id in CUP_ROLES)
        cls.calibrations={name:CameraCalibration(np.asarray(row['intrinsics']),Pose.from_xyzw(row['camera_to_world']))
            for name,row in cls.case['public_cameras'].items()}
        with np.load(ROOT/cls.case['sensor']['path'],allow_pickle=False) as stored:
            # This cache has current sensor/robot data only; no report/GT file
            # is opened anywhere in this test suite.
            cls.times=stored['times'][:2].copy()
            cls.depths={camera:stored['depth_'+camera][:2].copy() for camera in cls.calibrations}
            cls.masks={camera:stored['mask_'+camera][:2].copy() for camera in cls.calibrations}

    def observation(self,index=0,*,missing_depth=False,timestamp=None):
        now=float(self.times[index] if timestamp is None else timestamp)
        depths={camera:(np.zeros_like(values[index]) if missing_depth else values[index].copy())
            for camera,values in self.depths.items()}
        return dict(timestamp=now,**{'observation/calibrations':self.calibrations,
            'observation/depths':depths,'observation/instance_masks':{camera:values[index].copy() for camera,values in self.masks.items()},
            'observation/camera_timestamps':{camera:now for camera in self.calibrations}})

    def raw_points_and_views(self,observation,template):
        ids=tuple(value for component in template.components for value in component.instance_ids)
        views={camera:(calibration,observation['observation/depths'][camera],
            decode_instance_mask(observation['observation/instance_masks'][camera]))
            for camera,calibration in self.calibrations.items()}
        points=np.concatenate([calibration.backproject(depth,np.isin(labels,ids))
            for calibration,depth,labels in views.values()])
        return points,views

    def tracker(self):return MultiBodySE3Tracker(self.templates)

    def test_actual_adapter_matches_same_raw_RGBD_helper_and_measured_surface_gate(self):
        tracker=self.tracker();observation=self.observation()
        states,metadata=tracker.observe(observation,modes={role:'external' for role in CUP_ROLES})
        states={state.object_id:state for state in states};raw_counts=[]
        self.assertEqual(set(states),set(CUP_ROLES))
        for template in self.templates:
            role=template.role_id;points,views=self.raw_points_and_views(observation,template);raw_counts.append(len(points))
            direct=estimate_cup_feature_pose(points,self.models[role],views,reference_time=observation['timestamp'])
            pose=Pose.from_xyzw(direct['pose_xyzw'])
            self.assertTrue(np.allclose(states[role].pose.position,pose.position,rtol=0,atol=1e-12))
            self.assertTrue(np.allclose(states[role].pose.rotation,pose.rotation,rtol=0,atol=1e-12))
            distances=cKDTree(np.concatenate([c.points_body for c in template.components])).query(pose.inverse_apply(points))[0]
            inliers=distances<=2*tracker.config.maximum_registration_rms_m
            rms=float(np.sqrt(np.mean(distances[inliers]**2)))
            diag=metadata['bodies'][role];feature=diag['cup_feature_registration']
            self.assertEqual(feature['raw_current_points'],len(points))
            self.assertAlmostEqual(feature['observed_surface_rms_m'],rms,places=12)
            self.assertAlmostEqual(feature['observed_surface_inlier_fraction'],float(inliers.mean()),places=12)
            self.assertFalse(direct['ambiguous']);self.assertTrue(feature['full_rotation_ready'])
            self.assertTrue(diag['registration_valid']);self.assertTrue(diag['orientation_observable'])
            self.assertFalse(feature['GT_or_native_pose_used']);self.assertFalse(feature['parameter_search_used'])
        self.assertGreater(max(raw_counts),RegistrationConfig().maximum_points_per_component,
            'this fixture must exercise raw backprojection beyond the generic ICP384-point cap')

    def test_mask_without_current_depth_clears_orientation_and_current_feature_metadata(self):
        tracker=self.tracker();tracker.observe(self.observation(),modes={role:'external' for role in CUP_ROLES})
        states,metadata=tracker.observe(self.observation(1,missing_depth=True),modes={role:'external' for role in CUP_ROLES})
        states={state.object_id:state for state in states}
        for role in CUP_ROLES:
            diag=metadata['bodies'][role]
            self.assertFalse(diag['registration_valid']);self.assertFalse(diag['orientation_observable'])
            self.assertIsNone(diag['cup_feature_registration']);self.assertFalse(diag['directly_visible'])
            self.assertFalse(states[role].visible)
            self.assertEqual(len(tracker.history[role]),0);self.assertEqual(len(tracker.axis_history[role]),0)
            self.assertGreater(sum(diag['mask_pixels_by_camera'].values()),0,
                'mask identity alone must not authorize orientation when depth is absent')

    def test_two_real_sensor_frames_exercise_production_pose_history_without_writable_buffer_error(self):
        tracker=self.tracker();tracker.observe(self.observation(),modes={role:'external' for role in CUP_ROLES})
        states,metadata=tracker.observe(self.observation(1),modes={role:'external' for role in CUP_ROLES})
        for state in states:
            self.assertTrue(np.isfinite(state.angular_velocity).all())
            self.assertTrue(metadata['bodies'][state.object_id]['orientation_observable'])
            self.assertEqual(len(tracker.axis_history[state.object_id]),2)

    def test_wrong_model_provenance_role_and_nonfull_annotation_are_rejected(self):
        template=next(t for t in self.templates if t.role_id=='cup0')
        with self.assertRaisesRegex(ValueError,'exact TRAIN CAD'):
            replace(template,provenance=replace(template.provenance,evidence_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'exact TRAIN CAD'):
            replace(template,cup_feature_model=self.models['cup1'])
        for annotation in ('none','axis'):
            with self.subTest(annotation=annotation),self.assertRaisesRegex(ValueError,'exact TRAIN CAD'):
                replace(template,orientation_observability=annotation)
        with self.assertRaisesRegex(ValueError,'exact TRAIN CAD'):
            replace(template,role_id='frame')


if __name__=='__main__':unittest.main()
