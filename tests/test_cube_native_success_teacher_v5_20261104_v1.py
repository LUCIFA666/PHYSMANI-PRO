"""Pure acceptance/compile/driver contracts. No recorded private stream or NN."""
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from our_mani.cube_candidate_teacher_20261104_v5.labels import actual_local_grasp_contacts
from our_mani.cube_candidate_teacher_20261104_v5.data import compile_contact_episode,combine_episode_results,public_features
from our_mani.cube_candidate_matched_20261104_v2 import data as original
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('synthetic_fixture',Path(__file__).with_name('test_cube_candidate_matched_supervision_20261104_v2.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class NativeSuccessV5(unittest.TestCase):
    def test_success_short_unlifted_drifting_segment_accepts_first_held(self):
        public,initial,ticks=f.fixture();ticks=ticks[:32]
        for row in ticks:
            if row['held']:
                row['native_success']=row['time']>=1.6
                row['object_position']=[0,0,0]
                row['tcp_position']=[row['time'],0,.1]
        contacts=actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id'])
        self.assertEqual(len(contacts),1);c=contacts[0]
        self.assertEqual(c['acceptance_class'],'native_task_success_contact')
        self.assertAlmostEqual(c['time'],1.5);self.assertAlmostEqual(c['Native_task_success_observation_time'],1.6)
        self.assertFalse(c['extra_hold_gate_applied']);self.assertFalse(c['hold']['passed'])
        arrays,report=compile_contact_episode(public[:33],initial,ticks,parent_id=f.PARENT,episode_id=initial['episode_id'],expected_forecast_sha256=f.SHA)
        self.assertGreater(len(arrays['target_dt']),0);self.assertGreater(arrays['target_dt'].max(),.8)
        self.assertFalse(arrays['target_time_valid'][arrays['target_dt']>.8].any())
        self.assertEqual(report['selected_contact_acceptance_classes'],['native_task_success_contact'])

    def test_success_without_actual_held_or_contact_never_positive(self):
        _,initial,ticks=f.fixture(False)
        for row in ticks:row['native_success']=True
        self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])
        _,initial,ticks=f.fixture();ticks=ticks[:32]
        for row in ticks:row['native_success']=False
        self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])

    def test_local_only_keeps_original_lift_duration_drift_gate(self):
        _,initial,ticks=f.fixture()
        for row in ticks:row['native_success']=False
        c=actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id'])[0]
        self.assertEqual(c['acceptance_class'],'local_lift_hold_without_task_success')
        self.assertFalse(c['original_Native_task_success_observed']);self.assertTrue(c['extra_hold_gate_applied'])
        for row in ticks:
            if row['held']:row['object_position'][2]=.039
        self.assertEqual(actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id']),[])

    def test_later_native_success_prioritized_over_earlier_local_segment(self):
        public,initial,ticks=f.fixture()
        for row in ticks:
            t=row['time'];first=.45<=t<=1.05;second=t>=1.5
            row.update(held=bool(first or second),requested_opening=0. if(.4<=t<=1.05 or t>=1.45)else 1.,native_success=t>=1.6)
            row['object_position']=[0,0,.05 if row['held']else 0.]
            row['tcp_position']=[.01 if first else .02,0,.06 if row['held']else .1]
        for pub,row in zip(public[1:],ticks):pub['close_requested']=row['requested_opening']<.5
        contacts=actual_local_grasp_contacts(initial,ticks,episode_id=initial['episode_id'])
        self.assertEqual([c['acceptance_class']for c in contacts],['local_lift_hold_without_task_success','native_task_success_contact'])
        arrays,report=compile_contact_episode(public,initial,ticks,parent_id=f.PARENT,episode_id=initial['episode_id'],expected_forecast_sha256=f.SHA)
        self.assertEqual(report['selected_contact_acceptance_classes'],['native_task_success_contact'])
        self.assertTrue(np.allclose(arrays['target_position'][:,0],.02))

    def test_public_feature_identity_heldout_and_failure_denominator(self):
        self.assertIs(public_features,original.public_features)
        public,initial,ticks=f.fixture()
        with self.assertRaises(ValueError):compile_contact_episode(public,initial,ticks,parent_id='Cube_native_generated_ordinary_seed2611042006',episode_id=initial['episode_id'],expected_forecast_sha256=f.SHA)
        with self.assertRaises(ValueError):combine_episode_results([({},dict(parent_id=f.PARENT,role='fit'))])

    def test_driver_gate_and_public_replay_are_frozen_v4_math(self):
        old=ast.parse((ROOT/'scripts/compile_cube_teacher_candidate_dataset_20261104_v4.py').read_text())
        new=ast.parse((ROOT/'scripts/compile_cube_teacher_candidate_dataset_20261104_v5.py').read_text())
        def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
        class FreezeVersion(ast.NodeTransformer):
            def visit_Constant(self,n):
                if n.value=='Cube_corrected_GRU_exact_teacher2_public_prediction_freeze_20261104_v4':n.value='Cube_corrected_GRU_exact_teacher2_public_prediction_freeze_20261104_v5'
                return n
        for name in ('teacher_gate','predict_public','validate_public_freeze','metadata'):
            self.assertEqual(ast.dump(FreezeVersion().visit(fn(old,name)),include_attributes=False),ast.dump(fn(new,name),include_attributes=False),name)
        source=(ROOT/'scripts/compile_cube_teacher_candidate_dataset_20261104_v5.py').read_text()
        self.assertIn('from our_mani.cube_candidate_teacher_20261104_v3.replay import predict_public_episode',source)
        self.assertIn('from our_mani.cube_candidate_teacher_20261104_v5.data import compile_contact_episode',source)
        self.assertIn("choices=('fit','selection')",source)

if __name__=='__main__':unittest.main()
