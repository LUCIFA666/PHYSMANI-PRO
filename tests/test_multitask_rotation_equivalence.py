"""Finite geometric symmetry, source lineage and public ICP quotient checks."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask import rotation_equivalence as quotient
from our_mani.multitask.component_flags import SCHEMA,NATIVE_TRUE_VALUES
from our_mani.multitask.evaluation import file_sha,digest
from our_mani.multitask.training_geometry import ASSET_VERSION,_sha


def D4():
    spin=Rotation.from_rotvec(np.array([1.,0.,0.])[None]*np.arange(4)[:,None]*np.pi/2).as_matrix()
    flip=Rotation.from_rotvec([0.,np.pi,0.]).as_matrix()
    return np.concatenate((spin,spin@flip))


def asset(center=(0.,0.,0.)):
    v=np.array(list(itertools.product((-1.,1.),repeat=3)))*[.01,.025,.025]+center
    quads=((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
    triangles=[[face[0],face[1],face[2]] for face in quads]+[[face[0],face[2],face[3]] for face in quads]
    body=dict(components=[dict(name='square_ring',points_body=v.tolist())],
        owned_component_annotations=[dict(name='square_ring',flags=dict.fromkeys(NATIVE_TRUE_VALUES,True),
            native_getter_values=NATIVE_TRUE_VALUES.copy(),native_true_values=NATIVE_TRUE_VALUES.copy())],
        visual_component_names=['square_ring'],collision_component_names=['square_ring'],
        empty_visual_role=False,empty_collision_role=False,visual_geometry_status='renderable_components',
        collision_vertices_body=v.tolist(),collision_triangles=triangles)
    value=dict(version=ASSET_VERSION,split='train',shared_with_baseline=True,component_role_schema=SCHEMA,
        task='insert_onto_rotating_peg',source_group='train:insert_onto_rotating_peg:episode1',
        source_hdf5_sha256='a'*64,bodies=dict(ring=body))
    value['asset_sha256']=_sha(value);return value


class RotationEquivalenceTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def pointer(self,path):return dict(path=str(path.relative_to(self.root)),sha256=file_sha(path))
    def loaded(self,center=(0.,0.,0.)):
        source=self.write('our_mani/multitask/rotation_equivalence.py',{})
        codes=[self.write('our_mani/multitask/'+name,{}) for name in
            ('training_geometry.py','component_flags.py','safe_native_mesh.py')]
        launcher=self.write('scripts/run_local_sim.sh',{})
        plan=self.write('plan.json',dict(plan_sha256='plan',inputs={str(p):file_sha(p) for p in codes}))
        mesh=self.write('TRAIN_geometry.json',asset(center));entry=dict(split='train',task='insert_onto_rotating_peg',
            parent_source_group='train:insert_onto_rotating_peg:episode1')
        raw=self.write('raw_case.json',dict(case_id='case',status='completed',split='train',task=entry['task'],
            entry=entry,plan_sha256='plan',policy_history_contains_native_GT=False,process_id=42,
            case_isolation='fresh_simulator_process_per_episode',files_sha256={'TRAIN_geometry.json':file_sha(mesh)},
            native_reset_receipt=dict(source_hdf5_sha256='a'*64)))
        canonical=json.loads(raw.read_text());canonical['parent_process_receipt']=dict(
            child_case_path='raw_case.json',child_case_sha256=file_sha(raw),launcher='scripts/run_local_sim.sh',
            launcher_sha256=file_sha(launcher),command=['bash','scripts/run_local_sim.sh','--plan','plan.json'])
        receipt=self.write('case.json',canonical)
        manifest=self.write('manifest.json',dict(complete=True,completed=1,expected=1,missing_case_ids=[],TEST_used=False,
            entries=[entry],case_receipts=[dict(case_id='case',**self.pointer(receipt))],plan_sha256='plan'))
        with patch.object(quotient,'__file__',str(source)):
            value=quotient.build_train_certificate(self.root,self.pointer(mesh),'ring',D4(),center,
                TRAIN_manifest_pointer=self.pointer(manifest),TRAIN_receipt_pointer=self.pointer(receipt))
            certificate=self.write('certificate.json',value)
            verified=quotient.VerifiedFiniteSymmetry.load(self.root,self.pointer(certificate))
        return verified,certificate,source

    def observe(self,verified,poses,rms,**kwargs):
        return quotient.current_orientation_equivalence(verified,poses,rms,
            registration_valid=kwargs.get('registration_valid',True),directly_visible=kwargs.get('directly_visible',True),
            all_retained_hypotheses_provided=kwargs.get('all_retained_hypotheses_provided',True))

    def test_group_algebra_rejects_reflections_duplicates_nonclosure_and_C8(self):
        group,proof=quotient.finite_group(D4());self.assertEqual(proof['order'],8)
        reflection=D4();reflection[0,0,0]=-1
        with self.assertRaisesRegex(ValueError,'proper SO'):
            quotient.finite_group(reflection)
        duplicate=D4();duplicate[1]=duplicate[0]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            quotient.finite_group(duplicate)
        broken=D4();broken[1]=Rotation.from_rotvec([.1,0.,0.]).as_matrix()
        with self.assertRaisesRegex(ValueError,'closure'):
            quotient.finite_group(broken)
        C8=Rotation.from_rotvec(np.array([1.,0.,0.])[None]*np.arange(8)[:,None]*np.pi/4).as_matrix()
        with self.assertRaisesRegex(ValueError,'not D4'):
            quotient.finite_group(C8)

    def test_complete_triangle_surface_evidence_is_required_not_vertices_only(self):
        value=asset();proof=quotient.certify_train_geometry(value,'ring',D4(),[0.,0.,0.])
        self.assertTrue(proof['complete_visual_and_physical_geometric_surface_evidence'])
        self.assertTrue(all(row['convex_quadrilateral_face_bijection'] for row in proof['group_surface_matches']))
        self.assertFalse(proof['policy_skill_authorized']);self.assertTrue(proof['TRAIN_phase_affordance_permission_required'])
        value=asset();value['bodies']['ring']['collision_triangles'].pop();value['asset_sha256']=_sha({k:v for k,v in value.items() if k!='asset_sha256'})
        with self.assertRaisesRegex(ValueError,'closed two-manifold'):
            quotient.certify_train_geometry(value,'ring',D4(),[0.,0.,0.])
        value=asset();vertices=np.array(value['bodies']['ring']['collision_vertices_body']);vertices[:,2]*=1.2
        value['bodies']['ring']['collision_vertices_body']=vertices.tolist();value['asset_sha256']=_sha({k:v for k,v in value.items() if k!='asset_sha256'})
        with self.assertRaisesRegex(ValueError,'does not preserve'):
            quotient.certify_train_geometry(value,'ring',D4(),[0.,0.,0.])

    def test_file_loader_reproduces_real_acquisition_and_detects_source_change(self):
        verified,path,source=self.loaded();self.assertTrue(verified.certificate['TRAIN_only'])
        checksum=file_sha(path);source.write_text('changed')
        with patch.object(quotient,'__file__',str(source)),self.assertRaisesRegex(ValueError,'input changed'):
            quotient.VerifiedFiniteSymmetry.load(self.root,dict(path='certificate.json',sha256=checksum))

    def test_current_D4_hypotheses_can_agree_without_claiming_full_R_or_skill(self):
        verified,path,source=self.loaded();base=Pose(np.array([.3,.2,.5]),Rotation.from_rotvec([.2,.1,-.3]).as_matrix())
        poses=[base,Pose(base.position,base.rotation@D4()[4])]
        result=self.observe(verified,poses,[.002,.0021])
        self.assertTrue(result['orientation_equivalence_observable']);self.assertFalse(result['full_orientation_observable'])
        self.assertFalse(result['policy_skill_authorized']);self.assertFalse(result['absolute_registration_accuracy_proven'])
        self.assertEqual(result['representative_pose']['rotation'],base.rotation.tolist())

    def test_45degree_near_optimum_is_not_continuous_axis_equivalence(self):
        verified,path,source=self.loaded();poses=[Pose(np.zeros(3)),Pose(np.zeros(3),Rotation.from_rotvec([np.pi/4,0.,0.]).as_matrix())]
        result=self.observe(verified,poses,[.002,.0021])
        self.assertFalse(result['orientation_equivalence_observable']);self.assertIn('outside',result['reason'])

    def test_SE3_equivalence_uses_world_symmetry_center_not_raw_pose_translation(self):
        center=np.array([.1,.2,.3]);verified,path,source=self.loaded(center)
        base=Pose(np.array([.3,.2,.5]));g=D4()[1]
        equivalent=Pose(base.position+base.rotation@(center-g@center),base.rotation@g)
        self.assertTrue(self.observe(verified,[base,equivalent],[.002,.0021])['orientation_equivalence_observable'])
        wrong=Pose(equivalent.position+[.01,0.,0.],equivalent.rotation)
        self.assertFalse(self.observe(verified,[base,wrong],[.002,.0021])['orientation_equivalence_observable'])

    def test_missing_current_evidence_or_search_subset_fails_closed(self):
        verified,path,source=self.loaded();poses=[Pose(np.zeros(3)),Pose(np.zeros(3),D4()[1])]
        for field in ('registration_valid','directly_visible','all_retained_hypotheses_provided'):
            self.assertFalse(self.observe(verified,poses,[.002,.0021],**{field:False})['orientation_equivalence_observable'])
        self.assertFalse(self.observe(verified,poses,[.03,.0301])['orientation_equivalence_observable'])
        self.assertFalse(self.observe(verified,poses[:1],[.002])['orientation_equivalence_observable'])

    def test_loaded_certificate_cannot_be_silently_mutated(self):
        verified,path,source=self.loaded();verified.certificate['center_body']=[9.,9.,9.]
        with self.assertRaisesRegex(ValueError,'was mutated'):
            self.observe(verified,[Pose(np.zeros(3)),Pose(np.zeros(3),D4()[1])],[.002,.0021])

    def test_typed_current_envelope_binds_current_identity_time_pose_and_certificate(self):
        verified,path,source=self.loaded();pose=Pose(np.array([.3,.1,.6]))
        value=quotient.CurrentRotationEquivalence.from_root_hypotheses(verified,body_id='ring',timestamp=2.,
            root_candidate_poses=[pose,Pose(pose.position,pose.rotation@D4()[1])],candidate_rms_m=[.002,.0021],
            body_in_root=Pose(np.zeros(3)),registration_valid=True,directly_visible=True,
            all_retained_hypotheses_provided=True,representative_body_pose=pose)
        self.assertTrue(value.matches(body_id='ring',timestamp=2.,pose=pose,now=2.05,maximum_age=.1))
        self.assertFalse(value.matches(body_id='ring',timestamp=2.,pose=pose,now=2.2,maximum_age=.1))
        self.assertFalse(value.matches(body_id='peg0',timestamp=2.,pose=pose))
        self.assertFalse(value.matches(body_id='ring',timestamp=2.05,pose=pose))
        self.assertFalse(value.matches(body_id='ring',timestamp=2.,pose=Pose(pose.position+[.001,0.,0.])) )
        self.assertFalse(value.matches(body_id='ring',timestamp=2.,pose=pose,certificate_sha256='b'*64))
        value.evidence['timestamp']=3.
        with self.assertRaisesRegex(ValueError,'mutated'):value.receipt()

    def test_root_hypotheses_are_composed_into_actual_role_frame_before_D4_check(self):
        verified,path,source=self.loaded();local=Pose(np.array([.05,.03,.02]),Rotation.from_rotvec([.4,.2,.1]).as_matrix())
        body=Pose(np.array([.3,.1,.6]),Rotation.from_rotvec([.1,.2,.3]).as_matrix())
        second=Pose(body.position,body.rotation@D4()[1])
        roots=[Pose(p.position-p.rotation@local.rotation.T@local.position,p.rotation@local.rotation.T)
            for p in (body,second)]
        value=quotient.CurrentRotationEquivalence.from_root_hypotheses(verified,body_id='ring',timestamp=1.,
            root_candidate_poses=roots,candidate_rms_m=[.002,.0021],body_in_root=local,
            registration_valid=True,directly_visible=True,all_retained_hypotheses_provided=True,
            representative_body_pose=body)
        self.assertTrue(value.observable)
        self.assertFalse(value.evidence['full_orientation_observable'])
        with self.assertRaisesRegex(ValueError,'actual current observed'):
            quotient.CurrentRotationEquivalence.from_root_hypotheses(verified,body_id='ring',timestamp=1.,
                root_candidate_poses=roots,candidate_rms_m=[.002,.0021],body_in_root=local,
                registration_valid=True,directly_visible=True,all_retained_hypotheses_provided=True,
                representative_body_pose=Pose(body.position,body.rotation@D4()[1]))

    def test_explicit_metric_replay_guard_preserves_exact_structure_sources_and_other_numbers(self):
        verified,path,source=self.loaded();baseline=verified.certificate
        def resign(value):
            value['certificate_sha256']=digest({k:v for k,v in value.items() if k!='certificate_sha256'});return value
        narrow=deepcopy(baseline)
        narrow['group_surface_matches'][0]['original_surface_Hausdorff_upper_bound_m']+=quotient.SURFACE_LENGTH_REPLAY_GUARD_M/10
        receipt=quotient.replay_train_certificate(baseline,resign(narrow))
        self.assertTrue(receipt['accepted']);self.assertEqual(receipt['different_derived_float_fields'],1)
        length=deepcopy(baseline)
        length['group_surface_matches'][0]['original_surface_Hausdorff_upper_bound_m']+=quotient.SURFACE_LENGTH_REPLAY_GUARD_M*10
        with self.assertRaisesRegex(ValueError,'explicit m replay guard'):
            quotient.replay_train_certificate(baseline,resign(length))
        area=deepcopy(baseline);area['surface_coverage'][0]['triangle_area_m2']+=quotient.SURFACE_AREA_REPLAY_GUARD_M2*10
        with self.assertRaisesRegex(ValueError,'explicit m2 replay guard'):
            quotient.replay_train_certificate(baseline,resign(area))
        for field in ('source','structure','non_whitelisted_threshold','group_permutation','guard_policy'):
            changed=deepcopy(baseline)
            if field=='source':changed['CAD_asset_sha256']='f'*64
            elif field=='structure':changed['surface_coverage'].append(deepcopy(changed['surface_coverage'][0]))
            elif field=='non_whitelisted_threshold':changed['surface_coverage'][0]['plane_tolerance_m']+=1e-20
            elif field=='group_permutation':changed['group_surface_matches'][0]['convex_part_permutation'][0]=99
            else:changed['numerical_replay_policy']['absolute_length_guard_m']=1e-6
            with self.assertRaises(ValueError,msg=field):quotient.replay_train_certificate(baseline,resign(changed))
        unsigned=deepcopy(baseline);unsigned['group_surface_matches'][0]['original_surface_Hausdorff_upper_bound_m']+=1e-18
        with self.assertRaisesRegex(ValueError,'fingerprint'):quotient.replay_train_certificate(baseline,unsigned)

    def test_replay_guard_cannot_admit_a_surface_without_margin_to_original_bound(self):
        verified,path,source=self.loaded();changed=deepcopy(verified.certificate)
        changed['group_surface_matches'][0]['original_surface_Hausdorff_upper_bound_m']=changed['surface_tolerance_m']
        changed['certificate_sha256']=digest({k:v for k,v in changed.items() if k!='certificate_sha256'})
        with self.assertRaisesRegex(ValueError,'no margin'):quotient.replay_train_certificate(changed,changed)


if __name__=='__main__':unittest.main()
