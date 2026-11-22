from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.perception import RegistrationConfig
from our_mani.multitask import public_held_mesh_v33 as m

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_support_correction_rechecks_original_quality_and_does_not_mutate_pose(self):
        config=RegistrationConfig();points=np.array([[x,y,z] for x in (-.02,0,.02) for y in (-.02,0,.02) for z in (-.01,0,.01)])
        template=NS(components=[NS(name='shape',points_body=points)])
        pose=Pose(np.array([0.,0.,.009]),np.eye(3));clouds={'shape':points+np.array([0.,0.,.01])}
        result,row=m.supported_pose(pose,clouds,template,points,dict(height=0.,xy_min=[-1,-1],xy_max=[1,1]),config)
        self.assertTrue(row['accepted']);self.assertGreater(result.position[2],.01)
        np.testing.assert_array_equal(pose.position,[0,0,.009])
        # Same original gates reject a large correction rather than clipping geometry.
        bad=Pose(np.array([0.,0.,-.2]),np.eye(3));bad_clouds={'shape':points+bad.position}
        unchanged,row=m.supported_pose(bad,bad_clouds,template,points,dict(height=0.,xy_min=[-1,-1],xy_max=[1,1]),config)
        self.assertFalse(row['accepted']);self.assertIs(unchanged,bad)

    def test_frozen_asset_registration_retains_all_rotation_fallback_and_restores(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/task_executor_smoke_v22_setup/configuration.json').read_text())
        task='put_rubbish_in_moving_bin';b=m.binding(ROOT,c,task);c['public_held_mesh_v33']={task:b}
        asset=json.loads((ROOT/b['TRAIN_geometry']['path']).read_text());body=asset['bodies']['rubbish']
        cube=m.collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='none')
        calls=[]
        def original(role,v,f,**kwargs):calls.append((role,v,f,kwargs))
        service=NS(estimated_bodies={},register_estimated_geometry=original)
        with m.install_registration(service,c,ROOT,task):
            service.register_estimated_geometry('rubbish',cube.vertices,cube.triangles,source_sha256=asset['asset_sha256'])
        self.assertIs(service.register_estimated_geometry,original)
        _,v,f,kw=calls[0]
        self.assertEqual(m.mesh_sha256(v,f),b['primary_mesh_sha256'])
        self.assertEqual(m.mesh_sha256(kw['fallback_mesh']['vertices'],kw['fallback_mesh']['triangles']),b['fallback_mesh_sha256'])
        self.assertFalse(b['model_history_changed']);self.assertFalse(b['collision_permissions_changed'])

    def test_original_numpy_quality_boolean_is_admitted_without_truthiness_coercion(self):
        self.assertTrue(m.registration_role_valid({'per_role':{'rubbish':{'valid':np.bool_(True)}}},'rubbish'))
        self.assertFalse(m.registration_role_valid({'per_role':{'rubbish':{'valid':np.bool_(False)}}},'rubbish'))
        self.assertFalse(m.registration_role_valid({'per_role':{'rubbish':{'valid':'true'}}},'rubbish'))

    def test_contact_evidence_and_causal_age_are_mandatory(self):
        state=NS(robot=NS(contact=False,timestamp=1.))
        self.assertFalse(m.held_admissible(state,{'rubbish':'attached'},set(),{'timestamp':1.},.1))
        state.robot.contact=True
        self.assertFalse(m.held_admissible(state,{'rubbish':'external'},set(),{'timestamp':1.},.1))
        self.assertFalse(m.held_admissible(state,{'rubbish':'attached'},set(),None,.1))
        self.assertFalse(m.held_admissible(state,{'rubbish':'attached'},set(),{'timestamp':0.},.1))


if __name__=='__main__':unittest.main()
