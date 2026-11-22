"""TRAIN CAD export and optional mask binding contracts, without PyRep/GPU."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.specs import get_task_spec
from our_mani.multitask.training_geometry import (export_train_geometry, declared_instance_bindings,
    body_templates_from_asset, register_collision_asset)


class Shape:
    def __init__(self, name, handle, position, children=(), flags=(True, True, False)):
        self.name, self.handle, self.children = name, handle, tuple(children)
        self.matrix, self.forbid_pose = np.eye(4), False
        self.flags, self.forbid_flags, self.forbid_mesh = flags, False, False
        self.matrix[:3, 3] = position
    def get_name(self): return self.name
    def get_handle(self): return self.handle
    def get_objects_in_tree(self, **kwargs):
        return [self, *[part for child in self.children for part in child.get_objects_in_tree(**kwargs)]]
    def get_matrix(self):
        if self.forbid_pose: raise AssertionError('native pose queried outside TRAIN export')
        return self.matrix.copy()
    def _flag(self, index):
        if self.forbid_flags: raise AssertionError('current native flags queried outside TRAIN annotations')
        return self.flags[index]
    def is_renderable(self): return self._flag(0)
    def is_collidable(self): return self._flag(1)
    def is_respondable(self): return self._flag(2)
    def get_mesh_data(self):
        if self.forbid_mesh: raise AssertionError('neither component mesh must not enter CAD')
        return [0., 0., 0., .1, 0., 0., 0., .1, 0.], [0, 1, 2], []


def fixture():
    plate = Shape('target_button_topPlate', 22, [.3, 0., .85])
    wrap = Shape('target_button_wrap', 23, [.3, 0., .8])
    button = Shape('push_button_target', 21, [.3, 0., .8], [plate, wrap])
    joint = SimpleNamespace(get_handle=lambda: 24)
    task = SimpleNamespace(t=0., target_button=button, target_topPlate=plate, target_wrap=wrap, joint=joint)
    source = dict(task='push_moving_button', raw_episode_id='episode9', split='train')
    env = SimpleNamespace(task=task, spec=get_task_spec(source['task']), development_split='train',
        development_source=source, physics_time=lambda: 0., _aligned=True,
        resolve_scene_object=lambda name: None,
        reset_receipt=dict(source_hdf5_sha256='0' * 64,
            reset_kind='native_generated_synthetic_development_augmentation'))
    return env, source, (button, plate, wrap)


class GeometryTests(unittest.TestCase):
    def test_export_is_train_only_preserves_articulated_parts_and_triangle_surfaces(self):
        env, source, parts = fixture()
        asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        assert set(asset['bodies']) == {'button', 'plate', 'wrap'}
        for body in asset['bodies'].values():
            assert len(body['components']) == 1
            assert len(body['collision_vertices_body']) == 3 and body['collision_triangles'] == [[0, 1, 2]]
            assert np.asarray(body['components'][0]['points_body']).shape == (16, 3)
        assert asset['world_pose_or_motion_schedule_in_asset'] is False
        assert asset['source_original_motion_replayed'] is False
        assert asset == export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        env.development_split = 'validation'
        for part in parts: part.forbid_pose = True
        with self.assertRaises(MultitaskProtocolError): export_train_geometry(env, source, shape_type='shape')

    def test_current_bindings_are_explicit_shared_permission_and_never_read_poses(self):
        env, source, parts = fixture()
        asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        env.development_split = 'validation'
        for part in parts:
            part.forbid_pose = True
            part.forbid_flags = True
            part.handle += 100
        with self.assertRaises(MultitaskProtocolError): declared_instance_bindings(env, asset, shape_type='shape')
        bindings = declared_instance_bindings(env, asset, shared_permission=True, shape_type='shape')
        assert bindings['plate']['target_button_topPlate'] == (122,)
        templates = body_templates_from_asset(asset, bindings,
            orientation_observability={name: 'axis' for name in asset['bodies']},
            symmetry_axes={name: [0., 0., 1.] for name in asset['bodies']})
        assert all(template.provenance.source_kind == 'declared_cad' for template in templates)
        assert all(template.provenance.calibration_split == 'train' for template in templates)
        assert all(template.orientation_observability == 'axis' for template in templates)
        assert all(np.array_equal(template.symmetry_axis, [0., 0., 1.]) for template in templates)
        corrupt = deepcopy(asset);corrupt['bodies']['plate']['collision_vertices_body'][0][0] += .1
        with self.assertRaises(MultitaskProtocolError):
            declared_instance_bindings(env, corrupt, shared_permission=True, shape_type='shape')
        parts[1].name = 'changed_component'
        with self.assertRaises(MultitaskProtocolError):
            declared_instance_bindings(env, asset, shared_permission=True, shape_type='shape')

    def test_collision_registration_is_pure_asset_data_and_binds_train_fingerprint(self):
        env, source, _ = fixture()
        asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        calls = []
        service = SimpleNamespace(register_estimated_geometry=lambda *args, **kwargs: calls.append((args, kwargs)))
        register_collision_asset(service, asset,orientation_observability={name:'full' for name in asset['bodies']})
        assert len(calls) == 3
        assert all(kwargs['provenance']=='TRAIN_geometry_model' and kwargs['source_sha256']==asset['asset_sha256']
                   and kwargs['collision_geometry_receipt']['collision_geometry_kind']=='exact_TRAIN_triangles'
                   and kwargs['collision_geometry_receipt']['TRAIN_asset_sha256']==asset['asset_sha256']
                   for _, kwargs in calls)
        for args,kwargs in calls:
            body=asset['bodies'][args[0]]
            assert np.array_equal(args[1],body['collision_vertices_body'])
            assert np.array_equal(args[2],body['collision_triangles'])
            fallback=kwargs['fallback_mesh']
            assert fallback['collision_geometry_receipt']['collision_geometry_kind']=='rotation_uncertainty_cube'
            assert fallback['collision_geometry_receipt']['TRAIN_asset_sha256']==asset['asset_sha256']

    def test_axis_roles_require_unit_vectors_and_preserve_per_body_axes_without_promotion(self):
        env, source, _ = fixture();asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        bindings = declared_instance_bindings(env, asset, shared_permission=True, shape_type='shape')
        annotations = dict(button='axis', plate='none', wrap='full')
        with self.assertRaisesRegex(MultitaskProtocolError,'every and only'):
            body_templates_from_asset(asset, bindings, orientation_observability=annotations)
        for vector in ([0.,0.,0.], [2.,0.,0.], [np.nan,0.,1.], [0.,1.]):
            with self.assertRaisesRegex(MultitaskProtocolError,'unit vectors'):
                body_templates_from_asset(asset, bindings, orientation_observability=annotations,
                                          symmetry_axes={'button': vector})
        with self.assertRaisesRegex(MultitaskProtocolError,'every and only'):
            body_templates_from_asset(asset, bindings, orientation_observability=annotations,
                                      symmetry_axes={'button':[1.,0.,0.], 'plate':[0.,0.,1.]})
        templates = {t.role_id:t for t in body_templates_from_asset(asset, bindings,
            orientation_observability=annotations,symmetry_axes={'button':[1.,0.,0.]})}
        assert np.array_equal(templates['button'].symmetry_axis,[1.,0.,0.])
        assert {name:t.orientation_observability for name,t in templates.items()} == annotations

    def test_native_flags_split_visual_and_collision_preserve_ownership_without_name_heuristics(self):
        env, source, parts = fixture();button = parts[0]
        physical = Shape('boundary_but_physical', 31, [.3, 0., .8], flags=(False, False, True))
        visual = Shape('render_only', 32, [.3, 0., .8], flags=(True, False, False))
        neither = Shape('ordinary_helper', 33, [.3, 0., .8], flags=(False, False, False))
        neither.forbid_mesh = True
        button.children += (physical, visual, neither)
        asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        body = asset['bodies']['button']
        assert asset['version'] == 'r10_shared_native_TRAIN_mesh_v2'
        assert body['visual_component_names'] == ['push_button_target', 'render_only']
        assert body['collision_component_names'] == ['boundary_but_physical', 'push_button_target']
        assert len(body['owned_component_annotations']) == 4 and len(body['collision_vertices_body']) == 6
        assert len(body['components']) == 2
        env.development_split = 'validation'
        for part in (*parts, physical, visual, neither):
            part.forbid_pose = True;part.forbid_flags = True;part.handle += 100
        bindings = declared_instance_bindings(env, asset, shared_permission=True, shape_type='shape')
        assert bindings['button'] == {'push_button_target': (121,), 'render_only': (132,)}
        neither.name = 'topology_changed_even_though_neither'
        with self.assertRaises(MultitaskProtocolError):
            declared_instance_bindings(env, asset, shared_permission=True, shape_type='shape')

    def test_empty_collision_is_explicit_train_declaration_and_legacy_asset_rejected(self):
        env, source, parts = fixture();parts[2].flags = (True, False, False)
        asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        assert asset['bodies']['wrap']['empty_collision_role'] is True
        assert asset['bodies']['wrap']['collision_vertices_body'] == []
        calls, declarations = [], []
        service = SimpleNamespace(register_estimated_geometry=lambda *args, **kwargs:calls.append((args, kwargs)))
        with self.assertRaisesRegex(MultitaskProtocolError,'explicitly accept'):
            register_collision_asset(service, asset,orientation_observability={name:'none' for name in asset['bodies']})
        service.declare_noncolliding_geometry = lambda *args, **kwargs:declarations.append((args, kwargs))
        calls.clear();register_collision_asset(service, asset,orientation_observability={name:'none' for name in asset['bodies']})
        assert {row[0][0] for row in calls} == {'button', 'plate'}
        assert len(declarations)==1 and declarations[0][0]==('wrap',)
        receipt=declarations[0][1]['collision_geometry_receipt']
        assert receipt['collision_geometry_kind']=='declared_noncolliding'
        assert receipt['registered_mesh_sha256'] is None and receipt['vertex_count']==0
        legacy = deepcopy(asset);legacy['version'] = 'r10_shared_native_TRAIN_mesh_v1'
        with self.assertRaises(MultitaskProtocolError):
            declared_instance_bindings(env, legacy, shared_permission=True, shape_type='shape')

    def test_collision_annotations_are_required_complete_and_do_not_modify_original_asset(self):
        env,source,_=fixture();asset=export_train_geometry(env,source,surface_samples=16,shape_type='shape')
        before=deepcopy(asset);calls=[]
        service=SimpleNamespace(register_estimated_geometry=lambda *a,**k:calls.append((a,k)))
        with self.assertRaises(TypeError):register_collision_asset(service,asset)
        for flags in ({},dict(button='none',plate='none',wrap='unverified')):
            with self.assertRaises(MultitaskProtocolError):
                register_collision_asset(service,asset,orientation_observability=flags)
        result=register_collision_asset(service,asset,orientation_observability=dict(button='full',plate='axis',wrap='none'))
        assert asset==before
        assert result['button']['collision_geometry_kind']=='exact_TRAIN_triangles'
        assert result['plate']['collision_geometry_kind']==result['wrap']['collision_geometry_kind']=='rotation_uncertainty_cube'
        assert result['plate']['cavities_preserved'] is False
        assert 'fallback_mesh' in next(k for a,k in calls if a[0]=='button')
        assert all('fallback_mesh' not in k for a,k in calls if a[0] in ('plate','wrap'))

    def test_native_nonrenderable_role_keeps_identity_without_fake_visible_surface(self):
        env, source, parts = fixture();parts[1].flags = (False, True, False)
        asset = export_train_geometry(env, source, surface_samples=16, shape_type='shape')
        body = asset['bodies']['plate']
        assert body['empty_visual_role'] is True and body['components'] == []
        assert body['collision_component_names'] == ['target_button_topPlate']
        bindings = declared_instance_bindings(env, asset, shared_permission=True, shape_type='shape')
        assert bindings['plate'] == {}
        templates = {t.role_id:t for t in body_templates_from_asset(asset, bindings,
            orientation_observability={name:'none' for name in asset['bodies']})}
        assert templates['plate'].components == ()
        assert templates['plate'].visual_geometry_status == 'native_nonrenderable'


if __name__ == '__main__':
    unittest.main()
