"""No native simulator imports or mesh/role name heuristics."""
import unittest
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import numpy as np

from our_mani.multitask.component_flags import (
    component_roles, partition_components, bind_frozen_visual_components,
    read_native_component_flags,
)


def record(name, renderable, collidable, respondable):
    return dict(name=name, flags=dict(renderable=renderable, collidable=collidable,
                                    respondable=respondable))


class ComponentFlagChecks(unittest.TestCase):
    def test_four_categories_preserve_invisible_physics_and_visual_only(self):
        rows = (record('boundary_like_but_real', False, False, True),
                record('visual', True, False, False), record('body', True, True, False),
                record('ordinary_name', False, False, False))
        p = partition_components(rows)
        self.assertEqual(p['groups'], dict(visual_only=['visual'],
            collision_only=['boundary_like_but_real'], both=['body'], neither=['ordinary_name']))
        self.assertEqual(p['collision_component_names'], ['body', 'boundary_like_but_real'])
        self.assertEqual(p['visual_component_names'], ['body', 'visual'])
        self.assertFalse(p['classification_uses_component_names'])

    def test_real_pyrep_getters_accept_native_respondable_int(self):
        calls = []
        def getter(name, value):
            def read(): calls.append(name); return value
            return read
        s = SimpleNamespace(is_renderable=getter('r', False), is_collidable=getter('c', False),
                            is_respondable=getter('p', 1))
        self.assertEqual(read_native_component_flags(s)['category'], 'collision_only')
        self.assertEqual(calls, ['r', 'c', 'p'])

    def test_native_renderable_true_is_bit512_and_raw_value_preserved(self):
        shape = SimpleNamespace(is_renderable=lambda: 512, is_collidable=lambda: 1,
                                is_respondable=lambda: 0)
        result = read_native_component_flags(shape)
        self.assertEqual(result['flags'], dict(renderable=True, collidable=True, respondable=False))
        self.assertEqual(result['native_getter_values'], dict(renderable=512, collidable=1, respondable=0))
        shape.is_renderable = lambda: 2
        with self.assertRaises(ValueError): read_native_component_flags(shape)

    def test_missing_or_nonboolean_flags_fail_closed(self):
        for flags in ({}, dict(renderable=True, collidable=False),
                      dict(renderable='False', collidable=False, respondable=False),
                      dict(renderable=False, collidable=2, respondable=False)):
            with self.subTest(flags=flags), self.assertRaises(ValueError): component_roles(flags)
        with self.assertRaises(ValueError): partition_components([])
        with self.assertRaises(ValueError): partition_components([record('a', True, False, False)] * 2)

    def test_frozen_visual_names_not_runtime_flag_or_name_guess(self):
        rows = [record('invisible_phys', False, True, False),
                record('boundary', True, False, False), record('helper', False, False, False)]
        mapping = dict(invisible_phys=(10,), boundary=(20,), helper=(30,))
        self.assertEqual(bind_frozen_visual_components(rows, mapping), {'boundary': (20,)})
        with self.assertRaises(ValueError): bind_frozen_visual_components(rows, {**mapping, 'new': (40,)})
        with self.assertRaises(ValueError): bind_frozen_visual_components(rows, {'boundary': (20,)})

    def test_empty_role_risks_are_explicit_without_fake_geometry(self):
        p = partition_components([record('helper', False, False, False)])
        self.assertTrue(p['empty_visual_role']); self.assertTrue(p['empty_collision_role'])
        self.assertEqual(p['visual_component_names'], []); self.assertEqual(p['collision_component_names'], [])

    def test_audit_scope_and_raw_mesh_statistics_are_cpu_only(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/audit_train_component_flags_r10.py'
        spec = importlib.util.spec_from_file_location('component_audit_cpu_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(len(module.ordinary_tasks()), 7)
        stats = module.mesh_statistics([[0,0,0], [1,0,0], [0,1,0]], [[0,1,2]])
        self.assertEqual(stats['triangle_count'], 1); self.assertEqual(stats['surface_area_m2'], .5)
        self.assertFalse(stats['body_or_world_pose_included'])
        with self.assertRaises(ValueError): module.mesh_statistics([[0,0,0]], [[0,1,2]])
        with self.assertRaises(ValueError): module.mesh_statistics([[0,0,0]], [[0.,0.,0.]])
        self.assertIn(module.os.getpid(), module.process_ancestry())


if __name__ == '__main__': unittest.main()
