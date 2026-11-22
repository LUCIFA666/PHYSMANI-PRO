"""CPU-only new-control graph preparation; all inputs are temporary fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_public_held_repaired_graph_v1212_r10 as subject
from tests.test_repaired_policy_graph_v1143 import graph_spec, put


def metadata(root, name, value):
    return put(root, name, json.dumps(value, sort_keys=True).encode())


class PublicHeldGraphContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = graph_spec(self.root)
        selected = spec['physical']['factories'].pop('selected')
        spec['physical']['factories'] = dict(core_geometry=deepcopy(selected),
            block_geometry=dict(source=selected['source'], callable='integrated_block_components'),
            physmani=dict(source=spec['physical']['sources_map']['consumer'], callable='_baseline'))
        for row in spec['routes']:
            branch = 'physmani' if row['method'] == 'physmani' else (
                'block_geometry' if row['task'].startswith('pick_moving_target') else 'core_geometry')
            row['physical_factory'] = branch
            row['train']['physical_factory'] = row['evaluation']['physical_factory'] = branch
        self.old_ref = subject.graphs.write_policy_graph(self.root, spec,
            namespace='artifacts/framework/multitask_r10/old_fixture_graph')
        self.old = subject.graphs.read_policy_graph(self.root, self.old_ref)
        for name in (*subject.NEW_SOURCES, *subject.NEW_TESTS):
            put(self.root, name, b'def main():\n    return None\n')
        put(self.root, subject.PHYSICAL_FACTORY, b'def integrated_block_components(*args, **kwargs):\n    return None\n')
        self.factory_ref = subject.pointer(self.root, subject.PHYSICAL_FACTORY)
        core = self.old['physical']['factories']['core_geometry']
        self.overlay = dict(integration_complete=True,
            files={subject.PHYSICAL_FACTORY: self.factory_ref['sha256']},
            core_factory=dict(source=core['source']['path'], sha256=core['source']['sha256'], function=core['callable']),
            block_factory=dict(source=subject.PHYSICAL_FACTORY, sha256=self.factory_ref['sha256'],
                               function='integrated_block_components'))
        self.overlay_ref = metadata(self.root,
            'artifacts/framework/multitask_r10/physical_fixture/source_overlay_lock.json', self.overlay)
        self.development_ref = metadata(self.root,
            'artifacts/framework/multitask_r10/development_fixture/source_overlay_lock.json',
            dict(schema='r10_fixed_TRAIN_public_held_block_development_v1204_source_lock',
                 integration_complete=True,
                 factory=dict(source=subject.PHYSICAL_FACTORY, sha256=self.factory_ref['sha256']),
                 files={name: subject.pointer(self.root, name)['sha256'] for name in
                        (subject.PHYSICAL_FACTORY, subject.DEVELOPMENT_MODULE, subject.DEVELOPMENT_WORKER)}))
        factory_registry = {key: (value['source']['path'], value['callable'])
                            for key, value in self.old['physical']['factories'].items()}
        factory_registry['block_geometry'] = (subject.PHYSICAL_FACTORY, 'integrated_block_components')
        self.fake_worker = SimpleNamespace(__file__=str(self.root/subject.PREPARER),
            PHYSICAL_SOURCE_OVERLAY=self.overlay_ref, FACTORIES=factory_registry,
            CASE_SCHEMAS=dict(train=subject.CASE_SCHEMA+'_TRAIN_case',
                              validation=subject.CASE_SCHEMA+'_validation_case'))
        overlay_module = SimpleNamespace(read_overlay_lock=lambda root, reference: (deepcopy(self.overlay), self.overlay_ref))
        def imports(name):
            if name == subject.PHYSICAL_OVERLAY_MODULE: return overlay_module
            if name == subject.PREPARER[:-3].replace('/', '.'): return self.fake_worker
            raise AssertionError('unexpected import: '+name)
        patcher = patch.object(subject.importlib, 'import_module', side_effect=imports)
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(subject, 'PHYSICAL_FACTORY_SHA', self.factory_ref['sha256'])
        patcher.start(); self.addCleanup(patcher.stop)
        self.profile = metadata(self.root, subject.INPUT_PREFIX + 'parallel_resource_profile.json', dict(
            hard_limits=subject.HARD_LIMITS, target_utilization=.9,
            initial_limits=dict(native=2, GPU=1, CPU=6), automatic_retry=False,
            source_outcome_filtering=False))
        tool = put(self.root, 'control_tools/taskset', b'synthetic taskset control binary')
        self.affinity = dict(schema='r10_parallel_repaired_CPU_affinity_v1212',
            allowed_logical_cpus=list(range(32)), physical_core_count=16,
            core_groups=[dict(package=0, core=i, cpus=[i, i+16], L3_bytes=1024) for i in range(16)],
            native_slot_cpus=[[i, i+16] for i in range(12)],
            control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31], max_native_slots=12,
            taskset_root=str(self.root/'control_tools'),
            taskset_binary=dict(root='host_tools', path='taskset', sha256=tool['sha256']),
            taskset_runs_outside820_as_control_tool=True, bitwise_equivalence_claimed=False)
        metadata(self.root, subject.AFFINITY_NAME, self.affinity)
        name = str(self.root/'src/factory.py')
        self.inherited_map = {name: subject.pointer(self.root, 'src/factory.py')['sha256']}
        self.inherited_CPU_ref = metadata(self.root,
            subject.INPUT_PREFIX+'old_cpu/control_CPU_evidence.json',
            dict(complete=True, actual_tool_exit_observed=True, returncode=0,
                 native_started=False, GPU_started=False,
                 source_hashes_before=self.inherited_map, source_hashes_after=self.inherited_map,
                 inherited_source_hashes_before={}, inherited_source_hashes_after={}))
        self.stack = []
        for name, value in (('OLD_GRAPH', self.old_ref), ('PREDECESSOR_PHYSICAL_SHA', self.old['physical_policy_sha256']),
                            ('INHERITED_CPU_EVIDENCE', self.inherited_CPU_ref),
                            ('DEVELOPMENT_SOURCE_LOCK', self.development_ref)):
            patcher = patch.object(subject, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def evidence(self, **changes):
        inventory = subject.source_inventory(self.root)
        hashes = {name: row['source']['sha256'] for name, row in inventory.items()}
        probe = metadata(self.root, subject.INPUT_PREFIX + 'parent_lock_namespace_probe.json', dict(
            schema='r10_pure_OS_parent_lock_broker_probe_v1213', actual_tool_exit_observed=True,
            complete=True, returncode=0, checks={key: True for key in subject.PROBE_CHECKS},
            Native_started=False, GPU_started=False, global_permissions_changed=False,
            scientific_admission_granted=False, test_only=True,
            probe_source=subject.pointer(self.root, subject.PROBE_SOURCE),
            guard_source=subject.pointer(self.root, subject.COMPONENTS['resource_guard']),
            frozen_environment_manifest=self.old['physical']['environment_manifest']))
        record = dict(actual_tool_exit_observed=True, complete=True, returncode=0,
            tests_run=42, skipped=0, native_started=False, GPU_started=False,
            source_hashes_before=hashes, source_hashes_after=hashes, namespace_probe=probe,
            inherited_source_hashes_before=self.inherited_map, inherited_source_hashes_after=self.inherited_map,
            inherited_CPU_evidence=self.inherited_CPU_ref)
        record.update(changes)
        return metadata(self.root, subject.INPUT_PREFIX + 'control_CPU_evidence.json', record)

    def test_default_is_inert_without_reading_inputs(self):
        with patch.object(subject, 'control_spec', side_effect=AssertionError('no preparation')):
            result = subject.main([])
        self.assertTrue(result['default_inert'])
        self.assertFalse(result['execution_started'])

    def test_physical_and_all_nonworker_route_fields_remain_identical(self):
        spec, inventory, old = subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)
        new = subject.graphs.compile_policy_graph(self.root, spec)
        subject.require_expected_physical_change(old, new)
        self.assertNotEqual(new['physical'], old['physical'])
        self.assertNotEqual(new['physical_policy_sha256'], old['physical_policy_sha256'])
        self.assertNotEqual(new['control_graph_sha256'], old['control_graph_sha256'])
        self.assertEqual(len(new['routes']), 48)
        for current, previous in zip(new['routes'], old['routes']):
            self.assertEqual({k: v for k, v in current.items() if k not in ('train', 'evaluation')},
                             {k: v for k, v in previous.items() if k not in ('train', 'evaluation')})
            self.assertEqual(current['train']['worker']['path'], subject.WORKER)
            self.assertEqual(current['evaluation']['preparer']['path'], subject.PREPARER)
        self.assertEqual(set(inventory), set((*subject.NEW_SOURCES, *subject.NEW_TESTS)))

    def test_actual_cpu_failure_instability_or_missing_observation_cannot_freeze(self):
        for changes in (dict(returncode=1), dict(returncode=False), dict(skipped=1), dict(skipped=False),
                        dict(actual_tool_exit_observed=False), dict(native_started=True),
                        dict(source_hashes_after={}), dict(source_hashes_before={}),
                        dict(inherited_source_hashes_before={}), dict(inherited_source_hashes_after={}),
                        dict(inherited_CPU_evidence={})):
            ref = self.evidence(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError), \
                    patch.object(subject.graphs, 'write_policy_graph', side_effect=AssertionError('must not write')):
                subject.prepare_parallel_graph(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref, cpu_evidence_ref=ref,
                    namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)

    def test_missing_source_and_invalid_capacity_are_rejected(self):
        broken = deepcopy(self.profile)
        value, _ = subject.read(self.root, self.profile)
        for field, invalid in (('hard_limits', dict(subject.HARD_LIMITS, max_GPU_slots=30)),
                               ('initial_limits', dict(native=2, GPU=True, CPU=2)),
                               ('source_outcome_filtering', True)):
            changed = dict(value, **{field: invalid})
            ref = metadata(self.root, subject.INPUT_PREFIX + 'invalid/parallel_resource_profile.json', changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.control_spec(self.root, profile_ref=ref, physical_overlay_ref=self.overlay_ref)
        (self.root / subject.NEW_SOURCES[0]).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            subject.control_spec(self.root, profile_ref=broken, physical_overlay_ref=self.overlay_ref)

    def test_namespace_probe_observation_control_bytes_and_rejections_are_required(self):
        for change in (dict(actual_tool_exit_observed=False), dict(returncode=True),
                       dict(complete=False), dict(checks={}), dict(Native_started=True),
                       dict(probe_source={}), dict(guard_source={}), dict(frozen_environment_manifest={})):
            evidence_ref = self.evidence()
            evidence, _ = subject.read(self.root, evidence_ref)
            probe, _ = subject.read(self.root, evidence['namespace_probe'])
            probe.update(change)
            evidence['namespace_probe'] = metadata(self.root,
                subject.INPUT_PREFIX + 'parent_lock_namespace_probe.json', probe)
            evidence_ref = metadata(self.root, subject.INPUT_PREFIX + 'control_CPU_evidence.json', evidence)
            with self.subTest(change=change), self.assertRaises(ValueError), \
                    patch.object(subject.graphs, 'write_policy_graph', side_effect=AssertionError('no freeze')):
                subject.prepare_parallel_graph(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref, cpu_evidence_ref=evidence_ref,
                    namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)

    def test_valid_freeze_is_firstwrite_and_grants_no_execution_or_readiness(self):
        result = subject.prepare_parallel_graph(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref,
            cpu_evidence_ref=self.evidence(), namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)
        graph = subject.graphs.read_policy_graph(self.root, result['graph'])
        receipt, _ = subject.read(self.root, result['receipt'])
        self.assertNotEqual(graph['physical_policy_sha256'], self.old['physical_policy_sha256'])
        self.assertFalse(receipt['physical_declarations_byte_identical'])
        self.assertTrue(receipt['new_public_held_block_factory'])
        self.assertFalse(receipt['execution_started'])
        self.assertFalse(receipt['readiness_granted'])
        self.assertFalse(receipt['VAL_read'])
        with self.assertRaises((ValueError, FileExistsError)):
            subject.prepare_parallel_graph(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref,
                cpu_evidence_ref=self.evidence(), namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)

    def test_wrong_role_or_VAL_TEST_paths_are_rejected_before_any_read(self):
        names = ('artifacts/framework/protocol/validation_manifest.json',
                 'artifacts/framework/protocol/test_manifest.json',
                 'artifacts/framework/multitask_r10/other_test/case.json')
        for name in names:
            ref = dict(root='project', path=name, sha256='a' * 64)
            with self.subTest(name=name), patch.object(subject.graphs, 'read_project_json',
                    side_effect=AssertionError('source must not be parsed')):
                with self.assertRaises(ValueError):subject.control_spec(self.root, profile_ref=ref, physical_overlay_ref=self.overlay_ref)
                with self.assertRaises(ValueError):subject.prepare_parallel_graph(self.root,
                    profile_ref=self.profile, physical_overlay_ref=self.overlay_ref, cpu_evidence_ref=ref,
                    namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)
                with self.assertRaises(ValueError):subject.main(['--prepare', '--profile', name,
                    '--cpu-evidence', name, '--namespace', 'new', '--reviewed'])

    def test_core_partitions_cannot_overlap_and_outer_tool_bytes_are_pinned(self):
        broken = deepcopy(self.affinity)
        broken['control_and_CPU_preflight_cpus'].append(0)
        with self.assertRaises(ValueError):subject.validate_CPU_affinity(broken)
        spec, _, _ = subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)
        (self.root/'control_tools/taskset').write_bytes(b'changed tool')
        with self.assertRaises(ValueError):subject.graphs.compile_policy_graph(self.root, spec)

    def test_inherited_host_tool_root_cannot_be_retargeted(self):
        old = deepcopy(self.old)
        old['roots']['host_tools'] = self.affinity['taskset_root']
        with patch.object(subject.graphs, 'read_policy_graph', return_value=old):
            spec, _, _ = subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)
        self.assertEqual(spec['external_roots']['host_tools'], self.affinity['taskset_root'])
        old['roots']['host_tools'] = str(self.root/'different_tools')
        with patch.object(subject.graphs, 'read_policy_graph', return_value=old), self.assertRaises(ValueError):
            subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)


    def test_independent_physical_CPU_factory_pin_and_registry_are_required(self):
        for changes in (dict(integration_complete=False),
                        dict(core_factory={}), dict(block_factory={}),
                        dict(files={subject.PHYSICAL_FACTORY: '0'*64})):
            previous = deepcopy(self.overlay)
            self.overlay.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)
            self.overlay = previous
        self.fake_worker.PHYSICAL_SOURCE_OVERLAY = dict(path='other.json', sha256='0'*64)
        with self.assertRaises(ValueError):
            subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)

    def test_new_physical_graph_cannot_change_shared_rules_or_remove_frozen_sources(self):
        spec, _, old = subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)
        new = subject.graphs.compile_policy_graph(self.root, spec)
        for key in ('environment_manifest', 'shared_context_checkpoint', 'common_RML', 'outcome_sampling'):
            bad = deepcopy(new); bad['physical'][key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.require_expected_physical_change(old, bad)
        bad = deepcopy(new)
        bad['physical']['sources_map'].pop(next(iter(old['physical']['sources_map'])))
        with self.assertRaises(ValueError): subject.require_expected_physical_change(old, bad)

    def test_development_validator_runtime_dependencies_must_be_in_the_graph(self):
        spec, _, _ = subject.control_spec(self.root, profile_ref=self.profile, physical_overlay_ref=self.overlay_ref)
        ref = subject.pointer(self.root, subject.DEVELOPMENT_MODULE)
        spec['control']['sources_map'] = {key: value for key, value in spec['control']['sources_map'].items()
                                          if value != ref}
        with self.assertRaisesRegex(ValueError, 'validator dependency'):
            subject.verify_development_validator_sources(self.root, spec)

if __name__ == '__main__':
    unittest.main()
