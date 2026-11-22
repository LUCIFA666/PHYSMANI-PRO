"""CPU-only new-control graph preparation; all inputs are temporary fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import prepare_parallel_repaired_graph_v1186_r10 as subject
from tests.test_repaired_policy_graph_v1143 import graph_spec, put


def metadata(root, name, value):
    return put(root, name, json.dumps(value, sort_keys=True).encode())


class ParallelGraphContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = graph_spec(self.root)
        self.old_ref = subject.graphs.write_policy_graph(self.root, spec,
            namespace='artifacts/framework/multitask_r10/old_fixture_graph')
        self.old = subject.graphs.read_policy_graph(self.root, self.old_ref)
        for name in (*subject.NEW_SOURCES, *subject.NEW_TESTS):
            put(self.root, name, b'def main():\n    return None\n')
        self.profile = metadata(self.root, subject.INPUT_PREFIX + 'parallel_resource_profile.json', dict(
            hard_limits=subject.HARD_LIMITS, target_utilization=.9,
            initial_limits=dict(native=2, GPU=1, CPU=2), automatic_retry=False,
            source_outcome_filtering=False))
        tool = put(self.root, 'control_tools/taskset', b'synthetic taskset control binary')
        self.affinity = dict(schema='r10_parallel_repaired_CPU_affinity_v1186',
            allowed_logical_cpus=list(range(32)), physical_core_count=16,
            core_groups=[dict(package=0, core=i, cpus=[i, i+16], L3_bytes=1024) for i in range(16)],
            native_slot_cpus=[[i, i+16] for i in range(12)],
            control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31], max_native_slots=12,
            taskset_root=str(self.root/'control_tools'),
            taskset_binary=dict(root='host_tools', path='taskset', sha256=tool['sha256']),
            taskset_runs_outside820_as_control_tool=True, bitwise_equivalence_claimed=False)
        metadata(self.root, subject.AFFINITY_NAME, self.affinity)
        self.stack = []
        for name, value in (('OLD_GRAPH', self.old_ref), ('PHYSICAL_SHA', self.old['physical_policy_sha256'])):
            patcher = patch.object(subject, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def evidence(self, **changes):
        inventory = subject.source_inventory(self.root)
        hashes = {name: row['source']['sha256'] for name, row in inventory.items()}
        probe = metadata(self.root, subject.INPUT_PREFIX + 'parent_lock_namespace_probe.json', dict(
            schema='r10_pure_OS_parent_lock_broker_probe_v1179', actual_tool_exit_observed=True,
            complete=True, returncode=0, checks={key: True for key in subject.PROBE_CHECKS},
            Native_started=False, GPU_started=False, global_permissions_changed=False,
            scientific_admission_granted=False, test_only=True,
            probe_source=subject.pointer(self.root, subject.PROBE_SOURCE),
            guard_source=subject.pointer(self.root, subject.COMPONENTS['resource_guard']),
            frozen_environment_manifest=self.old['physical']['environment_manifest']))
        record = dict(actual_tool_exit_observed=True, complete=True, returncode=0,
            tests_run=42, skipped=0, native_started=False, GPU_started=False,
            source_hashes_before=hashes, source_hashes_after=hashes, namespace_probe=probe)
        record.update(changes)
        return metadata(self.root, subject.INPUT_PREFIX + 'control_CPU_evidence.json', record)

    def test_default_is_inert_without_reading_inputs(self):
        with patch.object(subject, 'control_spec', side_effect=AssertionError('no preparation')):
            result = subject.main([])
        self.assertTrue(result['default_inert'])
        self.assertFalse(result['execution_started'])

    def test_physical_and_all_nonworker_route_fields_remain_identical(self):
        spec, inventory, old = subject.control_spec(self.root, profile_ref=self.profile)
        new = subject.graphs.compile_policy_graph(self.root, spec)
        self.assertEqual(new['physical'], old['physical'])
        self.assertEqual(new['physical_policy_sha256'], old['physical_policy_sha256'])
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
                        dict(source_hashes_after={}), dict(source_hashes_before={})):
            ref = self.evidence(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError), \
                    patch.object(subject.graphs, 'write_policy_graph', side_effect=AssertionError('must not write')):
                subject.prepare_parallel_graph(self.root, profile_ref=self.profile, cpu_evidence_ref=ref,
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
                subject.control_spec(self.root, profile_ref=ref)
        (self.root / subject.NEW_SOURCES[0]).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            subject.control_spec(self.root, profile_ref=broken)

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
                subject.prepare_parallel_graph(self.root, profile_ref=self.profile, cpu_evidence_ref=evidence_ref,
                    namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)

    def test_valid_freeze_is_firstwrite_and_grants_no_execution_or_readiness(self):
        result = subject.prepare_parallel_graph(self.root, profile_ref=self.profile,
            cpu_evidence_ref=self.evidence(), namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)
        graph = subject.graphs.read_policy_graph(self.root, result['graph'])
        receipt, _ = subject.read(self.root, result['receipt'])
        self.assertEqual(graph['physical_policy_sha256'], self.old['physical_policy_sha256'])
        self.assertTrue(receipt['physical_declarations_byte_identical'])
        self.assertFalse(receipt['execution_started'])
        self.assertFalse(receipt['readiness_granted'])
        self.assertFalse(receipt['VAL_read'])
        with self.assertRaises((ValueError, FileExistsError)):
            subject.prepare_parallel_graph(self.root, profile_ref=self.profile,
                cpu_evidence_ref=self.evidence(), namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)

    def test_wrong_role_or_VAL_TEST_paths_are_rejected_before_any_read(self):
        names = ('artifacts/framework/protocol/validation_manifest.json',
                 'artifacts/framework/protocol/test_manifest.json',
                 'artifacts/framework/multitask_r10/other_test/case.json')
        for name in names:
            ref = dict(root='project', path=name, sha256='a' * 64)
            with self.subTest(name=name), patch.object(subject.graphs, 'read_project_json',
                    side_effect=AssertionError('source must not be parsed')):
                with self.assertRaises(ValueError):subject.control_spec(self.root, profile_ref=ref)
                with self.assertRaises(ValueError):subject.prepare_parallel_graph(self.root,
                    profile_ref=self.profile, cpu_evidence_ref=ref,
                    namespace='artifacts/framework/multitask_r10/new_fixture_graph', reviewed=True)
                with self.assertRaises(ValueError):subject.main(['--prepare', '--profile', name,
                    '--cpu-evidence', name, '--namespace', 'new', '--reviewed'])

    def test_core_partitions_cannot_overlap_and_outer_tool_bytes_are_pinned(self):
        broken = deepcopy(self.affinity)
        broken['control_and_CPU_preflight_cpus'].append(0)
        with self.assertRaises(ValueError):subject.validate_CPU_affinity(broken)
        spec, _, _ = subject.control_spec(self.root, profile_ref=self.profile)
        (self.root/'control_tools/taskset').write_bytes(b'changed tool')
        with self.assertRaises(ValueError):subject.graphs.compile_policy_graph(self.root, spec)


if __name__ == '__main__':
    unittest.main()
