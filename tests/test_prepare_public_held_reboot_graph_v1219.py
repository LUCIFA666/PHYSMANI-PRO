"""Synthetic control freeze and physical equality checks; no real roster reads."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import prepare_public_held_reboot_graph_v1219_r10 as subject
from tests.test_repaired_policy_graph_v1143 import graph_spec, put


def metadata(root, name, value):
    return put(root, name, json.dumps(value, sort_keys=True).encode())


class RebootGraphContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = graph_spec(self.root)
        tool = put(self.root, 'control_tools/taskset', b'synthetic taskset')
        spec['external_roots']['host_tools'] = str(self.root / 'control_tools')
        self.old_ref = subject.graphs.write_policy_graph(self.root, spec,
            namespace='artifacts/framework/multitask_r10/previous')
        self.old = subject.graphs.read_policy_graph(self.root, self.old_ref)
        for name in dict.fromkeys((*subject.NEW_SOURCES, *subject.NEW_TESTS, subject.PROBE_SOURCE)):
            put(self.root, name, b'def main():\n    return None\n')
        affinity = dict(schema='r10_parallel_repaired_CPU_affinity_v1212',
            allowed_logical_cpus=list(range(32)), physical_core_count=16,
            core_groups=[dict(package=0, core=i, cpus=[i, i+16], L3_bytes=1024) for i in range(16)],
            native_slot_cpus=[[i, i+16] for i in range(12)],
            control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31], max_native_slots=12,
            taskset_root=str(self.root/'control_tools'),
            taskset_binary=dict(root='host_tools', path='taskset', sha256=tool['sha256']),
            taskset_runs_outside820_as_control_tool=True, bitwise_equivalence_claimed=False)
        metadata(self.root, subject.AFFINITY_NAME, affinity)
        self.profile = metadata(self.root, subject.INPUT_PREFIX+'parallel_resource_profile.json',
            dict(hard_limits=subject.HARD_LIMITS, target_utilization=.9,
                initial_limits=dict(native=2, GPU=1, CPU=6), automatic_retry=False, source_outcome_filtering=False))
        self.inherited = {'src/factory.py': subject.pointer(self.root, 'src/factory.py')['sha256']}
        self.inherited_ref = metadata(self.root, subject.INPUT_PREFIX+'inherited/control_CPU_evidence.json',
            dict(complete=True, actual_tool_exit_observed=True, returncode=0, native_started=False, GPU_started=False,
                source_hashes_before=self.inherited, source_hashes_after=self.inherited,
                inherited_source_hashes_before={}, inherited_source_hashes_after={}))
        for key, value in (('OLD_GRAPH', self.old_ref), ('PHYSICAL_SHA', self.old['physical_policy_sha256']),
            ('INHERITED_CPU_EVIDENCE', self.inherited_ref)):
            manager = patch.object(subject, key, value); manager.start(); self.addCleanup(manager.stop)

    def evidence(self, **changes):
        inventory = subject.source_inventory(self.root)
        hashes = {name: item['source']['sha256'] for name, item in inventory.items()}
        probe = metadata(self.root, subject.INPUT_PREFIX+'parent_lock_namespace_probe.json',
            dict(schema='r10_pure_OS_parent_lock_broker_probe_v1213', actual_tool_exit_observed=True,
                complete=True, returncode=0, checks={k: True for k in subject.PROBE_CHECKS},
                Native_started=False, GPU_started=False, global_permissions_changed=False,
                test_only=True, scientific_admission_granted=False,
                probe_source=subject.pointer(self.root, subject.PROBE_SOURCE),
                guard_source=subject.pointer(self.root, subject.COMPONENTS['resource_guard']),
                frozen_environment_manifest=self.old['physical']['environment_manifest']))
        value = dict(complete=True, actual_tool_exit_observed=True, returncode=0, tests_run=1, skipped=0,
            native_started=False, GPU_started=False, namespace_probe=probe,
            source_hashes_before=hashes, source_hashes_after=hashes,
            inherited_source_hashes_before=self.inherited, inherited_source_hashes_after=self.inherited,
            inherited_CPU_evidence=self.inherited_ref)
        value.update(changes)
        return metadata(self.root, subject.INPUT_PREFIX+'control_CPU_evidence.json', value)

    def test_default_inert_and_wrong_input_domain_rejected_before_reads(self):
        with patch.object(subject, 'control_spec', side_effect=AssertionError('must remain inert')):
            self.assertTrue(subject.main([])['default_inert'])
        for name in ('runtime/TEST.json', '/tmp/control_CPU_evidence.json', subject.INPUT_PREFIX+'../control_CPU_evidence.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                subject.typed_input(name, 'control_CPU_evidence.json')

    def test_new_control_changes_no_physics_model_roster_or_consumer(self):
        spec, _, old = subject.control_spec(self.root, profile_ref=self.profile)
        new = subject.graphs.compile_policy_graph(self.root, spec)
        self.assertTrue(subject.require_physical_equivalence(old, new))
        self.assertNotEqual(old['control_graph_sha256'], new['control_graph_sha256'])
        self.assertEqual(old['physical_policy_sha256'], new['physical_policy_sha256'])
        self.assertEqual(new['routes'][0]['train']['worker']['path'], subject.WORKER)

    def test_model_physics_roster_consumer_or_missing_route_rejected(self):
        for label in ('physical', 'model', 'manifest', 'consumer', 'route'):
            changed = deepcopy(self.old)
            if label == 'physical': changed['physical']['complete_source_closure_declared'] = False
            elif label == 'model': changed['routes'][0]['models'] = {}
            elif label == 'manifest': changed['train_source_manifest']['sha256'] = 'f'*64
            elif label == 'consumer': changed['routes'][0]['train']['consumer']['sha256'] = 'f'*64
            else: changed['routes'].pop()
            with self.subTest(label=label), self.assertRaises(ValueError):
                subject.require_physical_equivalence(self.old, changed)

    def test_failed_boolean_exit_skip_or_source_drift_cannot_freeze(self):
        for invalid in (dict(returncode=False), dict(returncode=1), dict(skipped=False), dict(skipped=1),
            dict(actual_tool_exit_observed=False), dict(native_started=True), dict(GPU_started=True),
            dict(source_hashes_after={}), dict(inherited_source_hashes_after={}), dict(inherited_CPU_evidence={})):
            ref = self.evidence(**invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError), \
                patch.object(subject.graphs, 'write_policy_graph', side_effect=AssertionError('must not freeze')):
                subject.prepare_parallel_graph(self.root, profile_ref=self.profile, cpu_evidence_ref=ref,
                    namespace='artifacts/framework/multitask_r10/refused', reviewed=True)

    def test_full_source_cpu_probe_freeze_roundtrip_and_no_overwrite(self):
        ref = self.evidence()
        result = subject.prepare_parallel_graph(self.root, profile_ref=self.profile, cpu_evidence_ref=ref,
            namespace='artifacts/framework/multitask_r10/recovered', reviewed=True)
        new = subject.graphs.read_policy_graph(self.root, result['graph'])
        self.assertTrue(subject.require_physical_equivalence(self.old, new))
        receipt, _ = subject.read(self.root, result['receipt'])
        self.assertFalse(receipt['VAL_read']); self.assertFalse(receipt['TEST_read'])
        with self.assertRaises((ValueError, FileExistsError)):
            subject.prepare_parallel_graph(self.root, profile_ref=self.profile, cpu_evidence_ref=ref,
                namespace='artifacts/framework/multitask_r10/recovered', reviewed=True)

    def test_missing_control_and_drifting_inherited_bytes_rejected(self):
        (self.root / subject.PREPARER).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            subject.source_inventory(self.root)
        (self.root / 'src/factory.py').write_bytes(b'changed source')
        with self.assertRaises(ValueError):
            subject.inherited_CPU_sources(self.root, self.old)


if __name__ == '__main__':
    unittest.main()
