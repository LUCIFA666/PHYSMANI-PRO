"""Synthetic CPU recovery contracts; no actual source manifest or simulator."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts import run_public_held_reboot_owner_v1214_r10 as o
from tests import test_public_held_repaired_owner_v1207 as inherited


def partition_fixture():
    ids=[f'j{i:04d}' for i in range(192)]
    # Deliberately noncontiguous: a continuation may never subtract a prefix.
    accepted=ids[:53]+ids[54:86]
    remaining=[x for x in ids if x not in accepted]
    rejected=['j0100'];native=[x for x in remaining if x not in rejected][:12]
    cpu=[x for x in remaining if x not in rejected+native][:6]
    ready=[x for x in remaining if x not in rejected+native+cpu][:15]
    untouched=[x for x in remaining if x not in rejected+native+cpu+ready]
    records=[dict(job_id=x,native_success=i%2==0,native_terminal_failure=i%5==0) for i,x in enumerate(accepted)]
    saved=sorted(records+[dict(job_id='j0100',native_success=True)],key=lambda r:r['job_id'])
    audit=dict(accepted_job_ids=accepted,remaining_job_ids=remaining,accepted_records=records,
        rejected_saved_completed_job_ids=rejected,interrupted_native_job_ids=native,
        interrupted_CPU_job_ids=cpu,CPU_complete_native_unstarted_job_ids=ready,unstarted_job_ids=untouched)
    status=dict(completed=saved,active=[dict(job_id=x,phase='native') for x in native]+[dict(job_id=x,phase='CPU_check') for x in cpu])
    return audit,dict(jobs=[dict(job_id=x) for x in ids]),status


class RecoveryPartitionTests(unittest.TestCase):
    def test_noncontiguous85_and_all_outcomes_retained(self):
        a,p,s=partition_fixture();self.assertEqual(o.validate_recovery_partition(a,p,s),a['accepted_records'])
        self.assertNotEqual(a['accepted_job_ids'],[f'j{i:04d}' for i in range(85)])
        self.assertTrue(any(not r['native_success'] for r in a['accepted_records']))

    def test_success_based_dropping_rejected(self):
        a,p,s=partition_fixture();a['accepted_records']=[r for r in a['accepted_records'] if r['native_success']]
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_accepted_outcome_rewrite_rejected(self):
        a,p,s=partition_fixture();a=deepcopy(a);a['accepted_records'][1]['native_success']=True
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_missing_complement_or_padding_rejected(self):
        for change in ('missing','duplicate','accepted'):
            a,p,s=partition_fixture()
            if change=='missing':a['remaining_job_ids'].pop()
            elif change=='duplicate':a['remaining_job_ids'][-1]=a['remaining_job_ids'][0]
            else:a['remaining_job_ids'][-1]=a['accepted_job_ids'][0]
            with self.subTest(change=change),self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_reordering_sources_rejected(self):
        a,p,s=partition_fixture();p['jobs'][0],p['jobs'][1]=p['jobs'][1],p['jobs'][0]
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_damaged_saved_true_is_not_silently_accepted(self):
        a,p,s=partition_fixture();a['rejected_saved_completed_job_ids']=[]
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_partial_native_not_inferred_as_false(self):
        a,p,s=partition_fixture();s['completed'].append(dict(job_id=a['interrupted_native_job_ids'][0],native_success=False))
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_active_native_CPU_phase_mismatch_rejected(self):
        a,p,s=partition_fixture();s['active'][0]['phase']='CPU_check'
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)

    def test_overlapping_ready_and_interrupted_partition_rejected(self):
        a,p,s=partition_fixture();a['CPU_complete_native_unstarted_job_ids'][0]=a['interrupted_native_job_ids'][0]
        with self.assertRaises(ValueError):o.validate_recovery_partition(a,p,s)


class DurableMetadataTests(unittest.TestCase):
    def test_fresh_flushes_file_and_directory_and_never_overwrites(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'terminal.json';real_fsync=os.fsync
            with patch.object(o.os,'fsync',wraps=real_fsync) as sync:o.fresh(p,dict(returncode=0))
            self.assertEqual(json.loads(p.read_text()),dict(returncode=0));self.assertEqual(sync.call_count,2)
            with self.assertRaises(FileExistsError):o.fresh(p,dict(returncode=1))
            self.assertEqual(json.loads(p.read_text()),dict(returncode=0))

    def test_atomic_persists_rename_and_leaves_no_temporary(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'status.json';o.fresh(p,{'complete':False});real_fsync=os.fsync
            with patch.object(o.os,'fsync',wraps=real_fsync) as sync:o.atomic(p,{'complete':True})
            self.assertEqual(json.loads(p.read_text()),{'complete':True});self.assertEqual(sync.call_count,3)
            self.assertEqual([x.name for x in p.parent.iterdir()],['status.json'])

    def test_fsync_failure_is_an_error_not_successful_evidence(self):
        with TemporaryDirectory() as d,patch.object(o.os,'fsync',side_effect=OSError('synthetic durability failure')):
            with self.assertRaises(OSError):o.fresh(Path(d)/'terminal.json',{'returncode':0})


class PhysicalEqualityTests(unittest.TestCase):
    def fixture(self):
        old=dict(physical_policy_sha256=o.PHYSICAL_SHA,physical=dict(factory={'source':'frozen'}),roots={'project':'/synthetic'},
            routes=[dict(task=str(i//3),method=str(i%3),configuration={'sha256':'a'},train={'worker':'old','consumer':'same'},
                evaluation={'preparer':'old','consumer':'same'}) for i in range(48)])
        return old,deepcopy(old)

    def test_only_control_entry_changes_allowed(self):
        a,b=self.fixture();b['routes'][0]['train']['worker']='new'
        with patch.object(o.graphs,'read_policy_graph',return_value=a):self.assertEqual(o.physical_equivalence(None,None,b),a)

    def test_model_configuration_or_physical_change_rejected(self):
        for name in ('physical','configuration','consumer'):
            a,b=self.fixture()
            if name=='physical':b['physical']['factory']['source']='changed'
            elif name=='configuration':b['routes'][0]['configuration']['sha256']='changed'
            else:b['routes'][0]['train']['consumer']='changed'
            with patch.object(o.graphs,'read_policy_graph',return_value=a),self.subTest(name=name),self.assertRaises(ValueError):o.physical_equivalence(None,None,b)


class MetadataAffinityTests(unittest.TestCase):
    def fixture(self,root):
        graph={};affinity=inherited.affinity_fixture(root,graph)
        topology={k:deepcopy(affinity[k]) for k in ('allowed_logical_cpus','core_groups')}
        return graph,affinity,topology

    def test_narrow_metadata_auditor_checks_real_declared_topology_without_widening(self):
        with TemporaryDirectory() as d:
            graph,expected,topology=self.fixture(d,)
            with patch.object(o,'_hardware_snapshot',return_value=topology) as inspect,patch.object(o.os,'sched_setaffinity') as widen:
                self.assertEqual(o.read_CPU_affinity(d,graph,metadata_only=True),expected)
                inspect.assert_called_once_with(list(range(32)));widen.assert_not_called()

    def test_narrow_owner_startup_is_rejected(self):
        with TemporaryDirectory() as d:
            graph,expected,topology=self.fixture(d)
            topology['allowed_logical_cpus']=[0,16];topology['core_groups']=topology['core_groups'][:1]
            with patch.object(o,'_hardware_snapshot',return_value=topology) as inspect,self.assertRaises(ValueError):o.read_CPU_affinity(d,graph)
            inspect.assert_called_once_with(None)

    def test_metadata_topology_or_host_tool_mismatch_is_still_rejected(self):
        with TemporaryDirectory() as d:
            graph,expected,topology=self.fixture(d);topology['core_groups'][0]['L3_bytes']+=1
            with patch.object(o,'_hardware_snapshot',return_value=topology),self.assertRaises(ValueError):o.read_CPU_affinity(d,graph,metadata_only=True)


class InheritedSchedulerTests(inherited.SchedulerTests):
    """Exercise the copied scheduler with the original behavioral contract."""
    def setUp(self):self.original=inherited.o;inherited.o=o
    def tearDown(self):inherited.o=self.original


class InheritedQualificationWaveTests(inherited.QualificationWaveTests):
    def setUp(self):self.original=inherited.o;inherited.o=o
    def tearDown(self):inherited.o=self.original


if __name__=='__main__':unittest.main()
