"""Fixed aux200 scope and mocked CPU raw/base/hooks/parent/stop integration.

No simulator or subprocess is started. Native-shaped fixtures are explicitly
tagged as synthetic CPU fixtures and live only in removable test directories.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import collect_teacher_supplement_r10 as worker
from our_mani.multitask import teacher_supplement as audit
from our_mani.multitask import acquisition_lineage as lineage

ROOT=worker.ROOT
BASE=ROOT/'artifacts/framework/multitask_r10'
DRAFT=BASE/'TRAIN_teacher_supplement_basketball_r1_v1_setup/preparation_report.json'


class SupplementWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='aux_CPU_fixture_',dir=BASE)
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        self.prepared=worker.prepare(DRAFT,self.directory/'setup',self.directory/'collection')
        self.candidate=ROOT/self.prepared['execution_registration']['path']
        self.approved=self.directory/'approved_registration.json'
        worker.approve_registration(self.candidate,self.approved)
        self.context=audit.verify_registration(ROOT,lineage.pointer(ROOT,self.approved),live=True,require_ready=True)
        self.case=next(iter(self.context['expected_cases'].values()))

    def mutated_registry(self,modify):
        folder=self.directory/('mutation'+str(len(list(self.directory.glob('mutation*')))))
        folder.mkdir();registry=deepcopy(self.context['registry']);plan=deepcopy(self.context['plan']);modify(plan)
        plan['auxiliary_collection']=str((folder/'collection').relative_to(ROOT))
        plan.pop('plan_sha256');plan['plan_sha256']=lineage.canonical_sha(plan)
        plan_path=folder/'plan.json';worker.collection.write_json(plan_path,plan)
        snapshot=lineage.read(ROOT/registry['execution']['source_snapshot']['path'])
        snapshot['plan']=dict(**lineage.pointer(ROOT,plan_path),internal_sha256=plan['plan_sha256'])
        snapshot_path=folder/'source_snapshot.json';worker.collection.write_json(snapshot_path,snapshot)
        registry['execution'].update(plan=lineage.pointer(ROOT,plan_path),source_snapshot=lineage.pointer(ROOT,snapshot_path),
            collection=str((folder/'collection').relative_to(ROOT)))
        registry_path=folder/'registration.json';worker.collection.write_json(registry_path,registry)
        return registry_path

    def test_actual_fixed200_metadata_and_seed_fold_scope(self):
        rows=self.context['plan']['cases']
        self.assertEqual(len(rows),200)
        self.assertEqual({task:sum(row['task']==task for row in rows) for task in audit.TASKS},dict.fromkeys(audit.TASKS,100))
        self.assertEqual(len({row['parent_source_group'] for row in rows}),200)
        self.assertTrue(all(row['replica']==row['behavior_replica']==1 and row['teacher_used']
            and not row['context_primary_view'] and row['maximum_collection_ticks']==200 for row in rows))
        self.assertEqual(rows,audit.expected_cases(self.context['primary']))
        with self.assertRaisesRegex(ValueError,'first-job review'):
            audit.verify_registration(ROOT,lineage.pointer(ROOT,self.candidate),require_ready=True)
        self.assertFalse(self.context['current']['collection'].exists())
        self.assertFalse(lineage.read(ROOT/lineage.read(DRAFT)['plan']['path'])['native_execution_ready'])

    def test_rehashed_mutations_cannot_change_case_seed_variation_geometry_labels_or_budget(self):
        changes=[lambda p:p['cases'].pop(),lambda p:p['cases'][0].update(replica=2),
            lambda p:p['cases'][0]['source'].update(variation=1-p['cases'][0]['source']['variation']),lambda p:p['cases'][0].update(template_asset='wrong.json'),
            lambda p:p['cases'][0].update(context_primary_view=True),lambda p:p['cases'][0].update(maximum_collection_ticks=201),
            lambda p:p['protocol'].update(seed=1018),lambda p:p['protocol'].update(recovery_input={'path':'fake','sha256':'0'*64}),
            lambda p:p['augmentation_budget'].update(expected_cases=199)]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                path=self.mutated_registry(change)
                with self.assertRaises(ValueError):audit.verify_registration(ROOT,lineage.pointer(ROOT,path))

    def test_candidate_cannot_execute_and_incomplete200_stays_pending(self):
        with self.assertRaisesRegex(ValueError,'first-job review'):
            worker.collect_case(self.context['current']['path'],self.case['case_id'],self.directory/'wrong',
                ROOT/'unused_guard.json',self.candidate)
        proof=audit.audit_collection(ROOT,lineage.pointer(ROOT,self.approved))
        self.assertFalse(proof['complete']);self.assertEqual(proof['acquisition_projection']['completed'],0)
        self.assertEqual(len(proof['acquisition_projection']['missing_case_ids']),200)
        with self.assertRaisesRegex(ValueError,'complete fixed auxiliary200'):
            audit.audit_collection(ROOT,lineage.pointer(ROOT,self.approved),require_complete=True)

    def fixture_collector(self,plan_path,cid,output,guard):
        case=self.context['expected_cases'][cid]
        previous=lineage.read(BASE/'basketball_TRAIN_completed_skill_diagnostic_v1/report.json')['tasks'][case['task']]
        old=next(r for r in previous['case_receipts_and_adapter_diagnostics'] if r['parent_source_group']==case['parent_source_group'])
        raw=lineage.read(ROOT/old['canonical_receipt']['path']);raw.pop('parent_process_receipt',None)
        raw.pop('recovery_execution',None)
        # Recovered originals may carry base/hook files; only native collector
        # outputs belong to the synthetic CPU base fixture.
        raw['files_sha256']={name:checksum for name,checksum in raw['files_sha256'].items()
            if Path(name).name not in ('base_collector_receipt.json','recovery_hook_receipt.json')}
        out=worker.collection.case_directory(output,case);out.mkdir(parents=True)
        relocated={}
        for name,checksum in raw['files_sha256'].items():
            destination=out/Path(name).name;shutil.copyfile(ROOT/name,destination)
            relocated[name]=str(destination.relative_to(ROOT))
        raw['files_sha256']={relocated[name]:checksum for name,checksum in raw['files_sha256'].items()}
        entry=raw['entry']
        for name in ('raw_episode_id','behavior_replica','maximum_collection_ticks','teacher_used','context_primary_view','interaction_eligible'):
            entry[name]=case[name]
        entry.update(source_id=case['task']+':'+case['raw_episode_id'],augmentation_id=case['raw_episode_id'],augmentation_replica=1)
        for name in ('cache','teacher_command_receipt','phase_completion_events_receipt','motion_label_diagnostics'):
            if entry.get(name) in relocated:entry[name]=relocated[entry[name]]
        reset=lineage.expected_reset(case,self.context['plan']['protocol'])
        native=raw['native_reset_receipt'];native.update(reset,replica=1)
        permission=self.context['plan']['protocol']['development_reset_profile']
        physical=dict(version=lineage.read(ROOT/permission['path'])['version'],permission=permission,
            complete=True,task=case['task'],split='train',replica=1,source=case['source'],source_resampled=False,
            official_TEST_allowed=False,recovery_applied=False,**reset)
        native.update(reset_recovery_receipt=physical,development_reset_profile=physical['permission'])
        self.captures['simulations'].append(SimpleNamespace(reset_recovery_receipt=physical))
        self.captures['paths'].append(SimpleNamespace(recovery_receipt=dict(version='r10_explicit_owned_RML_recovery_adapter_v1',
            close_complete=True,physics_step_called_by_adapter=False,real_native_step_motion_replaced=False,
            historical_fault_reproduced_in_native_diagnostic=False)))
        raw.update(case_id=cid,plan_sha256=self.context['plan']['plan_sha256'],raw_episode_id=case['raw_episode_id'],
            process_id=123456,started_utc=worker.collection.utc(),finished_utc=worker.collection.utc(),CPU_fixture_not_native=True)
        worker.collection.write_json(out/'case.json',raw)
        self.base_bytes=(out/'case.json').read_bytes()
        return raw

    @contextmanager
    def fake_hooks(self,plan,captures):
        self.captures=captures
        yield

    def create_mocked_raw_child(self):
        output=self.context['current']['collection']/'infrastructure_attempts'/self.case['case_id']/'0/output'
        with patch.object(worker.recovery,'scoped_owned_hooks',self.fake_hooks), \
             patch.object(worker.collection,'collect_native_case',self.fixture_collector):
            worker.collect_case(self.context['current']['path'],self.case['case_id'],output,ROOT/'unused_guard.json',self.approved)
        return worker.collection.case_directory(output,self.case)/'case.json'

    def test_mocked_raw_child_preserves_base_bytes_and_requires_closed_actual_hooks(self):
        path=self.create_mocked_raw_child();raw=lineage.read(path)
        own=raw['teacher_supplement_execution']
        self.assertEqual((ROOT/own['base_collector_receipt']['path']).read_bytes(),self.base_bytes)
        self.assertEqual(raw['entry']['augmentation_replica'],1)
        self.assertTrue(raw['CPU_fixture_not_native'])
        audit.validate_result(ROOT,path,self.context,self.case)
        hook_path=ROOT/own['hook_receipt']['path'];hooks=lineage.read(hook_path)
        hooks['RML_path_recovery_receipts']=[];worker.collection.write_json(hook_path,hooks,overwrite=True)
        value=lineage.pointer(ROOT,hook_path);raw['teacher_supplement_execution']['hook_receipt']=value
        raw['files_sha256'][value['path']]=value['sha256'];worker.collection.write_json(path,raw,overwrite=True)
        with self.assertRaisesRegex(ValueError,'actual owned RML path witness'):
            audit.validate_result(ROOT,path,self.context,self.case)

    def test_raw_child_without_actual_parent_commit_is_never_complete(self):
        self.create_mocked_raw_child()
        proof=audit.audit_collection(ROOT,lineage.pointer(ROOT,self.approved))
        self.assertEqual(proof['acquisition_projection']['completed'],0)
        self.assertTrue(proof['pending'])

    def test_mocked_parent_child_chain_preserves_independent_pids_and_resumes_only_complete(self):
        def popen(command,**kwargs):
            cid=lineage.command_argument(command,'--case-id');out=Path(lineage.command_argument(command,'--output'))
            worker.collect_case(self.context['current']['path'],cid,out,ROOT/'unused_guard.json',self.approved)
            return SimpleNamespace(pid=123455,wait=lambda timeout=None:0)
        with patch.object(worker.recovery,'scoped_owned_hooks',self.fake_hooks), \
             patch.object(worker.collection,'collect_native_case',self.fixture_collector), \
             patch.object(worker.collection,'require_r09_release',return_value={'CPU_fixture_not_native':True}), \
             patch.object(worker.subprocess,'Popen',side_effect=popen) as process:
            report=worker.run_queue(self.context['current']['path'],self.context['current']['collection'],ROOT/'unused_guard.json',
                self.approved,case_ids=[self.case['case_id']],workers=1)
            self.assertEqual(report['completed'],1);self.assertFalse(report['complete']);self.assertEqual(process.call_count,1)
            canonical=worker.collection.case_directory(self.context['current']['collection'],self.case)/'case.json'
            raw=lineage.read(canonical)
            self.assertEqual(raw['process_id'],123456);self.assertEqual(raw['parent_process_receipt']['process_id'],123455)
            worker.run_queue(self.context['current']['path'],self.context['current']['collection'],ROOT/'unused_guard.json',
                self.approved,case_ids=[self.case['case_id']],workers=1)
            self.assertEqual(process.call_count,1)

    def test_first_infra_stops_new_dispatch_and_no_automatic_retry(self):
        with patch.object(worker.collection,'require_r09_release',return_value={'CPU_fixture_not_native':True}), \
             patch.object(worker.subprocess,'Popen',return_value=SimpleNamespace(pid=123455,wait=lambda timeout=None:1)) as process:
            result=worker.run_queue(self.context['current']['path'],self.context['current']['collection'],ROOT/'unused_guard.json',
                self.approved,workers=1)
            self.assertEqual(process.call_count,1)
            self.assertTrue(result['stopped_dispatch_after_first_infrastructure_error'])
            self.assertEqual(result['completed'],0);self.assertEqual(len(result['pending_case_ids']),200)
            proof=audit.audit_collection(ROOT,lineage.pointer(ROOT,self.approved))
            self.assertEqual(len(proof['infrastructure_errors']),1)
            with self.assertRaisesRegex(ValueError,'no automatic retry'):
                worker.run_queue(self.context['current']['path'],self.context['current']['collection'],ROOT/'unused_guard.json',self.approved)
            self.assertEqual(process.call_count,1)

    def test_prelaunch_failure_preserves_receipt_without_inventing_a_pid(self):
        with patch.object(worker.collection,'require_r09_release',return_value={'CPU_fixture_not_native':True}), \
             patch.object(worker.subprocess,'Popen',side_effect=OSError('CPU fixture launch failure')) as process:
            result=worker.run_queue(self.context['current']['path'],self.context['current']['collection'],ROOT/'unused_guard.json',
                self.approved,workers=1)
        self.assertEqual(process.call_count,1);self.assertEqual(result['completed'],0)
        parent=next((self.context['current']['collection']/'infrastructure_attempts').glob('*/0/parent_receipt.json'))
        receipt=lineage.read(parent)
        self.assertFalse(receipt['process_started']);self.assertIsNone(receipt['process_id']);self.assertIsNone(receipt['returncode'])
        proof=audit.audit_collection(ROOT,lineage.pointer(ROOT,self.approved))
        self.assertEqual(len(proof['infrastructure_errors']),1)

    def test_first_infra_stops_dispatch_but_both_inflight_parents_finish(self):
        barrier=threading.Barrier(2)
        def popen(command,**kwargs):
            def wait(timeout=None):
                barrier.wait(timeout=10)
                return 1
            return SimpleNamespace(pid=123455,wait=wait)
        ids=list(self.context['expected_cases'])[:3]
        with patch.object(worker.collection,'require_r09_release',return_value={'CPU_fixture_not_native':True}), \
             patch.object(worker.subprocess,'Popen',side_effect=popen) as process:
            result=worker.run_queue(self.context['current']['path'],self.context['current']['collection'],ROOT/'unused_guard.json',
                self.approved,case_ids=ids,workers=2)
        self.assertEqual(process.call_count,2)
        self.assertEqual(len(result['receipts']),2);self.assertTrue(result['inflight_jobs_allowed_to_finish'])
        self.assertEqual(len(audit.audit_collection(ROOT,lineage.pointer(ROOT,self.approved))['infrastructure_errors']),2)
        self.assertFalse((self.context['current']['collection']/'infrastructure_attempts'/ids[2]).exists())


if __name__=='__main__':unittest.main()
