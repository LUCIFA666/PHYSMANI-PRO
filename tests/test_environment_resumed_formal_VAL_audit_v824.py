"""CPU fake evidence only: preserve all historical outcomes and exact namespace bindings."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from scripts import audit_environment_resumed_formal_VAL_v824_r10 as audit
from our_mani.multitask.evaluation import digest


def fixture(imported_count=591):
    policy='e'*64;jobs=[]
    for index in range(945):
        method=('cv_geometry','context_geometry','physmani')[index%3]
        old_import=index<95;stage='runtime/framework/formal_VAL_v753' if old_import else audit.PREVIOUS_RUNTIME
        raw='episode'+str(index);task='beat_the_rotating_buzz'
        jobs.append(dict(job_id='job'+str(index),task=task,raw=raw,method=method,
            argv=['bash','scripts/run_local_sim.sh','751.py','--config','same.json',
                '--suite','752.json','--suite-lock','752_lock.json','--source-inputs','VAL_sources.json',
                '--split','validation','--task',task,'--raw-episode-id',raw,'--method',method,
                '--out',str(Path('original753' if old_import else 'original772')/task/raw/method)],
            evidence_runtime=stage,imported_completed=old_import,source_row_sha256='a'*64,
            worker_sha256='b'*64,case_worker=dict(path='743.py',sha256='c'*64)))
    old_plan=dict(jobs=jobs,split='validation',expected_cases=945,policy_sha256=policy)
    completed=[r['job_id'] for r in jobs[:imported_count]];remaining=[r['job_id'] for r in jobs[imported_count:]]
    failed=remaining[:3]
    status=dict(completed={name:dict(complete=True,native_success=index%2==0) for index,name in enumerate(completed)},
        failed={name:dict(stage='native',complete=False,returncode=1,interrupted=False,timed_out=False,automatic_retry=False) for name in failed},
        active=[],checking=[],stop_requested=False,drain_requested=True)
    terminal=dict(complete=False,drained=True,drain_requested=True,stop_requested=False,completed=len(completed),
        failed=len(failed),expected=945,policy_sha256=policy)
    control=dict(schema=audit.CONTROL_SCHEMA,split='validation',user_authorized_resume=True,expected_cases=945,
        standing_authorization_reused=True,new_user_event_invented=False,physical_policy_changed=False,
        previous_runtime=audit.PREVIOUS_RUNTIME,completed_job_ids=completed,remaining_job_ids=remaining,failed_job_ids=failed,
        completed_key_sha256=digest(sorted(completed)),remaining_key_sha256=digest(sorted(remaining)),
        failed_key_sha256=digest(sorted(failed)),policy_sha256=policy)
    new=deepcopy(jobs)
    for old,row in zip(jobs,new):
        imported=row['job_id'] in completed
        row.update(previous_row_sha256=digest(old),previous_imported_completed=old['imported_completed'],imported_completed=imported)
        if not imported:
            row['evidence_runtime']=audit.RUNTIME
            row['argv'][-1]=str(Path(audit.OUTPUT)/row['task']/row['raw']/row['method'])
    return control,old_plan,status,terminal,dict(jobs=new,split='validation',expected_cases=945,policy_sha256=policy)


def environment_fixture():
    control=dict(environment_wrapper=dict(path='scripts/run_frozen_system_environment_v820_r10.py',sha256='1'*64),
        environment_manifest=dict(path='snapshot/manifest.json',sha256='2'*64),
        environment_namespace_proof=dict(path='snapshot/CPUproof.json',sha256='3'*64))
    receipt=dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,
        manifest=control['environment_manifest'],wrapper=control['environment_wrapper'],
        namespace_qualification=control['environment_namespace_proof'],source_locks=[dict(path='oldlock',sha256='4'*64)],
        current_mount_namespace='mnt:[200]',mount_namespace='mnt:[200]',reference_mount_namespace='mnt:[100]',
        qualified_host_mount_namespace='mnt:[100]',observer_pid=123,pid=123,observer_start_ticks=1000,
        outer_wrapper_pid=100,outer_wrapper_start_ticks=500,checked_OS_files=6418,
        all_current_OS_targets_full_byte_SHA_verified=True,current_OS_target_SHA_map_sha256='5'*64,
        restored_four_target_SHA256={p:'6'*64 for p in audit.environment.RESTORED_TARGETS},
        host_PID_network_namespace_preserved=True,proc_or_dev_replaced=False,
        task_parameters_or_strategy_changed=False,project_writable=True)
    return control,receipt


class PlanTests(unittest.TestCase):
    def check(self,parts):return audit.require_resume_plan(*parts,runtime=audit.ROOT/audit.RUNTIME)

    def test_all591_accepts_include_native_false_and_preserve_both_real_old_tiers(self):
        parts=fixture();old,new=self.check(parts)
        self.assertEqual(len(new),945)
        self.assertEqual(sum(r['imported_completed'] for r in new.values()),591)
        self.assertEqual(sum(r['evidence_runtime'].endswith('v753') for r in new.values()),95)
        self.assertEqual(sum(r['evidence_runtime']==audit.PREVIOUS_RUNTIME for r in new.values()),496)
        self.assertTrue(any(not r['native_success'] for r in parts[2]['completed'].values()))
        self.assertTrue(new['job95']['imported_completed'])
        self.assertFalse(new['job95']['previous_imported_completed'])
        self.assertEqual(new['job95']['argv'],old['job95']['argv'])

    def test_complete_partition_is_dynamic_not_legacy95_or850(self):
        parts=fixture(580);self.check(parts)
        self.assertEqual(sum(r['imported_completed'] for r in parts[-1]['jobs']),580)

    def test_imported_tier_argv_and_previous_full_row_hash_are_immutable(self):
        for index in (0,95):
            for change in ('out','tier','old_flag','hash','source','current_import'):
                p=fixture();r=p[-1]['jobs'][index]
                if change=='out':r['argv'][-1]='rerun/or/overwrite'
                elif change=='tier':r['evidence_runtime']=audit.RUNTIME
                elif change=='old_flag':r['previous_imported_completed']=not r['previous_imported_completed']
                elif change=='hash':r['previous_row_sha256']='0'*64
                elif change=='source':r['source_row_sha256']='0'*64
                else:r['imported_completed']=False
                with self.subTest(index=index,change=change),self.assertRaises(ValueError):self.check(p)

    def test_remaining354_change_only_namespace_and_current_provenance(self):
        self.check(fixture())
        for change in ('source','config','suite','old_tier','old_out','fake_previous','wrong_method','duplicate_out'):
            p=fixture();r=p[-1]['jobs'][591]
            if change=='source':r['source_row_sha256']='f'*64
            elif change in ('config','suite'):r['argv'][r['argv'].index('--'+change)+1]='changed'
            elif change=='old_tier':r['evidence_runtime']=audit.PREVIOUS_RUNTIME
            elif change=='old_out':r['argv'][-1]=p[1]['jobs'][591]['argv'][-1]
            elif change=='fake_previous':r['previous_imported_completed']=True
            elif change=='wrong_method':r['method']='physmani'
            else:r['argv']+=['--out','another']
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(p)

    def test_grid_partition_cannot_skip_environment_error_or_replace_completed_failure(self):
        for change in ('drop_row','duplicate','skip_failure','select_success','completed_digest','failed_digest'):
            p=fixture();c=p[0]
            if change=='drop_row':p[-1]['jobs'].pop()
            elif change=='duplicate':p[-1]['jobs'][-1]=deepcopy(p[-1]['jobs'][0])
            elif change=='skip_failure':c['remaining_job_ids'].remove(c['failed_job_ids'][0])
            elif change=='select_success':c['completed_job_ids'][1]='job592'
            elif change=='completed_digest':c['completed_key_sha256']='0'*64
            else:c['failed_key_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(p)

    def test_no_new_user_stop_and_real_infrastructure_drain_required(self):
        for change in ('active','checking','stop','not_drained','native_success','timeout','interrupted','retry','invented_user','policy','split'):
            p=fixture()
            if change in ('active','checking'):p[2][change]=['still-live']
            elif change=='stop':p[2]['stop_requested']=True
            elif change=='not_drained':p[3]['drained']=False
            elif change in ('native_success','timeout','interrupted','retry'):
                r=p[2]['failed']['job591']
                r[{'native_success':'returncode','timeout':'timed_out','interrupted':'interrupted','retry':'automatic_retry'}[change]]=0 if change=='native_success' else True
            elif change=='invented_user':p[0]['new_user_event_invented']=True
            elif change=='policy':p[-1]['policy_sha256']='f'*64
            else:p[-1]['split']='test'
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(p)


class EnvironmentTests(unittest.TestCase):
    def test_actual823_owner_must_complete_all945_with_one_real_namespace(self):
        control,receipt=environment_fixture()
        protocol=dict(runtime_consumer={'path':'744','sha256':'a'*64},original_runtime={'path':'664','sha256':'b'*64})
        control.update(author_outcome_protocol=protocol,policy_sha256='c'*64,resume_owner={'path':audit.OWNER,'sha256':'d'*64})
        driver={'path':'751','sha256':'e'*64};guard={'path':'728','sha256':'f'*64}
        launch=dict(pid=123,source_sha256='d'*64,split='validation',suite_lock_sha256='1'*64,
            split_source_lock_sha256='2'*64,policy_sha256='c'*64,admission_driver=driver,admission_guard=guard,
            execution_protocol_changed=True,author_outcome_protocol=protocol,native_runtime=protocol['runtime_consumer'],
            original_native_runtime=protocol['original_runtime'],frozen_environment=receipt)
        terminal=dict(complete=True,expected=945,completed=945,failed=0,split='validation',policy_sha256='c'*64,
            author_outcome_protocol=protocol,frozen_environment=deepcopy(receipt))
        kwargs=dict(suite_lock_sha='1'*64,source_lock_sha='2'*64,driver=driver,guard=guard,expected_environment=receipt)
        audit.require_owner_launch(launch,terminal,control,**kwargs)
        for change in ('old_source','only354','namespace_changed','receipt_owner','old_policy'):
            bad_launch=deepcopy(launch);bad_terminal=deepcopy(terminal)
            if change=='old_source':bad_launch['source_sha256']='old772'
            elif change=='only354':bad_terminal['completed']=354
            elif change=='namespace_changed':
                bad_terminal['frozen_environment']['current_mount_namespace']='mnt:[201]'
                bad_terminal['frozen_environment']['mount_namespace']='mnt:[201]'
            elif change=='receipt_owner':bad_terminal['frozen_environment']['observer_pid']=124
            else:bad_launch['policy_sha256']='changed'
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.require_owner_launch(bad_launch,bad_terminal,control,**kwargs)

    def test_receipt_binds_actual_owner_full6418_bytes_and_host_visibility(self):
        control,receipt=environment_fixture()
        self.assertIs(audit.require_environment_receipt(receipt,control,owner_pid=123,expected=deepcopy(receipt)),receipt)
        for field,value in [('complete',False),('checked_OS_files',4),('checked_OS_files',6418.0),('all_current_OS_targets_full_byte_SHA_verified',False),
                            ('observer_pid',999),('outer_wrapper_start_ticks',0),('proc_or_dev_replaced',True),
                            ('task_parameters_or_strategy_changed',True),('host_PID_network_namespace_preserved',False),
                            ('current_OS_target_SHA_map_sha256','f'*64)]:
            bad=deepcopy(receipt);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.require_environment_receipt(bad,control,owner_pid=123,expected=receipt)
        bad=deepcopy(receipt);bad['current_mount_namespace']=bad['mount_namespace']='mnt:[100]'
        with self.assertRaises(ValueError):audit.require_environment_receipt(bad,control)
        bad=deepcopy(receipt);bad['namespace_qualification']={'path':'unqualified','sha256':'3'*64}
        with self.assertRaises(ValueError):audit.require_environment_receipt(bad,control)

    def test_child_must_inherit_exact_real_owner_receipt_and_namespace(self):
        control,receipt=environment_fixture();owner=dict(pid=123,frozen_environment=receipt)
        child=dict(inherited_frozen_environment=deepcopy(receipt),current_mount_namespace='mnt:[200]')
        audit.require_inherited_environment(child,owner,control,receipt)
        for change in ('namespace','receipt','owner_pid'):
            c=deepcopy(child);o=deepcopy(owner)
            if change=='namespace':c['current_mount_namespace']='mnt:[201]'
            elif change=='receipt':c['inherited_frozen_environment']['observer_pid']=999
            else:o['pid']=999
            with self.subTest(change=change),self.assertRaises(ValueError):audit.require_inherited_environment(c,o,control,receipt)

    def test_capture_calls_only_authoritative820_with_frozen_qualification(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(audit,'ROOT',Path(temporary)):
            control,receipt=environment_fixture();root=Path(temporary)
            for key in audit.ENVIRONMENT_FIELDS:
                path=root/control[key]['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(key.encode())
                control[key]=audit.pointer(path)
            receipt.update(wrapper=control['environment_wrapper'],manifest=control['environment_manifest'],
                           namespace_qualification=control['environment_namespace_proof'])
            with patch.object(audit.environment,'capture_namespace_evidence',return_value=receipt) as capture:
                audit.capture_environment(control)
                capture.assert_called_once_with(root/control['environment_manifest']['path'],control['environment_manifest']['sha256'],
                    root/control['environment_namespace_proof']['path'],control['environment_namespace_proof']['sha256'])
            control['environment_manifest']['sha256']='0'*64
            with patch.object(audit.environment,'capture_namespace_evidence') as capture,self.assertRaises(ValueError):
                audit.capture_environment(control)
            capture.assert_not_called()

    def test_clean_terminal_rejects_retries_timeouts_and_returncode_inconsistency(self):
        good=dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False)
        audit.require_clean_terminal(good)
        for field,value in [('complete',False),('returncode',1),('returncode',False),('timed_out',True),('interrupted',True),('automatic_retry',True)]:
            bad=dict(good);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.require_clean_terminal(bad)


class ReleaseTests(unittest.TestCase):
    def test_original_full945_gate_rejects_partial_and_duplicate_without_TEST_reads(self):
        path=Path(__file__).with_name('test_common_rml_release_v531.py')
        spec=importlib.util.spec_from_file_location('_environment_release531',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        routes,manifest,cases=module.Tests().fixture()
        self.assertTrue(audit.validation_release(cases,manifest,routes)['formal_test_permitted'])
        for bad in (cases[:-1],cases[:-1]+[cases[0]]):
            with self.assertRaises(ValueError):audit.validation_release(bad,manifest,routes)


if __name__=='__main__':unittest.main()
