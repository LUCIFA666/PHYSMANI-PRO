"""CPU fakes for nested evidence provenance and namespace/STOP admission."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json,os,tempfile,threading,unittest
from scripts import run_environment_resumed_formal_VAL_owner_v823_r10 as owner


class OwnerTests(unittest.TestCase):
    def fixture_rows(self):
        old=[]
        for name,tier,imported in [('legacy','runtime/framework/formal_VAL_v753',True),
                                   ('second',owner.PREVIOUS,False),('pending',owner.PREVIOUS,False)]:
            old.append(dict(job_id=name,task='task',raw=name,method='cv_geometry',argv=['bash','shell','driver','--out','old/'+name],
                evidence_runtime=tier,imported_completed=imported,worker_sha256='same-physical-worker'))
        fresh=deepcopy(old)
        for row in fresh:row['argv'][-1]='new/'+row['job_id']
        return old,fresh

    def test_imports_keep_both_real_tiers_old_argv_and_previous_markers(self):
        old,fresh=self.fixture_rows();before=deepcopy(old)
        got=owner.resumed_jobs(fresh,dict(jobs=old),dict(completed_job_ids=['legacy','second']),owner.RUNTIME)
        for row,previous in zip(got[:2],old[:2]):
            self.assertTrue(row['imported_completed']);self.assertEqual(row['argv'],previous['argv'])
            self.assertEqual(row['evidence_runtime'],previous['evidence_runtime'])
            self.assertEqual(row['previous_row_sha256'],owner.digest(previous))
            restored={k:v for k,v in row.items() if k not in ('previous_row_sha256','previous_imported_completed')}
            restored['imported_completed']=row['previous_imported_completed'];self.assertEqual(restored,previous)
        self.assertEqual(got[2]['argv'][-1],'new/pending');self.assertFalse(got[2]['imported_completed'])
        self.assertEqual(got[2]['evidence_runtime'],owner.RUNTIME)
        got[0]['argv'].append('mutated');self.assertEqual(old,before)

    def test_unknown_old_tier_is_not_flattened_or_guessed(self):
        old,fresh=self.fixture_rows();old[0]['evidence_runtime']='wrong'
        with self.assertRaisesRegex(ValueError,'evidence tier'):
            owner.resumed_jobs(fresh,dict(jobs=old),dict(completed_job_ids=['legacy']),owner.RUNTIME)

    def test_full_nested_plan_is_accepted_by_independent824_and_flattened_tier_is_rejected(self):
        from scripts import audit_environment_resumed_formal_VAL_v824_r10 as audit
        failures=['failed'+str(i) for i in range(3)]
        names=['job'+str(i) for i in range(942)]+failures
        oldrows=[dict(job_id=name,task='fake',raw=name,method='cv_geometry',argv=['--out','old/'+name],
            evidence_runtime='runtime/framework/formal_VAL_v753' if i<95 else owner.PREVIOUS,
            imported_completed=i<95) for i,name in enumerate(names)]
        completed=names[:591];remaining=sorted(set(names)-set(completed));policy='a'*64
        control=dict(schema=owner.CONTROL_SCHEMA,split='validation',user_authorized_resume=True,expected_cases=945,
            standing_authorization_reused=True,new_user_event_invented=False,physical_policy_changed=False,
            previous_runtime=owner.PREVIOUS,completed_job_ids=completed,remaining_job_ids=remaining,failed_job_ids=failures,
            completed_key_sha256=owner.digest(sorted(completed)),remaining_key_sha256=owner.digest(remaining),
            failed_key_sha256=owner.digest(sorted(failures)),policy_sha256=policy)
        oldplan=dict(jobs=oldrows,split='validation',expected_cases=945,policy_sha256=policy)
        oldstatus=dict(completed={name:{} for name in completed},failed={name:dict(stage='native',complete=False,interrupted=False,
            timed_out=False,returncode=1,automatic_retry=False) for name in failures},active=[],checking=[],drain_requested=True,stop_requested=False,
            native_slots=0,GPU_slots=0)
        terminal=dict(complete=False,completed=591,failed=3,expected=945,drained=True,drain_requested=True,stop_requested=False,policy_sha256=policy)
        fresh=deepcopy(oldrows)
        for row in fresh:row['argv'][-1]=str(Path(owner.OUTPUT)/row['task']/row['raw']/row['method'])
        rows=owner.resumed_jobs(fresh,oldplan,control,owner.RUNTIME)
        newplan=dict(jobs=rows,split='validation',expected_cases=945,policy_sha256=policy)
        audit.require_resume_plan(control,oldplan,oldstatus,terminal,newplan,runtime=owner.ROOT/owner.RUNTIME)
        rows[0]['evidence_runtime']=owner.PREVIOUS
        with self.assertRaises(ValueError):audit.require_resume_plan(control,oldplan,oldstatus,terminal,newplan,runtime=owner.ROOT/owner.RUNTIME)

    def test_STOP_never_starts_a_child_and_is_not_a_task_outcome(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(owner,'ROOT',Path(tmp)):
            event=threading.Event();event.set()
            with patch.object(owner.subprocess,'Popen') as popen:
                result=owner.child(dict(argv=['never']),Path(tmp)/'fake-child',event,check=True)
                popen.assert_not_called()
            self.assertFalse(result['complete']);self.assertTrue(result['interrupted'])

    def test_missing_namespace_never_starts_a_child(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(owner,'ROOT',Path(tmp)),patch.object(owner,'ENVIRONMENT_RECEIPT',None):
            with patch.object(owner.subprocess,'Popen') as popen:
                result=owner.child(dict(argv=['never']),Path(tmp)/'fake-child',threading.Event(),check=True)
                popen.assert_not_called()
            self.assertFalse(result['complete']);self.assertIn('namespace',result['error'])

    def control_fixture(self,root):
        mapping=dict(suite='suite.json',suite_lock='suite-lock.json',source_inputs='source.json',resume_owner='owner.py',
            resume_auditor='scripts/audit_environment_resumed_formal_VAL_v824_r10.py',
            previous_launch=owner.PREVIOUS+'/launch.json',previous_plan=owner.PREVIOUS+'/plan.json',
            previous_status=owner.PREVIOUS+'/status.json',previous_terminal=owner.PREVIOUS+'/terminal.json',
            previous_resume_control='artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup/resume_control.json',
            previous_resume_control_lock='artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup/resume_inputs.json',
            user_resume_record='runtime/framework/user_resume_v772.json',
            standing_continuation_record='runtime/framework/latest_executor_continuation_v817.json',
            environment_manifest='manifest.json',environment_wrapper='wrapper.py',environment_namespace_proof='proof.json')
        for name in mapping.values():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        (root/mapping['user_resume_record']).write_text(json.dumps(dict(user_authorized_resume=True,revokes_previous_stop=True)))
        (root/mapping['standing_continuation_record']).write_text(json.dumps(dict(continue_until_user_stop=True,user_stop_received=False)))
        c=dict(schema=owner.CONTROL_SCHEMA,user_authorized_resume=True,split='validation',expected_cases=945,
            previous_runtime=owner.PREVIOUS,standing_authorization_reused=True,new_user_event_invented=False,physical_policy_changed=False)
        c.update({k:owner.pointer(root/v) for k,v in mapping.items()})
        (root/'control.json').write_text(json.dumps(c));(root/'lock.json').write_text(json.dumps(dict(schema=owner.LOCK_SCHEMA,approved_for_execution=True,files={})))
        return c

    def test_admission_needs_actual_standing_permission_and_same_source_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with (patch.object(owner,'ROOT',root),patch.object(owner,'__file__',str(root/'owner.py')),
                  patch.object(owner,'require_frozen_environment',return_value={'fake':True}),
                  patch.object(owner,'verify_inputs'),patch.object(owner,'locked_pointer',side_effect=lambda r,f,p:r/p['path'])):
                c=self.control_fixture(root);args=('control.json','lock.json','suite.json','suite-lock.json','source.json')
                self.assertTrue(owner.load_resume_control(*args)[0]['standing_authorization_reused'])
                for field,bad in [('user_authorized_resume',False),('new_user_event_invented',True),('physical_policy_changed',True)]:
                    changed=dict(c,**{field:bad});(root/'control.json').write_text(json.dumps(changed))
                    with self.subTest(field=field),self.assertRaises(ValueError):owner.load_resume_control(*args)
                (root/'control.json').write_text(json.dumps(c));(root/'other.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'scope changed'):owner.load_resume_control(*args[:-1],'other.json')
                (root/c['standing_continuation_record']['path']).write_text(json.dumps(dict(continue_until_user_stop=False,user_stop_received=True)))
                with self.assertRaisesRegex(ValueError,'standing'):owner.load_resume_control(*args)

    def test_global_STOP_preempts_environment_and_lock_reads(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(owner,'ROOT',Path(tmp)):
            stop=Path(tmp)/'runtime/framework/multitask_r10_STOP';stop.parent.mkdir(parents=True);stop.touch()
            with patch.object(owner,'require_frozen_environment') as env:
                with self.assertRaises(InterruptedError):owner.load_resume_control('missing','missing','missing','missing','missing')
                env.assert_not_called()

    def fake_proc(self,root):
        proc=root/'proc';workspace=root/'project';workspace.mkdir()
        def process(pid,parent,script='unrelated',cwd=False):
            entry=proc/str(pid);entry.mkdir(parents=True)
            fields=['0']*20;fields[0]='S';fields[1]=str(parent);fields[19]=str(pid*10)
            (entry/'stat').write_text(str(pid)+' (fake) '+' '.join(fields))
            (entry/'cmdline').write_bytes(('python\0'+script+'\0').encode())
            if cwd:(entry/'cwd').symlink_to(workspace,target_is_directory=True)
            return entry
        process(1,0);(proc/'1/comm').write_text('systemd\n')
        process(80,1,'scripts/continue_environment_resumed_VAL_TEST_v827_r10.py')
        process(90,80,'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py')
        process(100,90)
        return proc,workspace,process

    def test_full_inventory_excludes_entire_bwrap_ancestor_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,workspace,process=self.fake_proc(Path(tmp));process(200,1,'scripts/run_task_executor_smoke_author_outcome_v743_r10.py',True)
            with patch.object(owner,'ROOT',workspace):result=owner.full_process_visibility(proc,pid=100,uid=os.getuid())
            self.assertEqual(result['excluded_self_and_ancestor_pids'],[1,80,90,100])
            self.assertEqual([x['pid'] for x in result['live_project_native_processes']],[200])
            self.assertFalse(result['same_uid_inventory_incomplete'])

    def test_unmatched_process_never_requires_denied_cwd_or_host_namespace_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,workspace,process=self.fake_proc(Path(tmp));process(200,1,'unrelated')
            with patch.object(owner,'ROOT',workspace),patch.object(owner.os,'readlink',side_effect=PermissionError) as readlink:
                result=owner.full_process_visibility(proc,pid=100,uid=os.getuid());readlink.assert_not_called()
            self.assertFalse(result['same_uid_inventory_incomplete']);self.assertFalse(result['host_namespace_links_claimed_readable'])

    def test_matching_sameUID_native_with_denied_cwd_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,workspace,process=self.fake_proc(Path(tmp));process(200,1,'scripts/run_formal_suite16_owner_v753_r10.py')
            with patch.object(owner,'ROOT',workspace),patch.object(owner.os,'readlink',side_effect=PermissionError):
                result=owner.full_process_visibility(proc,pid=100,uid=os.getuid())
            self.assertTrue(result['same_uid_inventory_incomplete'])
            self.assertIn('cwd unreadable',result['unresolved_matching_or_same_uid_processes'][0]['reason'])

    def test_original_v5_2_inventory_is_retained_without_claiming_all_metadata_readable(self):
        legacy=dict(host_visibility_available=True,same_uid_inventory_incomplete=True,original='actual-value')
        with patch.object(owner,'legacy_process_visibility',return_value=legacy),patch.object(owner,'full_process_visibility',return_value={'new_guard':True}):
            result=owner.process_visibility()
        self.assertEqual(result['legacy_v5_2_actual_result'],legacy)


if __name__=='__main__':unittest.main()
