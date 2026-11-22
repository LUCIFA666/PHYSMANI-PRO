"""CPU/temp-only six-tier audit contracts; no actual source graph or Native."""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from our_mani.multitask import composite_TEST_reboot_resume_v1055 as core
from our_mani.multitask.evaluation import digest
from scripts import audit_reboot_TEST_results_v1057_r10 as audit


def fixture_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
fifth_fakes=fixture_module('CPU1057_fifth1042_fakes','test_composite_TEST_results_v1042.py')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.parent=fifth_fakes.AuditTests();self.parent.setUp();self.fake=self.parent.fake

    def grid(self,original=134,middle=584,fourth_imported=1266,fifth_imported=1409):
        fc,pc,gc,oc,orec,mrec,grec,prec,fp=self.parent.grid(original,middle,fourth_imported,fifth_imported)
        ids=[r['job_id'] for r in fp['jobs']];saved_ids=ids[:fifth_imported+4]
        adopted=set(ids[:fifth_imported+2]);rejected=set(ids[fifth_imported+2:fifth_imported+4]);saved=set(saved_ids)
        fs=deepcopy(prec['status']);fs.update(completed={k:deepcopy(prec['status']['completed'][k])
            if k in prec['status']['completed'] else dict(complete=True,native_success=False,motion_pass=None,
                returncode=0,timed_out=False,interrupted=False,automatic_retry=False) for k in saved_ids},
            failed={},policy_sha256=fc['policy_sha256'],drain_requested=False,active=[ids[fifth_imported+4]],checking=[ids[fifth_imported+5]],
            native_slots=1,GPU_slots=1)
        c=deepcopy(fc);c.update(schema=core.CONTROL_SCHEMA,previous_runtime=core.PREVIOUS,
            previous_controller_runtime=core.PREVIOUS_CONTROLLER,explicit_host_reboot_recovery=True)
        partitions=dict(completed=adopted,failed=set(),remaining=set(ids)-adopted,
            saved_completed=saved,rejected_saved_completed=rejected)
        for name,values in partitions.items():
            c[name+'_job_ids']=sorted(values);c[name+'_key_sha256']=digest(sorted(values))
        boot=dict(schema='r10_readonly_host_reboot_TEST_evidence_observation_v1054',audit_complete=True,
            formal_full_TEST_complete=False,inputs_unchanged_through_audit=True,old_outputs_modified=False,
            missing_evidence_reconstructed=False,unknown_counted_as_NativeFalse=False,success_based_case_selection=False,
            new_terminal_claimed_for_interrupted_owner=False,Native_started=False,scientific_inputs_read=False,
            raw_data_read=False,model_read=False,full_source_graph_read=False,old_owner_process=dict(same_process_alive=False),
            current_boot_id='00000000-0000-0000-0000-000000000001',unregistered_clean_native_terminals=[],
            saved_status_completed_count=len(saved),recoverable_complete_count=len(adopted),damaged_registered_count=len(rejected),
            still_unclosed_count=1920-len(adopted),unknown_interrupted_cases=[])
        boot['completed_evidence']=[dict(job_id=k,recoverable_complete=k in adopted,
            native_success=fs['completed'][k]['native_success'] if k in adopted else None,
            errors=[] if k in adopted else ['missing_or_zero_byte_case']) for k in saved_ids]
        for name,values in dict(accepted_completed_job_ids=adopted,rejected_saved_completed_job_ids=rejected,
                saved_completed_job_ids=saved,remaining_job_ids=set(ids)-adopted).items():
            boot[name]=sorted(values);boot[name+'_sha256']=digest(sorted(values))
        np=dict(fp,jobs=core.resumed_jobs(fp,c,c['execution_entry']),imported_completed_cases=len(adopted))
        return c,fc,pc,gc,oc,orec,mrec,grec,prec,dict(plan=fp,status=fs),boot,np

    def test_six_tier_exact1920_grid_and_reboot_partition_with_literal_history(self):
        for parts in ((134,584,1266,1409),(7,19,31,49)):
            values=self.grid(*parts);before=deepcopy(values)
            a,b,c,d=parts
            self.assertEqual(audit.require_six_tier_plan(*values),{audit.ORIGINAL:a,audit.INTERMEDIATE:b-a,
                audit.GRANDPARENT:c-b,audit.PREVIOUS:d-c,audit.FIFTH:2,audit.RUNTIME:1920-d-2})
            self.assertEqual(values,before)
            self.assertNotIn('terminal',values[-3]);self.assertEqual(len(values[-3]['status']['completed']),d+4)
            adopted=set(values[0]['completed_job_ids']);rejected=set(values[0]['rejected_saved_completed_job_ids'])
            rows={r['job_id']:r for r in values[-1]['jobs']}
            self.assertTrue(all(rows[k]['imported_completed'] for k in adopted))
            self.assertTrue(all(not rows[k]['imported_completed'] and rows[k]['evidence_runtime']==audit.RUNTIME for k in rejected))

    def test_reboot_never_filters_nativeFalse_or_adopts_damaged_evidence(self):
        values=self.grid();audit.require_six_tier_plan(*values)
        for mode in ('saved_count','drop_nativeFalse','adopt_damaged','damaged_as_false','unregistered','old_literal_changed','boot_filter','reboot_flag'):
            bad=deepcopy(values)
            if mode=='saved_count':bad[-2]['saved_status_completed_count']-=1
            elif mode=='drop_nativeFalse':bad[0]['completed_job_ids'].pop()
            elif mode=='adopt_damaged':bad[0]['completed_job_ids'].append(bad[0]['rejected_saved_completed_job_ids'][0])
            elif mode=='damaged_as_false':next(r for r in bad[-2]['completed_evidence'] if not r['recoverable_complete'])['native_success']=False
            elif mode=='unregistered':bad[-2]['unregistered_clean_native_terminals']=[dict(job_id='clean_but_unadopted')]
            elif mode=='old_literal_changed':bad[-3]['status']['completed'][bad[1]['completed_job_ids'][0]]['native_success']=True
            elif mode=='boot_filter':bad[-2]['success_based_case_selection']=True
            else:bad[0]['explicit_host_reboot_recovery']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_six_tier_plan(*bad)

    def test_all_four_parent_lineages_and_all1920_coordinates_remain_bound(self):
        base=self.grid()
        for mode in ('955','1010','1040','1055','rowsha','runtime','entry','out','missing','duplicate'):
            values=deepcopy(base);row=values[-1]['jobs'][1409]
            if mode in ('955','1010','1040','1055'):
                field={'955':'resume_lineage','1010':'resume_lineage_v1010','1040':'resume_lineage_v1040','1055':'resume_lineage_v1055'}[mode]
                row[field]['parent_row_sha256']='f'*64
            elif mode=='rowsha':row['previous_row_sha256']='f'*64
            elif mode=='runtime':row['evidence_runtime']=audit.RUNTIME
            elif mode=='entry':row['execution_entry']['kind']='original751'
            elif mode=='out':row['physical_argv'][row['physical_argv'].index('--out')+1]='invented/path'
            elif mode=='missing':values[-1]['jobs'].pop()
            else:values[-1]['jobs'][-1]=deepcopy(values[-1]['jobs'][0])
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_six_tier_plan(*values)

    def test_scientific_and_leaf_receipt_helper_bodies_remain_exact_frozen1042(self):
        def bodies(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        root=Path(__file__).parents[1];old=bodies(root/'scripts/audit_composite_TEST_results_v1042_r10.py');new=bodies(root/audit.SOURCE)
        for name in ('require_case_outcome','audit_complete_grid','summarize','require_tier_launch'):
            self.assertEqual(old[name],new[name],name)

    def launch(self,runtime,check=True):
        values=self.grid();row=next(r for r in values[-1]['jobs'] if r['evidence_runtime']==runtime)
        control=values[0];bindings=self.fake.bindings();identity=dict(sha256='6'*64)
        pid={audit.ORIGINAL:7530,audit.INTERMEDIATE:9020,audit.GRANDPARENT:9560,audit.PREVIOUS:10110,
            audit.FIFTH:10410,audit.RUNTIME:10560}[runtime]
        owner=dict(pid=pid,start_ticks=pid*10)
        actual=dict(argv=deepcopy(row['argv'])+(['--check'] if check else []),check_only=check,
            launcher_pid=pid+1,launcher_start_ticks=pid+2,owner_pid=pid,execution_protocol_changed=True,
            native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
            expected_identity_sha256=None if check else identity['sha256'],**deepcopy(bindings))
        if runtime!=audit.ORIGINAL:
            environment,receipt=self.fake.environment(pid);receipt['observer_start_ticks']=owner['start_ticks']
            receipt.update(current_mount_namespace='mnt:[fake'+str(pid)+']',mount_namespace='mnt:[fake'+str(pid)+']')
            control.update(environment);owner['frozen_environment']=receipt
            actual.update(execution_entry=row['execution_entry'],physical_argv=row['physical_argv']+(['--check'] if check else []),
                inherited_frozen_environment=receipt,current_mount_namespace=receipt['current_mount_namespace'])
        return actual,row,bindings,identity,owner,control

    def test_all_six_real_CPU_native_tiers_keep_original_leaf_receipt_validation(self):
        for runtime in (audit.ORIGINAL,audit.INTERMEDIATE,audit.GRANDPARENT,audit.PREVIOUS,audit.FIFTH,audit.RUNTIME):
            for check in (True,False):
                actual,row,bindings,identity,owner,control=self.launch(runtime,check)
                audit.require_tier_launch(actual,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)
                changed=deepcopy(actual);changed['owner_pid']=123
                with self.subTest(runtime=runtime,check=check),self.assertRaises(ValueError):
                    audit.require_tier_launch(changed,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)

    def test_history_rechecks_exact_generations_without_rewriting_old_locks(self):
        for runtime,count in ((audit.ORIGINAL,5),(audit.INTERMEDIATE,4),(audit.GRANDPARENT,3),(audit.PREVIOUS,2),(audit.FIFTH,1)):
            batch=Mock();maps=[{'map':i} for i in range(5)];p=dict(path='CPU/only.json',sha256='a'*64)
            audit.require_historical_pointer(batch,*maps,p,runtime)
            self.assertEqual([c.args[1] for c in batch.locked_pointer.call_args_list],maps[:count])

    def owner_headers(self,root,*,fallback=False,lower=False):
        args,kw=self.parent.owner_headers(root,fallback=fallback,lower=lower)
        fc,pc,gc,oc,original,middle,grandparent,parent,fifth=args
        source=root/audit.OWNER;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('# CPU fake1056 source\n')
        c=deepcopy(fc);cp=self.fake.point(audit.CONTROL);lp=self.fake.point(audit.LOCK)
        c.update(previous_resume_control=kw['control_pointer'],previous_resume_control_lock=kw['lock_pointer'],
            completed_job_ids=['CPU_done_'+str(i) for i in range(1411)],resource_profile_changed=False,
            previous_resource_profile=deepcopy(fc['resource_profile']),previous_resource_qualification=deepcopy(fc['resource_qualification']),
            reboot_audit=self.fake.point('runtime/framework/reboot_TEST_evidence_v1054.json'))
        records=deepcopy(fifth);env=deepcopy(fifth['launch']['frozen_environment']);env.update(pid=10560,observer_pid=10560,
            observer_start_ticks=105600,current_mount_namespace='mnt:[fake1056]',mount_namespace='mnt:[fake1056]')
        for v in records.values():
            v.update(resume_control=cp,resume_control_lock=lp,frozen_environment=deepcopy(env),
                previous_resume_control=c['previous_resume_control'],previous_resume_control_lock=c['previous_resume_control_lock'],
                resource_profile_changed=False,previous_resource_profile=c['previous_resource_profile'],
                previous_resource_qualification=c['previous_resource_qualification'],reboot_audit=c['reboot_audit'],
                infrastructure_recovery=True,boot_id='00000000-0000-0000-0000-000000000001')
        records['launch'].update(pid=10560,start_ticks=105600,source_sha256=audit.previous_audit.file_sha(source),
            parent_stage_owner=audit.physical_owner.pointer(root/'scripts/run_composite_TEST_owner_v1041_r10.py'),imported_completed_cases=1411)
        expected={'--suite':c['suite']['path'],'--suite-lock':c['suite_lock']['path'],'--source-inputs':c['source_inputs']['path'],
            '--validation-release':c['validation_release']['path'],'--runtime':audit.RUNTIME,'--out':core.OUTPUT,
            '--resume-control':cp['path'],'--resume-lock':lp['path'],'--resume-lock-sha256':lp['sha256']}
        records['launch']['argv']=[audit.OWNER,*[x for pair in expected.items() for x in pair]]
        records['plan']['imported_completed_cases']=1411;records['status'].update(imported_completed=1411,CPU_launched=509)
        records['terminal'].update(imported_completed_cases=1411,resumed_new_cases=509)
        fifth.pop('terminal');fifth['status'].update(drain_requested=False,active=['interrupted_native'],checking=['interrupted_CPU'],native_slots=1,GPU_slots=1)
        return [c,fc,pc,gc,oc,original,middle,grandparent,parent,fifth,records],dict(source_sha=kw['source_sha'],driver=kw['driver'],guard=kw['guard'],
            control_pointer=cp,lock_pointer=lp,audit_environment=kw['audit_environment'])

    def test_six_owner_headers_keep_abrupt1041_interruption_and_actual_capacity(self):
        for mode in ('default','lower','fallback'):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name)
                with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                    args,kw=self.owner_headers(root,fallback=mode=='fallback',lower=mode=='lower');before=deepcopy(args)
                    owners=audit.require_bindings(*args,**kw)
                    self.assertEqual(len(owners),6);self.assertEqual(args,before)
                    self.assertNotIn('terminal',args[-2]);self.assertFalse(args[-1]['status']['drain_requested'])
                    self.assertFalse(args[0]['resource_profile_changed'])
                    if mode=='lower':self.assertNotEqual(args[0]['resource_profile'],args[0]['resource_capacity_envelope'])

    def test_bad_sixth_quota_boot_or_any_faked1041_terminal_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                base,kw=self.owner_headers(root)
                for mode in ('source','PID','namespace','actual','envelope_as_actual','profile_changed','parent_source','parent_control','unfinished','ready','stop','protocol','faked_old_terminal','boot','reboot_proof','infra'):
                    args=deepcopy(base);new=args[-1]
                    if mode=='source':new['launch']['source_sha256']='f'*64
                    elif mode=='PID':new['launch'].update(pid=10410,start_ticks=104100)
                    elif mode=='namespace':new['launch']['frozen_environment']=deepcopy(args[-2]['launch']['frozen_environment'])
                    elif mode=='actual':new['launch']['max_native_slots']=8
                    elif mode=='envelope_as_actual':new['terminal']['resource_capacity_envelope']={}
                    elif mode=='profile_changed':new['terminal']['resource_profile_changed']=True
                    elif mode=='parent_source':new['launch']['parent_stage_owner']=kw['control_pointer']
                    elif mode=='parent_control':new['plan']['previous_resume_control']={}
                    elif mode=='unfinished':new['terminal']['complete']=False
                    elif mode=='ready':new['status']['ready_jobs']=['pending']
                    elif mode=='stop':new['terminal']['stop_requested']=True
                    elif mode=='protocol':new['terminal']['author_outcome_protocol']={}
                    elif mode=='faked_old_terminal':args[-2]['terminal']={}
                    elif mode=='boot':new['terminal']['boot_id']='other_boot'
                    elif mode=='reboot_proof':new['terminal']['reboot_audit']={}
                    else:new['terminal']['infrastructure_recovery']=False
                    with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_bindings(*args,**kw)

    def test_original749_530_native_bool_clock_cleanup_and_unknown_motion_rules(self):
        for motion in (True,False,None):
            case,identity,protocol=self.fake.case(ticks=0,success=False,motion=motion)
            self.assertEqual(audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE),dict(native_success=False,motion_pass=motion,ticks=0))
        for mode in ('nativeNone','cleanup','RML','clock'):
            case,identity,protocol=self.fake.case(ticks=0,success=False,motion=None)
            if mode=='nativeNone':case['official_task_success']=None
            elif mode=='cleanup':case['cleanup_completed']=False
            elif mode=='RML':case['common_author_RML_parameters']['actual_parameters']['planning_acceleration']=6
            else:case['counted_clock']['ticks']=1
            with self.subTest(mode=mode),self.assertRaises(Exception):audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE)

    def test_scientific_full1920_false_None_zero_tick_grid_is_not_filtered(self):
        cases,manifest=self.parent.parent.parent.grid_cases();before=deepcopy(cases)
        with patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE),patch.object(audit,'require_attestation') as attest:
            result=audit.audit_complete_grid(cases,manifest,dict(common_author_RML_profile={}))
        self.assertEqual(result['method_cases'],1920);self.assertEqual(result['native_successes_by_method'],{m:0 for m in audit.METHODS})
        self.assertEqual(result['measured_motion_unknown_by_method'],{m:320 for m in audit.METHODS})
        self.assertIsNone(result['ours_measured_motion_passed']);self.assertEqual(attest.call_count,1920);self.assertEqual(cases,before)

    def test_complete_grid_rejects_missing_duplicate_changed_sources_and_infra_failures(self):
        original,manifest=self.parent.parent.parent.grid_cases()
        for mode in ('missing','duplicate','native_unknown','policy','fitted','infra','method','source'):
            cases=deepcopy(original)
            if mode=='missing':cases.pop()
            elif mode=='duplicate':cases[-1]=deepcopy(cases[0])
            elif mode=='native_unknown':cases[0]['official_task_success']=None
            elif mode=='policy':cases[0]['full16_frozen_execution']['policy_sha256']='f'*64
            elif mode=='fitted':cases[0]['TEST_used_for_fitting']=True
            elif mode=='infra':cases[0]['infrastructure_error']='lost_evidence'
            elif mode=='method':cases[0]['method']='unapproved'
            else:cases[0]['source_raw_episode_id']='not_an_original_source'
            with self.subTest(mode=mode),patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE),patch.object(audit,'require_attestation'),self.assertRaises(ValueError):
                audit.audit_complete_grid(cases,manifest,dict(common_author_RML_profile={}))

    def test_qualification_metadata_scopes_follow_actual_selected_capacity(self):
        from scripts import prepare_composite_TEST_resume_v1040_r10 as preparation
        names={'qualified1031':'runtime/framework/concurrent_TRAIN_resource_canary_v1031/plan.json',
            'qualified1005_fallback':'runtime/framework/concurrent_TRAIN_resource_canary_v1005/plan.json'}
        for kind,name in names.items():
            ledger=Mock();batch=object();files={'CPU_fake':True}
            audit.require_qualification_inputs(dict(inputs={name:'a'*64}),dict(resource_capacity_source=kind),preparation,batch,files,ledger)
            ledger.frozen.assert_called_once_with(batch,files,dict(path=name,sha256='a'*64),parse=False)
            for invalid in (names[next(k for k in names if k!=kind)],'models/unopened.pt','data/unopened.hdf5','../outside.json'):
                ledger=Mock()
                with self.subTest(kind=kind,invalid=invalid),self.assertRaises(ValueError):
                    audit.require_qualification_inputs(dict(inputs={invalid:'a'*64}),dict(resource_capacity_source=kind),preparation,batch,files,ledger)
                ledger.frozen.assert_not_called()

    def test_full_completion_gate_precedes_all_graph_and_TEST_reads(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/audit.CONTROL;p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(validation_release=self.fake.point('CPU/release.json'),completed_job_ids=[])))
            lock=root/audit.LOCK;lock.write_text(json.dumps(dict(schema=core.LOCK_SCHEMA,approved_for_execution=True,
                control=dict(path=audit.CONTROL,sha256=audit.previous_audit.file_sha(p)),
                files={audit.CONTROL:audit.previous_audit.file_sha(p)})))
            leaf=root/audit.RUNTIME;leaf.mkdir(parents=True)
            for k in ('launch','plan','status','terminal'):(leaf/(k+'.json')).write_text('{}')
            denied=Mock(side_effect=AssertionError('early full graph/TEST access'))
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'require_complete_VAL_release',return_value={}),patch.object(audit,'verify_inputs',denied),patch.object(audit,'LookupBatch',denied),patch.object(audit,'read_formal_source',denied),self.assertRaisesRegex(ValueError,'complete1056'):
                audit.main(['--resume-lock-sha256',audit.previous_audit.file_sha(lock)])
            denied.assert_not_called();self.assertFalse((root/audit.OUTPUT).exists())

    def test_STOP_external_lock_and_existing_result_do_not_touch_graph(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/audit.LOCK;p.parent.mkdir(parents=True);p.write_text('{}')
            denied=Mock(side_effect=AssertionError('actual graph touched'))
            with patch.object(audit,'ROOT',root),patch.object(audit,'verify_inputs',denied),self.assertRaisesRegex(ValueError,'external exact'):
                audit.main(['--resume-lock-sha256','f'*64])
            stop=root/'runtime/framework/multitask_r10_STOP';stop.parent.mkdir(parents=True);stop.touch()
            with patch.object(audit,'ROOT',root),patch.object(audit,'EvidenceLedger') as ledger,self.assertRaisesRegex(ValueError,'STOP'):
                audit.main(['--resume-lock-sha256','f'*64])
            ledger.assert_not_called();stop.unlink()
            out=root/audit.OUTPUT;out.parent.mkdir(parents=True,exist_ok=True);out.write_text('original')
            with patch.object(audit,'ROOT',root),self.assertRaisesRegex(ValueError,'never overwrite'):audit.main(['--resume-lock-sha256','f'*64])
            self.assertEqual(out.read_text(),'original');denied.assert_not_called()


if __name__=='__main__':unittest.main()
