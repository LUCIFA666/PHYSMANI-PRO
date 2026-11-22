"""Synthetic CPU/temp957 audits; never main on experiment data or Native."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from our_mani.multitask import composite_TEST_resume_v955 as core
from our_mani.multitask.evaluation import digest
from scripts import audit_composite_TEST_results_v957_r10 as audit


def fixture_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


fakes=fixture_module('CPU957_original906_fakes','test_concurrent_resumed_formal_TEST_audit_v906.py')


class AuditTests(unittest.TestCase):
    def setUp(self):self.fake=fakes.ConcurrentTESTAuditTests()

    def grid(self,old_count=134,adopted=584):
        old_control,original_plan,original_status,original_terminal,parent_plan=self.fake.grid()
        ids=[r['job_id'] for r in original_plan['jobs']]
        old_control.update(completed_job_ids=ids[:old_count],remaining_job_ids=ids[old_count:],
            completed_key_sha256=digest(sorted(ids[:old_count])),remaining_key_sha256=digest(sorted(ids[old_count:])))
        def record(i):return dict(complete=True,native_success=bool(i%3),motion_pass=None if i%3==0 else False,
            returncode=0,timed_out=False,interrupted=False,automatic_retry=False)
        original_status['completed']={name:record(i) for i,name in enumerate(ids[:old_count])}
        original_terminal['completed']=old_count
        parent_plan.update(jobs=audit.physical_owner.resumed_jobs(original_plan,old_control,old_control['execution_entry']),
            imported_completed_cases=old_count,automatic_retry=False)
        failed=ids[adopted:adopted+6]
        control=deepcopy(old_control);control.update(schema=core.CONTROL_SCHEMA,previous_runtime=core.PREVIOUS,
            previous_controller_runtime=core.PREVIOUS_CONTROLLER,completed_job_ids=ids[:adopted],failed_job_ids=failed,
            remaining_job_ids=ids[adopted:],completed_key_sha256=digest(sorted(ids[:adopted])),
            failed_key_sha256=digest(sorted(failed)),remaining_key_sha256=digest(sorted(ids[adopted:])))
        status=dict(split='test',expected=1920,completed={name:record(i) for i,name in enumerate(ids[:adopted])},
            failed={name:dict(complete=False,automatic_retry=False,error='synthetic environment failure') for name in failed},
            active=[],checking=[],native_slots=0,GPU_slots=0,drain_requested=True,stop_requested=False,automatic_retry=False)
        terminal=dict(expected=1920,completed=adopted,failed=6,complete=False,drained=True,drain_requested=True,
            stop_requested=False,automatic_retry=False,policy_sha256=control['policy_sha256'])
        plan=dict(parent_plan,jobs=core.resumed_jobs(parent_plan,control,control['execution_entry']),
            imported_completed_cases=adopted)
        return control,old_control,original_plan,original_status,original_terminal,parent_plan,status,terminal,plan

    def test_contract_exact_original_registry_and_three_tiers_134_450_1336(self):
        values=self.grid();before=deepcopy(values);counts=audit.require_three_tier_plan(*values)
        self.assertEqual(counts,{audit.ORIGINAL:134,audit.PREVIOUS:450,audit.RUNTIME:1336})
        self.assertEqual(values,before);self.assertEqual(set(core.TASKS),set(audit.TASKS))
        self.assertEqual(core.AUDITOR,audit.SOURCE)
        rows=values[-1]['jobs'];self.assertTrue(rows[134]['imported_completed'])
        self.assertFalse(rows[134]['resume_lineage']['parent_imported_completed'])
        self.assertEqual(rows[134]['evidence_runtime'],audit.PREVIOUS)
        self.assertIsNone(values[6]['completed'][rows[0]['job_id']]['motion_pass'])

    def test_dynamic_partition_is_not_hardcoded_to_historical_584(self):
        values=self.grid(old_count=7,adopted=19)
        self.assertEqual(audit.require_three_tier_plan(*values),{audit.ORIGINAL:7,audit.PREVIOUS:12,audit.RUNTIME:1901})

    def test_parent_row_flattening_changed_entry_and_missing_lineage_rejected(self):
        for mode in ('902_as753','902_as956','753_as902','lineage','old_marker','parent_sha','outer','physical','old_worker','filterFalse'):
            values=self.grid();row=values[-1]['jobs'][140]
            if mode=='902_as753':row['evidence_runtime']=audit.ORIGINAL
            elif mode=='902_as956':row['evidence_runtime']=audit.RUNTIME
            elif mode=='753_as902':values[-1]['jobs'][0]['evidence_runtime']=audit.PREVIOUS
            elif mode=='lineage':row.pop('resume_lineage')
            elif mode=='old_marker':row['resume_lineage']['parent_imported_completed']=True
            elif mode=='parent_sha':row['resume_lineage']['parent_row_sha256']='f'*64
            elif mode=='outer':row['argv'][2]=audit.physical_owner.ORIGINAL_ENTRY
            elif mode=='physical':row['physical_argv'][2]=audit.physical_owner.BATCH_ENTRY
            elif mode=='old_worker':row['previous_worker_sha256']='f'*64
            else:values[6]['completed'].pop(row['job_id'])
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_three_tier_plan(*values)

    def test_missing_duplicate_or_asymmetric_sources_and_infra_as_nativeFalse_rejected(self):
        for mode in ('missing','duplicate','source','failed_adopted','changed_historical_result'):
            values=self.grid()
            if mode=='missing':values[-1]['jobs'].pop()
            elif mode=='duplicate':values[-1]['jobs'][-1]=deepcopy(values[-1]['jobs'][0])
            elif mode=='source':values[5]['jobs'][-1]['raw']='different_source'
            elif mode=='failed_adopted':
                k=values[0]['failed_job_ids'][0];values[0]['completed_job_ids'].append(k)
                values[0]['completed_key_sha256']=digest(sorted(values[0]['completed_job_ids']))
            else:values[6]['completed'][values[0]['completed_job_ids'][0]]['native_success']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_three_tier_plan(*values)

    def launch(self,runtime,check=True):
        values=self.grid();rows=values[-1]['jobs'];row=next(r for r in rows if r['evidence_runtime']==runtime)
        control=values[0];bindings=self.fake.bindings();identity=dict(sha256='6'*64)
        pid={audit.ORIGINAL:7530,audit.PREVIOUS:9020,audit.RUNTIME:9560}[runtime]
        stage_owner=dict(pid=pid,start_ticks=90200 if runtime!=audit.ORIGINAL else 75300)
        actual=dict(argv=deepcopy(row['argv'])+(['--check'] if check else []),check_only=check,
            launcher_pid=pid+1,launcher_start_ticks=pid+2,owner_pid=pid,execution_protocol_changed=True,
            native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
            expected_identity_sha256=None if check else identity['sha256'],**deepcopy(bindings))
        if runtime!=audit.ORIGINAL:
            environment,receipt=self.fake.environment(pid);control.update(environment);stage_owner['frozen_environment']=receipt
            actual.update(execution_entry=row['execution_entry'],physical_argv=row['physical_argv']+(['--check'] if check else []),
                inherited_frozen_environment=receipt,current_mount_namespace=receipt['current_mount_namespace'])
        return actual,row,bindings,identity,stage_owner,control

    def test_all_three_actual_launch_owners_keep_real_argv_not_imported_boolean(self):
        for runtime in (audit.ORIGINAL,audit.PREVIOUS,audit.RUNTIME):
            for check in (True,False):
                actual,row,binding,identity,stage_owner,control=self.launch(runtime,check)
                audit.require_tier_launch(actual,row,binding,check=check,identity=identity,stage_owner=stage_owner,control=control)
                self.assertEqual(row['imported_completed'],runtime!=audit.RUNTIME)
                changed=deepcopy(actual);changed['owner_pid']=123
                with self.subTest(runtime=runtime,check=check),self.assertRaises(ValueError):
                    audit.require_tier_launch(changed,row,binding,check=check,identity=identity,stage_owner=stage_owner,control=control)

    def test_real902_import_cannot_acquire_956_namespace_or_direct751_claim(self):
        actual,row,binding,identity,stage_owner,control=self.launch(audit.PREVIOUS)
        for mode in ('namespace','entry','physical','receipt','sourcebinding'):
            changed=deepcopy(actual)
            if mode=='namespace':changed['current_mount_namespace']='mnt:[otherowner]'
            elif mode=='entry':changed['execution_entry']=dict(kind='original751',pointer=binding['admission_driver'])
            elif mode=='physical':changed['physical_argv'].append('--invented')
            elif mode=='receipt':changed['inherited_frozen_environment']['observer_pid']=9560
            else:changed['case_worker']['sha256']='f'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                audit.require_tier_launch(changed,row,binding,check=True,identity=identity,stage_owner=stage_owner,control=control)

    def test_original753_never_gets_fabricated_namespace_or901_evidence(self):
        actual,row,binding,identity,stage_owner,control=self.launch(audit.ORIGINAL)
        for field in ('execution_entry','physical_argv','inherited_frozen_environment','current_mount_namespace'):
            changed=dict(actual,**{field:{}})
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.require_tier_launch(changed,row,binding,check=True,identity=identity,stage_owner=stage_owner,control=control)

    def case(self,**kwargs):return self.fake.case(**kwargs)

    def test_original749_530_outcome_checks_retain_false_None_and_zero_ticks(self):
        for task in ('beat_the_rotating_buzz',next(iter(fakes.BLOCKS))):
            for method in audit.METHODS:
                for motion in (False,None,True):
                    case,identity,protocol=self.case(task=task,method=method,ticks=0,success=False,motion=motion)
                    value=audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE)
                    self.assertEqual(value,dict(native_success=False,motion_pass=motion,ticks=0))
                    self.assertIs(value['motion_pass'],motion)
        case,identity,protocol=self.case(ticks=2,success=False,motion=False)
        self.assertFalse(audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE)['motion_pass'])

    def test_cleanup_identity_clock_protocol_or_RML_cannot_be_diagnostic_exceptions(self):
        for mode in ('cleanup','identity','clock','protocol','RML','native_not_bool','domain'):
            case,identity,protocol=self.case(ticks=0,success=False,motion=None)
            if mode=='cleanup':case['cleanup_completed']=False
            elif mode=='identity':identity['sha256']='f'*64
            elif mode=='clock':case['counted_clock']['ticks']=1
            elif mode=='protocol':case['author_outcome_protocol']['runtime_consumer']['sha256']='f'*64
            elif mode=='RML':case['common_author_RML_parameters']['actual_parameters']['planning_acceleration']=6
            elif mode=='native_not_bool':case['official_task_success']=0
            else:case['component_domain']['purpose']='train_qualification'
            with self.subTest(mode=mode),self.assertRaises((ValueError,fakes.MultitaskProtocolError)):
                audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE)

    def grid_cases(self):
        sources=[dict(task=t,raw_episode_id='CPU_'+str(i),path='/UNOPENED_TEST/'+t+'/'+str(i)) for t in audit.TASKS for i in range(40)]
        manifest=dict(split='test',episodes=sources,mapping_status='verified',episode_count=640,
            training_calibration_or_parameter_selection_used=False)
        cases=[]
        for i,source in enumerate(sources):
            for method in audit.METHODS:
                motion=None if i%2==0 else False
                cases.append(dict(task=source['task'],source_raw_episode_id=source['raw_episode_id'],method=method,
                    split='test',complete=True,cleanup_completed=True,infrastructure_error=None,TEST_used_for_fitting=False,
                    official_task_success=False,native_terminal_failure=False,counted_clock=dict(ticks=0),
                    full16_frozen_execution=dict(policy_sha256=self.fake.POLICY,source_row_sha256=digest(source)),
                    all_stage_counted_joint_motion=dict(all_stage_actual_window_checks_passed=motion),
                    common_author_RML_parameters={'synthetic':True}))
        return cases,manifest

    def test_full1920_scientific_grid_keeps_all_False_and_reports_motion_unknown_separately(self):
        cases,manifest=self.grid_cases();before=deepcopy(cases)
        with patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE), \
                patch.object(audit,'require_attestation') as attestation:
            result=audit.audit_complete_grid(cases,manifest,dict(common_author_RML_profile={}))
        self.assertEqual(result['method_cases'],1920);self.assertEqual(result['source_cases'],640)
        self.assertEqual(result['native_successes_by_method'],{m:0 for m in audit.METHODS})
        self.assertEqual(result['measured_motion_unknown_by_method'],{m:320 for m in audit.METHODS})
        self.assertEqual(result['measured_motion_violations_by_method'],{m:320 for m in audit.METHODS})
        self.assertIsNone(result['ours_measured_motion_passed']);self.assertEqual(cases,before)
        self.assertEqual(attestation.call_count,1920)
        summary=audit.summarize(cases)
        self.assertEqual(len(summary),16)
        for row in summary:
            for value in row['methods'].values():
                self.assertEqual(value['episodes'],40);self.assertEqual(value['zero_tick_cases'],40)
                self.assertEqual(value['measured_motion_unknown_cases'],20);self.assertEqual(value['measured_motion_violation_cases'],20)
                self.assertIsNone(value['actual_window_acceleration_peak'])

    def test_full_grid_rejects_missing_duplicate_source_policy_and_nonformal_scope(self):
        cases,manifest=self.grid_cases()
        with patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE), \
                patch.object(audit,'require_attestation'):
            for mode in ('missing','duplicate','policy','source','demo','fit','infra'):
                rows=deepcopy(cases)
                if mode=='missing':rows.pop()
                elif mode=='duplicate':rows[-1]=deepcopy(rows[0])
                elif mode=='policy':rows[-1]['full16_frozen_execution']['policy_sha256']='f'*64
                elif mode=='source':rows[-1]['full16_frozen_execution']['source_row_sha256']='f'*64
                elif mode=='demo':rows[-1]['demo_only']=True
                elif mode=='fit':rows[-1]['TEST_used_for_fitting']=True
                else:rows[-1]['infrastructure_error']='partial'
                with self.subTest(mode=mode),self.assertRaises(ValueError):audit.audit_complete_grid(rows,manifest,dict(common_author_RML_profile={}))

    def test_ledger_temp_evidence_sha_alias_and_close_changes_are_checked(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'evidence.json';path.write_text('{"native":false}')
            ledger=audit.EvidenceLedger(root,cache=root/'private_cache')
            p=ledger.pointer('evidence.json');self.assertEqual(ledger.json('evidence.json'),{'native':False})
            with self.assertRaises(ValueError):ledger.read('evidence.json',expected='f'*64)
            alias=root/'alias.json';alias.symlink_to(path)
            with self.assertRaises(ValueError):ledger.pointer('alias.json')
            result=ledger.close();self.assertEqual(result[str(path)],p['sha256'])
            with self.assertRaises(ValueError):ledger.json('evidence.json')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'evidence.json';path.write_text('{}');ledger=audit.EvidenceLedger(root,cache=root/'private_cache')
            ledger.pointer('evidence.json');path.write_text('{"changed":true}')
            with self.assertRaises(Exception):ledger.close()

    def owner_headers(self,root):
        args,kwargs=self.fake.owner_records(root)
        old_control,original_launch,old_launch,old_plan,old_status,old_terminal=args
        old_control['suite_lock']['sha256']=kwargs['suite_lock_sha']
        old_cp=kwargs['resume_control'];old_lp=kwargs['resume_lock']
        control=deepcopy(old_control);control.update(completed_job_ids=['CPU_done_'+str(i) for i in range(584)],
            previous_resume_control=old_cp,previous_resume_control_lock=old_lp,
            environment_recheck=self.fake.point('CPU/environment_recheck.json'))
        cp=self.fake.point(audit.CONTROL);lp=self.fake.point(audit.LOCK)
        source=root/audit.OWNER;source.write_text('# invented956 owner source\n')
        old_records=dict(launch=old_launch,plan=old_plan,status=old_status,terminal=old_terminal)
        records=deepcopy(old_records);_,environment=self.fake.environment(9560);environment['observer_start_ticks']=95600
        environment.update(current_mount_namespace='mnt:[invented956]',mount_namespace='mnt:[invented956]')
        for value in records.values():
            value.update(resume_control=cp,resume_control_lock=lp,frozen_environment=deepcopy(environment),
                previous_resume_control=old_cp,previous_resume_control_lock=old_lp,environment_recheck=control['environment_recheck'])
        launch=records['launch'];launch.update(pid=9560,start_ticks=95600,
            source_sha256=audit.previous_audit.file_sha(source),parent_stage_owner=audit.physical_owner.pointer(
                root/'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'),imported_completed_cases=584)
        expected={'--suite':control['suite']['path'],'--suite-lock':control['suite_lock']['path'],
            '--source-inputs':control['source_inputs']['path'],'--validation-release':control['validation_release']['path'],
            '--runtime':audit.RUNTIME,'--out':core.OUTPUT,'--resume-control':cp['path'],'--resume-lock':lp['path'],
            '--resume-lock-sha256':lp['sha256']}
        launch['argv']=[audit.OWNER,*[value for pair in expected.items() for value in pair]]
        records['plan']['imported_completed_cases']=584
        records['status'].update(imported_completed=584,CPU_launched=1336)
        records['terminal'].update(imported_completed_cases=584,resumed_new_cases=1336)
        # Keep the real historical drain distinct from the only complete owner.
        old_records['status'].update(drain_requested=True,failed={'CPU_failed':{'complete':False}})
        old_records['terminal'].update(complete=False,drained=True,drain_requested=True,completed=584,failed=6)
        kwargs=dict(source_sha=kwargs['source_sha'],driver=kwargs['driver'],guard=kwargs['guard'],
            control_pointer=cp,lock_pointer=lp,audit_environment=kwargs['expected_environment'])
        return [control,old_control,original_launch,old_records,records],kwargs

    def test_three_owner_headers_keep_incomplete_history_and_distinct_current_namespace(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                args,kwargs=self.owner_headers(root);before=deepcopy(args)
                result=audit.require_bindings(*args,**kwargs)
                self.assertEqual({k:v['pid'] for k,v in result.items()},
                    {audit.ORIGINAL:7530,audit.PREVIOUS:9020,audit.RUNTIME:9560})
                self.assertFalse(args[3]['terminal']['complete']);self.assertEqual(args[3]['terminal']['failed'],6)
                self.assertTrue(args[4]['terminal']['complete']);self.assertEqual(args,before)
                self.assertNotEqual(result[audit.RUNTIME]['frozen_environment'],kwargs['audit_environment'])

    def test_owner_header_wrong_PIN_scope_source_PID_or_incomplete_final_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                base,kwargs=self.owner_headers(root)
                for mode in ('new_source','old_source','sharedpid','oldargv','newargv','bad_pin','protocol',
                        'new_parent','new_previous_control','namespace_flatten','drain','STOP','unfinished','missing_completed','active'):
                    args=deepcopy(base);old=args[3];new=args[4]
                    if mode=='new_source':new['launch']['source_sha256']='f'*64
                    elif mode=='old_source':old['launch']['source_sha256']='f'*64
                    elif mode=='sharedpid':new['launch'].update(pid=9020,start_ticks=90200)
                    elif mode in ('oldargv','newargv'):
                        actual=(old if mode=='oldargv' else new)['launch']['argv'];actual[actual.index('--runtime')+1]='invented_runtime'
                    elif mode=='bad_pin':new['launch']['argv'][new['launch']['argv'].index('--resume-lock-sha256')+1]='f'*64
                    elif mode=='protocol':new['terminal']['author_outcome_protocol']['runtime_consumer']['sha256']='f'*64
                    elif mode=='new_parent':new['launch']['parent_stage_owner']['sha256']='f'*64
                    elif mode=='new_previous_control':new['status']['previous_resume_control']['sha256']='f'*64
                    elif mode=='namespace_flatten':
                        for value in new.values():value['frozen_environment']=deepcopy(old['launch']['frozen_environment'])
                    elif mode=='drain':new['terminal']['drain_requested']=True
                    elif mode=='STOP':new['status']['stop_requested']=True
                    elif mode=='unfinished':new['terminal']['CPU_ready_unexecuted']=['CPU_missing']
                    elif mode=='missing_completed':new['status']['completed'].pop(next(iter(new['status']['completed'])))
                    else:new['status']['active']=['CPU_active']
                    with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_bindings(*args,**kwargs)

    def test_main_wrong_external_lock_pin_never_reads_control_or_opens_TEST(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);lock=root/audit.LOCK;lock.parent.mkdir(parents=True);lock.write_text('{}')
            denied=Mock(side_effect=AssertionError('early gate touched true data or resource'))
            with patch.object(audit,'ROOT',root),patch.object(audit,'verify_inputs',denied), \
                    patch.object(audit.environment_audit,'capture_environment',denied),patch.object(audit,'LookupBatch',denied), \
                    patch.object(audit,'read_formal_source',denied),self.assertRaisesRegex(ValueError,'external exact955'):
                audit.main(['--resume-lock-sha256','f'*64])
            denied.assert_not_called();self.assertFalse((root/audit.OUTPUT).exists())

    def test_main_VAL_gate_failure_precedes_full_graph_data_and_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);control=root/audit.CONTROL;control.parent.mkdir(parents=True)
            value=dict(validation_release=dict(path='CPU/VAL_release.json',sha256='e'*64))
            control.write_text(json.dumps(value));cp=dict(path=audit.CONTROL,sha256=audit.previous_audit.file_sha(control))
            lock=root/audit.LOCK;lock.write_text(json.dumps(dict(schema=core.LOCK_SCHEMA,approved_for_execution=True,files={audit.CONTROL:cp['sha256']})))
            denied=Mock(side_effect=AssertionError('invalid VAL gate touched TEST graph'))
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'require_complete_VAL_release',side_effect=ValueError('CPU invalidVAL')), \
                    patch.object(audit,'verify_inputs',denied),patch.object(audit.environment_audit,'capture_environment',denied), \
                    patch.object(audit,'LookupBatch',denied),patch.object(audit,'read_formal_source',denied),self.assertRaisesRegex(ValueError,'CPU invalidVAL'):
                audit.main(['--resume-lock-sha256',audit.previous_audit.file_sha(lock)])
            denied.assert_not_called();self.assertFalse((root/audit.OUTPUT).exists())

    def test_STOP_precedes_lock_or_result_access(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);stop=root/'runtime/framework/multitask_r10_STOP';stop.parent.mkdir(parents=True);stop.write_text('CPU_STOP')
            with patch.object(audit,'ROOT',root),patch.object(audit,'EvidenceLedger') as ledger,self.assertRaisesRegex(ValueError,'STOP'):
                audit.main(['--resume-lock-sha256','e'*64])
            ledger.assert_not_called()


if __name__=='__main__':unittest.main()
