"""CPU-only four-tier contracts and temp failure gates; no experiment IO or main."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from our_mani.multitask import composite_TEST_resume_v1010 as core
from our_mani.multitask.evaluation import digest
from scripts import audit_composite_TEST_results_v1012_r10 as audit

def fixture_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
parent_fakes=fixture_module('CPU1012_parent957_fakes','test_composite_TEST_results_v957.py')

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.parent=parent_fakes.AuditTests();self.parent.setUp();self.fake=self.parent.fake

    def grid(self,old_count=134,parent_count=584,adopted=1200):
        pc,oc,op,os,ot,mp,ms,mt,pp=self.parent.grid(old_count=old_count,adopted=parent_count)
        ids=[r['job_id'] for r in pp['jobs']]
        def record(i):return dict(complete=True,native_success=False,motion_pass=None if i%2 else False,
            returncode=0,timed_out=False,interrupted=False,automatic_retry=False)
        ps=deepcopy(ms);ps.update(completed={k:deepcopy(ms['completed'][k]) if k in ms['completed'] else record(i)
            for i,k in enumerate(ids[:adopted])},failed={})
        pt=deepcopy(mt);pt.update(completed=adopted,failed=0)
        control=deepcopy(pc);control.update(schema=core.CONTROL_SCHEMA,previous_runtime=core.PREVIOUS,
            previous_controller_runtime=core.PREVIOUS_CONTROLLER,completed_job_ids=ids[:adopted],failed_job_ids=[],
            remaining_job_ids=ids[adopted:],completed_key_sha256=digest(sorted(ids[:adopted])),failed_key_sha256=digest([]),
            remaining_key_sha256=digest(sorted(ids[adopted:])))
        np=dict(pp,jobs=core.resumed_jobs(pp,control,control['execution_entry']),imported_completed_cases=adopted)
        return control,pc,oc,dict(plan=op,status=os,terminal=ot),dict(plan=mp,status=ms,terminal=mt),dict(plan=pp,status=ps,terminal=pt),np

    def test_four_tier_exact_parent_lineages_and_dynamic_completed_partition(self):
        for c,p,n in ((134,584,1200),(7,19,31)):
            values=self.grid(c,p,n);before=deepcopy(values)
            self.assertEqual(audit.require_four_tier_plan(*values),{audit.ORIGINAL:c,audit.INTERMEDIATE:p-c,audit.PREVIOUS:n-p,audit.RUNTIME:1920-n})
            self.assertEqual(values,before)
            imported=values[-1]['jobs'][c]
            self.assertTrue(imported['imported_completed']);self.assertFalse(imported['resume_lineage']['parent_imported_completed'])
            self.assertTrue(imported['resume_lineage_v1010']['parent_imported_completed'])

    def test_whole956_row_SHA_and_both_prior_lineages_cannot_be_flattened(self):
        for mode in ('lineage955','lineage1010','oldsha','parent_sha','runtime','entry','out','old_result','missing','duplicate'):
            values=self.grid();r=values[-1]['jobs'][600]
            if mode=='lineage955':r['resume_lineage']['parent_row_sha256']='f'*64
            elif mode=='lineage1010':r.pop('resume_lineage_v1010')
            elif mode=='oldsha':r['previous_row_sha256']='f'*64
            elif mode=='parent_sha':r['resume_lineage_v1010']['parent_row_sha256']='f'*64
            elif mode=='runtime':r['evidence_runtime']=audit.RUNTIME
            elif mode=='entry':r['execution_entry']['kind']='original751'
            elif mode=='out':r['physical_argv'][r['physical_argv'].index('--out')+1]='invented/path'
            elif mode=='old_result':values[5]['status']['completed'][values[0]['completed_job_ids'][0]]['native_success']=True
            elif mode=='missing':values[-1]['jobs'].pop()
            else:values[-1]['jobs'][-1]=deepcopy(values[-1]['jobs'][0])
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_four_tier_plan(*values)

    def launch(self,runtime,check=True):
        values=self.grid();row=next(r for r in values[-1]['jobs'] if r['evidence_runtime']==runtime)
        control=values[0];bindings=self.fake.bindings();identity=dict(sha256='6'*64)
        pid={audit.ORIGINAL:7530,audit.INTERMEDIATE:9020,audit.PREVIOUS:9560,audit.RUNTIME:10110}[runtime]
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

    def test_all_four_actual_launch_tiers_keep_real_owner_argv_and_namespace(self):
        for runtime in (audit.ORIGINAL,audit.INTERMEDIATE,audit.PREVIOUS,audit.RUNTIME):
            for check in (True,False):
                actual,row,bindings,identity,owner,control=self.launch(runtime,check)
                audit.require_tier_launch(actual,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)
                self.assertEqual(row['imported_completed'],runtime!=audit.RUNTIME)
                changed=deepcopy(actual);changed['owner_pid']=123
                with self.subTest(runtime=runtime,check=check),self.assertRaises(ValueError):
                    audit.require_tier_launch(changed,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)

    def test_old753_never_acquires_invented901_or_namespace_fields(self):
        args=self.launch(audit.ORIGINAL)
        for field in ('execution_entry','physical_argv','inherited_frozen_environment','current_mount_namespace'):
            actual=deepcopy(args[0]);actual[field]={}
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.require_tier_launch(actual,args[1],args[2],check=True,identity=args[3],stage_owner=args[4],control=args[5])

    def test_historical_bytes_rechecked_in_exactly_the_declaring_maps(self):
        for runtime,count in ((audit.ORIGINAL,3),(audit.INTERMEDIATE,2),(audit.PREVIOUS,1)):
            batch=Mock();maps=[{'map':i} for i in range(3)];p=dict(path='CPU/only.json',sha256='a'*64)
            audit.require_historical_pointer(batch,*maps,p,runtime)
            self.assertEqual(batch.locked_pointer.call_count,count)
            self.assertEqual([c.args[1] for c in batch.locked_pointer.call_args_list],maps[:count])

    def test_scientific_full1920_false_None_zero_tick_grid_kept_without_filter(self):
        cases,manifest=self.parent.grid_cases();before=deepcopy(cases)
        with patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE),patch.object(audit,'require_attestation') as attest:
            result=audit.audit_complete_grid(cases,manifest,dict(common_author_RML_profile={}))
        self.assertEqual(result['method_cases'],1920);self.assertEqual(result['native_successes_by_method'],{m:0 for m in audit.METHODS})
        self.assertEqual(result['measured_motion_unknown_by_method'],{m:320 for m in audit.METHODS})
        self.assertIsNone(result['ours_measured_motion_passed']);self.assertEqual(attest.call_count,1920);self.assertEqual(cases,before)

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

    def owner_headers(self,root):
        args,kwargs=self.parent.owner_headers(root)
        pc,oc,original,middle,parent=args
        # The fake writes source bytes only under its isolated temporary root.
        source=root/audit.OWNER;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('# fake1011 source\n')
        c=deepcopy(pc);cp=self.fake.point(audit.CONTROL);lp=self.fake.point(audit.LOCK)
        c.update(previous_resume_control=kwargs['control_pointer'],previous_resume_control_lock=kwargs['lock_pointer'],
            completed_job_ids=['CPU_done_'+str(i) for i in range(1200)],
            resource_profile=dict(schema=audit.current_owner.PROFILE_SCHEMA,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16),
            resource_qualification=self.fake.point('CPU/new1005.json'),resource_profile_changed=True,
            previous_resource_profile=deepcopy(pc['resource_profile']),previous_resource_qualification=deepcopy(pc['resource_qualification']))
        records=deepcopy(parent);env=deepcopy(parent['launch']['frozen_environment']);env.update(pid=10110,observer_pid=10110,observer_start_ticks=101100,current_mount_namespace='mnt:[fake1011]',mount_namespace='mnt:[fake1011]')
        for v in records.values():
            v.update(resume_control=cp,resume_control_lock=lp,frozen_environment=deepcopy(env),
                resource_profile=c['resource_profile'],resource_qualification=c['resource_qualification'],
                previous_resume_control=c['previous_resume_control'],previous_resume_control_lock=c['previous_resume_control_lock'],
                resource_profile_changed=True,previous_resource_profile=c['previous_resource_profile'],previous_resource_qualification=c['previous_resource_qualification'])
        records['launch'].update(pid=10110,start_ticks=101100,source_sha256=audit.previous_audit.file_sha(source),
            parent_stage_owner=audit.physical_owner.pointer(root/'scripts/run_composite_TEST_owner_v956_r10.py'),imported_completed_cases=1200,
            max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4)
        expected={'--suite':c['suite']['path'],'--suite-lock':c['suite_lock']['path'],'--source-inputs':c['source_inputs']['path'],
            '--validation-release':c['validation_release']['path'],'--runtime':audit.RUNTIME,'--out':core.OUTPUT,
            '--resume-control':cp['path'],'--resume-lock':lp['path'],'--resume-lock-sha256':lp['sha256']}
        records['launch']['argv']=[audit.OWNER,*[x for pair in expected.items() for x in pair]]
        records['plan']['imported_completed_cases']=1200;records['status'].update(imported_completed=1200,CPU_launched=720)
        records['terminal'].update(imported_completed_cases=1200,resumed_new_cases=720)
        parent['status'].update(drain_requested=True);parent['terminal'].update(complete=False,drained=True,drain_requested=True,completed=1200,failed=0)
        return [c,pc,oc,original,middle,parent,records],dict(source_sha=kwargs['source_sha'],driver=kwargs['driver'],guard=kwargs['guard'],
            control_pointer=cp,lock_pointer=lp,audit_environment=kwargs['audit_environment'])

    def test_four_owner_headers_preserve_partial_history_and_changed_qualified_resource_profile(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                args,kw=self.owner_headers(root);before=deepcopy(args);owners=audit.require_bindings(*args,**kw)
                self.assertEqual({k:v['pid'] for k,v in owners.items()},{audit.ORIGINAL:7530,audit.INTERMEDIATE:9020,audit.PREVIOUS:9560,audit.RUNTIME:10110})
                self.assertFalse(args[4]['terminal']['complete']);self.assertEqual(args[4]['terminal']['failed'],6)
                self.assertFalse(args[5]['terminal']['complete']);self.assertEqual(args,before)
                self.assertNotEqual(owners[audit.RUNTIME]['frozen_environment'],kw['audit_environment'])

    def test_wrong_source_owner_namespace_profile_or_final_completion_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                base,kw=self.owner_headers(root)
                for mode in ('source','PID','namespace','oldprofile','profile_disclosure','parent_source','control','unfinished','ready','stop','protocol'):
                    args=deepcopy(base);new=args[-1]
                    if mode=='source':new['launch']['source_sha256']='f'*64
                    elif mode=='PID':new['launch'].update(pid=9560,start_ticks=95600)
                    elif mode=='namespace':new['launch']['frozen_environment']=deepcopy(args[-2]['launch']['frozen_environment'])
                    elif mode=='oldprofile':new['launch']['max_native_slots']=4
                    elif mode=='profile_disclosure':new['terminal']['previous_resource_profile']={}
                    elif mode=='parent_source':new['launch']['parent_stage_owner']=kw['control_pointer']
                    elif mode=='control':new['plan']['previous_resume_control']={}
                    elif mode=='unfinished':new['terminal']['complete']=False
                    elif mode=='ready':new['status']['ready_jobs']=['pending']
                    elif mode=='stop':new['terminal']['stop_requested']=True
                    else:new['terminal']['author_outcome_protocol']={}
                    with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_bindings(*args,**kw)

    def test_small_final_gate_accepts_all_nativeFalse_and_rejects_pending_or_failed_owner(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                args,_=self.owner_headers(root);records=args[-1]
                for value in records['status']['completed'].values():value['native_success']=False;value['motion_pass']=None
                audit.require_final_owner(records,1200)
                for kind in ('failed','pending','stop','drain','CPU_ready','missing_completed'):
                    bad=deepcopy(records)
                    if kind=='failed':bad['status']['failed']={'CPU_failure':{}}
                    elif kind=='pending':bad['terminal']['CPU_not_launched']=1
                    elif kind=='stop':bad['terminal']['stop_requested']=True
                    elif kind=='drain':bad['status']['drain_requested']=True
                    elif kind=='CPU_ready':bad['status']['ready_jobs']=['CPU_unexecuted']
                    else:bad['status']['completed'].pop(next(iter(bad['status']['completed'])))
                    with self.subTest(kind=kind),self.assertRaises(ValueError):audit.require_final_owner(bad,1200)

    def test_completion_gate_before_any_full_graph_or_TEST_read(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/audit.CONTROL;p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(validation_release=self.fake.point('CPU/release.json'),completed_job_ids=[])))
            lock=root/audit.LOCK;lock.write_text(json.dumps(dict(schema=core.LOCK_SCHEMA,approved_for_execution=True,
                control=dict(path=audit.CONTROL,sha256=audit.previous_audit.file_sha(p)),
                files={audit.CONTROL:audit.previous_audit.file_sha(p)})))
            leaf=root/audit.RUNTIME;leaf.mkdir(parents=True)
            for k in ('launch','plan','status','terminal'):(leaf/(k+'.json')).write_text('{}')
            denied=Mock(side_effect=AssertionError('early full graph/TEST access'))
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'require_complete_VAL_release',return_value={}),patch.object(audit,'verify_inputs',denied),patch.object(audit,'LookupBatch',denied),patch.object(audit,'read_formal_source',denied),self.assertRaisesRegex(ValueError,'complete1011'):
                audit.main(['--resume-lock-sha256',audit.previous_audit.file_sha(lock)])
            denied.assert_not_called();self.assertFalse((root/audit.OUTPUT).exists())

    def test_external_lock_failure_STOP_and_existing_output_never_touch_graph(self):
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
