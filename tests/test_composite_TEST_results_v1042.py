"""CPU/temp-only fifth-tier audit contracts; no real TEST/native/proc/graph."""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from our_mani.multitask import composite_TEST_resume_v1040 as core
from our_mani.multitask.evaluation import digest
from scripts import audit_composite_TEST_results_v1042_r10 as audit


def fixture_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
parent_fakes=fixture_module('CPU1042_parent1012_fakes','test_composite_TEST_results_v1012.py')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.parent=parent_fakes.AuditTests();self.parent.setUp();self.fake=self.parent.fake

    def grid(self,original=134,middle=584,parent_imported=1200,adopted=1400):
        pc,gc,oc,orec,mrec,grec,pp=self.parent.grid(original,middle,parent_imported)
        ids=[r['job_id'] for r in pp['jobs']]
        def record(i):return dict(complete=True,native_success=False,motion_pass=None if i%2 else False,
            returncode=0,timed_out=False,interrupted=False,automatic_retry=False)
        ps=deepcopy(grec['status']);ps.update(completed={k:deepcopy(grec['status']['completed'][k])
            if k in grec['status']['completed'] else record(i) for i,k in enumerate(ids[:adopted])},failed={})
        pt=deepcopy(grec['terminal']);pt.update(completed=adopted,failed=0)
        c=deepcopy(pc);c.update(schema=core.CONTROL_SCHEMA,previous_runtime=core.PREVIOUS,
            previous_controller_runtime=core.PREVIOUS_CONTROLLER,completed_job_ids=ids[:adopted],failed_job_ids=[],
            remaining_job_ids=ids[adopted:],completed_key_sha256=digest(sorted(ids[:adopted])),failed_key_sha256=digest([]),
            remaining_key_sha256=digest(sorted(ids[adopted:])))
        np=dict(pp,jobs=core.resumed_jobs(pp,c,c['execution_entry']),imported_completed_cases=adopted)
        return c,pc,gc,oc,orec,mrec,grec,dict(plan=pp,status=ps,terminal=pt),np

    def test_five_tier_exact_dynamic_partition_and_all_prior_rows_are_unchanged(self):
        for a,b,c,d in ((134,584,1200,1400),(7,19,31,49)):
            values=self.grid(a,b,c,d);before=deepcopy(values)
            self.assertEqual(audit.require_five_tier_plan(*values),{audit.ORIGINAL:a,audit.INTERMEDIATE:b-a,
                audit.GRANDPARENT:c-b,audit.PREVIOUS:d-c,audit.RUNTIME:1920-d})
            self.assertEqual(values,before)
            row=values[-1]['jobs'][c]
            self.assertTrue(row['imported_completed']);self.assertFalse(row['resume_lineage_v1040']['parent_imported_completed'])
            self.assertEqual(row['resume_lineage_v1010'],values[-2]['plan']['jobs'][c]['resume_lineage_v1010'])

    def test_whole1011_row_and_three_older_fields_cannot_be_flattened(self):
        for mode in ('lineage955','lineage1010','lineage1040','oldsha','parent_sha','runtime','entry','out','old_result','missing','duplicate'):
            values=self.grid();row=values[-1]['jobs'][1250]
            if mode=='lineage955':row['resume_lineage']['parent_row_sha256']='f'*64
            elif mode=='lineage1010':row['resume_lineage_v1010']['parent_row_sha256']='f'*64
            elif mode=='lineage1040':row.pop('resume_lineage_v1040')
            elif mode=='oldsha':row['previous_row_sha256']='f'*64
            elif mode=='parent_sha':row['resume_lineage_v1040']['parent_row_sha256']='f'*64
            elif mode=='runtime':row['evidence_runtime']=audit.RUNTIME
            elif mode=='entry':row['execution_entry']['kind']='original751'
            elif mode=='out':row['physical_argv'][row['physical_argv'].index('--out')+1]='invented/path'
            elif mode=='old_result':values[-2]['status']['completed'][values[0]['completed_job_ids'][0]]['native_success']=True
            elif mode=='missing':values[-1]['jobs'].pop()
            else:values[-1]['jobs'][-1]=deepcopy(values[-1]['jobs'][0])
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_five_tier_plan(*values)

    def launch(self,runtime,check=True):
        values=self.grid();row=next(r for r in values[-1]['jobs'] if r['evidence_runtime']==runtime)
        control=values[0];bindings=self.fake.bindings();identity=dict(sha256='6'*64)
        pid={audit.ORIGINAL:7530,audit.INTERMEDIATE:9020,audit.GRANDPARENT:9560,audit.PREVIOUS:10110,audit.RUNTIME:10410}[runtime]
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

    def test_all_five_real_CPU_native_tiers_keep_source_owner_PID_and_namespace(self):
        for runtime in (audit.ORIGINAL,audit.INTERMEDIATE,audit.GRANDPARENT,audit.PREVIOUS,audit.RUNTIME):
            for check in (True,False):
                actual,row,bindings,identity,owner,control=self.launch(runtime,check)
                audit.require_tier_launch(actual,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)
                changed=deepcopy(actual);changed['owner_pid']=123
                with self.subTest(runtime=runtime,check=check),self.assertRaises(ValueError):
                    audit.require_tier_launch(changed,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)

    def test_old753_never_acquires_invented901_namespace_or_capacity(self):
        args=self.launch(audit.ORIGINAL)
        for field in ('execution_entry','physical_argv','inherited_frozen_environment','current_mount_namespace'):
            actual=deepcopy(args[0]);actual[field]={}
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.require_tier_launch(actual,args[1],args[2],check=True,identity=args[3],stage_owner=args[4],control=args[5])

    def test_history_bytes_rechecked_only_in_their_four_declaring_lock_generations(self):
        for runtime,count in ((audit.ORIGINAL,4),(audit.INTERMEDIATE,3),(audit.GRANDPARENT,2),(audit.PREVIOUS,1)):
            batch=Mock();maps=[{'map':i} for i in range(4)];p=dict(path='CPU/only.json',sha256='a'*64)
            audit.require_historical_pointer(batch,*maps,p,runtime)
            self.assertEqual([c.args[1] for c in batch.locked_pointer.call_args_list],maps[:count])

    def test_scientific_helper_bodies_are_exact_frozen1012(self):
        def bodies(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        root=Path(__file__).parents[1];old=bodies(root/'scripts/audit_composite_TEST_results_v1012_r10.py');new=bodies(root/audit.SOURCE)
        for name in ('require_case_outcome','audit_complete_grid','summarize','require_tier_launch'):
            self.assertEqual(old[name],new[name],name)

    def test_qualification_inputs_follow_actual1031_or_real1005_fallback_scope(self):
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

    def owner_headers(self,root,*,fallback=False,lower=False):
        args,kw=self.parent.owner_headers(root)
        pc,gc,oc,original,middle,grandparent,parent=args
        source=root/audit.OWNER;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('# CPU fake1041 source\n')
        c=deepcopy(pc);cp=self.fake.point(audit.CONTROL);lp=self.fake.point(audit.LOCK)
        envelope=deepcopy(pc['resource_profile']) if fallback else dict(schema=audit.current_owner.PROFILE_SCHEMA,
            max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        actual=deepcopy(envelope)
        if lower:actual.update(max_native_slots=8,max_GPU_slots=4,max_CPU_preflight_slots=6,ready_limit=20)
        c.update(previous_resume_control=kw['control_pointer'],previous_resume_control_lock=kw['lock_pointer'],
            completed_job_ids=['CPU_done_'+str(i) for i in range(1400)],resource_profile=actual,
            resource_capacity_envelope=envelope,resource_profile_within_capacity=True,
            resource_capacity_source='qualified1005_fallback' if fallback else 'qualified1031',
            resource_qualifier=self.fake.point('CPU/new_capacity_qualifier.py'),
            resource_qualification=self.fake.point('CPU/new_capacity.json'),resource_profile_changed=actual!=pc['resource_profile'],
            previous_resource_profile=deepcopy(pc['resource_profile']),previous_resource_qualification=deepcopy(pc['resource_qualification']))
        records=deepcopy(parent);env=deepcopy(parent['launch']['frozen_environment']);env.update(pid=10410,observer_pid=10410,
            observer_start_ticks=104100,current_mount_namespace='mnt:[fake1041]',mount_namespace='mnt:[fake1041]')
        for v in records.values():
            v.update(resume_control=cp,resume_control_lock=lp,frozen_environment=deepcopy(env),resource_profile=actual,
                resource_qualification=c['resource_qualification'],previous_resume_control=c['previous_resume_control'],
                previous_resume_control_lock=c['previous_resume_control_lock'],resource_profile_changed=c['resource_profile_changed'],
                previous_resource_profile=c['previous_resource_profile'],previous_resource_qualification=c['previous_resource_qualification'],
                resource_capacity_envelope=envelope,resource_profile_within_capacity=True,resource_capacity_source=c['resource_capacity_source'])
            v['resource_qualifier']=c['resource_qualifier']
        records['launch'].update(pid=10410,start_ticks=104100,source_sha256=audit.previous_audit.file_sha(source),
            parent_stage_owner=audit.physical_owner.pointer(root/'scripts/run_composite_TEST_owner_v1011_r10.py'),imported_completed_cases=1400,
            **{k:actual[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')})
        expected={'--suite':c['suite']['path'],'--suite-lock':c['suite_lock']['path'],'--source-inputs':c['source_inputs']['path'],
            '--validation-release':c['validation_release']['path'],'--runtime':audit.RUNTIME,'--out':core.OUTPUT,
            '--resume-control':cp['path'],'--resume-lock':lp['path'],'--resume-lock-sha256':lp['sha256']}
        records['launch']['argv']=[audit.OWNER,*[x for pair in expected.items() for x in pair]]
        records['plan']['imported_completed_cases']=1400;records['status'].update(imported_completed=1400,CPU_launched=520)
        records['terminal'].update(imported_completed_cases=1400,resumed_new_cases=520)
        parent['status'].update(drain_requested=True);parent['terminal'].update(complete=False,drained=True,drain_requested=True,completed=1400,failed=0)
        return [c,pc,gc,oc,original,middle,grandparent,parent,records],dict(source_sha=kw['source_sha'],driver=kw['driver'],guard=kw['guard'],
            control_pointer=cp,lock_pointer=lp,audit_environment=kw['audit_environment'])

    def test_five_owner_headers_keep_three_partial_histories_and_capacity_distinct(self):
        for mode in ('default','lower','fallback'):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name)
                with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                    args,kw=self.owner_headers(root,fallback=mode=='fallback',lower=mode=='lower');before=deepcopy(args)
                    owners=audit.require_bindings(*args,**kw)
                    self.assertEqual(len(owners),5);self.assertEqual(args,before)
                    self.assertFalse(args[-2]['terminal']['complete']);self.assertEqual(args[5]['terminal']['failed'],6)
                    self.assertEqual(args[0]['resource_profile_changed'],mode!='fallback')
                    if mode=='lower':self.assertNotEqual(args[0]['resource_profile'],args[0]['resource_capacity_envelope'])

    def test_actual_quota_capacity_source_or_historical_profile_drift_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                base,kw=self.owner_headers(root)
                for mode in ('source','PID','namespace','actual','envelope_as_actual','profile_changed','parent_source','parent_control','unfinished','ready','stop','protocol'):
                    args=deepcopy(base);new=args[-1]
                    if mode=='source':new['launch']['source_sha256']='f'*64
                    elif mode=='PID':new['launch'].update(pid=10110,start_ticks=101100)
                    elif mode=='namespace':new['launch']['frozen_environment']=deepcopy(args[-2]['launch']['frozen_environment'])
                    elif mode=='actual':new['launch']['max_native_slots']=8
                    elif mode=='envelope_as_actual':new['terminal']['resource_capacity_envelope']={}
                    elif mode=='profile_changed':new['terminal']['resource_profile_changed']=False
                    elif mode=='parent_source':new['launch']['parent_stage_owner']=kw['control_pointer']
                    elif mode=='parent_control':new['plan']['previous_resume_control']={}
                    elif mode=='unfinished':new['terminal']['complete']=False
                    elif mode=='ready':new['status']['ready_jobs']=['pending']
                    elif mode=='stop':new['terminal']['stop_requested']=True
                    else:new['terminal']['author_outcome_protocol']={}
                    with self.subTest(mode=mode),self.assertRaises(ValueError):audit.require_bindings(*args,**kw)

    def test_scientific_full1920_false_None_zero_tick_grid_kept_without_filter(self):
        cases,manifest=self.parent.parent.grid_cases();before=deepcopy(cases)
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

    def test_small_final_gate_accepts_all_nativeFalse_and_rejects_pending_or_failed_owner(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                args,_=self.owner_headers(root);records=args[-1]
                for value in records['status']['completed'].values():value['native_success']=False;value['motion_pass']=None
                audit.require_final_owner(records,1400)
                for kind in ('failed','pending','stop','drain','CPU_ready','missing_completed'):
                    bad=deepcopy(records)
                    if kind=='failed':bad['status']['failed']={'CPU_failure':{}}
                    elif kind=='pending':bad['terminal']['CPU_not_launched']=1
                    elif kind=='stop':bad['terminal']['stop_requested']=True
                    elif kind=='drain':bad['status']['drain_requested']=True
                    elif kind=='CPU_ready':bad['status']['ready_jobs']=['CPU_unexecuted']
                    else:bad['status']['completed'].pop(next(iter(bad['status']['completed'])))
                    with self.subTest(kind=kind),self.assertRaises(ValueError):audit.require_final_owner(bad,1400)

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
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'require_complete_VAL_release',return_value={}),patch.object(audit,'verify_inputs',denied),patch.object(audit,'LookupBatch',denied),patch.object(audit,'read_formal_source',denied),self.assertRaisesRegex(ValueError,'complete1041'):
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
