"""CPU/temp-only seven-tier contracts; no actual TEST, Native or full graph."""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from our_mani.multitask import composite_TEST_reboot_resume_v1070 as core
from our_mani.multitask.evaluation import digest
from scripts import audit_second_reboot_TEST_results_v1072_r10 as audit


def fixture_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sixth_fakes = fixture_module('CPU1072_sixth1057_fakes', 'test_reboot_TEST_results_v1057.py')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.parent = sixth_fakes.AuditTests()
        self.parent.setUp()
        self.fake = self.parent.fake

    def grid(self, original=134, middle=584, fourth_imported=1266, fifth_imported=1409):
        sc, fc, pc, gc, oc, orec, mrec, grec, prec, frec, first, sp = self.parent.grid(
            original, middle, fourth_imported, fifth_imported)
        ids = [r['job_id'] for r in sp['jobs']]
        adopted = set(sc['completed_job_ids']) | set(ids[fifth_imported+2:fifth_imported+10])
        ss = deepcopy(frec['status'])
        ss['completed'] = {k: deepcopy(frec['status']['completed'][k]) if k in sc['completed_job_ids']
            else dict(complete=True, native_success=False, motion_pass=None, returncode=0,
                timed_out=False, interrupted=False, automatic_retry=False) for k in sorted(adopted)}
        ss.update(failed={}, active=[ids[fifth_imported+10]], checking=[ids[fifth_imported+11]])
        c = deepcopy(sc)
        c.update(schema=core.CONTROL_SCHEMA, previous_runtime=core.PREVIOUS,
            previous_controller_runtime=core.PREVIOUS_CONTROLLER)
        remaining = set(ids)-adopted
        for name, values in dict(completed=adopted, failed=set(), remaining=remaining,
                saved_completed=adopted, rejected_saved_completed=set()).items():
            c[name+'_job_ids'] = sorted(values)
            c[name+'_key_sha256'] = digest(sorted(values))
        inherited=set(sc['completed_job_ids'])
        fresh=adopted-inherited
        c.update(inherited_completed_job_ids=sorted(inherited),inherited_completed_key_sha256=digest(sorted(inherited)),
            fresh_completed_job_ids=sorted(fresh),fresh_completed_key_sha256=digest(sorted(fresh)),
            old_semantic_case_evidence_inherited=True,full_original_input_maps_reverified=True)
        second = deepcopy(first)
        second.update(schema='r10_readonly_host_second_reboot_TEST_evidence_observation_v1067',
            current_boot_id='00000000-0000-0000-0000-000000000002',
            saved_status_completed_count=len(adopted), recoverable_complete_count=len(adopted),
            damaged_registered_count=0, still_unclosed_count=len(remaining))
        second['completed_evidence'] = [dict(job_id=k, recoverable_complete=True,
            native_success=ss['completed'][k]['native_success'], errors=[]) for k in sorted(adopted)]
        for name, values in dict(accepted_completed_job_ids=adopted, rejected_saved_completed_job_ids=set(),
                saved_completed_job_ids=adopted, remaining_job_ids=remaining).items():
            second[name] = sorted(values)
            second[name+'_sha256'] = digest(sorted(values))
        np = dict(sp, jobs=core.resumed_jobs(sp,c,c['execution_entry']), imported_completed_cases=len(adopted))
        return (c,sc,fc,pc,gc,oc,orec,mrec,grec,prec,frec,dict(plan=sp,status=ss),first,second,np)

    def test_seven_tiers_keep_literal_history_and_both_damage_histories(self):
        for parts in ((134,584,1266,1409),(7,19,31,49)):
            values = self.grid(*parts)
            before = deepcopy(values)
            a,b,c,d = parts
            expected = {audit.ORIGINAL:a,audit.INTERMEDIATE:b-a,audit.GRANDPARENT:c-b,
                audit.PREVIOUS:d-c,audit.FIFTH:2,audit.SIXTH:8,audit.RUNTIME:1920-d-10}
            self.assertEqual(audit.require_seven_tier_plan(*values),expected)
            self.assertEqual(sum(expected.values()),1920)
            self.assertEqual(values,before)
            self.assertNotIn('terminal',values[10])
            self.assertNotIn('terminal',values[11])
            self.assertEqual(len(values[1]['rejected_saved_completed_job_ids']),2)
            self.assertEqual(values[0]['rejected_saved_completed_job_ids'],[])
            self.assertEqual(values[12]['damaged_registered_count'],2)
            self.assertEqual(values[13]['damaged_registered_count'],0)

    def test_reboot_partitions_retain_nativeFalse_and_unknown_motion_without_outcome_selection(self):
        base = self.grid()
        for mode in ('drop_false','saved_count','changed_old','changed_new','first_damage_as_false',
                'first_damage_erased','first_boot_filter','second_boot_filter','second_unregistered','reboot_flag',
                'inherited_partition','fresh_partition','inherited_claim','map_claim'):
            args = deepcopy(base)
            if mode=='drop_false': args[0]['completed_job_ids'].pop()
            elif mode=='saved_count': args[13]['saved_status_completed_count']-=1
            elif mode=='changed_old':
                changed=args[11]['status']['completed'][args[1]['completed_job_ids'][0]]
                changed['native_success']=not changed['native_success']
            elif mode=='changed_new': args[13]['completed_evidence'][-1]['native_success']=True
            elif mode=='first_damage_as_false': next(v for v in args[12]['completed_evidence'] if not v['recoverable_complete'])['native_success']=False
            elif mode=='first_damage_erased': args[1]['rejected_saved_completed_job_ids']=[]
            elif mode=='first_boot_filter': args[12]['success_based_case_selection']=True
            elif mode=='second_boot_filter': args[13]['success_based_case_selection']=True
            elif mode=='second_unregistered': args[13]['unregistered_clean_native_terminals']=[dict(job_id='extra')]
            elif mode=='reboot_flag': args[0]['explicit_host_reboot_recovery']=False
            elif mode=='inherited_partition': args[0]['inherited_completed_job_ids'].pop()
            elif mode=='fresh_partition': args[0]['fresh_completed_key_sha256']='f'*64
            elif mode=='inherited_claim': args[0]['old_semantic_case_evidence_inherited']=False
            else: args[0]['full_original_input_maps_reverified']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError): audit.require_seven_tier_plan(*args)

    def test_every_parent_lineage_and_all1920_coordinates_remain_bound(self):
        base = self.grid()
        fields = ('resume_lineage','resume_lineage_v1010','resume_lineage_v1040','resume_lineage_v1055','resume_lineage_v1070')
        for mode in (*fields,'previous_sha','runtime','entry','out','missing','duplicate'):
            args = deepcopy(base)
            row = args[-1]['jobs'][1411]
            if mode in fields: row[mode]['parent_row_sha256']='f'*64
            elif mode=='previous_sha': row['previous_row_sha256']='f'*64
            elif mode=='runtime': row['evidence_runtime']=audit.RUNTIME
            elif mode=='entry': row['execution_entry']['kind']='original751'
            elif mode=='out': row['physical_argv'][row['physical_argv'].index('--out')+1]='invented/path'
            elif mode=='missing': args[-1]['jobs'].pop()
            else: args[-1]['jobs'][-1]=deepcopy(args[-1]['jobs'][0])
            with self.subTest(mode=mode),self.assertRaises(ValueError): audit.require_seven_tier_plan(*args)

    def test_original_scientific_and_leaf_receipt_functions_have_exact_frozen1057_bodies(self):
        def bodies(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body
                if isinstance(n,ast.FunctionDef)}
        root = Path(__file__).parents[1]
        old,new = bodies(root/audit.sixth_audit.SOURCE),bodies(root/audit.SOURCE)
        for name in ('require_case_outcome','audit_complete_grid','summarize','require_tier_launch'):
            self.assertEqual(old[name],new[name],name)

    def launch(self,runtime,check):
        values = self.grid()
        row = next(v for v in values[-1]['jobs'] if v['evidence_runtime']==runtime)
        control = values[0]
        bindings = self.fake.bindings()
        identity = dict(sha256='6'*64)
        pid = {audit.ORIGINAL:7530,audit.INTERMEDIATE:9020,audit.GRANDPARENT:9560,
            audit.PREVIOUS:10110,audit.FIFTH:10410,audit.SIXTH:10560,audit.RUNTIME:10710}[runtime]
        owner = dict(pid=pid,start_ticks=pid*10)
        actual = dict(argv=deepcopy(row['argv'])+(['--check'] if check else []),check_only=check,
            launcher_pid=pid+1,launcher_start_ticks=pid+2,owner_pid=pid,execution_protocol_changed=True,
            native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
            expected_identity_sha256=None if check else identity['sha256'],**deepcopy(bindings))
        if runtime!=audit.ORIGINAL:
            environment,receipt = self.fake.environment(pid)
            receipt.update(observer_start_ticks=owner['start_ticks'],current_mount_namespace='mnt:[fake'+str(pid)+']',
                mount_namespace='mnt:[fake'+str(pid)+']')
            control.update(environment)
            owner['frozen_environment']=receipt
            actual.update(execution_entry=row['execution_entry'],physical_argv=row['physical_argv']+(['--check'] if check else []),
                inherited_frozen_environment=receipt,current_mount_namespace=receipt['current_mount_namespace'])
        return actual,row,bindings,identity,owner,control

    def test_seven_CPU_native_tiers_keep_their_actual_source_owner_and_namespace(self):
        for runtime in (audit.ORIGINAL,audit.INTERMEDIATE,audit.GRANDPARENT,audit.PREVIOUS,audit.FIFTH,audit.SIXTH,audit.RUNTIME):
            for check in (True,False):
                actual,row,bindings,identity,owner,control = self.launch(runtime,check)
                audit.require_tier_launch(actual,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)
                bad = deepcopy(actual)
                bad['owner_pid']=123
                with self.subTest(runtime=runtime,check=check),self.assertRaises(ValueError):
                    audit.require_tier_launch(bad,row,bindings,check=check,identity=identity,stage_owner=owner,control=control)

    def test_each_historical_pointer_is_rechecked_in_all_its_literal_generations(self):
        for runtime,count in ((audit.ORIGINAL,6),(audit.INTERMEDIATE,5),(audit.GRANDPARENT,4),
                (audit.PREVIOUS,3),(audit.FIFTH,2),(audit.SIXTH,1)):
            batch = Mock()
            maps = [{'map':i} for i in range(6)]
            pointer = dict(path='CPU/only.json',sha256='a'*64)
            audit.require_historical_pointer(batch,*maps,pointer,runtime)
            self.assertEqual([v.args[1] for v in batch.locked_pointer.call_args_list],maps[:count])

    def owner_headers(self,root,*,fallback=False,lower=False):
        args,kw = self.parent.owner_headers(root,fallback=fallback,lower=lower)
        sc,fc,pc,gc,oc,original,middle,grandparent,parent,fifth,sixth = args
        source=root/audit.OWNER
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('# CPU fake1071 source\n')
        c=deepcopy(sc)
        cp,lp=self.fake.point(audit.CONTROL),self.fake.point(audit.LOCK)
        c.update(previous_resume_control=kw['control_pointer'],previous_resume_control_lock=kw['lock_pointer'],
            completed_job_ids=['CPU_done_'+str(i) for i in range(1419)],resource_profile_changed=False,
            previous_resource_profile=deepcopy(sc['resource_profile']),previous_resource_qualification=deepcopy(sc['resource_qualification']),
            reboot_audit=self.fake.point('runtime/framework/second_reboot_TEST_evidence_v1067.json'))
        records=deepcopy(sixth)
        env=deepcopy(sixth['launch']['frozen_environment'])
        env.update(pid=10710,observer_pid=10710,observer_start_ticks=107100,
            current_mount_namespace='mnt:[fake1071]',mount_namespace='mnt:[fake1071]')
        for v in records.values():
            v.update(resume_control=cp,resume_control_lock=lp,frozen_environment=deepcopy(env),
                previous_resume_control=c['previous_resume_control'],previous_resume_control_lock=c['previous_resume_control_lock'],
                resource_profile_changed=False,previous_resource_profile=c['previous_resource_profile'],
                previous_resource_qualification=c['previous_resource_qualification'],reboot_audit=c['reboot_audit'],
                infrastructure_recovery=True,boot_id='00000000-0000-0000-0000-000000000002')
        records['launch'].update(pid=10710,start_ticks=107100,source_sha256=audit.previous_audit.file_sha(source),
            parent_stage_owner=audit.physical_owner.pointer(root/audit.sixth_audit.OWNER),imported_completed_cases=1419)
        expected={'--suite':c['suite']['path'],'--suite-lock':c['suite_lock']['path'],'--source-inputs':c['source_inputs']['path'],
            '--validation-release':c['validation_release']['path'],'--runtime':audit.RUNTIME,'--out':core.OUTPUT,
            '--resume-control':cp['path'],'--resume-lock':lp['path'],'--resume-lock-sha256':lp['sha256']}
        records['launch']['argv']=[audit.OWNER,*[x for pair in expected.items() for x in pair]]
        records['plan']['imported_completed_cases']=1419
        records['status'].update(imported_completed=1419,CPU_launched=501)
        records['terminal'].update(imported_completed_cases=1419,resumed_new_cases=501)
        sixth.pop('terminal')
        sixth['status'].update(drain_requested=False,active=['interrupted_native'],checking=['interrupted_CPU'],native_slots=1,GPU_slots=1)
        return [c,sc,fc,pc,gc,oc,original,middle,grandparent,parent,fifth,sixth,records],dict(
            source_sha=kw['source_sha'],driver=kw['driver'],guard=kw['guard'],control_pointer=cp,lock_pointer=lp,
            audit_environment=kw['audit_environment'])

    def test_seven_owner_headers_keep_both_abrupt_nonterminals_and_actual_capacity(self):
        for mode in ('default','lower','fallback'):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name)
                with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                    args,kw=self.owner_headers(root,fallback=mode=='fallback',lower=mode=='lower')
                    before=deepcopy(args)
                    owners=audit.require_bindings(*args,**kw)
                    self.assertEqual(len(owners),7)
                    self.assertEqual(args,before)
                    self.assertNotIn('terminal',args[-3])
                    self.assertNotIn('terminal',args[-2])
                    if mode=='lower': self.assertNotEqual(args[0]['resource_profile'],args[0]['resource_capacity_envelope'])

    def test_faked_interrupted_terminal_changed_boot_source_capacity_or_incomplete_current_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'ROOT',root):
                base,kw=self.owner_headers(root)
                for mode in ('source','PID','namespace','actual','capacity','changed','parent_source','parent_control',
                        'unfinished','ready','stop','protocol','faked1041','faked1056','boot','same_boot','reboot_proof','old_reboot_proof','infra'):
                    args=deepcopy(base)
                    new=args[-1]
                    if mode=='source': new['launch']['source_sha256']='f'*64
                    elif mode=='PID': new['launch'].update(pid=10560,start_ticks=105600)
                    elif mode=='namespace': new['launch']['frozen_environment']=deepcopy(args[-2]['launch']['frozen_environment'])
                    elif mode=='actual': new['launch']['max_native_slots']=8
                    elif mode=='capacity': new['terminal']['resource_capacity_envelope']={}
                    elif mode=='changed': new['terminal']['resource_profile_changed']=True
                    elif mode=='parent_source': new['launch']['parent_stage_owner']=kw['control_pointer']
                    elif mode=='parent_control': new['plan']['previous_resume_control']={}
                    elif mode=='unfinished': new['terminal']['complete']=False
                    elif mode=='ready': new['status']['ready_jobs']=['pending']
                    elif mode=='stop': new['terminal']['stop_requested']=True
                    elif mode=='protocol': new['terminal']['author_outcome_protocol']={}
                    elif mode=='faked1041': args[-3]['terminal']={}
                    elif mode=='faked1056': args[-2]['terminal']={}
                    elif mode=='boot': new['terminal']['boot_id']='other_boot'
                    elif mode=='same_boot':
                        for v in new.values(): v['boot_id']=args[-2]['launch']['boot_id']
                    elif mode=='reboot_proof': new['terminal']['reboot_audit']={}
                    elif mode=='old_reboot_proof': args[-2]['status']['reboot_audit']={}
                    else: new['terminal']['infrastructure_recovery']=False
                    with self.subTest(mode=mode),self.assertRaises(ValueError): audit.require_bindings(*args,**kw)

    def test_nativeFalse_zero_ticks_unknown_motion_and_original749_530_gates_are_retained(self):
        for motion in (True,False,None):
            case,identity,protocol=self.fake.case(ticks=0,success=False,motion=motion)
            self.assertEqual(audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE),
                dict(native_success=False,motion_pass=motion,ticks=0))
        for mode in ('nativeNone','cleanup','RML','clock'):
            case,identity,protocol=self.fake.case(ticks=0,success=False,motion=None)
            if mode=='nativeNone': case['official_task_success']=None
            elif mode=='cleanup': case['cleanup_completed']=False
            elif mode=='RML': case['common_author_RML_parameters']['actual_parameters']['planning_acceleration']=6
            else: case['counted_clock']['ticks']=1
            with self.subTest(mode=mode),self.assertRaises(Exception):
                audit.require_case_outcome(case,identity,protocol,self.fake.POLICY,self.fake.PROFILE)

    def test_scientific_full1920_grid_does_not_filter_nativeFalse_None_or_zero_ticks(self):
        cases,manifest=self.parent.parent.parent.parent.grid_cases()
        before=deepcopy(cases)
        with patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE),patch.object(audit,'require_attestation') as attest:
            result=audit.audit_complete_grid(cases,manifest,dict(common_author_RML_profile={}))
        self.assertEqual(result['method_cases'],1920)
        self.assertEqual(result['native_successes_by_method'],{m:0 for m in audit.METHODS})
        self.assertEqual(result['measured_motion_unknown_by_method'],{m:320 for m in audit.METHODS})
        self.assertIsNone(result['ours_measured_motion_passed'])
        self.assertEqual(attest.call_count,1920)
        self.assertEqual(cases,before)

    def test_incomplete_duplicate_unapproved_or_fitted_grid_never_releases(self):
        original,manifest=self.parent.parent.parent.parent.grid_cases()
        for mode in ('missing','duplicate','native_unknown','policy','fitted','infra','method','source'):
            cases=deepcopy(original)
            if mode=='missing': cases.pop()
            elif mode=='duplicate': cases[-1]=deepcopy(cases[0])
            elif mode=='native_unknown': cases[0]['official_task_success']=None
            elif mode=='policy': cases[0]['full16_frozen_execution']['policy_sha256']='f'*64
            elif mode=='fitted': cases[0]['TEST_used_for_fitting']=True
            elif mode=='infra': cases[0]['infrastructure_error']='lost_evidence'
            elif mode=='method': cases[0]['method']='unapproved'
            else: cases[0]['source_raw_episode_id']='changed_original_source'
            with self.subTest(mode=mode),patch.object(audit,'policy_fingerprint',return_value=self.fake.POLICY),patch.object(audit,'fingerprint',return_value=self.fake.PROFILE),patch.object(audit,'require_attestation'),self.assertRaises(ValueError):
                audit.audit_complete_grid(cases,manifest,dict(common_author_RML_profile={}))

    def test_selected_capacity_metadata_scope_does_not_open_other_graphs(self):
        from scripts import prepare_composite_TEST_resume_v1040_r10 as preparation
        names={'qualified1031':'runtime/framework/concurrent_TRAIN_resource_canary_v1031/plan.json',
            'qualified1005_fallback':'runtime/framework/concurrent_TRAIN_resource_canary_v1005/plan.json'}
        for kind,name in names.items():
            ledger=Mock()
            batch=object()
            files={'CPU_fake':True}
            audit.require_qualification_inputs(dict(inputs={name:'a'*64}),dict(resource_capacity_source=kind),preparation,batch,files,ledger)
            ledger.frozen.assert_called_once_with(batch,files,dict(path=name,sha256='a'*64),parse=False)
            for invalid in (names[next(k for k in names if k!=kind)],'models/unopened.pt','data/unopened.hdf5','../outside.json'):
                ledger=Mock()
                with self.subTest(kind=kind,invalid=invalid),self.assertRaises(ValueError):
                    audit.require_qualification_inputs(dict(inputs={invalid:'a'*64}),dict(resource_capacity_source=kind),preparation,batch,files,ledger)
                ledger.frozen.assert_not_called()

    def test_real_full_completion_gate_precedes_graph_and_TEST_reads(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            p=root/audit.CONTROL
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(validation_release=self.fake.point('CPU/release.json'),completed_job_ids=[])))
            lock=root/audit.LOCK
            lock.write_text(json.dumps(dict(schema=core.LOCK_SCHEMA,approved_for_execution=True,
                control=dict(path=audit.CONTROL,sha256=audit.previous_audit.file_sha(p)),
                files={audit.CONTROL:audit.previous_audit.file_sha(p)})))
            leaf=root/audit.RUNTIME
            leaf.mkdir(parents=True)
            for k in ('launch','plan','status','terminal'): (leaf/(k+'.json')).write_text('{}')
            denied=Mock(side_effect=AssertionError('early full graph/TEST access'))
            with patch.object(audit,'ROOT',root),patch.object(audit.physical_owner,'require_complete_VAL_release',return_value={}),patch.object(audit,'verify_inputs',denied),patch.object(audit,'LookupBatch',denied),patch.object(audit,'read_formal_source',denied),self.assertRaisesRegex(ValueError,'complete1071'):
                audit.main(['--resume-lock-sha256',audit.previous_audit.file_sha(lock)])
            denied.assert_not_called()
            self.assertFalse((root/audit.OUTPUT).exists())

    def test_STOP_external_SHA_and_existing_result_block_without_graph_reads(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            p=root/audit.LOCK
            p.parent.mkdir(parents=True)
            p.write_text('{}')
            denied=Mock(side_effect=AssertionError('actual graph touched'))
            with patch.object(audit,'ROOT',root),patch.object(audit,'verify_inputs',denied),self.assertRaisesRegex(ValueError,'external exact'):
                audit.main(['--resume-lock-sha256','f'*64])
            stop=root/'runtime/framework/multitask_r10_STOP'
            stop.parent.mkdir(parents=True)
            stop.touch()
            with patch.object(audit,'ROOT',root),patch.object(audit,'EvidenceLedger') as ledger,self.assertRaisesRegex(ValueError,'STOP'):
                audit.main(['--resume-lock-sha256','f'*64])
            ledger.assert_not_called()
            stop.unlink()
            out=root/audit.OUTPUT
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text('original')
            with patch.object(audit,'ROOT',root),self.assertRaisesRegex(ValueError,'never overwrite'):
                audit.main(['--resume-lock-sha256','f'*64])
            self.assertEqual(out.read_text(),'original')
            denied.assert_not_called()

    def test_historical_recovery_path_contains_no_old_live_boot_or_history_gate(self):
        source=(Path(__file__).parents[1]/audit.SOURCE).read_text()
        tree=ast.parse(source)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn('sixth_owner.require_history',calls)
        self.assertNotIn('sixth_audit.require_control',calls)
        self.assertIn('sixth_audit.require_six_tier_plan',calls)
        self.assertIn('core.require_resume_plan',calls)
        self.assertIn('require_historical_sixth_control',calls)
        self.assertIn('recovery_preparation.require_host_reboot_observation',calls)


if __name__=='__main__': unittest.main()
