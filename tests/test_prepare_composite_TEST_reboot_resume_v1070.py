"""Pure/temp-only reboot preparation checks; production prepare never invoked."""
from copy import deepcopy
from contextlib import ExitStack, contextmanager, redirect_stdout
import importlib.util
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts import prepare_composite_TEST_reboot_resume_v1070_r10 as prepare
spec=importlib.util.spec_from_file_location('fixture1070_CPUprepare',Path(__file__).with_name('test_composite_TEST_reboot_resume_v1070.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class RebootPreparationTests(unittest.TestCase):
    def test_disabled_default_and_external_pins_before_IO(self):
        with patch.object(prepare,'prepare',side_effect=AssertionError('inactive')),redirect_stdout(io.StringIO()) as out:
            prepare.main([])
        self.assertIn('"execution_enabled": false',out.getvalue())
        with patch.object(prepare,'stopped',side_effect=AssertionError('before IO')):
            for missing in ('environment_recheck_sha256','resource_qualification_sha256','reboot_audit_sha256'):
                args=dict(environment_recheck_sha256='a'*64,resource_qualification_sha256='b'*64,reboot_audit_sha256='c'*64);args[missing]=None
                with self.subTest(missing=missing),self.assertRaises(ValueError):prepare.prepare(**args)

    def test_literal_status1419_and_adopted1419_partition(self):
        _,p,s,a,*_=f.fixture();before=deepcopy((p,s,a));part=prepare.partition(p,s,a)
        self.assertEqual((len(part['saved']),len(part['completed']),len(part['rejected']),len(part['remaining'])),(1419,1419,0,501))
        self.assertEqual((p,s,a),before)

    def test_actual_absence_required_and_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=dict(absent_evidence={r+'/'+leaf:dict(path=r+'/'+leaf,exists=False,is_file=False)
                for r in (prepare.PREVIOUS,prepare.PREVIOUS_CONTROLLER) for leaf in ('terminal.json','failure.json','DRAIN','STOP')})
            prepare.require_absent_evidence(a,root)
            q=root/prepare.PREVIOUS/'terminal.json';q.parent.mkdir(parents=True);q.write_bytes(b'')
            with self.assertRaises(ValueError):prepare.require_absent_evidence(a,root)
            self.assertEqual(q.read_bytes(),b'')

    def test_interrupted_inventory_preserves_zero_bytes_and_missing_terminal(self):
        _,p,*_=f.fixture();row=p['jobs'][1411]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/prepare.core.option(row['physical_argv'],'--out');out.mkdir(parents=True)
            (out/'case.json').write_bytes(b'');leaf=root/prepare.PREVIOUS/'native'/row['job_id'];leaf.mkdir(parents=True)
            (leaf/'stdout.log').write_bytes(b'partial')
            value=prepare.interrupted_attempt_inventory(root,row,prepare.old_audit.EvidenceLedger(root))
            self.assertEqual(len(value['partial_files']),1);self.assertEqual(len(value['log_files']),1)
            self.assertTrue(value['outcome_unknown']);self.assertFalse((leaf/'terminal.json').exists())
            (out/'alias').symlink_to(root/'never_read_raw')
            with self.assertRaises(ValueError):prepare.interrupted_attempt_inventory(root,row,prepare.old_audit.EvidenceLedger(root))

    def test_old_tiers_751_901_paths_and_history_merge_conflict(self):
        _,p,*_=f.fixture()
        for row in (p['jobs'][0],p['jobs'][134],p['jobs'][584],p['jobs'][1266],p['jobs'][1409]):
            paths=prepare.old_evidence_paths(row)
            self.assertEqual(len(paths),11 if row['evidence_runtime']==prepare.core.ORIGINAL else 12)
            self.assertTrue(paths[0].startswith(row['evidence_runtime']+'/CPU/'))
        with self.assertRaises(ValueError):prepare.merge_maps({'historical':'a'*64},{'historical':'b'*64})

    def test_historical_closure_uses_old_pure_plans_without_old_current_boot_gate(self):
        _,p1056,_,_,prior,p1041,s1041,a1054,c1040,p1011,s1011,t1011,*_=f.fixture()
        prior=deepcopy(prior)
        prior['previous_resume_control']=dict(path=prepare.fifth_core.CONTROL,sha256='a'*64)
        prior['previous_resume_control_lock']=dict(path=prepare.fifth_core.LOCK,sha256='b'*64)
        for key,value in (('launch',{}),('plan',p1041),('status',s1041)):
            prior['previous_'+key]=dict(path=prepare.core.FIFTH+'/'+key+'.json',sha256='c'*64)
        prior['reboot_audit']=dict(path='CPU_only/first_boot_audit',sha256='d'*64)
        older=dict(plan=p1011,status=s1011,terminal=t1011)
        payloads={prepare.fifth_core.CONTROL:c1040,prepare.fifth_core.LOCK:dict(schema=prepare.fifth_core.LOCK_SCHEMA,
            approved_for_execution=True,files={'original-map-key':'e'*64}),'CPU_only/first_boot_audit':a1054}
        payloads.update({prepare.core.FIFTH+'/'+k+'.json':v for k,v in (('launch',{}),('plan',p1041),('status',s1041))})
        class Ledger:
            def json(self,path,*,expected):return deepcopy(payloads[path])
        with patch.object(prepare,'verify_inputs') as verify,patch.object(prepare.owner,'require_history',return_value=({}, {'CPU':'753'},older)) as old_history:
            original,returned_older,fifth=prepare.require_historical_chain(prior,{'original-map-key':'e'*64},Ledger(),p1056)
            self.assertEqual(original,{'CPU':'753'});self.assertEqual(returned_older,older);self.assertEqual(fifth['plan'],p1041)
            old_history.assert_called_once_with(c1040,{'original-map-key':'e'*64})
            verify.assert_called_once_with(prepare.ROOT,{'original-map-key':'e'*64})
        self.assertEqual(len(prior['rejected_saved_completed_job_ids']),2)
        with patch.object(prepare,'verify_inputs'),patch.object(prepare.owner,'require_history',side_effect=AssertionError('map must reject first')):
            with self.assertRaises(ValueError):prepare.require_historical_chain(prior,{'original-map-key':'f'*64},Ledger(),p1056)

    def test_lower_frozen_quota_uses_historical_capacity_without_train_rerun(self):
        profile=dict(prepare.PROFILE,max_native_slots=10,max_GPU_slots=4,max_CPU_preflight_slots=5,ready_limit=20)
        selected=prepare.capacity_selection('qualified1031',profile)
        self.assertEqual(selected['resource_profile'],profile)
        self.assertEqual(selected['resource_capacity_envelope'],prepare.PROFILE)
        with self.assertRaises(ValueError):prepare.capacity_selection('qualified1031',dict(profile,max_GPU_slots=6))

    def test_fsync_file_and_parent_is_real_temp_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);file=root/'control.json';file.write_text('{}')
            prepare.fsync_path(file);prepare.fsync_path(root)

    def test_complete_temp_prepare_publishes_durable_lock_with_literal_saved_state(self):
        c,plan,status,audit,prior,p1041,s1041,a1054,*rest=f.fixture()
        policy=plan['policy_sha256'];protocol={'CPU':'original741'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def point(name):return dict(path=name,sha256=prepare.file_sha(root/name))
            def write(name,value):
                q=root/name;q.parent.mkdir(parents=True,exist_ok=True)
                q.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode());return point(name)
            names={*prepare.REQUIRED_SOURCES,*prepare.bootstrap.SOURCES,prepare.core.PREVIOUS_OWNER,prepare.core.PREVIOUS_CONTROLLER_SOURCE}
            for name in names:write(name,('CPU source '+name).encode())
            manifest=dict(path='CPU_only/test_manifest.json',sha256='9'*64)
            suite=dict(author_outcome_protocol=protocol,source_manifests=dict(test=manifest),
                formal_case_driver=point(prepare.core.ORIGINAL_ENTRY),formal_admission_guard={'CPU':'guard'})
            write(prepare.SUITE,suite);write(prepare.VAL_RELEASE,dict(complete=True,policy_sha256=policy,author_outcome_protocol=protocol))
            stub=write('CPU_only/frozen_stub',b'original immutable bytes');files={str(root/stub['path']):stub['sha256']}
            write(prepare.SUITE_LOCK,dict(files=files));write(prepare.SOURCE_INPUTS,dict(schema='r10_full16_post_admission_split_source_lock_v1',split='test',source_count=640,
                policy_sha256=policy,TEST_used_for_fitting=False,source_manifest_sha256=manifest['sha256'],files=files))
            env={key:write(name,{'CPU':key}) for key,name in prepare.base.ENVIRONMENT_PATHS.items()}
            for name in prepare.base.ENVIRONMENT_EXTRA:write(name,{'CPU':'env extra'})
            standing=write(prepare.base.STANDING,dict(continue_until_user_stop=True,user_stop_received=False))
            auth=write(prepare.AUTHORIZATION,dict(user_authorized_concurrency_change=True))
            shared={name:point(name) for name in prepare.bootstrap.SOURCES}
            entry=deepcopy(prior['execution_entry']);bc=entry['batch_control']['path'];bl=entry['batch_lock']['path']
            write(bc,prepare.base.formal_bootstrap_control(suite));write(bl,dict(schema=prepare.bootstrap.LOCK_SCHEMA,execution_enabled=True,files={str(root/n):p['sha256'] for n,p in shared.items()}))
            entry.update(pointer=shared[prepare.core.BATCH_ENTRY],batch_control=point(bc),batch_lock=point(bl))
            for row in plan['jobs']:
                e=dict(kind='original751',pointer=shared[prepare.core.ORIGINAL_ENTRY]) if row['evidence_runtime']==prepare.core.ORIGINAL else deepcopy(entry)
                row.update(execution_entry=e,worker_sha256=e['pointer']['sha256'],argv=prepare.core.outer_argv(row['physical_argv'],e))
            original={'plan':{'CPU':'original'},'launch':{'pid':753},'status':{},'terminal':{}}
            for key in ('launch','plan','status','terminal'):prior['original_'+key]=write(prepare.core.ORIGINAL+'/'+key+'.json',original[key])
            prior.update(suite=point(prepare.SUITE),suite_lock=point(prepare.SUITE_LOCK),source_inputs=point(prepare.SOURCE_INPUTS),
                validation_release=point(prepare.VAL_RELEASE),policy_sha256=policy,author_outcome_protocol=protocol,
                execution_entry=entry,execution_implementation=prepare.owner.execution_implementation(entry,shared),standing_continuation_record=standing,
                concurrency_authorization_record=auth,resource_profile=prepare.PROFILE,resource_qualification=dict(path=prepare.QUALIFICATION,sha256='b'*64),**env)
            write(prepare.PREVIOUS_CONTROL,prior);write(prepare.PREVIOUS_LOCK,dict(schema=prepare.parent_core.LOCK_SCHEMA,approved_for_execution=True,files=files))
            common=dict(resume_control=point(prepare.PREVIOUS_CONTROL),resume_control_lock=point(prepare.PREVIOUS_LOCK))
            previous_launch=dict(common,pid=1041,start_ticks=37000,created_utc='2026-10-22T10:00:00+00:00',source_sha256=point(prepare.core.PREVIOUS_OWNER)['sha256'])
            for runtime in {r['evidence_runtime'] for r in plan['jobs']}:
                if runtime!=prepare.PREVIOUS:write(runtime+'/launch.json',dict(pid=111))
            write(prepare.PREVIOUS+'/launch.json',previous_launch);write(prepare.PREVIOUS+'/plan.json',plan);write(prepare.PREVIOUS+'/status.json',status)
            write(prepare.PREVIOUS_CONTROLLER+'/launch.json',dict(common,pid=1043,start_ticks=36000,source_sha256=point(prepare.core.PREVIOUS_CONTROLLER_SOURCE)['sha256']))
            write(prepare.PREVIOUS_CONTROLLER+'/status.json',dict(stage='formal_TEST_1920',automatic_retry=False))
            audit.update(source=point(prepare.REBOOT_AUDITOR),inputs={'previous_owner_'+name:dict(path=prepare.PREVIOUS+'/'+name+'.json',SHA_matches=True,
                sha256=point(prepare.PREVIOUS+'/'+name+'.json')['sha256']) for name in ('launch','plan','status')},
                absent_evidence={r+'/'+leaf:dict(path=r+'/'+leaf,exists=False,is_file=False) for r in (prepare.PREVIOUS,prepare.PREVIOUS_CONTROLLER)
                    for leaf in ('terminal.json','failure.json','DRAIN','STOP')})
            write(prepare.REBOOT_AUDIT,audit);boot=write(prepare.HOST_REBOOT_OBSERVATION,{'CPU':'boot point'})
            qlaunch=write(prepare.QUALIFICATION_LAUNCH,dict(CPU='independent historical1031 audit'))
            qc=write('CPU_only/train_control.json',{});ql=write('CPU_only/train_lock.json',{})
            qualification=write(prepare.QUALIFICATION,dict(actual_independent_audit=qlaunch,inputs={},control=qc,lock=ql))
            write(prepare.ENVIRONMENT_RECHECK,dict(source=point(prepare.RECHECK_SOURCE)))
            for key in c['rejected_saved_completed_job_ids']:
                row=next(r for r in plan['jobs'] if r['job_id']==key);write(prepare.core.option(row['physical_argv'],'--out')+'/case.json',b'')
            @contextmanager
            def idle(*args,**kwargs):yield dict(inventory={'CPU':'idle'},frozen_environment={'CPU':'820'})
            class Graph:
                def __init__(self,*args):pass
                def valid(self):return True
            older={'CPU':'older1011'}
            fifth=dict(plan=p1041,status=s1041)
            adopted=[]
            def freeze(row,record,*args):adopted.append(row['job_id']);return dict(native_success=record['native_success'])
            with ExitStack() as stack:
                for name,value in dict(ROOT=root,HOST_REBOOT_OBSERVATION_SHA=boot['sha256'],REQUIRED_SOURCES=tuple(names)).items():stack.enter_context(patch.object(prepare,name,value))
                for name in ('verify_inputs','require_previous_exit','require_host_reboot_observation','require_resource_proof','require_canonical_consistency'):
                    stack.enter_context(patch.object(prepare,name,return_value={}))
                stack.enter_context(patch.object(prepare,'stopped',return_value=False));stack.enter_context(patch.object(prepare,'idle_resources',side_effect=idle))
                stack.enter_context(patch.object(prepare,'FrozenInputIndex',Graph));stack.enter_context(patch.object(prepare,'freeze_completed',side_effect=freeze))
                stack.enter_context(patch.object(prepare.core,'require_environment_recheck'));stack.enter_context(patch.object(prepare.parent_core,'require_resume_plan'))
                stack.enter_context(patch.object(prepare,'require_historical_chain',return_value=(original,older,fifth)))
                stack.enter_context(patch.object(prepare.owner,'require_complete_VAL_release'));stack.enter_context(patch.object(prepare.owner,'require_resource_qualification'))
                stack.enter_context(patch.object(prepare.base,'VAL_RELEASE_SHA',point(prepare.VAL_RELEASE)['sha256']))
                stack.enter_context(patch.object(prepare.old_audit,'require_original_plan'))
                stack.enter_context(patch.object(prepare,'read_formal_source',return_value=({},[{}]*640,dict(policy_sha256=policy,routes=dict(common_author_RML_profile={})))) )
                stack.enter_context(patch.object(prepare,'outcome_protocol_binding',return_value=protocol));stack.enter_context(patch.object(prepare,'fingerprint',return_value='f'*64))
                result=prepare.prepare(root=root,environment_recheck_sha256=point(prepare.ENVIRONMENT_RECHECK)['sha256'],
                    resource_qualification_sha256=qualification['sha256'],reboot_audit_sha256=point(prepare.REBOOT_AUDIT)['sha256'])
            self.assertEqual((result['saved_completed'],result['imported_completed'],result['remaining']),(1419,1419,501))
            self.assertEqual(set(adopted),set(c['completed_job_ids'])-set(prior['completed_job_ids']))
            self.assertEqual(len(adopted),8)
            control=json.loads((root/prepare.CONTROL).read_bytes());lock=json.loads((root/prepare.LOCK).read_bytes())
            self.assertEqual(control['saved_completed_job_ids'],audit['saved_completed_job_ids'])
            self.assertEqual(len(json.loads((root/prepare.PREVIOUS/'status.json').read_bytes())['completed']),1419)
            self.assertTrue(all(lock['files'][k]==v for k,v in files.items()))
            for key in ('previous_terminal','previous_controller_failure','previous_DRAIN'):self.assertNotIn(key,control)
            self.assertEqual(control['host_reboot_observation'],boot)
            self.assertEqual(len(control['interrupted_attempt_evidence']),501)
            self.assertFalse((root/prepare.core.RUNTIME).exists());self.assertFalse((root/prepare.core.OUTPUT).exists())


if __name__=='__main__':unittest.main()
