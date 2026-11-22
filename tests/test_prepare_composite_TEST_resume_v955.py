"""Temporary/memory preparation checks; no actual scientific gate is run."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import importlib.util
import io
import json
import tempfile
from types import SimpleNamespace
import unittest

from scripts import prepare_composite_TEST_resume_v955_r10 as prepare

spec=importlib.util.spec_from_file_location('_composite955fixture',Path(__file__).with_name('test_composite_TEST_resume_v955.py'))
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
fixture=fixture_module.fixture


class PreparationTests(unittest.TestCase):
    def test_default_main_is_disabled_without_any_input_read(self):
        with patch.object(prepare,'prepare',side_effect=AssertionError('must stay inactive')),redirect_stdout(io.StringIO()) as output:
            prepare.main([])
        self.assertIn('"execution_enabled": false',output.getvalue())

    def test_partition_dynamic584_and_six_infra_kept(self):
        c,p,s,t,*_=fixture();before=deepcopy((p,s,t));value=prepare.partition(p,s,t)
        self.assertEqual((len(value['completed']),len(value['failed']),len(value['remaining'])),(584,6,1336))
        self.assertEqual((p,s,t),before)

    def test_evidence_paths_keep_real_tier_and_901_extra_sidecar(self):
        _,p,*_=fixture()
        original,new=p['jobs'][0],p['jobs'][134]
        for row,size in ((original,11),(new,12)):
            paths=prepare.old_evidence_paths(row);self.assertEqual(len(paths),size)
            self.assertTrue(paths[0].startswith(row['evidence_runtime']+'/CPU/'))
            out=prepare.core.option(row['physical_argv'],'--out')
            self.assertIn(out+'/case.json',paths)
        self.assertIn(prepare.core.option(new['physical_argv'],'--out')+'/batch_driver_receipt.json',prepare.old_evidence_paths(new))

    def test_true_failed_drain_not_a_fabricated_success_and_PID_live_refused(self):
        _,p,s,t,*_=fixture()
        previous=dict(plan=p,status=s,terminal=t,launch=dict(pid=902,start_ticks=12),controller_launch=dict(pid=907,start_ticks=13),
            controller_status=dict(stage='requires_concrete_repair',automatic_retry=False),
            controller_failure=dict(complete=False,user_stop=False,automatic_retry=False))
        self.assertEqual(prepare.require_previous_exit(previous,lambda _:False)['infrastructure_failed'],6)
        with self.assertRaises(ValueError):prepare.require_previous_exit(previous,lambda _:True)
        previous['controller_failure']['complete']=True
        with self.assertRaises(ValueError):prepare.require_previous_exit(previous,lambda _:False)

    def test_missing_future_source_rejected_before_ledger_or_mkdir(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'stopped',return_value=False),\
                    patch.object(prepare,'REQUIRED_SOURCES',('scripts/actual_future_missing957.py',)),\
                    patch.object(prepare.old_audit,'EvidenceLedger',side_effect=AssertionError('no input reads yet')):
                with self.assertRaisesRegex(ValueError,'source/test is not ready'):
                    prepare.prepare(root=root,environment_recheck_sha256='a'*64)
            self.assertFalse((root/prepare.SETUP).exists())

    def test_bad_external954_pin_rejected_before_any_environment_or_input_check(self):
        with patch.object(prepare,'stopped',side_effect=AssertionError('not before external SHA')):
            with self.assertRaisesRegex(ValueError,'external actual954'):
                prepare.prepare(environment_recheck_sha256=None)

    def test_merge_preserves_all_keys_and_rejects_conflict_before_any_setup(self):
        before={'original': 'a'*64};other={'new': 'b'*64}
        self.assertEqual(prepare.merge_maps(before,other),dict(before,**other))
        with self.assertRaises(ValueError):prepare.merge_maps(before,{'original':'c'*64})
        self.assertEqual(before,{'original':'a'*64})

    def test_failed_attempt_freeze_only_known_log_and_original_partial_dir(self):
        _,p,*_=fixture();row=p['jobs'][584]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/prepare.PREVIOUS/'CPU'/row['job_id']/'stderr.log'
            log.parent.mkdir(parents=True);log.write_bytes(b'invented infrastructure traceback')
            out=root/prepare.core.option(row['physical_argv'],'--out');out.mkdir(parents=True)
            partial=out/'partial.json';partial.write_bytes(b'{"incomplete":true}')
            ledger=prepare.old_audit.EvidenceLedger(root)
            value=prepare.failed_attempt_inventory(root,row,ledger)
            self.assertTrue(value['infrastructure_attempt_not_native_outcome'])
            self.assertEqual([x['path'] for x in value['log_files']],[str(log.relative_to(root))])
            self.assertEqual([x['path'] for x in value['partial_files']],[str(partial.relative_to(root))])
            ledger.close();self.assertEqual(partial.read_bytes(),b'{"incomplete":true}')

    def test_failed_partial_alias_refused_without_reading_target(self):
        _,p,*_=fixture();row=p['jobs'][584]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/prepare.core.option(row['physical_argv'],'--out');out.mkdir(parents=True)
            target=root/'never_read_raw';target.write_bytes(b'not an input');(out/'alias').symlink_to(target)
            ledger=prepare.old_audit.EvidenceLedger(root)
            with self.assertRaisesRegex(ValueError,'symlink'):prepare.failed_attempt_inventory(root,row,ledger)
            self.assertNotIn(str(target),ledger.used)

    def test_full_temp_preparation_retains_old_controls_and_freezes_all_three_stage_sources(self):
        """Exercise commit orchestration with temp metadata and injected gates.

        Scientific admission/case validators and resource helpers are mocks; no
        experiment or native source is imported from the invented source files.
        """
        core=prepare.core;_,parent,status,terminal,original,prior=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def write(name,value):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value,ensure_ascii=False))
                return point(name)
            def point(name):return dict(path=name,sha256=prepare.file_sha(root/name))
            names={*prepare.bootstrap.SOURCES,core.OWNER,core.AUDITOR,core.CONTROLLER,core.SOURCE,
                prepare.SOURCE,prepare.RECHECK_SOURCE,prepare.base.QUALIFIER,
                'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py','scripts/continue_concurrent_TEST_v907_r10.py'}
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('invented '+name).encode())
            protocol={'fixture':'original author outcome'};policy='e'*64
            manifest=dict(path='fixture/original_TEST_manifest.json',sha256='9'*64)
            suite=dict(author_outcome_protocol=protocol,source_manifests=dict(test=manifest),
                formal_case_driver=point(core.ORIGINAL_ENTRY),formal_admission_guard={'fixture':'guard'})
            write(prepare.VAL_RELEASE,dict(complete=True,policy_sha256=policy,author_outcome_protocol=protocol))
            write(prepare.SUITE,suite)
            seed_path=root/'fixture/frozen_scientific_stub';seed_path.parent.mkdir(parents=True,exist_ok=True);seed_path.write_bytes(b'invented bound stub')
            seed_files={str(seed_path):prepare.file_sha(seed_path)}
            write(prepare.SUITE_LOCK,dict(files=seed_files))
            write(prepare.SOURCE_INPUTS,dict(schema='r10_full16_post_admission_split_source_lock_v1',split='test',
                source_count=640,policy_sha256=policy,TEST_used_for_fitting=False,
                source_manifest_sha256=manifest['sha256'],files=seed_files))
            env={key:write(name,{'fixture':key}) for key,name in prepare.base.ENVIRONMENT_PATHS.items()}
            for name in prepare.base.ENVIRONMENT_EXTRA:write(name,{'fixture':'environment supplement'})
            standing=write(prepare.base.STANDING,dict(continue_until_user_stop=True,user_stop_received=False))
            authorization=write(prepare.base.AUTHORIZATION,dict(user_authorized_concurrency_change=True))
            shared={name:point(name) for name in prepare.bootstrap.SOURCES}
            bc=prior['execution_entry']['batch_control']['path'];bl=prior['execution_entry']['batch_lock']['path']
            write(bc,prepare.base.formal_bootstrap_control(suite))
            write(bl,dict(schema=prepare.bootstrap.LOCK_SCHEMA,execution_enabled=True,
                files={str(root/name):value['sha256'] for name,value in shared.items()}))
            entry=dict(kind='batched901',pointer=shared[core.BATCH_ENTRY],batch_control=point(bc),batch_lock=point(bl))
            for row in original['jobs']:
                row.update(worker_sha256=shared[core.ORIGINAL_ENTRY]['sha256'],admission_driver=shared[core.ORIGINAL_ENTRY])
            prior.update(execution_entry=entry,user_authorized_resume=True,standing_authorization_reused=True,
                new_user_event_invented=False,physical_policy_changed=False,policy_sha256=policy,
                completed_key_sha256=core.digest(sorted(prior['completed_job_ids'])),
                remaining_job_ids=sorted(r['job_id'] for r in original['jobs'][134:]),
                suite=point(prepare.SUITE),suite_lock=point(prepare.SUITE_LOCK),source_inputs=point(prepare.SOURCE_INPUTS),
                validation_release=point(prepare.VAL_RELEASE),author_outcome_protocol=protocol,
                resource_profile=prepare.PROFILE,resource_qualifier=point(prepare.base.QUALIFIER),
                execution_implementation=prepare.owner.execution_implementation(entry,shared),
                standing_continuation_record=standing,concurrency_authorization_record=authorization,**env)
            prior['remaining_key_sha256']=core.digest(prior['remaining_job_ids'])
            original_status=deepcopy(status);original_status.update(completed={key:deepcopy(status['completed'][key])
                for key in prior['completed_job_ids']},failed={})
            original_terminal=dict(terminal,completed=134,failed=0)
            for key,value in (('launch',dict(pid=753,start_ticks=75300)),('plan',original),
                    ('status',original_status),('terminal',original_terminal)):
                prior['previous_'+key]=write(core.ORIGINAL+'/'+key+'.json',value)
            audit_launch=write(prepare.base.QUALIFICATION_LAUNCH,{'fixture':'independent904 launch'})
            train_control=write('fixture/train_control.json',{'fixture':'TRAIN control'})
            train_lock=write('fixture/train_lock.json',{'fixture':'TRAIN lock'})
            qualification=write(prepare.base.QUALIFICATION,dict(actual_independent_audit=audit_launch,
                inputs={},control=train_control,lock=train_lock))
            prior['resource_qualification']=qualification
            write(prepare.PREVIOUS_CONTROL,prior);write(prepare.PREVIOUS_LOCK,
                dict(schema=prepare.old_audit.owner.LOCK_SCHEMA,approved_for_execution=True,files=seed_files))
            parent.update(jobs=prepare.old_audit.owner.resumed_jobs(original,prior,entry))
            common=dict(resume_control=point(prepare.PREVIOUS_CONTROL),resume_control_lock=point(prepare.PREVIOUS_LOCK))
            write(prepare.PREVIOUS+'/launch.json',dict(common,pid=902,start_ticks=90200,
                source_sha256=point('scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py')['sha256']))
            write(prepare.PREVIOUS+'/plan.json',parent);write(prepare.PREVIOUS+'/status.json',status)
            write(prepare.PREVIOUS+'/terminal.json',terminal);write(prepare.PREVIOUS+'/DRAIN',dict(fixture='real drained attempt'))
            write(prepare.PREVIOUS_CONTROLLER+'/launch.json',dict(common,pid=907,start_ticks=90700,
                source_sha256=point('scripts/continue_concurrent_TEST_v907_r10.py')['sha256']))
            write(prepare.PREVIOUS_CONTROLLER+'/status.json',dict(stage='requires_concrete_repair',automatic_retry=False))
            write(prepare.PREVIOUS_CONTROLLER+'/failure.json',dict(complete=False,user_stop=False,automatic_retry=False))
            for key,failed in status['failed'].items():
                kind='CPU' if failed['stage'].startswith('CPU') else 'native'
                for name in ('launch.json','terminal.json','stdout.log','stderr.log'):
                    write(prepare.PREVIOUS+'/'+kind+'/'+key+'/'+name,{'fixture':'retained failed metadata'})
            observation_control={key:prior[key] for key in ('suite_lock',*env)}
            proof=fixture_module.observation(observation_control)
            proof.update(source=point(prepare.RECHECK_SOURCE),suite_lock=prior['suite_lock'])
            proof['actual_environment'].update(manifest=env['environment_manifest'],wrapper=env['environment_wrapper'],
                namespace_qualification=env['environment_namespace_proof'])
            write(prepare.ENVIRONMENT_RECHECK,proof)
            @contextmanager
            def idle(*args,**kwargs):yield dict(inventory={'fixture':'idle'},frozen_environment={'fixture':'current820'})
            inspected=[]
            def freeze(row,record,*args):
                inspected.append((row['job_id'],row['evidence_runtime']))
                return dict(native_success=record['native_success'],motion_pass=record['motion_pass'])
            class Graph:
                def __init__(self,*args):pass
                def valid(self):return True
            with ExitStack() as stack:
                for module in (prepare,prepare.owner):stack.enter_context(patch.object(module,'ROOT',root))
                stack.enter_context(patch.object(prepare,'REQUIRED_SOURCES',tuple(sorted(names))))
                stack.enter_context(patch.object(prepare,'stopped',return_value=False))
                stack.enter_context(patch.object(prepare,'idle_resources',side_effect=idle))
                stack.enter_context(patch.object(prepare.owner,'alive',return_value=False))
                stack.enter_context(patch.object(prepare.owner,'require_complete_VAL_release',return_value={}))
                stack.enter_context(patch.object(prepare.base,'VAL_RELEASE_SHA',point(prepare.VAL_RELEASE)['sha256']))
                stack.enter_context(patch.object(prepare,'verify_inputs'))
                stack.enter_context(patch.object(prepare,'read_formal_source',return_value=({},[{} for _ in range(640)],
                    dict(policy_sha256=policy,routes=dict(common_author_RML_profile={'fixture':'RML'})))))
                stack.enter_context(patch.object(prepare,'outcome_protocol_binding',return_value=protocol))
                stack.enter_context(patch.object(prepare,'fingerprint',return_value='f'*64))
                stack.enter_context(patch.object(prepare.old_audit,'require_original_plan'))
                stack.enter_context(patch.object(prepare,'require_resource_proof'))
                stack.enter_context(patch.object(prepare.owner,'require_resource_qualification'))
                stack.enter_context(patch.object(prepare,'freeze_completed',side_effect=freeze))
                stack.enter_context(patch.object(prepare,'FrozenInputIndex',Graph))
                stack.enter_context(patch.object(prepare,'require_canonical_consistency'))
                old_control=(root/bc).read_bytes();old_lock=(root/bl).read_bytes()
                result=prepare.prepare(root=root,environment_recheck_sha256=point(prepare.ENVIRONMENT_RECHECK)['sha256'])
            self.assertEqual((len(inspected),result['historical_infrastructure_failed'],result['remaining']),(584,6,1336))
            self.assertEqual(sum(runtime==core.ORIGINAL for _,runtime in inspected),134)
            self.assertEqual(sum(runtime==core.PREVIOUS for _,runtime in inspected),450)
            control=json.loads((root/core.CONTROL).read_bytes());lock=json.loads((root/core.LOCK).read_bytes())
            self.assertEqual(control['execution_entry'],entry)
            self.assertEqual((root/bc).read_bytes(),old_control);self.assertEqual((root/bl).read_bytes(),old_lock)
            self.assertEqual(control['result_auditor'],control['resume_auditor'])
            self.assertEqual(control['resume_owner']['path'],core.OWNER)
            self.assertEqual(control['resume_controller']['path'],core.CONTROLLER)
            self.assertTrue(all(str(root/name) in lock['files'] for name in (core.OWNER,core.AUDITOR,core.CONTROLLER)))
            self.assertFalse((root/core.RUNTIME).exists());self.assertFalse((root/core.OUTPUT).exists())
            # Feed the exact newly prepared temp contract to the real956 loader.
            # All scientific/resource functions remain mocks; only temp metadata
            # and source stubs exist, and no case, raw or model bytes are supplied.
            args=SimpleNamespace(resume_control=core.CONTROL,resume_lock=core.LOCK,
                resume_lock_sha256=prepare.file_sha(root/core.LOCK),suite=prepare.SUITE,
                suite_lock=prepare.SUITE_LOCK,source_inputs=prepare.SOURCE_INPUTS,validation_release=prepare.VAL_RELEASE)
            with ExitStack() as stack:
                stack.enter_context(patch.object(prepare.owner,'ROOT',root))
                stack.enter_context(patch.object(prepare.owner,'__file__',str(root/core.OWNER)))
                stack.enter_context(patch.object(prepare.owner,'locked_pointer'))
                stack.enter_context(patch.object(prepare.owner,'verify_inputs'))
                stack.enter_context(patch.object(prepare.owner,'alive',return_value=False))
                stack.enter_context(patch.object(prepare.owner,'require_complete_VAL_release',return_value={}))
                stack.enter_context(patch.object(prepare.owner,'require_resource_qualification'))
                stack.enter_context(patch.object(prepare.owner.environment_owner,'require_frozen_environment',return_value={'fixture':'current820'}))
                admitted,admitted_files,admitted_sources=prepare.owner.load_control(args)
            self.assertEqual(admitted,control);self.assertEqual(admitted_files,lock['files'])
            self.assertEqual(admitted_sources,shared)


if __name__=='__main__':unittest.main()
