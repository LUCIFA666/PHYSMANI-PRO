"""Temp/memory-only preparation and loader contract; no production gate runs."""
from contextlib import ExitStack,contextmanager,redirect_stdout
from copy import deepcopy
import importlib.util
import io,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import prepare_composite_TEST_resume_v1010_r10 as prepare
from scripts import run_composite_TEST_owner_v956_r10 as parent_owner
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as middle_owner
spec=importlib.util.spec_from_file_location('CPU1010fixture',Path(__file__).with_name('test_composite_TEST_resume_v1010.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class PreparationTests(unittest.TestCase):
    def test_independent1005_proof_requires_three_real_witnesses_CPU4_and_actual_audit(self):
        ptr=lambda path:dict(path=path,sha256='a'*64)
        control=dict(resource_profile=prepare.PROFILE,execution_implementation={'same':'901'},policy_sha256='e'*64,
            author_outcome_protocol={'same':'741'},resource_qualifier=ptr(prepare.QUALIFIER))
        keys=[f'CPU_episode{i}_physmani' if i<3 else f'CPU_episode{i}_cv_geometry' for i in range(12)]
        intervals={key:[3 if i<8 else 12,10 if i<8 else 20,100+i,1000+i] for i,key in enumerate(keys)}
        CPU_intervals={key:[0 if i<4 else 1 if i<8 else 4,1 if i<4 else 2 if i<8 else 5,300+i,3000+i] for i,key in enumerate(keys)}
        q=dict(schema=prepare.owner.QUALIFICATION_SCHEMA,complete=True,split='train',source_sha256='a'*64,resource_profile=prepare.PROFILE,
            execution_entry=control['execution_implementation'],policy_sha256='e'*64,author_outcome_protocol=control['author_outcome_protocol'],case_count=12,
            infrastructure_failed=0,actual_CPU_preflight_peak=4,cleanup_completed=True,resource_bounds_verified=True,overlap_verified=True,
            independent_audit_complete=True,formal_score_contribution=False,TEST_read=False,observed_native_peak=8,observed_GPU_peak=3,
            joint_supply_overlap=dict(complete=True,independently_swept_all_child_intervals=True,native_slots=8,GPU_slots=3,CPU_preflight_slots=4,
                each_job_CPU_checks=1,each_job_Native_attempts=1,simultaneous_robot_ticks_or_hardware_saturation_claimed=False,
                witness=dict(monotonic_ns=4,native_job_ids=keys[:8],CPU_job_ids=keys[8:],native_slots=8,GPU_slots=3,CPU_preflight_slots=4)),
            actual_independent_audit=ptr(prepare.QUALIFICATION_LAUNCH),control=ptr('CPU/control.json'),lock=ptr('CPU/inputs.json'),
            launcher_overlap=dict(independently_swept_all_child_intervals=True,overlap_verified=True,observed_native_peak=8,observed_GPU_peak=3,intervals=intervals),
            CPU_preflight_overlap=dict(independently_swept_all_child_intervals=True,overlap_verified=True,observed_CPU_preflight_peak=4,intervals=CPU_intervals),
            hardware_observation=dict(actual_three_compute_contexts_observed=True,valid_samples=1,unknown_samples=0,required_headroom_mib=2048,minimum_sampled_headroom_mib=2048,
                actual_three_compute_contexts_during_joint8_3_4_observed=True,continuous_hardware_bound_claimed=False,
                witness=dict(jobs={key:dict(pid=200+i,start_ticks=2000+i) for i,key in enumerate(keys[:3])},headroom_mib=2048)),
            inputs={prepare.CANARY_RUNTIME+'/plan.json':'a'*64},native_failures_retained=True,raw_manifest_model_contents_read=False,wall_clock_contention_invariance_claimed=False)
        launch=dict(source_sha256='a'*64,pid=900,start_ticks=9000,control=q['control'],lock=q['lock'],frozen_environment={'CPU':'820'},
            argv=[prepare.QUALIFIER,'--control',q['control']['path'],'--lock',q['lock']['path'],'--lock-sha256','a'*64])
        with patch.object(prepare.owner,'alive',return_value=False),patch.object(prepare.base.environment_auditor,'require_environment_receipt',return_value=dict(observer_start_ticks=9000)):
            self.assertIs(prepare.require_resource_proof(q,control,audit_launch=launch),q)
            for mode in ('twoGPU','twoCPU','samePID','unknown','unknownbool','runner','noindependent','nojoint','nonoverlap','samejob','hardware_nojoint','rawinput'):
                changed=deepcopy(q)
                if mode=='twoGPU':changed['hardware_observation']['witness']['jobs'].pop(keys[0])
                elif mode=='twoCPU':changed['CPU_preflight_overlap']['observed_CPU_preflight_peak']=2
                elif mode=='samePID':changed['hardware_observation']['witness']['jobs'][keys[1]]=deepcopy(changed['hardware_observation']['witness']['jobs'][keys[0]])
                elif mode=='unknown':changed['hardware_observation']['unknown_samples']=1
                elif mode=='unknownbool':changed['hardware_observation']['unknown_samples']=False
                elif mode=='runner':changed['actual_independent_audit']['path']=prepare.CANARY_RUNTIME+'/run_summary.json'
                elif mode=='noindependent':changed['independent_audit_complete']=False
                elif mode=='nojoint':changed['joint_supply_overlap']={}
                elif mode=='nonoverlap':changed['CPU_preflight_overlap']['intervals'][keys[8]][0]=6
                elif mode=='samejob':changed['joint_supply_overlap']['witness']['CPU_job_ids'][0]=keys[0]
                elif mode=='hardware_nojoint':changed['hardware_observation']['actual_three_compute_contexts_during_joint8_3_4_observed']=False
                else:changed['inputs']={'data/raw.hdf5':'a'*64}
                with self.subTest(mode=mode),self.assertRaises(ValueError):prepare.require_resource_proof(changed,control,audit_launch=launch)

    def test_default_disabled_and_all_three_external_pins_before_IO(self):
        with patch.object(prepare,'prepare',side_effect=AssertionError('disabled')),redirect_stdout(io.StringIO()) as stream:prepare.main([])
        self.assertIn('"execution_enabled": false',stream.getvalue())
        with patch.object(prepare,'stopped',side_effect=AssertionError('must check all pins first')):
            for missing in ('environment_recheck_sha256','resource_qualification_sha256','concurrency_authorization_sha256'):
                args=dict(environment_recheck_sha256='a'*64,resource_qualification_sha256='b'*64,concurrency_authorization_sha256='c'*64);args[missing]=None
                with self.subTest(missing=missing),self.assertRaises(ValueError):prepare.prepare(**args)

    def test_dynamic_clean_DRAIN_partitions_no_nativeFalse_filter(self):
        _,p,s,t,*_=f.fixture();before=deepcopy((p,s,t));part=prepare.partition(p,s,t)
        self.assertEqual((len(part['completed']),len(part['failed']),len(part['remaining'])),(1200,0,720))
        self.assertEqual((p,s,t),before)

    def test_original_three_evidence_tiers_and_actual_751_901_paths(self):
        _,p,*_=f.fixture()
        for index,runtime,size in ((0,prepare.core.ORIGINAL,11),(134,prepare.core.INTERMEDIATE,12),(584,prepare.PREVIOUS,12)):
            row=p['jobs'][index];paths=prepare.old_evidence_paths(row)
            self.assertEqual(row['evidence_runtime'],runtime);self.assertEqual(len(paths),size)
            self.assertTrue(paths[0].startswith(runtime+'/CPU/'))

    def test_true_partial958_failure_and_exited956_required(self):
        _,p,s,t,*_=f.fixture()
        old=dict(plan=p,status=s,terminal=t,launch=dict(pid=956,start_ticks=95),controller_launch=dict(pid=958,start_ticks=98),
            controller_status=dict(stage='requires_concrete_repair',automatic_retry=False),controller_failure=dict(complete=False,user_stop=False,automatic_retry=False))
        self.assertEqual(prepare.require_previous_exit(old,lambda _:False)['infrastructure_failed'],0)
        for key in ('complete','user_stop'):
            changed=deepcopy(old);changed['controller_failure'][key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):prepare.require_previous_exit(changed,lambda _:False)
        with self.assertRaises(ValueError):prepare.require_previous_exit(old,lambda _:True)

    def test_missing_future_source_and_conflicting_map_precede_any_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'stopped',return_value=False),patch.object(prepare,'REQUIRED_SOURCES',('missing1012.py',)), \
                    patch.object(prepare.old_audit,'EvidenceLedger',side_effect=AssertionError('no ledger')):
                with self.assertRaises(ValueError):prepare.prepare(root=root,environment_recheck_sha256='a'*64,resource_qualification_sha256='b'*64,concurrency_authorization_sha256='c'*64)
            self.assertFalse((root/prepare.SETUP).exists())
        with self.assertRaises(ValueError):prepare.merge_maps({'original':'a'*64},{'original':'b'*64})

    def test_failed_partial_inventory_is_bounded_and_alias_refused(self):
        _,p,*_=f.fixture();row=p['jobs'][1200]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/prepare.core.option(row['physical_argv'],'--out');out.mkdir(parents=True)
            (out/'partial.json').write_bytes(b'{"partial":true}')
            ledger=prepare.old_audit.EvidenceLedger(root);value=prepare.failed_attempt_inventory(root,row,ledger)
            self.assertTrue(value['infrastructure_attempt_not_native_outcome']);self.assertEqual(len(value['partial_files']),1)
            (out/'alias').symlink_to(root/'unopened_raw')
            with self.assertRaises(ValueError):prepare.failed_attempt_inventory(root,row,ledger)

    def test_qualification_allowlist_only_known12TRAIN_metadata(self):
        valid=prepare.CANARY_OUTPUT+'/pick_moving_target_on_the_table/episode105/physmani/case.json'
        self.assertTrue(prepare.qualification_path_allowed(valid))
        for value in ('data/raw.hdf5',prepare.QUALIFICATION,prepare.CANARY_OUTPUT+'/pick_moving_target_on_the_table/episode999/physmani/case.json'):
            self.assertFalse(prepare.qualification_path_allowed(value))

    def test_full_temp_prepare_and_actual1011_loader_bind_four_tier_control(self):
        core=prepare.core;c,parent,status,terminal,middle,prior,original,ancestor,middle_status,middle_terminal=f.fixture()
        policy='e'*64;protocol={'CPU':'same741'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def point(name):return dict(path=name,sha256=prepare.file_sha(root/name))
            def write(name,value):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode());return point(name)
            names={*prepare.bootstrap.SOURCES,*prepare.QUALIFICATION_SOURCES,core.SOURCE,core.OWNER,core.AUDITOR,core.CONTROLLER,prepare.SOURCE,prepare.RECHECK_SOURCE,
                'scripts/run_composite_TEST_owner_v956_r10.py','scripts/continue_composite_TEST_v958_r10.py'}
            for name in names:write(name,('CPU source '+name).encode())
            manifest=dict(path='CPU_only/test_manifest.json',sha256='9'*64)
            suite=dict(author_outcome_protocol=protocol,source_manifests=dict(test=manifest),formal_case_driver=point(core.ORIGINAL_ENTRY),formal_admission_guard={'CPU':'guard'})
            write(prepare.SUITE,suite);write(prepare.VAL_RELEASE,dict(complete=True,policy_sha256=policy,author_outcome_protocol=protocol))
            seed=write('CPU_only/input_stub',b'CPU bytes');files={str(root/seed['path']):seed['sha256']}
            write(prepare.SUITE_LOCK,dict(files=files));write(prepare.SOURCE_INPUTS,dict(schema='r10_full16_post_admission_split_source_lock_v1',split='test',source_count=640,
                policy_sha256=policy,TEST_used_for_fitting=False,source_manifest_sha256=manifest['sha256'],files=files))
            env={key:write(name,{'CPU':key}) for key,name in prepare.base.ENVIRONMENT_PATHS.items()}
            for name in prepare.base.ENVIRONMENT_EXTRA:write(name,{'CPU':'env extra'})
            standing=write(prepare.base.STANDING,dict(continue_until_user_stop=True,user_stop_received=False))
            authorization=write(prepare.AUTHORIZATION,dict(user_authorized_concurrency_change=True))
            shared={name:point(name) for name in prepare.bootstrap.SOURCES}
            bc=prior['execution_entry']['batch_control']['path'];bl=prior['execution_entry']['batch_lock']['path']
            write(bc,prepare.base.formal_bootstrap_control(suite));write(bl,dict(schema=prepare.bootstrap.LOCK_SCHEMA,execution_enabled=True,files={str(root/n):p['sha256'] for n,p in shared.items()}))
            entry=dict(kind='batched901',pointer=shared[core.BATCH_ENTRY],batch_control=point(bc),batch_lock=point(bl))
            for row in original['jobs']:row.update(worker_sha256=shared[core.ORIGINAL_ENTRY]['sha256'],admission_driver=shared[core.ORIGINAL_ENTRY])
            ancestor.update(execution_entry=entry)
            middle['jobs']=middle_owner.resumed_jobs(original,ancestor,entry)
            prior.update(execution_entry=entry,policy_sha256=policy,author_outcome_protocol=protocol,
                suite=point(prepare.SUITE),suite_lock=point(prepare.SUITE_LOCK),source_inputs=point(prepare.SOURCE_INPUTS),validation_release=point(prepare.VAL_RELEASE),
                execution_implementation=prepare.owner.execution_implementation(entry,shared),resource_profile=deepcopy(prepare.base.PROFILE),
                resource_qualification={'path':'CPU_only/original904.json','sha256':'a'*64},resource_qualifier={'path':'CPU_only/old904.py','sha256':'b'*64},
                standing_continuation_record=standing,concurrency_authorization_record={'path':'CPU_only/old_authorization.json','sha256':'c'*64},**env)
            parent['jobs']=prepare.parent_core.resumed_jobs(middle,prior,entry)
            original_status=dict(status,completed={r['job_id']:deepcopy(status['completed'][r['job_id']]) for r in original['jobs'][:134]},failed={})
            original_terminal=dict(terminal,completed=134,failed=0)
            for key,value in (('launch',dict(pid=753,start_ticks=75300)),('plan',original),('status',original_status),('terminal',original_terminal)):
                prior['original_'+key]=write(core.ORIGINAL+'/'+key+'.json',value)
            for key,value in (('launch',dict(pid=902,start_ticks=90200)),('plan',middle),('status',middle_status),('terminal',middle_terminal)):
                prior['previous_'+key]=write(core.INTERMEDIATE+'/'+key+'.json',value)
            prior['previous_resume_control']=write(prepare.base.CONTROL,ancestor)
            prior['previous_resume_control_lock']=write(prepare.base.LOCK,dict(schema=prepare.old_audit.owner.LOCK_SCHEMA,approved_for_execution=True,files=files))
            write(prepare.PREVIOUS_CONTROL,prior);write(prepare.PREVIOUS_LOCK,dict(schema=prepare.parent_core.LOCK_SCHEMA,approved_for_execution=True,files=files))
            common=dict(resume_control=point(prepare.PREVIOUS_CONTROL),resume_control_lock=point(prepare.PREVIOUS_LOCK))
            previous_launch=dict(common,pid=956,start_ticks=95600,source_sha256=point('scripts/run_composite_TEST_owner_v956_r10.py')['sha256'])
            write(prepare.PREVIOUS+'/launch.json',previous_launch);write(prepare.PREVIOUS+'/plan.json',parent);write(prepare.PREVIOUS+'/status.json',status);write(prepare.PREVIOUS+'/terminal.json',terminal)
            write(prepare.PREVIOUS+'/DRAIN',dict(CPU='actual drain'))
            write(prepare.PREVIOUS_CONTROLLER+'/launch.json',dict(common,pid=958,start_ticks=95800,source_sha256=point('scripts/continue_composite_TEST_v958_r10.py')['sha256']))
            write(prepare.PREVIOUS_CONTROLLER+'/status.json',dict(stage='requires_concrete_repair',automatic_retry=False));write(prepare.PREVIOUS_CONTROLLER+'/failure.json',dict(complete=False,user_stop=False,automatic_retry=False))
            audit_launch=write(prepare.QUALIFICATION_LAUNCH,dict(CPU='independent1005'))
            train_control=write('CPU_only/train_control.json',{'CPU':'TRAIN control'});train_lock=write('CPU_only/train_lock.json',{'CPU':'TRAIN lock'})
            qualification=write(prepare.QUALIFICATION,dict(actual_independent_audit=audit_launch,inputs={},control=train_control,lock=train_lock))
            proof=f.old.observation({key:prior[key] for key in ('suite_lock',*env)})
            proof.update(source=point(prepare.RECHECK_SOURCE),suite_lock=prior['suite_lock']);proof['actual_environment'].update(manifest=env['environment_manifest'],wrapper=env['environment_wrapper'],namespace_qualification=env['environment_namespace_proof'])
            write(prepare.ENVIRONMENT_RECHECK,proof)
            @contextmanager
            def idle(*args,**kwargs):yield dict(inventory={'CPU':'idle'},frozen_environment={'CPU':'820'})
            class Graph:
                def __init__(self,*args):pass
                def valid(self):return True
            inspected=[]
            def freeze(row,record,*args):inspected.append(row['evidence_runtime']);return dict(native_success=record['native_success'],motion_pass=record['motion_pass'])
            with ExitStack() as stack:
                for module in (prepare,prepare.owner):stack.enter_context(patch.object(module,'ROOT',root))
                for name,value in (('REQUIRED_SOURCES',tuple(names)),):stack.enter_context(patch.object(prepare,name,value))
                stack.enter_context(patch.object(prepare,'stopped',return_value=False));stack.enter_context(patch.object(prepare,'idle_resources',side_effect=idle))
                stack.enter_context(patch.object(prepare.owner,'alive',return_value=False));stack.enter_context(patch.object(prepare.owner,'require_complete_VAL_release',return_value={}))
                stack.enter_context(patch.object(prepare.base,'VAL_RELEASE_SHA',point(prepare.VAL_RELEASE)['sha256']))
                stack.enter_context(patch.object(prepare,'verify_inputs'));stack.enter_context(patch.object(prepare,'read_formal_source',return_value=({},[{}]*640,dict(policy_sha256=policy,routes=dict(common_author_RML_profile={'CPU':'RML'})))))
                stack.enter_context(patch.object(prepare,'outcome_protocol_binding',return_value=protocol));stack.enter_context(patch.object(prepare,'fingerprint',return_value='f'*64))
                stack.enter_context(patch.object(prepare.old_audit,'require_original_plan'));stack.enter_context(patch.object(prepare.old_audit,'require_resume_plan'))
                stack.enter_context(patch.object(prepare,'require_resource_proof'));stack.enter_context(patch.object(prepare.owner,'require_resource_qualification'))
                stack.enter_context(patch.object(prepare,'freeze_completed',side_effect=freeze));stack.enter_context(patch.object(prepare,'FrozenInputIndex',Graph));stack.enter_context(patch.object(prepare,'require_canonical_consistency'))
                old_bytes=(root/bc).read_bytes(),(root/bl).read_bytes()
                result=prepare.prepare(root=root,environment_recheck_sha256=point(prepare.ENVIRONMENT_RECHECK)['sha256'],resource_qualification_sha256=qualification['sha256'],concurrency_authorization_sha256=authorization['sha256'])
            self.assertEqual((result['imported_completed'],result['historical_infrastructure_failed'],result['remaining']),(1200,0,720))
            self.assertEqual([inspected.count(r) for r in (core.ORIGINAL,core.INTERMEDIATE,core.PREVIOUS)],[134,450,616])
            self.assertEqual(((root/bc).read_bytes(),(root/bl).read_bytes()),old_bytes)
            control=json.loads((root/core.CONTROL).read_bytes());lock=json.loads((root/core.LOCK).read_bytes())
            self.assertEqual(control['execution_entry'],entry);self.assertEqual(control['resource_profile'],prepare.PROFILE)
            self.assertTrue(control['resource_profile_changed']);self.assertEqual(control['previous_resource_profile'],prior['resource_profile'])
            self.assertTrue(all(str(root/n) in lock['files'] for n in (core.OWNER,core.AUDITOR,core.CONTROLLER)))
            self.assertFalse((root/core.RUNTIME).exists());self.assertFalse((root/core.OUTPUT).exists())
            args=SimpleNamespace(resume_control=core.CONTROL,resume_lock=core.LOCK,resume_lock_sha256=prepare.file_sha(root/core.LOCK),suite=prepare.SUITE,suite_lock=prepare.SUITE_LOCK,source_inputs=prepare.SOURCE_INPUTS,validation_release=prepare.VAL_RELEASE)
            with ExitStack() as stack:
                stack.enter_context(patch.object(prepare.owner,'ROOT',root));stack.enter_context(patch.object(prepare.owner,'__file__',str(root/core.OWNER)))
                stack.enter_context(patch.object(prepare.owner,'locked_pointer'));stack.enter_context(patch.object(prepare.owner,'verify_inputs'));stack.enter_context(patch.object(prepare.owner,'alive',return_value=False))
                stack.enter_context(patch.object(prepare.owner,'require_complete_VAL_release',return_value={}));stack.enter_context(patch.object(prepare.owner,'require_resource_qualification'))
                stack.enter_context(patch.object(prepare.owner,'require_history',return_value=(prior,{'launch':{'CPU':'753'}},{'launch':previous_launch})))
                stack.enter_context(patch.object(prepare.owner.environment_owner,'require_frozen_environment',return_value={'CPU':'820'}))
                admitted,admitted_files,admitted_sources=prepare.owner.load_control(args)
            self.assertEqual(admitted,control);self.assertEqual(admitted_files,lock['files']);self.assertEqual(admitted_sources,shared)


if __name__=='__main__':unittest.main()
