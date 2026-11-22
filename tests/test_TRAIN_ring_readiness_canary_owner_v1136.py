"""CPU-only bounded owner checks; every child and predecessor is synthetic."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import run_TRAIN_ring_readiness_canary_v1136_r10 as owner


def roster():
    first={'insert_onto_rotating_peg':['episode101','episode102','episode105','episode112'],
        'beat_the_rotating_buzz':['episode10','episode104','episode105','episode107']}
    return dict(split='train',episodes=[dict(task=t,raw_episode_id=r) for t,rows in first.items() for r in rows+['episode999']])




class BoundedOwner(unittest.TestCase):
    def protocol_fixture(self, root, row, lock):
        preparer='our_mani/multitask/train_ring_readiness_overlay_v1135.py'
        for name in (owner.WORKER,preparer):
            path=root/name
            if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic reviewed source')
        ref=lambda name:dict(path=name,sha256=owner.sha(root/name))
        inherited=dict(schema='r10_full16_fixed_TRAIN_public_phase_overlay_v1132',
            source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
            execution_branch='core_selected_integration',core_factory=dict(module='our_mani.multitask.public_task_phase_integration_v1131'),
            changed_factor=owner.PARENT_FACTOR,
            public_phase_component_binding=dict(task_rule=dict(task=row['task'],
                source_sha256='00116825ce901934bcf6ea72fbe2b4727e0a02fa4783a100aaf3504c5914318e'),
                integration=dict(schema='r10_estimated_public_ring_seating_and_buzz_approach_integration_v1131')))
        return dict(schema=owner.DIAGNOSTIC_SCHEMA,diagnostic_source_overlay_lock=lock,
            actual_worker=ref(owner.WORKER),actual_preparer=ref(preparer),diagnostic_only=True,
            parent_source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
            inherited_1132_execution_protocol=inherited,inherited_1132_identity_sha256='a'*64,
            formal_score_contribution=False,formal_VAL_TEST_qualification=False)

    def quality_fixture(self):
        count=dict(calls=1,returns=1,exceptions=0)
        probe=dict(schema='r10_evaluation_existing_public_readiness_probe_v1134',
            restored=True,errors=[],recording_errors=0,dropped_events=0,
            restoration_checks=[dict(restored=True)],events=[
                dict(method=m,call_sequence=i+1,original_returned=True,original_return=False)
                for i,m in enumerate(('frontend.ready','collision_port.readiness',
                    'source_policy.candidates','generator.observed_reference'))],
            method_call_counts={m:dict(count) for m in ('frontend.ready','collision_port.readiness',
                'source_policy.candidates','generator.observed_reference')},
            geometry_calls=1,geometry_returns=1,geometry_exceptions=0,
            restoration_error=None,scope_exception_type=None,sysprofile_installed=False,
            original_return_or_exception_rewritten=False,extra_readiness_reference_forecast_or_native_calls=0)
        trace=dict(schema='r10_evaluation_existing_public_execution_trace_v1126',restored=True,
            restoration_error=None,restoration_checks=[dict(restored=True)],errors=[],scope_exception_type=None,
            frame_calls=1,frame_returns=1,frame_exceptions=0,plan_calls=1,plan_returns=1,plan_exceptions=0,
            frames=[dict(timestamp=0.)],plans=[dict(original_returned=True)])
        for key in ('recording_errors','frame_dropped','plan_dropped','plan_value_truncations',
                'extra_env_observe_calls','extra_native_getter_or_condition_calls','extra_prediction_calls',
                'extra_path_step_or_FK_calls'):trace[key]=0
        return dict(schema=owner.DIAGNOSTIC_SCHEMA+'_observers',restored=True,factory_restored=True,
            writer_restored=True,parent_annotation_restored=True,observer_finalization_errors=[],
            readiness_probe=dict(readiness_probe=probe,restored=True),
            public_execution_trace=dict(trace_receipt=trace,restored=True))

    def clock_fixture(self):
        return dict(counted_clock=dict(ticks=4,dt=.05,seconds=.2),
            alignment_receipt=dict(clock=dict(ticks=4,dt=.05,simulation_seconds=.2),outcomes_are_policy_observations=False),
            native_outcome_sampling=dict(last_tick=4,active_action=None,aborted=False,completed_actions=[
                dict(start_tick=0,end_tick=4,ticks=4,max_ticks=10,complete=True)]))

    def test_default_inert_and_exact_one_first_four_source_without_queue(self):
        with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no child')),\
             patch.object(owner,'prepare',side_effect=AssertionError('no prepare')),\
             patch.object(owner,'execute',side_effect=AssertionError('no execute')),\
             patch('sys.stdout',new_callable=io.StringIO) as output:owner.main([])
        self.assertFalse(json.loads(output.getvalue())['Native_started'])
        jobs=owner.build_jobs(roster())
        self.assertEqual([(r['task'],r['raw'],r['method']) for r in jobs],list(owner.CANARIES))
        self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['kind'],'ring')
        self.assertEqual(len({r['out'] for r in jobs}),1);self.assertEqual(len({r['runtime'] for r in jobs}),1)
        self.assertTrue(all('/observed_v1136/' in r['out'] and '/observed_v1136/' in r['runtime'] for r in jobs))
        self.assertEqual(owner.FIXED_SOURCES,[dict(task=r['task'],raw_episode_id=r['raw'],method=r['method']) for r in jobs])

    def test_invalid_roster_and_exact_candidate_cli(self):
        for bad in ('split','missing','duplicate'):
            value=roster()
            if bad=='split':value['split']='test'
            if bad=='missing':value['episodes']=[r for r in value['episodes'] if r['raw_episode_id']!='episode101']
            if bad=='duplicate':value['episodes'][1]=deepcopy(value['episodes'][0])
            with self.assertRaises(ValueError):owner.build_jobs(value)
        row=owner.build_jobs(roster())[0];native=owner.command(row)
        self.assertEqual(owner.command(row,True),native+['--check'])
        self.assertIn(owner.WORKER,native)
        self.assertEqual(native[native.index('--diagnostic-lock')+1],owner.DIAGNOSTIC_LOCK)
        self.assertNotIn('--overlay-lock',native)

    def predecessor_fixture(self, root, *, bad=None):
        owner.fresh(root/owner.PREDECESSOR_PLAN,{'synthetic':'plan'})
        path=root/owner.PREDECESSOR_SOURCE;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source')
        plan_sha=owner.sha(root/owner.PREDECESSOR_PLAN);source_sha=owner.sha(path)
        ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])+(0 if bad=='live' else 1)
        launch=dict(pid=os.getpid(),start_ticks=ticks,plan=dict(path=owner.PREDECESSOR_PLAN,sha256=plan_sha))
        completed=[]
        for index,(task,raw,method) in enumerate(owner.PREDECESSOR_CANARIES):
            def ref(label):
                name=owner.PREDECESSOR+f'/synthetic/{index}_{label}.json'
                owner.fresh(root/name,{'synthetic':label});return dict(path=name,sha256=owner.sha(root/name))
            completed.append(dict(task=task,raw=raw,method=method,case=ref('case'),owned_hooks=ref('hooks'),
                CPU_check={k:ref('cpu_'+k) for k in ('launch','terminal','sidecar')},
                native={k:ref('native_'+k) for k in ('launch','terminal','sidecar')}))
        status=dict(stage='complete',active=None,failed=[],completed=completed)
        terminal=dict(complete=True,completed=2,infrastructure_failed=0,automatic_retry=False)
        owner.fresh(root/(owner.PREDECESSOR+'/launch.json'),launch)
        owner.fresh(root/(owner.PREDECESSOR+'/terminal.json'),terminal)
        completion=dict(schema=owner.EXIT_SCHEMA,unified_exec_session=16893,actual_tool_exit_observed=True,returncode=0,
            owner_pid=launch['pid'],owner_start_ticks=ticks,owner_alive=False,natural_completion=True,
            native_cases_complete=2,completed_case_infrastructure_failures=0,plan=launch['plan'],
            terminal=dict(path=owner.PREDECESSOR+'/terminal.json',sha256=owner.sha(root/(owner.PREDECESSOR+'/terminal.json'))))
        if bad=='running':status['stage']='running'
        if bad=='active':status['active']={'unexpected':'child'}
        if bad=='failed':status['failed']=[{'infrastructure':'error'}]
        if bad=='order':status['completed']=list(reversed(completed))
        if bad=='unobserved':completion['actual_tool_exit_observed']=False
        if bad=='returncode':completion['returncode']=1
        if bad=='session':completion['unified_exec_session']=58539
        if bad=='boolcode':completion['returncode']=False
        if bad=='unnatural':completion['natural_completion']=False
        if bad=='terminalsha':completion['terminal']['sha256']='changed'
        if bad=='casesha':status['completed'][0]['case']['sha256']='changed'
        owner.fresh(root/(owner.PREDECESSOR+'/status.json'),status)
        if bad!='missing_exit':owner.fresh(root/owner.PREDECESSOR_COMPLETION,completion)
        return plan_sha,source_sha

    def test_predecessor_requires_real_natural_complete_and_dead_identity_and_tool_exit0(self):
        for bad in (None,'live','running','active','failed','order','unobserved','returncode','session','boolcode','unnatural','terminalsha','casesha','missing_exit'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);plan_sha,source_sha=self.predecessor_fixture(root,bad=bad)
                with patch.object(owner,'ROOT',root),patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha),\
                     patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha):
                    if bad is None:self.assertTrue(owner.predecessor_complete()['natural_completion'])
                    else:
                        with self.assertRaises((ValueError,FileNotFoundError)):owner.predecessor_complete()

    def test_root_review_binds_exact_order_sources_exec_and_candidate_lock(self):
        for bad in (None,'order','exec','lock','released','reviewer'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);ref={'path':owner.DIAGNOSTIC_LOCK,'sha256':'candidate'}
                predecessor={'actual_exec_completion':{'path':owner.PREDECESSOR_COMPLETION,'sha256':'exit'}}
                review=dict(schema=owner.REVIEW_SCHEMA,review_complete=True,release_bounded_one=True,reviewer='root',
                    fixed_TRAIN_sources=deepcopy(owner.FIXED_SOURCES),exec_completion=predecessor['actual_exec_completion'],
                    diagnostic_lock=ref,diagnostic_only=True)
                if bad=='order':review['fixed_TRAIN_sources'][0]['raw_episode_id']='episode102'
                if bad=='exec':review['exec_completion']={'sha256':'wrong'}
                if bad=='lock':review['diagnostic_lock']={'sha256':'wrong'}
                if bad=='released':review['release_bounded_one']=False
                if bad=='reviewer':review['reviewer']='unreviewed'
                owner.fresh(root/owner.ROOT_REVIEW,review)
                with patch.object(owner,'ROOT',root):
                    if bad is None:self.assertEqual(owner.root_review(ref,predecessor)['path'],owner.ROOT_REVIEW)
                    else:
                        with self.assertRaises(ValueError):owner.root_review(ref,predecessor)

    def test_prepare_blocked_before_overlay_or_writes_while_predecessor_runs(self):
        with TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)),\
             patch.object(owner,'predecessor_complete',side_effect=ValueError('still running')),\
             patch.object(owner,'overlay_module',side_effect=AssertionError('no candidate activation')),\
             patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no child')):
            with self.assertRaisesRegex(ValueError,'still running'):owner.prepare()
            self.assertFalse((Path(directory)/owner.PLAN).exists())



    def plan_fixture(self,root):
        plan=dict(jobs=owner.build_jobs(roster()),source_overlay_lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'diag'},remaining_queue=[])
        owner.fresh(root/owner.PLAN,plan);return plan,owner.sha(root/owner.PLAN)

    def test_execute_cpu_then_one_native_natural_end_no_retries_and_preserves_false(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def child(row,check,status):
                status['active']=dict(job_id=row['job_id'],stage='CPU_check' if check else 'native');calls.append((row['job_id'],check));return {'check':check}
            def complete(row,cpu,native,lock):
                self.assertTrue(cpu['check']);self.assertFalse(native['check']);return dict(job_id=row['job_id'],native_success=False)
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),patch.object(owner,'predecessor_complete',return_value={}),\
                 patch.object(owner,'child_phase',side_effect=child),patch.object(owner,'completed_case',side_effect=complete),\
                 patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no real child')),patch('sys.stdout',new_callable=io.StringIO):
                plan,checksum=self.plan_fixture(root);owner.execute(checksum)
            self.assertEqual(calls,[(r['job_id'],check) for r in plan['jobs'] for check in (True,False)])
            status=json.loads((root/owner.RUNTIME/'status.json').read_text());self.assertEqual(status['stage'],'complete')
            self.assertEqual(len(status['completed']),1);self.assertEqual(status['failed'],[])
            self.assertTrue(all(item['native_success'] is False for item in status['completed']))
            self.assertEqual(json.loads((root/owner.RUNTIME/'terminal.json').read_text())['completed'],1)

    def test_repeat_wrongsha_and_failed_result_review_stop_without_retry(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),patch.object(owner,'predecessor_complete',return_value={}):
                plan,checksum=self.plan_fixture(root)
                with self.assertRaises(ValueError):owner.execute('wrongsha')
                owner.fresh(root/owner.RUNTIME/'launch.json',{'already':'launched'})
                with self.assertRaises(ValueError):owner.execute(checksum)
        with TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def child(row,check,status):
                status['active']=dict(job_id=row['job_id'],stage='CPU_check' if check else 'native');calls.append(check);return {}
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),patch.object(owner,'predecessor_complete',return_value={}),\
                 patch.object(owner,'child_phase',side_effect=child),patch.object(owner,'completed_case',side_effect=ValueError('candidate result incomplete')):
                plan,checksum=self.plan_fixture(root)
                with self.assertRaisesRegex(ValueError,'candidate result'):owner.execute(checksum)
            self.assertEqual(calls,[True,False])
            status=json.loads((root/owner.RUNTIME/'status.json').read_text())
            self.assertEqual(status['completed'],[]);self.assertFalse(status['failed'][0]['native_failure_inferred'])
            self.assertEqual(status['failed'][0]['stage'],'result_review');self.assertFalse((root/owner.RUNTIME/'terminal.json').exists())

    def worker_fixture(self,root):
        row=owner.build_jobs(roster())[0];ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        status=dict(completed=[],failed=[],active=None,stage='running',source_overlay_lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'candidate'})
        value=dict(schema=owner.DIAGNOSTIC_SCHEMA,complete=True,source_overlay_lock=status['source_overlay_lock'],
            formal_score_contribution=False,diagnostic_only=True,
            parent_source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
            selection=dict(task=row['task'],method=row['method'],raw_episode_id=row['raw']),process=dict(pid=9876,start_ticks=123),
            frozen_OS_environment=dict(outer_wrapper_pid=os.getpid(),outer_wrapper_start_ticks=ticks,observer_pid=9876,observer_start_ticks=123),
            execution=dict(mode='CPU_check',complete=True,native_started=False,GPU_started=False,actual_split='train',
                native_execution_ready=True,
                execution_protocol=self.protocol_fixture(root,row,status['source_overlay_lock'])))
        name=row['runtime']+'/CPU_check/worker_sidecar.json';owner.fresh(root/name,value)
        return row,status,value,name

    def test_child_actual_pid_returncode_and_sidecar_namespace_binding(self):
        for bad in (None,'returncode','outerpid','observerpid','factor','started'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);row,status,value,name=self.worker_fixture(root)
                if bad=='outerpid':value['frozen_OS_environment']['outer_wrapper_pid']+=1
                if bad=='observerpid':value['frozen_OS_environment']['observer_start_ticks']+=1
                if bad=='factor':value['execution']['execution_protocol']['inherited_1132_execution_protocol']['changed_factor']='wrong_family_factor'
                if bad=='started':value['execution']['native_started']=True
                (root/name).write_text(json.dumps(value));code=7 if bad=='returncode' else 0
                child=NS(pid=os.getpid(),wait=lambda:code,poll=lambda:code)
                with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SCHEMA=owner.DIAGNOSTIC_SCHEMA)),\
                     patch.object(owner.subprocess,'Popen',return_value=child) as spawn:
                    if bad is None:self.assertEqual(owner.child_phase(row,True,status)['process']['pid'],9876)
                    else:
                        with self.assertRaises(ValueError):owner.child_phase(row,True,status)
                self.assertEqual(spawn.call_count,1)
                terminal=json.loads((root/owner.RUNTIME/'launches'/row['job_id']/'CPU_check/terminal.json').read_text())
                self.assertEqual(terminal['returncode'],code);self.assertEqual(terminal['pid'],os.getpid())

    def test_interrupted_child_exit_remains_unknown_not_normal_terminal(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);row=owner.build_jobs(roster())[0];status=dict(completed=[],failed=[],stage='running')
            def interrupted():raise KeyboardInterrupt('synthetic interruption')
            child=NS(pid=os.getpid(),wait=interrupted,poll=lambda:None)
            with patch.object(owner,'ROOT',root),patch.object(owner.subprocess,'Popen',return_value=child):
                with self.assertRaises(KeyboardInterrupt):owner.child_phase(row,False,status)
            leaf=root/owner.RUNTIME/'launches'/row['job_id']/'native'
            interruption=json.loads((leaf/'owner_interruption.json').read_text())
            self.assertTrue(interruption['child_exit_unknown']);self.assertFalse(interruption['normal_child_terminal_claimed'])
            self.assertFalse((leaf/'terminal.json').exists())

    def test_completed_case_retains_native_false_with_quality_checked_passive_streams(self):
        for bad in (None,'casehash','identity','cleanup','factor','clock','parent_identity'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);row=owner.build_jobs(roster())[0]
                lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'candidate'}
                protocol=self.protocol_fixture(root,row,lock)
                if bad=='factor':protocol['inherited_1132_execution_protocol']['changed_factor']='wrong_factor'
                identity=dict(sha256='identity',inherited_1132_identity_sha256='a'*64,
                    execution_protocol_schema=owner.DIAGNOSTIC_SCHEMA,execution_protocol_sha256=owner.digest(protocol),
                    source_overlay_lock_sha256=lock['sha256'])
                inherited_identity=dict(sha256='a'*64,source_overlay_lock_sha256=owner.PARENT_LOCK_SHA,
                    execution_protocol_schema='r10_full16_fixed_TRAIN_public_phase_overlay_v1132',
                    execution_protocol_sha256=owner.digest(protocol['inherited_1132_execution_protocol']))
                if bad=='parent_identity':inherited_identity['sha256']='b'*64
                case=dict(version=owner.DIAGNOSTIC_SCHEMA+'_case',complete=True,cleanup_completed=bad!='cleanup',split='train',
                    task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],process_id=9876,
                    official_task_success=False,native_terminal_failure=False,source_overlay_lock=lock,formal_score_contribution=False,
                    parent_source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),diagnostic_only=True,
                    original_745_execution_acceptance_granted=False,
                    identity=identity,execution_protocol=protocol,infrastructure_error=None,**self.clock_fixture())
                if bad=='clock':case['counted_clock']['seconds']=.3
                owner.fresh(root/row['out']/'case.json',case)
                owner.fresh(root/row['out']/'owned_hook_receipt.json',dict(actual_split='train',reset=[{'complete':True}],paths=[{'close_complete':True}]))
                with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SCHEMA=owner.DIAGNOSTIC_SCHEMA)):
                    cpu={'execution':dict(identity=identity,execution_protocol=protocol,inherited_1132_identity=inherited_identity)}
                    native=dict(process={'pid':9876},execution=dict(identity={'sha256':'changed'} if bad=='identity' else identity,
                        case=owner.pointer(row['out']+'/case.json'),native_success=False,native_failure=False,diagnostic_events=self.quality_fixture()))
                    if bad=='casehash':native['execution']['case']['sha256']='changed'
                    if bad is None:self.assertIs(owner.completed_case(row,cpu,native,lock)['native_success'],False)
                    else:
                        with self.assertRaises(ValueError):owner.completed_case(row,cpu,native,lock)

    def test_protocol_inherits_exact1132_ring_factor_and_new_diagnostic_worker(self):
        row=owner.build_jobs(roster())[0]
        for bad in (None,'other_factor','worker','rule_source','rule_task','core','parent','diaglock'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'diagnostic'}
                protocol=self.protocol_fixture(root,row,lock);inherited=protocol['inherited_1132_execution_protocol']
                if bad=='other_factor':inherited['changed_factor']='another_factor'
                if bad=='worker':protocol['actual_worker']['sha256']='changed'
                if bad=='rule_source':inherited['public_phase_component_binding']['task_rule']['source_sha256']='changed'
                if bad=='rule_task':inherited['public_phase_component_binding']['task_rule']['task']='another_task'
                if bad=='core':inherited['core_factory']['module']='old1121_without1131'
                if bad=='parent':protocol['parent_source_overlay_lock']['sha256']='changed'
                if bad=='diaglock':protocol['diagnostic_source_overlay_lock']={'sha256':'changed'}
                with patch.object(owner,'ROOT',root):
                    if bad is None:owner.validate_protocol(row,protocol,lock)
                    else:
                        with self.assertRaises(ValueError):owner.validate_protocol(row,protocol,lock)

    def test_original_clock_budget_action_accounting_and_early_terminal_are_preserved(self):
        for bad in (None,'budget','nan','seconds','unpaired','aborted','action_delta','negative','policy_outcome'):
            with self.subTest(bad=bad):
                case=self.clock_fixture()
                if bad=='budget':case['counted_clock']['ticks']=201
                if bad=='nan':case['counted_clock']['seconds']=float('nan')
                if bad=='seconds':case['counted_clock']['seconds']=.3
                if bad=='unpaired':case['native_outcome_sampling']['active_action']={'open':'action'}
                if bad=='aborted':case['native_outcome_sampling']['aborted']=True
                if bad=='action_delta':case['native_outcome_sampling']['completed_actions'][0]['end_tick']=5
                if bad=='negative':case['native_outcome_sampling']['completed_actions'][0]['ticks']=-4
                if bad=='policy_outcome':case['alignment_receipt']['outcomes_are_policy_observations']=True
                if bad is None:self.assertEqual(owner.validate_clock(case)['counted_clock'],case['counted_clock'])
                else:
                    with self.assertRaises(ValueError):owner.validate_clock(case)

    def test_quality_accepts_original_false_readiness_empty_candidates_and_helper_exception(self):
        row=owner.build_jobs(roster())[0]
        value=self.quality_fixture()
        probe=value['readiness_probe']['readiness_probe']
        probe['events'][2]['original_return']=dict(candidate_count=0,first_candidates=[])
        probe['events'][3].pop('original_return')
        probe['events'][3].update(original_returned=False,original_exception=dict(type='ValueError'))
        probe['method_call_counts']['generator.observed_reference'].update(returns=0,exceptions=1)
        trace=value['public_execution_trace']['trace_receipt']
        trace.update(plan_returns=0,plan_exceptions=1,plans=[dict(exception_type='ConfigurationPathError')])
        quality=owner.diagnostic_quality(row,value)
        self.assertTrue(quality['complete'])
        self.assertEqual(quality['original_method_exceptions'],1)
        self.assertEqual(quality['original_plan_exceptions'],1)
        self.assertFalse(quality['native_failure_inferred_from_missing_diagnostics'])

    def test_quality_rejects_empty_probe_or_missing_each_original_method(self):
        row=owner.build_jobs(roster())[0]
        for bad in ('empty','frontend.ready','collision_port.readiness','source_policy.candidates','generator.observed_reference','geometry'):
            with self.subTest(bad=bad):
                value=self.quality_fixture();probe=value['readiness_probe']['readiness_probe']
                if bad=='empty':probe['events']=[]
                elif bad=='geometry':probe.update(geometry_calls=0,geometry_returns=0)
                else:probe['method_call_counts'][bad].update(calls=0,returns=0)
                with self.assertRaisesRegex(ValueError,'diagnostic collection incomplete'):
                    owner.diagnostic_quality(row,value)

    def test_quality_rejects_drops_errors_unrestored_and_unpaired_calls(self):
        row=owner.build_jobs(roster())[0]
        for bad in ('probe_drop','probe_error','probe_restore','outer_restore','finalization',
                    'unpaired','trace_drop','trace_error','trace_restore','trace_trunc'):
            with self.subTest(bad=bad):
                value=self.quality_fixture();probe=value['readiness_probe']['readiness_probe']
                trace=value['public_execution_trace']['trace_receipt']
                if bad=='probe_drop':probe['dropped_events']=1
                if bad=='probe_error':probe['recording_errors']=1
                if bad=='probe_restore':probe['restoration_checks'][0]['restored']=False
                if bad=='outer_restore':value['factory_restored']=False
                if bad=='finalization':value['observer_finalization_errors']=['failed']
                if bad=='unpaired':probe['method_call_counts']['frontend.ready']['returns']=0
                if bad=='trace_drop':trace['frame_dropped']=1
                if bad=='trace_error':trace['errors']=['bad observer']
                if bad=='trace_restore':trace['restored']=False
                if bad=='trace_trunc':trace['plan_value_truncations']=1
                with self.assertRaises(ValueError):owner.diagnostic_quality(row,value)

    def test_quality_rejects_empty_trace_and_counter_only_records(self):
        row=owner.build_jobs(roster())[0]
        for bad in ('frame_calls','frames','plan_calls','plans','counter'):
            with self.subTest(bad=bad):
                value=self.quality_fixture();trace=value['public_execution_trace']['trace_receipt']
                if bad in ('frames','plans'):trace[bad]=[]
                if bad=='frame_calls':trace.update(frame_calls=0,frame_returns=0,frames=[])
                if bad=='plan_calls':trace.update(plan_calls=0,plan_returns=0,plans=[])
                if bad=='counter':trace['plan_calls']=2
                with self.assertRaisesRegex(ValueError,'diagnostic collection incomplete'):
                    owner.diagnostic_quality(row,value)


if __name__=='__main__':unittest.main()
