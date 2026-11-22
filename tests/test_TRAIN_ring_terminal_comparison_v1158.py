"""CPU-only bounded4 owner contracts; all launches are mocks or temp files."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import run_TRAIN_ring_terminal_comparison_v1158_r10 as owner

PROJECT=Path(__file__).resolve().parents[1]


def synthetic_grid():
    tasks={task:raw for task,raw,_ in owner.CANARIES}
    roster=dict(split='train',episodes=[dict(task=task,raw_episode_id=raw) for task,first in tasks.items()
        for raw in (first,'second','third','fourth')])
    routes=[dict(task=task,method=method,GPU_required=False,configuration=dict(path='configs/'+task+'.json',sha256='config'))
        for task in tasks for method in owner.METHODS]
    routes.extend(dict(task='other'+str(i),method='cv_geometry',GPU_required=False,configuration={}) for i in range(48-len(routes)))
    return roster,dict(routes=routes)


def sampler_case(task, *, success=False, fatal=False):
    block=False;outcome=dict(success=success,terminate=success or fatal,terminal_failure=fatal)
    action=dict(label='geometry_execute',start_tick=0,end_tick=4,ticks=4,max_ticks=10,complete=True,sampled_outcome=outcome)
    receipt=dict(schema='r10_original_no_failure_block_action_sampling_v744' if block else 'r10_explicit_author_action_outcome_sampling_v741',
        completed_actions=[action],last_tick=4,active_action=None,aborted=False,global_tick_modulus_used=False)
    alignment=dict(clock=dict(ticks=4,dt=.05,simulation_seconds=.2),native_success_latched=success,native_terminal_failure_latched=fatal)
    if block:
        receipt.update(original_native_success_latch_preserved=True,native_failure_conditions=0,extra_task_success_calls=0)
        action['extra_task_success_calls']=0
    else:
        action.update(action_index=0,current_native_result=outcome)
        receipt.update(aborted_actions=[],success_sampled_each_tick=True,failure_sampled_at_explicit_action_end=True,
            reset_failure_is_diagnostic_only=True,reset_success_semantics_preserved=True,
            sampled_outcome=outcome,initial_native_result=dict(success=False,terminate=False,terminal_failure=False),intermediate_failure_samples=0)
        alignment.update(native_terminal_latched=success or fatal,native_outcome=outcome,native_terminal_tick=4 if success or fatal else None)
    alignment['native_outcome_sampling']=deepcopy(receipt)
    return dict(version=owner.CANDIDATE_SCHEMA+'_case',inherited_consumer_record_version='r10_full16_explicit_author_action_outcome_v744',
        task=task,method='context_geometry',complete=True,cleanup_completed=True,infrastructure_error=None,
        official_task_success=success,native_terminal_failure=fatal,counted_clock=dict(ticks=4,dt=.05,seconds=.2),
        alignment_receipt=alignment,native_outcome_sampling=receipt,author_outcome_protocol=dict(schema='r10_author_action_outcome_protocol_v741'))


class OwnerTests(unittest.TestCase):
    def setUp(self):
        pins=dict(owner.FROZEN_PINS);pins[owner.CORE_SOURCE]='b'*64
        self.pins_patch=patch.object(owner,'FROZEN_PINS',pins);self.pins_patch.start();self.addCleanup(self.pins_patch.stop)

    def jobs(self):return owner.build_jobs(*synthetic_grid())

    def protocol(self,root,row,lock):
        for name in (owner.WORKER,owner.PREPARER):
            path=root/name
            if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU synthetic source')
        return dict(schema=owner.CANDIDATE_SCHEMA,source_overlay_lock=lock,
            actual_worker=dict(path=owner.WORKER,sha256=owner.sha(root/owner.WORKER)),
            actual_preparer=dict(path=owner.PREPARER,sha256=owner.sha(root/owner.PREPARER)),
            execution_branch='core_selected_integration',formal_score_contribution=False,formal_VAL_TEST_qualification=False,
            comparison_parent_source_overlay_lock=owner.COMPARISON_PARENT_LOCK,core_changed_relative_to1154=True,
            block_changed_relative_to1154=False,baseline_changed_relative_to1154=False,
            original_route=dict(author_outcome_protocol=dict(schema='r10_author_action_outcome_protocol_v741')),
            core_factory=dict(module=owner.CORE_MODULE,function='integrated_components',source=owner.CORE_SOURCE,
                sha256=owner.FROZEN_PINS[owner.CORE_SOURCE]),changed_factor=owner.CHANGED_FACTOR,
            public_phase_component_binding=dict(integration=dict(source_sha256=owner.FROZEN_PINS[owner.CORE_SOURCE],
                schema='r10_author_ring_terminal_canonical_goal_integration_v1156'),
                task_rule=dict(task=row['task'],source_sha256=owner.FROZEN_PINS[owner.CORE_SOURCE],changed_factor=owner.CHANGED_FACTOR)))

    def test_default_inert_exact4_unique_cases_and2_parents(self):
        with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no launch')),patch.object(owner,'prepare',side_effect=AssertionError('no prepare')),patch('sys.stdout',new_callable=io.StringIO) as output:owner.main([])
        self.assertFalse(json.loads(output.getvalue())['Native_started']);rows=self.jobs()
        self.assertEqual([(r['task'],r['raw'],r['method']) for r in rows],list(owner.CANARIES))
        self.assertEqual(len(rows),4);self.assertEqual(len({(r['task'],r['raw']) for r in rows}),2)
        self.assertEqual(len({r['out'] for r in rows}),4);self.assertTrue(all('/matched_v1158/' in r['out'] for r in rows))

    def test_real_TRAIN_manifest_and48routes_match_all_fixed_first_parents(self):
        manifest=json.loads((PROJECT/'artifacts/framework/protocol/train_manifest.json').read_text())
        routes=json.loads((PROJECT/'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json').read_text())
        jobs=owner.build_jobs(manifest,routes)
        self.assertEqual([j['raw'] for j in jobs],['episode101','episode101','episode102','episode102'])
        for job in jobs:self.assertEqual(owner.sha(PROJECT/job['config']),job['config_sha256'])

    def test_bad_roster_or_original_route_is_rejected(self):
        for change in ('split','parent','duplicate','route','gpu'):
            roster,routes=synthetic_grid()
            if change=='split':roster['split']='validation'
            if change=='parent':roster['episodes'][0]['raw_episode_id']='not-fixed'
            if change=='duplicate':roster['episodes'][1]=deepcopy(roster['episodes'][0])
            if change=='route':routes['routes'][-1]=deepcopy(routes['routes'][0])
            if change=='gpu':routes['routes'][1]['GPU_required']=True
            with self.subTest(change=change),self.assertRaises(ValueError):owner.build_jobs(roster,routes)

    def test_command_is_original820_local_sim1157_and_check_only_adds_flag(self):
        row=self.jobs()[0];args=owner.command(row)
        self.assertIn(owner.WORKER,args);self.assertIn(owner.WRAPPER,args);self.assertIn(owner.SIM,args)
        self.assertEqual(owner.command(row,True),args+['--check']);self.assertEqual(args[args.index('--overlay-lock')+1],owner.SOURCE_LOCK)
        self.assertNotIn('--prepare',args);self.assertNotIn('--manifest',args)

    def predecessor(self,root,bad=None):
        plan=dict(jobs=[dict(task=t,raw=r,method=m) for t,r,m in owner.PREDECESSOR_CASES],source_overlay_lock={})
        owner.fresh(root/owner.PREDECESSOR_PLAN,plan)
        path=root/owner.PREDECESSOR_SOURCE;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU source')
        plan_sha=owner.sha(root/owner.PREDECESSOR_PLAN);source_sha=owner.sha(path)
        ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])+(0 if bad=='live' else 1)
        launch=dict(pid=os.getpid(),start_ticks=ticks,plan=dict(path=owner.PREDECESSOR_PLAN,sha256=plan_sha))
        def ref(label):
            name=owner.PREDECESSOR+'/evidence/'+label+'.json';owner.fresh(root/name,{'retained':label})
            return dict(path=name,sha256=owner.sha(root/name))
        rows=[]
        for i,(task,raw,method) in enumerate(owner.PREDECESSOR_CASES):
            row=dict(task=task,raw=raw,method=method,native_success=False,case=ref(str(i)+'case'),owned_hooks=ref(str(i)+'hooks'),
                CPU_check={k:ref(str(i)+'CPU_'+k) for k in ('launch','terminal','sidecar')},
                native={k:ref(str(i)+'native_'+k) for k in ('launch','terminal','sidecar')})
            rows.append(row)
        status=dict(stage='complete',active=None,failed=[],completed=rows)
        terminal=dict(complete=True,completed=3,infrastructure_failed=0,automatic_retry=False)
        owner.fresh(root/(owner.PREDECESSOR+'/launch.json'),launch);owner.fresh(root/(owner.PREDECESSOR+'/terminal.json'),terminal)
        completion=dict(schema=owner.EXIT_SCHEMA,actual_tool_exit_observed=True,returncode=0,
            unified_exec_session=owner.PREDECESSOR_EXEC_SESSION,owner_pid=os.getpid(),owner_start_ticks=ticks,owner_alive=False,
            natural_completion=True,native_cases_complete=3,completed_case_infrastructure_failures=0,plan=launch['plan'],
            terminal=dict(path=owner.PREDECESSOR+'/terminal.json',sha256=owner.sha(root/(owner.PREDECESSOR+'/terminal.json'))))
        if bad=='running':status['stage']='running'
        if bad=='failed':status['failed']=[{}]
        if bad=='active':status['active']={}
        if bad=='tool':completion['actual_tool_exit_observed']=False
        if bad=='rc':completion['returncode']=1
        if bad=='boolrc':completion['returncode']=False
        if bad=='session':completion['unified_exec_session']+=1
        if bad=='partial':status['completed'].pop()
        if bad=='case':rows[0]['case']['sha256']='wrong'
        owner.fresh(root/(owner.PREDECESSOR+'/status.json'),status)
        if bad!='missing':owner.fresh(root/owner.PREDECESSOR_COMPLETION,completion)
        return plan_sha,source_sha

    def test_predecessor_requires_actual_natural_exit0_dead_identity_and_all_evidence(self):
        for bad in (None,'live','running','failed','active','tool','rc','boolrc','session','partial','case','missing'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);plan_sha,source_sha=self.predecessor(root,bad)
                with patch.object(owner,'ROOT',root),patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha),patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha),patch.object(owner,'predecessor_record_quality',return_value=None):
                    if bad is None:self.assertTrue(owner.predecessor_complete()['natural_completion'])
                    else:
                        with self.assertRaises((ValueError,FileNotFoundError)):owner.predecessor_complete()

    def test_prepare_blocks_before_overlay_or_writes_if_predecessor_unfinished(self):
        with TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)),patch.object(owner,'predecessor_complete',side_effect=ValueError('running')),patch.object(owner,'overlay_module',side_effect=AssertionError('not ready')):
            with self.assertRaises(ValueError):owner.prepare()
            self.assertFalse((Path(directory)/owner.PLAN).exists())

    def test_pending_pins_refuse_before_reading_sources(self):
        with patch.object(owner,'SOURCE_LOCK_SHA',None),patch.object(owner,'sha',side_effect=AssertionError('pending must be inert')):
            with self.assertRaisesRegex(ValueError,'pending1156/1157'):owner.overlay_module()

    def test_predecessor_incomplete_diagnostic_quality_blocks_before_new_prepare(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);plan_sha,source_sha=self.predecessor(root)
            with patch.object(owner,'ROOT',root),patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha),\
                 patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha),\
                 patch.object(owner,'predecessor_record_quality',side_effect=ValueError('missing leaf or zero trace')),\
                 patch.object(owner,'overlay_module',side_effect=AssertionError('must not reach new overlay')):
                with self.assertRaisesRegex(ValueError,'missing leaf'):owner.prepare()
            self.assertFalse((root/owner.PLAN).exists())

    def test_predecessor_readonly_verifier_is_called_once_and_changed_row_rejects(self):
        from scripts import run_TRAIN_block_preclose_comparison_v1155_r10 as previous
        row=dict(task=owner.PREDECESSOR_CASES[0][0],raw='episode102',method='cv_geometry',CPU_check={},native={})
        job=dict(row);lock={'source':'synthetic'}
        with patch.object(previous,'completed_case',return_value=row) as observed:
            owner.predecessor_record_quality(row,[job],lock)
            observed.assert_called_once_with(job,row['CPU_check'],row['native'],lock)
        with patch.object(previous,'completed_case',return_value=dict(row,diagnostic_quality=False)):
            with self.assertRaisesRegex(ValueError,'integrity receipt changed'):owner.predecessor_record_quality(row,[job],lock)

    def test_root_review_binds_only_exact4_sources_lock_and_actual_completion(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);lock=dict(path=owner.SOURCE_LOCK,sha256='lock');pred=dict(actual_exec_completion=dict(path=owner.PREDECESSOR_COMPLETION,sha256='exit'))
            review=dict(schema=owner.REVIEW_SCHEMA,review_complete=True,release_bounded_four=True,reviewer='root',fixed_TRAIN_sources=owner.FIXED_SOURCES,
                source_overlay_lock=lock,exec_completion=pred['actual_exec_completion'],development_regression_only=True,formal192_readiness=False,changed_factor=owner.CHANGED_FACTOR)
            owner.fresh(root/owner.ROOT_REVIEW,review)
            with patch.object(owner,'ROOT',root):self.assertEqual(owner.root_review(lock,pred)['path'],owner.ROOT_REVIEW)
            review['fixed_TRAIN_sources']=list(reversed(owner.FIXED_SOURCES));(root/owner.ROOT_REVIEW).write_text(json.dumps(review))
            with patch.object(owner,'ROOT',root),self.assertRaises(ValueError):owner.root_review(lock,pred)

    def test_core741_sampler_accepts_false_success_and_fatal(self):
        for task in owner.RINGS:
            for success,fatal in ((False,False),(True,False),*(([(False,True)]) if True else [])):
                case=sampler_case(task,success=success,fatal=fatal);before=deepcopy(case)
                result=owner.validate_clock_and_sampler(case);self.assertEqual(result['completed_actions'],1);self.assertEqual(case,before)
                self.assertIn('native_outcome',case['alignment_receipt'])

    def test_actual_old_TRAIN_ring_shapes_pass_new_header_readonly_projection(self):
        for suffix in ('insert_onto_rotating_peg/episode101/cv_geometry','insert_onto_rotating_peg_high_speed/episode102/context_geometry'):
            name=PROJECT/'artifacts/framework/multitask_r10/author_outcome_qualification_v745/development_demo_only'/suffix/'case.json'
            case=json.loads(name.read_text());case['inherited_consumer_record_version']=case['version'];case['version']=owner.CANDIDATE_SCHEMA+'_case'
            before=deepcopy(case);quality=owner.validate_clock_and_sampler(case)
            self.assertEqual(quality['sampler_schema'],'r10_explicit_author_action_outcome_sampling_v741');self.assertEqual(case,before)


    def test_sampler_rejects_uncovered_ticks_extra_block_calls_fatal_and_false_headers(self):
        for bad in ('clock','actions','failure_or','version','inherited'):
            case=sampler_case(owner.CANARIES[0][0])
            if bad=='clock':case['counted_clock']['seconds']=.3
            if bad=='actions':case['native_outcome_sampling']['completed_actions'][0]['start_tick']=1
            if bad=='failure_or':case['native_outcome_sampling']['global_tick_modulus_used']=True
            if bad=='version':case['version']='old-case'
            if bad=='inherited':case['inherited_consumer_record_version']='unknown'
            with self.subTest(bad=bad),self.assertRaises(ValueError):owner.validate_clock_and_sampler(case)

    def test_protocol_checks_exact_new_ring_terminal_binding_and_factor(self):
        with TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
            root=Path(directory);lock={'sha256':'lock'}
            for row in self.jobs():
                proof=self.protocol(root,row,lock);owner.validate_protocol(row,proof,lock)
                proof['execution_branch']='incorrect'
                with self.assertRaises(ValueError):owner.validate_protocol(row,proof,lock)

    def completed_fixture(self,root,row,bad=None,*,success=False,fatal=False):
        lock={'path':owner.SOURCE_LOCK,'sha256':'lock'};protocol=self.protocol(root,row,lock);identity={'sha256':'case-identity'}
        case=sampler_case(row['task'],success=success,fatal=fatal);case.update(method=row['method'],split='train',source_raw_episode_id=row['raw'],process_id=12345,source_overlay_lock=lock,
            formal_score_contribution=False,original_745_execution_acceptance_granted=False,identity=identity,execution_protocol=protocol)
        if bad=='cleanup':case['cleanup_completed']=False
        owner.fresh(root/row['out']/'case.json',case)
        hooks=dict(actual_split='train',paths=[],reset=[{'complete':True}],shared_development_hooks_applied=True)
        if bad=='hooks':hooks['reset']=[]
        owner.fresh(root/row['out']/'owned_hook_receipt.json',hooks)
        CPU=dict(execution=dict(identity=identity,execution_protocol=protocol));Native=dict(process=dict(pid=12345),execution=dict(identity=identity,
            case=dict(path=row['out']+'/case.json',sha256=owner.sha(root/row['out']/'case.json')),native_success=success,native_failure=fatal))
        if bad=='identity':Native['execution']['identity']={'sha256':'wrong'}
        refs=[]
        for stage,value in (('CPU_check',CPU),('native',Native)):
            path=row['runtime']+'/'+stage+'/worker_sidecar.json';owner.fresh(root/path,value)
            phase_refs=dict(sidecar=dict(path=path,sha256=owner.sha(root/path)))
            for key in ('launch','terminal'):
                name=row['runtime']+'/'+stage+'/'+key+'.json';owner.fresh(root/name,{'synthetic_evidence':key})
                phase_refs[key]=dict(path=name,sha256=owner.sha(root/name))
            refs.append(phase_refs)
        return refs[0],refs[1],lock

    def test_completed_ring_false_and_fatal_cases_keep_compact_pointer_only_status(self):
        for row in (self.jobs()[0],self.jobs()[2]):
            for success,fatal in ((False,False),(True,False),(False,True)):
                with self.subTest(success=success,fatal=fatal),TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
                    root=Path(directory);cpu,native,lock=self.completed_fixture(root,row,success=success,fatal=fatal)
                    result=owner.completed_case(row,cpu,native,lock)
                    self.assertIs(result['native_success'],success);self.assertIs(result['native_terminal_failure'],fatal)
                    self.assertNotIn('execution',json.dumps(result));self.assertEqual(set(result['CPU_check']),{'sidecar','launch','terminal'})
                    self.assertNotIn('identity',result);self.assertNotIn('execution_protocol',result)

    def test_completed_case_rejects_identity_cleanup_and_wrong_reset_interface(self):
        for row in (self.jobs()[0],self.jobs()[2]):
            for bad in ('identity','cleanup','hooks'):
                with self.subTest(kind=row['kind'],bad=bad),TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
                    cpu,native,lock=self.completed_fixture(Path(directory),row,bad)
                    with self.assertRaises(ValueError):owner.completed_case(row,cpu,native,lock)

    def test_CPU_or_native_launch_terminal_mutation_rejects_before_completed_status(self):
        for phase,key in [('CPU_check','launch'),('CPU_check','terminal'),('native','launch'),('native','terminal')]:
            with self.subTest(phase=phase,key=key),TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
                root=Path(directory);row=self.jobs()[0];cpu,native,lock=self.completed_fixture(root,row)
                ref=(cpu if phase=='CPU_check' else native)[key]
                (root/ref['path']).write_text('{"changed_after_phase":true}')
                with self.assertRaisesRegex(ValueError,'evidence changed'):owner.completed_case(row,cpu,native,lock)

    def test_execute_is_serial4_cpu_before_native_and_retains_all_false(self):
        with TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
            root=Path(directory);jobs=self.jobs();plan=dict(jobs=jobs,source_overlay_lock={'sha256':'lock'});owner.fresh(root/owner.PLAN,plan);calls=[]
            def child(row,check,status):calls.append((row['job_id'],check));status['active']={'job_id':row['job_id'],'stage':'CPU_check' if check else 'native'};return {'pointers':'only'}
            with patch.object(owner,'validate',return_value={}),patch.object(owner,'predecessor_complete',return_value={}),patch.object(owner,'child_phase',side_effect=child),patch.object(owner,'completed_case',side_effect=lambda row,*args:dict(job_id=row['job_id'],native_success=False)),patch('sys.stdout',new_callable=io.StringIO):
                owner.execute(owner.sha(root/owner.PLAN))
                with self.assertRaises(ValueError):owner.execute(owner.sha(root/owner.PLAN))
            self.assertEqual(calls,[(r['job_id'],c) for r in jobs for c in (True,False)])
            status=json.loads((root/owner.RUNTIME/'status.json').read_text());self.assertEqual(status['stage'],'complete');self.assertEqual(len(status['completed']),4)
            self.assertEqual(json.loads((root/owner.RUNTIME/'terminal.json').read_text())['completed'],4)

    def test_stop_after_CPU_and_infrastructure_failure_stop_without_retry(self):
        for mode in ('stop','infra'):
            with self.subTest(mode=mode),TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
                root=Path(directory);jobs=self.jobs();owner.fresh(root/owner.PLAN,dict(jobs=jobs,source_overlay_lock={}));calls=[]
                def child(row,check,status):
                    calls.append(check);status['active']={'job_id':row['job_id'],'stage':'CPU_check'}
                    if mode=='infra':raise RuntimeError('original infrastructure')
                    (root/owner.RUNTIME/'STOP').touch();return {}
                with patch.object(owner,'validate',return_value={}),patch.object(owner,'predecessor_complete',return_value={}),patch.object(owner,'child_phase',side_effect=child):
                    with self.assertRaises((InterruptedError,RuntimeError)):owner.execute(owner.sha(root/owner.PLAN))
                self.assertEqual(calls,[True]);status=json.loads((root/owner.RUNTIME/'status.json').read_text());self.assertEqual(status['stage'],'failed')
                self.assertFalse(status['failed'][0]['native_failure_inferred']);self.assertFalse((root/owner.RUNTIME/'terminal.json').exists())

    def test_child_nonzero_or_interrupted_return_is_preserved_and_not_retried(self):
        for mode in ('nonzero','interrupted'):
            with self.subTest(mode=mode),TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
                root=Path(directory);row=self.jobs()[0];status={'source_overlay_lock':{}}
                def wait():
                    if mode=='interrupted':raise KeyboardInterrupt('original interruption')
                    return 7
                child=NS(pid=os.getpid(),wait=wait,poll=lambda:None)
                with patch.object(owner.subprocess,'Popen',return_value=child) as launch:
                    with self.assertRaises((ValueError,KeyboardInterrupt)):owner.child_phase(row,True,status)
                self.assertEqual(launch.call_count,1);leaf=root/owner.RUNTIME/'launches'/row['job_id']/'CPU_check'
                if mode=='nonzero':self.assertEqual(json.loads((leaf/'terminal.json').read_text())['returncode'],7)
                else:
                    self.assertTrue(json.loads((leaf/'owner_interruption.json').read_text())['child_exit_unknown']);self.assertFalse((leaf/'terminal.json').exists())

    def test_child_actual_PID_namespace_selection_protocol_and_compact_sidecar_pointers(self):
        for row in self.jobs():
            for bad in (None,'namespace','outer','selection','lock','protocol'):
                with self.subTest(kind=row['kind'],bad=bad),TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)):
                    root=Path(directory);lock={'path':owner.SOURCE_LOCK,'sha256':'lock'};protocol=self.protocol(root,row,lock)
                    ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]);worker_pid=os.getpid()+1000
                    proof=dict(complete=True,outer_wrapper_pid=os.getpid(),outer_wrapper_start_ticks=ticks,observer_pid=worker_pid,observer_start_ticks=42)
                    if bad=='namespace':proof['observer_pid']+=1
                    if bad=='outer':proof['outer_wrapper_start_ticks']+=1
                    selected=dict(task=row['task'],raw_episode_id=row['raw'],method=row['method'])
                    if bad=='selection':selected['raw_episode_id']='wrong'
                    if bad=='protocol':protocol['execution_branch']='wrong'
                    sidecar=dict(schema=owner.CANDIDATE_SCHEMA,complete=True,source_overlay_lock=lock if bad!='lock' else {},formal_score_contribution=False,
                        selection=selected,process=dict(pid=worker_pid,start_ticks=42),frozen_OS_environment=proof,
                        execution=dict(mode='CPU_check',complete=True,native_started=False,GPU_started=False,actual_split='train',native_execution_ready=True,
                            execution_protocol=protocol))
                    def wait():
                        owner.fresh(root/row['runtime']/'CPU_check/worker_sidecar.json',sidecar);return 0
                    child=NS(pid=os.getpid(),wait=wait,poll=lambda:0)
                    with patch.object(owner.subprocess,'Popen',return_value=child) as launch:
                        if bad is None:
                            result=owner.child_phase(row,True,{'source_overlay_lock':lock})
                            self.assertEqual(set(result),{'launch','terminal','sidecar'});self.assertNotIn('execution',result)
                            for reference in result.values():self.assertEqual(owner.sha(root/reference['path']),reference['sha256'])
                        else:
                            with self.assertRaises(ValueError):owner.child_phase(row,True,{'source_overlay_lock':lock})
                    self.assertEqual(launch.call_count,1)


if __name__=='__main__':unittest.main()
