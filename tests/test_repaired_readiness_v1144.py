"""Pure CPU synthetic contracts; no real candidate, dataset or simulation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from our_mani.multitask import repaired_readiness_v1144 as subject
from our_mani.multitask import repaired_policy_graph_v1143 as graphs
from tests.test_repaired_policy_graph_v1143 import graph_spec, put


def write(root, name, value):
    return dict(root='project', **put(root,name,json.dumps(value,ensure_ascii=False,
        sort_keys=True,allow_nan=False).encode()))


def protocol(implementation=None, consumer=None):
    pointer = dict(root='project',path='src/consumer.py',sha256='a'*64)
    return dict(schema='r10_author_action_outcome_protocol_v741',
        implementation=implementation or pointer,runtime_consumer=consumer or pointer,original_runtime=consumer or pointer,
        failure_sampling='current_native_result_at_explicit_action_end',success_sampling='original_per_tick_success_latch',
        pose_action_max_ticks=10,public_hold_max_ticks=1,budget_partial_action_consumed=True,
        native_RML_parameters_unchanged=True,block_native_fail_conditions=0,block_extra_task_success_calls=0)


def outcome(success=False, fatal=False):
    return dict(success=success,terminate=success or fatal,terminal_failure=fatal)


def sampled_case(*, block=False, method='cv_geometry', success=False, fatal=False, ticks=1):
    task=subject.BLOCKS[0] if block else next(t for t in subject.TASKS if t not in subject.BLOCKS)
    final=outcome(success,fatal)
    rows=[];cursor=0
    while cursor<ticks:
        end=min(cursor+10,ticks);row=dict(complete=True,label='baseline_execute' if method=='physmani' else 'geometry_execute',
            start_tick=cursor,end_tick=end,ticks=end-cursor,max_ticks=10,
            sampled_outcome=final if end==ticks else outcome())
        if block:row['extra_task_success_calls']=0
        else:row.update(action_index=len(rows),current_native_result=deepcopy(row['sampled_outcome']))
        rows.append(row);cursor=end
    receipt=dict(schema=subject.BLOCK_SCHEMA if block else subject.CORE_SCHEMA,
        active_action=None,aborted=False,aborted_actions=[],last_tick=ticks,
        global_tick_modulus_used=False,completed_actions=rows)
    if block:receipt.update(original_native_success_latch_preserved=True,native_failure_conditions=0,extra_task_success_calls=0)
    else:receipt.update(sampled_outcome=final,initial_native_result=outcome(success and ticks==0),
        intermediate_failure_samples=0,success_sampled_each_tick=True,failure_sampled_at_explicit_action_end=True,
        reset_failure_is_diagnostic_only=True,reset_success_semantics_preserved=True)
    alignment=dict(clock=dict(ticks=ticks,dt=.05,simulation_seconds=ticks*.05,budget_seconds=10.),
        actual_physics_seconds=ticks*.05,actual_task_seconds=ticks*.05,
        native_outcome_sampling=deepcopy(receipt),native_success_latched=success,
        native_terminal_failure_latched=fatal,outcomes_are_policy_observations=False)
    if not block:alignment.update(native_terminal_latched=success or fatal,native_outcome=final,
        native_terminal_tick=ticks if success or fatal else None)
    return dict(version='new_actual_worker_case',complete=True,cleanup_completed=True,infrastructure_error=None,
        task=task,method=method,official_task_success=success,native_terminal_failure=fatal,
        counted_clock=dict(ticks=ticks,dt=.05,seconds=ticks*.05),native_outcome_sampling=receipt,
        alignment_receipt=alignment,author_outcome_protocol=protocol())


class FreshFixture:
    """192 simulated JSON receipts, deliberately all NativeFalse/motionFalse.

    These isolated dummy source files are test evidence only. The validator
    never constructs a candidate or claims this fixture ran a real robot.
    """
    def __init__(self,root,mutate=None):
        self.root=Path(root);spec=graph_spec(root)
        auditor=put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
        spec['control']['sources_map']['train_auditor']=auditor
        spec['control']['components']['train_auditor']=auditor
        for name in ('scripts/run_aligned_gpu.sh','scripts/run_local_sim.sh'):
            spec['control']['sources_map'][name]=put(root,name)
        for index,name in enumerate((*subject.RML_LOCAL_SOURCES,*subject.RML_NATIVE_SOURCES)):
            spec['physical']['sources_map']['common_RML_'+str(index)]=put(root,name)
        physical=spec['physical']
        source_map={p['path']:p['sha256'] for p in physical['sources_map'].values()}
        self.profile=dict(schema='r10_common_author_RML_profile_v530',methods=list(subject.METHODS),
            parameters=deepcopy(subject.PARAMETERS),user_authorization=subject.RML_AUTHORIZATION,
            original_waypoint_filter_and_RML_algorithm=True,motion_metrics_required_for_all_methods=True,
            measured_acceleration6_is_diagnostic_only=True,native_task_success_unchanged=True,
            sources_sha256={p:source_map[p] for p in subject.RML_LOCAL_SOURCES},
            native_sources_sha256={p:source_map[p] for p in subject.RML_NATIVE_SOURCES})
        physical['common_RML']['profile']=write(root,'metadata/common_profile.json',self.profile)
        self.protocol=protocol(physical['outcome_sampling']['source'],physical['sources_map']['consumer'])
        physical['outcome_sampling']['protocol']=write(root,'metadata/outcome_protocol.json',self.protocol)
        episodes=[dict(task=t,raw_episode_id='episode'+str(i),path='never-opened/'+t+'/'+str(i)+'.h5')
            for t in subject.TASKS for i in range(100)]
        spec['train_source_manifest']=write(root,'metadata/original_TRAIN_manifest.json',dict(split='train',episodes=episodes))
        self.graph_ref=graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/synthetic_graph')
        self.graph=graphs.read_policy_graph(root,self.graph_ref)
        self.stamp=dict(graph=self.graph_ref,**{k:self.graph[k] for k in subject.HASHES})
        self.jobs=[dict(task=t,raw_episode_id='episode'+str(i),method=m,
            out='artifacts/framework/multitask_r10/synthetic_cases/'+t+'/'+str(i)+'/'+m,
            runtime='runtime/framework/synthetic_cases/'+t+'/'+str(i)+'/'+m)
            for t in subject.TASKS for i in range(4) for m in subject.METHODS]
        self.plan=dict(schema=subject.PLAN_SCHEMA,jobs=self.jobs,automatic_retry=False,
            source_outcome_filtering=False,reused_cases=[],**self.stamp)
        self.plan_ref=write(root,'runtime/framework/synthetic_owner/plan.json',self.plan)
        self.records=[];route_map={(r['task'],r['method']):r for r in self.graph['routes']}
        for index,job in enumerate(self.jobs):
            route=route_map[(job['task'],job['method'])]
            source=next(r for r in episodes if (r['task'],r['raw_episode_id'])==(job['task'],job['raw_episode_id']))
            witness=subject.make_graph_witness(self.graph_ref,self.graph,route,source,split='train')
            case=sampled_case(block=job['task'] in subject.BLOCKS,method=job['method'])
            case.update(task=job['task'],split='train',source_raw_episode_id=job['raw_episode_id'],
                process_id=100000+2*index+1,automatic_retry=False,TEST_used_for_fitting=False,graph_witness=witness,
                component_domain=dict(schema='r10_source_bound_RML_component_purpose_v640',
                    purpose='train_qualification',split='train',VAL_TEST_relabelled_as_TRAIN=False,policy_sha256=None,
                    task=job['task'],method=job['method'],raw_episode_id=job['raw_episode_id']),
                author_outcome_protocol=deepcopy(self.protocol),
                case_isolation='fresh_simulator_process_per_episode',formal_test_used=False,original_source_motion_replayed=False,
                raw_episode_id=job['raw_episode_id']+'__policy_aug_r0',parent_raw_episode_id=job['raw_episode_id'],
                recorded_initial_conditions=dict(metric_only=True,policy_use_allowed=False),
                all_stage_counted_joint_motion=dict(schema='r10_original_baseline_measured6_independent_of_unused_plan_model_v345',
                    events=[dict(tick_before=0,tick_after=1,phase='hold',actual_issued_at=0.,reference_target_at=.05,
                        measured_q_before=[0.]*7,measured_q_after=[0.]*7,backward_window_velocity=[0.]*7,
                        backward_window_acceleration=None,position_passed=False,speed_passed=True,acceleration_passed=None,
                        diagnostic_only=True,native_parameters_changed=False)],violations=[1],
                    all_stage_actual_window_checks_passed=False,peak_actual_window_acceleration=0.,
                    extra_physics_ticks=0,extra_camera_queries=0,policy_or_physics_modified=False,
                    original_planning_model_acceleration_unchanged=True,planning_metadata_used_as_actual_acceleration_cap=False,
                    actual_window_acceleration_criterion_rad_s2=6.),
                common_author_RML_parameters=dict(schema='r10_observed_common_author_RML_parameters_v530',
                    complete=True,profile_sha256=subject.digest(self.profile),actual_parameters=deepcopy(subject.PARAMETERS),
                    native_parameters_mutated=False,actual_motion_bound_claimed=False))
            case['alignment_receipt']['all_stage_counted_joint_motion_v191']=deepcopy(case['all_stage_counted_joint_motion'])
            execution_protocol=dict(schema='new_actual_worker_protocol',actual_worker=route['train']['worker'],
                actual_preparer=route['train']['preparer'],inherited_consumer=route['train']['consumer'],
                source_overlay_lock=self.graph['physical']['source_overlay_lock'],selected_physical_factory=route['physical_factory'],
                factory_binding=witness['factory_binding'],execution_branch=witness['model_branch'],actual_split='train',
                component_purpose='train_qualification',automatic_retry=False,original_745_execution_acceptance_granted=False,
                qualification_granted_by_worker=False,formal_score_contribution=False,
                additional_native_getter_observe_predict_or_success_calls=0,unchanged_action_outcome_sampling=self.protocol,**self.stamp)
            case.update(execution_protocol=execution_protocol,source_overlay_lock=self.graph['physical']['source_overlay_lock'],
                original_745_execution_acceptance_granted=False,qualification_granted_by_worker=False,formal_score_contribution=False)
            case['identity']=subject.bind_graph_identity(dict(version='r10_fresh_case_identity_v1',split='train',
                task=job['task'],method=job['method'],raw_episode_id=case['raw_episode_id'],parent_raw_episode_id=job['raw_episode_id'],
                source_raw_episode_id=job['raw_episode_id'],replica=0,exploration_seed=None,demo_only=True,
                development_demo_only=True,fitting_source=False,formal_validation_metric=False,
                execution_protocol_schema=execution_protocol['schema'],execution_protocol_sha256=subject.digest(execution_protocol),
                config_sha256=route['configuration']['sha256'],frozen_inputs_sha256=route['frozen_inputs']['sha256'],
                source_row_sha256=subject.digest(source)),witness)
            if mutate:mutate('case',index,case)
            case_ref=write(root,job['out']+'/case.json',case)
            hooks=dict(actual_split='train',reset=[] if job['task'] in subject.BLOCKS else [dict(complete=True)],
                paths=[dict(close_complete=True)])
            if mutate:mutate('hooks',index,hooks)
            row=dict(task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],attempt=1,case=case_ref,
                owned_hooks=write(root,job['out']+'/owned_hook_receipt.json',hooks))
            for check in (True,False):
                phase='CPU_check' if check else 'native';base=job['runtime']+'/'+phase
                inner=dict(pid=100000+2*index+int(not check),start_ticks=40000+2*index+int(not check))
                outer=dict(pid=200000+2*index+int(not check),start_ticks=50000+2*index+int(not check))
                launcher='scripts/run_aligned_gpu.sh' if route['GPU_required'] else 'scripts/run_local_sim.sh'
                argv=['python',self.graph['control']['components']['environment_wrapper']['path'],
                    '--manifest-sha256',self.graph['physical']['environment_manifest']['sha256'],
                    '--','bash',launcher,route['train']['worker']['path'],'--graph',self.graph_ref['path'],
                    '--task',job['task'],'--raw-episode-id',job['raw_episode_id'],'--method',job['method'],
                    '--split','train','--out',job['out'],'--runtime',job['runtime']]+(['--check'] if check else [])
                launch=dict(process=outer,actual_worker=route['train']['worker'],argv=argv,**self.stamp)
                terminal=dict(process=outer,returncode=0,automatic_retry=False)
                execution=dict(complete=True,identity=deepcopy(case['identity']),graph_witness=deepcopy(witness),
                    component_domain=deepcopy(case['component_domain']),
                    execution_protocol=deepcopy(case['execution_protocol']),
                    mode='CPU_check' if check else 'graph_bound_native_TRAIN',actual_split='train',
                    native_started=not check,GPU_started=False,native_false_preserved=True,
                    runtime_scope_restoration=dict(complete=True,restored=True),
                    case=case_ref,native_success=case['official_task_success'],native_failure=case['native_terminal_failure'])
                sidecar=dict(complete=True,process=inner,graph_witness=deepcopy(witness),execution=execution,
                    actual_worker=route['train']['worker'],actual_preparer=route['train']['preparer'],**self.stamp,
                    frozen_OS_environment=dict(complete=True,outer_wrapper_pid=outer['pid'],outer_wrapper_start_ticks=outer['start_ticks'],
                        observer_pid=inner['pid'],observer_start_ticks=inner['start_ticks']))
                if mutate:
                    mutate(phase+'_launch',index,launch);mutate(phase+'_terminal',index,terminal);mutate(phase,index,sidecar)
                row[phase]=dict(launch=write(root,base+'/launch.json',launch),terminal=write(root,base+'/terminal.json',terminal),
                    sidecar=write(root,base+'/worker_sidecar.json',sidecar))
            if mutate:mutate('record',index,row)
            self.records.append(row)
        self.owner_process=dict(pid=2147483000,start_ticks=70000)
        owner_launch=dict(source=self.graph['control']['components']['train_owner'],process=self.owner_process,
            plan=self.plan_ref,**self.stamp)
        self.owner_terminal=dict(complete=True,completed=192,attempted=192,infrastructure_failed=0,
            automatic_retry=False,active=None)
        self.owner_launch_ref=write(root,'runtime/framework/synthetic_owner/launch.json',owner_launch)
        self.owner_terminal_ref=write(root,'runtime/framework/synthetic_owner/terminal.json',self.owner_terminal)
        self.owner_completion=dict(process=self.owner_process,actual_tool_exit_observed=True,natural_completion=True,
            returncode=0,plan=self.plan_ref,terminal=self.owner_terminal_ref)
        self.cohort=dict(schema=subject.COHORT_SCHEMA,complete=True,automatic_retry=False,source_outcome_filtering=False,
            reused_cases=[],plan=self.plan_ref,records=self.records,**self.stamp,
            owner=dict(launch=self.owner_launch_ref,terminal=self.owner_terminal_ref,
                exec_completion=write(root,'runtime/framework/synthetic_owner/exec_completion.json',self.owner_completion)))
        self.refresh()

    def refresh(self):
        self.cohort_ref=write(self.root,'runtime/framework/synthetic_owner/cohort.json',self.cohort)

    def audit(self):return subject.audit_fresh_train192(self.root,self.graph_ref,self.cohort_ref)


class SamplingContracts(unittest.TestCase):
    def validate(self,case):
        return subject.validate_sampling_case(case,case['author_outcome_protocol'],
            block=case['task'] in subject.BLOCKS,case_version='new_actual_worker_case')

    def test_all_native_false_success_and_fatal_are_legal_measured_outcomes(self):
        for block in (False,True):
            for method in subject.METHODS:
                for success,fatal in ((False,False),(True,False),(False,True)):
                    if block and fatal:continue
                    with self.subTest(block=block,method=method,success=success,fatal=fatal):
                        case=sampled_case(block=block,method=method,success=success,fatal=fatal,ticks=17)
                        self.assertEqual(len(self.validate(case)['completed_actions']),2)

    def test_zero_initial_success_and_zero_task_failure_remain_legal(self):
        for block in (False,True):
            for success in (False,True):self.validate(sampled_case(block=block,success=success,ticks=0))

    def test_block_inherited_observe_success_calls_are_not_forbidden_new_sampler_calls(self):
        case=sampled_case(block=True);case['inherited_observe_success_calls']=2
        self.validate(case)
        case['native_outcome_sampling']['extra_task_success_calls']=1
        case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])
        with self.assertRaises(subject.ReadinessError):self.validate(case)

    def test_old_version_nan_bool_tick_and_clock_spoof_rejected(self):
        for field,value in (('version','r10_full16_explicit_author_action_outcome_v744'),
                ('official_task_success',1),('complete',1),('cleanup_completed',False)):
            case=sampled_case();case[field]=value
            with self.subTest(field=field),self.assertRaises(subject.ReadinessError):self.validate(case)
        for field,value in (('ticks',True),('ticks',201),('seconds',float('nan')),('dt',True),('seconds',.1)):
            case=sampled_case();case['counted_clock'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(subject.ReadinessError):self.validate(case)

    def test_action_gaps_overlap_hold_limit_abort_and_failure_OR_rejected(self):
        changes=(('start_tick',1),('end_tick',2),('ticks',True),('max_ticks',True),('label','public_hold'))
        for field,value in changes:
            case=sampled_case();case['native_outcome_sampling']['completed_actions'][0][field]=value
            case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])
            with self.subTest(field=field),self.assertRaises(subject.ReadinessError):self.validate(case)
        case=sampled_case(fatal=True);case['native_outcome_sampling']['completed_actions'][0]['current_native_result']=outcome()
        case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])
        with self.assertRaises(subject.ReadinessError):self.validate(case)
        case=sampled_case();case['native_outcome_sampling']['aborted_actions']=[dict(ticks=1)]
        case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])
        with self.assertRaises(subject.ReadinessError):self.validate(case)

    def test_action_after_terminal_and_uncovered_counted_ticks_rejected(self):
        case=sampled_case(ticks=11);case['native_outcome_sampling']['completed_actions'][0]['sampled_outcome']=outcome(success=True)
        case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])
        with self.assertRaises(subject.ReadinessError):self.validate(case)
        case=sampled_case(ticks=11);case['native_outcome_sampling']['completed_actions'].pop()
        case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])
        with self.assertRaises(subject.ReadinessError):self.validate(case)

    def test_public_hold_budget_one_and_partial_pose_action_are_valid(self):
        case=sampled_case();row=case['native_outcome_sampling']['completed_actions'][0]
        row.update(label='public_hold',max_ticks=1)
        case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling']);self.validate(case)
        self.validate(sampled_case(ticks=199))

    def test_native_task_clock_and_budget_cannot_be_replaced_by_a_counted_flag(self):
        for field,value in (('actual_task_seconds',0.),('actual_physics_seconds',1.)):
            case=sampled_case();case['alignment_receipt'][field]=value
            with self.subTest(field=field),self.assertRaises(subject.ReadinessError):self.validate(case)
        case=sampled_case();case['alignment_receipt']['clock']['budget_seconds']=True
        with self.assertRaises(subject.ReadinessError):self.validate(case)

    def test_original_manifest_first_four_are_ordered_rows_without_reading_samples(self):
        rows=[dict(task=t,raw_episode_id='episode'+str(i),path='never-opened.h5') for t in subject.TASKS for i in range(100)]
        manifest=dict(split='train',episodes=rows);sources=subject._sources(manifest)
        self.assertEqual(len(sources),64);self.assertIn((subject.TASKS[0],'episode3'),sources)
        self.assertNotIn((subject.TASKS[0],'episode4'),sources)
        manifest['episodes'][-1]=deepcopy(rows[0])
        with self.assertRaises(subject.ReadinessError):subject._sources(manifest)

    def test_witness_and_identity_are_pure_and_split_explicit(self):
        with TemporaryDirectory() as directory:
            graph=graphs.compile_policy_graph(directory,graph_spec(directory));route=graph['routes'][0]
            graph_ref=dict(path='metadata/graph.json',sha256='a'*64)
            source=dict(task=route['task'],raw_episode_id='episode0',path='never-opened.h5')
            original=deepcopy((graph,route,source));base=dict(config_sha256='c'*64,sha256='old')
            for split,purpose in (('train','train_qualification'),('validation','evaluation')):
                witness=subject.make_graph_witness(graph_ref,graph,route,source,split=split)
                self.assertEqual(witness['actual_split'],split);self.assertEqual(witness['purpose'],purpose)
                identity=subject.bind_graph_identity(base,witness);without=dict(identity);without.pop('sha256')
                self.assertEqual(identity['sha256'],subject.digest(without));self.assertEqual(base['sha256'],'old')
            self.assertEqual((graph,route,source),original)


class Fresh192Contracts(unittest.TestCase):
    def fixture(self,mutate=None):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        return FreshFixture(temporary.name,mutate)

    def test_exact192_with_zero_success_and_failed_motion_qualifies_completeness_only(self):
        fixture=self.fixture();before=(fixture.root/fixture.graph_ref['path']).read_bytes();proof=fixture.audit()
        self.assertEqual(proof['completed'],192);self.assertEqual(len(proof['cases']),192)
        self.assertTrue(all(not r['native_success'] and not r['motion_pass'] for r in proof['cases']))
        self.assertEqual(proof['source_count'],64);self.assertEqual(len(proof['groups']),48)
        self.assertTrue(all(r['completed']==4 and r['native_successes']==0 for r in proof['groups']))
        self.assertFalse(proof['VAL_read']);self.assertFalse(proof['TEST_read'])
        self.assertFalse(proof['block_only_tick_success_call_claimed'])
        self.assertEqual((fixture.root/fixture.graph_ref['path']).read_bytes(),before)
        proof_ref=write(fixture.root,'runtime/framework/synthetic_owner/proof.json',proof)
        self.assertEqual(subject.require_readiness_proof(fixture.root,fixture.graph_ref,proof_ref),proof)
        changed=deepcopy(proof);changed['cases'][0]['native_success']=True
        proof_ref=write(fixture.root,proof_ref['path'],changed)
        with self.assertRaises(subject.ReadinessError):subject.require_readiness_proof(fixture.root,fixture.graph_ref,proof_ref)

    def test_missing_partial_duplicate_retry_and_outcome_filtering_are_not_ready(self):
        fixture=self.fixture();original=deepcopy(fixture.cohort)
        for change in ('missing','duplicate','retry','filter','reuse','attempt'):
            fixture.cohort=deepcopy(original)
            if change=='missing':fixture.cohort['records'].pop()
            elif change=='duplicate':fixture.cohort['records'][-1]=deepcopy(fixture.cohort['records'][0])
            elif change=='retry':fixture.cohort['automatic_retry']=True
            elif change=='filter':fixture.cohort['source_outcome_filtering']=True
            elif change=='reuse':fixture.cohort['reused_cases']=[dict(case='old744')]
            else:fixture.cohort['records'][0]['attempt']=True
            fixture.refresh()
            with self.subTest(change=change),self.assertRaises(subject.ReadinessError):fixture.audit()

    def test_owner_exit_is_actual_integer_zero_and_not_a_flag_only(self):
        fixture=self.fixture()
        for field,value in (('returncode',False),('returncode',1),('actual_tool_exit_observed',False),('natural_completion',False)):
            completion=deepcopy(fixture.owner_completion);completion[field]=value
            fixture.cohort['owner']['exec_completion']=write(fixture.root,'runtime/framework/synthetic_owner/exec_completion.json',completion)
            fixture.refresh()
            with self.subTest(field=field,value=value),self.assertRaises(subject.ReadinessError):fixture.audit()

    def test_spoofed_actual_worker_argv_pid_and_child_returncode_rejected(self):
        for phase,field,value in (('CPU_check_launch','worker','src/evaluation_worker.py'),
                ('native_terminal','returncode',False),('native','namespace',1),('CPU_check','GPU_started',True)):
            def mutate(kind,index,record):
                if index or kind!=phase:return
                if field=='worker':record['argv'][7]=value
                elif field=='returncode':record[field]=value
                elif field=='namespace':record['frozen_OS_environment']['outer_wrapper_pid']=value
                else:record['execution'][field]=value
            with self.subTest(phase=phase,field=field),self.assertRaises(subject.ReadinessError):self.fixture(mutate).audit()

    def test_actual820_manifest_and_method_specific_launcher_are_required(self):
        for replacement,index in (('src/not820.py',1),('a'*64,3),('scripts/run_aligned_gpu.sh',6)):
            def mutate(kind,row_index,record):
                if row_index==0 and kind=='native_launch':record['argv'][index]=replacement
            with self.subTest(index=index),self.assertRaises(subject.ReadinessError):self.fixture(mutate).audit()

    def test_success_and_fatal_are_retained_in_full_grid_without_selection(self):
        def mutate(kind,index,record):
            if kind!='case' or index not in (0,1):return
            sample=sampled_case(method=record['method'],success=index==0,fatal=index==1)
            for key in ('official_task_success','native_terminal_failure','native_outcome_sampling','alignment_receipt'):
                if key=='alignment_receipt':record[key].update(sample[key])
                else:record[key]=sample[key]
        result=self.fixture(mutate).audit()
        self.assertEqual(sum(r['native_success'] for r in result['cases']),1)
        self.assertEqual(sum(r['native_terminal_failure'] for r in result['cases']),1)
        self.assertEqual(result['completed'],192)

    def test_CPU_native_worker_process_cannot_be_reused(self):
        def mutate(kind,index,record):
            if index==0 and kind=='CPU_check':
                record['process'].update(pid=100001,start_ticks=40001)
                record['frozen_OS_environment'].update(observer_pid=100001,observer_start_ticks=40001)
        with self.assertRaises(subject.ReadinessError):self.fixture(mutate).audit()

    def test_changed_declared_530_profile_is_not_accepted_as_original_protocol(self):
        fixture=self.fixture();graph=deepcopy(fixture.graph)
        for change in ('authorization','source','parameter'):
            value=deepcopy(fixture.profile)
            if change=='authorization':value['user_authorization']='different profile'
            elif change=='source':value['sources_sha256'][subject.RML_LOCAL_SOURCES[0]]='a'*64
            else:value['parameters']['max_velocity']=True
            graph['physical']['common_RML']['profile']=write(fixture.root,'runtime/framework/changed_profile.json',value)
            with self.subTest(change=change),self.assertRaises(subject.ReadinessError):subject._profile(graph,fixture.root)

    def test_legacy_readiness_or_complete_flag_without_cohort_never_grants_access(self):
        fixture=self.fixture()
        for value in (dict(schema='r10_exact16_fixed_four_common_author_RML_readiness_v532',complete=True),
                dict(schema=subject.READINESS_SCHEMA,complete=True,completed=192)):
            ref=write(fixture.root,'runtime/framework/synthetic_owner/fake_proof.json',value)
            with self.assertRaises(subject.ReadinessError):subject.require_readiness_proof(fixture.root,fixture.graph_ref,ref)

    def test_wrong_model_branch_identity_and_source_witness_rejected(self):
        for field in ('model_branch','actual_worker','actual_preparer','physical_factory','source_raw_episode_id'):
            def mutate(kind,index,record):
                if index==0 and kind=='case':record['graph_witness'][field]='spoofed'
            with self.subTest(field=field),self.assertRaises(subject.ReadinessError):self.fixture(mutate).audit()
        def wrong_identity(kind,index,record):
            if index==0 and kind=='case':record['identity']['source_row_sha256']='a'*64
        with self.assertRaises(subject.ReadinessError):self.fixture(wrong_identity).audit()

    def test_original_RML_cleanup_scope_and_motion_evidence_cannot_be_omitted(self):
        for kind,field in (('case','common_author_RML_parameters'),('case','all_stage_counted_joint_motion'),
                           ('native','runtime_scope_restoration'),('hooks','reset')):
            def mutate(actual,index,record):
                if index or actual!=kind:return
                if field=='reset':record['reset']=[]
                elif actual=='native':record['execution'].pop(field)
                else:record.pop(field)
            with self.subTest(kind=kind,field=field),self.assertRaises(subject.ReadinessError):self.fixture(mutate).audit()

    def test_stale_file_SHA_source_graph_change_and_missing_artifact_rejected(self):
        fixture=self.fixture();case=fixture.records[0]['case'];path=fixture.root/case['path']
        old=path.read_bytes();path.write_bytes(old+b' ')
        with self.assertRaises(subject.ReadinessError):fixture.audit()
        path.write_bytes(old);source=fixture.graph['physical']['factories']['selected']['source']
        (fixture.root/source['path']).write_text('changed frozen physical policy')
        with self.assertRaises(subject.ReadinessError):fixture.audit()
        with self.assertRaises(subject.ReadinessError):
            subject.audit_fresh_train192(fixture.root,fixture.graph_ref,dict(path='runtime/missing.json',sha256='a'*64))


if __name__=='__main__':unittest.main()
