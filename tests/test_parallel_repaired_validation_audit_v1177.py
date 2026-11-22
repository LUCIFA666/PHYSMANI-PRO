"""CPU-only synthetic315x3 receipts; no actual VAL/TEST or simulator access."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from our_mani.multitask import parallel_repaired_validation_audit_v1177 as subject
from tests.test_repaired_policy_graph_v1143 import graph_spec, put
from tests.test_repaired_readiness_v1144 import protocol, sampled_case, write
from tests.test_parallel_repaired_readiness_v1174 import make_session, add_resource_phase, make_CPU_affinity_fixture


class ValidationFixture:
    """All reset bytes and process receipts below are temporary test fixtures."""
    def __init__(self, root):
        self.root = Path(root); spec = graph_spec(root)
        self.affinity = make_CPU_affinity_fixture(root,spec)
        auditor = put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
        spec['control']['components']['evaluation_auditor'] = auditor
        spec['control']['sources_map']['evaluation_auditor'] = auditor
        # Use actual new controller bytes in the synthetic source graph. The
        # fixtures still never invoke its model loader or physical loop.
        inherited_entry=put(root,'scripts/run_repaired_policy_case_v1167_r10.py',(Path(__file__).resolve().parents[1]/'scripts/run_repaired_policy_case_v1167_r10.py').read_bytes())
        guard_source='our_mani/multitask/parallel_repaired_resource_guard_v1171.py'
        guard_ref=put(root,guard_source,(Path(__file__).resolve().parents[1]/guard_source).read_bytes())
        spec['control']['components']['resource_guard']=guard_ref
        spec['control']['sources_map']['resource_guard']=guard_ref
        control={name:put(root,name,(Path(__file__).resolve().parents[1]/name).read_bytes())
            for name in (subject.worker.WORKER,subject.worker.SOURCE,subject.worker.RUNTIME)}
        spec['control']['sources_map'].update(actual_worker1172=control[subject.worker.WORKER],
            actual_preparer1172=control[subject.worker.SOURCE])
        spec['physical']['sources_map']['consumer']=control[subject.worker.RUNTIME]
        for route in spec['routes']:
            for purpose,split in (('train','train'),('evaluation','validation')):
                route[purpose].update(worker=control[subject.worker.WORKER],preparer=control[subject.worker.SOURCE],
                    consumer=control[subject.worker.RUNTIME],case_schema=subject.worker.CASE_SCHEMAS[split])
        writer = put(root,subject.inputs.SOURCE,b'synthetic post-admission source writer')
        spec['control']['components']['split_source_lock_writer'] = writer
        spec['control']['sources_map']['split_source_lock_writer'] = writer
        spec['control']['contracts']['split_source_lock_schema'] = subject.gate.SPLIT_LOCK_SCHEMA
        qualification = write(root,'metadata/namespace_qualification.json',dict(complete=True))
        spec['control']['static_inputs']['environment_namespace_qualification'] = qualification
        for name in ('scripts/run_aligned_gpu.sh','scripts/run_local_sim.sh'):
            spec['control']['sources_map'][name] = put(root,name)
        for i,name in enumerate((*subject.base.RML_LOCAL_SOURCES,*subject.base.RML_NATIVE_SOURCES)):
            spec['physical']['sources_map']['common_RML_'+str(i)] = put(root,name)
        source_map = {p['path']:p['sha256'] for p in spec['physical']['sources_map'].values()}
        self.profile = dict(schema='r10_common_author_RML_profile_v530',methods=list(subject.METHODS),
            parameters=deepcopy(subject.base.PARAMETERS),user_authorization=subject.base.RML_AUTHORIZATION,
            original_waypoint_filter_and_RML_algorithm=True,motion_metrics_required_for_all_methods=True,
            measured_acceleration6_is_diagnostic_only=True,native_task_success_unchanged=True,
            sources_sha256={p:source_map[p] for p in subject.base.RML_LOCAL_SOURCES},
            native_sources_sha256={p:source_map[p] for p in subject.base.RML_NATIVE_SOURCES})
        spec['physical']['common_RML']['profile'] = write(root,'metadata/common_profile.json',self.profile)
        self.protocol = protocol(spec['physical']['outcome_sampling']['source'],spec['physical']['sources_map']['consumer'])
        spec['physical']['outcome_sampling']['protocol'] = write(root,'metadata/outcome_protocol.json',self.protocol)
        checkpoint=spec['physical']['shared_context_checkpoint'];model=spec['routes'][0]['models']['trained_model']
        configuration=write(root,'metadata/configuration.json',dict(context_checkpoint=checkpoint,
            baseline_checkpoint=str(self.root/checkpoint['path']),learned_artifacts={t:dict(geometry=model,
                interaction=model,rigid_layout=model,colour=model) for t in subject.TASKS}))
        for route in spec['routes']:route['configuration']=configuration
        self.graph_ref = subject.base.graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/synthetic_graph')
        self.graph = subject.base.graphs.read_policy_graph(root,self.graph_ref)
        self.stamp = dict(graph=self.graph_ref,**{k:self.graph[k] for k in subject.HASHES})
        self.rows=[]; frozen_files={}
        for index,task in enumerate(subject.TASKS):
            for i in range(20 if index < 11 else 19):
                name=f'val_package_compressed/{task}/episode{i}/data.h5'; ref=put(root,name,b'synthetic reset bytes')
                absolute=str(self.root/name); frozen_files[absolute]=ref['sha256']
                self.rows.append(dict(task=task,raw_episode_id=f'episode{i}',path=absolute))
        self.manifest_ref=write(root,'metadata/synthetic_validation_manifest.json',dict(split='validation',episodes=self.rows))
        frozen_files[str(self.root/self.manifest_ref['path'])]=self.manifest_ref['sha256']
        routes_ref=write(root,'metadata/preflight_routes.json',{})
        actual_routes_ref=write(root,'metadata/actual_routes.json',{'actual':'1172'})
        projections=dict(preflight_routes=routes_ref,actual_routes=actual_routes_ref,inherited742_preflight_entry=inherited_entry)
        readiness_ref=write(root,'metadata/synthetic_fresh_TRAIN192_proof.json',{})
        self.admission=dict(routes=routes_ref,readiness=readiness_ref,tasks=list(subject.TASKS),methods=list(subject.METHODS),
            source_manifests=dict(validation=self.manifest_ref),**projections)
        self.admission_ref=write(root,'artifacts/framework/multitask_r10/synthetic_admission/admission.json',self.admission)
        source_inputs=dict(schema=subject.gate.SPLIT_LOCK_SCHEMA,producer_schema=subject.inputs.SCHEMA,
            producer=self.graph['control']['components']['split_source_lock_writer'],split='validation',source_count=315,
            source_manifest_sha256=self.manifest_ref['sha256'],admission=self.admission_ref,files=frozen_files,
            TEST_used_for_fitting=False,source_read_after_admission=True,original_VAL_bytes_reused_for_new_policy=True,
            new_independent_validation_dataset=False,**self.stamp,**projections)
        self.source_inputs_ref=write(root,subject.BUNDLE_PREFIX+'synthetic_run/source_inputs.json',source_inputs)
        suite=dict(schema=subject.gate.SUITE_SCHEMA,admission=self.admission_ref,source_inputs=self.source_inputs_ref,
            source_manifests=dict(validation=self.manifest_ref),readiness=readiness_ref,routes=routes_ref,
            formal_TEST_permitted=False,TEST_used_for_fitting=False,tasks=list(subject.TASKS),methods=list(subject.METHODS),**self.stamp,**projections)
        self.suite_ref=write(root,subject.BUNDLE_PREFIX+'synthetic_run/suite.json',suite)
        lock=dict(schema=subject.inputs.SCHEMA+'_suite_evidence_lock',split='validation',admission=self.admission_ref,
            formal_TEST_permitted=False,TEST_used_for_fitting=False,files={},**self.stamp)
        for ref in (self.graph_ref,self.admission_ref,self.suite_ref,self.source_inputs_ref,routes_ref,self.manifest_ref):
            lock['files'][str(self.root/ref['path'])]=ref['sha256']
        self.suite_lock_ref=write(root,subject.BUNDLE_PREFIX+'synthetic_run/suite_lock.json',lock)
        bundle=dict(schema=subject.inputs.SCHEMA,admission=self.admission_ref,suite=self.suite_ref,suite_lock=self.suite_lock_ref,
            source_inputs=self.source_inputs_ref,source_count=315,Native_started=False,TEST_read=False,**self.stamp)
        self.bundle_ref=write(root,subject.BUNDLE_PREFIX+'synthetic_run/bundle.json',bundle)
        self.metadata=dict(suite=self.suite_ref,suite_lock=self.suite_lock_ref,source_inputs=self.source_inputs_ref,
            admission=self.admission_ref,manifest=self.manifest_ref)
        self.owner_runtime=subject.OWNER_PREFIX+'synthetic'
        self.jobs=[];self.records=[];routes={(r['task'],r['method']):r for r in self.graph['routes']}
        for source in self.rows:
            for method in subject.METHODS:
                index=len(self.jobs); task=source['task'];raw=source['raw_episode_id'];route=routes[(task,method)]
                job=dict(job_id=f'j{index:04d}',split='validation',task=task,raw_episode_id=raw,method=method,configuration=route['configuration'],
                    source_row_sha256=subject.base.digest(source),
                    out=subject.worker.ARTIFACT_PREFIX+f'validation/synthetic/{task}/{raw}/{method}',
                    runtime=subject.worker.RUNTIME_PREFIX+f'validation/synthetic/{task}/{raw}/{method}')
                self.jobs.append(job)
                witness=subject.base.make_graph_witness(self.graph_ref,self.graph,route,source,split='validation')
                # Include a true, a fatal, and many false/motion-false outcomes.
                case=sampled_case(block=task in subject.BLOCKS,method=method,success=index==0,fatal=index==1)
                ep=dict(schema=subject.worker.SCHEMA,actual_worker=route['evaluation']['worker'],
                    actual_preparer=route['evaluation']['preparer'],inherited_consumer=route['evaluation']['consumer'],
                    source_overlay_lock=self.graph['physical']['source_overlay_lock'],selected_physical_factory=route['physical_factory'],
                    factory_binding=witness['factory_binding'],execution_branch=witness['model_branch'],actual_split='validation',
                    component_purpose='evaluation',formal_score_contribution=True,formal_validation_admitted=True,
                    post_admission_metadata=self.metadata,original_745_execution_acceptance_granted=False,
                    qualification_granted_by_worker=False,automatic_retry=False,TEST_supported=False,
                    inherited742_preflight_entry=inherited_entry,actual_worker_verified_by_new_control_graph=True,preflight_entry_is_actual_outer_worker=False,
                    additional_native_getter_observe_predict_or_success_calls=0,unchanged_action_outcome_sampling=self.protocol,**self.stamp)
                domain=dict(schema='r10_source_bound_RML_component_purpose_v640',purpose='evaluation',split='validation',
                    task=task,method=method,raw_episode_id=raw,policy_sha256=self.graph['policy_sha256'],VAL_TEST_relabelled_as_TRAIN=False)
                identity=subject.base.bind_graph_identity(dict(version='r10_fresh_case_identity_v1',task=task,method=method,
                    split='validation',raw_episode_id=raw+'__policy_aug_r0',parent_raw_episode_id=raw,source_raw_episode_id=raw,
                    replica=0,exploration_seed=None,demo_only=False,config_sha256=route['configuration']['sha256'],
                    frozen_inputs_sha256=route['frozen_inputs']['sha256'],source_row_sha256=subject.base.digest(source),
                    source_input_lock_sha256=self.source_inputs_ref['sha256'],
                    execution_protocol_schema=ep['schema'],execution_protocol_sha256=subject.base.digest(ep)),witness)
                model_proof=dict(GPU_started=False,split_source_input_lock=self.source_inputs_ref)
                if method=='physmani':
                    model_proof.update(original_author_checkpoint=dict(path=str(self.root/checkpoint['path']),sha256=checkpoint['sha256']),
                        original_neural_policy_and_online_world_model=True)
                    component=None
                elif task in subject.BLOCKS:
                    binding=dict(task=task,checkpoint_sha256=checkpoint['sha256'])
                    model_proof['actual_shared16_block_adapter']=binding
                    component=dict(actual_shared16_block_adapter=binding)
                else:
                    ci=dict(schema='r10_direct_public_RML_component_identity_v643',task=task,method=method,
                        shared_context_checkpoint=checkpoint,learned_operation_artifacts=dict(geometry=model,
                            interaction=model,rigid_layout=model,colour=model))
                    ci['sha256']=subject.base.digest(ci)
                    model_proof.update(actual_model_loader_validation=True,native_started=False,
                        old_TRAIN_actor_constructor_executed=False,component_identity=ci,component_domain=domain)
                    component=deepcopy(ci)
                motion=dict(schema='r10_original_baseline_measured6_independent_of_unused_plan_model_v345',
                    events=[dict(tick_before=0,tick_after=1,phase='hold',actual_issued_at=0.,reference_target_at=.05,
                        measured_q_before=[0.]*7,measured_q_after=[0.]*7,backward_window_velocity=[0.]*7,
                        backward_window_acceleration=None,position_passed=False,speed_passed=True,acceleration_passed=None,
                        diagnostic_only=True,native_parameters_changed=False)],violations=[1],all_stage_actual_window_checks_passed=False,
                    peak_actual_window_acceleration=0.,extra_physics_ticks=0,extra_camera_queries=0,policy_or_physics_modified=False,
                    original_planning_model_acceleration_unchanged=True,planning_metadata_used_as_actual_acceleration_cap=False,
                    actual_window_acceleration_criterion_rad_s2=6.)
                case.update(version=route['evaluation']['case_schema'],task=task,split='validation',source_raw_episode_id=raw,raw_episode_id=identity['raw_episode_id'],
                    parent_raw_episode_id=raw,identity=identity,process_id=1200000001+index*4,automatic_retry=False,
                    TEST_used_for_fitting=False,graph_witness=witness,component_domain=domain,author_outcome_protocol=self.protocol,
                    case_isolation='fresh_simulator_process_per_episode',formal_test_used=False,original_source_motion_replayed=False,
                    recorded_initial_conditions=dict(metric_only=True,policy_use_allowed=False),execution_protocol=ep,
                    source_overlay_lock=self.graph['physical']['source_overlay_lock'],original_745_execution_acceptance_granted=False,
                    qualification_granted_by_worker=False,formal_score_contribution=True,formal_TEST_metric=False,
                    demo_only=False,development_demo_only=False,formal_validation_metric=True,
                    shared_context_checkpoint=checkpoint,trained_component_identity=component,
                    actor_metrics=dict(original_baseline_algorithm_unchanged=method=='physmani'),
                    common_author_RML_parameters=dict(schema='r10_observed_common_author_RML_parameters_v530',complete=True,
                        profile_sha256=subject.base.digest(self.profile),actual_parameters=deepcopy(subject.base.PARAMETERS),
                        native_parameters_mutated=False,actual_motion_bound_claimed=False),all_stage_counted_joint_motion=motion)
                case['alignment_receipt']['all_stage_counted_joint_motion_v191']=deepcopy(motion)
                case_ref=write(root,job['out']+'/case.json',case)
                hooks=dict(actual_split='validation',reset=[] if task in subject.BLOCKS else [dict(complete=True)],
                    paths=[dict(close_complete=True)],execution_protocol=ep,graph_witness=witness)
                record=dict(task=task,raw_episode_id=raw,method=method,attempt=1,case=case_ref,
                    owned_hooks=write(root,job['out']+'/owned_hook_receipt.json',hooks))
                for check in (True,False):
                    phase='CPU_check' if check else 'native';base_path=job['runtime']+'/'+phase
                    inner=dict(pid=1200000000+index*4+int(not check),start_ticks=10000+index*4+int(not check))
                    outer=dict(pid=1200000002+index*4+int(not check),start_ticks=10002+index*4+int(not check))
                    argv=['python',self.graph['control']['components']['environment_wrapper']['path'],
                        '--manifest-sha256',self.graph['physical']['environment_manifest']['sha256'],'--','bash',
                        'scripts/run_aligned_gpu.sh' if route['GPU_required'] else 'scripts/run_local_sim.sh',route['evaluation']['worker']['path']]
                    for flag,value in (('graph',self.graph_ref['path']),('task',task),('raw-episode-id',raw),('method',method),
                            ('split','validation'),('out',job['out']),('runtime',job['runtime']),('config',route['configuration']['path']),
                            *[(name.replace('_','-'),ref['path']) for name,ref in self.metadata.items() if name!='manifest']):
                        argv.extend(('--'+flag,value))
                    if check:argv.append('--check')
                    launch=dict(process=outer,actual_worker=route['evaluation']['worker'],actual_preparer=route['evaluation']['preparer'],
                        argv=argv,automatic_retry=False,**self.stamp)
                    terminal=dict(process=outer,returncode=0,automatic_retry=False)
                    execution=dict(complete=True,identity=identity,graph_witness=witness,component_domain=domain,execution_protocol=ep,
                        mode='CPU_check' if check else 'graph_bound_native_validation',actual_split='validation',native_started=not check,
                        GPU_started=False,physical_factory_constructed=False,native_false_preserved=True,qualification_granted_by_worker=False,
                        source_and_model_preflight=model_proof,
                        runtime_scope_restoration=dict(complete=True,restored=True,checks=dict(writer_restored=True),extra_native_calls=0),
                        case=case_ref,native_success=case['official_task_success'],native_failure=case['native_terminal_failure'])
                    ns=dict(complete=True,outer_wrapper_pid=outer['pid'],outer_wrapper_start_ticks=outer['start_ticks'],
                        observer_pid=inner['pid'],observer_start_ticks=inner['start_ticks'],
                        wrapper=self.graph['control']['components']['environment_wrapper'],manifest=self.graph['physical']['environment_manifest'],
                        namespace_qualification=self.graph['control']['static_inputs']['environment_namespace_qualification'])
                    sidecar=dict(complete=True,process=inner,identity=identity,automatic_retry=False,
                        resource_proof=dict(complete=True,host_inventory=dict(host_visibility_available=True,
                            ancestor_chain_complete=True,same_uid_inventory_incomplete=False,live_project_native_processes=[])),
                        actual_worker=route['evaluation']['worker'],
                        actual_preparer=route['evaluation']['preparer'],execution=execution,graph_witness=witness,
                        frozen_OS_environment=ns,**self.stamp)
                    outer_base=self.owner_runtime+'/children/'+task+'/'+raw+'/'+method+'/'+phase
                    record[phase]=dict(launch=write(root,outer_base+'/launch.json',launch),terminal=write(root,outer_base+'/terminal.json',terminal),
                        sidecar=write(root,base_path+'/worker_sidecar.json',sidecar))
                self.records.append(record)
        self.plan=dict(schema=subject.PLAN_SCHEMA,split='validation',expected_cases=945,jobs=self.jobs,
            profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24),
            CPU_affinity=self.affinity['CPU_affinity'],
            owner_runtime=self.owner_runtime,
            validation_bundle=self.bundle_ref,automatic_retry=False,source_outcome_filtering=False,reused_cases=[],**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        process=dict(pid=2147483000,start_ticks=70000)
        owner_launch=write(root,self.owner_runtime+'/launch.json',dict(process=process,
            source=self.graph['control']['components']['evaluation_owner'],plan=self.plan_ref,**self.stamp))
        terminal=write(root,self.owner_runtime+'/terminal.json',dict(complete=True,completed=945,attempted=945,
            infrastructure_failed=0,active=None,automatic_retry=False,process=process,plan=self.plan_ref,**self.stamp))
        completion=write(root,self.owner_runtime+'/exec_completion.json',dict(process=process,
            actual_tool_exit_observed=True,natural_completion=True,returncode=0,plan=self.plan_ref,terminal=terminal))
        self.cohort=dict(schema=subject.COHORT_SCHEMA,complete=True,split='validation',expected_cases=945,
            validation_bundle=self.bundle_ref,records=self.records,plan=self.plan_ref,automatic_retry=False,
            source_outcome_filtering=False,reused_cases=[],owner=dict(launch=owner_launch,terminal=terminal,exec_completion=completion),**self.stamp)
        self.augment_resources(process)
        self.refresh()

    def augment_resources(self,owner):
        session_ref=make_session(self.root,self.owner_runtime,self.graph_ref,self.graph,
            owner=owner,plan_ref=self.plan_ref,jobs=self.jobs)
        for job,record in zip(self.jobs,self.records):
            for check in (True,False):
                phase='CPU_check' if check else 'native';refs=record[phase]
                sidecar=json.loads((self.root/refs['sidecar']['path']).read_text())
                launch=json.loads((self.root/refs['launch']['path']).read_text())
                sidecar,launch=add_resource_phase(self.root,self.owner_runtime,session_ref,self.graph_ref,
                    self.graph,job,sidecar,launch,check=check)
                if not check:sidecar['execution']['actual_native_interval']=dict(start_monotonic=1.,end_monotonic=2.)
                record[phase]['launch']=write(self.root,refs['launch']['path'],launch)
                record[phase]['sidecar']=write(self.root,refs['sidecar']['path'],sidecar)
        for kind in ('launch','terminal'):
            ref=self.cohort['owner'][kind]
            master=json.loads((self.root/ref['path']).read_text());master['resource_session']=session_ref
            self.cohort['owner'][kind]=write(self.root,ref['path'],master)
        ref=self.cohort['owner']['exec_completion']
        completion=json.loads((self.root/ref['path']).read_text());completion['terminal']=self.cohort['owner']['terminal']
        self.cohort['owner']['exec_completion']=write(self.root,ref['path'],completion)

    def refresh(self):
        self.cohort_ref=write(self.root,self.owner_runtime+'/cohort.json',self.cohort)

    @contextmanager
    def gate(self,*args):
        yield dict(graph=self.graph,graph_ref=self.graph_ref,admission=self.admission,admission_ref=self.admission_ref,permit=object())

    @contextmanager
    def authorized(self):
        with patch.object(subject.gate,'admission_scope',side_effect=self.gate),\
             patch.object(subject.gate,'read_formal_source',return_value=({},self.rows,dict(actual_split='validation',
                component_purpose='evaluation',policy_sha256=self.graph['policy_sha256'],formal_TEST_permitted=False))),\
             patch.object(subject.gate,'_permit',return_value={}),\
             patch.object(subject.gate,'_graph',return_value=self.graph):
            yield

    def audit(self):
        with self.authorized():return subject.audit_validation(self.root,self.graph_ref,self.bundle_ref,self.cohort_ref,permit=object())

    @contextmanager
    def modified(self, mutate):
        """Restore all fixture bytes and original pointers after one adversary."""
        before=deepcopy(self.cohort); saved={}
        def change(ref,fn):
            path=self.root/ref['path'];saved.setdefault(path,path.read_bytes());v=json.loads(path.read_bytes());fn(v)
            return write(self.root,ref['path'],v)
        try:
            mutate(self.cohort,change);self.refresh();yield
        finally:
            for path,data in saved.items():path.write_bytes(data)
            self.cohort=before;self.refresh()


class ValidationAuditContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=TemporaryDirectory();cls.fixture=ValidationFixture(cls.directory.name)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_actual1172_worker_preparer_consumer_and_schema_cannot_be_old_controller(self):
        original=self.fixture.graph
        for field in ('worker','preparer','consumer','case_schema'):
            changed=deepcopy(original);declaration=changed['routes'][0]['evaluation']
            if field=='case_schema':declaration[field]='r10_graph_bound_repaired_policy_case_v1146_validation_case'
            else:
                declaration[field]=dict(root='project',path='metadata/old_'+field+'.py',sha256='0'*64)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'actual1172 worker/preparer'):
                subject._worker_declarations(self.fixture.root,changed)

    def test_old_owner_cohort_namespace_is_refused_before_any_read(self):
        bad=dict(self.fixture.cohort_ref,path=self.fixture.cohort_ref['path'].replace('v1173/','v1149/'))
        with patch.object(subject,'read_json',side_effect=AssertionError('no metadata/source read')):
            with self.assertRaisesRegex(ValueError,'typed project metadata'):
                subject.audit_validation(self.fixture.root,self.fixture.graph_ref,self.fixture.bundle_ref,bad,permit=object())

    def test_complete945_accepts_false_fatal_and_motion_false_without_success_gate(self):
        result=self.fixture.audit()
        self.assertEqual((result['source_count'],result['case_count']),(315,945))
        self.assertEqual(sum(r['native_success'] for r in result['cases']),1)
        self.assertEqual(sum(r['native_terminal_failure'] for r in result['cases']),1)
        self.assertTrue(all(r['motion_pass'] is False for r in result['cases']))
        self.assertFalse(result['formal_TEST_permitted']);self.assertFalse(result['new_independent_validation_dataset'])
        self.assertEqual(result,self.fixture.audit())

    def test_denied_scope_precedes_source_reader_hash_or_cohort_access(self):
        @contextmanager
        def denied(*args):raise ValueError('fresh192 missing');yield
        with patch.object(subject.gate,'admission_scope',side_effect=denied),\
             patch.object(subject.gate,'read_formal_source',side_effect=AssertionError('unauthorized reader')),\
             patch.object(subject.inputs,'_hash_file',side_effect=AssertionError('unauthorized source hash')):
            with self.assertRaisesRegex(ValueError,'fresh192 missing'):
                subject.audit_validation(self.fixture.root,self.fixture.graph_ref,self.fixture.bundle_ref,
                    self.fixture.cohort_ref,permit=object())

    def test_manifest_or_TEST_pointer_cannot_be_read_as_bootstrap_bundle(self):
        for name in ('artifacts/framework/protocol/validation_manifest.json',
                     'artifacts/framework/protocol/test_manifest.json','metadata/source_inputs.json'):
            with self.subTest(name=name),patch.object(subject,'read_json',side_effect=AssertionError('forbidden read')):
                with self.assertRaisesRegex(ValueError,'metadata bundle required'):
                    subject.audit_validation(self.fixture.root,self.fixture.graph_ref,dict(path=name,sha256='a'*64),
                        self.fixture.cohort_ref,permit=object())

    def test_source_lock_and_model_loader_must_match_actual_graph_branch(self):
        changes=[lambda s:s['execution']['source_and_model_preflight']['split_source_input_lock'].update(sha256='a'*64),
            lambda s:s['execution']['source_and_model_preflight']['component_identity'].update(shared_context_checkpoint=dict(path='other.pt',sha256='a'*64)),
            lambda s:s['execution']['source_and_model_preflight'].update(actual_model_loader_validation=False)]
        for fn in changes:
            def edit(c,change):c['records'][0]['CPU_check']['sidecar']=change(c['records'][0]['CPU_check']['sidecar'],fn)
            with self.subTest(change=changes.index(fn)),self.fixture.modified(edit):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_partial_duplicate_retry_filter_or_train_schema_cannot_become_validation(self):
        changes=[lambda c:c['records'].pop(),lambda c:c['records'].__setitem__(1,deepcopy(c['records'][0])),
            lambda c:c.update(automatic_retry=True),lambda c:c.update(source_outcome_filtering=True),
            lambda c:c.update(schema=subject.base.COHORT_SCHEMA),lambda c:c.update(split='train'),
            lambda c:c.update(reused_cases=[c['records'][0]['case']]),lambda c:c['records'][0].update(attempt=2)]
        for mutate in changes:
            with self.subTest(change=changes.index(mutate)),self.fixture.modified(lambda c,_:mutate(c)):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_missing_case_or_child_evidence_is_not_an_outcome(self):
        for key in ('case','owned_hooks'):
            with self.subTest(key=key),self.fixture.modified(lambda c,_:c['records'][0].update({key:dict(path='missing.json',sha256='0'*64)})):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_wrong_worker_or_820_binding_or_factory_restoration_rejected(self):
        changes=[lambda s:s.update(actual_worker=dict(path='other.py',sha256='a'*64)),
            lambda s:s['frozen_OS_environment'].update(observer_pid=999),
            lambda s:s['execution']['runtime_scope_restoration'].update(restored=False),
            lambda s:s['execution']['runtime_scope_restoration']['checks'].update(writer_restored=False),
            lambda s:s['execution'].update(actual_split='train'),lambda s:s['execution'].update(native_failure=True)]
        for mutate in changes:
            def edit(c,change):c['records'][0]['native']['sidecar']=change(c['records'][0]['native']['sidecar'],mutate)
            with self.subTest(change=changes.index(mutate)),self.fixture.modified(edit):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_child_bool_exit_and_missing_formal_argv_rejected(self):
        for kind,fn in [('terminal',lambda v:v.update(returncode=False)),
                        ('launch',lambda v:v['argv'].remove('--suite')),
                        ('launch',lambda v:v.update(automatic_retry=True))]:
            def edit(c,change):c['records'][0]['CPU_check'][kind]=change(c['records'][0]['CPU_check'][kind],fn)
            with self.subTest(kind=kind),self.fixture.modified(edit):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_owner_must_really_exit_naturally_and_cover_all945(self):
        changes=[('exec_completion',lambda v:v.update(returncode=False)),
            ('exec_completion',lambda v:v.update(actual_tool_exit_observed=False)),
            ('exec_completion',lambda v:v.update(natural_completion=False)),
            ('terminal',lambda v:v.update(completed=944)),('terminal',lambda v:v.update(infrastructure_failed=1))]
        for kind,fn in changes:
            def edit(c,change):c['owner'][kind]=change(c['owner'][kind],fn)
            with self.subTest(kind=kind),self.fixture.modified(edit):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_no_case_source_model_or_graph_identity_substitution(self):
        changes=[lambda c:c['identity'].update(source_row_sha256='a'*64),
            lambda c:c['identity'].update(config_sha256='a'*64),
            lambda c:c['identity'].update(frozen_inputs_sha256='a'*64),
            lambda c:c['component_domain'].update(purpose='train_qualification'),
            lambda c:c['execution_protocol'].update(selected_physical_factory='other'),
            lambda c:c.update(cleanup_completed=False),lambda c:c['counted_clock'].update(seconds=.1)]
        for fn in changes:
            def edit(c,change):c['records'][0]['case']=change(c['records'][0]['case'],fn)
            with self.subTest(change=changes.index(fn)),self.fixture.modified(edit):
                with self.assertRaises(ValueError):self.fixture.audit()

    def test_empty_block_reset_is_valid_but_extra_core_reset_is_not(self):
        index=next(i for i,r in enumerate(self.fixture.records) if r['task'] not in subject.BLOCKS)
        def edit(c,change):
            c['records'][index]['owned_hooks']=change(c['records'][index]['owned_hooks'],lambda h:h.update(reset=[]))
        with self.fixture.modified(edit):
            with self.assertRaises(ValueError):self.fixture.audit()

    def test_changed_reset_bytes_and_wrong_source_writer_are_rejected(self):
        path=Path(self.fixture.rows[0]['path']);old=path.read_bytes()
        try:
            path.write_bytes(b'changed after freeze')
            with self.assertRaisesRegex(ValueError,'reset bytes changed'):self.fixture.audit()
        finally:path.write_bytes(old)
        original=deepcopy(self.fixture.graph)
        try:
            self.fixture.graph['control']['components']['split_source_lock_writer']['sha256']='a'*64
            with self.assertRaises(ValueError):self.fixture.audit()
        finally:self.fixture.graph=original

    def test_actual_auditor_must_have_been_frozen_into_graph(self):
        original=deepcopy(self.fixture.graph)
        try:
            self.fixture.graph['control']['components']['evaluation_auditor']['sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'auditor must have been frozen'):self.fixture.audit()
        finally:self.fixture.graph=original


class ParallelResourceNegativeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=TemporaryDirectory();cls.fixture=ValidationFixture(cls.directory.name)
    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_resource_claim_and_terminal_substitution_rejected(self):
        changes=[lambda s:s['resource_scope_receipt'].update(test_only=True),
            lambda s:s['resource_scope_receipt'].update(restored=False),
            lambda s:s['resource_scope_receipt'].update(child_empty_inventory_claimed=True),
            lambda s:s['resource_scope_receipt'].update(slot=99),
            lambda s:s.update(resource_slot=1),
            lambda s:s['resource_scope_receipt'].update(phase='CPU_check'),
            lambda s:s['resource_scope_receipt'].update(session={'path':'forbidden_TEST/session.json','sha256':'a'*64})]
        for fn in changes:
            def edit(c,change):c['records'][0]['native']['sidecar']=change(c['records'][0]['native']['sidecar'],fn)
            with self.subTest(change=changes.index(fn)),self.fixture.modified(edit):
                with self.assertRaises(ValueError):self.fixture.audit()


if __name__=='__main__':unittest.main()
