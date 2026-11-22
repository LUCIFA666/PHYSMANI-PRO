"""Three-method interfaces with actual basketball library identities and real raw chains.

Native runtime closure/calibration, model certificate provenance/continuation construction and
physical parity use CPU fixtures. Actual library loading/semantic identity,
wrapper VERSION source parsing, raw/base/parent/hook chains, task subset
baseline validator, three-method coverage and selection audit are unmocked.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import unittest
from types import SimpleNamespace
from dataclasses import make_dataclass
from unittest.mock import patch

from our_mani.multitask import geometry_task_release as gate
from our_mani.multitask import task_release as old
from our_mani.multitask import baseline_task_component as baseline
from our_mani.multitask.evaluation import digest,file_sha
METHODS=gate.METHODS
from our_mani.multitask.interaction import InteractionLibrary
from our_mani.multitask.specs import TASK_SPECS
from scripts.run_multitask_fresh_queue_r10 import build_jobs
from tests import test_multitask_baseline_task_component as baseline_fixtures
from tests import test_multitask_task_release as task_fixtures
from tests import test_project_GPU_budget_history_r10 as budget_fixtures
from our_mani.multitask import gpu_budget as budget
from tests import test_multitask_reset_parity as parity_fixtures
from tests.test_multitask_release_preflight import inventories

PROJECT=Path(__file__).resolve().parents[1]


class GeometryTaskReleaseTests(unittest.TestCase):
    def setUp(self):
        self.f=baseline_fixtures.BaselineTaskComponentTests(methodName='runTest');self.f.setUp();self.f.fixture();f=self.f
        self.root=f.root;self.task=f.task
        actual=PROJECT/'artifacts/framework/multitask_r10/TRAIN_artifacts_full_goalv6_semanticv9_3100_v1'/self.task/'interaction.json'
        library_path=f.root/'actual_basketball_library.json';library_path.write_bytes(actual.read_bytes())
        self.library=InteractionLibrary.load(library_path);self.library_file=f.ptr(library_path)
        assert self.library.sha256!=self.library_file['sha256']
        wrapper=f.root/'scripts/run_geometry_task_formal_case_r10.py';wrapper.write_text("VERSION='r10_trained_geometry_task_formal_case_wrapper_v1'\n# CPU source-contract fixture; not executed.\n")
        parent=f.root/'ours_queue.py';parent.write_text('# CPU parent source fixture only\n')
        local=f.root/'scripts/run_local_sim.sh';local.write_text('# CPU actuator launcher fixture only\n')
        current=PROJECT/'our_mani/multitask/baseline_task_component.py'
        target=f.root/'our_mani/multitask/baseline_task_component.py';target.write_bytes(current.read_bytes())
        profile=f.put('profile.json',dict(TRAIN_evidence=self.library_file))
        f.config.update(context_checkpoint=self.library_file,initial_condition_tolerance_profile=profile,
            RML_factory_hook=f.config['development_reset_profile'],
            continuation_identities={self.task:{predictor:'a'*64 for predictor in ('cv','context')}})
        f.config['learned_artifacts'][self.task]=dict(geometry=self.library_file,interaction=self.library_file,
            colour=None,rigid_layout=None)
        counts={name:sum(row['task']==name for row in f.rows) for name in TASK_SPECS}
        coverage=dict(schema=gate.COVERAGE,task=self.task,complete=True,formal_model_coverage_passed=True,errors=[],
            task_comparison_sha256=old.task_comparison_sha256(f.config),full_experiment_tasks=list(TASK_SPECS),
            TRAIN_expected_parents=100,VAL_expected_parents=20,coverage_source_sha256=file_sha(gate.__file__),legacy_model_provenance_source_sha256=file_sha(old.__file__),
            methods=list(METHODS),Value_required=False,Value_artifacts_loaded=False,
            original_model_gate_source_sha256=file_sha(old.models.__file__),actual_TEST_read=False,model_fitting_performed=False,
            input_files_sha256={self.library_file['path']:self.library_file['sha256']},
            sections=dict(canonical_train=dict(per_task=dict.fromkeys(TASK_SPECS,100)),canonical_validation=dict(per_task=counts),
                context=dict(passed=True,TRAIN_parent_sources=1400,VAL_parent_sources=275,optimizer_parent_sources=1400),
                geometry=dict(passed=True,task=self.task,shared_with_baseline=True,asset=self.library_file),
                library=dict(passed=True,native_acquisition_parent_sources=100,original_TRAIN_priors=1400,
                    completed_parent_sources_by_skill={p.name+'|CPU_fixture':8 for p in TASK_SPECS[self.task].phase_skills}),
                continuations={p:dict(sha256='a'*64) for p in ('cv','context')},colour=dict(passed=True,required=False)))
        self.coverage=f.put('coverage.json',coverage);f.config['task_model_coverage_certificate']=self.coverage
        f.cfg=f.put('common_config.json',f.config)
        f.contract_value['projected_config_sha256']=digest(baseline.original.project_baseline_config(f.config))
        f.contract=f.put('execution_contract.json',f.contract_value);f.closure['contract']=f.contract_value
        f.files.update({str(f.root/p['path']):p['sha256'] for p in (f.cfg,f.contract,self.coverage,self.library_file,profile)})
        f.lock=f.put('baseline_actual_lock.json',dict(files=f.files))
        self.experiment=f.put('experiment.json',dict(remaining_tasks=14,methods=list(METHODS),comparison_scope='geometry_only_three_methods',Value_removed_by_user_instruction=True,original_sources=dict(validation=275,official_test=560),
            stopping_rule=dict(preregistered_before_geometry_only_VAL_and_TEST=True,deployment_or_success_module_benefit_assumed=False,
                performance_screen_failure='complete fixed comparison without TEST tuning')))
        union={**f.files,str(f.root/f.lock['path']):f.lock['sha256'],str(wrapper):file_sha(wrapper),str(parent):file_sha(parent),
            str(local):file_sha(local),str(target):file_sha(target),str(f.root/self.experiment['path']):self.experiment['sha256']}
        f.common_lock=f.put('common_union_lock.json',dict(files=union));self.files=old.models.normalized_lock(f.root,union)
        f.plan_value.update(config=f.cfg,frozen_inputs=f.lock,baseline_execution_contract=f.contract)
        f.plan_value['jobs']=build_jobs(f.rows,'validation',['physmani'],f.cfg['sha256'],f.lock['sha256'])
        f.plan_value['sha256']=digest({k:v for k,v in f.plan_value.items() if k!='sha256'});f.plan=f.put('plan.json',f.plan_value)
        shutil.rmtree(f.root/f.plan_value['output']);f.jobs=[job for job in f.plan_value['jobs'] if job['row']['task']==self.task]
        f.raw=[];f.canonical=[]
        for index,job in enumerate(f.jobs):f.case(job,index)
        f.status=f.put(f.plan_value['output']+'/status.json',dict(complete=False,queue_running=True,expected=275,completed=20,
            plan_sha256=f.plan_value['sha256'],errors=[],stop_requested=False))
        selected=[row for row in f.rows if row['task']==self.task]
        self.canonical={(row['task'],row['raw_episode_id']):row for row in selected}
        self.plan=dict(version='r10_fresh_policy_queue_v1',split='validation',methods=list(METHODS[:-1]),replicas=1,source_episodes=20,expected_cases=40,
            official_TEST_used_for_fitting=False,case_isolation='fresh_simulator_process_per_episode',config=f.cfg,manifest=f.roster,
            frozen_inputs=f.common_lock,parent_source=f.ptr(parent),child_source=f.ptr(wrapper),output=str((f.base/'ours').relative_to(f.root)),
            jobs=build_jobs(selected,'validation',METHODS[:-1],f.cfg['sha256'],f.common_lock['sha256']),
            task_execution_contract=dict(schema=old.EXECUTION,task=self.task,original_evaluator=f.ptr(f.root/'scripts/evaluate_multitask_case_r10.py'),
                wrapper=f.ptr(wrapper),wrapper_schema_version='r10_trained_geometry_task_formal_case_wrapper_v1',
                launchers=dict(ours=f.ptr(local),physmani=f.ptr(f.root/'scripts/run_aligned_gpu.sh'))))
        self.plan['sha256']=digest(self.plan);self.plan_pointer=f.put('ours_plan.json',self.plan)
        h=task_fixtures.TaskReleaseTests(methodName='runTest');self.h=h
        h.root=f.root;h.base=f.base;h.task=self.task;h.plan=self.plan;h.config_value=f.config;h.config=f.root/f.cfg['path'];h.child=wrapper
        h.original=f.root/'scripts/evaluate_multitask_case_r10.py';h.artifact=library_path;h.coverage=f.root/self.coverage['path']
        h.manifest=f.root/f.roster['path'];h.lock=f.root/f.common_lock['path'];h.parent=parent;h.raw_cases(wrapper=True)
        for index,path in enumerate(h.canonical_paths):
            record=h.read(path);witness=record['task_execution_witness'];base_record=h.read(f.root/witness['base_evaluator_receipt']['path'])
            base_record['interaction_library_sha256']=self.library.sha256;h.put(f.root/witness['base_evaluator_receipt']['path'],base_record)
            witness['base_evaluator_receipt']=h.pointer(f.root/witness['base_evaluator_receipt']['path'])
            hook_path=f.root/witness['hook_receipt']['path'];hook=h.read(hook_path);hook['version']='r10_trained_geometry_task_formal_case_wrapper_v1';h.put(hook_path,hook)
            witness['hook_receipt']=h.pointer(hook_path);record['interaction_library_sha256']=self.library.sha256
            h.put(h.raw[index],{k:v for k,v in record.items() if k!='parent_case_process_receipt'})
            record['parent_case_process_receipt']['child_sha256']=file_sha(h.raw[index]);h.put(path,record)
        spec=dict(schema=gate.COMPOSITION,task=self.task,ours_queue_plan=self.plan_pointer,baseline_queue_plan=f.plan,
            baseline_execution_contract=f.contract,canonical_validation_manifest=f.roster,raw_identities_rewritten=False,
            expected_source_cases=20,expected_method_cases=60,resource_budget_receipt=None)
        spec['sha256']=digest(spec);self.composition=f.put('composition.json',spec)
    def tearDown(self):self.f.tearDown()
    def plan_config(self):return dict(self.plan,_config=self.f.config)
    def finalize(self):
        with patch.object(baseline,'verify_execution_contract',return_value=self.f.closure),\
                patch.object(baseline,'calibration_dependency',side_effect=lambda row:{'fixture_row':row}),\
                patch.object(gate,'profile_problems',return_value=[]),\
                patch.object(gate,'compare_three_way',return_value=dict(complete=True,accepted=True,checks=[{}]*3)),\
                patch.object(gate,'_continuation_identity',return_value=dict(sha256='a'*64)):
            return gate.finalize_task_composite_validation_release(self.root,self.task,self.composition,self.coverage,self.f.base/'release_v2.json',
                canonical_validation_manifest=self.root/self.f.roster['path'],experiment_plan=self.root/self.experiment['path'])
    def test_actual_basketball_semantic_library_sha_and_geometry_v1_source_acceptance(self):
        self.assertNotEqual(self.library.sha256,self.library_file['sha256'])
        with patch.object(gate,'_continuation_identity',return_value=dict(sha256='a'*64)):
            groups,receipts,status=gate._cases(gate.Proof(self.root),self.plan_config(),self.task,self.canonical,self.files)
        self.assertEqual(len(receipts),40);self.assertTrue(gate.audit_selection_usage(groups)['diagnostics_valid'])
    def test_library_file_sha_is_not_a_valid_runtime_semantic_sha(self):
        self.h.update_record(0,lambda x:x.update(interaction_library_sha256=self.library_file['sha256']))
        with self.assertRaisesRegex(ValueError,'rewrote|continuation/model'):gate._cases(gate.Proof(self.root),self.plan_config(),self.task,self.canonical,self.files)
    def test_hook_v2_cannot_claim_geometry_contract_even_with_consistent_raw_SHA(self):
        record=self.h.read(self.h.canonical_paths[0]);pointer=record['task_execution_witness']['hook_receipt'];path=self.root/pointer['path']
        hook=self.h.read(path);hook['version']='r10_trained_task_fresh_case_wrapper_v2_first_actionable_value';self.h.put(path,hook)
        self.h.update_record(0,lambda x:x['task_execution_witness'].update(hook_receipt=self.h.pointer(path)))
        with self.assertRaisesRegex(ValueError,'hook receipt'):gate._cases(gate.Proof(self.root),self.plan_config(),self.task,self.canonical,self.files)
    def test_unknown_or_wrong_literal_wrapper_version_is_rejected(self):
        plan=deepcopy(self.plan);plan['task_execution_contract']['wrapper_schema_version']='unreviewed_version'
        with self.assertRaisesRegex(ValueError,'exact reviewed'):gate._execution_contract(gate.Proof(self.root),plan,self.files,self.task)
    def test_missing_parent_returncode_or_second_attempt_cannot_be_promoted(self):
        path=self.h.canonical_paths[0];record=self.h.read(path);record['parent_case_process_receipt'].pop('returncode');self.h.put(path,record)
        with self.assertRaisesRegex(ValueError,'returncode'):gate._cases(gate.Proof(self.root),self.plan_config(),self.task,self.canonical,self.files)
        record['parent_case_process_receipt']['returncode']=0;self.h.put(path,record)
        (self.h.raw[0].parent.parent/'attempt0001').mkdir()
        with self.assertRaisesRegex(ValueError,'unique planned'):gate._cases(gate.Proof(self.root),self.plan_config(),self.task,self.canonical,self.files)
    def test_full40_ours_plus_complete20_baseline_releases_task_without_other255(self):
        r=self.finalize();self.assertTrue(r['complete'],r.get('errors'));self.assertEqual(r['schema'],gate.RELEASE)
        self.assertEqual(r['completed_method_cases'],60);self.assertNotEqual(r['actual_baseline_lock'],r['validated_input_lock'])
        self.assertTrue(r['single_full_config_identity_claimed']);self.assertFalse(r['single_lock_identity_claimed'])
        self.assertEqual(r['coverage_source_sha256'],file_sha(gate.__file__))
    def test_baseline_task19_of20_is_pending_even_if_ours40_complete(self):
        self.f.canonical[0].unlink();r=self.finalize();self.assertFalse(r['complete']);self.assertEqual(r['stage'],'fresh_task_baseline_component')
    def test_one_geometry_method_cannot_release_task_TEST(self):
        spec=self.f.read(self.composition['path']);spec['expected_method_cases']=20
        spec['sha256']=digest({k:v for k,v in spec.items() if k!='sha256'});self.composition=self.f.put('geometry_only.json',spec)
        r=self.finalize();self.assertFalse(r['complete']);self.assertEqual(r['stage'],'explicit_task_composition')
    def test_declared_GPU_budget_is_pending_until_real_auditor_receipt(self):
        with self.assertRaisesRegex(ValueError,'verified source-bound receipt'):
            gate._resource_budget(gate.Proof(self.root),dict(GPU_execution_resource_budget={'profile':{}}),{},dict(resource_budget_receipt=None))
    def test_new_TEST_checker_requires_complete_three_way_and_original560(self):
        report=self.finalize();self.assertTrue(report['complete'],report.get('errors'))
        release=self.f.ptr(self.f.base/'release_v2.json')
        config=dict(self.f.config,formal_comparison_ready=True,task_validation_release=release)
        rows=[dict(task=task,raw_episode_id='test'+str(i),path='/CPU/archive/'+task+'/test'+str(i),mapping_status='verified')
            for task in TASK_SPECS for i in range(40)]
        test=self.f.put(gate.DEFAULT_TEST,dict(split='test',episodes=rows,episode_count=560,mapping_status='verified',
            training_calibration_or_parameter_selection_used=False))
        self.f.put(gate.DEFAULT_VAL,self.f.read(self.f.roster['path']))
        files=dict(self.files,**report['input_files_sha256']);files[release['path']]=release['sha256'];files[test['path']]=test['sha256']
        with patch.object(gate,'_continuation_identity',return_value=dict(sha256='a'*64)):
            result=gate.require_task_test_release(self.root,self.task,release,config,files)
        self.assertEqual(result['TEST_method_cases'],120)
        config['unreviewed_policy_change']=True
        with self.assertRaisesRegex(ValueError,'changed policy'):gate.require_task_test_release(self.root,self.task,release,config,files)
    def test_old_task_release_cannot_be_mixed_into_new_TEST_checker(self):
        report=self.finalize();self.assertTrue(report['complete'],report.get('errors'))
        report['schema']=old.RELEASE;ref=self.f.put('foreign_old_release.json',report)
        with self.assertRaisesRegex(ValueError,'failed, stale, foreign'):
            gate.require_task_test_release(self.root,self.task,ref,self.f.config,self.files)
    def test_TEST_canonical_VAL_rows_cannot_change_at_same_denominator(self):
        report=self.finalize();self.assertTrue(report['complete'],report.get('errors'))
        changed=self.f.read(self.f.roster['path']);changed['episodes'][0]['path']='/different/source'
        self.f.put(gate.DEFAULT_VAL,changed)
        with self.assertRaisesRegex(ValueError,'original task source'):
            gate.require_task_test_release(self.root,self.task,self.f.ptr(self.f.base/'release_v2.json'),self.f.config,self.files)
    def test_model_coverage_does_not_call_or_read_any_Value_artifact(self):
        train,_,_=inventories()
        self.f.put(gate.DEFAULT_TRAIN,dict(split='train',episodes=[dict(task=task,raw_episode_id=raw) for task,raw in train]))
        self.f.put(gate.DEFAULT_VAL,self.f.read(self.f.roster['path']))
        for module in (gate,gate.models):
            target=self.root/'our_mani/multitask'/Path(module.__file__).name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(module.__file__).read_bytes())
        summary=self.f.read(self.coverage['path'])['sections']
        with patch.object(gate.models,'_context',return_value=summary['context']),\
                patch.object(gate.models,'_library',return_value=(self.library,summary['library'])),\
                patch.object(gate,'_geometry',return_value=summary['geometry']),\
                patch.object(gate,'_continuation_identity',return_value=dict(sha256='a'*64)),\
                patch.object(gate.models,'_critic',side_effect=AssertionError('Value gate must not run')),\
                patch('our_mani.multitask.candidate_success.CandidateSuccessModel',side_effect=AssertionError('Value artifact must not load')):
            report=gate.audit_task_model_coverage(self.root,self.task,self.f.config,context_training_view={'CPU_fixture':True})
        self.assertTrue(report['formal_model_coverage_passed'],report['errors'])
        self.assertFalse(report['Value_required']);self.assertFalse(report['Value_artifacts_loaded'])
        self.assertNotIn('critics',report['sections'])
    def test_skill_support_seven_is_still_rejected_without_Value(self):
        report=self.f.read(self.coverage['path']);key=next(iter(report['sections']['library']['completed_parent_sources_by_skill']))
        report['sections']['library']['completed_parent_sources_by_skill'][key]=7
        ref=self.f.put('unsupported_geometry_coverage.json',report)
        with self.assertRaisesRegex(ValueError,'skill coverage'):
            gate.require_task_coverage_certificate(self.root,self.task,ref,self.f.config,self.files)
    def test_actual_eighteen_and_nineteen_parent_denominators_use_two_ours_methods(self):
        for task,n in (('insert_onto_rotating_peg',18),('insert_onto_rotating_peg_high_speed',19)):
            rows=[row for row in self.f.rows if row['task']==task]
            self.assertEqual(len(rows),n)
            canonical={(task,row['raw_episode_id']):row for row in rows}
            plan=deepcopy(self.plan);plan.update(source_episodes=n,expected_cases=2*n,
                jobs=build_jobs(rows,'validation',METHODS[:-1],self.plan['config']['sha256'],self.plan['frozen_inputs']['sha256']))
            plan['task_execution_contract']['task']=task
            plan['sha256']=digest({key:value for key,value in plan.items() if key!='sha256'})
            pointer=self.f.put(task+'_plan.json',plan)
            self.assertEqual(len(gate._plan(gate.Proof(self.root),self.root/pointer['path'],canonical,task,methods=METHODS[:-1])['jobs']),2*n)
    def test_duplicate_ours_job_cannot_hide_a_missing_parent_method(self):
        plan=deepcopy(self.plan);plan['jobs'][1]=plan['jobs'][0]
        plan['sha256']=digest({key:value for key,value in plan.items() if key!='sha256'})
        pointer=self.f.put('duplicate_job_plan.json',plan)
        with self.assertRaisesRegex(ValueError,'duplicated or changed'):
            gate._plan(gate.Proof(self.root),self.root/pointer['path'],self.canonical,self.task,methods=METHODS[:-1])


class ThreeWayPhysicalParityTests(unittest.TestCase):
    def test_exact_three_pairs_use_original_recorded_condition_comparator(self):
        record,_=parity_fixtures.record();profile=parity_fixtures.pending_tolerance_profile(parity_fixtures.LEGACY)
        records={method:deepcopy(record) for method in METHODS}
        result=gate.compare_three_way(records,profile,require_frozen=False)
        self.assertTrue(result['accepted']);self.assertTrue(result['complete'])
        self.assertEqual({(row['a'],row['b']) for row in result['checks']},
            {('cv_geometry','context_geometry'),('cv_geometry','physmani'),('context_geometry','physmani')})
    def test_actual_body_delta_and_pending_TRAIN_profile_still_reject(self):
        record,_=parity_fixtures.record();profile=parity_fixtures.pending_tolerance_profile(parity_fixtures.LEGACY)
        records={method:deepcopy(record) for method in METHODS}
        self.assertFalse(gate.compare_three_way(records,profile,require_frozen=True)['accepted'])
        records['physmani']['reset_receipt']['native_initial_snapshot']['bodies']['ball']['pose_xyzw'][0]+=.001
        result=gate.compare_three_way(records,profile,require_frozen=False)
        self.assertFalse(result['accepted']);self.assertEqual(sum(not row['accepted'] for row in result['checks']),2)
    def test_missing_or_old_five_method_scope_cannot_claim_complete_three_way(self):
        for methods in (METHODS[:-1],METHODS+('cv_value','context_value')):
            result=gate.compare_three_way(dict.fromkeys(methods,{}),{})
            self.assertFalse(result['complete']);self.assertEqual(result['checks'],[])


class ActualFeedbackIdentityTests(unittest.TestCase):
    """Real source/library/identity APIs; only controller runtime state is a CPU fixture."""
    def setUp(self):
        from our_mani.multitask.feedback_actor_backend import policy_overlay,feedback_continuation_identity
        from our_mani.multitask.feedback_interaction import feedback_binding,ObservedFeedbackController
        from our_mani.multitask.interaction import CandidateConfig,MPCConfig
        self.root=PROJECT;self.task='moving_basketball_in_hoop'
        self.protocol=json.loads((PROJECT/'artifacts/framework/multitask_r10/basketball_Context_geometry_TRAIN_development_demo_v1_setup/configuration.json').read_text())
        self.protocol['policy_overlay']=policy_overlay(PROJECT,{})
        ref=self.protocol['learned_artifacts'][self.task]['interaction'];self.library=InteractionLibrary.load(PROJECT/ref['path'])
        self.identities={}
        for predictor in ('cv','context'):
            identity=feedback_continuation_identity(PROJECT,self.protocol,self.task,predictor,self.library,
                CandidateConfig(**self.protocol['candidate_config']),MPCConfig(**self.protocol['mpc_config']))
            self.protocol['continuation_identities'][self.task][predictor]=identity['sha256'];self.identities[predictor]=identity
        fixture=SimpleNamespace(feedback_binding=feedback_binding({}),continuation_sha256=self.identities['cv']['sha256'],
            feedback_step_calls=0,feedback_events=[],phase_timing_events=[])
        self.receipt=ObservedFeedbackController.feedback_metrics(fixture)
        self.case=dict(task=self.task,predictor='cv',continuation_sha256=self.identities['cv']['sha256'],
            continuation_identity=self.identities['cv'],actor_metrics=dict(observed_feedback_controller=self.receipt))
    def check(self):
        return gate._verify_feedback_case(gate.models._Inputs(PROJECT),self.case,dict(task=self.task,predictor='cv'),self.identities['cv'])
    def test_actual_extended_identity_binds_both_predictors_with_no_Value_loading(self):
        with patch('our_mani.multitask.candidate_success.CandidateSuccessModel',side_effect=AssertionError('Value load forbidden')):
            for predictor in ('cv','context'):
                actual=gate._continuation_identity(gate.models._Inputs(PROJECT),self.protocol,self.task,predictor,self.library)
                self.assertEqual(actual,self.identities[predictor]);self.assertNotEqual(actual['sha256'],actual['original_continuation']['sha256'])
        self.check() # A zero-candidate controller receipt is not censored or invented.
    def test_old_continuation_or_missing_actual_controller_cannot_claim_new_runtime(self):
        self.case['continuation_identity']=self.identities['cv']['original_continuation']
        with self.assertRaisesRegex(ValueError,'complete actual extended'):self.check()
        self.case['continuation_identity']=self.identities['cv'];self.case['actor_metrics']={}
        with self.assertRaisesRegex(ValueError,'installed measured-feedback'):self.check()
    def test_controller_source_config_and_completion_claims_are_strict(self):
        for mutate in (lambda r:r['source'].update(sha256='0'*64),lambda r:r.update(configuration_sha256='0'*64),
                lambda r:r.update(nominal_completion_claimed=True),lambda r:r.update(total_budget_extended=True)):
            self.case['actor_metrics']['observed_feedback_controller']=deepcopy(self.receipt)
            mutate(self.case['actor_metrics']['observed_feedback_controller'])
            with self.assertRaisesRegex(ValueError,'controller/config/source/gates'):self.check()
    def test_wrong_map_to_original_id_is_rejected(self):
        self.protocol['continuation_identities'][self.task]['cv']=self.identities['cv']['original_continuation']['sha256']
        with self.assertRaisesRegex(ValueError,'actual original source'):
            gate._continuation_identity(gate.models._Inputs(PROJECT),self.protocol,self.task,'cv',self.library)
    def test_declared_slack_replays_real_helper_math_and_cannot_extend_task_budget(self):
        from our_mani.multitask.feedback_actor_backend import policy_overlay,feedback_continuation_identity
        from our_mani.multitask.feedback_interaction import FeedbackConfig,feedback_binding,ObservedFeedbackController
        from our_mani.multitask.interaction import CandidateConfig,MPCConfig
        cfg=FeedbackConfig(phase_tracking_slack_seconds=2.)
        self.protocol['policy_overlay']=policy_overlay(PROJECT,cfg)
        identity=feedback_continuation_identity(PROJECT,self.protocol,self.task,'cv',self.library,
            CandidateConfig(**self.protocol['candidate_config']),MPCConfig(**self.protocol['mpc_config']))
        goal_type=make_dataclass('CPUGoal',['candidate_id','selected_at','phase_budget','suffix_budget'])
        state=SimpleNamespace(robot=SimpleNamespace(timestamp=1.))
        fixture=SimpleNamespace(goal=goal_type('CPU_candidate',1.,.2,1.),_phase_slack_tokens=set(),
            attempts=SimpleNamespace(records=[]),feedback_config=cfg,phase_timing_events=[],feedback_events=[],
            feedback_binding=feedback_binding(cfg),continuation_sha256=identity['sha256'],feedback_step_calls=1)
        ObservedFeedbackController._grant_phase_slack(fixture,state,9.)
        receipt=ObservedFeedbackController.feedback_metrics(fixture)
        self.assertEqual(receipt['phase_timing_events'][0]['granted_slack_seconds'],2.)
        self.case.update(continuation_identity=identity,continuation_sha256=identity['sha256'],
            actor_metrics=dict(observed_feedback_controller=receipt));self.identities['cv']=identity
        self.check()
        receipt['phase_timing_events'][0]['granted_slack_seconds']=9.
        with self.assertRaisesRegex(ValueError,'slack exceeds'):self.check()


class ResourceBudgetCompositionTests(unittest.TestCase):
    """Use the real historical validator; daemon/proc/control only are CPU fakes."""
    def setUp(self):
        self.f=budget_fixtures.BudgetHistoryTests(methodName='runTest');self.f.setUp()
        self.record=self.f.work/'baseline_native_case.json';self.record.write_text(json.dumps(dict(process_id=904)))
        self.component=dict(sources=1,completed_cases=[dict(raw_child_receipt=budget.pointer(self.record))])
        self.config=dict(GPU_execution_resource_budget=dict(profile=self.f.prepared['profile'],
            environment_contract=self.f.prepared['execution_environment_contract']))
        self.spec=dict(resource_budget_receipt=budget.pointer(self.f.receipt_path))
    def tearDown(self):self.f.tearDown()
    def test_actual_resource_chain_is_replayed_and_all_files_frozen(self):
        proof=gate.Proof(PROJECT);result=gate._resource_budget(proof,self.config,self.component,self.spec)
        self.assertEqual(result['required_case_process_ids'],[904]);self.assertTrue(result['valid'])
        for ref in self.f.receipt['runtime_events']+self.f.receipt['actual_MPS_log_files']:
            self.assertEqual(proof.files[proof.name(ref['path'])],ref['sha256'])
        self.assertIn(proof.name(self.f.metadata/'profile.json'),proof.files)
    def test_resource_case_PID_missing_from_actual_clients_is_rejected(self):
        self.record.write_text(json.dumps(dict(process_id=999)))
        self.component['completed_cases'][0]['raw_child_receipt']=budget.pointer(self.record)
        with self.assertRaisesRegex(ValueError,'case PID'):gate._resource_budget(gate.Proof(PROJECT),self.config,self.component,self.spec)
    def test_different_environment_contract_cannot_claim_common_conditions(self):
        alternate=self.f.work/'other_environment_contract.json';alternate.write_text('{}')
        self.config['GPU_execution_resource_budget']['environment_contract']=budget.pointer(alternate)
        with self.assertRaisesRegex(ValueError,'common configuration'):
            gate._resource_budget(gate.Proof(PROJECT),self.config,self.component,self.spec)
    def test_resource_raw_bytes_or_cleanup_cannot_be_replaced_by_summary(self):
        event=self.f.receipt['runtime_events'][0];Path(event['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):
            gate._resource_budget(gate.Proof(PROJECT),self.config,self.component,self.spec)


if __name__=='__main__':unittest.main()
