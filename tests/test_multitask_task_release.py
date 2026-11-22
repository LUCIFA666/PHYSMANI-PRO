"""Task gates on real JSON/raw chains; native/optimization never run.

Model training and recorded physical parity are mocked only in the explicitly
named coverage/finalization tests. Roster, locks, raw child/wrapper evidence,
five-method coverage, selection ledgers and TEST admission use real validators.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import task_release as gate
from our_mani.multitask.evaluation import METHODS,case_identity,digest,file_sha,native_completion_record
from our_mani.multitask.specs import TASK_SPECS
from tests.test_multitask_release_preflight import inventories
from scripts.run_multitask_fresh_queue_r10 import build_jobs


class TaskReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.base=self.root/'artifacts/framework/multitask_r10';self.base.mkdir(parents=True)
        self.train,self.val,self.counts=inventories();self.task='moving_basketball_in_hoop'
    def tearDown(self):self.tmp.cleanup()
    def put(self,name,data):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data));return path
    def pointer(self,path):return dict(path=str(Path(path).relative_to(self.root)),sha256=file_sha(path))
    def read(self,path):return json.loads(Path(path).read_text())
    def fixture(self,task=None,*,wrapper=False):
        self.task=task or self.task;task=self.task
        valrows=[dict(task=t,raw_episode_id=raw,path='/CPU/val_package_compressed/'+t+'/'+raw+'/data.h5') for t,raw in sorted(self.val)]
        self.manifest=self.put(gate.DEFAULT_VAL,dict(split='validation',episodes=valrows))
        self.put(gate.DEFAULT_TRAIN,dict(split='train',episodes=[dict(task=t,raw_episode_id=raw) for t,raw in sorted(self.train)]))
        self.artifact=self.put('artifact.json',dict(CPU_model_fixture_only=True))
        self.profile=self.put('profile.json',dict(TRAIN_evidence=self.pointer(self.artifact)))
        self.child=self.put('child.py',dict(CPU_source_fixture_only=True));self.parent=self.put('parent.py',{})
        self.config_value=dict(context_checkpoint=self.pointer(self.artifact),baseline_checkpoint='artifact.json',baseline_executor='CPU_fixture',
            learned_artifacts={task:{'geometry':self.pointer(self.artifact),'interaction':self.pointer(self.artifact),
                'cv_value':self.pointer(self.artifact),'context_value':self.pointer(self.artifact)}},
            continuation_identities={task:{p:'a'*64 for p in ('cv','context')}},
            initial_condition_tolerance_profile=self.pointer(self.profile),
            development_reset_profile=self.pointer(self.artifact),RML_factory_hook=self.pointer(self.artifact))
        self.config=self.put('config.json',self.config_value)
        required={p.name+'|fixture':8 for p in TASK_SPECS[task].phase_skills if not p.optional_after_native_completion}
        self.coverage_value=dict(schema=gate.COVERAGE,task=task,complete=True,formal_model_coverage_passed=True,errors=[],
            task_comparison_sha256=gate.task_comparison_sha256(self.config_value),full_experiment_tasks=list(TASK_SPECS),
            TRAIN_expected_parents=100,VAL_expected_parents=self.counts[task],
            coverage_source_sha256=file_sha(gate.__file__),original_model_gate_source_sha256=file_sha(gate.models.__file__),
            actual_TEST_read=False,model_fitting_performed=False,input_files_sha256={'artifact.json':file_sha(self.artifact)},
            sections=dict(canonical_train=dict(per_task=dict.fromkeys(TASK_SPECS,100)),canonical_validation=dict(per_task=self.counts),
                context=dict(passed=True,TRAIN_parent_sources=1400,VAL_parent_sources=275,optimizer_parent_sources=1400),
                geometry=dict(passed=True,task=task,shared_with_baseline=True,asset=self.pointer(self.artifact)),
                library=dict(passed=True,native_acquisition_parent_sources=100,original_TRAIN_priors=1400,completed_parent_sources_by_skill=required),
                critics={p:dict(passed=True,TRAIN_acquired_parent_sources=100,VAL_acquired_parent_sources=self.counts[task],
                    ranking_enabled=False,geometry_fallback_is_legal=True) for p in ('cv','context')},
                colour=dict(passed=True,required=TASK_SPECS[task].family in ('insert_onto_rotating_peg','push_moving_button'))))
        self.coverage=self.put('coverage.json',self.coverage_value)
        self.config_value['task_model_coverage_certificate']=self.pointer(self.coverage)
        self.put(self.config,self.config_value)
        self.experiment=self.put('experiment.json',dict(remaining_tasks=14,methods=list(METHODS),original_sources=dict(validation=275,official_test=560),
            stopping_rule=dict(preregistered_before_remaining14_VAL_and_TEST=True,deployment_or_success_module_benefit_assumed=False,
                performance_screen_failure='Report negative performance; complete fixed comparison without tuning TEST')))
        paths=[self.manifest,self.artifact,self.profile,self.child,self.parent,self.config,self.coverage,self.experiment]
        self.original=self.child
        if wrapper:
            self.original=self.put('original.py',{})
            self.local_launcher=self.put('scripts/run_local_sim.sh',{});self.gpu_launcher=self.put('scripts/run_aligned_gpu.sh',{})
            paths.extend([self.original,self.local_launcher,self.gpu_launcher])
        self.files={str(p.relative_to(self.root)):file_sha(p) for p in paths}
        self.lock=self.put('lock.json',dict(files=self.files))
        selected=[row for row in valrows if row['task']==task]
        jobs=build_jobs(selected,'validation',METHODS,file_sha(self.config),file_sha(self.lock))
        self.plan=dict(version='r10_fresh_policy_queue_v1',split='validation',methods=list(METHODS),replicas=1,
            source_episodes=len(selected),expected_cases=len(jobs),official_TEST_used_for_fitting=False,
            case_isolation='fresh_simulator_process_per_episode',config=self.pointer(self.config),manifest=self.pointer(self.manifest),
            frozen_inputs=self.pointer(self.lock),parent_source=self.pointer(self.parent),child_source=self.pointer(self.child),
            output=str((self.base/'VAL').relative_to(self.root)),jobs=jobs)
        if wrapper:self.plan['task_execution_contract']=dict(schema=gate.EXECUTION,task=task,
            original_evaluator=self.pointer(self.original),wrapper=self.pointer(self.child),
            launchers=dict(ours=self.pointer(self.local_launcher),physmani=self.pointer(self.gpu_launcher)))
        self.plan['sha256']=digest(self.plan);self.plan_path=self.put('plan.json',self.plan)
        self.canonical={(row['task'],row['raw_episode_id']):row for row in selected}
        return self.plan

    def raw_cases(self,*,wrapper=False):
        output=self.root/self.plan['output'];self.raw=[];self.canonical_paths=[]
        for index,job in enumerate(self.plan['jobs']):
            identity=job['identity'];ticks=3
            alignment=dict(clock=dict(ticks=ticks,dt=.05,budget_seconds=10.,simulation_seconds=.15),actual_physics_seconds=.15,
                actual_task_seconds=.15,native_success_latched=False,native_terminal_failure_latched=False)
            record=native_completion_record(identity,process_id=100+index,success=False,fatal=False,ticks=ticks,
                initial_conditions=dict(metric_only=True,policy_use_allowed=False),alignment_receipt=alignment)
            method=identity['method'];predictor=method.split('_')[0]
            record.update(cleanup_completed=True,evaluation_process_pid=100+index,entrypoint_sha256=file_sha(self.original),
                config_sha256=file_sha(self.config),manifest_sha256=file_sha(self.manifest),source_row_sha256=identity['source_row_sha256'],
                frozen_input_lock=self.plan['frozen_inputs'],video=None,recorded_video_in_formal_metrics=False,
                compute_seconds=.01,wall_seconds=.02,simulation_seconds=.15,continuation_attempts=[])
            if method=='physmani':record.update(checkpoint_sha256=file_sha(self.artifact),baseline_executor='CPU_fixture')
            else:record.update(predictor=predictor,continuation_sha256='a'*64,continuation_identity=dict(sha256='a'*64),
                interaction_library_sha256=file_sha(self.artifact),unexecuted_candidates_labelled=False,
                first_selection_policy='value_once' if method.endswith('_value') else 'geometry',downstream_selection_policy='geometry',
                actor_metrics=dict(candidate_selection_events=[]))
            child=output/'cases'/job['job_id']/'attempt0000/case.json'
            if wrapper:
                base=self.put(child.parent/'evaluate_run_case_receipt.json',record)
                hooks=self.put(child.parent/'trained_task_hook_receipt.json',dict(version='r10_trained_task_fresh_case_wrapper_v1',
                    identity=identity,actual_wrapper=self.pointer(self.child),actual_process_id=100+index,split='validation',
                    shared_development_hooks_applied=True,complete=True,error=None,original_TEST_reset_unmodified=False,
                    source_resampled=False,automatic_retry=False,policy_failure_label_assigned=False,
                    permissions={key:self.config_value[key] for key in ('development_reset_profile','RML_factory_hook')},
                    reset_recovery_receipts=[dict(complete=True,permission=self.config_value['development_reset_profile'],task=self.task,
                        split='validation',replica=0,source_resampled=False,source=dict(raw_episode_id=job['source_raw_episode_id']))],
                    RML_path_recovery_receipts=[]))
                record['task_execution_witness']=dict(schema=gate.EXECUTION,task=self.task,original_evaluator=self.pointer(self.original),
                    actual_wrapper=self.pointer(self.child),actual_process_id=100+index,split='validation',
                    coverage_certificate=self.pointer(self.coverage),task_validation_release=None,
                    base_evaluator_receipt=self.pointer(base),hook_receipt=self.pointer(hooks))
            self.put(child,record);self.raw.append(child)
            log=self.put(child.parent/'launcher.log',dict(CPU_fixture_no_process_started=True))
            launcher='scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh'
            command=['bash',launcher,self.plan['child_source']['path'],'--config',self.plan['config']['path'],'--manifest',self.plan['manifest']['path'],
                '--frozen-inputs',self.plan['frozen_inputs']['path'],'--split','validation','--task',self.task,'--raw-episode-id',job['source_raw_episode_id'],
                '--method',method,'--replica','0','--out',str(child.parent.relative_to(self.root))]
            record['parent_case_process_receipt']=dict(child_path=str(child.relative_to(self.root)),child_sha256=file_sha(child),
                child_process_id=100+index,launcher_process_id=9000+index,command=command,log_path=str(log.relative_to(self.root)),log_sha256=file_sha(log),
                plan_sha256=self.plan['sha256'],parent_source_sha256=file_sha(self.parent),returncode=0)
            canonical=self.put(output/'results'/(job['job_id']+'.json'),record);self.canonical_paths.append(canonical)
        self.put(output/'status.json',dict(stage='complete',complete=True,expected=len(self.plan['jobs']),completed=len(self.plan['jobs']),
            plan_sha256=self.plan['sha256'],errors=[],stop_requested=False))

    def parsed_plan(self):return dict(self.plan,_config=self.config_value)
    def update_record(self,index,mutate):
        raw=self.read(self.raw[index]);mutate(raw);self.put(self.raw[index],raw)
        parent=self.read(self.canonical_paths[index])['parent_case_process_receipt'];parent['child_sha256']=file_sha(self.raw[index])
        self.put(self.canonical_paths[index],dict(raw,parent_case_process_receipt=parent))
    def finalize(self):
        # Only physical repetition/parity are mocked. Actual raw/source/cases,
        # immutable model certificate, selection ledger and denominator gates run.
        with patch.object(gate,'profile_problems',return_value=[]),patch.object(gate,'compare_five_way',return_value=dict(complete=True,accepted=True,checks=[{}]*10)):
            return gate.finalize_task_validation_release(self.root,self.task,self.plan_path,self.pointer(self.coverage),self.base/'release.json',
                canonical_validation_manifest=self.manifest,experiment_plan=self.experiment)

    def test_full_original_roster_retained_but_only_task_five_methods(self):
        self.fixture();canonical,info=gate._canonical_task(gate.Proof(self.root),self.manifest,'validation',self.task)
        self.assertEqual(info['full_original_sources'],275);self.assertEqual(len(canonical),20)
        self.assertEqual(len(gate._plan(gate.Proof(self.root),self.plan_path,canonical,self.task)['jobs']),100)
    def test_real_shortfall_denominators_eighteen_and_nineteen(self):
        for task,n in [('insert_onto_rotating_peg',18),('insert_onto_rotating_peg_high_speed',19)]:
            with self.subTest(task=task):
                self.fixture(task);self.assertEqual(self.plan['expected_cases'],5*n)
                gate._plan(gate.Proof(self.root),self.plan_path,self.canonical,task)
    def test_four_methods_missing_or_duplicate_jobs_are_not_five_way(self):
        self.fixture()
        for mutate in (lambda p:p.update(methods=list(METHODS[:-1])),lambda p:p['jobs'].pop(),lambda p:p['jobs'].__setitem__(1,p['jobs'][0])):
            p=deepcopy(self.plan);mutate(p);p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'});self.put(self.plan_path,p)
            with self.assertRaises(ValueError):gate._plan(gate.Proof(self.root),self.plan_path,self.canonical,self.task)
    def test_subset_canonical_inventory_cannot_hide_remaining_tasks(self):
        self.fixture();self.put(self.manifest,dict(split='validation',episodes=list(self.canonical.values())))
        with self.assertRaisesRegex(ValueError,'fourteen'):gate._canonical_task(gate.Proof(self.root),self.manifest,'validation',self.task)
    def test_certificate_does_not_require_unrelated_libraries_or_colour(self):
        self.fixture();value=gate.require_task_coverage_certificate(self.root,self.task,self.pointer(self.coverage),self.config_value,self.files)
        self.assertTrue(value['passed']);self.assertEqual(set(self.config_value['learned_artifacts']),{self.task})
    def test_certificate_rejects_missing_critic_unsupported_skill_and_stale_source(self):
        self.fixture()
        for mutate in (lambda x:x['sections']['critics'].pop('context'),
                lambda x:x['sections']['library']['completed_parent_sources_by_skill'].update(approach=7),
                lambda x:x.update(coverage_source_sha256='0'*64),lambda x:x.update(task='put_rubbish_in_moving_bin')):
            x=deepcopy(self.coverage_value);mutate(x);p=self.put('mutated_coverage.json',x);files=dict(self.files,mutated_coverage_json='unused')
            files['mutated_coverage.json']=file_sha(p)
            with self.assertRaises(ValueError):gate.require_task_coverage_certificate(self.root,self.task,self.pointer(p),self.config_value,files)
    def test_unknown_config_change_invalidates_certificate_release_pointers_do_not(self):
        self.fixture();c=deepcopy(self.config_value);c['unknown_model_parameter']=1
        with self.assertRaisesRegex(ValueError,'stale'):gate.require_task_coverage_certificate(self.root,self.task,self.pointer(self.coverage),c,self.files)
        c=deepcopy(self.config_value);c.update(formal_comparison_ready=True,task_validation_release={'path':'future','sha256':'a'*64})
        self.assertEqual(gate.task_comparison_sha256(c),gate.task_comparison_sha256(self.config_value))
    def test_actual_all100_raw_records_and_zero_candidate_traces_pass(self):
        self.fixture();self.raw_cases();groups,receipts,status=gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
        self.assertEqual(len(groups),20);self.assertEqual(len(receipts),100);self.assertTrue(gate.audit_selection_usage(groups)['diagnostics_valid'])
    def test_raw_tamper_cannot_become_policy_negative(self):
        self.fixture();self.raw_cases();raw=self.read(self.raw[0]);raw['official_task_success']=True;self.put(self.raw[0],raw)
        with self.assertRaisesRegex(ValueError,'SHA'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_failed_queue_missing_raw_and_parent_command_are_rejected(self):
        self.fixture();self.raw_cases();record=self.read(self.canonical_paths[0]);record['parent_case_process_receipt']['command'][-1]='wrong/out'
        self.put(self.canonical_paths[0],record)
        with self.assertRaisesRegex(ValueError,'launcher command'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_model_or_demo_rewrite_rejected_even_after_rehashing_parent(self):
        self.fixture();self.raw_cases();self.update_record(0,lambda x:x.update(interaction_library_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'continuation/model'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_actual_wrapper_base_and_reset_receipts_pass(self):
        self.fixture(wrapper=True);self.raw_cases(wrapper=True)
        groups,receipts,status=gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
        self.assertEqual(len(receipts),100)
    def test_wrapper_cannot_rewrite_base_or_claim_missing_reset(self):
        self.fixture(wrapper=True);self.raw_cases(wrapper=True)
        self.update_record(0,lambda x:x.update(compute_seconds=9.))
        with self.assertRaisesRegex(ValueError,'rewrote'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_full_task_release_passes_actual_raws_not_other_tasks(self):
        self.fixture();self.raw_cases();result=self.finalize()
        self.assertTrue(result['complete'],result.get('errors'));self.assertEqual(result['completed_method_cases'],100)
        self.assertFalse(result['other_tasks_released']);self.assertFalse(result['post_VAL_threshold_invented'])
    def test_incomplete_writes_pending_never_release(self):
        self.fixture();self.raw_cases();self.canonical_paths[0].unlink();result=self.finalize()
        self.assertFalse(result['complete']);self.assertFalse(result['formal_test_permitted']);self.assertFalse((self.base/'release.json').exists())
    def test_real_parity_failure_blocks_even_all100_raw_complete(self):
        self.fixture();self.raw_cases()
        with patch.object(gate,'profile_problems',return_value=[]),patch.object(gate,'compare_five_way',return_value=dict(complete=True,accepted=False,checks=[{}]*10)):
            result=gate.finalize_task_validation_release(self.root,self.task,self.plan_path,self.pointer(self.coverage),self.base/'release.json',
                canonical_validation_manifest=self.manifest,experiment_plan=self.experiment)
        self.assertFalse(result['complete']);self.assertEqual(result['stage'],'recorded_condition_parity')
    def test_TEST_admission_requires_original_all560_and_same_frozen_task_config(self):
        self.fixture();self.raw_cases();release=self.finalize();self.assertTrue(release['complete'],release.get('errors'))
        release_path=self.base/'release.json';config=dict(self.config_value,formal_comparison_ready=True,task_validation_release=self.pointer(release_path))
        testrows=[dict(task=t,raw_episode_id='test'+str(i),path='/CPU/archive/'+t+'/test'+str(i),mapping_status='verified')
            for t in TASK_SPECS for i in range(40)]
        test=self.put(gate.DEFAULT_TEST,dict(split='test',episodes=testrows,episode_count=560,mapping_status='verified',training_calibration_or_parameter_selection_used=False))
        files=dict(self.files);files.update(release['input_files_sha256']);files[str(release_path.relative_to(self.root))]=file_sha(release_path)
        files[gate.DEFAULT_TEST]=file_sha(test)
        value=gate.require_task_test_release(self.root,self.task,self.pointer(release_path),config,files)
        self.assertEqual(value['TEST_method_cases'],200);self.assertFalse(value['other_tasks_released'])
        config['new_unknown_candidate_rule']=True
        with self.assertRaisesRegex(ValueError,'changed policy'):gate.require_task_test_release(self.root,self.task,self.pointer(release_path),config,files)
    def test_colour_constructor_is_not_called_for_basketball_or_bin(self):
        for task in ('moving_basketball_in_hoop','put_rubbish_in_moving_bin'):
            with patch('our_mani.multitask.public_roles.PublicRoleResolver',side_effect=AssertionError('unrelated colour gate')):
                result=gate._colour(None,{},task,{})
            self.assertFalse(result['required'])
    def test_model_audit_invokes_only_task_library_and_both_task_critics(self):
        self.fixture()
        for module in (gate,gate.models):
            target=self.root/'our_mani/multitask'/Path(module.__file__).name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(module.__file__).read_bytes())
        library=SimpleNamespace(templates=[SimpleNamespace(phase=p.name) for p in TASK_SPECS[self.task].phase_skills])
        with patch.object(gate.models,'_context',return_value=self.coverage_value['sections']['context']) as context,\
                patch.object(gate,'_geometry',return_value=self.coverage_value['sections']['geometry']),\
                patch.object(gate.models,'_library',return_value=(library,self.coverage_value['sections']['library'])) as fit,\
                patch.object(gate.models,'_critic',side_effect=lambda inputs,protocol,task,predictor,*rest:self.coverage_value['sections']['critics'][predictor]) as critic:
            result=gate.audit_task_model_coverage(self.root,self.task,self.config_value,context_training_view={})
        self.assertTrue(result['formal_model_coverage_passed'],result['errors'])
        self.assertEqual(fit.call_count,1);self.assertEqual(fit.call_args.args[2],self.task)
        self.assertEqual({call.args[3] for call in critic.call_args_list},{'cv','context'});self.assertEqual(context.call_count,1)
        # Only model loading/fit evidence are mocked here; full inventories and
        # final direct input rehash execute and no other task model is queried.
    def test_missing_model_certificate_never_reaches_case_acceptance(self):
        self.fixture();self.raw_cases();bad=deepcopy(self.coverage_value);bad['formal_model_coverage_passed']=False
        pointer=self.pointer(self.put('failed_coverage.json',bad))
        result=gate.finalize_task_validation_release(self.root,self.task,self.plan_path,pointer,self.base/'release.json',
            canonical_validation_manifest=self.manifest,experiment_plan=self.experiment)
        self.assertFalse(result['complete']);self.assertEqual(result['stage'],'task_model_coverage')
    def test_native_clock_failure_is_not_policy_negative(self):
        self.fixture();self.raw_cases();self.update_record(0,lambda x:x['alignment_receipt'].update(actual_physics_seconds=.2))
        with self.assertRaisesRegex(Exception,'actual native clock'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_wrapper_reset_receipt_incomplete_is_rejected_with_consistent_hashes(self):
        self.fixture(wrapper=True);self.raw_cases(wrapper=True)
        raw=self.read(self.raw[0]);pointer=raw['task_execution_witness']['hook_receipt'];path=self.root/pointer['path']
        hooks=self.read(path);hooks['reset_recovery_receipts']=[];self.put(path,hooks)
        self.update_record(0,lambda x:x['task_execution_witness'].update(hook_receipt=self.pointer(path)))
        with self.assertRaisesRegex(ValueError,'shared reset receipt'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_parent_failed_process_is_not_accepted_after_committing_case(self):
        self.fixture();self.raw_cases();x=self.read(self.canonical_paths[0]);x['parent_case_process_receipt']['returncode']=1
        self.put(self.canonical_paths[0],x)
        with self.assertRaisesRegex(ValueError,'returncode'):gate._cases(gate.Proof(self.root),self.parsed_plan(),self.task,self.canonical,self.files)
    def test_official_inventory_cannot_drop_other_task_or_forge_mapping(self):
        self.fixture()
        rows=[dict(task=t,raw_episode_id='test'+str(i),path='/CPU/archive/'+t+'/'+str(i),mapping_status='verified')
            for t in TASK_SPECS for i in range(40)]
        value=dict(split='test',episodes=rows,episode_count=560,mapping_status='verified',training_calibration_or_parameter_selection_used=False)
        path=self.put(gate.DEFAULT_TEST,value)
        selected,info=gate._canonical_task(gate.Proof(self.root),path,'test',self.task);self.assertEqual(len(selected),40)
        bad=deepcopy(value);bad['episodes'].pop();self.put(path,bad)
        with self.assertRaisesRegex(ValueError,'560-row'):gate._canonical_task(gate.Proof(self.root),path,'test',self.task)
        bad=deepcopy(value);bad['episodes'][-1]['mapping_status']='unverified';self.put(path,bad)
        with self.assertRaisesRegex(ValueError,'row mapping'):gate._canonical_task(gate.Proof(self.root),path,'test',self.task)


class TaskComponentReleaseTests(unittest.TestCase):
    """Real275 baseline validator + real80 task raw chains, closure-only mocked."""
    def setUp(self):
        from tests.test_multitask_composite_validation_release import CompositeValidationReleaseTests
        from our_mani.multitask import baseline_component
        self.baseline=baseline_component;self.fx=CompositeValidationReleaseTests();self.fx.setUp();f=self.fx
        self.task='moving_basketball_in_hoop'
        # Both actual configs retain this additional field identically; an
        # unknown early/final difference must still fail the real validator.
        f.early['task_model_coverage_certificate']=f.coverage
        f.final['task_model_coverage_certificate']=f.coverage
        f.early_config=f.put('early_config.json',f.early);f.final_config=f.put('final_config.json',f.final)
        experiment=json.loads((f.root/f.experiment['path']).read_text())
        experiment.update(remaining_tasks=14);experiment['original_sources']['official_test']=560
        f.experiment=f.put('experiment.json',experiment)
        for label,config in [('early',f.early_config),('final',f.final_config)]:
            old=getattr(f,label+'_lock');value=json.loads((f.root/old['path']).read_text())
            value['files'][config['path']]=config['sha256'];value['files'][f.experiment['path']]=f.experiment['sha256']
            setattr(f,label+'_lock',f.put(label+'_lock.json',value))
        f.early_plan_value=f.plan(['physmani'],f.early_config,f.early_lock,'early_child.py','early_fresh')
        f.early_plan=f.put('early_plan.json',f.early_plan_value)
        f.ours_plan_value=f.plan(list(METHODS[:-1]),f.final_config,f.final_lock,'final_child.py','ours_task')
        f.ours_plan_value['jobs']=[job for job in f.ours_plan_value['jobs'] if job['row']['task']==self.task]
        f.ours_plan_value.update(source_episodes=20,expected_cases=80)
        f.ours_plan_value['sha256']=digest({k:v for k,v in f.ours_plan_value.items() if k!='sha256'})
        f.ours_plan=f.put('ours_plan.json',f.ours_plan_value)
        f.closure['contract']['execution_entrypoints'].update(parent_source=f.early_plan_value['parent_source'],child_source=f.early_plan_value['child_source'])
        f.baseline_paths=f.raw_cases(f.early_plan_value,f.early,baseline_only=True)
        f.ours_paths=f.raw_cases(f.ours_plan_value,f.final,baseline_only=False)
        for canonical in f.ours_paths:
            record=json.loads(canonical.read_text());parent=record.pop('parent_case_process_receipt');child=f.root/parent['child_path']
            record.update(interaction_library_sha256=f.asset['sha256'],recorded_video_in_formal_metrics=False,
                continuation_attempts=[],actor_metrics=dict(candidate_selection_events=[]))
            f.put(child,record);parent['child_sha256']=file_sha(child)
            f.put(canonical,dict(record,parent_case_process_receipt=parent))
        value=dict(schema=gate.COMPOSITION,task=self.task,ours_queue_plan=f.ours_plan,
            baseline_early_queue_plan=f.early_plan,baseline_execution_contract=f.contract,canonical_validation_manifest=f.roster,
            raw_identities_rewritten=False,expected_source_cases=20,expected_method_cases=100)
        value['sha256']=digest(value);self.composition=f.put('task_composition.json',value)
    def tearDown(self):self.fx.tearDown()
    def run_release(self):
        f=self.fx
        with patch.object(self.baseline,'verify_execution_contract',return_value=f.closure),\
                patch.object(self.baseline,'calibration_dependency',side_effect=lambda row:{'fixture_row':row}),\
                patch.object(gate,'require_task_coverage_certificate',return_value=dict(passed=True)),\
                patch.object(gate,'profile_problems',return_value=[]),\
                patch.object(gate,'compare_five_way',return_value=dict(complete=True,accepted=True,checks=[{}]*10)):
            return gate.finalize_task_composite_validation_release(f.root,self.task,self.composition,f.coverage,f.base/'task_release.json',
                canonical_validation_manifest=f.root/f.roster['path'],experiment_plan=f.root/f.experiment['path'])
    def test_complete_task80_and_fresh_actual_full275_baseline_release100(self):
        result=self.run_release()
        self.assertTrue(result['complete'],result.get('errors'));self.assertEqual(result['completed_method_cases'],100)
        self.assertFalse(result['single_full_config_identity_claimed']);self.assertFalse(result['raw_case_identities_rewritten'])
        self.assertEqual(result['actual_baseline_config'],self.fx.early_config)
        self.assertNotEqual(result['actual_baseline_config'],result['validated_config'])
    def test_partial_baseline_cannot_be_sliced_into_task_release(self):
        f=self.fx;path=f.root/f.early_plan_value['output']/'status.json';value=json.loads(path.read_text())
        value.update(complete=False,completed=274);f.put(path,value)
        result=self.run_release();self.assertFalse(result['complete']);self.assertEqual(result['stage'],'fresh_full_baseline_component')
    def test_raw_baseline_change_rejected_by_real_component_validator(self):
        f=self.fx;record=json.loads(f.baseline_paths[0].read_text());child=f.root/record['parent_case_process_receipt']['child_path']
        child.write_text(child.read_text()+' ')
        result=self.run_release();self.assertFalse(result['complete']);self.assertEqual(result['stage'],'fresh_full_baseline_component')


if __name__=='__main__':unittest.main()
