"""Release composition and raw receipt boundaries using temporary CPU fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import validation_release as release
from our_mani.multitask.evaluation import (METHODS,case_identity,digest,file_sha,native_completion_record)
from our_mani.multitask.specs import TASK_SPECS
from scripts.run_multitask_fresh_queue_r10 import build_jobs
from tests.test_multitask_release_preflight import inventories


class ValidationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=self.root/'artifacts/framework/multitask_r10';self.base.mkdir(parents=True)
        self.train,self.val,self.counts=inventories()
    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return path
    def pointer(self,path):return dict(path=str(path.relative_to(self.root)),sha256=file_sha(path))
    def fixture(self):
        rows=[dict(task=task,raw_episode_id=raw,path='/CPU/val_package_compressed/'+task+'/'+raw+'/data.h5')
            for task,raw in sorted(self.val)]
        manifest=self.write('manifest.json',dict(split='validation',episodes=rows))
        artifact=self.write('artifact.json',dict(CPU_fixture_only=True))
        profile=self.write('profile.json',dict(TRAIN_evidence=self.pointer(artifact)))
        child=self.write('child.py',{});parent=self.write('parent.py',{})
        config=self.write('config.json',dict(context_checkpoint=self.pointer(artifact),baseline_checkpoint=str(artifact.relative_to(self.root)),
            baseline_executor='CPU_fixture',initial_condition_tolerance_profile=self.pointer(profile),
            learned_artifacts={task:dict(geometry=self.pointer(artifact),interaction=self.pointer(artifact)) for task in TASK_SPECS},
            continuation_identities={task:{p:'a'*64 for p in ('cv','context')} for task in TASK_SPECS}))
        experiment=self.write('experiment.json',dict(methods=list(METHODS),original_sources=dict(validation=275),stopping_rule=dict(
            preregistered_before_remaining14_VAL_and_TEST=True,deployment_or_success_module_benefit_assumed=False,
            performance_screen_failure='Complete the fixed comparison, report negative results without tuning TEST')))
        paths=(manifest,artifact,profile,child,parent,config,experiment)
        lock=self.write('lock.json',dict(files={str(p.relative_to(self.root)):file_sha(p) for p in paths}))
        jobs=build_jobs(rows,'validation',METHODS,file_sha(config),file_sha(lock))
        plan=dict(version='r10_fresh_policy_queue_v1',split='validation',methods=list(METHODS),replicas=1,
            source_episodes=275,expected_cases=1375,official_TEST_used_for_fitting=False,
            case_isolation='fresh_simulator_process_per_episode',config=self.pointer(config),manifest=self.pointer(manifest),
            frozen_inputs=self.pointer(lock),parent_source=self.pointer(parent),child_source=self.pointer(child),
            output=str((self.base/'VAL').relative_to(self.root)),jobs=jobs)
        plan['sha256']=digest(plan);plan_path=self.write('plan.json',plan)
        self.write(plan['output']+'/status.json',dict(stage='incomplete',complete=False,completed=1374,expected=1375,
            plan_sha256=plan['sha256'],errors=[],stop_requested=False))
        coverage=self.write('coverage.json',dict(input_files_sha256={}))
        return plan,plan_path,manifest,config,experiment,coverage

    def test_full_plan_identity_coverage_and_missing_job(self):
        plan,path,manifest,config,experiment,coverage=self.fixture();proof=release.Proof(self.root)
        canonical=release._manifest(proof,manifest)
        self.assertEqual(len(release._plan(proof,path,canonical)['jobs']),1375)
        plan['jobs'].pop();plan['sha256']=digest({k:v for k,v in plan.items() if k!='sha256'});path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'paired five-method VAL job coverage'):
            release._plan(release.Proof(self.root),path,canonical)

    def test_incomplete_queue_writes_pending_and_no_formal_release(self):
        plan,path,manifest,config,experiment,coverage=self.fixture()
        out=self.base/'VAL_release.json'
        result=release.finalize_validation_release(self.root,path,self.pointer(coverage),out,
            canonical_validation_manifest=manifest,experiment_plan=experiment)
        self.assertFalse(result['complete']);self.assertFalse(result['formal_test_permitted'])
        self.assertEqual(result['stage'],'completed_native_cases')
        self.assertFalse(out.exists());self.assertTrue((self.root/result['pending_report']['path']).is_file())

    def single_raw_case(self):
        plan,path,manifest,config,experiment,coverage=self.fixture()
        job=next(job for job in plan['jobs'] if job['identity']['method']=='cv_geometry')
        plan['jobs']=[job];plan['_config']=json.loads(config.read_text())
        output=self.root/plan['output'];identity=job['identity'];ticks=3
        alignment=dict(clock=dict(ticks=ticks,dt=.05,budget_seconds=10.,simulation_seconds=ticks*.05),
            actual_physics_seconds=ticks*.05,actual_task_seconds=ticks*.05,native_success_latched=False,
            native_terminal_failure_latched=False)
        record=native_completion_record(identity,process_id=42,success=False,fatal=False,ticks=ticks,
            initial_conditions=dict(metric_only=True,policy_use_allowed=False),alignment_receipt=alignment)
        record.update(cleanup_completed=True,evaluation_process_pid=42,entrypoint_sha256=plan['child_source']['sha256'],
            config_sha256=plan['config']['sha256'],manifest_sha256=plan['manifest']['sha256'],
            source_row_sha256=identity['source_row_sha256'],frozen_input_lock=plan['frozen_inputs'],video=None,
            compute_seconds=.01,wall_seconds=.02,simulation_seconds=ticks*.05,predictor='cv',
            continuation_sha256='a'*64,continuation_identity=dict(sha256='a'*64),
            unexecuted_candidates_labelled=False,first_selection_policy='geometry',downstream_selection_policy='geometry')
        child=output/'cases'/job['job_id']/'attempt0000/case.json';self.write(child,record)
        log=child.parent.parent/'attempt0000.log';log.write_text('CPU fixture, never launched')
        command=['bash','scripts/run_local_sim.sh',plan['child_source']['path'],'--config',plan['config']['path'],
            '--manifest',plan['manifest']['path'],'--frozen-inputs',plan['frozen_inputs']['path'],
            '--split','validation','--task',identity['task'],'--raw-episode-id',job['source_raw_episode_id'],
            '--method','cv_geometry','--replica','0','--out',str(child.parent.relative_to(self.root))]
        record['parent_case_process_receipt']=dict(child_path=str(child.relative_to(self.root)),child_sha256=file_sha(child),
            child_process_id=42,launcher_process_id=41,command=command,log_path=str(log.relative_to(self.root)),
            log_sha256=file_sha(log),plan_sha256=plan['sha256'],parent_source_sha256=plan['parent_source']['sha256'])
        canonical=output/'results'/(job['job_id']+'.json');self.write(canonical,record)
        self.write(output/'status.json',dict(stage='complete',complete=True,completed=1375,expected=1375,
            plan_sha256=plan['sha256'],errors=[],stop_requested=False))
        return plan,child,canonical

    def test_actual_case_receipt_keeps_distinct_launcher_and_native_PID(self):
        plan,child,canonical=self.single_raw_case()
        groups,receipts,status=release._cases(release.Proof(self.root),plan)
        self.assertEqual(len(groups),1);self.assertEqual(len(receipts),1)
        # This helper test isolates one raw receipt. Public _plan separately
        # requires all 1375 jobs before this helper can release any experiment.
        record=json.loads(canonical.read_text());record['parent_case_process_receipt']['command'][-1]='wrong/output'
        canonical.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'raw launcher command differs'):
            release._cases(release.Proof(self.root),plan)

    def test_raw_child_change_cannot_be_accepted_as_policy_failure(self):
        plan,child,canonical=self.single_raw_case();data=json.loads(child.read_text());data['official_task_success']=True
        child.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            release._cases(release.Proof(self.root),plan)

    def groups(self):
        result={}
        for task,raw in self.val:
            result[(task,raw)]={method:dict(official_task_success=method!='context_value') for method in METHODS}
        return result

    def test_performance_reports_negative_differences_without_inventing_gate(self):
        result=release._performance(self.groups())
        self.assertIsNone(result['passed']);self.assertEqual(result['status'],'not_preregistered_numeric_gate')
        self.assertEqual(len(result['negative_value_geometry_differences']),14)
        self.assertFalse(result['success_module_benefit_established'])
        self.assertEqual(result['pooled']['physmani']['episodes'],275)

    def test_release_composition_uses_real_VALconfig_pointer_and_no_self_hash_cycle(self):
        plan,path,manifest,config,experiment,coverage=self.fixture();groups=self.groups()
        source=self.root/'scripts/finalize_multitask_validation_release_r10.py';source.parent.mkdir(parents=True)
        source.write_text('# temporary CPU schema fixture only')
        trace=self.root/'our_mani/multitask/trace_evidence.py';trace.parent.mkdir(parents=True);trace.write_text('# CPU fixture source pointer only')
        comparison=dict(complete=True,accepted=True,checks=[dict(accepted=True)]*10)
        out=self.base/'VAL_release.json'
        with patch.object(release,'_cases',return_value=(groups,[dict(CPU_mock_receipt=True)]*1375,dict(complete=True))),\
             patch.object(release,'require_frozen_coverage_certificate',return_value=dict(passed=True)),\
             patch.object(release,'profile_problems',return_value=[]),\
             patch.object(release,'audit_selection_usage',return_value=dict(complete=True,diagnostics_valid=True)),\
             patch.object(release,'compare_five_way',return_value=comparison):
            result=release.finalize_validation_release(self.root,path,self.pointer(coverage),out,
                canonical_validation_manifest=manifest,experiment_plan=experiment)
        self.assertTrue(result['complete'],result.get('errors'))
        self.assertTrue(result['formal_test_permitted']);self.assertEqual(result['validated_config'],self.pointer(config))
        self.assertEqual(result['config_sha256'],file_sha(config));self.assertEqual(result['recorded_initial_conditions']['pairs'],2750)
        self.assertIsNone(result['development_performance_screen_passed'])
        self.assertFalse(result['success_rate_benefit_validated'])
        self.assertNotIn(str(out.relative_to(self.root)),result['input_files_sha256'])
        self.assertTrue(out.is_file())

    def test_complete_counts_cannot_override_failed_recorded_initial_conditions(self):
        plan,path,manifest,config,experiment,coverage=self.fixture();out=self.base/'VAL_release.json'
        with patch.object(release,'_cases',return_value=(self.groups(),[],dict(complete=True))),\
             patch.object(release,'require_frozen_coverage_certificate',return_value=dict(passed=True)),\
             patch.object(release,'profile_problems',return_value=[]),\
             patch.object(release,'compare_five_way',return_value=dict(complete=True,accepted=False,checks=[{}]*10)):
            result=release.finalize_validation_release(self.root,path,self.pointer(coverage),out,
                canonical_validation_manifest=manifest,experiment_plan=experiment)
        self.assertFalse(result['formal_test_permitted']);self.assertEqual(result['stage'],'recorded_initial_conditions')
        self.assertFalse(out.exists());self.assertTrue(result['no_infrastructure_failure_scored_as_policy_negative'])

    def test_missing_coverage_proof_saves_pending_instead_of_releasing_complete_queue(self):
        plan,path,manifest,config,experiment,coverage=self.fixture();out=self.base/'VAL_release.json'
        with patch.object(release,'_cases',return_value=(self.groups(),[],dict(complete=True))),\
             patch.object(release,'require_frozen_coverage_certificate',side_effect=ValueError('pilot coverage rejected')):
            result=release.finalize_validation_release(self.root,path,self.pointer(coverage),out,
                canonical_validation_manifest=manifest,experiment_plan=experiment)
        self.assertFalse(result['complete']);self.assertEqual(result['stage'],'coverage_certificate')
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
