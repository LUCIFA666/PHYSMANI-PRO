"""CPU boundary tests for real 1100+275 raw VAL receipt composition.

Fixtures retain every original source and every raw child/launcher receipt.
Only native model-coverage certification, native initial-state comparison,
selection diagnostics, and the independently tested baseline runtime closure
are mocked. No policy, simulator, GPU, fitting, or TEST data is executed.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_component as baseline
from our_mani.multitask import composite_validation_release as composite
from our_mani.multitask import validation_release as single
from our_mani.multitask.evaluation import METHODS, digest, file_sha, native_completion_record
from our_mani.multitask.specs import TASK_SPECS
from scripts.run_multitask_fresh_queue_r10 import build_jobs
from tests import test_multitask_baseline_component as baseline_fixtures


class CompositeValidationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=baseline.ROOT/'runtime')
        self.root = Path(self.temp.name)
        self.base = self.root/'artifacts/framework/multitask_r10'
        self.base.mkdir(parents=True)
        self.rows = []
        for task in TASK_SPECS:
            n = (18 if task in ('insert_onto_rotating_peg', 'put_rubbish_in_moving_bin_high_speed')
                 else 19 if task == 'insert_onto_rotating_peg_high_speed' else 20)
            self.rows += [dict(task=task, raw_episode_id='episode'+str(i),
                path='/source/val_package_compressed/'+task+'/'+str(i)+'/data.h5') for i in range(n)]
        self.roster = self.put('roster.json', dict(split='validation', episodes=self.rows))
        self.asset = self.put('TRAIN_asset.json', dict(CPU_fixture_only=True))
        self.profile = self.put('profile.json', dict(TRAIN_evidence=self.asset))
        permission = self.put('physical_permission.json', dict(CPU_fixture_only=True))
        self.sources = {}
        for name in ('queue.py', 'early_child.py', 'final_child.py', 'epoch_99999.pth', 'python-executable',
                     'scripts/run_aligned_gpu.sh', 'scripts/run_local_sim.sh',
                     'our_mani/multitask/composite_validation_release.py',
                     'our_mani/multitask/validation_release.py',
                     'our_mani/multitask/baseline_component.py', 'our_mani/multitask/trace_evidence.py',
                     'scripts/finalize_multitask_composite_validation_release_r10.py'):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# frozen CPU fixture execution source/data\n')
            self.sources[str(path)] = file_sha(path)
        self.early = baseline_fixtures.config()
        self.early.update(baseline_checkpoint=str(self.root/'epoch_99999.pth'),
            development_reset_profile=permission, initial_condition_tolerance_profile=self.profile,
            learned_artifacts={task:dict(geometry=self.asset, interaction=self.asset,
                cv_value=None, context_value=None) for task in TASK_SPECS})
        self.final = deepcopy(self.early)
        self.final.update(version='final', formal_comparison_ready=True, context_checkpoint=self.asset,
            continuation_identities={task:{predictor:'a'*64 for predictor in ('cv','context')}
                for task in TASK_SPECS})
        for artifacts in self.final['learned_artifacts'].values():
            artifacts.update(cv_value=self.asset, context_value=self.asset)
        self.early_config = self.put('early_config.json', self.early)
        self.final_config = self.put('final_config.json', self.final)
        self.contract = self.put('execution_contract.json', dict(frozen_utc='2026-01-01T00:00:00+00:00'))
        self.experiment = self.put('experiment.json', dict(methods=list(METHODS),
            original_sources=dict(validation=275), stopping_rule=dict(
                preregistered_before_remaining14_VAL_and_TEST=True,
                deployment_or_success_module_benefit_assumed=False,
                performance_screen_failure='Finish the fixed comparison without TEST tuning')))
        self.coverage = self.put('coverage.json', dict(input_files_sha256={}))
        common = dict(self.sources)
        for pointer in (self.roster, self.asset, self.profile, permission, self.contract, self.experiment):
            common[pointer['path']] = pointer['sha256']
        self.early_lock = self.put('early_lock.json', dict(files={**common,
            self.early_config['path']:self.early_config['sha256']}))
        self.final_lock = self.put('final_lock.json', dict(files={**common,
            self.final_config['path']:self.final_config['sha256']}))
        self.early_plan_value = self.plan(['physmani'], self.early_config, self.early_lock,
            'early_child.py', 'early')
        self.early_plan = self.put('early_plan.json', self.early_plan_value)
        self.ours_plan_value = self.plan(list(composite.OURS), self.final_config, self.final_lock,
            'final_child.py', 'ours')
        self.ours_plan = self.put('ours_plan.json', self.ours_plan_value)
        self.baseline_paths = self.raw_cases(self.early_plan_value, self.early, baseline_only=True)
        self.ours_paths = self.raw_cases(self.ours_plan_value, self.final, baseline_only=False)
        self.composition_value = dict(schema=composite.COMPOSITION, created_utc='2026-01-04T00:00:00+00:00',
            ours_queue_plan=self.ours_plan, baseline_early_queue_plan=self.early_plan,
            baseline_execution_contract=self.contract, canonical_validation_manifest=self.roster,
            expected_source_cases=275, expected_method_cases=1375, raw_identities_rewritten=False)
        self.composition_value['sha256'] = digest(self.composition_value)
        self.composition = self.put('composition.json', self.composition_value)
        self.out = self.base/'release.json'
        self.release_attempt = 0
        self.closure = dict(contract=dict(frozen_utc='2026-01-01T00:00:00+00:00',
            execution_entrypoints=dict(parent_source=self.early_plan_value['parent_source'],
                child_source=self.early_plan_value['child_source'],
                launcher_source=self.pointer(self.root/'scripts/run_aligned_gpu.sh'))),
            dependencies=self.sources, fingerprint=dict(CPU_fixture_only=True),mutable_state_permission=None)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        return self.pointer(path)

    def pointer(self, path):
        return dict(path=str(path.relative_to(self.root)), sha256=file_sha(path))

    def plan(self, methods, config, lock, child, name):
        jobs = build_jobs(self.rows, 'validation', methods, config['sha256'], lock['sha256'])
        plan = dict(version='r10_fresh_policy_queue_v1', created_utc='2026-01-02T00:00:00+00:00',
            split='validation', methods=methods, replicas=1, source_episodes=275,
            expected_cases=len(jobs), official_TEST_used_for_fitting=False,
            case_isolation='fresh_simulator_process_per_episode', config=config,
            frozen_inputs=lock, manifest=self.roster, parent_source=self.pointer(self.root/'queue.py'),
            child_source=self.pointer(self.root/child), output=str((self.base/name).relative_to(self.root)), jobs=jobs)
        plan['sha256'] = digest(plan)
        return plan

    def raw_cases(self, plan, config, *, baseline_only):
        paths = []
        for job in plan['jobs']:
            identity = job['identity']; row = job['row']; method = identity['method']
            initial = dict(metric_only=True, policy_use_allowed=False)
            if baseline_only:
                reset = dict(complete=True, permission=config['development_reset_profile'],
                    source_resampled=False, official_TEST_allowed=False)
                witness = dict(version=baseline.VERSION, execution_contract=self.contract,
                    projected_config_sha256=digest(baseline.project_baseline_config(config)),
                    full_config_identity_preserved=True, complete=True,
                    projection_helper_sha256=self.sources[str(self.root/'our_mani/multitask/baseline_component.py')],
                    calibration_binding_sha256=digest({'fixture_row':row}), split='validation', method='physmani',
                    actual_python_executable=str(self.root/'python-executable'),
                    loaded_native_files_sha256={str(self.root/'python-executable'):self.sources[str(self.root/'python-executable')]},
                    imported_dependency_files_sha256={str(self.root/'early_child.py'):self.sources[str(self.root/'early_child.py')]},
                    shared_reset_receipt=reset)
                initial.update(reset_receipt={'reset_recovery_receipt':reset}, baseline_execution_dependency_witness=witness)
            record = native_completion_record(identity, process_id=123, success=False, fatal=False, ticks=200,
                initial_conditions=initial, alignment_receipt=dict(clock=dict(ticks=200, dt=.05,
                    budget_seconds=10., simulation_seconds=10.), actual_physics_seconds=10.,
                    actual_task_seconds=10., native_success_latched=False, native_terminal_failure_latched=False))
            record.update(video=None, cleanup_completed=True, evaluation_process_pid=123,
                compute_seconds=1., wall_seconds=2., simulation_seconds=10.,
                config_sha256=plan['config']['sha256'], manifest_sha256=plan['manifest']['sha256'],
                source_row_sha256=identity['source_row_sha256'], entrypoint_sha256=plan['child_source']['sha256'],
                frozen_input_lock=dict(path=str(self.root/plan['frozen_inputs']['path']), sha256=plan['frozen_inputs']['sha256']))
            if baseline_only:
                record.update(baseline_executor=config['baseline_executor'],
                    checkpoint_sha256=self.sources[str(self.root/'epoch_99999.pth')])
            else:
                predictor = method.split('_')[0]
                record.update(predictor=predictor, continuation_sha256='a'*64,
                    continuation_identity=dict(sha256='a'*64), unexecuted_candidates_labelled=False,
                    first_selection_policy='value_once' if method.endswith('_value') else 'geometry',
                    downstream_selection_policy='geometry')
            child = self.root/plan['output']/'cases'/job['job_id']/'attempt0000/case.json'
            self.put(child, record)
            log = child.parent.parent/'native.log'; log.write_text('CPU raw fixture; native never launched\n')
            wrapper = 'scripts/run_aligned_gpu.sh' if baseline_only else 'scripts/run_local_sim.sh'
            command = ['bash', wrapper, plan['child_source']['path'], '--config', plan['config']['path'],
                '--manifest', plan['manifest']['path'], '--frozen-inputs', plan['frozen_inputs']['path'],
                '--split', 'validation', '--task', row['task'], '--raw-episode-id', row['raw_episode_id'],
                '--method', method, '--replica', '0', '--out', str(child.parent.relative_to(self.root))]
            record['parent_case_process_receipt'] = dict(child_path=str(child.relative_to(self.root)),
                child_sha256=file_sha(child), child_process_id=123, launcher_process_id=122,
                plan_sha256=plan['sha256'], parent_source_sha256=plan['parent_source']['sha256'],
                command=command, log_path=str(log.relative_to(self.root)), log_sha256=file_sha(log))
            canonical = self.root/plan['output']/'results'/(job['job_id']+'.json')
            self.put(canonical, record); paths.append(canonical)
        self.put(plan['output']+'/status.json', dict(stage='complete', complete=True,
            completed=len(plan['jobs']), expected=len(plan['jobs']), plan_sha256=plan['sha256'],
            errors=[], stop_requested=False))
        return paths

    def mocks(self, **changes):
        stack = ExitStack()
        options = dict(coverage=dict(passed=True), parity=dict(complete=True, accepted=True, checks=[{}]*10),
            selection=dict(complete=True, diagnostics_valid=True))
        options.update(changes)
        stack.enter_context(patch.object(baseline, 'verify_execution_contract', return_value=self.closure))
        stack.enter_context(patch.object(baseline, 'calibration_dependency', side_effect=lambda row:{'fixture_row':row}))
        for name, key in (('require_frozen_coverage_certificate','coverage'), ('compare_five_way','parity'),
                          ('audit_selection_usage','selection')):
            kwargs = dict(side_effect=options[key]) if isinstance(options[key], Exception) else dict(return_value=options[key])
            stack.enter_context(patch.object(composite, name, **kwargs))
        stack.enter_context(patch.object(composite, 'profile_problems', return_value=[]))
        return stack

    def run_release(self, **changes):
        self.out = self.base/('release_'+str(self.release_attempt)+'.json')
        self.release_attempt += 1
        with self.mocks(**changes):
            return composite.finalize_composite_validation_release(self.root, self.composition,
                self.coverage, self.out, canonical_validation_manifest=self.root/self.roster['path'],
                experiment_plan=self.root/self.experiment['path'])

    def reject(self, stage, **changes):
        result = self.run_release(**changes)
        self.assertFalse(result['complete'], dict(stage=result.get('stage'), errors=result.get('errors')))
        self.assertFalse(result['formal_test_permitted'])
        self.assertEqual(result['stage'], stage, result.get('errors'))
        self.assertFalse(self.out.exists())
        self.assertTrue((self.root/result['pending_report']['path']).is_file())
        return result

    def rehash_plan(self, value, name):
        value['sha256'] = digest({k:v for k,v in value.items() if k != 'sha256'})
        pointer = self.put(name, value)
        key = 'baseline_early_queue_plan' if name == 'early_plan.json' else 'ours_queue_plan'
        self.composition_value[key] = pointer
        self.composition_value['sha256'] = digest({k:v for k,v in self.composition_value.items() if k != 'sha256'})
        self.composition = self.put('composition.json', self.composition_value)

    def rebind_baseline_parent_plan(self, value):
        """Keep all other raw-chain checks valid to isolate plan-field guards."""
        for path in self.baseline_paths:
            record = json.loads(path.read_text())
            record['parent_case_process_receipt']['plan_sha256'] = value['sha256']
            self.put(path, record)
        self.put(value['output']+'/status.json', dict(stage='complete', complete=True,
            completed=275, expected=275, plan_sha256=value['sha256'], errors=[], stop_requested=False))

    def rebuild_with_frozen_experiment(self, value):
        """Make a genuinely frozen invalid rule; do not test only SHA tamper."""
        import shutil
        self.experiment = self.put('experiment.json', value)
        for which, config, baseline_only in (('early', self.early, True), ('ours', self.final, False)):
            lock_pointer = self.early_lock if baseline_only else self.final_lock
            lock = json.loads((self.root/lock_pointer['path']).read_text())
            lock['files'][self.experiment['path']] = self.experiment['sha256']
            lock_pointer = self.put(lock_pointer['path'], lock)
            plan = self.early_plan_value if baseline_only else self.ours_plan_value
            plan['frozen_inputs'] = lock_pointer
            plan['jobs'] = build_jobs(self.rows, 'validation', plan['methods'],
                plan['config']['sha256'], lock_pointer['sha256'])
            shutil.rmtree(self.root/plan['output'])
            self.rehash_plan(plan, 'early_plan.json' if baseline_only else 'ours_plan.json')
            paths = self.raw_cases(plan, config, baseline_only=baseline_only)
            if baseline_only:
                self.early_lock = lock_pointer; self.baseline_paths = paths
            else:
                self.final_lock = lock_pointer; self.ours_paths = paths

    def test_complete275_pairs_retain_actual_early_and_final_identity_and_freshly_validate(self):
        with patch.object(composite, 'validate_baseline_component', wraps=baseline.validate_baseline_component) as validator:
            result = self.run_release()
        self.assertTrue(result['complete'], result.get('errors'))
        validator.assert_called_once()
        self.assertEqual(result['completed_method_cases'], 1375)
        self.assertEqual(result['validated_config'], self.final_config)
        self.assertEqual(result['actual_baseline_config'], self.early_config)
        self.assertNotEqual(result['validated_input_lock'], result['actual_baseline_lock'])
        self.assertFalse(result['single_full_config_identity_claimed'])
        self.assertFalse(result['raw_case_identities_rewritten'])
        self.assertFalse(result['actual_R10_TEST_read'])
        self.assertIsNone(result['development_performance_screen_passed'])
        receipts = json.loads((self.root/result['canonical_cases']['path']).read_text())['entries']
        self.assertEqual(len(receipts), 1375)
        self.assertEqual(sum(row['method']=='physmani' for row in receipts), 275)
        self.assertEqual(result['recorded_initial_conditions']['pairs'], 2750)
        self.assertNotIn(str(self.out.relative_to(self.root)), result['input_files_sha256'])

    def test_missing_ours_job_cannot_be_replaced_by_baseline_or_cached_component(self):
        self.ours_plan_value['jobs'].pop()
        self.rehash_plan(self.ours_plan_value, 'ours_plan.json')
        with patch.object(composite, 'validate_baseline_component') as validator:
            self.reject('four_ours_plan')
        validator.assert_not_called()

    def test_duplicate_ours_job_is_rejected_with_unchanged_total_count(self):
        self.ours_plan_value['jobs'][-1] = deepcopy(self.ours_plan_value['jobs'][0])
        self.rehash_plan(self.ours_plan_value, 'ours_plan.json')
        self.reject('four_ours_plan')

    def test_changed_source_row_rejected_before_reading_native_records(self):
        self.ours_plan_value['jobs'][0]['row']['path'] = '/substituted/source.h5'
        self.rehash_plan(self.ours_plan_value, 'ours_plan.json')
        self.reject('four_ours_plan')

    def test_baseline_missing_canonical_case_cannot_be_hidden_by_complete_status(self):
        self.baseline_paths[0].unlink()
        self.reject('fresh_baseline_component_validation')

    def test_raw_baseline_change_cannot_be_hidden_by_cached_complete_component(self):
        # A forged earlier component.json is never used as acceptance input.
        self.put('cached_component.json', dict(schema=baseline.COMPONENT, complete=True, sources=275))
        record = json.loads(self.baseline_paths[0].read_text())
        child = self.root/record['parent_case_process_receipt']['child_path']
        raw = json.loads(child.read_text()); raw['compute_seconds'] = 12345.
        self.put(child, raw)
        self.reject('fresh_baseline_component_validation')

    def test_unknown_final_config_change_is_not_classified_as_training_metadata(self):
        final = deepcopy(self.final); final['new_training_metadata'] = True
        self.final_config = self.put('final_config.json', final)
        self.ours_plan_value['config'] = self.final_config
        lock = json.loads((self.root/self.final_lock['path']).read_text())
        lock['files'][self.final_config['path']] = self.final_config['sha256']
        self.final_lock = self.put('final_lock.json', lock)
        self.ours_plan_value['frozen_inputs'] = self.final_lock
        self.ours_plan_value['jobs'] = build_jobs(self.rows, 'validation', composite.OURS,
            self.final_config['sha256'], self.final_lock['sha256'])
        # Rebind genuine ours records under the new actual config/lock. The
        # independent baseline projection still must reject the new key.
        output = self.root/self.ours_plan_value['output']
        import shutil
        shutil.rmtree(output)
        self.rehash_plan(self.ours_plan_value, 'ours_plan.json')
        self.ours_paths = self.raw_cases(self.ours_plan_value, final, baseline_only=False)
        self.reject('fresh_baseline_component_validation')

    def test_changed_baseline_source_sha_rejected_without_outcome_selection(self):
        (self.root/'early_child.py').write_text('# executable changed after execution\n')
        self.reject('four_ours_plan')  # Final lock already binds this shared closure.

    def test_failed_model_coverage_blocks_complete1375_raw_cases(self):
        self.reject('final_model_coverage', coverage=ValueError('required learned model coverage missing'))

    def test_failed_parity_or_missing_tenth_pair_blocks_complete_counts(self):
        for parity in (dict(complete=True, accepted=False, checks=[{}]*10),
                       dict(complete=True, accepted=True, checks=[{}]*9)):
            with self.subTest(parity=parity):
                self.reject('recorded_initial_conditions', parity=parity)

    def test_selection_trace_and_original_stopping_rule_cannot_be_skipped(self):
        self.reject('selection_trace', selection=ValueError('missing executed candidate trace'))
        value = json.loads((self.root/self.experiment['path']).read_text())
        value['stopping_rule']['preregistered_before_remaining14_VAL_and_TEST'] = False
        self.rebuild_with_frozen_experiment(value)
        self.reject('preregistered_stopping_rule')

    def test_TEST_or_demo_plan_fields_are_rejected_before_baseline_reuse(self):
        for field, value in (('split','test'), ('official_TEST_used_for_fitting',True)):
            original = deepcopy(self.ours_plan_value)
            self.ours_plan_value[field] = value
            self.rehash_plan(self.ours_plan_value, 'ours_plan.json')
            self.reject('four_ours_plan')
            self.ours_plan_value = original
        self.ours_plan_value['jobs'][0]['record_demo'] = True
        self.rehash_plan(self.ours_plan_value, 'ours_plan.json')
        self.reject('four_ours_plan')

    def test_old_public_single_queue_release_still_rejects_four_methods(self):
        result = single.finalize_validation_release(self.root, self.root/self.ours_plan['path'],
            self.coverage, self.out, canonical_validation_manifest=self.root/self.roster['path'],
            experiment_plan=self.root/self.experiment['path'])
        self.assertFalse(result['complete']); self.assertEqual(result['stage'], 'queue_plan')
        self.assertFalse(self.out.exists())

    def test_baseline_launcher_pid_child_name_and_unknown_command_flags_fail_closed(self):
        path = self.baseline_paths[0]; original = path.read_text()
        for mutation in ('launcher', 'child_name', 'command'):
            record = json.loads(original); parent = record['parent_case_process_receipt']
            if mutation == 'launcher': parent.pop('launcher_process_id')
            elif mutation == 'command': parent['command'] += ['--seed','12345']
            else:
                child = self.root/parent['child_path']; other = child.with_name('unreviewed.json')
                other.write_bytes(child.read_bytes()); parent['child_path'] = str(other.relative_to(self.root))
            self.put(path, record)
            with self.subTest(mutation=mutation): self.reject('fresh_baseline_component_validation')
            path.write_text(original)

    def test_baseline_plan_noTEST_fresh_isolation_and_source_raw_are_mandatory(self):
        for mutation in ('TEST', 'isolation', 'source_raw'):
            value = deepcopy(self.early_plan_value)
            if mutation == 'TEST': value['official_TEST_used_for_fitting'] = True
            elif mutation == 'isolation': value['case_isolation'] = 'reused_simulator'
            else: value['jobs'][0]['source_raw_episode_id'] = 'different_parent'
            self.rehash_plan(value, 'early_plan.json')
            self.rebind_baseline_parent_plan(value)
            with self.subTest(mutation=mutation): self.reject('fresh_baseline_component_validation')


if __name__ == '__main__':
    unittest.main()
