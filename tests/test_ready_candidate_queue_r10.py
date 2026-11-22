"""Meaningful TRAIN/CAL/geometry-VAL isolation and execution-receipt boundaries."""
from argparse import Namespace
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import ready_candidate_queue as q
from our_mani.multitask.evaluation import atomic_json, digest, file_sha
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.specs import TASK_SPECS
from scripts import run_trained_task_case_r10 as child
from scripts import run_ready_candidate_sequence_r10 as sequence
from tests.test_evaluate_multitask_case_r10 import MockBackend


class ReadyQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=child.evaluator.BASE)
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def put(self, name, value):
        path = self.root / name; atomic_json(path, value)
        return path

    def original(self, split):
        rows = []
        for index, task in enumerate(TASK_SPECS):
            count = 100 if split == 'train' else 20 if index < 9 else 19
            rows += [dict(task=task, raw_episode_id='episode' + str(i), path='/read_only/' + split)
                     for i in range(count)]
        return dict(split=split, episodes=rows)

    def test_full_original_parent_roster_is_selected_without_rewriting_rows(self):
        for split in ('train', 'validation'):
            source = self.original(split); task = next(iter(TASK_SPECS))
            rows = q.select_original_rows(source, split, [task])
            self.assertEqual(rows, [row for row in source['episodes'] if row['task'] == task])
            self.assertEqual(len(rows), 100 if split == 'train' else 20)
            rows[0]['path'] = 'altered copy'
            self.assertNotEqual(rows[0], source['episodes'][0])

    def test_missing_duplicate_test_and_invented_source_rosters_fail(self):
        task = next(iter(TASK_SPECS)); source = self.original('train')
        broken = deepcopy(source); broken['episodes'].pop()
        with self.assertRaisesRegex(ValueError, 'complete original'): q.select_original_rows(broken, 'train', [task])
        broken = deepcopy(source); broken['episodes'][1] = broken['episodes'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate'): q.select_original_rows(broken, 'train', [task])
        with self.assertRaisesRegex(ValueError, 'no TEST'): q.select_original_rows(source, 'test', [task])
        with self.assertRaisesRegex(ValueError, 'unique declared'): q.select_original_rows(source, 'train', ['fake_task'])

    def test_train_explores_once_and_cal_val_use_distinct_frozen_identities(self):
        row = dict(task='moving_basketball_in_hoop', raw_episode_id='episode104', path='/read_only')
        train = q.build_jobs([row], 'train', 'a'*64, 'b'*64, 7)
        cal = q.build_jobs([row], 'cal', 'a'*64, 'b'*64, 7)
        val = q.build_jobs([row], 'geometry_val', 'c'*64, 'd'*64, 7)
        self.assertEqual(train, q.build_jobs([row], 'train', 'a'*64, 'b'*64, 7))
        self.assertTrue(all(type(job['exploration_seed']) is int for job in train))
        self.assertTrue(all(job['exploration_seed'] is None for job in cal + val))
        self.assertTrue(all(not job['record_demo'] and job['replica'] == 0 for job in train + cal + val))
        self.assertEqual(Counter(job['identity']['method'] for job in train), Counter(q.METHODS))
        self.assertTrue({job['job_id'] for job in cal}.isdisjoint(job['job_id'] for job in val))

    def fixture_child(self):
        task = 'moving_basketball_in_hoop'
        row = dict(task=task, raw_episode_id='episode104', path='/read_only')
        config = self.put('config.json', dict(seed=3))
        manifest = self.put('manifest.json', dict(split='validation', episodes=[row]))
        files = {str(path):file_sha(path) for path in (config, manifest, Path(child.evaluator.__file__), Path(child.__file__))}
        lock = self.put('lock.json', dict(files=files))
        job = q.build_jobs([row], 'geometry_val', file_sha(config), file_sha(lock), 3)[0]
        prepared = dict(args=Namespace(method='cv_geometry', split='validation', task=task,
            raw_episode_id='episode104', replica=0, exploration_seed=None, demo_only=False,
            record_demo=False, development_demo=False), config={'seed':3}, config_path=config,
            manifest_path=manifest, manifest_sha256=file_sha(manifest), row=row, original_row=row,
            out=self.root/'case', identity=job['identity'], files=files, lock_path=lock,
            lock_sha256=file_sha(lock), r09_release={}, r10_validation_release=None,
            record_demo=False, shared_hook_plan={'protocol':{},'inputs':files},
            original_source_manifest=child.pointer(manifest))
        backend = MockBackend()
        @contextmanager
        def hooks(plan, captures):
            backend.env.reset_recovery_receipt = dict(complete=True, source_resampled=False)
            captures['simulations'].append(backend.env)
            yield
        record = child.run_prepared(prepared, backend=backend, hooks=hooks)
        plan = dict(stage='geometry_val', jobs=[job], sha256='e'*64,
            child_source=child.pointer(child.__file__), original_evaluator=child.pointer(child.evaluator.__file__),
            per_task_original_parents={task:1})
        actor_config = dict(learned_artifacts={task:{'interaction':{'sha256':'library'}}},
            continuation_identities={task:{'cv':'controller'}},
            executed_candidate_collection_scope={'interaction_library_identities':{task:'library'}})
        return plan, job, prepared['out']/'case.json', actor_config, record

    def test_actual_original_cpu_mock_evaluator_wrapper_receipts_validate(self):
        plan, job, path, config, record = self.fixture_child()
        actual = q.validate_raw_child(child.ROOT, plan, job, path, config)
        self.assertEqual(actual, record)
        self.assertTrue(actual['official_task_success'])

    def test_policy_negative_is_retained_but_changed_wrapper_receipt_rejected(self):
        plan, job, path, config, record = self.fixture_child()
        # This explicit CPU fixture exercises the independent native False branch.
        record['official_task_success'] = False
        record['alignment_receipt']['native_success_latched'] = False
        record['continuation_attempts'][0]['official_task_success'] = False
        base = child.ROOT/record['task_execution_witness']['base_evaluator_receipt']['path']
        atomic_json(base, {k:v for k,v in record.items() if k != 'task_execution_witness'}, overwrite=True)
        record['task_execution_witness']['base_evaluator_receipt']['sha256'] = file_sha(base)
        atomic_json(path, record, overwrite=True)
        self.assertFalse(q.validate_raw_child(child.ROOT, plan, job, path, config)['official_task_success'])
        record['task_execution_witness']['actual_wrapper']['sha256'] = '0'*64
        atomic_json(path, record, overwrite=True)
        with self.assertRaisesRegex(ValueError, 'wrapper/process'): q.validate_raw_child(child.ROOT, plan, job, path, config)

    def test_demo_teacher_or_changed_raw_receipts_do_not_become_critic_labels(self):
        plan, job, path, config, record = self.fixture_child()
        record['development_demo_execution'] = {'fitting_source':False}
        atomic_json(path, record, overwrite=True)
        with self.assertRaisesRegex(ValueError, 'development videos'): q.validate_raw_child(child.ROOT, plan, job, path, config)
        record.pop('development_demo_execution'); record['first_selection_policy'] = 'teacher'
        atomic_json(path, record, overwrite=True)
        with self.assertRaisesRegex(ValueError, 'first-selection'): q.validate_raw_child(child.ROOT, plan, job, path, config)
        record['first_selection_policy'] = 'geometry'; atomic_json(path, record, overwrite=True)
        base = child.ROOT/record['task_execution_witness']['base_evaluator_receipt']['path']
        base.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'): q.validate_raw_child(child.ROOT, plan, job, path, config)

    def test_formal_geometry_val_cannot_be_relabelled_cal_and_missing_jobs_cannot_fit(self):
        plan, job, path, config, record = self.fixture_child()
        with self.assertRaisesRegex(ValueError, 'replace critic CAL'): q.critic_manifest(plan, [(job,path)], child.ROOT)
        plan['stage'] = 'cal'
        with self.assertRaisesRegex(ValueError, 'complete declared queue'): q.critic_manifest(plan, [], child.ROOT)
        manifest = q.critic_manifest(plan, [(job,path)], child.ROOT)
        self.assertFalse(manifest['teacher_hold_and_development_demo_sources_used'])
        self.assertEqual(manifest['entries'][0]['parent_raw_episode_id'], 'episode104')

    def test_v1_value_preflight_counterexample_real_library_first_phase_only_head(self):
        # Regression evidence for ROOT's separately versioned runner repair.
        # Library and continuation are real; only the loaded critic is a CPU mock.
        from our_mani.multitask.factory import artifact_path, continuation_identity
        from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
        task = 'moving_basketball_in_hoop'
        config = json.loads((child.ROOT/'artifacts/framework/multitask_r10/basketball_Context_geometry_TRAIN_development_demo_v1_setup/configuration.json').read_text())
        library_path = artifact_path(child.ROOT, config['learned_artifacts'][task]['interaction'])
        library = InteractionLibrary.load(library_path)
        config['continuation_identities'][task]['cv'] = continuation_identity(child.ROOT, config, task, 'cv', library,
            CandidateConfig(**config['candidate_config']), MPCConfig(**config['mpc_config']))['sha256']
        mock_critic = SimpleNamespace(heads={'|'.join((task,'cv',library.phase_order[0])):{}}, metadata={})
        original_artifact_path = artifact_path
        def paths(root, value): return self.root/'mock_first_phase_critic.json' if value is None else original_artifact_path(root,value)
        with patch('our_mani.multitask.factory.artifact_path', side_effect=paths), patch(
                'our_mani.multitask.candidate_success.CandidateSuccessModel', return_value=mock_critic):
            with self.assertRaisesRegex(MultitaskProtocolError, 'required phase: ' + library.phase_order[1]):
                child.actor_model_preflight(dict(args=Namespace(task=task,method='cv_value'), config=config, files={}))

    def test_v2_accepts_exact_first_phase_critic_and_rejects_bad_query_population(self):
        from our_mani.multitask.interaction import InteractionLibrary
        from scripts import run_trained_task_case_v2_r10 as v2
        task = 'moving_basketball_in_hoop'
        path = child.ROOT/'artifacts/framework/multitask_r10/TRAIN_artifacts_full_goalv6_semanticv9_3100_v1'/task/'interaction.json'
        library = InteractionLibrary.load(path)
        critic = SimpleNamespace(heads={'|'.join((task,'cv',library.phase_order[0])):{}}, metadata=dict(
            continuation_selection_policy='single_value_selection_then_frozen_geometry',
            classifier_and_calibration_population='episode_first_actionable_decision_only',
            value_use='first actionable selection once per episode including fallback; all later selections geometry',
            omitted_downstream_heads=[dict(phase=phase,
                reason='unused downstream phase; frozen geometry continuation has no critic query')
                for phase in library.phase_order[1:]]))
        self.assertFalse(v2.verify_value_query_heads(library, critic, 'cv')['downstream_value_queries'])
        for key, value in (('classifier_and_calibration_population','all_retry_decisions'),
                           ('value_use','rerank_all_phases'), ('omitted_downstream_heads',[])):
            broken = deepcopy(critic); broken.metadata[key] = value
            with self.assertRaises(MultitaskProtocolError): v2.verify_value_query_heads(library, broken, 'cv')
        wrong = deepcopy(critic); wrong.heads['|'.join((task,'cv',library.phase_order[1]))] = {}
        with self.assertRaisesRegex(MultitaskProtocolError, 'exactly the first'): v2.verify_value_query_heads(library, wrong, 'cv')

    def test_sequence_runs_all_train_then_cal_with_one_shared_worker_budget(self):
        paths = [self.put(stage+'_plan.json', {'stage':stage}) for stage in ('train','cal')]
        lock = self.put('sequence_lock.json', {'files':{}})
        common = dict(config={'path':'mock','sha256':'a'*64}, frozen_inputs=child.pointer(lock),
            per_task_original_parents={'moving_basketball_in_hoop':100})
        plans = [dict(common,stage=stage) for stage in ('train','cal')]
        calls = []
        with patch.object(sequence.queue,'verify_plan',side_effect=[(plan,{'fixed':True}) for plan in plans]), patch.object(
                sequence.queue,'locked'), patch.object(sequence.queue,'run',side_effect=lambda path,workers:
                calls.append((path,workers)) or {'complete':True}):
            report = sequence.sequence(paths[0],paths[1],self.root/'sequence.json',4)
        self.assertEqual(calls,[(paths[0],4),(paths[1],4)])
        self.assertTrue(report['complete']); self.assertTrue(report['TRAIN_completed']); self.assertTrue(report['CAL_completed'])

    def test_sequence_does_not_start_cal_after_incomplete_train(self):
        paths = [self.put(stage+'_plan.json', {'stage':stage}) for stage in ('train','cal')]
        lock = self.put('sequence_lock.json', {'files':{}})
        common = dict(config={'path':'mock','sha256':'a'*64}, frozen_inputs=child.pointer(lock),
            per_task_original_parents={'moving_basketball_in_hoop':100})
        plans = [dict(common,stage=stage) for stage in ('train','cal')]
        with patch.object(sequence.queue,'verify_plan',side_effect=[(plan,{}) for plan in plans]), patch.object(
                sequence.queue,'locked'), patch.object(sequence.queue,'run',return_value={'complete':False}) as run:
            with self.assertRaisesRegex(ValueError,'TRAIN incomplete'):
                sequence.sequence(paths[0],paths[1],self.root/'sequence.json',4)
            self.assertEqual(run.call_count,1)
        report = json.loads((self.root/'sequence.json').read_text())
        self.assertFalse(report['TRAIN_completed']); self.assertFalse(report['CAL_completed']); self.assertFalse(report['complete'])


if __name__ == '__main__': unittest.main()
