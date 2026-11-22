"""CPU fake evidence only: a complete task is not a full-VAL release."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_completed_VAL_task_v833_r10 as audit
from our_mani.multitask.evaluation import case_identity, digest


METHODS = ('cv_geometry', 'context_geometry', 'physmani')
TASK = 'place_cups_on_rotating_frame_high_speed'
REASON = 'public_forecast_collision_on_RML_target_sweep'


def pointer(path, character='a'):
    return dict(path=path, sha256=character * 64)


def task_fixture():
    """Only one two-source task is done; the other 939 jobs remain pending."""
    sources = [dict(task=TASK, raw_episode_id='episode8'),
               dict(task=TASK, raw_episode_id='episode11')]
    sources += [dict(task='another_task', raw_episode_id='episode' + str(index))
                for index in range(313)]
    jobs = []
    completed = {}
    for source in sources:
        for method in METHODS:
            task, raw = source['task'], source['raw_episode_id']
            name = f'{task}_{raw}_{method}'
            jobs.append(dict(job_id=name, task=task, raw=raw, method=method,
                             evidence_runtime='runtime/framework/formal_VAL_resume_v772'))
            if task == TASK:
                leaf = f'artifacts/fake_validation/{task}/{raw}/{method}'
                completed[name] = dict(
                    complete=True, returncode=0, timed_out=False,
                    interrupted=False, automatic_retry=False,
                    case=pointer(leaf + '/case.json'),
                    admission_driver_receipt=pointer(leaf + '/admission_driver_receipt.json', 'b'),
                    native_success=False, motion_pass=False)
    plan = dict(split='validation', expected_cases=945, jobs=list(reversed(jobs)))
    status = dict(split='validation', expected=945, completed=completed, failed={})
    return plan, status, sources


def ordinary_case():
    return dict(official_task_success=False, native_terminal_failure=False,
                policy_failure=None, counted_clock=dict(ticks=199, dt=.05, seconds=9.95))


def CPU_fixture(method='context_geometry'):
    """Match 742's real VAL augmentation identity and 751's single JSON output."""
    row = dict(task=TASK, raw='episode8', method=method)
    source_row = dict(task=TASK, raw_episode_id=row['raw'], path='fake_VAL/data.h5')
    source_sha = 'c' * 64
    route = dict(configuration=pointer('fake_route/config.json', 'd'),
                 frozen_inputs=pointer('fake_route/frozen_inputs.json', 'e'))
    expected = dict(admission_driver=pointer('scripts/fake751.py'),
                    admission_guard=pointer('our_mani/fake728.py', 'b'),
                    case_worker=pointer('scripts/fake743.py', 'f'),
                    native_runtime=pointer('scripts/fake744.py', '0'),
                    original_native_runtime=pointer('scripts/fake664.py', '1'),
                    author_outcome_protocol=dict(schema='fake_author_outcome_protocol'))
    identity = case_identity(
        split='validation', task=TASK, raw_episode_id=row['raw'] + '__policy_aug_r0',
        method=method, source_row=source_row, config_sha256=route['configuration']['sha256'],
        inputs_sha256=route['frozen_inputs']['sha256'], parent_raw_episode_id=row['raw'],
        replica=0, demo_only=False, exploration_seed=None)
    identity.pop('sha256')
    identity['source_input_lock_sha256'] = source_sha
    identity['sha256'] = digest(identity)
    CPU = dict(complete=True, native_started=False, GPU_started=False,
               execution_protocol_changed=True, actual_split='validation', identity=identity,
               component_domain=dict(schema='r10_source_bound_RML_component_purpose_v640',
                   purpose='evaluation', task=TASK, method=method, split='validation',
                   raw_episode_id=row['raw'], policy_sha256='2' * 64,
                   VAL_TEST_relabelled_as_TRAIN=False), **expected)
    return CPU, deepcopy(CPU), row, expected, source_row, source_sha, route


class CompleteTaskTests(unittest.TestCase):
    def select(self, parts):
        plan, status, sources = parts
        return audit.select_complete_task(plan, status, TASK, sources)

    def test_exact_task_can_finish_before_other_tasks_and_retains_all_native_false(self):
        plan, status, sources = task_fixture()
        rows = self.select((plan, status, sources))
        self.assertEqual(len(rows), 6)
        self.assertEqual({(row['raw'], row['method']) for row in rows},
                         {(raw, method) for raw in ('episode8', 'episode11') for method in METHODS})
        self.assertEqual([(row['raw'], row['method']) for row in rows],
                         sorted(((row['raw'], row['method']) for row in rows),
                                key=lambda pair: (pair[0], METHODS.index(pair[1]))))
        self.assertTrue(all(status['completed'][row['job_id']]['native_success'] is False
                            for row in rows))
        self.assertEqual(len(plan['jobs']), 945)
        self.assertEqual(len(status['completed']), 6)

    def test_missing_pending_failed_or_dirty_selected_case_cannot_make_task_complete(self):
        for change in ('missing', 'failed', 'complete', 'returncode', 'bool_returncode',
                       'float_returncode', 'timed_out', 'interrupted', 'automatic_retry'):
            parts = task_fixture()
            status = parts[1]
            name = sorted(status['completed'])[0]
            record = status['completed'][name]
            if change == 'missing':
                del status['completed'][name]
            elif change == 'failed':
                status['failed'][name] = dict(stage='native', complete=False, returncode=1)
            elif change == 'complete':
                record['complete'] = False
            elif change in ('returncode', 'bool_returncode', 'float_returncode'):
                record['returncode'] = {'returncode': 1, 'bool_returncode': False,
                                        'float_returncode': 0.0}[change]
            else:
                record[change] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select(parts)

    def test_completed_case_and_external_receipt_require_whole_pointers(self):
        for field in ('case', 'admission_driver_receipt'):
            for change in ('missing', 'not_mapping', 'empty_path', 'missing_sha', 'invalid_sha',
                           'absolute_path', 'parent_path', 'wrong_filename'):
                parts = task_fixture()
                record = next(iter(parts[1]['completed'].values()))
                if change == 'missing':
                    del record[field]
                elif change == 'not_mapping':
                    record[field] = 'unbound.json'
                elif change == 'empty_path':
                    record[field]['path'] = ''
                elif change == 'missing_sha':
                    del record[field]['sha256']
                elif change == 'invalid_sha':
                    record[field]['sha256'] = 'z' * 64
                elif change == 'absolute_path':
                    record[field]['path'] = '/' + record[field]['path']
                elif change == 'parent_path':
                    record[field]['path'] = '../' + record[field]['path']
                else:
                    record[field]['path'] = 'artifacts/fake_validation/other.json'
                with self.subTest(field=field, change=change), self.assertRaises(ValueError):
                    self.select(parts)

    def test_source_grid_cannot_drop_add_duplicate_or_relabel_a_method(self):
        for change in ('drop_row', 'extra_raw', 'duplicate_job', 'duplicate_pair',
                       'wrong_method', 'wrong_job_id', 'duplicate_source', 'no_task_source'):
            parts = task_fixture()
            plan, status, sources = parts
            selected = [row for row in plan['jobs'] if row['task'] == TASK]
            row = selected[0]
            if change == 'drop_row':
                plan['jobs'].remove(row)
            elif change == 'extra_raw':
                row['raw'] = 'episode999'
                row['job_id'] = f"{TASK}_{row['raw']}_{row['method']}"
            elif change == 'duplicate_job':
                plan['jobs'].append(deepcopy(row))
            elif change == 'duplicate_pair':
                duplicate = deepcopy(row)
                duplicate['job_id'] = 'invented_job'
                plan['jobs'].append(duplicate)
            elif change == 'wrong_method':
                row['method'] = 'Value'
            elif change == 'wrong_job_id':
                row['job_id'] = 'some_other_case'
            elif change == 'duplicate_source':
                sources.append(deepcopy(sources[0]))
            else:
                sources[:] = [source for source in sources if source['task'] != TASK]
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select(parts)

    def test_task_selection_is_validation_only(self):
        for target in (0, 1):
            parts = task_fixture()
            parts[target]['split'] = 'test'
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.select(parts)


class ClassificationTests(unittest.TestCase):
    def test_precedence_preserves_native_success_then_current_fatal_then_policy(self):
        case = ordinary_case()
        case.update(official_task_success=True, native_terminal_failure=False,
                    policy_failure='explicit controller exception')
        case['counted_clock'] = dict(ticks=200, dt=.05, seconds=10.)
        self.assertEqual(audit.classify_case(case), 'native_success')
        case['official_task_success'] = False
        case['native_terminal_failure'] = True
        self.assertEqual(audit.classify_case(case), 'native_terminal_failure')
        case['native_terminal_failure'] = False
        self.assertEqual(audit.classify_case(case), 'explicit_policy_failure')
        case['policy_failure'] = None
        self.assertEqual(audit.classify_case(case), 'budget_exhausted_no_native_terminal')

    def test_only_nonempty_string_is_an_explicit_policy_failure(self):
        for value in (None, '', '   ', False, 0, [], {}):
            case = ordinary_case()
            case['policy_failure'] = value
            with self.subTest(value=value):
                self.assertEqual(audit.classify_case(case), 'unknown')
        case = ordinary_case()
        case['policy_failure'] = 'candidate exception'
        self.assertEqual(audit.classify_case(case), 'explicit_policy_failure')

    def test_native_flags_are_strict_booleans_and_cannot_conflict(self):
        for field in ('official_task_success', 'native_terminal_failure'):
            for value in (None, 0, 1, 'false', []):
                case = ordinary_case()
                case[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    audit.classify_case(case)
        case = ordinary_case()
        case.update(official_task_success=True, native_terminal_failure=True)
        with self.assertRaises(ValueError):
            audit.classify_case(case)

    def test_budget_requires_strict_count_full_duration_and_finite_clock(self):
        for field, value in (('ticks', 199), ('ticks', 201), ('ticks', 200.0),
                             ('ticks', True), ('dt', .1), ('dt', True),
                             ('seconds', 9.99), ('seconds', float('nan')),
                             ('seconds', float('inf')), ('seconds', True),
                             ('seconds', '10.0')):
            case = ordinary_case()
            case['counted_clock'] = dict(ticks=200, dt=.05, seconds=10.)
            case['counted_clock'][field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual(audit.classify_case(case), 'unknown')
        case = ordinary_case()
        case['counted_clock'] = dict(ticks=200, dt=.05, seconds=10.0000001)
        self.assertEqual(audit.classify_case(case), 'budget_exhausted_no_native_terminal')

    def test_missing_or_early_clock_does_not_create_a_failure_cause(self):
        for clock in (None, {}, dict(ticks=200), dict(ticks=99, dt=.05, seconds=4.95)):
            case = ordinary_case()
            if clock is None:
                del case['counted_clock']
            else:
                case['counted_clock'] = clock
            with self.subTest(clock=clock):
                self.assertEqual(audit.classify_case(case), 'unknown')


class GeometryDiagnosticTests(unittest.TestCase):
    def test_baseline_missing_geometry_is_unknown_not_zero(self):
        result = audit.geometry_diagnostics(dict(actor_metrics=dict(baseline_steps=[])), 'physmani')
        self.assertIsNone(result['final_geometry_phase'])
        self.assertIsNone(result['public_forecast_collision_rejections'])
        self.assertIsNone(result['recorded_rejection_reasons'])
        self.assertFalse(result['admission_events_recorded'])

    def test_geometry_uses_current_phase_and_counts_only_exact_false_rejections(self):
        events = [dict(accepted=False, proof=dict(reason=REASON)),
                  dict(accepted=False, proof=dict(reason=REASON)),
                  dict(accepted=True, proof=dict(reason=REASON)),
                  dict(accepted=False, proof=dict(reason=REASON + '_extra')),
                  dict(accepted=False, proof=dict(reason='another_rejection'), reason=REASON)]
        case = dict(actor_metrics=dict(
            geometry_intents=dict(current_phase='hang', phase='stale_phase'),
            physmani_geometry_executor=dict(generated_target_admissions=events)))
        result = audit.geometry_diagnostics(case, 'context_geometry')
        self.assertEqual(result['final_geometry_phase'], 'hang')
        self.assertEqual(result['public_forecast_collision_rejections'], 2)
        self.assertEqual(result['recorded_rejection_reasons'],
                         {REASON: 2, REASON + '_extra': 1, 'another_rejection': 1})
        self.assertTrue(result['admission_events_recorded'])
        self.assertNotIn('failure_cause', result)

    def test_known_empty_admissions_are_zero_but_missing_admissions_remain_unknown(self):
        case = dict(actor_metrics=dict(geometry_intents=dict(current_phase='align'),
                    physmani_geometry_executor=dict(generated_target_admissions=[])))
        result = audit.geometry_diagnostics(case, 'cv_geometry')
        self.assertEqual(result['final_geometry_phase'], 'align')
        self.assertEqual(result['public_forecast_collision_rejections'], 0)
        self.assertEqual(result['recorded_rejection_reasons'], {})
        self.assertTrue(result['admission_events_recorded'])
        del case['actor_metrics']['physmani_geometry_executor']['generated_target_admissions']
        result = audit.geometry_diagnostics(case, 'cv_geometry')
        self.assertIsNone(result['public_forecast_collision_rejections'])
        self.assertIsNone(result['recorded_rejection_reasons'])
        self.assertFalse(result['admission_events_recorded'])

    def test_malformed_admission_does_not_claim_complete_exact_counts(self):
        for event in (dict(accepted=0, proof=dict(reason=REASON)),
                      dict(proof=dict(reason=REASON)),
                      dict(accepted=False, reason=REASON),
                      dict(accepted=False, proof=dict(reason='')), 'invalid event'):
            case = dict(actor_metrics=dict(geometry_intents=dict(current_phase='hang'),
                        physmani_geometry_executor=dict(generated_target_admissions=[
                            dict(accepted=False, proof=dict(reason=REASON)), event])))
            with self.subTest(event=event):
                result = audit.geometry_diagnostics(case, 'cv_geometry')
                self.assertEqual(result['final_geometry_phase'], 'hang')
                self.assertIsNone(result['public_forecast_collision_rejections'])
                self.assertIsNone(result['recorded_rejection_reasons'])
                self.assertFalse(result['admission_events_recorded'])


class CaseOwnerTests(unittest.TestCase):
    def test_both_old_tiers_retain_real_owner_without_requiring_new_environment(self):
        for tier in ('runtime/framework/formal_VAL_v753',
                     'runtime/framework/formal_VAL_resume_v772'):
            row = dict(evidence_runtime=tier, imported_completed=True)
            with self.subTest(tier=tier), patch.object(audit.resumed, 'require_inherited_environment') as inherited:
                audit.require_case_owner(dict(owner_pid=753), dict(pid=753), row, {},
                                         expected_environment={'unused': True})
                inherited.assert_not_called()

    def test_all_tiers_reject_wrong_missing_bool_float_or_nonpositive_owner_pids(self):
        for tier in audit.OWNER_SOURCES:
            row = dict(evidence_runtime=tier, imported_completed=tier != audit.resumed.RUNTIME)
            for child_pid, owner_pid in ((124, 123), (None, 123), (True, 1), (123., 123),
                                         (123, 123.), (0, 0), (-1, -1), (False, False)):
                with self.subTest(tier=tier, child=child_pid, owner=owner_pid), \
                        patch.object(audit.resumed, 'require_inherited_environment') as inherited, \
                        self.assertRaises(ValueError):
                    audit.require_case_owner(dict(owner_pid=child_pid), dict(pid=owner_pid), row,
                                             {}, expected_environment={})
                inherited.assert_not_called()

    def test_new823_requires_exact_actual_inherited_environment_and_current_provenance(self):
        row = dict(evidence_runtime=audit.resumed.RUNTIME, imported_completed=False)
        actual, owner = dict(owner_pid=823), dict(pid=823)
        control, environment = {'control': 'fake'}, {'namespace': 'fake'}
        with patch.object(audit.resumed, 'require_inherited_environment') as inherited:
            audit.require_case_owner(actual, owner, row, control, expected_environment=environment)
            inherited.assert_called_once_with(actual, owner, control, environment)
        with patch.object(audit.resumed, 'require_inherited_environment',
                          side_effect=ValueError('wrong namespace')), self.assertRaises(ValueError):
            audit.require_case_owner(actual, owner, row, control, expected_environment=environment)

    def test_tier_and_import_provenance_cannot_be_relabelled(self):
        for tier, imported in (('runtime/fabricated', True),
                               ('runtime/framework/formal_VAL_v753', False),
                               ('runtime/framework/formal_VAL_resume_v772', False),
                               (audit.resumed.RUNTIME, True),
                               (audit.resumed.RUNTIME, 0)):
            with self.subTest(tier=tier, imported=imported), \
                    patch.object(audit.resumed, 'require_inherited_environment') as inherited, \
                    self.assertRaises(ValueError):
                audit.require_case_owner(dict(owner_pid=823), dict(pid=823),
                                         dict(evidence_runtime=tier, imported_completed=imported), {},
                                         expected_environment={})
            inherited.assert_not_called()


class CPUEvidenceTests(unittest.TestCase):
    def test_real_augmented_VAL_identity_binds_all_three_methods_and_keeps_native_false(self):
        for method in METHODS:
            parts = CPU_fixture(method)
            with self.subTest(method=method):
                identity = audit.require_CPU_evidence(*parts, policy_sha256='2' * 64)
                self.assertIs(identity, parts[0]['identity'])
                self.assertEqual(identity['raw_episode_id'], 'episode8__policy_aug_r0')
                self.assertEqual(identity['parent_raw_episode_id'], 'episode8')
                self.assertEqual(identity['source_raw_episode_id'], 'episode8')
                case = dict(ordinary_case(), identity=identity)
                case['counted_clock'] = dict(ticks=200, dt=.05, seconds=10.)
                self.assertIs(case['official_task_success'], False)
                self.assertEqual(audit.classify_case(case), 'budget_exhausted_no_native_terminal')

    def test_same_source_and_config_hashes_cannot_reuse_another_row_identity(self):
        for field, value in (('task', 'another_task'), ('method', 'physmani'),
                             ('split', 'train'), ('raw_episode_id', 'episode8'),
                             ('raw_episode_id', 'episode99__policy_aug_r0'),
                             ('parent_raw_episode_id', 'episode99'),
                             ('source_raw_episode_id', 'episode99'),
                             ('replica', 0.0), ('replica', False), ('exploration_seed', 0)):
            parts = CPU_fixture()
            CPU, _, row, _, source_row, _, route = parts
            identity = CPU['identity']
            identity[field] = value
            identity['sha256'] = digest({key: item for key, item in identity.items() if key != 'sha256'})
            parts = (CPU, deepcopy(CPU), *parts[2:])
            self.assertEqual(identity['source_row_sha256'], digest(source_row))
            self.assertEqual(identity['config_sha256'], route['configuration']['sha256'])
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.require_CPU_evidence(*parts)
        for field, value in (('task', 'another_task'), ('method', 'physmani'),
                             ('split', 'train'), ('raw_episode_id', 'episode99'),
                             ('raw_episode_id', 'episode8__policy_aug_r0'),
                             ('policy_sha256', '9' * 64)):
            parts = CPU_fixture()
            CPU = parts[0]
            CPU['component_domain'][field] = value
            with self.subTest(domain_field=field, value=value), self.assertRaises(ValueError):
                audit.require_CPU_evidence(CPU, deepcopy(CPU), *parts[2:], policy_sha256='2' * 64)

    def test_CPU_only_completion_and_exact_stdout_are_required(self):
        for field, value in (('complete', False), ('complete', 1),
                             ('native_started', True), ('native_started', 0),
                             ('GPU_started', True), ('GPU_started', 0),
                             ('execution_protocol_changed', False), ('actual_split', 'test'),
                             ('stdout', 'changed'), ('component_purpose', 'train_qualification'),
                             ('identity_sha256', '0' * 64)):
            parts = CPU_fixture()
            CPU, stdout = parts[:2]
            if field == 'stdout':
                stdout['identity']['method'] = 'physmani'
            elif field == 'component_purpose':
                CPU['component_domain']['purpose'] = value
                stdout = deepcopy(CPU)
            elif field == 'identity_sha256':
                CPU['identity']['sha256'] = value
                stdout = deepcopy(CPU)
            else:
                CPU[field] = value
                stdout = deepcopy(CPU)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.require_CPU_evidence(CPU, stdout, *parts[2:])


class EvidenceTests(unittest.TestCase):
    def test_mutable_status_snapshot_retains_original_text_without_freezing_live_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'status.json'
            original = '{"completed":{"fake_case":{"native_success":false}},"expected":945}\n'
            path.write_text(original)
            evidence = audit.Evidence(root)
            parsed, receipt, text = evidence.snapshot('status.json')
            path.write_text('{"completed":{},"expected":945,"updated":true}\n')
            self.assertEqual(parsed, json.loads(original))
            self.assertEqual(text, original)
            self.assertEqual(receipt, dict(path='status.json',
                             sha256=hashlib.sha256(original.encode('utf-8')).hexdigest()))
            self.assertEqual(hashlib.sha256(text.encode('utf-8')).hexdigest(), receipt['sha256'])
            self.assertNotIn(str(path), evidence.inputs)
            self.assertEqual(evidence.inputs, {})

    def test_stable_json_bytes_bind_recorded_sha_and_project_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'evidence.json'
            data = b'{"complete":false,"native_success":false}\n'
            path.write_bytes(data)
            checksum = hashlib.sha256(data).hexdigest()
            evidence = audit.Evidence(root)
            value, receipt = evidence.read('evidence.json', expected=checksum)
            self.assertEqual(value, dict(complete=False, native_success=False))
            self.assertEqual(receipt, dict(path='evidence.json', sha256=checksum))
            self.assertEqual(evidence.inputs, {str(path): checksum})
            self.assertEqual(evidence.pointer(receipt, {str(path): checksum}), value)
            self.assertEqual(evidence.pointer(receipt, json_value=False), data)

    def test_sha_mismatch_lock_mismatch_and_changed_prior_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'evidence.json'
            path.write_text('{"version":1}')
            evidence = audit.Evidence(root)
            _, receipt = evidence.read('evidence.json')
            with self.assertRaises(ValueError):
                evidence.read('evidence.json', expected='0' * 64)
            with self.assertRaises(ValueError):
                evidence.pointer(receipt, {})
            with self.assertRaises(ValueError):
                evidence.pointer(receipt, {str(path): '0' * 64})
            path.write_text('{"version":2}')
            with self.assertRaises(ValueError):
                evidence.read('evidence.json')
            with self.assertRaises(ValueError):
                audit.Evidence(root).pointer(receipt)

    def test_mutation_during_read_is_rejected_before_recording_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'evidence.json'
            path.write_bytes(b'{}')
            original_read = Path.read_bytes

            def mutate_after_read(candidate):
                data = original_read(candidate)
                if candidate == path:
                    candidate.write_bytes(b'{"changed":true}')
                return data

            evidence = audit.Evidence(root)
            with patch.object(Path, 'read_bytes', mutate_after_read), self.assertRaises(ValueError):
                evidence.read('evidence.json')
            self.assertEqual(evidence.inputs, {})

    def test_parent_or_symlink_escape_never_reads_outside_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / 'project'
            root.mkdir()
            outside = parent / 'outside.json'
            outside.write_text('{"outside":true}')
            (root / 'escape.json').symlink_to(outside)
            evidence = audit.Evidence(root)
            for path in ('../outside.json', 'escape.json', str(outside)):
                with self.subTest(path=path), patch.object(Path, 'read_bytes') as read, \
                        self.assertRaises(ValueError):
                    evidence.read(path)
                read.assert_not_called()
            self.assertEqual(evidence.inputs, {})


class FrozenHistoricalOwnerTests(unittest.TestCase):
    def test_both_historical_launches_match_their_exact_frozen_bytes(self):
        for tier in ('runtime/framework/formal_VAL_v753',
                     'runtime/framework/formal_VAL_resume_v772'):
            with self.subTest(tier=tier):
                receipt = pointer(tier + '/launch.json')
                files = {str((audit.ROOT / receipt['path']).resolve()): receipt['sha256']}
                audit.require_historical_owner_pointer(tier, receipt, files)

    def test_missing_or_changed_historical_launch_SHA_is_rejected(self):
        for tier in ('runtime/framework/formal_VAL_v753',
                     'runtime/framework/formal_VAL_resume_v772'):
            receipt = pointer(tier + '/launch.json')
            for files in ({}, {str((audit.ROOT / receipt['path']).resolve()): 'b' * 64}):
                with self.subTest(tier=tier, files=files), self.assertRaises(ValueError):
                    audit.require_historical_owner_pointer(tier, receipt, files)

    def test_current823_launch_is_created_after_the_frozen_control(self):
        receipt = pointer(audit.resumed.RUNTIME + '/launch.json')
        audit.require_historical_owner_pointer(audit.resumed.RUNTIME, receipt, {})

    def test_historical_tier_cannot_use_another_owners_launch_pointer(self):
        receipt = pointer('runtime/framework/formal_VAL_resume_v772/launch.json')
        files = {str((audit.ROOT / receipt['path']).resolve()): receipt['sha256']}
        with self.assertRaises(ValueError):
            audit.require_historical_owner_pointer('runtime/framework/formal_VAL_v753', receipt, files)


if __name__ == '__main__':
    unittest.main()
