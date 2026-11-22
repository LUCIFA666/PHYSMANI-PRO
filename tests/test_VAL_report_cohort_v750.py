"""CPU-only checks for cohort isolation and a preserved reporting deadline."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/report_formal_VAL_halfhour_v750_r10.py'
SPEC = importlib.util.spec_from_file_location('report750_tested', SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(report, 'ROOT', self.root).start()
        self.runtime = self.root / 'runtime/framework/formal_VAL_new'
        self.old = self.root / 'runtime/framework/formal_VAL_old'
        self.controller = self.root / 'runtime/framework/new_controller'
        self.release = self.root / 'artifacts/framework/multitask_r10/new/VAL_release.json'
        self.source = dict(runtime=self.runtime, controller=self.controller,
                           validation_release=self.release, protocol_status='repair_acceptance',
                           superseded_runtime=self.old, repair_evidence=None)
        self.write(self.old / 'status.json', dict(completed={
            'old1': dict(complete=True, native_success=True),
            'old2': dict(complete=True, native_success=False)},
            failed={'cancelled': dict(interrupted=True)}, active=['stale'], GPU_slots=1))
        self.write(self.controller / 'status.json', {'stage': 'repair_fixed_TRAIN_qualification'})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def evidence(self, value=4):
        path = self.root / 'runtime/framework/repair/evidence.json'
        self.write(path, {'confirmed_archived_VAL_cases': value})
        return dict(path=str(path.relative_to(self.root)), sha256=report.sha_bytes(path.read_bytes()))

    def formal_plan(self):
        jobs = []
        for task in report.TASK_LABELS:
            count = (18 if task in ('insert_onto_rotating_peg', 'put_rubbish_in_moving_bin_high_speed')
                     else 19 if task == 'insert_onto_rotating_peg_high_speed' else 20)
            for index in range(count):
                for method in report.METHODS:
                    jobs.append(dict(job_id=task + str(index) + method, task=task,
                                     method=method, argv=['driver', '--split', 'validation']))
        self.assertEqual(len(jobs), 945)
        self.write(self.runtime / 'plan.json', {'jobs': jobs})
        self.write(self.runtime / 'status.json', dict(completed={
            jobs[0]['job_id']: dict(complete=True, native_success=True)},
            failed={'administrative': dict(interrupted=True)}, active=[], GPU_slots=0))
        return jobs

    def test_repair_mode_hides_old_rates_and_cancelled_failure_bits(self):
        # A stale release and old owner cannot enable the new formal cohort.
        self.source['runtime'] = self.old
        data = report.collect(self.source)
        self.assertEqual(data['archived'], 2)
        self.assertEqual(sum(data['completed'].values()), 0)
        self.assertEqual(data['failed'], 0)
        self.assertFalse(data['release_complete'])
        output = report.render(data, self.source, 1792468800, 1792470600, '验收中', [])
        self.assertIn('旧口径已封存2例', output)
        self.assertIn('正式新口径VAL尚未开始', output)
        self.assertNotIn('50.00%', output)
        self.assertNotIn('GPU=1', output)

    def test_frozen_evidence_can_include_naturally_finished_unharvested_cases(self):
        self.source['repair_evidence'] = self.evidence()
        data = report.collect(self.source)
        self.assertEqual(data['archived'], 4)
        self.assertEqual(data['archived_registered'], 2)
        self.assertIn('confirmed_archived_VAL_cases', data['archived_origin'])
        self.source['repair_evidence']['sha256'] = 'f' * 64
        rejected = report.collect(self.source)
        self.assertEqual(rejected['archived'], 2)
        self.assertTrue(any('SHA' in text for text in rejected['errors']))

    def test_old_namespace_cannot_be_relabelled_as_new_formal_VAL(self):
        self.source.update(runtime=self.old, protocol_status='formal_VAL')
        self.assertTrue(report.collect(self.source)['quarantined'])

    def test_new_cohort_uses_only_new_VAL_plan_and_completed_records(self):
        self.formal_plan()
        self.source['protocol_status'] = 'formal_VAL'
        data = report.collect(self.source)
        self.assertFalse(data['quarantined'])
        self.assertTrue(data['plan_valid'])
        self.assertEqual(sum(data['expected'].values()), 945)
        self.assertEqual(sum(data['completed'].values()), 1)
        self.assertEqual(sum(data['success'].values()), 1)
        self.assertEqual(data['failed'], 0)
        self.assertEqual(data['cancelled'], 1)
        output = report.render(data, self.source, 1792468800, 1792470600, '新口径', [])
        self.assertIn('旧口径另行封存2例', output)
        self.assertIn('总体完成：1/945', output)

    def test_TRAIN_plan_is_not_counted_as_VAL(self):
        jobs = self.formal_plan()
        jobs[0]['argv'][-1] = 'train'
        self.write(self.runtime / 'plan.json', {'jobs': jobs})
        self.source['protocol_status'] = 'formal_VAL'
        data = report.collect(self.source)
        self.assertFalse(data['plan_valid'])
        self.assertEqual(sum(data['completed'].values()), 0)

    def test_due_is_inherited_and_immediate_snapshots_do_not_postpone_it(self):
        now = report.parse_utc('2026-10-20T04:00:00+00:00')
        due = report.parse_utc('2026-10-20T04:08:21Z')
        state = self.root / 'runtime/framework/reporter750'
        old_state = self.root / 'runtime/framework/reporter739'
        self.write(old_state / 'chat_ready.json', {'next_update_utc': '2026-10-20T04:08:21+00:00'})
        self.assertEqual(report.initial_deadline(None, old_state, state, now, 1800), due)
        self.assertEqual(report.advance_deadline(due, now + 300, 1800), due)
        self.assertEqual(report.advance_deadline(due, due, 1800), due + 1800)
        data = report.collect(self.source)
        history = []
        state.mkdir(parents=True)
        output = self.root / 'VAL.txt'
        for stamp in (now, now + 300):
            with patch.object(report.time, 'time', return_value=stamp):
                report.save_report(state, output, data, self.source, 'a' * 64, 1800,
                                   '修复即时状态', history, next_due=due)
            ready = json.loads((state / 'chat_ready.json').read_text())
            self.assertEqual(report.parse_utc(ready['next_update_utc']), due)
            self.assertIn('旧口径已封存2例', ready['text'])
            self.assertIn('正式新口径VAL尚未开始', ready['text'])
            self.assertNotIn('成功/已完成', ready['text'])
        self.assertEqual(len(list((state / 'snapshots').glob('*.txt'))), 2)
        self.assertFalse((state / 'chat_ack.json').exists())

    def test_explicit_deadline_overrides_inherited_deadline(self):
        due = report.initial_deadline('2026-10-20T04:08:21Z', None, self.root, 0, 1800)
        self.assertEqual(report.clock_text(due), '2026-10-20 12:08:21（北京时间）')
        with self.assertRaises(ValueError):
            report.parse_utc('2026-10-20T04:08:21')

    def test_source_config_loads_optional_cohort_fields_without_reading_TEST(self):
        path = self.root / 'runtime/framework/reporter/source.json'
        self.write(path, dict(runtime=str(self.runtime), controller=str(self.controller),
            validation_release=str(self.release), protocol_status='repair_acceptance',
            superseded_runtime=str(self.old), repair_evidence=self.evidence()))
        source, digest = report.load_source(path)
        self.assertEqual(source['superseded_runtime'], self.old)
        self.assertEqual(source['protocol_status'], 'repair_acceptance')
        self.assertEqual(len(digest), 64)


if __name__ == '__main__':
    unittest.main()
