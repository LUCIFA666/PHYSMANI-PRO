"""Full TRAIN progress uses committed linked receipts, not child attempts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_multitask_r10 import full_TRAIN_progress, full_TRAIN_runtime


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


class FullTRAINProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root/'full_collection'
        cases = []
        for parent in range(1400):
            for teacher in (True, False):
                behavior = ('TRAIN_teacher_committed_waypoint' if teacher
                            else 'TRAIN_hold_motion_identification')
                cases.append(dict(case_id=f'{parent:04d}-{"teacher" if teacher else "hold"}',
                    split='train', task=f'task{parent // 100}', raw_episode_id=f'parent{parent}__{behavior}_r0',
                    parent_raw_episode_id=f'parent{parent}', parent_source_group=f'train:task{parent // 100}:parent{parent}',
                    behavior=behavior, behavior_replica=0, teacher_used=teacher,
                    context_primary_view=not teacher, interaction_eligible=teacher,
                    maximum_collection_ticks=200 if teacher else 60))
        self.plan = dict(split='train', cases=cases)
        self.plan['plan_sha256'] = digest(self.plan)
        self.plan_path = self.root/'full_plan.json'
        self.write(self.plan_path, self.plan)
        self.status = dict(active_full_TRAIN_directory='full_collection',
            active_full_TRAIN_plan=dict(path='full_plan.json',
                sha256=hashlib.sha256(self.plan_path.read_bytes()).hexdigest()))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    def commit(self, index):
        case = self.plan['cases'][index]
        attempt = self.folder/'infrastructure_attempts'/case['case_id']/'0'
        child = attempt/'output/cases/train'/case['task']/case['case_id']/'case.json'
        entry = deepcopy(case)
        record = {k: case[k] for k in ('case_id', 'split', 'task', 'raw_episode_id', 'parent_source_group')}
        record.update(status='completed', plan_sha256=self.plan['plan_sha256'], process_id=321, entry=entry)
        self.write(child, record)
        command = ['bash', 'run_local_sim.sh', 'collector.py', '--case-id', case['case_id'],
                   '--plan', str(self.plan_path)]
        parent = dict(case_id=case['case_id'], status='completed', returncode=0,
                      process_id=123, command=command)
        self.write(attempt/'parent_receipt.json', parent)
        record['parent_process_receipt'] = dict(process_id=123, command=command,
            child_case_path=str(child.relative_to(self.root)),
            child_case_sha256=hashlib.sha256(child.read_bytes()).hexdigest())
        canonical = self.folder/'cases/train'/case['task']/case['case_id']/'case.json'
        self.write(canonical, record)
        return canonical, child, attempt/'parent_receipt.json'

    def progress(self):
        return full_TRAIN_progress(self.status, root=self.root)

    def test_partial_full_plan_counts_teacher_hold_and_parents_with_fixed_denominator(self):
        self.commit(0); self.commit(1); self.commit(2)
        # An old/partial manifest claiming completion must not set denominator.
        self.write(self.folder/'manifest.json', dict(expected=3, completed=3, complete=True))
        result = self.progress()
        self.assertEqual((result['committed'], result['teacher'], result['hold'], result['parent_sources']),
                         (3, 2, 1, 2))
        self.assertEqual((result['expected'], result['expected_teacher'], result['expected_hold']),
                         (2800, 1400, 1400))
        self.assertFalse(result['complete'])
        self.assertFalse(result['invalid_receipts'])

    def test_raw_attempt_only_is_not_counted_and_parent_pid_mismatch_is_excluded(self):
        canonical, child, parent = self.commit(0)
        canonical.unlink()
        self.assertEqual(self.progress()['committed'], 0)
        canonical, child, parent = self.commit(1)
        raw_parent = json.loads(parent.read_text()); raw_parent['process_id'] = 999
        self.write(parent, raw_parent)
        result = self.progress()
        self.assertEqual(result['committed'], 0)
        self.assertEqual(len(result['invalid_receipts']), 1)

    def test_changed_child_or_canonical_entry_is_not_counted(self):
        canonical, child, parent = self.commit(0)
        raw = json.loads(child.read_text()); raw['entry']['behavior'] = 'other'
        self.write(child, raw)
        canonical2, child2, parent2 = self.commit(1)
        raw = json.loads(canonical2.read_text()); raw['entry']['context_primary_view'] = False
        self.write(canonical2, raw)
        result = self.progress()
        self.assertEqual(result['committed'], 0)
        self.assertEqual(len(result['invalid_receipts']), 2)

    def test_infrastructure_attempts_are_reported_separately_from_canonical_errors(self):
        for index in (0, 1):
            canonical, child, parent = self.commit(index)
            canonical.unlink()
            attempt = json.loads(parent.read_text())
            attempt.update(status='infrastructure_error', returncode=1,
                           policy_failure_label_assigned=False)
            self.write(parent, attempt)
        result = self.progress()
        self.assertEqual(result['committed'], 0)
        self.assertEqual(result['infrastructure_attempt_count'], 2)
        self.assertFalse(result['invalid_receipts'])
        self.assertTrue(all(not row['policy_failure_label_assigned']
                            for row in result['infrastructure_attempts']))

    def test_wrong_frozen_plan_sha_or_invalid_json_receipt_fails_closed(self):
        canonical, child, parent = self.commit(0)
        self.status['active_full_TRAIN_plan']['sha256'] = '0' * 64
        result = self.progress()
        self.assertEqual(result['committed'], 0)
        self.assertIn('configuration_error', result)
        self.status['active_full_TRAIN_plan']['sha256'] = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.write(canonical, ['unexpected JSON array'])
        result = self.progress()
        self.assertEqual(result['committed'], 0)
        self.assertEqual(len(result['invalid_receipts']), 1)

    def test_live_worker_count_excludes_shells_other_plans_and_other_projects(self):
        proc_root = self.root/'proc'
        collector = str(self.root/'scripts/collect_multitask_training_r10.py')
        matching = ['/env/bin/python', collector, '--collect-case', '--plan', str(self.plan_path)]
        inventories = {101: matching,
            102: ['/bin/bash', collector, '--collect-case', '--plan', str(self.plan_path)],
            103: ['/env/bin/python', collector, '--collect-case', '--plan', str(self.root/'other_plan.json')],
            104: ['/env/bin/python', '/other_project/train.py'],
            105: ['/env/bin/python', collector, '--run-queue', '--plan', str(self.plan_path)]}
        for pid, args in inventories.items():
            folder = proc_root/str(pid); folder.mkdir(parents=True)
            (folder/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
        result = full_TRAIN_runtime(self.status, root=self.root, proc_root=proc_root)
        self.assertEqual(result['active_worker_count'], 1)
        self.assertEqual(result['active_worker_pids'], [101])


if __name__ == '__main__':
    unittest.main()
