"""CPU-only owner orchestration checks; every child boundary is mocked."""
from contextlib import ExitStack
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts import run_TRAIN_preclose_comparison_v1124_r10 as owner


FIRST4 = {
    'remove_cups_from_rotating_frame': ['episode102', 'episode104', 'episode111', 'episode112'],
    'remove_cups_from_rotating_frame_high_speed': ['episode1', 'episode102', 'episode104', 'episode116'],
    'insert_onto_rotating_peg': ['episode101', 'episode102', 'episode105', 'episode112'],
    'insert_onto_rotating_peg_high_speed': ['episode102', 'episode103', 'episode110', 'episode111'],
    'beat_the_rotating_buzz': ['episode10', 'episode104', 'episode105', 'episode107'],
    'beat_the_rotating_buzz_high_speed': ['episode102', 'episode103', 'episode104', 'episode107'],
}


def roster():
    return dict(split='train', episodes=[dict(task=task, raw_episode_id=raw)
        for task, raws in FIRST4.items() for raw in raws + ['episode999']])


class BoundedOwner(unittest.TestCase):
    def test_matched4_fixed_sources_order_and_no_remaining_queue(self):
        jobs = owner.build_jobs(roster())
        keys = [(r['task'], r['raw'], r['method']) for r in jobs]
        self.assertEqual(keys[:4], list(owner.CANARIES))
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 4)
        self.assertFalse(set(keys[:4]) & set(keys[4:]))
        self.assertNotIn('episode999', {r['raw'] for r in jobs})
        self.assertEqual(len({(r['task'], r['raw']) for r in jobs}), 4)
        self.assertEqual(len({r['out'] for r in jobs}), 4)
        self.assertEqual(len({r['runtime'] for r in jobs}), 4)
        self.assertTrue(all('/matched_preclose_v1124/' in r['out'] for r in jobs))
        self.assertTrue(all('/matched_preclose_v1124/' in r['runtime'] for r in jobs))
        self.assertNotIn('physmani', {r['method'] for r in jobs})
        self.assertFalse(hasattr(owner, 'await_review'))

    def test_nontrain_or_missing_canary_or_duplicate_parents_rejected(self):
        bad = roster(); bad['split'] = 'test'
        with self.assertRaises(ValueError):
            owner.build_jobs(bad)
        bad = roster()
        bad['episodes'] = [r for r in bad['episodes'] if r['raw_episode_id'] != 'episode104']
        with self.assertRaises(ValueError):
            owner.build_jobs(bad)
        bad = roster(); bad['episodes'][1] = deepcopy(bad['episodes'][0])
        with self.assertRaises(ValueError):
            owner.build_jobs(bad)

    def test_command_exact_newworker_overlay_and_check_delta(self):
        row = owner.build_jobs(roster())[0]
        native, check = owner.command(row), owner.command(row, True)
        self.assertEqual(check, native + ['--check'])
        self.assertIn(owner.WORKER, native)
        self.assertEqual(native[native.index('--overlay-lock')+1], owner.OVERLAY_LOCK)
        self.assertEqual(native[native.index('--raw-episode-id')+1], 'episode104')
        self.assertNotIn('--execute', native)

    def test_default_is_inert(self):
        with patch.object(owner.subprocess, 'Popen', side_effect=AssertionError('no child')), \
             patch.object(owner, 'prepare', side_effect=AssertionError('no prepare')), \
             patch.object(owner, 'execute', side_effect=AssertionError('no execute')), \
             patch('sys.stdout', new_callable=io.StringIO) as out:
            owner.main([])
        self.assertFalse(json.loads(out.getvalue())['Native_started'])

    def fixture(self, root):
        jobs = owner.build_jobs(roster())
        plan = dict(jobs=jobs, source_overlay_lock=dict(path=owner.OVERLAY_LOCK, sha256='locksha'),
                    remaining_queue=[])
        path = root / owner.PLAN
        owner.fresh(path, plan)
        return plan, owner.sha(path)

    def test_execute_rejects_wrongsha_and_repeated_launch_without_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'validate'), patch.object(owner, 'predecessor_quiescent', return_value={}), \
                 patch.object(owner.subprocess, 'Popen', side_effect=AssertionError('no child')):
                plan, checksum = self.fixture(root)
                with self.assertRaises(ValueError):
                    owner.execute('wrongsha')
                owner.fresh(root / owner.RUNTIME / 'launch.json', dict(already=True))
                with self.assertRaises(ValueError):
                    owner.execute(checksum)

    def test_execution_serial_all4_preserves_native_false_and_completes_without_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            def child(row, check, status):
                status['active'] = dict(job_id=row['job_id'], stage='CPU_check' if check else 'native')
                calls.append((row['job_id'], check))
                return dict(check=check)
            def completed(row, cpu, native, lock):
                self.assertTrue(cpu['check']); self.assertFalse(native['check'])
                return dict(job_id=row['job_id'], native_success=False)
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'validate'), patch.object(owner, 'predecessor_quiescent', return_value={}), \
                 patch.object(owner, 'child_phase', side_effect=child), \
                 patch.object(owner, 'completed_case', side_effect=completed), \
                 patch.object(owner.subprocess, 'Popen', side_effect=AssertionError('no real child')), \
                 patch('sys.stdout', new_callable=io.StringIO):
                plan, checksum = self.fixture(root)
                owner.execute(checksum)
            self.assertEqual(calls, [(r['job_id'], c) for r in plan['jobs'] for c in (True, False)])
            status = json.loads((root / owner.RUNTIME / 'status.json').read_text())
            self.assertEqual(len(status['completed']), 4)
            self.assertTrue(all(r['native_success'] is False for r in status['completed']))
            self.assertEqual(status['failed'], [])
            self.assertEqual(status['stage'], 'complete')

    def test_infrastructure_failure_stops_retains_previous_and_failed_case_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            def child(row, check, status):
                status['active'] = dict(job_id=row['job_id'], stage='CPU_check' if check else 'native')
                calls.append((row['job_id'], check))
                if len(calls) == 3:
                    raise RuntimeError('mock CPU infrastructure failure')
                return {}
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'validate'), patch.object(owner, 'predecessor_quiescent', return_value={}), \
                 patch.object(owner, 'child_phase', side_effect=child), \
                 patch.object(owner, 'completed_case', side_effect=lambda row,*args: dict(job_id=row['job_id'], native_success=False)), \
                 patch.object(owner.subprocess, 'Popen', side_effect=AssertionError('no real child')), \
                 patch('sys.stdout', new_callable=io.StringIO):
                plan, checksum = self.fixture(root)
                with self.assertRaisesRegex(RuntimeError, 'mock CPU'):
                    owner.execute(checksum)
            status = json.loads((root / owner.RUNTIME / 'status.json').read_text())
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(status['completed']), 1)
            self.assertEqual(status['failed'][0]['job_id'], plan['jobs'][1]['job_id'])
            self.assertEqual(status['failed'][0]['stage'], 'CPU_check')
            self.assertFalse((root / owner.RUNTIME / 'terminal.json').exists())
            self.assertTrue((root / owner.RUNTIME / 'failure.json').exists())

    def test_postprocessing_failure_keeps_case_as_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def child(row, check, status):
                status['active'] = dict(job_id=row['job_id'], stage='CPU_check' if check else 'native')
                return {}
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'validate'), patch.object(owner, 'predecessor_quiescent', return_value={}), \
                 patch.object(owner, 'child_phase', side_effect=child), \
                 patch.object(owner, 'completed_case', side_effect=ValueError('bad sidecar')), \
                 patch.object(owner.subprocess, 'Popen', side_effect=AssertionError('no real child')):
                plan, checksum = self.fixture(root)
                with self.assertRaisesRegex(ValueError, 'bad sidecar'):
                    owner.execute(checksum)
            status = json.loads((root / owner.RUNTIME / 'status.json').read_text())
            self.assertEqual(status['failed'][0]['stage'], 'result_review')
            self.assertFalse(status['failed'][0]['native_failure_inferred'])



    def worker_fixture(self, root, *, check=True):
        row = owner.build_jobs(roster())[0]
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        state = dict(completed=[], failed=[], active=None, updated_utc='now', stage='running',
                     source_overlay_lock={'path':owner.OVERLAY_LOCK, 'sha256':'source'})
        sidecar = dict(schema='worker_schema', complete=True, source_overlay_lock=state['source_overlay_lock'],
            selection=dict(task=row['task'], method=row['method'], raw_episode_id=row['raw']),
            process=dict(pid=9876, start_ticks=123), frozen_OS_environment=dict(
                outer_wrapper_pid=os.getpid(), outer_wrapper_start_ticks=ticks,
                observer_pid=9876, observer_start_ticks=123),
            execution=dict(mode='CPU_check' if check else 'new_protocol_native_TRAIN', complete=True,
                native_started=not check, GPU_started=False, actual_split='train', native_execution_ready=True))
        name = row['runtime'] + ('/CPU_check/' if check else '/native/') + 'worker_sidecar.json'
        owner.fresh(root / name, sidecar)
        return row, state, sidecar, name

    def test_child_records_actual_returncode_and_refuses_nonzero_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = owner.build_jobs(roster())[0]
            state = dict(completed=[], failed=[], active=None, updated_utc='now', stage='running')
            child = SimpleNamespace(pid=os.getpid(), wait=lambda:7, poll=lambda:7)
            with patch.object(owner, 'ROOT', root), patch.object(owner.subprocess, 'Popen', return_value=child) as spawn:
                with self.assertRaisesRegex(ValueError, 'infrastructure failure'):
                    owner.child_phase(row, True, state)
            self.assertEqual(spawn.call_count, 1)
            leaf = root / owner.RUNTIME / 'launches' / row['job_id'] / 'CPU_check'
            self.assertEqual(json.loads((leaf/'terminal.json').read_text())['returncode'], 7)
            self.assertFalse((leaf/'owner_interruption.json').exists())

    def test_interrupted_wait_records_unknown_child_exit_without_claiming_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = owner.build_jobs(roster())[0]
            state = dict(completed=[], failed=[], active=None, updated_utc='now', stage='running')
            def interrupted():
                raise KeyboardInterrupt('mock owner interruption')
            child = SimpleNamespace(pid=os.getpid(), wait=interrupted, poll=lambda:None)
            with patch.object(owner, 'ROOT', root), patch.object(owner.subprocess, 'Popen', return_value=child) as spawn:
                with self.assertRaises(KeyboardInterrupt):
                    owner.child_phase(row, False, state)
            leaf = root / owner.RUNTIME / 'launches' / row['job_id'] / 'native'
            saved = json.loads((leaf/'owner_interruption.json').read_text())
            self.assertTrue(saved['child_exit_unknown'])
            self.assertIsNone(saved['observed_returncode'])
            self.assertFalse(saved['normal_child_terminal_claimed'])
            self.assertFalse((leaf/'terminal.json').exists())
            self.assertEqual(spawn.call_count, 1)

    def test_worker_sidecar_binds_actual_wrapper_and_worker_processes(self):
        for bad in (None, 'outer_ticks', 'observer_ticks', 'source', 'incomplete', 'CPU_started_native'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row, state, sidecar, name = self.worker_fixture(root)
                if bad is not None:
                    if bad == 'outer_ticks': sidecar['frozen_OS_environment']['outer_wrapper_start_ticks'] += 1
                    if bad == 'observer_ticks': sidecar['frozen_OS_environment']['observer_start_ticks'] += 1
                    if bad == 'source': sidecar['source_overlay_lock'] = {'sha256':'wrong'}
                    if bad == 'incomplete': sidecar['complete'] = False
                    if bad == 'CPU_started_native': sidecar['execution']['native_started'] = True
                    (root/name).write_text(json.dumps(sidecar))
                child = SimpleNamespace(pid=os.getpid(), wait=lambda:0, poll=lambda:0)
                with patch.object(owner, 'ROOT', root), \
                     patch.object(owner, 'overlay_module', return_value=SimpleNamespace(SCHEMA='worker_schema')), \
                     patch.object(owner.subprocess, 'Popen', return_value=child):
                    if bad is None:
                        result = owner.child_phase(row, True, state)
                        self.assertEqual(result['process']['pid'], 9876)
                    else:
                        with self.assertRaises(ValueError):
                            owner.child_phase(row, True, state)
                leaf = root / owner.RUNTIME / 'launches' / row['job_id'] / 'CPU_check'
                self.assertEqual(json.loads((leaf/'terminal.json').read_text())['returncode'], 0)

    def test_completed_case_native_false_and_sha_cleanup_identity_binding(self):
        for bad in (None, 'sha', 'cleanup', 'identity', 'CPU_identity'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = owner.build_jobs(roster())[0]
                lock = {'path':owner.OVERLAY_LOCK,'sha256':'s'}
                case = dict(version='worker_schema_case', complete=True, cleanup_completed=True, split='train',
                    task=row['task'], method=row['method'], source_raw_episode_id=row['raw'], process_id=9876,
                    official_task_success=False, source_overlay_lock=lock, formal_score_contribution=False,
                    original_745_execution_acceptance_granted=False)
                if bad == 'cleanup': case['cleanup_completed'] = False
                if bad == 'identity': case['source_raw_episode_id'] = 'episode999'
                owner.fresh(root / row['out'] / 'case.json', case)
                owner.fresh(root / row['out'] / 'owned_hook_receipt.json',
                            dict(actual_split='train', reset=[dict(complete=True)], paths=[dict(close_complete=True)]))
                with patch.object(owner, 'ROOT', root), \
                     patch.object(owner, 'overlay_module', return_value=SimpleNamespace(SCHEMA='worker_schema')):
                    cpu = dict(execution=dict(identity={'sha256':'identity'}))
                    native = dict(process=dict(pid=9876), execution=dict(identity={'sha256':'identity'},
                        case=owner.pointer(row['out']+'/case.json'), native_success=False))
                    if bad == 'sha': native['execution']['case']['sha256'] = 'wrong'
                    if bad == 'CPU_identity': cpu['execution']['identity'] = {'sha256':'different'}
                    if bad is None:
                        self.assertIs(owner.completed_case(row, cpu, native, lock)['native_success'], False)
                    else:
                        with self.assertRaises(ValueError):
                            owner.completed_case(row, cpu, native, lock)

    def predecessor_fixture(self, root, *, live=False, bad=None):
        owner.fresh(root/owner.PREDECESSOR_PLAN, {'original':'frozen plan'})
        source = root/owner.PREDECESSOR_SOURCE
        source.parent.mkdir(parents=True, exist_ok=True); source.write_text('frozen source')
        plan_sha, source_sha = owner.sha(root/owner.PREDECESSOR_PLAN), owner.sha(source)
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]) + (0 if live else 1)
        launch = dict(pid=os.getpid(), start_ticks=ticks,
                      plan=dict(path=owner.PREDECESSOR_PLAN, sha256=plan_sha))
        completed=[]
        for index,(task,raw,method) in enumerate(owner.CANARIES):
            refs=[]
            for kind in ('case', 'sidecar'):
                name=owner.PREDECESSOR+f'/fixture/{index}_{kind}.json'
                owner.fresh(root/name, dict(complete=True))
                refs.append(dict(path=name,sha256=owner.sha(root/name)))
            completed.append(dict(task=task,raw=raw,method=method,case=refs[0],native=dict(sidecar=refs[1])))
        status=dict(completed=completed,active=None,failed=[],stage='failed',error=dict(type='InterruptedError'))
        control=dict(schema='r10_actual_controlled_pilot_owner_exit_v1121',actual_tool_exit_observed=True,
            returncode=1,owner_pid=launch['pid'],owner_start_ticks=ticks,owner_alive=False,
            controlled_end=True,native_cases_complete=4,completed_case_infrastructure_failures=0,
            remaining44_not_launched=True,normal_full48_completion_claimed=False)
        decision=dict(owner_process=dict(pid=launch['pid'],start_ticks=ticks),plan=launch['plan'],remaining44_released=False)
        if bad == 'active': status['active'] = {'job_id':'unexpected'}
        if bad == 'count': status['completed'] = completed[:3]
        if bad == 'changed_order': status['completed'] = list(reversed(completed))
        if bad == 'unobserved_exit': control['actual_tool_exit_observed'] = False
        if bad == 'queue_released': control['remaining44_not_launched'] = False
        if bad == 'sha': completed[0]['case']['sha256'] = 'wrong'
        owner.fresh(root/(owner.PREDECESSOR+'/launch.json'), launch)
        owner.fresh(root/(owner.PREDECESSOR+'/status.json'), status)
        owner.fresh(root/owner.PREDECESSOR_CONTROL, control)
        owner.fresh(root/owner.PREDECESSOR_DECISION, decision)
        return plan_sha,source_sha

    def test_predecessor_live_actual_identity_blocks_even_if_receipt_claims_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan_sha,source_sha=self.predecessor_fixture(root,live=True)
            with patch.object(owner,'ROOT',root), patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha), \
                 patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha), \
                 patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no child')):
                with self.assertRaisesRegex(ValueError,'still alive'):
                    owner.predecessor_quiescent()

    def test_predecessor_controlled_stop_binding_allfour_and_evidence_sha(self):
        for bad in (None,'active','count','changed_order','unobserved_exit','queue_released','sha'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                plan_sha,source_sha=self.predecessor_fixture(root,bad=bad)
                with patch.object(owner,'ROOT',root), patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha), \
                     patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha), \
                     patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no child')):
                    if bad is None:
                        proof=owner.predecessor_quiescent()
                        self.assertTrue(proof['process_identity_not_live'])
                        self.assertFalse(proof['natural_exit_claimed'])
                    else:
                        with self.assertRaises(ValueError):
                            owner.predecessor_quiescent()


if __name__ == '__main__':
    unittest.main()
