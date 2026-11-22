"""Synthetic orchestration/source-archive checks; no Native, images or labels."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dual_cf_queue', ROOT/'scripts/run_current_cube_dual_crop_capture_queue_20261104_v1.py')
queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
from our_mani.local_mpc_20261103 import current_cube_dual_crop_capture_v3 as core


def plan():
    return json.loads((ROOT/'config/framework/current_cube_dual_crop_CF_pilot40_cap12_20261104_v3/plan.json').read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def closed_prefix(base):
    p = plan(); p['source_sha256'] = {}; p['external_source_sha256'] = {}
    pp = base/'plan.json'; save(pp, p); digest = queue.sha(pp)
    root = base/'capture'; first = p['parents'][0]; ident = first['parent_id']; case = root/ident
    source = {'robot_source': 'synthetic'}
    save(case/'run_inputs.json', dict(parent_id=ident, plan_sha256=digest,
        source_sha256={}, external_source_sha256={}, Native_sources_sha256=source))
    shots = [dict(scene_id=ident+'/'+s['shot_id'], shot_id=s['shot_id'], condition=s['condition'],
        condition_detail=s['condition_detail'], position_band=s['position_band'],
        independent_scene=True, physics_ticks=0) for s in first['shots']]
    save(case/'result.json', dict(parent_id=ident, complete=True, native_closed=True,
        source_unchanged=True, controlled_physics_ticks=0, shots=shots,
        bootstrap_legal_Native_baseline_restored=True, Native_sources_sha256_after=source))
    save(root/(ident+'.owner.actual_exit.json'), dict(actual_process_returncode=0,
        actual_wait_completed=True, stop_reason=None, source_unchanged=True,
        parent_id=ident, output=str(case.resolve())))
    save(case/'source_snapshot/manifest.json', {'files': []})
    (case/'source_snapshot/plan.json').write_bytes(pp.read_bytes())
    return root, p, digest


class DualCFQueueTests(unittest.TestCase):
    def test_preparation_execute_refuses_before_child_or_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            output = Path(temp)/'notcreated'
            args = ['--plan', str(ROOT/'config/framework/current_cube_dual_crop_CF_pilot40_cap12_20261104_v3/plan.json'),
                '--output-root', str(output), '--execute']
            with patch.object(queue.subprocess, 'run', side_effect=AssertionError('No child')), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): queue.main(args)
            self.assertFalse(output.exists())

    def test_only_fixed_closed_first_prefix_may_continue_and_never_retry_remaining(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            root, p, digest = closed_prefix(Path(temp))
            self.assertEqual(queue.validate_prefix(root, p, digest, 1)['parent_id'], queue.EXPECTED[0])
            with self.assertRaises(FileExistsError): queue.validate_prefix(root, p, digest, 0)
            with self.assertRaises(ValueError): queue.validate_prefix(root, p, digest, 2)
            (root/p['parents'][1]['parent_id']).mkdir()
            with self.assertRaises(FileExistsError): queue.validate_prefix(root, p, digest, 1)

    def test_typed_exit_and_same_plan_archive_required_before_prefix_acceptance(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            root, p, digest = closed_prefix(Path(temp)); ident = p['parents'][0]['parent_id']
            owner = root/(ident+'.owner.actual_exit.json'); original = json.loads(owner.read_text())
            wrong = dict(original, actual_process_returncode=False); save(owner, wrong)
            with self.assertRaises(ValueError): queue.validate_prefix(root, p, digest, 1)
            save(owner, original)
            (root/ident/'source_snapshot/plan.json').write_text('{}')
            with self.assertRaises(ValueError): queue.validate_prefix(root, p, digest, 1)

    def test_stage_zero_dispatches_one_then_stage_one_dispatches_only_remaining39(self):
        """Mock only launcher/admission/closed callbacks; no production certificate."""
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            base = Path(temp); pp = base/'plan.json'; p = plan()
            p.update(preparation_only=False, execution_authorized=True)
            save(pp, p); output = base/'capture'; calls = []
            def child(command, **_kwargs):
                calls.append(command[command.index('--parent-id')+1])
                return SimpleNamespace(returncode=0)
            def audit(_root, parent, *_args): return {'parent_id': parent['parent_id'], 'actual_process_returncode': 0}
            argv = ['--plan', str(pp), '--plan-sha256', queue.sha(pp), '--output-root', str(output), '--execute']
            with (patch.object(queue, 'FAMILY', base), patch.object(queue, 'verify_plan', return_value=p),
                 patch.object(core, 'check_capture_plan', return_value={'capture_ready': True}),
                 patch.object(queue, 'audit_closed', side_effect=audit),
                 patch.object(queue.subprocess, 'run', side_effect=child), redirect_stdout(io.StringIO())):
                self.assertEqual(queue.main(argv), 0)
                self.assertEqual(calls, [queue.EXPECTED[0]])
                preview = [json.loads(line) for line in (output/'first_fit_preview_queue_v3.jsonl').read_text().splitlines()]
                self.assertFalse(preview[-1]['all40_actual_closed'])
                self.assertFalse((output/'serial_queue_v3.jsonl').exists())
                self.assertEqual(queue.main(argv+['--start-index', '1']), 0)
            self.assertEqual(calls, list(queue.EXPECTED))
            final = [json.loads(line) for line in (output/'serial_queue_v3.jsonl').read_text().splitlines()]
            self.assertTrue(final[-1]['all40_actual_closed'])


if __name__ == '__main__':
    unittest.main()
