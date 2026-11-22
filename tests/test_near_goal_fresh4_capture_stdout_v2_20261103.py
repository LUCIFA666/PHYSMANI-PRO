"""Mock stdout repair and source closure only; no Native, data capture or fit."""
from contextlib import redirect_stdout
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.local_mpc_20261103.depth_owner_capture import plain


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'


def module(name):
    spec = importlib.util.spec_from_file_location('independent_'+name, SCRIPTS/(name+'.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class FreshCaptureStdoutV2Tests(unittest.TestCase):
    def test_capture_nested_numpy_serializes_once_preserving_scope_and_false_qualification(self):
        cli = module('collect_near_goal_fresh4_calibration_20261103_v2')
        native_result = dict(complete=np.bool_(True), native_closed=np.bool_(True),
            source_unchanged=np.bool_(True), physics_ticks=np.int64(0),
            model_qualified=np.bool_(False), accepted_for_training=np.bool_(False),
            scope='fresh4_calibration', label_method='counterfactual_current_depth_owner',
            nested=[dict(position=np.array([.12, -.08, .94], np.float32),
                         clock=(np.float64(0.), np.int32(0), (np.bool_(True),)))],
            semantic_roles=np.array([[0, 1, 255]], np.uint8))
        output = io.StringIO()
        with patch.object(cli.producer, 'capture_parent', return_value=native_result) as capture, \
             patch.object(cli.producer, 'check_capture_plan', side_effect=AssertionError('wrong branch')) as check, \
             redirect_stdout(output):
            code = cli.main(['--capture', '--plan', 'synthetic_plan.json', '--parent-id', 'synthetic_parent', '--output', 'synthetic_output'])
        self.assertEqual(code, 0)
        capture.assert_called_once_with(cli.ROOT, Path('synthetic_plan.json'), 'synthetic_parent', Path('synthetic_output'))
        check.assert_not_called()
        self.assertIs(cli.plain, plain)
        decoded = json.loads(output.getvalue())
        self.assertEqual(decoded, plain(native_result))
        self.assertIs(decoded['model_qualified'], False); self.assertIs(decoded['accepted_for_training'], False)
        self.assertEqual(decoded['scope'], 'fresh4_calibration')
        self.assertEqual(decoded['nested'][0]['clock'], [0., 0, [True]])
        self.assertEqual(decoded['semantic_roles'], [[0, 1, 255]])
        self.assertIsInstance(native_result['complete'], np.bool_)
        self.assertIsInstance(native_result['nested'][0]['position'], np.ndarray)

    def test_default_check_never_enters_capture(self):
        cli = module('collect_near_goal_fresh4_calibration_20261103_v2')
        result = dict(complete=np.bool_(True), Native_started=False, model_qualified=False)
        output = io.StringIO()
        with patch.object(cli.producer, 'capture_parent', side_effect=AssertionError('Native forbidden')) as capture, \
             patch.object(cli.producer, 'check_capture_plan', return_value=result) as check, redirect_stdout(output):
            self.assertEqual(cli.main(['--plan', 'synthetic_plan.json']), 0)
        capture.assert_not_called(); check.assert_called_once_with(cli.ROOT, Path('synthetic_plan.json'), None)
        self.assertIs(json.loads(output.getvalue())['Native_started'], False)

    def test_nonfinite_result_is_rejected_without_capture_retry(self):
        cli = module('collect_near_goal_fresh4_calibration_20261103_v2')
        with patch.object(cli.producer, 'capture_parent', return_value=dict(value=np.array([np.nan]))) as capture, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                cli.main(['--capture', '--plan', 'synthetic_plan.json', '--parent-id', 'p', '--output', 'synthetic_output'])
        capture.assert_called_once()

    def test_original_stdout_bug_and_old_CLI_are_preserved(self):
        old = module('collect_near_goal_fresh4_calibration_20261103')
        with patch.object(old.producer, 'capture_parent', return_value=dict(value=np.array([1., 2.]))) as capture, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(TypeError):
                old.main(['--capture', '--plan', 'synthetic_plan.json', '--parent-id', 'p', '--output', 'synthetic_output'])
        capture.assert_called_once()

    def test_new_collector_AST_only_adds_frozen_plain_at_stdout(self):
        old = ast.parse((SCRIPTS/'collect_near_goal_fresh4_calibration_20261103.py').read_text())
        new = ast.parse((SCRIPTS/'collect_near_goal_fresh4_calibration_20261103_v2.py').read_text())
        class UndoPlain(ast.NodeTransformer):
            removed = 0
            unwrapped = 0
            def visit_ImportFrom(self, node):
                if node.module == 'our_mani.local_mpc_20261103.depth_owner_capture':
                    self.assert_import = [(n.name, n.asname) for n in node.names]
                    if self.assert_import != [('plain', None)]: raise AssertionError('Unexpected import')
                    self.removed += 1; return None
                return node
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == 'plain':
                    if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Name) or node.args[0].id != 'result':
                        raise AssertionError('Only final result plain conversion allowed')
                    self.unwrapped += 1; return node.args[0]
                return node
        undo = UndoPlain(); normalized = undo.visit(new)
        self.assertEqual((undo.removed, undo.unwrapped), (1, 1))
        self.assertEqual(ast.dump(normalized, include_attributes=False), ast.dump(old, include_attributes=False))

    def test_other_v2_scripts_only_repoint_paths_and_builder_version(self):
        stems = ('launch_near_goal_fresh4_calibration_20261103',
                 'run_near_goal_fresh4_calibration_queue_20261103',
                 'prepare_near_goal_fresh4_calibration_manifest_20261103')
        replacements = {stem+'_v2': stem for stem in (*stems, 'run_near_goal_fresh4_calibration_20261103',
                          'collect_near_goal_fresh4_calibration_20261103')}
        replacements['near_goal_depth_influence_fresh4_calibration_manifest_builder_20261103_v2'] = 'near_goal_depth_influence_fresh4_calibration_manifest_builder_20261103_v1'
        class UndoPaths(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    value = node.value
                    for new, old in sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True): value = value.replace(new, old)
                    return ast.copy_location(ast.Constant(value=value), node)
                return node
        for stem in stems:
            with self.subTest(stem=stem):
                a = ast.parse((SCRIPTS/(stem+'.py')).read_text())
                b = UndoPaths().visit(ast.parse((SCRIPTS/(stem+'_v2.py')).read_text()))
                self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False))
        old_shell = (SCRIPTS/'run_near_goal_fresh4_calibration_20261103.sh').read_text()
        new_shell = (SCRIPTS/'run_near_goal_fresh4_calibration_20261103_v2.sh').read_text()
        self.assertEqual(new_shell.replace('collect_near_goal_fresh4_calibration_20261103_v2.py',
                         'collect_near_goal_fresh4_calibration_20261103.py'), old_shell)

    def test_original_plan_all138_repo_and12_external_sources_unchanged(self):
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        path = ROOT/'config/framework/P3_near_goal_depth_owner_fresh4_calibration_20261103_v1.json'
        self.assertEqual(sha(path), '06fd0c6e3f1d0b8836d609205f3eea5611e5dda4b6681e6594e02ff81b99b036')
        plan = json.loads(path.read_text())
        self.assertEqual(len(plan['source_sha256']), 138); self.assertEqual(len(plan['external_source_sha256']), 12)
        for name, digest in plan['source_sha256'].items(): self.assertEqual(sha(ROOT/name), digest, name)
        for name, digest in plan['external_source_sha256'].items(): self.assertEqual(sha(name), digest, name)
        producer = 'our_mani/local_mpc_20261103/depth_owner_fresh4_calibration.py'
        self.assertEqual(plan['source_sha256'][producer], '2aaccf5f5fbe453c1118dd31345da66a95ff1fb3c99bf82fba5e6bec55c8283d')


if __name__ == '__main__':
    unittest.main()
