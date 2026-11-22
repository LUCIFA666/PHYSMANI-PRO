"""Static closure preparation tests: no native imports or external runtime."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_dependencies as inventory
from our_mani.multitask.baseline_component import GROUPS, REVIEW_SOURCE_FILES
from our_mani.multitask.specs import TASK_SPECS


class BaselineDependencyInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runtime')
        self.root = Path(self.temp.name); self.repo = self.root/'repo'; self.env = self.root/'env'
        self.python = self.file('env/bin/python3.10', b'CPU fixture interpreter bytes')
        self.runtime = self.root/'native_runtime'
        self.file('native_runtime/libcoppeliaSim.so', b'CPU native fixture bytes')
        self.file('native_runtime/runtime.sh', b'export CUDA_VISIBLE_DEVICES=0\n')
        self.file('env/lib/libstdc++.so.6', b'C++ runtime fixture')
        self.file('env/targets/x86_64-linux/lib/libcudart.so', b'CUDA runtime fixture')
        self.metadata = dict(executable=str(self.python), prefix=str(self.env), base_prefix=str(self.env),
            native_or_GPU_packages_imported=[], sys_path=[str(self.env/'lib/python3.10/site-packages')],
            sysconfig_paths={'stdlib':str(self.env/'lib/python3.10')}, packages={})
        for name in inventory.EXTERNAL_MODULES:
            path = self.env/'lib/python3.10/site-packages'/name
            self.file(path/'__init__.py', b'# fixture dependency\n')
            self.metadata['packages'][name] = dict(origin=str(path/'__init__.py'), roots=[str(path)])
        for name in inventory.PACKAGE_NAMES:
            path = self.repo/'packages'/name/name
            self.file(path/'__init__.py', b'# actual editable package\n')
            self.file(path/'extra.py', b'# must not be missed by wrong repository glob\n')
            self.metadata['packages'][name] = dict(origin=str(path/'__init__.py'), roots=[str(path)])
        for name in ('RLBench/rlbench', 'PyRep/pyrep'):
            self.file(self.repo/'third_party'/name/'__init__.py', b'# native source fixture\n')
        scene = self.repo/'third_party/RLBench/rlbench'
        self.file(scene/'task_design.ttt', b'base scene')
        for task in TASK_SPECS:
            self.file(scene/'task_ttms'/(task+'.ttm'), b'task scene '+task.encode())
        self.file('epoch_99999.pth', b'published checkpoint fixture')
        cache = self.file('cache/RN50.pt', b'RN50 weights fixture')
        link = self.root/'runtime/framework/clip/RN50.pt'; link.parent.mkdir(parents=True)
        link.symlink_to(cache)
        self.file(self.repo/'tasks/18_peract_tasks_location_bounds.json', b'{}')
        self.file(self.repo/'instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl', b'instructions')
        for name in REVIEW_SOURCE_FILES:
            self.file(name, b'# fixture reviewed local source\n')
        self.file('scripts/run_aligned_gpu.sh', b'source '+str(self.runtime/'runtime.sh').encode()+b'\n')
        diagnostic = self.file('diagnostic.json', b'{"CPU_fixture":true}')
        native = scene/'backend/scene.py'; self.file(native, b'# exact native reset source\n')
        permission = self.file('permission.json', json.dumps(dict(diagnostic_report=self.pointer(diagnostic),
            native_inputs_sha256={'third_party/RLBench/rlbench/backend/scene.py':inventory._sha(native)})).encode())
        self.config = dict(baseline_repo=str(self.repo), baseline_checkpoint='epoch_99999.pth',
            development_reset_profile=self.pointer(permission))
        self.commands = {name:str(self.file('env/bin/'+name, b'CPU launcher fixture'))
            for name in ('bash','Xvfb','xvfb-run','xauth')}

    def tearDown(self):
        self.temp.cleanup()

    def file(self, path, data):
        path = self.root/path; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        return path

    def pointer(self, path):
        return dict(path=str(path.relative_to(self.root)), sha256=inventory._sha(path))

    def run_inventory(self, **options):
        arguments = dict(python_executable=self.python, runtime_roots=[self.runtime],
            local_sources=['scripts/run_aligned_gpu.sh'], python_metadata=self.metadata)
        arguments.update(options)
        with patch.object(inventory.shutil, 'which', side_effect=lambda name:self.commands.get(name)), \
             patch.object(inventory.subprocess, 'run') as process:
            result = inventory.inventory_baseline_dependencies(self.root, self.config, **arguments)
        process.assert_not_called()
        return result

    def test_actual_three_package_trees_and_native_scenes_runtime_assets_are_bound(self):
        result = self.run_inventory()
        self.assertEqual(tuple(result['dependency_groups']), GROUPS)
        self.assertFalse(result['approved']); self.assertTrue(result['prepared_only'])
        self.assertFalse(result['native_or_GPU_launched'])
        self.assertFalse(result['runtime_inventory']['complete'])
        self.assertFalse(result['runtime_inventory']['CUDA_and_native_extensions_bound'])
        self.assertEqual(result['gaps'], [])
        self.assertEqual(len(result['complete_source_root_inventories']), 3)
        groups = result['dependency_groups']
        for name in inventory.PACKAGE_NAMES:
            self.assertIn(str(self.repo/'packages'/name/name/'extra.py'), groups['policy_and_WM_sources'])
        self.assertIn(str(self.env/'lib/libstdc++.so.6'), groups['runtime_and_extension_assets'])
        self.assertIn(str(self.runtime/'libcoppeliaSim.so'), groups['runtime_and_extension_assets'])
        self.assertIn(str(self.runtime/'runtime.sh'), groups['runtime_and_extension_assets'])
        self.assertIn(str(self.root/'cache/RN50.pt'), groups['model_assets'])
        self.assertIn(str(self.root/'diagnostic.json'), groups['calibration_inputs'])
        for task in TASK_SPECS:
            path = self.repo/'third_party/RLBench/rlbench/task_ttms'/(task+'.ttm')
            self.assertIn(str(path), groups['native_sources_and_scenes'])

    def test_cache_git_pyc_exclusions_and_late_python_file_changes_are_visible(self):
        package = self.repo/'packages/physmani_wm/physmani_wm'
        self.file(package/'__pycache__/generated.pyc', b'cache')
        self.file(package/'.git/config', b'git')
        self.file(package/'unused.pyc', b'cache')
        before = self.run_inventory()
        self.file(package/'new_late_import.py', b'# late dependency\n')
        after = self.run_inventory()
        left = before['dependency_groups']['policy_and_WM_sources']
        right = after['dependency_groups']['policy_and_WM_sources']
        self.assertEqual(set(right)-set(left), {str(package/'new_late_import.py')})
        self.assertFalse(any('.pyc' in name or '/.git/' in name for name in right))

    def test_dry_inventory_has_paths_and_sizes_but_no_fake_sha_or_approval(self):
        result = self.run_inventory(dry_run=True)
        self.assertTrue(result['dry_run']); self.assertGreater(result['total_bytes'], 0)
        self.assertTrue(all(not values for values in result['dependency_groups'].values()))
        self.assertFalse(result['runtime_inventory']['prepared_files_hashed'])
        self.assertIsNone(result['source_archive'])
        with self.assertRaisesRegex(ValueError, 'dry inventory'):
            self.run_inventory(dry_run=True, archive_directory='archive')

    def test_runtime_launcher_candidates_do_not_depend_on_preparation_shell_PATH(self):
        xvfb = self.file(self.runtime/'usr/bin/Xvfb', b'actual runtime Xvfb fixture')
        runner = self.file(self.runtime/'usr/bin/xvfb-run', b'actual runtime xvfb-run fixture')
        del self.commands['Xvfb']; del self.commands['xvfb-run']
        result = self.run_inventory()
        self.assertFalse(any(row['kind']=='launcher_program_missing' for row in result['gaps']))
        programs = result['runtime_inventory']['launcher_programs']
        self.assertIsNone(programs['Xvfb']['current_PATH_resolution'])
        self.assertEqual(programs['Xvfb']['frozen_candidates'], [str(xvfb)])
        self.assertIn(str(runner), result['dependency_groups']['runtime_and_extension_assets'])
        self.assertTrue(programs['Xvfb']['final_launcher_PATH_requires_review'])

    def test_missing_real_package_and_wrong_installed_root_never_look_complete(self):
        wrong = self.file('wrong/physmani_policy/__init__.py', b'# wrong package\n')
        data = deepcopy(self.metadata)
        data['packages']['physmani_policy'] = dict(origin=str(wrong), roots=[str(wrong.parent)])
        result = self.run_inventory(python_metadata=data, runtime_roots=[])
        self.assertIn('actual_installed_package_differs', {gap['kind'] for gap in result['gaps']})
        self.assertIn('explicit_native_system_runtime_roots_required', {gap['kind'] for gap in result['gaps']})
        self.assertFalse(result['runtime_inventory']['complete'])

    def test_permission_sha_native_hash_and_required_model_asset_tamper_are_reported(self):
        path = self.root/'permission.json'
        original = path.read_bytes()
        value = json.loads(original); value['native_inputs_sha256']['third_party/RLBench/rlbench/backend/scene.py'] = 'f'*64
        path.write_text(json.dumps(value))
        result = self.run_inventory()
        mismatches = [gap['path'] for gap in result['gaps'] if gap['kind']=='declared_pointer_SHA_differs']
        self.assertIn(str(path), mismatches)
        self.assertIn(str(self.repo/'third_party/RLBench/rlbench/backend/scene.py'), mismatches)
        path.write_bytes(original)
        (self.root/'epoch_99999.pth').unlink()
        result = self.run_inventory()
        self.assertTrue(any(gap['kind']=='missing_dependency' and gap['path'].endswith('epoch_99999.pth')
            for gap in result['gaps']))

    def test_exact_local_relative_imports_are_followed_and_dynamic_or_release_imports_require_review(self):
        source = self.file('our_mani/example/start.py', b'from .neighbor import Thing\nimport importlib\n'
            b'importlib.import_module(runtime_name)\nfrom our_mani.multitask import validation_release\n')
        neighbor = self.file('our_mani/example/neighbor.py', b'Thing = 1\n')
        release = self.file('our_mani/multitask/validation_release.py', b'# unrelated release builder\n')
        result = self.run_inventory(local_sources=[source])
        local = result['dependency_groups']['local_sources']
        self.assertIn(str(neighbor), local); self.assertNotIn(str(release), local)
        kinds = {gap['kind'] for gap in result['gaps']}
        self.assertIn('dynamic_import_requires_review', kinds)
        self.assertIn('release_import_requires_branch_review', kinds)

    def test_historical_source_archive_is_fresh_complete_and_byte_exact(self):
        result = self.run_inventory(archive_directory='new_archive')
        pointer = result['source_archive']; archive = self.root/pointer['path']
        self.assertEqual(inventory._sha(archive), pointer['sha256'])
        proof = json.loads(archive.read_text()); self.assertTrue(proof['all_inputs_verified_before_and_after'])
        expected = {name:sha for files in result['dependency_groups'].values() for name,sha in files.items()
            if Path(name).suffix in inventory.CODE_EXTENSIONS}
        self.assertEqual({row['original_path']:row['sha256'] for row in proof['entries']}, expected)
        for row in proof['entries']:
            self.assertEqual(inventory._sha(self.root/row['path']), row['sha256'])
        with self.assertRaisesRegex(ValueError, 'fresh owned archive'):
            self.run_inventory(archive_directory='new_archive')

    def test_archive_copy_or_original_source_mutation_is_not_silently_certified(self):
        actual_copy = inventory.shutil.copyfile
        changed = []
        def corrupt(source, target):
            result = actual_copy(source, target)
            if not changed:
                Path(target).write_bytes(b'corrupted archive bytes'); changed.append(True)
            return result
        with patch.object(inventory.shutil, 'copyfile', side_effect=corrupt), \
             self.assertRaisesRegex(ValueError, 'source changed'):
            self.run_inventory(archive_directory='bad_archive')

    def test_interpreter_metadata_query_never_imports_policy_or_native_package(self):
        response = SimpleNamespace(stdout=json.dumps(self.metadata))
        with patch.object(inventory.subprocess, 'run', return_value=response) as process:
            self.assertEqual(inventory.python_runtime_metadata(self.python), self.metadata)
        command = process.call_args.args[0]
        self.assertEqual(command[0], str(self.python)); self.assertEqual(command[1], '-B')
        self.assertNotIn('import torch', command[3]); self.assertNotIn('import pyrep', command[3])
        bad = deepcopy(self.metadata); bad['native_or_GPU_packages_imported'] = ['torch']
        with patch.object(inventory.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(bad))), \
             self.assertRaisesRegex(ValueError, 'prohibited native/GPU'):
            inventory.python_runtime_metadata(self.python)


if __name__ == '__main__':
    unittest.main()
