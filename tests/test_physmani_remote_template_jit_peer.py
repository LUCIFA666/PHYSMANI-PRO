"""Independent narrow CPU peer for the fixed post-load remote template.

The actual-stock class runs only in the native Python environment. It loads
the stock generated module's top level, but calls none of its functions, RPC
entry points, CUDA operations or simulator code. Historical checks use bytes.
"""
import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_jit_artifacts as jit

ROOT = Path(__file__).resolve().parents[1]
SITE = Path('/home/ubuntu/physmani-conda/envs/physmani5090/lib/python3.10/site-packages')
NATIVE_PYTHON = Path('/home/ubuntu/physmani-conda/envs/physmani5090/bin/python')


def stock_bytes():
    return {role: (SITE / relative).read_bytes() for role, relative in jit.REMOTE_SOURCES.items()}


class RemoteRecipePeerTests(unittest.TestCase):
    def test_fixed_recipe_matches_preserved_new_CPU_stock_bytes(self):
        expected, recipe = jit.stock_remote_template_recipe(stock_bytes())
        preserved = ROOT / 'artifacts/framework/multitask_r10/PhysMani_remote_template_CPU_peer_v1/new_CPU_actual_stock_generated_bytes.py'
        self.assertEqual(expected, preserved.read_bytes())
        self.assertEqual(recipe['module_name'], '_remote_module_non_scriptable')
        self.assertEqual(recipe['arguments'], jit.REMOTE_ARGUMENTS)
        self.assertIs(recipe['enable_moving_cpu_tensors_to_cuda'], True)
        codes = jit.remote_function_codes(expected, '/CPU/not_executed.py', 0)
        self.assertEqual(set(codes), {'forward', 'forward_async', '_remote_forward'})

    def test_recipe_rejects_different_scope_parameters_and_formatter(self):
        raw = stock_bytes()
        bad = dict(raw); bad.pop('nn_entry')
        with self.assertRaises(ValueError): jit.stock_remote_template_recipe(bad)
        for before, after in [(b'"*args"', b'"arg"'),
                              (b'generated_module_name, str_dict, True', b'generated_module_name, str_dict, False')]:
            self.assertIn(before, raw['instantiator'])
            bad = dict(raw); bad['instantiator'] = raw['instantiator'].replace(before, after)
            with self.assertRaises(ValueError): jit.stock_remote_template_recipe(bad)
        bad = dict(raw)
        bad['template'] = raw['template'].replace(b'if enable_moving_cpu_tensors_to_cuda', b'if True')
        with self.assertRaises(ValueError): jit.stock_remote_template_recipe(bad)

    def test_exact_temp_location_and_code_optimization_scope(self):
        directory = ROOT / 'runtime/framework/tmp/tmpCPU_peer'
        path = directory / (jit.REMOTE_NAME + '.py')
        self.assertEqual(jit.check_remote_template_path(ROOT, path, directory), path)
        for bad_path, bad_dir in [(path.with_name('other.py'), directory),
                                  (ROOT / 'other.py', directory),
                                  (directory / '../other.py', directory),
                                  (Path('/tmp/tmpCPU') / path.name, Path('/tmp/tmpCPU'))]:
            with self.assertRaises(ValueError): jit.check_remote_template_path(ROOT, bad_path, bad_dir)
        expected, _ = jit.stock_remote_template_recipe(stock_bytes())
        for optimization in (True, -1, 3, '0'):
            with self.assertRaises(ValueError): jit.remote_function_codes(expected, path, optimization)


@unittest.skipUnless(Path(sys.executable).resolve() == NATIVE_PYTHON.resolve(), 'actual native stock Python CPU test')
class ActualRemotePostloadPeerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_tempdir = tempfile.tempdir
        cls.previous_tmpdir = os.environ.get('TMPDIR')
        directory = ROOT / 'runtime/framework/tmp'
        directory.mkdir(parents=True, exist_ok=True)
        # This test is intended to be launched separately before Torch import.
        if 'torch.distributed.nn.jit.instantiator' in sys.modules:
            raise AssertionError('run actual remote peer in a fresh CPU process')
        os.environ['TMPDIR'] = str(directory); tempfile.tempdir = str(directory)
        cls.rpc_calls = []; cls.generated_calls = []
        def profiler(frame, event, arg):
            if event != 'call': return
            name = frame.f_code.co_name; module = frame.f_globals.get('__name__', '')
            if module.startswith('torch.distributed.rpc') and name in ('rpc_async', 'rpc_sync', 'remote', 'init_rpc', '_init_rpc_backend'):
                cls.rpc_calls.append((module, name))
            if module == jit.REMOTE_NAME and name in jit.REMOTE_FUNCTIONS:
                cls.generated_calls.append(name)
        cls.previous_profiler = sys.getprofile(); sys.setprofile(profiler)
        cls.torch = importlib.import_module('torch')
        cls.rpc = importlib.import_module('torch.distributed.rpc')
        cls.modules = {role: importlib.import_module(name) for role, name in dict(
            nn_entry='torch.distributed.nn', API='torch.distributed.nn.api.remote_module',
            instantiator='torch.distributed.nn.jit.instantiator',
            template='torch.distributed.nn.jit.templates.remote_module_template').items()}
        cls.module = sys.modules[jit.REMOTE_NAME]
        cls.path = Path(cls.module.__file__)
        cls.payload = cls.path.read_bytes()
        cls.dependencies = {str(Path(module.__file__).resolve()): jit.file_sha(module.__file__)
                            for module in cls.modules.values()}
        cls.dependencies[str(Path(jit.__file__).resolve())] = jit.file_sha(jit.__file__)
        if cls.torch.cuda.is_initialized() or cls.rpc._is_current_rpc_agent_set():
            raise AssertionError('unexpected CUDA/RPC initialization')

    @classmethod
    def tearDownClass(cls):
        sys.setprofile(cls.previous_profiler)
        tempfile.tempdir = cls.previous_tempdir
        if cls.previous_tmpdir is None: os.environ.pop('TMPDIR', None)
        else: os.environ['TMPDIR'] = cls.previous_tmpdir
        if cls.rpc_calls or cls.generated_calls or cls.torch.cuda.is_initialized() or cls.rpc._is_current_rpc_agent_set():
            raise AssertionError('CPU peer called generated/RPC functions or initialized CUDA')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='remote_JIT_peer_CPU_', dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup)
        self.profile = dict(schema=jit.PROFILE, cache_paths=jit.cache_paths(ROOT),
                            fresh_empty_initial_namespace=True, stock_compile_autotune_static_flags_unchanged=True)
        self.identity = {'actual_stock_remote_postload_CPU_peer': self._testMethodName}
        self.session = jit.ArtifactSession(ROOT, Path(self.temp.name)/'case', self.profile,
                                           dict(self.dependencies), identity=self.identity)
        self.assertEqual(self.path.read_bytes(), self.payload)

    def authorize(self):
        return self.session.authorize_python(jit.REMOTE_NAME, self.module, self.path, jit.sha_bytes(self.path.read_bytes()))

    def validate(self, event):
        return jit.validate_event(ROOT, event, self.profile, self.dependencies)

    def test_actual_stock_postload_repeat_and_historical_bindings_without_live_source(self):
        binding = self.authorize()
        self.assertEqual(binding, self.authorize())
        self.assertEqual(len(self.session.events), 1)
        event = self.session.events[0]
        self.validate(event)
        self.assertIs(event['inputs']['writer_observed'], False)
        self.assertIs(event['inputs']['loader_call_observed'], False)
        pointer = self.session.close()
        witness = dict(generated_python_module_bindings={jit.REMOTE_NAME: binding},
                       imported_dependency_files_sha256={str(self.path): jit.sha_bytes(self.payload)})
        # Historical validation must remain possible after process-owned temp cleanup.
        missing = self.path.with_suffix('.temporarily_hidden_CPU_peer')
        self.path.rename(missing)
        try:
            self.validate(event)
            files = jit.validate_observed_bindings(ROOT, pointer, self.profile, self.dependencies, witness,
                identity=self.identity, process_id=os.getpid())
            self.assertEqual(files, {str(self.path): jit.sha_bytes(self.payload)})
        finally: missing.rename(self.path)

    def test_source_bytes_changed_after_load_rejected_and_qualification_not_complete(self):
        self.path.write_bytes(self.payload + b'\n# CPU peer mutation\n')
        try:
            with self.assertRaises(ValueError): self.authorize()
        finally: self.path.write_bytes(self.payload)
        self.assertNotIn(id(self.module), self.session._module_authorities)
        pointer = self.session.close()
        with self.assertRaises(ValueError):
            jit.validate_execution(ROOT, pointer, self.profile, self.dependencies)

    def test_clone_registry_alias_and_loader_substitution_get_no_authority(self):
        clone = ModuleType(jit.REMOTE_NAME); clone.__dict__.update(self.module.__dict__)
        with self.assertRaises(ValueError):
            self.session.authorize_python(jit.REMOTE_NAME, clone, self.path, jit.sha_bytes(self.payload))
        with patch.object(self.modules['API'], '_NON_SCRIPTABLE_REMOTE_MODULE_MODULE', clone):
            with self.assertRaises(ValueError): self.authorize()
        with patch.object(self.module, '__loader__', object()):
            with self.assertRaises(ValueError): self.authorize()
        self.assertNotIn(id(self.module), self.session._module_authorities)

    def test_used_globals_generated_method_order_and_loaded_code_are_checked(self):
        original = self.module.forward.__code__
        replacement = next(item for item in compile('def forward(self,*args,**kwargs):\n return None\n', str(self.path), 'exec').co_consts if isinstance(item, CodeType))
        with patch.object(self.module, 'rpc', SimpleNamespace()):
            with self.assertRaises(ValueError): self.authorize()
        with patch.object(self.module, '_generated_methods', list(reversed(self.module._generated_methods))):
            with self.assertRaises(ValueError): self.authorize()
        try:
            self.module.forward.__code__ = replacement
            with self.assertRaises(ValueError): self.authorize()
        finally: self.module.forward.__code__ = original
        self.assertNotIn(id(self.module), self.session._module_authorities)

    def test_qualified_module_cannot_swap_equal_code_function_object(self):
        self.authorize()
        old = self.module.forward
        same_code = FunctionType(old.__code__, old.__globals__, old.__name__)
        same_code.__module__ = old.__module__; same_code.__qualname__ = old.__qualname__
        with patch.object(self.module, 'forward', same_code), patch.object(self.module, '_generated_methods', [self.module.forward_async, same_code]):
            with self.assertRaises(ValueError): self.authorize()
        pointer = self.session.close()
        with self.assertRaises(ValueError): jit.validate_execution(ROOT, pointer, self.profile, self.dependencies)

    def test_unbound_stock_source_and_unknown_registry_name_rejected(self):
        self.session.dependencies.pop(str(Path(self.modules['template'].__file__).resolve()))
        with self.assertRaises(ValueError): self.authorize()
        with self.assertRaises(ValueError):
            self.session.authorize_python('_remote_module_other', self.module, self.path, jit.sha_bytes(self.payload))
        self.assertNotIn(id(self.module), self.session._module_authorities)

    def test_historical_scope_code_owner_path_and_observed_binding_tamper_rejected(self):
        binding = self.authorize(); event = self.session.events[0]
        for field, value in [('writer_observed', True), ('loader_call_observed', True), ('validation_scope', 'preexec_writer_trace')]:
            changed = deepcopy(event); changed['inputs'][field] = value
            with self.assertRaises(ValueError): self.validate(changed)
        changed = deepcopy(event)
        changed['result']['function_codes']['forward']['actual'] = changed['result']['function_codes']['forward_async']['actual']
        with self.assertRaises(ValueError): self.validate(changed)
        changed = deepcopy(event); changed['source_owner']['qualified_name'] = 'instantiate_scriptable_remote_module_template'
        with self.assertRaises(ValueError): self.validate(changed)
        changed = deepcopy(event); changed['inputs']['path'] = str(ROOT/'runtime/framework/tmp/other.py')
        with self.assertRaises(ValueError): self.validate(changed)
        pointer = self.session.close()
        witness = dict(generated_python_module_bindings={jit.REMOTE_NAME: binding}, imported_dependency_files_sha256={})
        with self.assertRaises(ValueError):
            jit.validate_observed_bindings(ROOT, pointer, self.profile, self.dependencies, witness,
                identity=self.identity, process_id=os.getpid())


if __name__ == '__main__': unittest.main()
