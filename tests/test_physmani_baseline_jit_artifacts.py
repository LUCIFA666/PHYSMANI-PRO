"""Actual stock Python loader CPU tests; no CUDA/simulator initialization."""
from copy import deepcopy
import importlib
import io
import itertools
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_jit_artifacts as jit
from our_mani.multitask.evaluation import file_sha

ROOT=Path(__file__).resolve().parents[1]


class ActualStockCPUObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        assert not torch.cuda.is_initialized()
        names=['torch._inductor.codecache','torch._inductor.output_code','torch._inductor.async_compile',
            'torch._inductor.runtime.compile_tasks','torch._inductor.runtime.triton_heuristics',
            'torch._inductor.runtime.static_cuda_launcher','torch._functorch._aot_autograd.autograd_cache',
            'torch._inductor.compile_worker.subproc_pool',
            'triton.runtime.build','triton.backends.nvidia.driver','triton.compiler.compiler']
        cls.modules={name:importlib.import_module(name) for name in names}
        cls.dependencies={str(Path(module.__file__).resolve()):file_sha(module.__file__) for module in cls.modules.values()}
        cls.dependencies[str(Path(jit.__file__).resolve())]=file_sha(jit.__file__)

    def test_actual_stock_options_replay_matches_v8_metadata_without_JSON_tuple_conversion(self):
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.compiler import CUDABackend,knobs
        backend=CUDABackend(GPUTarget('cuda',120,32))
        files=list((ROOT/'runtime/framework/physmani_r10_JIT_v8/triton').rglob('triton_poi_fused_cat_cos_mul_sin_0.json'))
        self.assertEqual(len(files),2)
        for path in files:
            metadata=json.loads(path.read_text())
            cm=dict(device_type='cuda',cc=120,num_warps=metadata['num_warps'],
                num_stages=metadata['num_stages'],debug=metadata['debug'])
            # The offline test interpreter lives in another environment. Bind
            # this isolated replay to the actual v8 external-lib path, rather
            # than weakening production's exact metadata path comparison.
            with patch.object(knobs.nvidia,'libdevice_path',dict(metadata['extern_libs'])['libdevice']):
                options,witness=jit.stock_cuda_compile_options(backend,cm,metadata)
            self.assertEqual(options.hash(),'156354d5b010ae72f5947ac4acc4ed730ed45a11920f1db67e91be6cd30b6f96')
            self.assertNotEqual(options.hash(),backend.parse_options(metadata).hash())
            self.assertIsInstance(options.cluster_dims,tuple)
            self.assertIsInstance(metadata['cluster_dims'],list)
            self.assertEqual(witness['request'],dict(num_warps=4,num_stages=1,debug=True,sanitize_overflow=False))
            for field in ('cluster_dims','num_warps','debug','sanitize_overflow','extern_libs'):
                changed=deepcopy(metadata);changed[field]=[9] if isinstance(changed[field],list) else 999
                with self.subTest(field=field),self.assertRaises(ValueError):
                    jit.stock_cuda_compile_options(backend,cm,changed)
            for field,value in [('device_type','cpu'),('cc',120.0),('debug',1),('num_warps',True),('num_stages',0)]:
                changed=deepcopy(cm);changed[field]=value
                with self.subTest(compile_meta=field),self.assertRaises(ValueError):
                    jit.stock_cuda_compile_options(backend,changed,metadata)
        self.assertFalse(self.torch.cuda.is_initialized())

    def _CPU_pool(self):
        return SimpleNamespace(pickler=SimpleNamespace(dumps=lambda unused:b'CPU_placeholder_not_unpickled'),
            futures_lock=threading.Lock(),write_lock=threading.Lock(),pending_futures={},
            job_id_count=itertools.count(),write_pipe=io.BytesIO(),running=True,process=SimpleNamespace(pid=1234))

    def test_actual_stock_submit_observes_exact_owned_guard_and_restores(self):
        pool=self.modules['torch._inductor.compile_worker.subproc_pool']
        session=self.session();original=pool._send_msg
        source_owner=session.source_owner(self.modules['torch._inductor.async_compile'].AsyncCompile.triton,
            'torch._inductor.async_compile','AsyncCompile.triton')
        worker=session.source_owner(self.modules['torch._inductor.runtime.compile_tasks']._worker_compile_triton,
            'torch._inductor.runtime.compile_tasks','_worker_compile_triton')
        event=session.begin('triton_parent_task_source',source_owner,dict(kernel_name='CPU_IPC_fixture',
            source=session.archive_bytes(b'CPU_source_not_executed','triton_python'),worker_source=worker,
            worker_full_trace=False,worker_PID_observed=False,
            cache_environment=dict(TORCHINDUCTOR_CACHE_DIR=self.profile['cache_paths']['inductor'],
                TRITON_CACHE_DIR=self.profile['cache_paths']['triton'])))
        session.finish(event,result=dict(parent_source_observed_before_stock_submit=True))
        session._local.task_source=session._event_pointer(event)
        session.install_pickle_hooks()
        try:
            obj=self._CPU_pool();future=pool.SubprocPool.submit(obj,lambda:None)
            self.assertTrue(future.running());self.assertTrue(obj.write_pipe.getvalue())
            messages=[e for e in session.events if e['kind']=='stock_worker_task_message']
            self.assertEqual(len(messages),1);self.assertEqual(messages[0]['status'],'complete')
            self.assertEqual(messages[0]['inputs']['actual_parent_future_identity'],id(future))
        finally:pointer=session.close()
        self.assertIs(pool._send_msg,original)
        jit.validate_execution(self.root,pointer,self.profile,self.dependencies)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_unknown_IPC_wrapper_and_foreign_guard_identity_reject_without_send(self):
        pool=self.modules['torch._inductor.compile_worker.subproc_pool']
        session=self.session();original=pool._send_msg
        session._local.task_source={'CPU_fixture':'no producer needed because call is rejected before event'}
        session.install_pickle_hooks()
        try:
            obj=self._CPU_pool()
            # Direct application call has the correct guard but no exact stock
            # submit caller. It must not emit an authorized task message.
            with self.assertRaisesRegex(ValueError,'caller is not stock submit'):
                pool._send_msg(obj.write_pipe,0,b'CPU_NOT_sent')
            bound=session._guarded_calls[(id(pool),'_send_msg')]
            old=bound['replacement'];bound['replacement']=lambda *args:None
            try:
                with self.assertRaisesRegex(ValueError,'not the exact current owned guard'):
                    pool.SubprocPool.submit(obj,lambda:None)
            finally:bound['replacement']=old
            self.assertFalse(obj.write_pipe.getvalue())
            self.assertFalse(any(e['kind']=='stock_worker_task_message' for e in session.events))
        finally:session.close(RuntimeError('expected CPU unknown wrapper rejections'))
        self.assertIs(pool._send_msg,original)
        self.assertFalse(self.torch.cuda.is_initialized())

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='JIT_actual_CPU_',dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'project';self.root.mkdir()
        self.profile=dict(schema=jit.PROFILE,cache_paths=jit.cache_paths(self.root),
            fresh_empty_initial_namespace=True,stock_compile_autotune_static_flags_unchanged=True)
        env=patch.dict(os.environ,{});env.start();self.addCleanup(env.stop)
        jit.initialize_cache_environment(self.root,self.profile)
        self.codecache=self.modules['torch._inductor.codecache']
        self.existing=set(sys.modules)
        self.addCleanup(self.cleanup_modules)

    def cleanup_modules(self):
        for key in set(sys.modules)-self.existing:
            if key.startswith('torch._inductor.runtime.compile_tasks.c'):sys.modules.pop(key,None)
        self.codecache.PyCodeCache.cache_clear()
        self.assertFalse(self.torch.cuda.is_initialized())

    def session(self,case='case0'):
        return jit.ArtifactSession(self.root,self.root/'cases'/case,self.profile,self.dependencies,
            identity={'CPU_fixture':True,'case':case})

    def test_profile_metadata_deduplicates_site_aliases_but_rejects_distinct_ABIs(self):
        prefix=self.root/'CPU_fake_environment';python=prefix/'bin/python3.10'
        python.parent.mkdir(parents=True);python.write_bytes(b'not executed, metadata fixture')
        site=prefix/'lib/python3.10/site-packages'
        for name in jit.STOCK_RELATIVE_SOURCES:
            path=site/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('not executed, stock source fixture')
        (prefix/'lib/python3.1').symlink_to(prefix/'lib/python3.10',target_is_directory=True)
        profile=jit.draft_profile(self.root,python)
        self.assertFalse(profile['reviewed_before_execution'])
        self.assertIn(str((site/'triton/runtime/build.py').resolve()),profile['source_bindings_sha256'])
        (prefix/'lib/python3.11/site-packages').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'exact interpreter site-packages root'):jit.draft_profile(self.root,python)

    def test_stock_load_nonempty_extra_exact_module_bytes_and_pure_historical_validation(self):
        session=self.session();original=self.codecache.PyCodeCache.__dict__['load_by_key_path']
        session.install()
        try:
            module=self.codecache.PyCodeCache.load('\nanswer=42\n','actual_nonempty_extra')
            self.assertEqual(module.answer,42)
            checksum=file_sha(module.__file__)
            authority=session.authorize_python(module.__name__,module,module.__file__,checksum)
            self.assertEqual(authority['key'],jit.py_code_key('\nanswer=42\n','actual_nonempty_extra'))
            again=self.codecache.PyCodeCache.load('\nanswer=42\n','actual_nonempty_extra')
            self.assertIs(again,module)
        finally:pointer=session.close()
        self.assertIs(self.codecache.PyCodeCache.__dict__['load_by_key_path'],original)
        receipt=jit.validate_execution(self.root,pointer,self.profile,self.dependencies,
            identity={'CPU_fixture':True,'case':'case0'})
        self.assertTrue(receipt['complete']);self.assertTrue(receipt['hook_restoration_completed'])
        self.assertFalse(receipt['worker_full_trace'])
        self.assertTrue(any(e['kind']=='python_module_exec' for e in receipt['events']))

    def test_unobserved_direct_load_is_rejected_before_exec_and_hook_restored(self):
        source='raise AssertionError("unobserved source executed")';key=jit.py_code_key(source)
        path=Path(self.profile['cache_paths']['inductor'])/key[1:3]/(key+'.py')
        path.parent.mkdir();path.write_text(source)
        session=self.session();session.install()
        try:
            with self.assertRaisesRegex(ValueError,'unobserved same-run'):
                self.codecache.PyCodeCache.load_by_key_path(key,str(path))
        finally:session.close(RuntimeError('expected CPU rejection'))
        self.assertFalse(jit.read_pointer(self.root,session.receipt_pointer)['complete'])

    def test_generated_module_identity_and_unknown_native_never_get_prefix_authority(self):
        session=self.session();session.install()
        try:
            module=self.codecache.PyCodeCache.load('answer=123')
            fake=type(module)(module.__name__);fake.__dict__.update(module.__dict__)
            with self.assertRaises(ValueError):session.authorize_python(module.__name__,fake,module.__file__,file_sha(module.__file__))
            with self.assertRaises(ValueError):session.authorize_native(Path(self.profile['cache_paths']['triton'])/'unknown.so','0'*64)
        finally:session.close()

    def test_earlier_complete_artifact_survives_later_failed_case_only_exact_bytes_reusable(self):
        first=self.session('failed_case');first.install()
        module=self.codecache.PyCodeCache.load('answer=7')
        path=module.__file__;key=module.key
        first.close(RuntimeError('later infrastructure fixture'))
        self.codecache.PyCodeCache.cache_clear()
        second=self.session('next_case');second.install()
        try:
            loaded=self.codecache.PyCodeCache.load_by_key_path(key,path)
            self.assertEqual(loaded.answer,7)
        finally:pointer=second.close()
        self.assertTrue(jit.validate_execution(self.root,pointer,self.profile,self.dependencies)['complete'])
        Path(path).write_text('answer=8')
        third=self.session('tampered');third.install()
        try:
            with self.assertRaisesRegex(ValueError,'unobserved same-run'):
                self.codecache.PyCodeCache.load_by_key_path(key,path)
        finally:third.close(RuntimeError('expected tamper'))

    def test_archive_and_recipe_tamper_rejected_without_executing_python(self):
        session=self.session();session.install()
        self.codecache.PyCodeCache.load('answer=42','extra')
        pointer=session.close();value=jit.read_pointer(self.root,pointer)
        event=next(e for e in value['events'] if e['kind']=='python_source_write')
        changed=deepcopy(event);changed['inputs']['key']='c'+'z'*51
        with self.assertRaises(ValueError):jit.validate_event(self.root,changed,self.profile,self.dependencies)
        archive=self.root/event['artifacts'][0]['archive']['path'];archive.write_bytes(b'raise Exception("must never execute")')
        with self.assertRaises(ValueError):jit.validate_execution(self.root,pointer,self.profile,self.dependencies)

    def test_actual_stock_C_helper_build_load_command_and_same_run_reuse_without_CUDA(self):
        compiler=Path(shutil.which(os.environ.get('CC') or 'gcc')).resolve()
        self.dependencies[str(compiler)]=file_sha(compiler)
        source='''#include <Python.h>
static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "cpu_jit_fixture", NULL, -1, NULL};
PyMODINIT_FUNC PyInit_cpu_jit_fixture(void) { return PyModule_Create(&mod); }
'''
        build=self.modules['triton.runtime.build'];session=self.session();session.install()
        try:
            module=build.compile_module_from_src(source,'cpu_jit_fixture')
            authority=session.authorize_native(module.__file__,file_sha(module.__file__))
            self.assertEqual(authority['sha256'],file_sha(module.__file__))
            again=build.compile_module_from_src(source,'cpu_jit_fixture')
            self.assertEqual(again.__file__,module.__file__)
        finally:pointer=session.close()
        receipt=jit.validate_execution(self.root,pointer,self.profile,self.dependencies)
        commands=[e['inputs']['actual_compiler_command'] for e in receipt['events'] if e['kind']=='native_helper_build']
        self.assertEqual(len(commands),1);self.assertIn('-shared',commands[0])
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_actual_component_imports_maps_and_historical_generated_bindings(self):
        from our_mani.multitask import baseline_component as component
        from our_mani.multitask.evaluation import atomic_json
        session=self.session();session.install()
        try:
            generated=self.codecache.PyCodeCache.load('answer=9001')
            # Freeze the actually imported CPU files and maps for this isolated
            # unit fixture. Production obtains these through its full contract.
            dependencies=dict(self.dependencies)
            for registry,module in list(sys.modules.items()):
                alias=component.torch_virtual_owner(registry,module)
                name=str(alias[0]) if alias else getattr(module,'__file__',None)
                if not name or not Path(name).is_absolute():continue
                path=Path(name).resolve()
                if path.suffix=='.pyc':
                    try:path=Path(importlib.util.source_from_cache(str(path))).resolve()
                    except ValueError:path=path.with_suffix('.py')
                if path.is_file() and str(path)!=generated.__file__:dependencies[str(path)]=file_sha(path)
            for line in Path('/proc/self/maps').read_text().splitlines():
                fields=line.split(maxsplit=5)
                if len(fields)==6 and fields[5].startswith('/') and '.so' in Path(fields[5]).name:
                    path=Path(fields[5]).resolve();dependencies[str(path)]=file_sha(path)
            dependencies[str(Path(sys.executable).resolve())]=file_sha(Path(sys.executable).resolve())
            contract=self.root/'CPU_fixture_contract.json'
            atomic_json(contract,dict(dependency_groups={'CPU_fixture_only':dependencies}))
            backend=component.ProjectedBaselineBackend(execution_contract=jit.path_pointer(self.root,contract),
                root=self.root,inner=object())
            backend.jit_session=session;backend.witness=dict(imported_dependency_files_sha256={},loaded_native_files_sha256={})
            backend._imports()
            self.assertIn(generated.__name__,backend.witness['generated_python_module_bindings'])
            self.assertEqual(backend.witness['imported_dependency_files_sha256'][generated.__file__],file_sha(generated.__file__))
        finally:pointer=session.close()
        authorized=jit.validate_observed_bindings(self.root,pointer,self.profile,self.dependencies,backend.witness,
            identity={'CPU_fixture':True,'case':'case0'},process_id=os.getpid())
        self.assertEqual(authorized,{generated.__file__:file_sha(generated.__file__)})
        tampered=deepcopy(backend.witness)
        tampered['generated_python_module_bindings'][generated.__name__]['actual_module_identity']+=1
        with self.assertRaises(ValueError):jit.validate_observed_bindings(self.root,pointer,self.profile,self.dependencies,tampered,
            identity={'CPU_fixture':True,'case':'case0'},process_id=os.getpid())


if __name__=='__main__':unittest.main()
