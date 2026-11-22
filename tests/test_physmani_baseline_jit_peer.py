"""Independent CPU boundaries for initial JIT primitives, not native authority.

Stock key code is extracted as pure AST definitions from the actual frozen
installation. Torch/Triton are not imported, GPU code is never loaded, and
all fixtures live under removable our-mani artifact directories. Hooks and
historical production qualification are separate tests once implemented.
"""
import ast
import base64
from copy import deepcopy
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import os
import pickle
import re
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_jit_artifacts as jit

ROOT=Path(__file__).resolve().parents[1]
SITE=Path('/home/ubuntu/physmani-conda/envs/physmani5090/lib/python3.10/site-packages')


def CPU_hook_original():return 'CPU_original'


def CPU_hook_replacement():return 'CPU_replacement'


class CPUOptionsFixture:
    def __init__(self,request):
        self.__dict__.update(request);self.cluster_dims=(1,1,1)
    def hash(self):return 'CPU_options_fixture'


def actual_stock_CUDA_backend_AST():
    """Execute named stock definitions only, no generated kernel/module."""
    path=SITE/'triton/backends/nvidia/compiler.py';tree=ast.parse(path.read_text())
    selected=[deepcopy(node) for node in tree.body if getattr(node,'name',None) in ('file_hash','CUDAOptions','CUDABackend')]
    backend=next(node for node in selected if node.name=='CUDABackend');backend.bases=[]
    backend.body=[node for node in backend.body if getattr(node,'name',None) in ('_parse_arch','parse_options')]
    namespace=dict(dataclass=dataclasses.dataclass,functools=functools,Path=Path,Tuple=Tuple,Optional=Optional,
        Any=Any,Dict=Dict,hashlib=hashlib,re=re,__file__=str(path),
        knobs=SimpleNamespace(nvidia=SimpleNamespace(libdevice_path=None),runtime=SimpleNamespace(override_arch=None),
            language=SimpleNamespace(default_fp_fusion=True)))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    value=namespace['CUDABackend']();value.target=SimpleNamespace(arch=120)
    return value


class StockJITPrimitivePeerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='JIT_peer_CPU_',dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);self.root=self.base/'project';self.root.mkdir()
        self.profile=dict(schema=jit.PROFILE,cache_paths=jit.cache_paths(self.root),
            fresh_empty_initial_namespace=True,stock_compile_autotune_static_flags_unchanged=True)
        self.env=patch.dict(os.environ,dict(TORCHINDUCTOR_CACHE_DIR='CPU_old_inductor',TRITON_CACHE_DIR='CPU_old_triton'))
        self.env.start();self.addCleanup(self.env.stop)

    def session(self):
        return jit.ArtifactSession(self.root,self.root/'cases/case0',self.profile,{},identity={'CPU_fixture_not_native':True})

    def test_python_keys_match_actual_stock_write_AST_including_strip_and_nonempty_extra(self):
        path=SITE/'torch/_inductor/codecache.py';tree=ast.parse(path.read_text())
        selected=[node for node in tree.body if isinstance(node,ast.FunctionDef)
            and node.name in ('sha256_hash','code_hash','get_hash','write')]
        self.assertEqual({node.name for node in selected},{'sha256_hash','code_hash','get_hash','write'})
        namespace=dict(Union=Union,Optional=Optional,base64=base64,hashlib=hashlib,os=os,
            get_path=lambda key,extension,specified_dir='':(key,str(self.root),str(self.root/(key+'.'+extension))),
            write_atomic=lambda *args,**kwargs:None)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
        for source in ('answer=42',' \nanswer=42\n ',b'\nanswer=42\n',b'kernel\x00bytes'):
            for extra in ('','device=sm120',b'nonemptybytes'):
                with self.subTest(source=source,extra=extra):
                    self.assertEqual(jit.py_code_key(source,extra),namespace['write'](source,'py',extra=extra)[0])
        self.assertEqual(jit.py_code_key(' x=1 '),jit.py_code_key('x=1'))
        self.assertNotEqual(jit.sha_bytes(b' x=1 '),jit.sha_bytes(b'x=1'))
        self.assertNotEqual(jit.py_code_key('x=1','first'),jit.py_code_key('x=1','second'))

    def test_helper_key_matches_actual_stock_source_expression_without_compilation(self):
        path=SITE/'triton/runtime/build.py';tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_module_from_src')
        key=next(n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets))
        source=(SITE/'triton/backends/nvidia/driver.c').read_text();platform='x86_64,Linux,64bit,ELF'
        expected=eval(compile(ast.Expression(key.value),str(path),'eval'),dict(hashlib=hashlib,src=source,platform_key=lambda:platform))
        self.assertEqual(jit.helper_cache_key(source,platform),expected)
        self.assertNotEqual(jit.helper_cache_key(source,platform),jit.helper_cache_key(source,platform+'-other'))

    def test_cache_path_checks_location_only_and_rejects_relative_outside_wrong_kind_and_symlink(self):
        for name in self.profile['cache_paths'].values():Path(name).mkdir(parents=True)
        path=Path(self.profile['cache_paths']['inductor'])/'generated.py';path.write_text('CPU fixture')
        self.assertEqual(jit.check_cache_path(self.root,self.profile,path,kind='inductor'),path)
        self.assertFalse(self.session()._loaded_modules)
        for bad,kind in ((Path('relative.py'),None),(self.base/'outside.py',None),(path,'triton')):
            with self.subTest(path=bad,kind=kind),self.assertRaises(ValueError):jit.check_cache_path(self.root,self.profile,bad,kind=kind)
        outside=self.base/'outside.py';outside.write_text('CPU fixture')
        link=path.parent/'symlink.py';link.symlink_to(outside)
        with self.assertRaises(ValueError):jit.check_cache_path(self.root,self.profile,link)

    def test_initialization_records_empty_once_and_does_not_authorize_later_files(self):
        receipt=jit.initialize_cache_environment(self.root,self.profile)
        self.assertEqual(receipt['initial_files'],[]);self.assertFalse(receipt['old_cache_or_helper_copied'])
        self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],self.profile['cache_paths']['inductor'])
        self.assertEqual(os.environ['TRITON_CACHE_DIR'],self.profile['cache_paths']['triton'])
        path=Path(self.profile['cache_paths']['inductor'])/'unobserved.py';path.write_text('CPU fixture')
        self.assertEqual(jit.initialize_cache_environment(self.root,self.profile)['initialization'],receipt['initialization'])
        self.assertFalse(self.session()._loaded_modules)
        marker=self.root/receipt['initialization']['path'];value=json.loads(marker.read_text());value['profile_sha256']='0'*64
        marker.write_text(json.dumps(value))
        with self.assertRaises(ValueError):jit.initialize_cache_environment(self.root,self.profile)

    def test_nonempty_first_namespace_rejected_before_environment_change(self):
        path=Path(self.profile['cache_paths']['inductor']);path.mkdir(parents=True);(path/'old.py').write_text('CPU fixture')
        with self.assertRaises(ValueError):jit.initialize_cache_environment(self.root,self.profile)
        self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],'CPU_old_inductor')
        self.assertEqual(os.environ['TRITON_CACHE_DIR'],'CPU_old_triton')

    def test_empty_symlink_cache_namespace_rejected_before_external_write_or_environment_change(self):
        outside=self.base/'outside_empty';outside.mkdir()
        path=Path(self.profile['cache_paths']['inductor']);path.parent.mkdir(parents=True);path.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):jit.initialize_cache_environment(self.root,self.profile)
        self.assertEqual(list(outside.iterdir()),[])
        self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],'CPU_old_inductor')

    def test_archived_bytes_exact_and_tamper_detected_without_execution(self):
        session=self.session();data=b'CPU fixture only, not executable GPU or ELF'
        archive=session.archive_bytes(data,'fixture');path=self.root/archive['path']
        self.assertEqual(path.read_bytes(),data);self.assertEqual(archive['sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(session.archive_bytes(data,'fixture'),archive)
        path.write_bytes(b'tampered CPU fixture')
        with self.assertRaises(ValueError):session.archive_bytes(data,'fixture')
        with self.assertRaises(ValueError):session.archive_bytes('not bytes','fixture')

    def test_archive_kind_cannot_create_paths_outside_case_archive(self):
        session=self.session()
        for kind in ('../escape','nested/file','x/../../../../../../escaped'):
            with self.subTest(kind=kind),self.assertRaises(ValueError):session.archive_bytes(b'CPU fixture',kind)
        self.assertFalse((self.base/'escaped').exists())

    def test_partial_event_facts_remain_unqualified_and_snapshot_is_a_copy(self):
        session=self.session();event=session.begin('CPU_fixture_only',{'fixture':True},{'source_SHA':'a'*64})
        partial=session.snapshot()
        self.assertFalse(partial['complete']);self.assertFalse(partial['hook_restoration_completed'])
        self.assertFalse(partial['worker_full_trace']);self.assertTrue(partial['events'][0]['parent_observed'])
        partial['events'][0]['inputs']['source_SHA']='b'*64
        self.assertEqual(session.events[0]['inputs']['source_SHA'],'a'*64)
        session.fail(event,RuntimeError('synthetic CPU failure'))
        self.assertEqual(session.snapshot()['events'][0]['status'],'failed')
        self.assertFalse(session.snapshot()['complete'])

    def test_event_mutation_requires_exact_owned_started_event(self):
        session=self.session();event=session.begin('CPU_fixture_only',{'fixture':True},{})
        for foreign in (deepcopy(event),{'status':'started'}):
            with self.subTest(action='finish',foreign=foreign),self.assertRaises(ValueError):session.finish(foreign)
            with self.subTest(action='fail',foreign=foreign),self.assertRaises(ValueError):session.fail(foreign,RuntimeError('foreign'))
        session.finish(event,result={'CPU_fixture_not_native':True})
        self.assertEqual(session.events[0]['status'],'complete')
        with self.assertRaises(ValueError):session.finish(event)
        with self.assertRaises(ValueError):session.fail(event,RuntimeError('terminal rewrite'))


class StockJITHistoricalPeerTests(unittest.TestCase):
    """Synthetic CPU receipts test validators; they never establish GPU evidence."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='JIT_historical_peer_CPU_',dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);self.root=self.base/'project';self.root.mkdir()
        self.profile=dict(schema=jit.PROFILE,cache_paths=jit.cache_paths(self.root),
            fresh_empty_initial_namespace=True,stock_compile_autotune_static_flags_unchanged=True)
        self.dependencies={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [SITE/'torch/_inductor/codecache.py',SITE/'torch/_inductor/runtime/compile_tasks.py',
                SITE/'torch/_inductor/output_code.py',SITE/'torch/_inductor/async_compile.py',
                SITE/'torch/_inductor/runtime/triton_heuristics.py',SITE/'torch/_inductor/runtime/static_cuda_launcher.py',
                SITE/'triton/runtime/build.py',SITE/'triton/compiler/compiler.py',Path(jit.__file__),Path(__file__)]}
        self.session=jit.ArtifactSession(self.root,self.root/'cases/CPU_receipt_fixture',self.profile,
            self.dependencies,identity={'synthetic_CPU_receipt_fixture':True})

    def owner(self,module,qualified):
        path=SITE/(module.replace('.','/')+'.py')
        return dict(module=module,qualified_name=qualified,path=str(path),sha256=self.dependencies[str(path)])

    def event(self,kind,module,qualified,inputs,artifacts=(),result=None):
        event=self.session.begin(kind,self.owner(module,qualified),inputs)
        self.session.finish(event,artifacts=artifacts,result=result or {})
        return event

    def pointer(self,event):return self.session._event_pointer(event)

    def validate(self,event):return jit.validate_event(self.root,event,self.profile,self.dependencies)

    def python(self):
        source=b'CPU_fixture_value=42\n';extra=b'CPU_fixture_extra';key=jit.py_code_key(source,extra)
        path=Path(self.profile['cache_paths']['inductor'])/key[1:3]/(key+'.py')
        path.parent.mkdir(parents=True);path.write_bytes(source)
        artifact=self.session.archive_file(path,'python')
        producer=self.event('python_source_write','torch._inductor.codecache','write',
            dict(key=key,requested_source=self.session.archive_bytes(source,'requested_python'),
                extra=self.session.archive_bytes(extra,'extra')),[artifact],dict(key=key,path=str(path)))
        loader=self.event('python_module_exec','torch._inductor.runtime.compile_tasks','_reload_python_module',
            dict(key=key,path=str(path),set_sys_modules=True,source_envelope=self.pointer(producer)),
            [artifact],dict(module_name='torch._inductor.runtime.compile_tasks.'+key,key=key,path=str(path),
                actual_module_identity=42,registry_identity=True))
        return producer,loader

    def stock_ASTSource(self):
        path=SITE/'triton/compiler/compiler.py';tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ASTSource')
        namespace=dict(hashlib=hashlib,Language=SimpleNamespace(TRITON='CPU_schema_fixture'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        return namespace['ASTSource']

    def kernel(self):
        """CPU metadata fixtures use actual ASTSource schema, no compiled code."""
        worker=self.owner('torch._inductor.runtime.compile_tasks','_worker_compile_triton')
        task=self.event('triton_parent_task_source','torch._inductor.async_compile','AsyncCompile.triton',
            dict(kernel_name='CPU_fixture_kernel',source=self.session.archive_bytes(b'CPU_source_not_executed','triton_python'),
                worker_source=worker,worker_full_trace=False,worker_PID_observed=False,
                cache_environment=dict(TORCHINDUCTOR_CACHE_DIR=self.profile['cache_paths']['inductor'],
                    TRITON_CACHE_DIR=self.profile['cache_paths']['triton'])),
            result=dict(parent_source_observed_before_stock_submit=True))
        fn=SimpleNamespace(__name__='CPU_fixture_kernel',arg_names=['x','BLOCK'],cache_key='CPU_fake_fn_key')
        cm=dict(signature={'x':'*fp32','BLOCK':'constexpr'},constants={'BLOCK':128},configs=[{(0,):[['tt.divisibility',16]]}],
            device_type='cuda',cc=120,num_warps=4,num_stages=1,debug=True)
        source=self.stock_ASTSource()(fn,cm['signature'],cm['constants'],cm['configs'][0])
        recipe=dict(triton_key='CPU_triton_fixture',source_hash=source.hash(),backend_hash='CPU_backend_fixture',
            options_hash='CPU_options_fixture',sorted_environment='[]')
        key=jit.sha_bytes('-'.join(recipe.values()).encode())
        directory=Path(self.profile['cache_paths']['triton'])/base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
        directory.mkdir(parents=True)
        metadata=directory/'CPU_fixture_kernel.json'
        request=dict(num_warps=4,num_stages=1,debug=True,sanitize_overflow=False)
        projected=json.loads(json.dumps(CPUOptionsFixture(request).__dict__))
        metadata.write_text(json.dumps(dict(hash=key,target=dict(backend='cuda',arch=120,warp_size=32),**projected)))
        cubin=directory/'CPU_fixture_kernel.cubin';cubin.write_bytes(b'CPU_fixture_only_NOT_real_CUDA_binary')
        group=directory/'__grp__CPU_fixture_kernel.json'
        group.write_text(json.dumps(dict(child_paths={metadata.name:str(metadata),cubin.name:str(cubin)})))
        artifacts=[self.session.archive_file(path,'kernel_artifact') for path in (metadata,cubin)]
        result=self.event('triton_compile_result','torch._inductor.runtime.triton_heuristics','StaticTritonCompileResult.make_launcher',
            dict(source_event=self.pointer(task),kernel_hash=key,recipe=recipe,
                metadata_group=self.session.archive_file(group,'metadata_group'),
                source_hash_schema=jit.attributes_schema(dict(signature=source.signature,constants=source.constants,attrs=source.attrs)),
                compile_meta_schema=jit.attributes_schema(cm),target=dict(backend='cuda',arch=120,warp_size=32),
                stock_compile_options=dict(schema='r10_stock_CUDA_four_option_replay_v1',request=request,
                    projected_options=projected,all_projected_fields_match_actual_metadata=True),
                worker_frozen_source=worker,worker_full_trace=False,worker_PID_observed=False),
            artifacts,dict(kernel_hash=key,actual_kernel_identity=777,worker_full_trace=False,
                provenance_scope='CPU_fixture_not_native'))
        static=self.event('static_cuda_binary_load','torch._inductor.runtime.static_cuda_launcher',
            'StaticallyLaunchedCudaKernel.load_kernel',dict(compile_result_event=self.pointer(result),kernel_hash=key,
                name='CPU_fixture_kernel',shared=0,device=0,actual_kernel_identity=777,preload_cubin_path=str(cubin)),
            [artifacts[1]],dict(actual_function_handle=123,n_regs=8,n_spills=0,cubin_path_cleared=True,cubin_raw_cleared=True))
        nonstatic=self.event('nonstatic_cuda_binary_load','triton.compiler.compiler','CompiledKernel._init_handles',
            dict(compile_result_event=self.pointer(result),kernel_hash=key,name='CPU_fixture_kernel',shared=0,
                actual_kernel_identity=777,binary=artifacts[1]['archive']),
            result=dict(actual_module_handle=456,actual_function_handle=123,n_regs=8,n_spills=0))
        return dict(task=task,result=result,static=static,nonstatic=nonstatic,fn=fn,cm=cm,group=group,
            directory=directory,cubin=cubin,worker=worker)

    def helper(self):
        source=b'CPU_helper_source_fixture_not_compiled';platform='CPU_platform_fixture'
        key=jit.helper_cache_key(source.decode(),platform)
        directory=Path(self.profile['cache_paths']['triton'])/base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
        directory.mkdir(parents=True);path=directory/'CPU_helper.so';path.write_bytes(b'CPU_fixture_NOT_real_ELF')
        artifact=self.session.archive_file(path,'native_so')
        compiler=Path('/usr/bin/gcc').resolve()
        self.dependencies[str(compiler)]=hashlib.sha256(compiler.read_bytes()).hexdigest()
        build=self.event('native_helper_build','triton.runtime.build','_build',
            dict(name='CPU_helper',key=key,source=self.session.archive_bytes(source,'helper_c'),
                compiler_command_observed=True,actual_compiler_command=[str(compiler),'CPU_fixture_not_executed.c'],
                actual_compiler_executable=dict(path=str(compiler),sha256=self.dependencies[str(compiler)])),
            [dict(archive=artifact['archive'],sha256=artifact['sha256'],size_bytes=artifact['size_bytes'])],
            dict(actual_build_output_sha256=artifact['sha256'],actual_build_path='CPU_temporary_not_compiled.so'))
        load=self.event('native_helper_load','triton.runtime.build','_load_module_from_path',
            dict(name='CPU_helper',key=key,platform_key=platform,requested_source=self.session.archive_bytes(source,'helper_c'),
                path=str(path),actual_build_event=self.pointer(build)),[artifact],
            dict(module_name='CPU_helper',actual_module_identity=888,spec_origin=str(path)))
        return build,load

    def test_positive_synthetic_CPU_python_and_both_binary_receipt_interfaces(self):
        for event in self.python():self.validate(event)
        fixture=self.kernel()
        for kind in ('task','result','static','nonstatic'):self.validate(fixture[kind])

    def test_missing_archive_and_archive_byte_tamper_rejected(self):
        producer,_=self.python();changed=deepcopy(producer);changed['artifacts'][0].pop('archive')
        with self.assertRaises((ValueError,KeyError)):self.validate(changed)
        path=self.root/producer['artifacts'][0]['archive']['path'];path.write_bytes(b'CPU_tamper_not_executed')
        with self.assertRaises(ValueError):self.validate(producer)

    def test_unbound_source_owner_rejected(self):
        producer,_=self.python();producer=deepcopy(producer);producer['source_owner']['sha256']='0'*64
        with self.assertRaises(ValueError):self.validate(producer)

    def test_bound_source_file_with_wrong_stock_module_or_qualname_rejected(self):
        producer,_=self.python()
        for field,value in (('module','CPU_forged_owner'),('qualified_name','CPU_unrelated_bound_callable')):
            changed=deepcopy(producer);changed['source_owner'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(changed)

    def test_unknown_completed_event_role_rejected(self):
        producer,_=self.python();producer=deepcopy(producer);producer['kind']='unknown_cache_authority'
        with self.assertRaises(ValueError):self.validate(producer)

    def test_python_nonempty_extra_recipe_key_tamper_rejected(self):
        producer,_=self.python();producer=deepcopy(producer);producer['inputs']['key']='c'+'a'*51
        with self.assertRaises(ValueError):self.validate(producer)

    def test_python_module_registry_identity_false_rejected(self):
        _,loader=self.python();loader=deepcopy(loader);loader['result']['registry_identity']=False
        with self.assertRaises(ValueError):self.validate(loader)

    def test_python_load_without_completed_source_envelope_rejected(self):
        producer,loader=self.python();producer['status']='failed'
        path=self.root/self.pointer(producer)['path'];path.write_text(json.dumps(producer))
        loader=deepcopy(loader);loader['inputs']['source_envelope']=jit.path_pointer(self.root,path)
        with self.assertRaises(ValueError):self.validate(loader)

    def test_worker_full_trace_claim_rejected(self):
        fixture=self.kernel()
        for kind,where in (('task','inputs'),('result','inputs'),('result','result')):
            changed=deepcopy(fixture[kind]);changed[where]['worker_full_trace']=True
            with self.subTest(kind=kind,where=where),self.assertRaises(ValueError):self.validate(changed)

    def test_parent_return_and_precompile_cannot_claim_unobserved_worker_full_trace(self):
        fixture=self.kernel()
        returned=self.event('triton_parent_task_return','torch._inductor.async_compile','AsyncCompile.triton',
            dict(source_event=self.pointer(fixture['task'])),
            result=dict(actual_return_type='CPU_future_fixture',actual_return_identity=1,worker_full_trace=False))
        precompile=self.event('triton_parent_precompile','torch._inductor.runtime.triton_heuristics','CachingAutotuner.precompile',
            dict(source_event=self.pointer(fixture['task']),kernel_name='CPU_fixture_kernel',full_worker_trace=False),
            result=dict(actual_compile_results=1,actual_launchers=1,worker_full_trace=False))
        self.validate(returned);self.validate(precompile)
        for original,where,key in ((returned,'result','worker_full_trace'),
                (precompile,'result','worker_full_trace'),(precompile,'inputs','full_worker_trace')):
            changed=deepcopy(original);changed[where][key]=True
            with self.subTest(kind=original['kind'],where=where,key=key),self.assertRaises(ValueError):self.validate(changed)

    def test_worker_frozen_source_must_have_exact_role_and_binding(self):
        fixture=self.kernel()
        for kind,field in (('task','worker_source'),('result','worker_frozen_source')):
            for key,value in (('qualified_name','CPU_wrong_worker'),('sha256','0'*64)):
                changed=deepcopy(fixture[kind]);changed['inputs'][field][key]=value
                with self.subTest(kind=kind,key=key),self.assertRaises(ValueError):self.validate(changed)

    def test_metadata_child_paths_actual_schema_and_outside_path_rejected(self):
        fixture=self.kernel();changed=deepcopy(fixture['result'])
        group=json.loads(fixture['group'].read_text())
        group['child_paths'][fixture['cubin'].name]=str(self.base/'outside.cubin')
        payload=json.dumps(group).encode();archive=self.session.archive_bytes(payload,'metadata_group')
        changed['inputs']['metadata_group'].update(archive=archive,sha256=jit.sha_bytes(payload),size_bytes=len(payload))
        with self.assertRaises(ValueError):self.validate(changed)

    def test_metadata_hash_differs_from_actual_kernel_recipe_rejected(self):
        fixture=self.kernel();changed=deepcopy(fixture['result']);artifact=next(a for a in changed['artifacts'] if a['path'].endswith('.json'))
        payload=json.dumps(dict(hash='0'*64,target=dict(backend='cuda',arch=120,warp_size=32))).encode()
        artifact.update(archive=self.session.archive_bytes(payload,'kernel_artifact'),sha256=jit.sha_bytes(payload),size_bytes=len(payload))
        with self.assertRaises(ValueError):self.validate(changed)

    def test_kernel_recipe_hash_mismatch_rejected(self):
        fixture=self.kernel();changed=deepcopy(fixture['result']);changed['inputs']['recipe']['source_hash']='CPU_changed'
        with self.assertRaises(ValueError):self.validate(changed)

    def test_actual_GPU_object_identity_mismatch_rejected_for_both_routes(self):
        fixture=self.kernel()
        for kind in ('static','nonstatic'):
            changed=deepcopy(fixture[kind]);changed['inputs']['actual_kernel_identity']=778
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate(changed)

    def test_loaded_binary_mismatch_rejected_for_both_routes(self):
        fixture=self.kernel();payload=b'CPU_other_NOT_real_CUDA_binary';archive=self.session.archive_bytes(payload,'cubin')
        for kind in ('static','nonstatic'):
            changed=deepcopy(fixture[kind])
            if kind=='static':changed['artifacts'][0].update(archive=archive,sha256=jit.sha_bytes(payload),size_bytes=len(payload))
            else:changed['inputs']['binary']=archive
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate(changed)

    def test_static_postload_cleared_fields_and_nonstatic_handle_required(self):
        fixture=self.kernel()
        for kind,field,value in (('static','cubin_path_cleared',False),('static','cubin_raw_cleared',False),
                ('static','actual_function_handle',None),('nonstatic','actual_function_handle',None)):
            changed=deepcopy(fixture[kind]);changed['result'][field]=value
            with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):self.validate(changed)

    def test_actual_stock_ASTSource_tuple_keys_preserved_without_string_collisions(self):
        fixture=self.kernel();schema=fixture['result']['inputs']['source_hash_schema']['fields']
        for field,expected in (('constants',1),('attrs',0)):
            mapping=schema[field];self.assertEqual(mapping['type'],'typed_mapping')
            self.assertEqual(mapping['items'][0]['key'],dict(type='tuple',items=[dict(type='int',value=expected)]))
        ambiguous=jit.attributes_schema({(1,):'tuple','(1,)':'string',1:'integer'})
        self.assertEqual(len(ambiguous['items']),3)
        with self.assertRaises(ValueError):jit.attributes_schema({frozenset({1}):'unknown_key'})

    def test_actual_stock_group_and_ASTSource_accepted_by_provenance_reader_CPU(self):
        fixture=self.kernel();self.session._local.tuner=(SimpleNamespace(fn=fixture['fn']),dict(event=self.pointer(fixture['task'])))
        self.session._hook_owners['stock_worker_compile']=fixture['worker']
        compiler=SimpleNamespace(ASTSource=self.stock_ASTSource(),GPUTarget=lambda **kw:SimpleNamespace(**kw),
            make_backend=lambda target:SimpleNamespace(hash=lambda:'CPU_backend_fixture',
                parse_options=lambda request:CPUOptionsFixture(request)),
            get_cache_invalidating_env_vars=lambda:{},triton_key=lambda:'CPU_triton_fixture')
        kernel=SimpleNamespace(hash=fixture['result']['inputs']['kernel_hash'],name='CPU_fixture_kernel',cubin_raw=fixture['cubin'].read_bytes())
        evidence=self.session._kernel_provenance(SimpleNamespace(kernel=kernel,compile_meta=fixture['cm']),compiler)
        self.assertEqual(evidence['binary_sha256'],jit.sha_bytes(kernel.cubin_raw))
        self.assertEqual(evidence['inputs']['recipe'],fixture['result']['inputs']['recipe'])
        self.assertFalse(evidence['inputs']['worker_full_trace'])

    def test_kernel_group_symlink_escape_rejected_before_outside_archive_CPU(self):
        fixture=self.kernel();self.session._local.tuner=(SimpleNamespace(fn=fixture['fn']),dict(event=self.pointer(fixture['task'])))
        self.session._hook_owners['stock_worker_compile']=fixture['worker']
        compiler=SimpleNamespace(ASTSource=self.stock_ASTSource(),GPUTarget=lambda **kw:SimpleNamespace(**kw),
            make_backend=lambda target:SimpleNamespace(hash=lambda:'CPU_backend_fixture',
                parse_options=lambda request:CPUOptionsFixture(request)),
            get_cache_invalidating_env_vars=lambda:{},triton_key=lambda:'CPU_triton_fixture')
        outside=self.base/'outside.cubin';outside.write_bytes(fixture['cubin'].read_bytes())
        fixture['cubin'].unlink();fixture['cubin'].symlink_to(outside)
        kernel=SimpleNamespace(hash=fixture['result']['inputs']['kernel_hash'],name='CPU_fixture_kernel',cubin_raw=outside.read_bytes())
        with self.assertRaises(ValueError):self.session._kernel_provenance(SimpleNamespace(kernel=kernel,compile_meta=fixture['cm']),compiler)

    def test_empty_restored_CPU_receipt_not_substitute_for_wrong_identity_or_unrestored_hooks(self):
        pointer=self.session.close();original=jit.read_pointer(self.root,pointer)
        jit.validate_execution(self.root,pointer,self.profile,self.dependencies,identity={'synthetic_CPU_receipt_fixture':True})
        for field,value in (('identity',{'synthetic_CPU_receipt_fixture':False}),('hook_restoration_completed',False),
                ('worker_full_trace',True),('violations',['CPU_fixture_violation'])):
            changed=deepcopy(original);changed[field]=value
            path=self.session.output/(field+'.json');path.write_text(json.dumps(changed))
            with self.subTest(field=field),self.assertRaises(ValueError):jit.validate_execution(self.root,
                jit.path_pointer(self.root,path),self.profile,self.dependencies,identity={'synthetic_CPU_receipt_fixture':True})

    def test_helper_build_and_load_CPU_interface_requires_compiler_source_and_bytes(self):
        build,load=self.helper();self.validate(build);self.validate(load)
        for event,field,value in ((build,'compiler_command_observed',False),
                (build,'actual_compiler_executable',dict(path='/CPU_unbound_cc',sha256='0'*64)),
                (load,'key','0'*64)):
            changed=deepcopy(event);changed['inputs'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(changed)
        changed=deepcopy(load);changed['inputs']['requested_source']=self.session.archive_bytes(b'CPU_other_source','helper_c')
        changed['inputs']['key']=jit.helper_cache_key('CPU_other_source',changed['inputs']['platform_key'])
        with self.assertRaises(ValueError):self.validate(changed)

    def test_historical_python_path_must_match_key_directory_and_basename(self):
        producer,loader=self.python()
        bad=str(Path(self.profile['cache_paths']['inductor'])/'xx'/'CPU_wrong_key.py')
        for original in (producer,loader):
            changed=deepcopy(original);changed['artifacts'][0]['path']=bad;changed['result']['path']=bad
            if 'path' in changed['inputs']:changed['inputs']['path']=bad
            with self.subTest(kind=original['kind']),self.assertRaises(ValueError):self.validate(changed)

    def test_historical_helper_path_must_match_source_platform_key_directory(self):
        _,load=self.helper();changed=deepcopy(load)
        bad=str(Path(self.profile['cache_paths']['triton'])/'CPU_wrong_key'/'CPU_helper.so')
        changed['inputs']['path']=bad;changed['artifacts'][0]['path']=bad;changed['result']['spec_origin']=bad
        with self.assertRaises(ValueError):self.validate(changed)

    def test_historical_kernel_group_path_must_match_recipe_hash_directory(self):
        fixture=self.kernel();changed=deepcopy(fixture['result'])
        bad_dir=Path(self.profile['cache_paths']['triton'])/'CPU_wrong_recipe_key'
        for artifact in changed['artifacts']:artifact['path']=str(bad_dir/Path(artifact['path']).name)
        group=dict(child_paths={Path(a['path']).name:a['path'] for a in changed['artifacts']});payload=json.dumps(group).encode()
        changed['inputs']['metadata_group'].update(path=str(bad_dir/'__grp__CPU_fixture_kernel.json'),
            archive=self.session.archive_bytes(payload,'metadata_group'),sha256=jit.sha_bytes(payload),size_bytes=len(payload))
        with self.assertRaises(ValueError):self.validate(changed)

    def test_hook_descriptor_restoration_is_exact_and_replaced_hook_marks_partial(self):
        target=SimpleNamespace(call=CPU_hook_original)
        original=inspect.getattr_static(target,'call')
        self.session.patch(target,'call',CPU_hook_replacement,module=__name__,qualified_name='CPU_hook_original')
        self.assertEqual(target.call(),'CPU_replacement')
        self.assertIsNot(inspect.getattr_static(target,'call'),original)
        target.call=lambda:'CPU_unapproved_replacement'
        with self.assertRaisesRegex(ValueError,'replaced during observation'):self.session.close()
        self.assertIs(inspect.getattr_static(target,'call'),original)
        receipt=jit.read_pointer(self.root,self.session.receipt_pointer)
        self.assertFalse(receipt['complete']);self.assertFalse(receipt['hook_restoration_completed'])
        self.assertTrue(receipt['violations'])
        with self.assertRaises(ValueError):jit.validate_execution(self.root,self.session.receipt_pointer,self.profile,self.dependencies)

    def test_historical_CUDA_option_witness_rejects_changed_request_target_and_projection(self):
        fixture=self.kernel();self.validate(fixture['result'])
        changes=[('request_debug',lambda e:e['inputs']['stock_compile_options']['request'].update(debug=False)),
            ('sanitize',lambda e:e['inputs']['stock_compile_options']['request'].update(sanitize_overflow=True)),
            ('projection',lambda e:e['inputs']['stock_compile_options']['projected_options'].update(cluster_dims=[2,1,1])),
            ('target',lambda e:e['inputs'].update(target=dict(backend='cuda',arch=90,warp_size=32))),
            ('compile_meta',lambda e:e['inputs']['compile_meta_schema']['fields'].update(debug=dict(type='bool',value=False))),
            ('claim',lambda e:e['inputs']['stock_compile_options'].update(all_projected_fields_match_actual_metadata=False))]
        for name,change in changes:
            event=deepcopy(fixture['result']);change(event)
            with self.subTest(name=name),self.assertRaises(ValueError):self.validate(event)


class StockJITCUDAOptionsPeerTests(unittest.TestCase):
    """Actual frozen option-definition CPU replay, including JSON type loss."""
    def setUp(self):
        self.backend=actual_stock_CUDA_backend_AST()
        self.cm=dict(device_type='cuda',cc=120,num_warps=4,num_stages=1,debug=True)
        original=self.backend.parse_options(dict(num_warps=4,num_stages=1,debug=True,sanitize_overflow=False))
        self.metadata=dict(target=dict(backend='cuda',arch=120,warp_size=32),**json.loads(json.dumps(original.__dict__)))

    def test_exact_four_request_recovers_tuple_sensitive_actual_stock_hash(self):
        options,witness=jit.stock_cuda_compile_options(self.backend,self.cm,self.metadata)
        naive=self.backend.parse_options(self.metadata)
        self.assertEqual(options.hash(),'156354d5b010ae72f5947ac4acc4ed730ed45a11920f1db67e91be6cd30b6f96')
        self.assertNotEqual(naive.hash(),options.hash())
        for key in ('cluster_dims','supported_fp8_dtypes','deprecated_fp8_dot_operand_dtypes','allowed_dot_input_precisions'):
            self.assertIs(type(options.__dict__[key]),tuple)
            self.assertIs(type(naive.__dict__[key]),list)
            self.assertEqual(list(options.__dict__[key]),self.metadata[key])
        self.assertEqual(witness['request'],dict(num_warps=4,num_stages=1,debug=True,sanitize_overflow=False))
        self.assertEqual(witness['projected_options'],json.loads(json.dumps(options.__dict__)))

    def test_option_metadata_is_checked_instead_of_cast_or_repaired(self):
        for key,value in (('cluster_dims',[2,1,1]),('debug',1),('num_warps',4.0),('sanitize_overflow',True)):
            metadata=deepcopy(self.metadata);metadata[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):jit.stock_cuda_compile_options(self.backend,self.cm,metadata)
        metadata=deepcopy(self.metadata);metadata.pop('supported_fp8_dtypes')
        with self.assertRaises(ValueError):jit.stock_cuda_compile_options(self.backend,self.cm,metadata)

    def test_unknown_target_or_compile_meta_type_rejected(self):
        for key,value in (('device_type','cpu'),('cc',True),('num_warps',True),('num_stages',0),('debug',1)):
            cm=deepcopy(self.cm);cm[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):jit.stock_cuda_compile_options(self.backend,cm,self.metadata)
        metadata=deepcopy(self.metadata);metadata['target']['arch']=90
        with self.assertRaises(ValueError):jit.stock_cuda_compile_options(self.backend,self.cm,metadata)


class StockJITActualEntryPeerTests(unittest.TestCase):
    """Only native-environment CPU entry tests; CUDA and policy stay unused."""
    @classmethod
    def setUpClass(cls):
        import torch
        if Path(torch.__file__).resolve().parent.parent!=SITE:
            raise unittest.SkipTest('Actual stock entry fixture requires the fixed native Torch installation')
        cls.torch=torch
        assert not torch.cuda.is_initialized()
        names=['torch._inductor.codecache','torch._inductor.output_code','torch._inductor.async_compile',
            'torch._inductor.runtime.compile_tasks','torch._inductor.runtime.triton_heuristics',
            'torch._inductor.runtime.static_cuda_launcher','torch._functorch._aot_autograd.autograd_cache',
            'torch._inductor.compile_worker.subproc_pool',
            'triton.runtime.build','triton.backends.nvidia.driver','triton.compiler.compiler']
        cls.modules={name:importlib.import_module(name) for name in names}
        cls.dependencies={str(Path(module.__file__).resolve()):jit.file_sha(module.__file__) for module in cls.modules.values()}
        cls.dependencies[str(Path(jit.__file__).resolve())]=jit.file_sha(jit.__file__)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='JIT_actual_entry_peer_CPU_',dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'project';self.root.mkdir()
        self.CPU_session_count=0
        self.profile=dict(schema=jit.PROFILE,cache_paths=jit.cache_paths(self.root),
            fresh_empty_initial_namespace=True,stock_compile_autotune_static_flags_unchanged=True)
        env=patch.dict(os.environ,{});env.start();self.addCleanup(env.stop)
        jit.initialize_cache_environment(self.root,self.profile)
        self.addCleanup(lambda:self.assertFalse(self.torch.cuda.is_initialized()))

    def session(self,deps=None):
        self.CPU_session_count+=1
        return jit.ArtifactSession(self.root,self.root/'cases'/f'CPU_actual_entry_fixture_{self.CPU_session_count}',self.profile,
            self.dependencies if deps is None else deps,identity={'actual_stock_entry_CPU_only':True})

    def test_partial_install_unbound_stock_owner_restores_already_patched_entries(self):
        cache=self.modules['torch._inductor.codecache'];tasks=self.modules['torch._inductor.runtime.compile_tasks']
        original_write=cache.write;original_reload=tasks._reload_python_module;original_alias=cache._reload_python_module
        dependencies=dict(self.dependencies);dependencies.pop(str(Path(tasks.__file__).resolve()))
        session=self.session(dependencies)
        with self.assertRaisesRegex(ValueError,'stock callable origin/source'):session.install()
        self.assertIs(cache.write,original_write);self.assertIs(tasks._reload_python_module,original_reload)
        self.assertIs(cache._reload_python_module,original_alias)
        pointer=session.close(RuntimeError('expected missing owner CPU fixture'))
        self.assertFalse(jit.read_pointer(self.root,pointer)['complete'])

    def test_unknown_stock_guarded_cache_pickle_rejected_before_any_pickle_side_effect(self):
        cache=self.modules['torch._inductor.codecache']
        directory=Path(self.profile['cache_paths']['inductor'])/'fxgraph'/'CPU_fixture';directory.mkdir(parents=True)
        marker=self.root/'CPU_unpickle_executed_UNEXPECTED'
        class SideEffectFixture:
            def __reduce__(self):return os.mkdir,(str(marker),)
        (directory/'CPU_unknown_pickle').write_bytes(pickle.dumps(SideEffectFixture()))
        class CPUCandidateCache(cache.GuardedCache):
            @classmethod
            def _get_tmp_dir_for_key(cls,key):return str(directory)
        session=self.session();original_loads=pickle.loads;session.install()
        try:
            # Stock iterator catches its candidate exceptions; owned violation
            # must survive the stock fallback and prevent complete qualification.
            self.assertEqual(list(CPUCandidateCache.iterate_over_candidates(True,None,'CPU_fixture')),[])
            self.assertFalse(marker.exists())
            self.assertTrue(session._violations)
            self.assertEqual(pickle.loads(pickle.dumps({'ordinary_application_pickle':True})),
                {'ordinary_application_pickle':True})
        finally:pointer=session.close()
        self.assertIs(pickle.loads,original_loads)
        self.assertFalse(jit.read_pointer(self.root,pointer)['complete'])
        with self.assertRaises(ValueError):jit.validate_execution(self.root,pointer,self.profile,self.dependencies)

    def test_replaced_stock_callable_rejected_and_prior_install_slots_restored(self):
        cache=self.modules['torch._inductor.codecache'];tasks=self.modules['torch._inductor.runtime.compile_tasks']
        original_write=cache.write;original_reload=tasks._reload_python_module
        session=self.session()
        with patch.object(tasks,'_reload_python_module',CPU_hook_replacement):
            with self.assertRaisesRegex(ValueError,'stock callable origin/source'):session.install()
            self.assertIs(cache.write,original_write)
            self.assertIs(tasks._reload_python_module,CPU_hook_replacement)
        self.assertIs(tasks._reload_python_module,original_reload)
        self.assertFalse(jit.read_pointer(self.root,session.close(RuntimeError('expected callable replacement CPU fixture')))['complete'])

    def CPU_parent_task_source(self,session):
        task_path=SITE/'torch/_inductor/async_compile.py';worker_path=SITE/'torch/_inductor/runtime/compile_tasks.py'
        event=session.begin('triton_parent_task_source',dict(module='torch._inductor.async_compile',
            qualified_name='AsyncCompile.triton',path=str(task_path),sha256=self.dependencies[str(task_path)]),
            dict(kernel_name='CPU_no_compile_no_worker',source=session.archive_bytes(b'CPU_not_generated_code','triton_python'),
                worker_source=dict(module='torch._inductor.runtime.compile_tasks',qualified_name='_worker_compile_triton',
                    path=str(worker_path),sha256=self.dependencies[str(worker_path)]),
                worker_full_trace=False,worker_PID_observed=False,
                cache_environment=dict(TORCHINDUCTOR_CACHE_DIR=self.profile['cache_paths']['inductor'],
                    TRITON_CACHE_DIR=self.profile['cache_paths']['triton'])))
        session.finish(event,result=dict(parent_source_observed_before_stock_submit=True))
        return session._event_pointer(event)

    def CPU_fake_pool(self,writes):
        import itertools
        import threading
        return SimpleNamespace(pickler=SimpleNamespace(dumps=lambda function:b'CPU_payload_not_a_pickle'),
            futures_lock=threading.Lock(),write_lock=threading.Lock(),pending_futures={},job_id_count=itertools.count(),
            write_pipe=SimpleNamespace(write=lambda payload:writes.append(payload),flush=lambda:None),
            running=True,process=SimpleNamespace(pid=1234))

    def test_exact_owned_guard_observes_actual_stock_submit_without_worker_or_pickle_execution(self):
        pool_module=self.modules['torch._inductor.compile_worker.subproc_pool']
        original=pool_module._send_msg;session=self.session();source=self.CPU_parent_task_source(session)
        session.install();writes=[];pool=self.CPU_fake_pool(writes)
        session._local.task_source=source
        try:
            future=pool_module.SubprocPool.submit(pool,lambda:None)
            messages=[event for event in session.events if event['kind']=='stock_worker_task_message']
            self.assertEqual(len(messages),1);self.assertEqual(messages[0]['status'],'complete')
            self.assertEqual(messages[0]['inputs']['actual_parent_future_identity'],id(future))
            self.assertEqual(messages[0]['inputs']['actual_pool_server_process_id'],1234)
            self.assertFalse(messages[0]['inputs']['worker_process_id_observed'])
            self.assertFalse(messages[0]['inputs']['worker_full_trace'])
            self.assertEqual(writes[-1],b'CPU_payload_not_a_pickle')
        finally:
            session._local.task_source=None;pointer=session.close()
        self.assertIs(pool_module._send_msg,original)
        self.assertTrue(jit.validate_execution(self.root,pointer,self.profile,self.dependencies)['complete'])

    def test_unknown_IPC_wrapper_and_non_submit_caller_get_no_message_authority(self):
        pool_module=self.modules['torch._inductor.compile_worker.subproc_pool']
        for variant in ('unknown_outer_wrapper','direct_non_stock_submit'):
            session=self.session();source=self.CPU_parent_task_source(session);original=pool_module._send_msg
            session.install();guard=pool_module._send_msg;writes=[];pool=self.CPU_fake_pool(writes)
            session._local.task_source=source
            try:
                if variant=='unknown_outer_wrapper':
                    def unknown(*args,**kwargs):return guard(*args,**kwargs)
                    pool_module._send_msg=unknown
                    with self.assertRaisesRegex(ValueError,'exact current owned guard'):
                        pool_module.SubprocPool.submit(pool,lambda:None)
                else:
                    with self.assertRaisesRegex(ValueError,'not stock submit'):
                        pool_module._send_msg(pool.write_pipe,0,b'CPU_no_send')
                self.assertEqual(writes,[])
                self.assertFalse(any(event['kind']=='stock_worker_task_message' for event in session.events))
                self.assertTrue(session._violations)
            finally:
                pool_module._send_msg=guard;session._local.task_source=None;pointer=session.close()
            self.assertIs(pool_module._send_msg,original)
            self.assertFalse(jit.read_pointer(self.root,pointer)['complete'])

    def test_preinstall_replaced_submit_or_pickler_caller_rejected_before_authority_and_restored(self):
        pool=self.modules['torch._inductor.compile_worker.subproc_pool']
        cache=self.modules['torch._inductor.codecache']
        for owner,name in ((pool.SubprocPool,'submit'),(pool.SubprocPickler,'loads')):
            original_send=pool._send_msg;original_write=cache.write
            original=inspect.getattr_static(owner,name);session=self.session()
            with patch.object(owner,name,CPU_hook_replacement):
                with self.assertRaisesRegex(ValueError,'stock callable origin/source'):
                    session.install()
                self.assertIs(inspect.getattr_static(owner,name),CPU_hook_replacement)
                self.assertIs(pool._send_msg,original_send);self.assertIs(cache.write,original_write)
                self.assertFalse(any(event['kind'].startswith('stock_worker_') for event in session.events))
                pointer=session.close(RuntimeError('Expected preinstall unknown caller CPU rejection'))
                self.assertFalse(jit.read_pointer(self.root,pointer)['complete'])
            self.assertIs(inspect.getattr_static(owner,name),original)

    def test_actual_preserved_v8_source86_and_two_stock_kernel_keys_replay_without_generated_exec(self):
        from triton.compiler import compiler
        from triton.runtime.jit import DependenciesFinder
        import triton.language as tl
        from torch._inductor.runtime.triton_helpers import math as tl_math,libdevice
        case=ROOT/'artifacts/framework/multitask_r10/PhysMani_early_VAL_approved_v8/executions/cases/3abaa978021bf3143b88/attempt0000'
        event=json.loads((case/'jit_artifacts/events/000086.json').read_text())
        source=jit.read_archive(ROOT,event['inputs']['source']).decode()
        tree=ast.parse(source);function=next(node for node in tree.body if isinstance(node,ast.FunctionDef))
        text=source[re.search(r'^def\s+',source,re.MULTILINE).start():]
        finder=DependenciesFinder(function.name,dict(tl=tl,tl_math=tl_math,libdevice=libdevice),{},text)
        finder.visit(ast.parse(text))
        line=min([function.lineno]+[node.lineno for node in function.decorator_list])
        self.assertEqual(finder.ret+str(line),'5c32d24a25874046bdf61f6e70ca3d0097e54877fbbe8df33649d54ba4adc6ad10')
        meta=next(keyword.value for keyword in function.decorator_list[0].keywords if keyword.arg=='triton_meta')
        parts={ast.literal_eval(key):value for key,value in zip(meta.keys,meta.values)}
        signature=ast.literal_eval(parts['signature']);attrs=ast.literal_eval(parts['configs'])[0]
        fn=SimpleNamespace(__name__=function.name,arg_names=[a.arg for a in function.args.args],cache_key=finder.ret+str(line))
        report_path=ROOT/'artifacts/framework/multitask_r10/PhysMani_v8_first_JIT_readonly_diagnosis/native_CPU_components.json'
        report=json.loads(report_path.read_text())
        rows=report['actual_CPU_component_replay']['exact_actual_kernel_key_matches']
        self.assertEqual({row['XBLOCK'] for row in rows},{128,256})
        for row in rows:
            metadata_path=Path(row['metadata_path']);metadata=json.loads(metadata_path.read_text())
            expected=next(x for x in report['options_replay'] if x['metadata_path']==str(metadata_path))
            self.assertEqual(jit.file_sha(metadata_path),expected['metadata_sha256'])
            cm=dict(device_type='cuda',cc=120,num_warps=4,num_stages=1,debug=True)
            backend=compiler.make_backend(compiler.GPUTarget(**metadata['target']))
            options,witness=jit.stock_cuda_compile_options(backend,cm,metadata)
            source_hash=compiler.ASTSource(fn,signature,{'XBLOCK':row['XBLOCK']},attrs).hash()
            expression='-'.join((compiler.triton_key(),source_hash,backend.hash(),options.hash(),
                str(sorted(compiler.get_cache_invalidating_env_vars().items()))))
            self.assertEqual(jit.sha_bytes(expression.encode()),metadata['hash'])
            self.assertEqual(metadata['hash'],{128:'9e316c53ffacdb340cdb939e631bc56ba31c34c83c60f8e925e57593cae98147',
                256:'98a62e08e6ab2bfd87ddb2e270825e0d40843fa0e0641cb1722c268d6afb3125'}[row['XBLOCK']])
            self.assertEqual(source_hash,row['source_hash']);self.assertEqual(options.hash(),row['options_hash'])
            self.assertTrue(witness['all_projected_fields_match_actual_metadata'])
            self.assertNotEqual(backend.parse_options(metadata).hash(),options.hash())
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
