"""Owned observation of stock Torch/Triton generated execution artifacts.

This module is imported without Torch, Triton, CUDA or simulator imports.
Only explicitly observed stock producer/loader events can authorize generated
files. A cache-directory prefix never grants execution authority.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import base64
import ast
import builtins
import hashlib
import inspect
import importlib
import importlib.machinery
import json
import marshal
import os
from pathlib import Path
import sys
import shutil
import threading
import functools
import tempfile
from types import CodeType, FunctionType, ModuleType

from .evaluation import atomic_json, digest, file_sha

PROFILE = 'r10_stock_PhysMani_JIT_derived_artifact_profile_v1'
EXECUTION = 'r10_observed_stock_PhysMani_JIT_artifacts_v1'
EVENT = 'r10_observed_stock_JIT_artifact_event_v1'
CACHE_RELATIVE = 'runtime/framework/physmani_r10_JIT_v11'
ROOT = Path(__file__).resolve().parents[2]
STOCK_RELATIVE_SOURCES=(
    'torch/_inductor/codecache.py','torch/_inductor/output_code.py','torch/_inductor/async_compile.py',
    'torch/_inductor/runtime/compile_tasks.py','torch/_inductor/runtime/triton_heuristics.py',
    'torch/_inductor/runtime/static_cuda_launcher.py','torch/_functorch/_aot_autograd/autograd_cache.py',
    'torch/_inductor/compile_worker/subproc_pool.py','torch/_inductor/compile_worker/__main__.py',
    'triton/runtime/build.py','triton/runtime/cache.py','triton/compiler/compiler.py',
    'triton/backends/nvidia/driver.py','triton/backends/nvidia/driver.c','triton/backends/nvidia/compiler.py',
    'torch/distributed/nn/__init__.py','torch/distributed/nn/api/remote_module.py',
    'torch/distributed/nn/jit/instantiator.py','torch/distributed/nn/jit/templates/remote_module_template.py')
REMOTE_NAME='_remote_module_non_scriptable'
REMOTE_EVENT='stock_remote_non_scriptable_postload_verification'
REMOTE_BINDING='r10_exact_postload_stock_remote_non_scriptable_module_v1'
REMOTE_SOURCES={
    'nn_entry':'torch/distributed/nn/__init__.py',
    'API':'torch/distributed/nn/api/remote_module.py',
    'instantiator':'torch/distributed/nn/jit/instantiator.py',
    'template':'torch/distributed/nn/jit/templates/remote_module_template.py'}
REMOTE_ARGUMENTS=dict(assign_module_interface_cls='module_interface_cls = None',args='*args',kwargs='**kwargs',
    arg_types='*args, **kwargs',arrow_and_return_type='',arrow_and_future_return_type='',jit_script_decorator='')
REMOTE_FUNCTIONS=('forward_async','forward','_remote_forward')
EVENT_ROLES={
    'python_source_write':(('torch._inductor.codecache','write'),),
    'python_deserialized_source':(('torch._inductor.output_code','CompiledFxGraph.write_to_disk'),),
    'python_module_exec':(('torch._inductor.runtime.compile_tasks','_reload_python_module'),),
    'python_module_load':(('torch._inductor.codecache','PyCodeCache.load_by_key_path'),),
    'native_helper_source':(('triton.runtime.build','compile_module_from_src'),),
    'native_helper_build':(('triton.runtime.build','_build'),),
    'native_helper_load':(('triton.runtime.build','_load_module_from_path'),),
    'triton_parent_task_source':(('torch._inductor.async_compile','AsyncCompile.triton'),),
    'triton_parent_task_return':(('torch._inductor.async_compile','AsyncCompile.triton'),),
    'triton_parent_precompile':(('torch._inductor.runtime.triton_heuristics','CachingAutotuner.precompile'),),
    'triton_compile_result':(('torch._inductor.runtime.triton_heuristics','StaticTritonCompileResult.make_launcher'),
        ('torch._inductor.runtime.triton_heuristics','TritonCompileResult.make_launcher')),
    'static_cuda_binary_load':(('torch._inductor.runtime.static_cuda_launcher','StaticallyLaunchedCudaKernel.load_kernel'),),
    'nonstatic_cuda_binary_load':(('triton.compiler.compiler','CompiledKernel._init_handles'),),
    'stock_cache_serialization':(('torch._inductor.codecache','FxGraphCache._save_graph'),
        ('torch._functorch._aot_autograd.autograd_cache','AOTAutogradCache.save')),
    'stock_cache_before_unpickle':(('torch._inductor.codecache','GuardedCache.iterate_over_candidates'),)}
EVENT_ROLES.update(stock_worker_task_message=(('torch._inductor.compile_worker.subproc_pool','_send_msg'),),
    stock_worker_result_before_unpickle=(('torch._inductor.compile_worker.subproc_pool','SubprocPool._read_thread'),),
    stock_remote_non_scriptable_postload_verification=(('torch.distributed.nn.jit.instantiator',
        'instantiate_non_scriptable_remote_module_template'),))


def draft_profile(root,python_executable,evidence=()):
    """Metadata only: no stock imports, cache creation or executable loads."""
    root=Path(root).resolve();python=Path(python_executable).resolve()
    version=python.parent.parent/'lib'
    # This native environment exposes python3.1 as an alias of python3.10.
    # Deduplicate resolved roots; never guess between genuinely different ABIs.
    candidates=sorted({path.resolve() for path in version.glob('python*/site-packages')})
    require(len(candidates)==1,'exact interpreter site-packages root required')
    site=candidates[0]
    sources=[Path(__file__).resolve(),python,*[site/name for name in STOCK_RELATIVE_SOURCES]]
    require(all(path.is_file() for path in sources),'stock JIT source closure is incomplete')
    return dict(schema=PROFILE,reviewed_before_execution=False,cache_paths=cache_paths(root),
        fresh_empty_initial_namespace=True,stock_compile_autotune_static_flags_unchanged=True,
        old_cache_or_helper_copy_allowed=False,full_worker_trace_claimed=False,
        scope='original275_PhysMani_VAL_replica0_only',
        source_bindings_sha256={str(path.resolve()):file_sha(path) for path in sources},
        evidence=list(evidence),actual_native_or_GPU_execution=False)


def load_profile(root,pointer,dependencies,*,require_reviewed=True):
    value=read_pointer(root,pointer)
    require(set(value)=={'schema','reviewed_before_execution','cache_paths','fresh_empty_initial_namespace',
        'stock_compile_autotune_static_flags_unchanged','old_cache_or_helper_copy_allowed','full_worker_trace_claimed',
        'scope','source_bindings_sha256','evidence','actual_native_or_GPU_execution'},'JIT profile fields differ')
    require(value['schema']==PROFILE and value['reviewed_before_execution'] is require_reviewed
        and value['cache_paths']==cache_paths(root) and value['fresh_empty_initial_namespace'] is True
        and value['stock_compile_autotune_static_flags_unchanged'] is True
        and value['old_cache_or_helper_copy_allowed'] is False and value['full_worker_trace_claimed'] is False
        and value['scope']=='original275_PhysMani_VAL_replica0_only'
        and value['actual_native_or_GPU_execution'] is False,'JIT scope/profile was widened or misqualified')
    bindings=value['source_bindings_sha256']
    require(isinstance(bindings,dict) and bindings and all(dependencies.get(name)==checksum
        and Path(name).is_absolute() for name,checksum in bindings.items()),'stock JIT binding is outside frozen closure')
    require(bindings.get(str(Path(__file__).resolve()))==file_sha(__file__),'owned JIT source binding differs')
    for suffix in STOCK_RELATIVE_SOURCES:
        require(sum(name.endswith('/'+suffix) for name in bindings)==1,'stock JIT exact source role missing: '+suffix)
    for item in value['evidence']:
        read_pointer(root,item)
        require(dependencies.get(str((Path(root)/item['path']).resolve()))==item['sha256'],'JIT source review evidence is unbound')
    return value


def require(value, message):
    if not value:
        raise ValueError('baseline exact JIT artifacts: '+message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def py_code_key(source, extra=''):
    """Stock codecache.write rule, with execution bytes kept separately."""
    require(isinstance(source, (str, bytes)) and isinstance(extra, (str, bytes)),
            'source/extra must be exact strings or bytes')
    payload = source.encode('utf-8') if isinstance(source, str) else source
    payload = payload.strip()
    additional = extra.encode('utf-8') if isinstance(extra, str) else extra
    if additional:
        payload += b'||'+additional
    return 'c'+base64.b32encode(hashlib.sha256(payload).digest())[:51].decode().lower()


def helper_cache_key(source, platform_key):
    require(isinstance(source, str) and isinstance(platform_key, str), 'invalid helper source/platform')
    return hashlib.sha256((source+platform_key).encode('utf-8')).hexdigest()


def path_pointer(root, path):
    path = Path(path).resolve(); root = Path(root).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'archive pointer escaped ROOT')
    return dict(path=str(path.relative_to(root)), sha256=file_sha(path))


def cache_paths(root):
    base = Path(root).resolve()/CACHE_RELATIVE
    return dict(inductor=str(base/'inductor'), triton=str(base/'triton'))


def check_cache_path(root, profile, path, *, kind=None):
    """Check location only; callers must separately prove event provenance."""
    path = Path(path)
    require(path.is_absolute(), 'generated path must be absolute')
    resolved = path.resolve()
    roots = profile['cache_paths']
    require(roots == cache_paths(root), 'cache profile widened ROOT locations')
    allowed = [Path(roots[k]).resolve() for k in ([kind] if kind else roots)]
    require(any(resolved.is_relative_to(directory) and resolved != directory for directory in allowed),
            'generated path escaped the exact run-owned caches')
    require(all(Path(roots[k]).resolve() == Path(cache_paths(root)[k]) for k in roots),
            'cache profile widened ROOT locations')
    return resolved


def initialize_cache_environment(root, profile):
    """Set two cache locations before any Torch/driver initialization.

    First initialization records an empty namespace. Later native children in
    this same run must validate its exact profile and use artifact receipts.
    """
    root = Path(root).resolve()
    require(profile.get('schema') == PROFILE and profile.get('cache_paths') == cache_paths(root)
        and profile.get('fresh_empty_initial_namespace') is True
        and profile.get('stock_compile_autotune_static_flags_unchanged') is True,
        'unknown or widened cache initialization profile')
    base = root/CACHE_RELATIVE
    require(base.resolve() == base and base.is_relative_to(root), 'cache base is a symlink escape')
    require(all(Path(path).resolve() == Path(path) for path in profile['cache_paths'].values()),
            'cache directory is a symlink escape')
    marker = base/'initial_empty_namespace.json'
    lock_path = base/'initialization.lock'
    require(all(path.resolve() == path and path.is_relative_to(root)
                for path in [base, marker, lock_path, *map(Path, profile['cache_paths'].values())]),
            'cache initialization path is a symlink escape')
    base.mkdir(parents=True, exist_ok=True)
    # A lock makes the first parent/child and later parallel children agree on
    # the one actual empty initialization; it is outside the cache directories.
    import fcntl
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if marker.exists():
            value = json.loads(marker.read_text())
            require(value.get('profile_sha256') == digest(profile)
                and value.get('cache_paths') == profile['cache_paths']
                and value.get('initial_files') == [], 'cache initialization belongs to another profile')
        else:
            existing = []
            for name in ('inductor', 'triton'):
                directory = Path(profile['cache_paths'][name]); directory.mkdir(parents=True, exist_ok=True)
                existing += [str(path.resolve()) for path in directory.rglob('*') if path.is_file()]
            require(not existing, 'fresh cache namespace already contains preexisting files')
            value = dict(schema='r10_actual_empty_JIT_namespace_initialization_v1',
                created_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid(),
                profile_sha256=digest(profile), cache_paths=deepcopy(profile['cache_paths']),
                initial_files=[], old_cache_or_helper_copied=False, policy_outcome_used=False)
            atomic_json(marker, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = profile['cache_paths']['inductor']
    os.environ['TRITON_CACHE_DIR'] = profile['cache_paths']['triton']
    return dict(initialization=path_pointer(root, marker), **deepcopy(value))


def attributes_schema(value):
    """Record actual constant schemas without inventing tensor-value hashes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return dict(type=type(value).__name__, value=value)
    if isinstance(value, dict):
        if all(isinstance(k,str) for k in value):
            return dict(type='dict', fields={k: attributes_schema(v) for k, v in sorted(value.items())})
        require(all(type(k) in (str,int) or isinstance(k,tuple) and all(type(i) is int for i in k)
            for k in value),'unknown typed positional mapping key')
        return dict(type='typed_mapping',items=[dict(key=attributes_schema(k),value=attributes_schema(v))
            for k,v in sorted(value.items(),key=lambda item:repr(item[0]))])
    if isinstance(value, (list, tuple)):
        return dict(type=type(value).__name__, items=[attributes_schema(v) for v in value])
    if type(value).__module__.startswith('torch') and hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return dict(type=type(value).__module__+'.'+type(value).__qualname__,
            dtype=str(value.dtype), shape=list(value.shape), device=str(value.device),
            tensor_values_hashed=False)
    return dict(type=type(value).__module__+'.'+type(value).__qualname__, values_hashed=False)


def stock_cuda_compile_options(backend, compile_meta, metadata):
    """Replay the four CUDA options passed by frozen _precompile_config.

    Compiler metadata is a JSON projection of options and compilation outputs.
    It cannot reconstruct Python tuple types used by the stock options hash.
    Rebuild from actual compile_meta instead, then check every projected option
    against the actual worker metadata without changing its types or bytes.
    """
    require(isinstance(compile_meta,dict) and compile_meta.get('device_type')=='cuda'
        and type(compile_meta.get('cc')) is int
        and metadata.get('target')==dict(backend='cuda',arch=compile_meta['cc'],warp_size=32),
        'stock CUDA compile target differs')
    require(type(compile_meta.get('num_warps')) is int and compile_meta['num_warps']>0
        and type(compile_meta.get('num_stages')) is int and compile_meta['num_stages']>0
        and type(compile_meta.get('debug')) is bool,'stock CUDA compile options types differ')
    request=dict(num_warps=compile_meta['num_warps'],num_stages=compile_meta['num_stages'],
        debug=compile_meta['debug'],sanitize_overflow=False)
    options=backend.parse_options(request)
    projected=json.loads(json.dumps(options.__dict__,allow_nan=False))
    require(isinstance(projected,dict) and projected,'stock CUDA options projection is empty')
    for field,value in projected.items():
        require(field in metadata and json.dumps(metadata[field],sort_keys=True,allow_nan=False)==
            json.dumps(value,sort_keys=True,allow_nan=False),'worker metadata option differs: '+field)
    return options,dict(schema='r10_stock_CUDA_four_option_replay_v1',request=request,
        projected_options=projected,all_projected_fields_match_actual_metadata=True)


def stock_remote_template_recipe(source_bytes):
    """Derive only the fixed non-scriptable stock recipe using AST literals.

    No generated module, formatter function, RPC function or CUDA code executes.
    """
    require(set(source_bytes)==set(REMOTE_SOURCES) and all(isinstance(v,bytes) for v in source_bytes.values()),
        'remote template stock source roles differ')
    template=ast.parse(source_bytes['template'].decode('utf-8'))
    strings={}
    for node in template.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('_TEMPLATE_PREFIX','_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA',
                '_REMOTE_FORWARD_TEMPLATE'):
                strings[node.targets[0].id]=ast.literal_eval(node.value)
    require(len(strings)==3 and all(type(v) is str for v in strings.values()),'stock remote template literals differ')
    function=next((n for n in template.body if isinstance(n,ast.FunctionDef) and n.name=='get_remote_module_template'),None)
    expected=ast.parse('_TEMPLATE_PREFIX + (_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA '
        'if enable_moving_cpu_tensors_to_cuda else _REMOTE_FORWARD_TEMPLATE)',mode='eval').body
    require(function is not None and len(function.body)==1 and isinstance(function.body[0],ast.Return)
        and ast.dump(function.body[0].value)==ast.dump(expected),'stock remote formatter expression differs')
    instantiator=ast.parse(source_bytes['instantiator'].decode('utf-8'))
    function=next((n for n in instantiator.body if isinstance(n,ast.FunctionDef)
        and n.name=='instantiate_non_scriptable_remote_module_template'),None)
    require(function is not None,'fixed non-scriptable stock instantiator is absent')
    assignment=next((n for n in function.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='str_dict' for t in n.targets)),None)
    require(assignment is not None and isinstance(assignment.value,ast.Call)
        and isinstance(assignment.value.func,ast.Name) and assignment.value.func.id=='dict'
        and not assignment.value.args,'stock non-scriptable parameters differ')
    arguments={item.arg:ast.literal_eval(item.value) for item in assignment.value.keywords}
    require(arguments==REMOTE_ARGUMENTS,'stock non-scriptable parameters widened')
    returned=function.body[-1]
    require(isinstance(returned,ast.Return) and isinstance(returned.value,ast.Call)
        and isinstance(returned.value.func,ast.Name) and returned.value.func.id=='_do_instantiate_remote_module_template'
        and len(returned.value.args)==3 and not returned.value.keywords
        and ast.dump(returned.value.args[0])==ast.dump(ast.Name(id='generated_module_name',ctx=ast.Load()))
        and ast.dump(returned.value.args[1])==ast.dump(ast.Name(id='str_dict',ctx=ast.Load()))
        and isinstance(returned.value.args[2],ast.Constant) and returned.value.args[2].value is True,
        'stock non-scriptable instantiation call differs')
    text=(strings['_TEMPLATE_PREFIX']+strings['_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA']).format(**arguments)
    return text.encode('utf-8'),dict(schema='r10_fixed_stock_non_scriptable_remote_recipe_v1',module_name=REMOTE_NAME,
        arguments=arguments,enable_moving_cpu_tensors_to_cuda=True,generated_source_sha256=sha_bytes(text.encode('utf-8')))


def check_remote_template_path(root, path, directory):
    """Exact temporary-directory/file location; never an execution exemption."""
    root=Path(root).resolve();path=Path(path);directory=Path(directory)
    require(path.is_absolute() and directory.is_absolute() and path==directory/(REMOTE_NAME+'.py')
        and directory.parent==root/'runtime/framework/tmp'
        and directory.name.startswith('tmp') and len(directory.name)>3
        and all(part not in ('.','..') for part in path.parts), 'remote template path differs from exact stock temp directory')
    return path


def remote_function_codes(source, path, optimization):
    require(type(optimization) is int and optimization in (0,1,2),'remote Python optimization scope differs')
    code=compile(source.decode('utf-8'),str(path),'exec',dont_inherit=True,optimize=optimization)
    functions={item.co_name:item for item in code.co_consts if isinstance(item,CodeType)}
    require(set(functions)==set(REMOTE_FUNCTIONS),'fixed remote template function set differs')
    # Version 0 serializes code structure without reference/intern tracking;
    # newer marshal formats can differ solely with Python object refcounts.
    return {name:marshal.dumps(functions[name],0) for name in REMOTE_FUNCTIONS}


class ArtifactSession:
    """Per-case append-only event authority and immutable byte archives."""
    def __init__(self, root, case_directory, profile, dependencies, *, identity=None):
        self.root = Path(root).resolve(); self.case_directory = Path(case_directory).resolve()
        require(self.case_directory.is_relative_to(self.root), 'case archive escaped ROOT')
        require(profile.get('schema') == PROFILE and profile.get('cache_paths') == cache_paths(self.root),
                'session profile differs from exact ROOT cache layout')
        self.profile, self.dependencies = deepcopy(profile), dependencies
        self.identity = deepcopy(identity); self.process_id = os.getpid()
        self.output = self.case_directory/'jit_artifacts'; self.output.mkdir(parents=True, exist_ok=True)
        require(self.output.resolve() == self.output, 'case artifact directory is a symlink escape')
        self.events = []; self._lock = threading.RLock(); self._patches = []
        self._source_envelopes = {}; self._loaded_modules = {}; self._native_files = {}
        self._binary_objects = {}; self._future_sources = {}; self._closed = False
        self._local = threading.local(); self._hook_owners = {}
        self._generated_files = {}; self._violations = []; self._module_authorities = {}
        self._stock_functions={};self._callable_identities={};self._guarded_calls={}
        self._helper_builds = {}; self._kernel_envelopes = {}; self._serialized = {}
        self._worker_messages={}
        self.hook_restoration_completed = False
        self.initialization=None;self.initialized_before_Torch_Triton_import=False
        self.stock_flags_initial=None;self.stock_flags_final=None

    def archive_bytes(self, value, kind):
        require(isinstance(value, bytes), 'artifact must contain actual bytes')
        require(isinstance(kind, str) and 0 < len(kind) <= 32
            and kind.replace('_', '').replace('-', '').isalnum(), 'unknown archive artifact kind')
        checksum = sha_bytes(value); path = self.output/'bytes'/(checksum+'.'+kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        require(path.resolve() == path and path.is_relative_to(self.case_directory), 'byte archive escaped case')
        with self._lock:
            if path.exists():
                require(path.read_bytes() == value, 'existing archive bytes changed')
            else:
                with path.open('xb') as stream:
                    stream.write(value)
        return dict(path=str(path.relative_to(self.root)), sha256=checksum, size_bytes=len(value))

    def archive_file(self, path, kind):
        path = check_cache_path(self.root, self.profile, path)
        before = path.stat(); payload = path.read_bytes(); after = path.stat()
        fields = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
        require(fields(before) == fields(after), 'generated artifact changed during archive read')
        return dict(path=str(path), sha256=sha_bytes(payload), size_bytes=len(payload),
                    archive=self.archive_bytes(payload, kind))

    def begin(self, kind, source_owner, inputs):
        with self._lock:
            require(not self._closed, 'event started after session close')
            row = dict(schema=EVENT, event_id=len(self.events), kind=kind, status='started',
                process_id=self.process_id, parent_observed=True, worker_full_trace=False,
                started_utc=datetime.now(timezone.utc).isoformat(), source_owner=deepcopy(source_owner),
                inputs=deepcopy(inputs), artifacts=[])
            self.events.append(row)
            return row

    def finish(self, event, *, artifacts=(), result=None):
        with self._lock:
            require(any(item is event for item in self.events) and event['status'] == 'started',
                    'event completion differs')
            event.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat(),
                artifacts=deepcopy(list(artifacts)), result=deepcopy(result or {}))
            self._persist_event(event)

    def fail(self, event, error):
        with self._lock:
            require(any(item is event for item in self.events) and event['status'] == 'started',
                    'event failure differs')
            event.update(status='failed', ended_utc=datetime.now(timezone.utc).isoformat(),
                         error_class=type(error).__name__, error=str(error))
            if isinstance(error,ValueError) and str(error).startswith('baseline exact JIT artifacts:'):
                self._violations.append(str(error))
            self._persist_event(event)

    def _persist_event(self, event):
        directory = self.output/'events'; directory.mkdir(exist_ok=True)
        path = directory/(str(event['event_id']).zfill(6)+'.json')
        atomic_json(path, event)
        # Completed generation is useful even when the containing policy case
        # later fails. The ledger references immutable bytes/events, not SR.
        if event['status'] == 'complete' and event['kind'] != REMOTE_EVENT:
            for artifact in event['artifacts']:
                if 'path' not in artifact or not Path(artifact['path']).is_absolute(): continue
                generated = check_cache_path(self.root, self.profile, artifact['path'])
                if artifact.get('sha256') is None: continue
                key = digest(dict(path=str(generated), sha256=artifact['sha256']))
                ledger = self.root/CACHE_RELATIVE/'provenance'/self.profile_digest/(key+'.json')
                ledger.parent.mkdir(parents=True, exist_ok=True)
                require(ledger.resolve() == ledger, 'provenance ledger symlink escape')
                value = dict(schema='r10_completed_stock_JIT_artifact_reference_v1',
                    profile_sha256=self.profile_digest, path=str(generated), sha256=artifact['sha256'],
                    artifact=deepcopy(artifact), event=path_pointer(self.root,path),
                    case_directory=str(self.case_directory), policy_outcome_used=False)
                # A new event may observe the same bytes; keep the first actual
                # provenance. Parallel children use an exclusive creation.
                try:
                    with ledger.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2)
                except FileExistsError:
                    old=json.loads(ledger.read_text())
                    require(old.get('path')==value['path'] and old.get('sha256')==value['sha256']
                        and old.get('profile_sha256')==self.profile_digest,'ledger provenance changed')
                self._generated_files[(str(generated),artifact['sha256'])] = path_pointer(self.root,ledger)

    @property
    def profile_digest(self): return digest(self.profile)

    def previous_artifact(self, path, checksum):
        path = check_cache_path(self.root,self.profile,path)
        key=digest(dict(path=str(path),sha256=checksum))
        ledger=self.root/CACHE_RELATIVE/'provenance'/self.profile_digest/(key+'.json')
        require(ledger.is_file() and ledger.resolve()==ledger,'unobserved same-run cache artifact: '+str(path))
        value=json.loads(ledger.read_text())
        require(value.get('schema')=='r10_completed_stock_JIT_artifact_reference_v1'
            and value.get('profile_sha256')==self.profile_digest and value.get('path')==str(path)
            and value.get('sha256')==checksum and value.get('policy_outcome_used') is False,
            'same-run provenance reference differs')
        event=read_pointer(self.root,value['event'])
        validate_event(self.root,event,self.profile,self.dependencies)
        require(event['status']=='complete' and value['artifact'] in event['artifacts'],
                'same-run provenance has no completed producer event')
        return path_pointer(self.root,ledger)

    def violation(self, message):
        self._violations.append(str(message))
        raise ValueError('baseline exact JIT artifacts: '+str(message))

    def source_owner(self, function, module_name, qualified_name):
        """Require an actual frozen Python owner and exact callable identity."""
        fn = getattr(function, '__func__', function)
        require(hasattr(fn, '__code__') and hasattr(fn, '__qualname__'), 'stock owner must be a Python callable')
        path = Path(inspect.getsourcefile(fn) or '').resolve()
        module=sys.modules.get(module_name)
        require(type(module) is ModuleType and fn.__globals__ is module.__dict__
            and Path(module.__file__).resolve()==path,'stock callable origin/source module registry/globals identity differs')
        role=(module_name,qualified_name)
        if role in self._stock_functions:
            require(self._stock_functions[role] is fn,'stock callable identity changed after capture')
        else:
            actual=module
            for part in qualified_name.split('.'):actual=getattr(actual,part)
            require(getattr(actual,'__func__',actual) is fn,'stock callable differs from its canonical owner slot')
            self._stock_functions[role]=fn
        require(fn.__module__ == module_name and fn.__qualname__ == qualified_name
            and path.is_file() and Path(fn.__code__.co_filename).resolve() == path
            and self.dependencies.get(str(path)) == file_sha(path),
            'stock callable origin/source bytes differ: '+module_name+'.'+qualified_name)
        return dict(module=module_name, qualified_name=qualified_name,
                    path=str(path), sha256=file_sha(path))

    def snapshot(self, *, complete=False, error=None):
        with self._lock:
            require(not complete or (self.hook_restoration_completed and not self._violations
                and all(event['status']!='started' for event in self.events)), 'incomplete observer cannot claim completion')
            return dict(schema=EXECUTION, profile_sha256=digest(self.profile), profile=deepcopy(self.profile),
                identity=deepcopy(self.identity), process_id=self.process_id, complete=complete,
                hook_restoration_completed=self.hook_restoration_completed,
                events=deepcopy(self.events), worker_full_trace=False,
                observer_source=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)),
                hook_owners=deepcopy(self._hook_owners),violations=deepcopy(self._violations),
                stock_callable_identities=deepcopy(self._callable_identities),
                initialization=deepcopy(self.initialization),
                initialized_before_Torch_Triton_import=self.initialized_before_Torch_Triton_import,
                stock_flags_initial=deepcopy(self.stock_flags_initial),stock_flags_final=deepcopy(self.stock_flags_final),
                worker_scope='parent observations plus explicitly bound stock worker sources only',
                error=None if error is None else dict(class_name=type(error).__name__, message=str(error)))

    def patch(self, owner, name, replacement, *, module, qualified_name):
        descriptor=inspect.getattr_static(owner,name)
        original=getattr(owner,name)
        binding=self.source_owner(original,module,qualified_name)
        key=module+'.'+qualified_name
        self._hook_owners[key]=binding
        fn=getattr(original,'__func__',original)
        self._callable_identities[key]=dict(actual_callable_identity=id(fn),actual_code_identity=id(fn.__code__),
            canonical_role=key,source=deepcopy(binding))
        if isinstance(descriptor,classmethod):
            @functools.wraps(replacement)
            def guarded(cls,*args,**kwargs):
                if os.getpid()!=self.process_id:return original(*args,**kwargs)
                return replacement(cls,*args,**kwargs)
            wrapped=classmethod(guarded)
        else:
            @functools.wraps(replacement)
            def guarded(*args,**kwargs):
                if os.getpid()!=self.process_id:return original(*args,**kwargs)
                return replacement(*args,**kwargs)
            wrapped=staticmethod(guarded) if isinstance(descriptor,staticmethod) else guarded
        setattr(owner,name,wrapped)
        self._guarded_calls[(id(owner),name)]=dict(owner=owner,wrapped=wrapped,guard=guarded,
            replacement=replacement,original=original)
        self._patches.append((owner,name,descriptor,wrapped))
        return original,binding

    def owned_patch_caller(self, frame, *, owner, name, replacement, original, caller_code):
        """Recognize one exact owned guard frame, never walk arbitrary callers."""
        bound=self._guarded_calls.get((id(owner),name))
        valid=(bound is not None and bound['owner'] is owner
            and bound['replacement'] is replacement and bound['original'] is original
            and inspect.getattr_static(owner,name) is bound['wrapped']
            and frame.f_code is bound['guard'].__code__
            and frame.f_locals.get('self') is self
            and frame.f_locals.get('replacement') is replacement
            and frame.f_locals.get('original') is original)
        if not valid:self.violation('worker IPC caller is not the exact current owned guard')
        frame=frame.f_back
        if frame is None or frame.f_code is not caller_code:
            self.violation('worker IPC owned guard caller is not stock submit')
        return frame

    def restore(self):
        failures=[]
        for owner,name,descriptor,wrapped in reversed(self._patches):
            if inspect.getattr_static(owner,name) is not wrapped:
                failures.append(str(owner)+'.'+name)
            setattr(owner,name,descriptor)
        self._patches.clear();self.hook_restoration_completed=not failures
        if failures:self.violation('stock hooks were replaced during observation: '+repr(failures))

    def authorize_python(self, registry_key, module, path, checksum):
        if registry_key==REMOTE_NAME:
            return self._authorize_remote_template(module,path,checksum)
        row=self._module_authorities.get(id(module))
        require(row is not None and row['module'] is module and row['name']==registry_key
            and row['path']==str(Path(path).resolve()) and row['sha256']==checksum
            and sys.modules.get(registry_key) is module,
            'unobserved or replaced generated Python module: '+str(path))
        return deepcopy(row['witness'])

    def _authorize_remote_template(self, module, path, checksum):
        """Verify a fixed stock-derived module after load; no writer claim."""
        names=dict(nn_entry='torch.distributed.nn',API='torch.distributed.nn.api.remote_module',
            instantiator='torch.distributed.nn.jit.instantiator',
            template='torch.distributed.nn.jit.templates.remote_module_template')
        modules={role:sys.modules.get(name) for role,name in names.items()}
        require(all(type(value) is ModuleType for value in modules.values()),'remote stock modules are not already loaded')
        instantiator=modules['instantiator'];api=modules['API']
        owner=self.source_owner(instantiator.instantiate_non_scriptable_remote_module_template,
            names['instantiator'],'instantiate_non_scriptable_remote_module_template')
        stock_sources={};raw_sources={}
        for role,value in modules.items():
            source=Path(value.__file__).resolve()
            require(source.as_posix().endswith('/'+REMOTE_SOURCES[role])
                and self.dependencies.get(str(source))==file_sha(source),'remote stock source is unbound: '+role)
            raw_sources[role]=source.read_bytes()
            stock_sources[role]=dict(path=str(source),sha256=sha_bytes(raw_sources[role]),
                archive=self.archive_bytes(raw_sources[role],'remote_stock_source'))
        expected,recipe=stock_remote_template_recipe(raw_sources)
        directory=Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH);path=check_remote_template_path(self.root,path,directory)
        require(type(instantiator._TEMP_DIR) is tempfile.TemporaryDirectory
            and instantiator._TEMP_DIR.name==str(directory)
            and directory.resolve()==directory and path.resolve()==path
            and directory.parent.resolve()==self.root/'runtime/framework/tmp',
            'remote stock TemporaryDirectory identity or canonical location differs')
        before=path.stat();payload=path.read_bytes();after=path.stat()
        require((before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==
            (after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev)
            and sha_bytes(payload)==checksum,'actual remote template bytes changed during verification')
        existing=self._module_authorities.get(id(module))
        event=None
        if existing is None:
            event=self.begin(REMOTE_EVENT,owner,dict(module_name=REMOTE_NAME,path=str(path),
                stock_directory=str(directory),stock_temporary_directory_identity=id(instantiator._TEMP_DIR),
                stock_temporary_directory_class='tempfile.TemporaryDirectory',stock_temporary_directory_name=instantiator._TEMP_DIR.name,
                stock_sources=stock_sources,recipe=recipe,actual_source=self.archive_bytes(payload,'remote_actual_source'),
                expected_source=self.archive_bytes(expected,'remote_expected_source'),optimization=sys.flags.optimize,
                code_representation='CPython_marshal_v0_without_reference_tracking',
                validation_scope='postload_fixed_stock_recipe_and_loaded_function_code',writer_observed=False,loader_call_observed=False))
        try:
            require(payload==expected,'actual remote Python bytes differ from fixed stock recipe')
            loader=getattr(module,'__loader__',None);spec=getattr(module,'__spec__',None)
            require(type(module) is ModuleType and module.__name__==REMOTE_NAME
                and sys.modules.get(REMOTE_NAME) is module and api._NON_SCRIPTABLE_REMOTE_MODULE_MODULE is module
                and module.__file__==str(path) and spec is not None and spec.name==REMOTE_NAME and spec.origin==str(path)
                and type(loader) is importlib.machinery.SourceFileLoader and spec.loader is loader
                and loader.name==REMOTE_NAME and loader.path==str(path),'remote module registry/stock loader/origin identity differs')
            torch=sys.modules.get('torch');rpc=sys.modules.get('torch.distributed.rpc')
            internal=sys.modules.get('torch._jit_internal');typing=sys.modules.get('typing')
            require(all(type(value) is ModuleType for value in (torch,rpc,internal,typing)),
                'remote canonical global modules are missing')
            globals_expected=dict(torch=torch,rpc=rpc,Tensor=torch.Tensor,Future=internal.Future,RRef=rpc.RRef,
                Tuple=typing.Tuple,__builtins__=builtins.__dict__)
            require(module.module_interface_cls is None and all(module.__dict__.get(k) is value
                for k,value in globals_expected.items()),'remote actual used global identities differ')
            expected_codes=remote_function_codes(expected,path,sys.flags.optimize)
            functions={name:getattr(module,name,None) for name in REMOTE_FUNCTIONS};codes={}
            for name,function in functions.items():
                require(type(function) is FunctionType and function.__name__==function.__qualname__==name
                    and function.__module__==REMOTE_NAME and function.__globals__ is module.__dict__
                    and function.__defaults__ is None and function.__kwdefaults__ is None and function.__closure__ is None
                    and function.__builtins__ is builtins.__dict__ and function.__code__.co_filename==str(path),
                    'remote loaded function identity/globals/defaults/closure differ: '+name)
                actual=marshal.dumps(function.__code__,0)
                require(actual==expected_codes[name],'remote loaded function code structure differs: '+name)
                codes[name]=dict(actual=self.archive_bytes(actual,'remote_loaded_code'),
                    expected=self.archive_bytes(expected_codes[name],'remote_expected_code'),
                    actual_function_identity=id(function),actual_code_identity=id(function.__code__))
            require(type(module._generated_methods) is list and len(module._generated_methods)==2
                and all(a is b for a,b in zip(module._generated_methods,[functions['forward_async'],functions['forward']])),
                'remote generated method order/identity differs')
            if existing is not None:
                require(existing['module'] is module and existing['name']==REMOTE_NAME
                    and existing['path']==str(path) and existing['sha256']==checksum
                    and existing['remote_loader'] is loader and existing['remote_tempdir'] is instantiator._TEMP_DIR
                    and all(existing['remote_functions'][name][0] is function
                        and existing['remote_functions'][name][1] is function.__code__
                        for name,function in functions.items()),'remote module/function identity changed after qualification')
                return deepcopy(existing['witness'])
            artifact=dict(path=str(path),sha256=checksum,size_bytes=len(payload),archive=event['inputs']['actual_source'])
            self.finish(event,artifacts=[artifact],result=dict(module_name=REMOTE_NAME,path=str(path),
                actual_module_identity=id(module),registry_identity=True,parent_API_alias_identity=id(module),
                actual_loader_identity=id(loader),spec_loader_identity=id(spec.loader),loader_class='_frozen_importlib_external.SourceFileLoader',
                function_codes=codes,global_bindings={k:dict(actual_identity=id(value),canonical_identity=id(value))
                    for k,value in globals_expected.items()},module_interface_cls_is_None=True,
                generated_method_identities=[id(functions['forward_async']),id(functions['forward'])],
                writer_observed=False,loader_call_observed=False,
                validation_scope='postload_fixed_stock_recipe_and_loaded_function_code'))
            binding=dict(schema=REMOTE_BINDING,module_name=REMOTE_NAME,path=str(path),sha256=checksum,
                actual_module_identity=id(module),registry_identity=True,verification_event=self._event_pointer(event))
            self._module_authorities[id(module)]=dict(module=module,name=REMOTE_NAME,path=str(path),sha256=checksum,
                remote_functions={name:(fn,fn.__code__) for name,fn in functions.items()},
                remote_loader=loader,remote_tempdir=instantiator._TEMP_DIR,witness=binding)
            return deepcopy(binding)
        except Exception as error:
            if event is not None and event['status']=='started':self.fail(event,error)
            else:self._violations.append(str(error))
            raise

    def authorize_native(self, path, checksum):
        row=self._native_files.get(str(Path(path).resolve()))
        require(row is not None and row['sha256']==checksum,'unobserved generated native library: '+str(path))
        return deepcopy(row)

    def _event_pointer(self,event):
        return path_pointer(self.root,self.output/'events'/(str(event['event_id']).zfill(6)+'.json'))

    def _python_input(self,key,path):
        path=check_cache_path(self.root,self.profile,path,kind='inductor')
        require(path.name==key+'.py' and path.parent.name==key[1:3], 'Python stock key/path differs')
        payload=path.read_bytes(); checksum=sha_bytes(payload)
        envelope=self._source_envelopes.get((key,str(path)))
        if envelope is None:
            prior=self.previous_artifact(path,checksum)
            return payload,dict(previous_artifact=prior)
        require(py_code_key(payload,envelope['extra'])==key, 'Python source differs from submitted key/extra')
        return payload,dict(source_envelope=envelope['event'],extra=deepcopy(envelope['extra_archive']))

    def _register_python(self,module,key,path,checksum,event,set_sys_modules):
        self._validate_python(module,key,path,set_sys_modules)
        registry=bool(sys.modules.get(module.__name__) is module)
        binding=dict(schema='r10_exact_observed_stock_generated_python_module_v1',
            module_name=module.__name__,key=key,path=str(Path(path).resolve()),sha256=checksum,
            actual_module_identity=id(module),registry_identity=registry,loader_event=self._event_pointer(event))
        self._module_authorities[id(module)]=dict(module=module,name=module.__name__,path=str(Path(path).resolve()),
            sha256=checksum,witness=binding)
        return binding

    def _validate_python(self,module,key,path,set_sys_modules):
        require(type(module) is ModuleType and module.__name__=='torch._inductor.runtime.compile_tasks.'+key
            and Path(module.__file__).resolve()==Path(path).resolve() and getattr(module,'key',None)==key,
            'stock loader returned a different module identity/origin')
        registry=bool(sys.modules.get(module.__name__) is module)
        require(not set_sys_modules or registry,'stock generated registry identity differs')

    def install_python_hooks(self):
        codecache=importlib.import_module('torch._inductor.codecache')
        tasks=importlib.import_module('torch._inductor.runtime.compile_tasks')
        original_write=codecache.write
        write_owner=self.source_owner(original_write,codecache.__name__,'write')
        @functools.wraps(original_write)
        def observed_write(content,extension,extra='',hash_type='code',specified_dir='',key=None):
            if extension!='py':return original_write(content,extension,extra,hash_type,specified_dir,key)
            require(hash_type=='code' and not specified_dir and key in (None,py_code_key(content,extra)),
                'unknown Python explicit-key/hash/directory mode')
            payload=content.encode() if isinstance(content,str) else content
            extra_bytes=extra.encode() if isinstance(extra,str) else extra
            expected=py_code_key(content,extra)
            event=self.begin('python_source_write',write_owner,dict(key=expected,
                requested_source=self.archive_bytes(payload,'requested_python'),extra=self.archive_bytes(extra_bytes,'extra')))
            try:
                actual_key,path=original_write(content,extension,extra,hash_type,specified_dir,key)
                artifact=self.archive_file(path,'python')
                require(actual_key==expected and py_code_key(Path(path).read_bytes(),extra)==expected,
                        'stock Python writer key/executed bytes differ')
                self.finish(event,artifacts=[artifact],result=dict(key=actual_key,path=str(Path(path).resolve())))
                self._source_envelopes[(expected,str(Path(path).resolve()))]=dict(event=self._event_pointer(event),
                    extra=extra,extra_archive=event['inputs']['extra'])
                return actual_key,path
            except Exception as error:self.fail(event,error);raise
        self.patch(codecache,'write',observed_write,module=codecache.__name__,qualified_name='write')
        original_reload=tasks._reload_python_module
        reload_owner=self.source_owner(original_reload,tasks.__name__,'_reload_python_module')
        @functools.wraps(original_reload)
        def observed_reload(key,path,set_sys_modules=True):
            payload,provenance=self._python_input(key,path)
            artifact=self.archive_file(path,'python')
            event=self.begin('python_module_exec',reload_owner,dict(key=key,path=str(Path(path).resolve()),
                set_sys_modules=bool(set_sys_modules),**provenance))
            try:
                module=original_reload(key,path,set_sys_modules)
                require(Path(path).read_bytes()==payload,'Python bytes changed during exec')
                self._validate_python(module,key,path,set_sys_modules)
                result=dict(module_name=module.__name__,key=key,path=str(Path(path).resolve()),
                    actual_module_identity=id(module),registry_identity=sys.modules.get(module.__name__) is module)
                self.finish(event,artifacts=[artifact],result=result)
                self._register_python(module,key,path,artifact['sha256'],event,set_sys_modules)
                return module
            except Exception as error:self.fail(event,error);raise
        self.patch(tasks,'_reload_python_module',observed_reload,module=tasks.__name__,qualified_name='_reload_python_module')
        require(codecache._reload_python_module is original_reload,'unreviewed codecache reload alias')
        # Alias restoration preserves the original exact module slot.
        self.patch(codecache,'_reload_python_module',observed_reload,module=tasks.__name__,qualified_name='_reload_python_module')
        original_load=codecache.PyCodeCache.load_by_key_path
        load_owner=self.source_owner(original_load,codecache.__name__,'PyCodeCache.load_by_key_path')
        @functools.wraps(original_load)
        def observed_load(cls,key,path,linemap=None,attrs=None):
            payload,provenance=self._python_input(key,path)
            prior=cls.modules_no_attr.get(path) if attrs is None else None
            if prior is not None:
                self.authorize_python(prior.__name__,prior,path,sha_bytes(payload))
            event=self.begin('python_module_load',load_owner,dict(key=key,path=str(Path(path).resolve()),
                attributes=attributes_schema(attrs),memory_hit=prior is not None,**provenance))
            try:
                module=original_load(key,path,linemap,attrs)
                require(Path(path).read_bytes()==payload and (prior is None or module is prior)
                    and any(item is module for item in cls.modules), 'Python load/module list differs')
                self._validate_python(module,key,path,True)
                artifact=self.archive_file(path,'python')
                self.finish(event,artifacts=[artifact],result=dict(module_name=module.__name__,key=key,
                    path=str(Path(path).resolve()),actual_module_identity=id(module),
                    registry_identity=sys.modules.get(module.__name__) is module))
                self._register_python(module,key,path,artifact['sha256'],event,True)
                return module
            except Exception as error:self.fail(event,error);raise
        self.patch(codecache.PyCodeCache,'load_by_key_path',observed_load,module=codecache.__name__,
            qualified_name='PyCodeCache.load_by_key_path')
        # after_deserialization bypasses PyCodeCache.write. Capture its actual
        # source before stock write_to_disk writes/loads it.
        output=importlib.import_module('torch._inductor.output_code')
        original_disk=output.CompiledFxGraph.write_to_disk
        disk_owner=self.source_owner(original_disk,output.__name__,'CompiledFxGraph.write_to_disk')
        @functools.wraps(original_disk)
        def observed_disk(graph,*args,**kwargs):
            source=graph.source_code;key=graph.cache_key
            require(py_code_key(source)==key,'FXGraph stored source/key differs')
            event=self.begin('python_deserialized_source',disk_owner,dict(key=key,
                requested_source=self.archive_bytes(source.encode(),'requested_python'),
                extra=self.archive_bytes(b'','extra')))
            try:
                result=original_disk(graph,*args,**kwargs)
                # This method returns path; load happens in after_deserialization.
                path=result
                artifact=self.archive_file(path,'python')
                require(py_code_key(Path(path).read_bytes())==key,'FXGraph executed source/key differs')
                self.finish(event,artifacts=[artifact],result=dict(key=key,path=str(Path(path).resolve())))
                self._source_envelopes[(key,str(Path(path).resolve()))]=dict(event=self._event_pointer(event),
                    extra='',extra_archive=event['inputs']['extra'])
                return result
            except Exception as error:self.fail(event,error);raise
        self.patch(output.CompiledFxGraph,'write_to_disk',observed_disk,module=output.__name__,
            qualified_name='CompiledFxGraph.write_to_disk')

    def install_helper_hooks(self):
        build=importlib.import_module('triton.runtime.build')
        driver=importlib.import_module('triton.backends.nvidia.driver')
        original_compile=build.compile_module_from_src
        compile_owner=self.source_owner(original_compile,build.__name__,'compile_module_from_src')
        @functools.wraps(original_compile)
        def observed_compile(src,name,library_dirs=None,include_dirs=None,libraries=None):
            platform=build.platform_key();key=helper_cache_key(src,platform)
            event=self.begin('native_helper_source',compile_owner,dict(name=name,key=key,platform_key=platform,
                requested_source=self.archive_bytes(src.encode(),'helper_c'),
                library_dirs=list(library_dirs or []),include_dirs=list(include_dirs or []),libraries=list(libraries or [])))
            previous=getattr(self._local,'helper',None)
            self._local.helper=dict(event=event,key=key,name=name,source=src,builds=[])
            try:
                result=original_compile(src,name,library_dirs,include_dirs,libraries)
                require(isinstance(result,ModuleType),'helper compiler returned non-module')
                self.finish(event,result=dict(module_name=result.__name__,actual_module_identity=id(result),
                    actual_path=str(Path(result.__file__).resolve()),build_events=deepcopy(self._local.helper['builds'])))
                return result
            except Exception as error:self.fail(event,error);raise
            finally:self._local.helper=previous
        self.patch(build,'compile_module_from_src',observed_compile,module=build.__name__,qualified_name='compile_module_from_src')
        require(driver.compile_module_from_src is original_compile,'unreviewed NVIDIA compile alias')
        self.patch(driver,'compile_module_from_src',observed_compile,module=build.__name__,qualified_name='compile_module_from_src')
        original_build=build._build
        build_owner=self.source_owner(original_build,build.__name__,'_build')
        @functools.wraps(original_build)
        def observed_build(name,src,srcdir,library_dirs,include_dirs,libraries):
            context=getattr(self._local,'helper',None)
            require(context is not None and name==context['name'] and Path(src).read_text()==context['source'],
                'helper build has no exact current source request')
            require(build.knobs.build.impl is None,'unreviewed custom Triton compiler implementation')
            event=self.begin('native_helper_build',build_owner,dict(name=name,key=context['key'],
                source=self.archive_bytes(Path(src).read_bytes(),'helper_c'),
                library_dirs=list(library_dirs),include_dirs=list(include_dirs),libraries=list(libraries),
                compiler_command_observed=True))
            previous_build=getattr(self._local,'helper_build',None);self._local.helper_build=event
            try:
                path=original_build(name,src,srcdir,library_dirs,include_dirs,libraries)
                payload=Path(path).read_bytes();archive=self.archive_bytes(payload,'native_so')
                self.finish(event,artifacts=[dict(archive=archive,sha256=sha_bytes(payload),size_bytes=len(payload))],
                    result=dict(actual_build_path=str(path),actual_build_output_sha256=sha_bytes(payload)))
                self._helper_builds[(context['key'],sha_bytes(payload))]=self._event_pointer(event)
                context['builds'].append(self._event_pointer(event))
                return path
            except Exception as error:self.fail(event,error);raise
            finally:self._local.helper_build=previous_build
        self.patch(build,'_build',observed_build,module=build.__name__,qualified_name='_build')
        subprocess=build.subprocess;original_check_call=subprocess.check_call
        def observed_check_call(argv,*args,**kwargs):
            if os.getpid()!=self.process_id or sys._getframe(1).f_code is not original_build.__code__:
                return original_check_call(argv,*args,**kwargs)
            event=getattr(self._local,'helper_build',None)
            require(event is not None and isinstance(argv,list) and all(isinstance(item,str) for item in argv),
                'stock helper compiler call has no current build event')
            compiler_path=Path(shutil.which(argv[0]) or argv[0]).resolve()
            require(compiler_path.is_file() and self.dependencies.get(str(compiler_path))==file_sha(compiler_path),
                'actual helper compiler executable is outside frozen closure')
            require('actual_compiler_command' not in event['inputs'],'multiple unexpected compiler invocations')
            event['inputs']['actual_compiler_command']=deepcopy(argv)
            event['inputs']['actual_compiler_executable']=dict(path=str(compiler_path),sha256=file_sha(compiler_path))
            return original_check_call(argv,*args,**kwargs)
        subprocess.check_call=observed_check_call
        self._patches.append((subprocess,'check_call',original_check_call,observed_check_call))
        original_load=build._load_module_from_path
        load_owner=self.source_owner(original_load,build.__name__,'_load_module_from_path')
        @functools.wraps(original_load)
        def observed_load(name,path):
            context=getattr(self._local,'helper',None)
            require(context is not None and context['name']==name,'native helper loader has no exact source request')
            resolved=check_cache_path(self.root,self.profile,path,kind='triton')
            expected_dir=base64.b32encode(bytes.fromhex(context['key'])).decode().rstrip('=')
            require(resolved.parent.name==expected_dir,'native helper cache key/path differs')
            artifact=self.archive_file(resolved,'native_so');checksum=artifact['sha256']
            build_event=self._helper_builds.get((context['key'],checksum))
            provenance=dict(actual_build_event=build_event) if build_event else dict(
                previous_artifact=self.previous_artifact(resolved,checksum))
            event=self.begin('native_helper_load',load_owner,dict(name=name,key=context['key'],
                platform_key=context['event']['inputs']['platform_key'],
                requested_source=context['event']['inputs']['requested_source'],path=str(resolved),**provenance))
            try:
                module=original_load(name,path)
                require(type(module) is ModuleType and Path(module.__file__).resolve()==resolved
                    and module.__spec__ is not None and Path(module.__spec__.origin).resolve()==resolved
                    and file_sha(resolved)==checksum,'loaded helper identity/origin/bytes differ')
                self.finish(event,artifacts=[artifact],result=dict(module_name=module.__name__,
                    actual_module_identity=id(module),spec_origin=str(resolved)))
                self._native_files[str(resolved)]=dict(schema='r10_exact_observed_generated_native_library_v1',
                    path=str(resolved),sha256=checksum,loader_event=self._event_pointer(event))
                self._module_authorities[id(module)]=dict(module=module,name=module.__name__,path=str(resolved),
                    sha256=checksum,witness=deepcopy(self._native_files[str(resolved)]))
                return module
            except Exception as error:self.fail(event,error);raise
        self.patch(build,'_load_module_from_path',observed_load,module=build.__name__,qualified_name='_load_module_from_path')

    def install_kernel_hooks(self):
        async_module=importlib.import_module('torch._inductor.async_compile')
        heuristic=importlib.import_module('torch._inductor.runtime.triton_heuristics')
        static=importlib.import_module('torch._inductor.runtime.static_cuda_launcher')
        compiler=importlib.import_module('triton.compiler.compiler')
        tasks=importlib.import_module('torch._inductor.runtime.compile_tasks')
        worker_owner=self.source_owner(tasks._worker_compile_triton,tasks.__name__,'_worker_compile_triton')
        self._hook_owners['stock_worker_compile']=worker_owner
        original_task=async_module.AsyncCompile.triton
        task_owner=self.source_owner(original_task,async_module.__name__,'AsyncCompile.triton')
        @functools.wraps(original_task)
        def observed_task(obj,kernel_name,source_code,*args,**kwargs):
            event=self.begin('triton_parent_task_source',task_owner,dict(kernel_name=kernel_name,
                source=self.archive_bytes(source_code.encode(),'triton_python'),
                worker_source=worker_owner,worker_full_trace=False,worker_PID_observed=False,
                cache_environment={key:os.environ.get(key) for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')},
                call_arguments=attributes_schema(dict(args=list(args),kwargs=kwargs))))
            # Submit source authority is immutable before stock call: synchronous
            # compilation can already load a GPU binary inside that call.
            self.finish(event,result=dict(parent_source_observed_before_stock_submit=True))
            self._kernel_envelopes[kernel_name]=dict(event=self._event_pointer(event),source=source_code)
            call=self.begin('triton_parent_task_return',task_owner,dict(source_event=self._event_pointer(event)))
            previous_task=getattr(self._local,'task_source',None);self._local.task_source=self._event_pointer(event)
            try:
                result=original_task(obj,kernel_name,source_code,*args,**kwargs)
                self.finish(call,result=dict(actual_return_type=type(result).__module__+'.'+type(result).__qualname__,
                    actual_return_identity=id(result),worker_full_trace=False))
                return result
            except Exception as error:self.fail(call,error);raise
            finally:self._local.task_source=previous_task
        self.patch(async_module.AsyncCompile,'triton',observed_task,module=async_module.__name__,qualified_name='AsyncCompile.triton')
        original_precompile=heuristic.CachingAutotuner.precompile
        precompile_owner=self.source_owner(original_precompile,heuristic.__name__,'CachingAutotuner.precompile')
        @functools.wraps(original_precompile)
        def observed_precompile(tuner,*args,**kwargs):
            name=tuner.inductor_meta.get('kernel_name',getattr(tuner.fn,'__name__',None))
            envelope=self._kernel_envelopes.get(name)
            require(envelope is not None and tuner.fn.src.strip() in envelope['source'],
                'autotuner has no exact parent-submitted kernel source')
            prior=getattr(self._local,'tuner',None);self._local.tuner=(tuner,envelope)
            event=self.begin('triton_parent_precompile',precompile_owner,dict(source_event=envelope['event'],
                kernel_name=name,arguments=attributes_schema(dict(args=list(args),kwargs=kwargs)),
                full_worker_trace=False))
            try:
                result=original_precompile(tuner,*args,**kwargs)
                self.finish(event,result=dict(actual_compile_results=len(tuner.compile_results),
                    actual_launchers=len(tuner.launchers),worker_full_trace=False))
                return result
            except Exception as error:self.fail(event,error);raise
            finally:self._local.tuner=prior
        self.patch(heuristic.CachingAutotuner,'precompile',observed_precompile,module=heuristic.__name__,
            qualified_name='CachingAutotuner.precompile')
        # The parent receives either normal or static CompileResults. It can
        # validate the same recipe even when worker __setstate__ bypassed init.
        for cls in (heuristic.StaticTritonCompileResult,heuristic.TritonCompileResult):
            original=cls.make_launcher;owner=self.source_owner(original,heuristic.__name__,cls.__name__+'.make_launcher')
            def make_wrapper(original,owner):
                @functools.wraps(original)
                def observed_make(result,*args,**kwargs):
                    provenance=self._kernel_provenance(result,compiler)
                    self._binary_objects[id(result.kernel)]=dict(object=result.kernel,**provenance)
                    event=self.begin('triton_compile_result',owner,provenance['inputs'])
                    self.finish(event,artifacts=provenance['artifacts'],result=dict(kernel_hash=result.kernel.hash,
                        actual_kernel_identity=id(result.kernel),worker_full_trace=False,
                        provenance_scope='parent-returned bytes with recomputed stock recipe and frozen worker source'))
                    self._binary_objects[id(result.kernel)]['event']=self._event_pointer(event)
                    return original(result,*args,**kwargs)
                return observed_make
            self.patch(cls,'make_launcher',make_wrapper(original,owner),module=heuristic.__name__,
                qualified_name=cls.__name__+'.make_launcher')
        original_static=static.StaticallyLaunchedCudaKernel.load_kernel
        static_owner=self.source_owner(original_static,static.__name__,'StaticallyLaunchedCudaKernel.load_kernel')
        @functools.wraps(original_static)
        def observed_static(kernel,device):
            if kernel.function is not None:return original_static(kernel,device)
            bound=self._binary_objects.get(id(kernel))
            require(bound is not None and bound['object'] is kernel,'static CUDA load bypassed observed compile result')
            artifact=self.archive_file(kernel.cubin_path,'cubin')
            require(artifact['sha256']==bound['binary_sha256'] and
                (kernel.cubin_raw is None or sha_bytes(kernel.cubin_raw)==artifact['sha256']),
                'static CUDA actual bytes differ from observed compile result')
            event=self.begin('static_cuda_binary_load',static_owner,dict(compile_result_event=bound['event'],
                kernel_hash=kernel.hash,name=kernel.name,shared=kernel.shared,device=device,
                actual_kernel_identity=id(kernel),preload_cubin_path=str(kernel.cubin_path)))
            try:
                result=original_static(kernel,device)
                require(kernel.function is not None and kernel.cubin_path is None and kernel.cubin_raw is None,
                    'static stock loader handles/cleared fields differ')
                self.finish(event,artifacts=[artifact],result=dict(actual_function_handle=kernel.function,
                    n_regs=kernel.n_regs,n_spills=kernel.n_spills,cubin_path_cleared=True,cubin_raw_cleared=True))
                return result
            except Exception as error:self.fail(event,error);raise
        self.patch(static.StaticallyLaunchedCudaKernel,'load_kernel',observed_static,module=static.__name__,
            qualified_name='StaticallyLaunchedCudaKernel.load_kernel')
        original_handles=compiler.CompiledKernel._init_handles
        handles_owner=self.source_owner(original_handles,compiler.__name__,'CompiledKernel._init_handles')
        @functools.wraps(original_handles)
        def observed_handles(kernel):
            if kernel.module is not None:return original_handles(kernel)
            bound=self._binary_objects.get(id(kernel))
            require(bound is not None and bound['object'] is kernel,'CUDA load bypassed observed compile result')
            require(isinstance(kernel.kernel,bytes) and sha_bytes(kernel.kernel)==bound['binary_sha256'],
                'nonstatic CUDA bytes differ from observed compile result')
            event=self.begin('nonstatic_cuda_binary_load',handles_owner,dict(compile_result_event=bound['event'],
                kernel_hash=kernel.hash,name=kernel.name,shared=kernel.metadata.shared,
                actual_kernel_identity=id(kernel),binary=self.archive_bytes(kernel.kernel,'cubin')))
            try:
                result=original_handles(kernel)
                require(kernel.module is not None and kernel.function is not None,'CUDA native handles are missing')
                self.finish(event,result=dict(actual_module_handle=kernel.module,actual_function_handle=kernel.function,
                    n_regs=kernel.n_regs,n_spills=kernel.n_spills))
                return result
            except Exception as error:self.fail(event,error);raise
        self.patch(compiler.CompiledKernel,'_init_handles',observed_handles,module=compiler.__name__,
            qualified_name='CompiledKernel._init_handles')

    def _kernel_provenance(self,result,compiler):
        context=getattr(self._local,'tuner',None)
        require(context is not None,'compiled result has no parent-observed autotuner context')
        tuner,envelope=context;kernel=result.kernel;key=kernel.hash
        directory=Path(self.profile['cache_paths']['triton'])/base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
        check_cache_path(self.root,self.profile,directory,kind='triton')
        name=kernel.name
        metadata_path=directory/(name+'.json')
        # Normal CompiledKernel.name can differ from original fn name; the cache
        # metadata filename uses the source name, which static kernels retain.
        if not metadata_path.exists():metadata_path=directory/(tuner.fn.__name__[:150]+'.json')
        metadata=json.loads(metadata_path.read_text())
        require(metadata.get('hash')==key,'kernel metadata hash differs from returned object')
        target=compiler.GPUTarget(**metadata['target']);backend=compiler.make_backend(target)
        cm=result.compile_meta
        source=compiler.ASTSource(tuner.fn,cm['signature'],cm['constants'],cm['configs'][0])
        options,options_witness=stock_cuda_compile_options(backend,cm,metadata)
        env=compiler.get_cache_invalidating_env_vars()
        recipe=dict(triton_key=compiler.triton_key(),source_hash=source.hash(),backend_hash=backend.hash(),
            options_hash=options.hash(),sorted_environment=str(sorted(env.items())))
        expression='-'.join(recipe[field] for field in ('triton_key','source_hash','backend_hash','options_hash','sorted_environment'))
        require(sha_bytes(expression.encode())==key,'returned worker kernel stock compiler recipe differs')
        # Validate the metadata group's exact child paths before reading any
        # child, instead of trusting a cache-directory name.
        group_path=directory/('__grp__'+metadata_path.name)
        group=json.loads(group_path.read_text())
        children=group.get('child_paths')
        require(set(group)=={'child_paths'} and isinstance(children,dict)
            and metadata_path.name in children,'kernel metadata group differs')
        files=[]
        for child,child_path in children.items():
            require(isinstance(child,str) and Path(child).name==child and isinstance(child_path,str)
                and Path(child_path).is_absolute() and Path(child_path)==directory/child
                and Path(child_path).resolve()==directory/child,'kernel metadata group child escaped directory')
            files.append(self.archive_file(child_path,'kernel_artifact'))
        cubins=[item for item in files if item['path'].endswith('.cubin')]
        require(len(cubins)==1,'kernel result must contain one actual CUDA binary')
        binary=cubins[0];raw=getattr(kernel,'cubin_raw',None)
        if raw is None and hasattr(kernel,'kernel'):raw=kernel.kernel
        require(raw is None or isinstance(raw,bytes) and sha_bytes(raw)==binary['sha256'],
            'returned worker binary bytes differ from actual cache group')
        return dict(binary_sha256=binary['sha256'],artifacts=files,
            inputs=dict(source_event=envelope['event'],kernel_hash=key,recipe=recipe,
                metadata_group=self.archive_file(group_path,'metadata_group'),
                source_hash_schema=attributes_schema(dict(signature=source.signature,constants=source.constants,attrs=source.attrs)),
                compile_meta_schema=attributes_schema(cm),target=metadata['target'],
                stock_compile_options=options_witness,
                worker_frozen_source=self._hook_owners['stock_worker_compile'],
                worker_full_trace=False,worker_PID_observed=False))

    def install_pickle_hooks(self):
        """Observe the two stock cache serializers and guard before unpickling.

        Other application pickle uses are untouched. Unknown cache content is
        rejected before stock pickle execution, including remote cache input.
        """
        import pickle
        codecache=importlib.import_module('torch._inductor.codecache')
        aot=importlib.import_module('torch._functorch._aot_autograd.autograd_cache')
        serializers={}
        for fn,module,name in ((codecache.FxGraphCache._save_graph,codecache.__name__,'FxGraphCache._save_graph'),
            (aot.AOTAutogradCache.save,aot.__name__,'AOTAutogradCache.save')):
            serializers[fn.__code__]=self.source_owner(fn,module,name)
        iterator=codecache.GuardedCache.iterate_over_candidates.__func__
        iterator_owner=self.source_owner(iterator,codecache.__name__,'GuardedCache.iterate_over_candidates')
        self._hook_owners['cache_candidate_before_unpickle']=iterator_owner
        original_dumps=pickle.dumps;original_loads=pickle.loads
        pool_module=importlib.import_module('torch._inductor.compile_worker.subproc_pool')
        submit=pool_module.SubprocPool.submit
        self._hook_owners['stock_pool_submit']=self.source_owner(submit,pool_module.__name__,'SubprocPool.submit')
        submit_code=submit.__code__
        read_thread=pool_module.SubprocPool._read_thread
        read_owner=self.source_owner(read_thread,pool_module.__name__,'SubprocPool._read_thread')
        pickler_load=pool_module.SubprocPickler.loads
        self._hook_owners['stock_pickler_loads']=self.source_owner(pickler_load,pool_module.__name__,'SubprocPickler.loads')
        pickler_load_code=pickler_load.__code__
        original_send=pool_module._send_msg
        send_owner=self.source_owner(original_send,pool_module.__name__,'_send_msg')
        @functools.wraps(original_send)
        def observed_send(write_pipe,job_id,job_data=b''):
            frame=sys._getframe(1);source=getattr(self._local,'task_source',None)
            if os.getpid()!=self.process_id or source is None:
                return original_send(write_pipe,job_id,job_data)
            frame=self.owned_patch_caller(frame,owner=pool_module,name='_send_msg',
                replacement=observed_send,original=original_send,caller_code=submit_code)
            pool=frame.f_locals['self'];future=pool.pending_futures[job_id]
            event=self.begin('stock_worker_task_message',send_owner,dict(source_event=source,
                actual_job_id=job_id,actual_pool_server_process_id=pool.process.pid,
                actual_parent_future_identity=id(future),payload=self.archive_bytes(job_data,'worker_task_pickle'),
                worker_process_id_observed=False,worker_full_trace=False))
            try:
                # Register before sending: the read thread may receive a fast
                # result before original_send returns.
                message=dict(event=event,future=future,ready=threading.Event())
                self._worker_messages[(id(pool),job_id)]=message
                result=original_send(write_pipe,job_id,job_data)
                self.finish(event,result=dict(actual_parent_send_returned=True))
                return result
            except Exception as error:self.fail(event,error);raise
            finally:
                if (id(pool),job_id) in self._worker_messages:
                    self._worker_messages[(id(pool),job_id)]['ready'].set()
        self.patch(pool_module,'_send_msg',observed_send,module=pool_module.__name__,qualified_name='_send_msg')
        def observed_dumps(obj,*args,**kwargs):
            if os.getpid()!=self.process_id:return original_dumps(obj,*args,**kwargs)
            frame=sys._getframe(1);owner=serializers.get(frame.f_code)
            if owner is None:return original_dumps(obj,*args,**kwargs)
            event=self.begin('stock_cache_serialization',owner,dict(actual_object_schema=attributes_schema(obj),
                arguments=attributes_schema(dict(args=list(args),kwargs=kwargs))))
            try:
                payload=original_dumps(obj,*args,**kwargs);archive=self.archive_bytes(payload,'cache_pickle')
                self.finish(event,artifacts=[dict(archive=archive,sha256=sha_bytes(payload),size_bytes=len(payload))],
                    result=dict(serialized_bytes_sha256=sha_bytes(payload),pickle_executed_by_validator=False))
                self._serialized[sha_bytes(payload)]=self._event_pointer(event)
                directory=self.root/CACHE_RELATIVE/'serializations'/self.profile_digest
                directory.mkdir(parents=True,exist_ok=True);path=directory/(sha_bytes(payload)+'.json')
                require(path.resolve()==path,'serializer provenance symlink escape')
                value=dict(schema='r10_stock_cache_serialization_reference_v1',profile_sha256=self.profile_digest,
                    sha256=sha_bytes(payload),event=self._event_pointer(event),policy_outcome_used=False)
                try:
                    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2)
                except FileExistsError:pass
                return payload
            except Exception as error:self.fail(event,error);raise
        def observed_loads(payload,*args,**kwargs):
            if os.getpid()!=self.process_id:return original_loads(payload,*args,**kwargs)
            frame=sys._getframe(1)
            if frame.f_code is pickler_load_code:
                outer=frame.f_back
                if outer is not None and outer.f_code is read_thread.__code__:
                    pool=outer.f_locals['self'];job_id=outer.f_locals['job_id']
                    message=self._worker_messages.get((id(pool),job_id))
                    if message is not None:
                        require(pool.pending_futures.get(job_id) is message['future'],
                            'worker IPC future identity differs')
                        # Sending returns almost immediately. Hold only the
                        # metadata lock until its immutable event is published.
                        message['ready'].wait()
                        with self._lock:
                            send=message['event']
                            require(send['status']=='complete','worker result arrived before complete send witness')
                            event=self.begin('stock_worker_result_before_unpickle',read_owner,dict(
                                task_message_event=self._event_pointer(send),actual_job_id=job_id,
                                actual_pool_server_process_id=pool.process.pid,
                                actual_parent_future_identity=id(message['future']),
                                payload=self.archive_bytes(payload,'worker_result_pickle'),
                                before_original_pickle_loads=True,worker_process_id_observed=False,worker_full_trace=False))
                        try:
                            value=original_loads(payload,*args,**kwargs)
                            self.finish(event,result=dict(actual_return_type=type(value).__module__+'.'+type(value).__qualname__,
                                actual_return_identity=id(value),worker_full_trace=False))
                            return value
                        except Exception as error:self.fail(event,error);raise
            if frame.f_code is not iterator.__code__:return original_loads(payload,*args,**kwargs)
            require(isinstance(payload,bytes),'stock cache pickle payload is not exact bytes')
            checksum=sha_bytes(payload)
            path=self.root/CACHE_RELATIVE/'serializations'/self.profile_digest/(checksum+'.json')
            try:
                require(path.is_file() and path.resolve()==path,'unobserved cache pickle before execution')
                reference=json.loads(path.read_text())
                require(reference.get('schema')=='r10_stock_cache_serialization_reference_v1'
                    and reference.get('profile_sha256')==self.profile_digest and reference.get('sha256')==checksum
                    and reference.get('policy_outcome_used') is False,'cache serializer provenance differs')
                producer=read_pointer(self.root,reference['event'])
                validate_event(self.root,producer,self.profile,self.dependencies)
                require(producer['kind']=='stock_cache_serialization' and producer['status']=='complete'
                    and producer['result']['serialized_bytes_sha256']==checksum,
                    'cache pickle lacks complete observed serializer event')
            except Exception as error:
                self.violation('cache content rejected before unpickle: '+str(error))
            event=self.begin('stock_cache_before_unpickle',iterator_owner,dict(serialization_event=reference['event'],
                payload=self.archive_bytes(payload,'cache_pickle'),before_original_pickle_loads=True))
            try:
                value=original_loads(payload,*args,**kwargs)
                self.finish(event,result=dict(actual_return_type=type(value).__module__+'.'+type(value).__qualname__,
                    actual_return_identity=id(value)))
                return value
            except Exception as error:self.fail(event,error);raise
        # These are CPython extension callables, so their execution owner is
        # the exact frozen stock caller above, not a fictitious Python owner.
        for name,replacement,original in [('dumps',observed_dumps,original_dumps),('loads',observed_loads,original_loads)]:
            setattr(pickle,name,replacement);self._patches.append((pickle,name,original,replacement))
        self._hook_owners.update({'serializer:'+owner['qualified_name']:owner for owner in serializers.values()})

    def install(self):
        require(not self._patches and not self._closed,'observer already installed/closed')
        require(os.environ.get('TORCHINDUCTOR_CACHE_DIR')==self.profile['cache_paths']['inductor']
            and os.environ.get('TRITON_CACHE_DIR')==self.profile['cache_paths']['triton'],
            'cache environment must be initialized before stock imports')
        try:
            # The driver captures its C-helper compiler alias at import time.
            # Observe it before other stock imports can initialize that driver.
            self.install_helper_hooks();self.install_python_hooks();self.install_kernel_hooks();self.install_pickle_hooks()
            self.stock_flags_initial=stock_flags()
        except Exception:
            self.restore();raise
        return self

    def close(self,error=None):
        if self._closed:return self.receipt_pointer
        try:
            if self.stock_flags_initial is not None:
                self.stock_flags_final=stock_flags()
                require(self.stock_flags_final==self.stock_flags_initial,'stock compile/autotune/static flags changed')
            self.restore()
        except BaseException as qualification_error:
            # A changed flag is an infra violation; restoration is mandatory
            # even when qualification itself raises.
            if self._patches:self.restore()
            if error is None:error=qualification_error
            raise
        finally:
            self._closed=True
            complete=error is None and not self._violations and all(e['status']!='started' for e in self.events)
            value=self.snapshot(complete=complete,error=error)
            path=self.output/'execution.json';atomic_json(path,value)
            self.receipt_pointer=path_pointer(self.root,path)
        return self.receipt_pointer


def read_pointer(root,pointer):
    require(isinstance(pointer,dict) and set(pointer)>={'path','sha256'},'archive pointer schema differs')
    path=Path(root)/pointer['path']
    require(not Path(pointer['path']).is_absolute() and path.resolve()==path and path.is_relative_to(Path(root).resolve())
        and path.is_file() and file_sha(path)==pointer['sha256'],'archive pointer bytes/path differ')
    return json.loads(path.read_text())


def stock_flags():
    config=importlib.import_module('torch._inductor.config')
    return {name:getattr(config,name) for name in ('use_static_cuda_launcher','compile_threads',
        'max_autotune','max_autotune_gemm','max_autotune_pointwise')}


def read_archive(root,archive):
    require(isinstance(archive,dict) and set(archive)=={'path','sha256','size_bytes'},'byte archive schema differs')
    require(type(archive['size_bytes']) is int and archive['size_bytes']>=0,'byte archive size type differs')
    path=Path(root)/archive['path']
    require(not Path(archive['path']).is_absolute() and path.resolve()==path and path.is_relative_to(Path(root).resolve())
        and path.is_file(),'byte archive path differs')
    payload=path.read_bytes()
    require(len(payload)==archive['size_bytes'] and sha_bytes(payload)==archive['sha256'],'byte archive SHA/size differs')
    return payload


def validate_event(root,event,profile,dependencies):
    """Pure historical verification. Never imports or executes generated code."""
    require(event.get('schema')==EVENT and type(event.get('event_id')) is int and event['event_id']>=0
        and event.get('parent_observed') is True and event.get('worker_full_trace') is False
        and type(event.get('process_id')) is int and event['process_id']>0
        and event.get('status') in ('started','complete','failed'),'event schema/scope differs')
    owner=event.get('source_owner',{})
    require(set(owner)=={'module','qualified_name','path','sha256'}
        and dependencies.get(str(Path(owner['path']).resolve()))==owner['sha256']
        and Path(owner['path']).is_absolute(),'stock event owner is not frozen')
    inputs=event.get('inputs',{});kind=event['kind']
    def verify_embedded_archives(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256','size_bytes'}:read_archive(root,value)
            elif set(value)=={'path','sha256'} and not Path(value['path']).is_absolute():read_pointer(root,value)
            else:
                for item in value.values():verify_embedded_archives(item)
        elif isinstance(value,list):
            for item in value:verify_embedded_archives(item)
    verify_embedded_archives(inputs)
    require((owner['module'],owner['qualified_name']) in EVENT_ROLES.get(kind,()),
            'event owner has the wrong exact stock role')
    require(owner['path'].endswith('/'+owner['module'].replace('.','/')+'.py'),
            'event owner module path differs from exact stock module')
    allowed={'python_source_write','python_deserialized_source','python_module_exec','python_module_load',
        'native_helper_source','native_helper_build','native_helper_load','triton_parent_task_source',
        'triton_parent_task_return','triton_parent_precompile','triton_compile_result',
        'static_cuda_binary_load','nonstatic_cuda_binary_load','stock_cache_serialization','stock_cache_before_unpickle',
        'stock_worker_task_message','stock_worker_result_before_unpickle',REMOTE_EVENT}
    require(kind in allowed,'unknown event kind')
    for artifact in event.get('artifacts',[]):
        payload=read_archive(root,artifact['archive'])
        require(artifact.get('sha256')==sha_bytes(payload) and artifact.get('size_bytes')==len(payload),
            'event artifact differs from archived bytes')
        if 'path' in artifact:
            if kind==REMOTE_EVENT:check_remote_template_path(root,artifact['path'],inputs['stock_directory'])
            else:check_cache_path(root,profile,artifact['path'])
    if event['status']!='complete':return
    require(event.get('ended_utc')>=event['started_utc'] and isinstance(event.get('result'),dict),'event terminal clock/result differs')
    if kind==REMOTE_EVENT:
        validate_remote_template_event(root,event,dependencies)
    elif kind in ('python_source_write','python_deserialized_source'):
        source=read_archive(root,inputs['requested_source']);extra=read_archive(root,inputs['extra'])
        require(py_code_key(source,extra)==inputs['key'] and event['result']['key']==inputs['key'],
            'Python request source/extra/key recipe differs')
        require(len(event['artifacts'])==1 and py_code_key(read_archive(root,event['artifacts'][0]['archive']),extra)==inputs['key'],
            'executed Python bytes/key recipe differs')
        require(event['artifacts'][0]['path']==str(Path(profile['cache_paths']['inductor'])/inputs['key'][1:3]/(inputs['key']+'.py')),
            'Python key/path recipe differs')
    elif kind in ('python_module_exec','python_module_load'):
        require(len(event['artifacts'])==1 and event['result']['key']==inputs['key']
            and event['result']['module_name']=='torch._inductor.runtime.compile_tasks.'+inputs['key']
            and event['result']['path']==inputs['path'] and type(event['result']['actual_module_identity']) is int,
            'generated Python module identity witness differs')
        require(inputs['path']==event['artifacts'][0]['path'],'Python load file differs')
        require(inputs['path']==str(Path(profile['cache_paths']['inductor'])/inputs['key'][1:3]/(inputs['key']+'.py')),
            'loaded Python key/path recipe differs')
        if 'source_envelope' in inputs:
            producer=read_pointer(root,inputs['source_envelope']);validate_event(root,producer,profile,dependencies)
            require(producer['kind'] in ('python_source_write','python_deserialized_source')
                and producer['status']=='complete' and producer['inputs']['key']==inputs['key']
                and py_code_key(read_archive(root,event['artifacts'][0]['archive']),read_archive(root,producer['inputs']['extra']))==inputs['key'],
                'Python load source envelope differs')
        else:validate_previous(root,inputs['previous_artifact'],profile,dependencies,event['artifacts'][0])
        require(event['result'].get('registry_identity') is True,'generated parent module registry identity differs')
    elif kind in ('native_helper_source','native_helper_load'):
        source=read_archive(root,inputs['requested_source']).decode()
        require(helper_cache_key(source,inputs['platform_key'])==inputs['key'],'helper key/source/platform differs')
        if kind=='native_helper_load':
            require(len(event['artifacts'])==1 and event['result']['spec_origin']==inputs['path']
                and event['artifacts'][0]['path']==inputs['path'],'helper native load bytes/origin differs')
            expected_dir=Path(profile['cache_paths']['triton'])/base64.b32encode(bytes.fromhex(inputs['key'])).decode().rstrip('=')
            require(Path(inputs['path']).parent==expected_dir and Path(inputs['path']).name.startswith(inputs['name']+'.')
                and Path(inputs['path']).name.endswith('.so'),'helper stock key/path recipe differs')
            if inputs.get('actual_build_event'):
                producer=read_pointer(root,inputs['actual_build_event']);validate_event(root,producer,profile,dependencies)
                require(producer['kind']=='native_helper_build' and producer['status']=='complete'
                    and producer['inputs']['key']==inputs['key']
                    and read_archive(root,producer['inputs']['source'])==read_archive(root,inputs['requested_source'])
                    and producer['result']['actual_build_output_sha256']==event['artifacts'][0]['sha256'],
                    'helper binary lacks actual observed build bytes')
            else:validate_previous(root,inputs['previous_artifact'],profile,dependencies,event['artifacts'][0])
    elif kind=='native_helper_build':
        require(len(event['artifacts'])==1 and event['result']['actual_build_output_sha256']==event['artifacts'][0]['sha256']
            and inputs.get('compiler_command_observed') is True,'helper build output or observation scope differs')
        command=inputs.get('actual_compiler_command');program=inputs.get('actual_compiler_executable',{})
        require(isinstance(command,list) and command and all(isinstance(item,str) for item in command)
            and dependencies.get(program.get('path'))==program.get('sha256'),
            'helper actual compiler command/executable is unbound')
        read_archive(root,inputs['source'])
    elif kind=='triton_parent_task_source':
        read_archive(root,inputs['source'])
        require(inputs.get('worker_full_trace') is False and inputs.get('worker_PID_observed') is False
            and inputs.get('cache_environment')==dict(TORCHINDUCTOR_CACHE_DIR=profile['cache_paths']['inductor'],
                TRITON_CACHE_DIR=profile['cache_paths']['triton']),'parent task scope/cache environment differs')
        validate_worker_owner(inputs['worker_source'],dependencies)
    elif kind=='triton_compile_result':
        recipe=inputs['recipe'];require(set(recipe)=={'triton_key','source_hash','backend_hash','options_hash','sorted_environment'},
            'kernel compiler recipe schema differs')
        expression='-'.join(recipe[field] for field in ('triton_key','source_hash','backend_hash','options_hash','sorted_environment'))
        require(sha_bytes(expression.encode())==inputs['kernel_hash']==event['result']['kernel_hash'],
            'kernel compiler hash recipe differs')
        producer=read_pointer(root,inputs['source_event']);validate_event(root,producer,profile,dependencies)
        require(producer['kind']=='triton_parent_task_source' and producer['status']=='complete'
            and inputs['worker_full_trace'] is False and inputs['worker_PID_observed'] is False,
            'kernel result has no actual parent source task')
        validate_worker_owner(inputs['worker_frozen_source'],dependencies)
        require(inputs['worker_frozen_source']==producer['inputs']['worker_source'],'kernel worker owner differs from submitted task')
        group=inputs['metadata_group'];payload=read_archive(root,group['archive'])
        require(group['sha256']==sha_bytes(payload),'metadata group bytes differ')
        children=json.loads(payload)['child_paths'];actual={Path(a['path']).name:a['path'] for a in event['artifacts']}
        require(children==actual and len(actual)==len(event['artifacts']),'kernel metadata group children differ')
        expected_dir=Path(profile['cache_paths']['triton'])/base64.b32encode(bytes.fromhex(inputs['kernel_hash'])).decode().rstrip('=')
        require(Path(group['path']).parent==expected_dir,'kernel compiler key/path recipe differs')
        for artifact in event['artifacts']:
            require(Path(artifact['path']).parent==Path(group['path']).parent,'kernel child path escaped metadata group')
        metadata=[a for a in event['artifacts'] if a['path'].endswith('.json')]
        require(len(metadata)==1 and json.loads(read_archive(root,metadata[0]['archive']))['hash']==inputs['kernel_hash'],
            'kernel metadata hash differs')
        actual_metadata=json.loads(read_archive(root,metadata[0]['archive']))
        witness=inputs['stock_compile_options'];request=witness['request']
        require(set(witness)=={'schema','request','projected_options','all_projected_fields_match_actual_metadata'}
            and witness['schema']=='r10_stock_CUDA_four_option_replay_v1'
            and witness['all_projected_fields_match_actual_metadata'] is True
            and set(request)=={'num_warps','num_stages','debug','sanitize_overflow'}
            and request['sanitize_overflow'] is False,'kernel stock CUDA option witness differs')
        fields=inputs['compile_meta_schema']['fields']
        require(fields['device_type']==dict(type='str',value='cuda')
            and type(fields['cc']['value']) is int
            and actual_metadata['target']==inputs['target']==dict(backend='cuda',arch=fields['cc']['value'],warp_size=32),
            'kernel stock CUDA target witness differs')
        for key,kind in [('num_warps','int'),('num_stages','int'),('debug','bool')]:
            require(fields[key]==dict(type=kind,value=request[key])
                and type(request[key]).__name__==kind,'kernel compile_meta options differ')
        projected=witness['projected_options']
        require(isinstance(projected,dict) and projected and all(key in projected
            and json.dumps(projected[key],sort_keys=True,allow_nan=False)==json.dumps(value,sort_keys=True,allow_nan=False)
            for key,value in request.items()),'kernel options request/projection differs')
        for key,value in projected.items():
            require(key in actual_metadata and json.dumps(actual_metadata[key],sort_keys=True,allow_nan=False)==
                json.dumps(value,sort_keys=True,allow_nan=False),'kernel metadata option projection differs: '+key)
        require(event['result'].get('worker_full_trace') is False,'unobserved worker tracing claimed')
    elif kind in ('static_cuda_binary_load','nonstatic_cuda_binary_load'):
        producer=read_pointer(root,inputs['compile_result_event']);validate_event(root,producer,profile,dependencies)
        require(producer['kind']=='triton_compile_result' and producer['status']=='complete'
            and producer['inputs']['kernel_hash']==inputs['kernel_hash']
            and producer['result']['actual_kernel_identity']==inputs['actual_kernel_identity'],
            'GPU load object differs from observed parent compile result')
        binaries=[a for a in producer['artifacts'] if a['path'].endswith('.cubin')]
        require(len(binaries)==1,'GPU load lacks exact binary group')
        payload=read_archive(root,event['artifacts'][0]['archive'] if kind=='static_cuda_binary_load' else inputs['binary'])
        require(sha_bytes(payload)==binaries[0]['sha256'] and event['result'].get('actual_function_handle') is not None,
            'actual GPU loaded binary/handle differs')
        if kind=='static_cuda_binary_load':require(event['result'].get('cubin_path_cleared') is True
            and event['result'].get('cubin_raw_cleared') is True,'static postload fields differ')
    elif kind=='stock_cache_serialization':
        require(len(event['artifacts'])==1 and event['result']['serialized_bytes_sha256']==event['artifacts'][0]['sha256']
            and event['result']['pickle_executed_by_validator'] is False,'cache serializer bytes differ')
    elif kind=='stock_cache_before_unpickle':
        producer=read_pointer(root,inputs['serialization_event']);validate_event(root,producer,profile,dependencies)
        require(producer['kind']=='stock_cache_serialization' and producer['status']=='complete'
            and producer['result']['serialized_bytes_sha256']==sha_bytes(read_archive(root,inputs['payload']))
            and inputs['before_original_pickle_loads'] is True,'unpickle has no previous exact serializer authority')
    elif kind in ('triton_parent_task_return','triton_parent_precompile'):
        producer=read_pointer(root,inputs['source_event']);validate_event(root,producer,profile,dependencies)
        require(producer['kind']=='triton_parent_task_source' and producer['status']=='complete','parent task reference differs')
        require(event['result'].get('worker_full_trace') is False,'parent task return falsely claims full worker tracing')
        if kind=='triton_parent_precompile':require(inputs.get('full_worker_trace') is False,
            'parent precompile falsely claims full worker tracing')
    elif kind in ('stock_worker_task_message','stock_worker_result_before_unpickle'):
        read_archive(root,inputs['payload'])
        require(type(inputs['actual_job_id']) is int and inputs['actual_job_id']>=0
            and type(inputs['actual_pool_server_process_id']) is int and inputs['actual_pool_server_process_id']>0
            and type(inputs['actual_parent_future_identity']) is int and inputs['actual_parent_future_identity']>0
            and inputs['worker_process_id_observed'] is False and inputs['worker_full_trace'] is False,
            'parent worker IPC scope/identity differs')
        producer=read_pointer(root,inputs['source_event'] if kind=='stock_worker_task_message' else inputs['task_message_event'])
        validate_event(root,producer,profile,dependencies)
        require(producer['status']=='complete' and producer['kind']==('triton_parent_task_source'
            if kind=='stock_worker_task_message' else 'stock_worker_task_message'),'worker IPC parent task chain differs')
        if kind=='stock_worker_result_before_unpickle':
            require(all(producer['inputs'][key]==inputs[key] for key in ('actual_job_id',
                'actual_pool_server_process_id','actual_parent_future_identity'))
                and inputs['before_original_pickle_loads'] is True and event['result']['worker_full_trace'] is False,
                'worker IPC returned payload differs from submitted parent future')


def validate_remote_template_event(root,event,dependencies):
    """Historical postload verification from archives, never a live tmp read."""
    inputs=event['inputs'];result=event['result']
    require(inputs['module_name']==result['module_name']==REMOTE_NAME
        and inputs['path']==result['path']
        and inputs['writer_observed'] is result['writer_observed'] is False
        and inputs['loader_call_observed'] is result['loader_call_observed'] is False
        and inputs['validation_scope']==result['validation_scope']=='postload_fixed_stock_recipe_and_loaded_function_code'
        and inputs['code_representation']=='CPython_marshal_v0_without_reference_tracking',
        'remote verification scope/observation claim differs')
    check_remote_template_path(root,inputs['path'],inputs['stock_directory'])
    require(inputs['stock_temporary_directory_name']==inputs['stock_directory']
        and inputs['stock_temporary_directory_class']=='tempfile.TemporaryDirectory'
        and type(inputs['stock_temporary_directory_identity']) is int and inputs['stock_temporary_directory_identity']>0,
        'remote stock temporary-directory witness differs')
    sources=inputs['stock_sources'];require(set(sources)==set(REMOTE_SOURCES),'remote archived source roles differ')
    raw={}
    for role,source in sources.items():
        require(set(source)=={'path','sha256','archive'} and Path(source['path']).is_absolute()
            and source['path'].endswith('/'+REMOTE_SOURCES[role])
            and dependencies.get(source['path'])==source['sha256'],'remote archived stock source is unbound: '+role)
        raw[role]=read_archive(root,source['archive'])
        require(sha_bytes(raw[role])==source['sha256'],'remote archived stock source bytes differ')
    require(event['source_owner']['path']==sources['instantiator']['path']
        and event['source_owner']['sha256']==sources['instantiator']['sha256'],'remote owner/source role differs')
    expected,recipe=stock_remote_template_recipe(raw)
    require(inputs['recipe']==recipe and read_archive(root,inputs['expected_source'])==expected
        and read_archive(root,inputs['actual_source'])==expected,'remote archived source differs from exact fixed stock recipe')
    artifacts=event['artifacts']
    require(len(artifacts)==1 and artifacts[0]['path']==inputs['path']
        and artifacts[0]['archive']==inputs['actual_source'] and artifacts[0]['sha256']==sha_bytes(expected),
        'remote source artifact differs')
    require(type(result['actual_module_identity']) is int and result['actual_module_identity']>0
        and result['parent_API_alias_identity']==result['actual_module_identity'] and result['registry_identity'] is True
        and type(result['actual_loader_identity']) is int and result['actual_loader_identity']>0
        and result['spec_loader_identity']==result['actual_loader_identity']
        and result['loader_class']=='_frozen_importlib_external.SourceFileLoader'
        and result['module_interface_cls_is_None'] is True,'remote actual registry/loader identity witness differs')
    codes=result['function_codes'];require(set(codes)==set(REMOTE_FUNCTIONS),'remote loaded function set differs')
    expected_codes=remote_function_codes(expected,inputs['path'],inputs['optimization'])
    for name,code in codes.items():
        require(set(code)=={'actual','expected','actual_function_identity','actual_code_identity'}
            and type(code['actual_function_identity']) is int and code['actual_function_identity']>0
            and type(code['actual_code_identity']) is int and code['actual_code_identity']>0
            and read_archive(root,code['actual'])==read_archive(root,code['expected'])==expected_codes[name],
            'remote loaded code archive/structure differs: '+name)
    require(len({v['actual_function_identity'] for v in codes.values()})==3
        and len({v['actual_code_identity'] for v in codes.values()})==3
        and result['generated_method_identities']==[codes[name]['actual_function_identity'] for name in ('forward_async','forward')],
        'remote generated method ordering/identity differs')
    globals=result['global_bindings']
    require(set(globals)=={'torch','rpc','Tensor','Future','RRef','Tuple','__builtins__'}
        and all(set(value)=={'actual_identity','canonical_identity'}
            and type(value['actual_identity']) is int and value['actual_identity']>0
            and value['actual_identity']==value['canonical_identity'] for value in globals.values()),
        'remote canonical used globals identity witness differs')


def validate_previous(root,pointer,profile,dependencies,artifact):
    reference=read_pointer(root,pointer)
    require(reference.get('schema')=='r10_completed_stock_JIT_artifact_reference_v1'
        and reference.get('profile_sha256')==digest(profile) and reference.get('path')==artifact['path']
        and reference.get('sha256')==artifact['sha256'] and reference.get('policy_outcome_used') is False,
        'previous artifact reference differs')
    producer=read_pointer(root,reference['event']);validate_event(root,producer,profile,dependencies)
    require(producer['status']=='complete' and reference['artifact'] in producer['artifacts'],
        'previous artifact event is not a completed exact byte producer')


def validate_worker_owner(owner,dependencies):
    require(isinstance(owner,dict) and set(owner)=={'module','qualified_name','path','sha256'}
        and owner['module']=='torch._inductor.runtime.compile_tasks'
        and owner['qualified_name']=='_worker_compile_triton'
        and owner['path'].endswith('/torch/_inductor/runtime/compile_tasks.py')
        and dependencies.get(owner['path'])==owner['sha256'],'worker exact stock role/source is unbound')


def validate_execution(root,pointer,profile,dependencies,*,identity=None,require_complete=True):
    value=read_pointer(root,pointer)
    require(value.get('schema')==EXECUTION and value.get('profile')==profile
        and value.get('profile_sha256')==digest(profile) and value.get('worker_full_trace') is False
        and (identity is None or value.get('identity')==identity)
        and type(value.get('process_id')) is int and value['process_id']>0,
        'execution profile/identity/scope differs')
    observer=value.get('observer_source',{})
    require(dependencies.get(observer.get('path'))==observer.get('sha256'),'JIT observer source is unbound')
    if require_complete:
        require(value.get('complete') is True and value.get('hook_restoration_completed') is True
            and value.get('violations')==[] and value.get('error') is None,'JIT execution is incomplete/violated')
        if value.get('stock_flags_initial') is not None:
            require(value['stock_flags_final']==value['stock_flags_initial'],'stock compile/autotune/static flags changed')
        if profile.get('reviewed_before_execution') is True:
            initialization=value.get('initialization',{})
            marker=read_pointer(root,initialization.get('initialization',{}))
            require(value.get('initialized_before_Torch_Triton_import') is True
                and marker.get('schema')=='r10_actual_empty_JIT_namespace_initialization_v1'
                and marker.get('profile_sha256')==digest(profile) and marker.get('cache_paths')==profile['cache_paths']
                and marker.get('initial_files')==[] and marker.get('old_cache_or_helper_copied') is False
                and marker.get('policy_outcome_used') is False,'actual pre-Torch empty namespace witness differs')
    for index,event in enumerate(value.get('events',[])):
        require(event.get('event_id')==index and event.get('process_id')==value['process_id']
            and (not require_complete or event['status']!='started'),'event sequence/PID/completeness differs')
        validate_event(root,event,profile,dependencies)
    return value


def validate_observed_bindings(root,pointer,profile,dependencies,witness,*,identity,process_id):
    """Return only files authorized by exact current-case loader witnesses."""
    value=validate_execution(root,pointer,profile,dependencies,identity=identity)
    require(value['process_id']==process_id,'JIT execution PID differs from canonical raw child')
    receipt_path=Path(root)/pointer['path'];events={}
    for event in value['events']:
        path=receipt_path.parent/'events'/(str(event['event_id']).zfill(6)+'.json')
        reference=path_pointer(root,path)
        require(read_pointer(root,reference)==event,'standalone event differs from final case snapshot')
        events[(reference['path'],reference['sha256'])]=event
    files={}
    for registry,binding in witness.get('generated_python_module_bindings',{}).items():
        if binding.get('schema')==REMOTE_BINDING:
            require(set(binding)=={'schema','module_name','path','sha256','actual_module_identity',
                'registry_identity','verification_event'} and registry==binding['module_name']==REMOTE_NAME
                and binding['registry_identity'] is True,'remote postload binding schema/registry differs')
            reference=binding['verification_event'];event=events.get((reference['path'],reference['sha256']))
            require(event is not None and event['kind']==REMOTE_EVENT and event['status']=='complete'
                and event['result']['module_name']==registry
                and event['result']['actual_module_identity']==binding['actual_module_identity']
                and event['inputs']['path']==binding['path'] and event['artifacts'][0]['sha256']==binding['sha256'],
                'remote module binding has no exact postload verification')
            require(witness.get('imported_dependency_files_sha256',{}).get(binding['path'])==binding['sha256'],
                'remote module binding was not actually observed in sys.modules')
            files[binding['path']]=binding['sha256'];continue
        require(set(binding)=={'schema','module_name','key','path','sha256','actual_module_identity',
            'registry_identity','loader_event'} and binding['schema']=='r10_exact_observed_stock_generated_python_module_v1'
            and binding['module_name']==registry and binding['registry_identity'] is True,
            'generated Python binding schema/registry differs')
        reference=binding['loader_event'];event=events.get((reference['path'],reference['sha256']))
        require(event is not None and event['kind'] in ('python_module_exec','python_module_load')
            and event['status']=='complete' and event['result']['module_name']==registry
            and event['result']['actual_module_identity']==binding['actual_module_identity']
            and event['inputs']['key']==binding['key'] and event['inputs']['path']==binding['path']
            and event['artifacts'][0]['sha256']==binding['sha256'],'generated Python binding has no exact observed object/load')
        require(witness.get('imported_dependency_files_sha256',{}).get(binding['path'])==binding['sha256'],
            'generated Python binding was not actually observed in sys.modules')
        files[binding['path']]=binding['sha256']
    for path,binding in witness.get('generated_native_file_bindings',{}).items():
        require(set(binding)=={'schema','path','sha256','loader_event'}
            and binding['schema']=='r10_exact_observed_generated_native_library_v1' and binding['path']==path,
            'generated native binding schema/path differs')
        reference=binding['loader_event'];event=events.get((reference['path'],reference['sha256']))
        require(event is not None and event['kind']=='native_helper_load' and event['status']=='complete'
            and event['inputs']['path']==path and event['artifacts'][0]['sha256']==binding['sha256'],
            'generated native binding has no exact observed source/build/load')
        require(witness.get('loaded_native_files_sha256',{}).get(path)==binding['sha256']
            or witness.get('imported_dependency_files_sha256',{}).get(path)==binding['sha256'],
            'generated native binding was not actually observed')
        files[path]=binding['sha256']
    return files


@contextmanager
def scoped_stock_jit_observer(root,case_directory,profile,dependencies,*,identity=None):
    session=ArtifactSession(root,case_directory,profile,dependencies,identity=identity)
    error=None
    try:
        session.install();yield session
    except BaseException as caught:
        error=caught;raise
    finally:session.close(error)

