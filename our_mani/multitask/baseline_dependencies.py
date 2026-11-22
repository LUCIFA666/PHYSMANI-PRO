"""Static baseline dependency preparation, without policy/native imports.

This produces drafts and inventories, never an approved execution contract.
The execution wrapper must still reject every import/mapped library outside
the reviewed frozen closure. Python metadata collection imports stdlib only.
"""
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from . import baseline_import_scope as import_scope

SCHEMA = 'r10_PhysMani_static_dependency_preparation_v1'
RUNTIME_SCHEMA = 'r10_PhysMani_actual_runtime_dependency_inventory_v1'
ARCHIVE_SCHEMA = 'r10_frozen_PhysMani_actual_execution_sources_v1'
GROUPS = ('local_sources', 'native_sources_and_scenes', 'policy_and_WM_sources',
          'model_assets', 'runtime_and_extension_assets', 'calibration_inputs')
CODE_EXTENSIONS = frozenset(('.py', '.sh', '.c', '.cpp', '.cu', '.cuh', '.h', '.hpp'))
PACKAGE_NAMES = ('physmani_policy', 'physmani_wm', 'physmani_bench')
EXTERNAL_MODULES = ('physmani_policy', 'physmani_wm', 'physmani_bench', 'rlbench', 'pyrep',
    'clip', 'diff_gaussian_rasterization', 'simple_knn', 'gsplat', 'pytorch3d',
    'torch', 'torchvision', 'open3d', 'sklearn', 'diffusers', 'flash_attn',
    'flash_attn_2_cuda', 'fps_cuda', 'functorch', 'numpy', 'scipy', 'PIL', 'cv2',
    'h5py', 'einops', 'pyquaternion', 'transformers', 'safetensors', 'rich',
    'tqdm', 'packaging', 'blosc', 'tap', 'termcolor', 'natsort', 'imageio',
    'requests', 'regex', 'ftfy')
ENVIRONMENT_KEYS = ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'COPPELIASIM_ROOT',
    'QT_QPA_PLATFORM_PLUGIN_PATH', 'CUDA_HOME', 'TORCH_EXTENSIONS_DIR',
    'TORCH_CUDA_ARCH_LIST', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
    'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CC', 'CXX', 'CPATH', 'LIBRARY_PATH')
UNRELATED_RELEASE_MODULES = frozenset(('our_mani.multitask.validation_release',
    'our_mani.multitask.composite_validation_release'))
CURRENT_EXECUTION_PERMISSION_FIELDS=('development_reset_profile','initial_condition_tolerance_profile')


def _require(value, message):
    if not value:
        raise ValueError('baseline dependency preparation: '+message)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            result.update(data)
    return result.hexdigest()


def _paths(root):
    """Match the production contract's exact inventory exclusions."""
    root = Path(root).resolve()
    if root.is_file():
        return {root}
    return {path.resolve() for path in root.rglob('*') if path.is_file()
        and '__pycache__' not in path.parts and '.git' not in path.parts and path.suffix != '.pyc'}


def python_runtime_metadata(python_executable):
    """Query locations only: never import torch, PyRep or any native package."""
    executable = Path(python_executable).resolve()
    _require(executable.is_file(), 'explicit Python interpreter is missing')
    query = '''import sys, sysconfig, importlib.util, json, os
names = json.loads(sys.argv[1]); specs = {}
for name in names:
    try:
        item = importlib.util.find_spec(name)
        specs[name] = None if item is None else {'origin':item.origin, 'roots':list(item.submodule_search_locations or [])}
    except Exception as error:
        specs[name] = {'error':type(error).__name__+': '+str(error)}
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
    'sys_path':sys.path,'sysconfig_paths':sysconfig.get_paths(),'packages':specs,
    'native_or_GPU_packages_imported':[name for name in ('torch','pyrep','rlbench','open3d','gsplat') if name in sys.modules]}))
'''
    process = subprocess.run([str(executable), '-B', '-c', query, json.dumps(EXTERNAL_MODULES)],
        check=True, text=True, capture_output=True, timeout=60)
    metadata = json.loads(process.stdout)
    _require(not metadata['native_or_GPU_packages_imported'],
        'interpreter startup imported a prohibited native/GPU package')
    _require(Path(metadata['executable']).resolve() == executable,
        'queried Python interpreter identity differs')
    return metadata


def _module_files(root, module):
    parts = module.split('.') if module else []
    candidate = root.joinpath(*parts)
    files = []
    for index in range(1, len(parts)+1):
        init = root.joinpath(*parts[:index])/'__init__.py'
        if init.is_file():
            files.append(init.resolve())
    if candidate.with_suffix('.py').is_file():
        files.append(candidate.with_suffix('.py').resolve())
    return files


def local_import_closure(root, initial_sources, *, excluded_import_nodes=()):
    """Follow local imports by AST; record unresolved dynamic import sites.

    Imports in delayed functions are included conservatively. A release module
    is never silently ignored: encountering one makes the draft require review.
    """
    root = Path(root).resolve(); queue = deque(Path(root/path).resolve() for path in initial_sources)
    found = set(); gaps = []
    excluded={(node['source_path'],node['scope'],node['line'],node['statement']) for node in excluded_import_nodes}
    while queue:
        path = queue.popleft()
        if path in found:
            continue
        if not path.is_file():
            gaps.append(dict(kind='missing_local_source', path=str(path))); continue
        _require(path.is_relative_to(root), 'local source outside owned project: '+str(path))
        found.add(path)
        if path.suffix != '.py':
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            gaps.append(dict(kind='unreadable_Python_source', path=str(path), error=str(error))); continue
        relative = path.relative_to(root).with_suffix('')
        package = list(relative.parts[:-1])
        if path.name == '__init__.py':
            package = list(relative.parts[:-1])
        modules = set()
        for node in import_scope.import_nodes(root,path):
            key=(node['source_path'],node['scope'],node['line'],node['statement'])
            if key not in excluded:modules.update(node['modules'])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                dynamic = ((isinstance(func, ast.Attribute) and func.attr == 'import_module')
                    or (isinstance(func, ast.Name) and func.id == '__import__'))
                if dynamic and node.args:
                    if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        modules.add(node.args[0].value)
                    else:
                        gaps.append(dict(kind='dynamic_import_requires_review', path=str(path), line=node.lineno))
        for module in sorted(modules):
            if module in UNRELATED_RELEASE_MODULES:
                gaps.append(dict(kind='release_import_requires_branch_review', path=str(path), module=module))
                continue
            queue.extend(_module_files(root, module))
    return found, gaps


def _literal_shell_sources(path):
    sources, dynamic = [], []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if words and words[0] in ('source', '.') and len(words) >= 2:
            if '$' in words[1] or '`' in words[1]:
                dynamic.append(dict(kind='dynamic_shell_source_requires_review', path=str(path), line=number))
            else:
                source = Path(words[1])
                sources.append(source if source.is_absolute() else Path(path).parent/source)
    return sources, dynamic


def inventory_baseline_dependencies(root, config, *, python_executable, runtime_roots,
        extra_source_roots=(), local_sources=(), calibration_inputs=(), dry_run=False,
        archive_directory=None, python_metadata=None, mutable_state_permission=None,import_edge_review=None):
    """Prepare hash-bound closure drafts, with explicit unresolved gaps.

    ``runtime_roots`` must include the reviewed Coppelia/system/driver roots.
    All files in the selected interpreter's lib/bin/targets trees are bound,
    including installed Python dependencies, extension binaries and data.
    ``python_metadata`` is an injectable stdlib-query result for CPU fixtures.
    The caller publishes/approves a contract only after reviewing this draft.
    """
    root = Path(root).resolve(); repo = (root/config['baseline_repo']).resolve()
    executable = (root/python_executable).resolve()
    _require(executable.is_file(), 'missing explicit Python interpreter')
    metadata = python_runtime_metadata(executable) if python_metadata is None else python_metadata
    _require(Path(metadata['executable']).resolve() == executable
        and not metadata.get('native_or_GPU_packages_imported'), 'Python metadata identity/import guard differs')
    prefix = Path(metadata['prefix']).resolve()
    _require(executable.is_relative_to(prefix), 'interpreter is outside declared environment prefix')
    groups = {name:set() for name in GROUPS}; gaps = []; inventories = []; runtime_directories=set()
    edge_review=None
    if import_edge_review is not None:
        edge_review=import_scope.load_review(root,import_edge_review,require_reviewed=False)
    state_permission=None; state_before=None; excluded=set()
    if mutable_state_permission is not None:
        from . import baseline_runtime_state as state
        state_permission=state.load_permission(root,mutable_state_permission,config,require_reviewed=False)
        excluded={Path(state_permission['resolved_path'])}
        state_before=state.snapshot(state_permission['resolved_path'])

    def add(path, group, *, required=True, inventory=False):
        path = (root/path).resolve()
        if not path.exists():
            if required: gaps.append(dict(kind='missing_dependency', group=group, path=str(path)))
            return
        files = _paths(path)
        if group=='runtime_and_extension_assets':
            if path.is_dir():runtime_directories.add(path)
            files-=excluded
        else:
            _require(not (files & excluded),'mutable state cannot masquerade as source/config/model input')
        if not files:
            gaps.append(dict(kind='empty_dependency_root', group=group, path=str(path)))
        groups[group].update(files)
        if inventory:
            inventories.append(dict(root=str(path), paths=files))

    for name in PACKAGE_NAMES:
        directory = repo/'packages'/name/name
        add(directory, 'policy_and_WM_sources', inventory=True)
        if not (directory/'__init__.py').is_file():
            gaps.append(dict(kind='missing_actual_package_init', path=str(directory)))
        spec = metadata.get('packages', {}).get(name)
        if spec is None or {Path(value).resolve() for value in spec.get('roots', [])} != {directory}:
            gaps.append(dict(kind='actual_installed_package_differs', package=name, expected=str(directory), actual=spec))
    for path in ('third_party/RLBench/rlbench', 'third_party/PyRep/pyrep'):
        add(repo/path, 'native_sources_and_scenes')
    scene = repo/'third_party/RLBench/rlbench'
    add(scene/'task_design.ttt', 'native_sources_and_scenes')
    # Use the project's static task list; importing it starts no native runtime.
    from .specs import TASK_SPECS
    for task in TASK_SPECS:
        add(scene/'task_ttms'/(task+'.ttm'), 'native_sources_and_scenes')
    for path in (root/config['baseline_checkpoint'], root/'runtime/framework/clip/RN50.pt',
                 repo/'tasks/18_peract_tasks_location_bounds.json',
                 repo/'instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl'):
        add(path, 'model_assets')

    # Runtime metadata enumerates locations rather than importing these modules.
    for name, spec in metadata.get('packages', {}).items():
        if spec is None or 'error' in spec:
            if name in EXTERNAL_MODULES:
                gaps.append(dict(kind='unresolved_required_package', package=name, actual=spec))
            continue
        for directory in spec.get('roots', []):
            resolved = Path(directory).resolve()
            if name not in PACKAGE_NAMES and name not in ('rlbench','pyrep'):
                add(resolved, 'runtime_and_extension_assets')
        origin = spec.get('origin')
        if origin and origin not in ('built-in','frozen'):
            add(origin, 'runtime_and_extension_assets')
    for suffix in ('lib', 'lib64', 'bin', 'targets', 'etc/conda'):
        add(prefix/suffix, 'runtime_and_extension_assets', required=False)
    add(executable, 'runtime_and_extension_assets')
    for directory in runtime_roots:
        add(directory, 'runtime_and_extension_assets')
    if not runtime_roots:
        gaps.append(dict(kind='explicit_native_system_runtime_roots_required'))
    for directory in extra_source_roots:
        add(directory, 'runtime_and_extension_assets')
    launcher_programs = {}
    for name in ('bash', 'Xvfb', 'xvfb-run', 'xauth'):
        command = shutil.which(name)
        candidates = {Path(command).resolve()} if command else set()
        for directory in runtime_roots:
            for suffix in (name, 'bin/'+name, 'usr/bin/'+name):
                candidate = (root/directory/suffix).resolve()
                if candidate.is_file(): candidates.add(candidate)
        for candidate in candidates: add(candidate, 'runtime_and_extension_assets')
        launcher_programs[name] = dict(current_PATH_resolution=str(Path(command).resolve()) if command else None,
            frozen_candidates=sorted(str(path) for path in candidates),
            final_launcher_PATH_requires_review=True)
        if not candidates: gaps.append(dict(kind='launcher_program_missing', executable=name))

    from .baseline_component import REVIEW_SOURCE_FILES
    reviewed_origins=() if edge_review is None else tuple(edge_review['source_bindings_sha256'])
    owned, imports = local_import_closure(root, tuple(REVIEW_SOURCE_FILES)+tuple(local_sources)+reviewed_origins,
        excluded_import_nodes=() if edge_review is None else edge_review['excluded_import_nodes'])
    groups['local_sources'].update(owned); gaps.extend(imports)
    queue = deque(path for path in owned if path.suffix == '.sh')
    shell_seen = set()
    while queue:
        path = queue.popleft().resolve()
        if path in shell_seen: continue
        shell_seen.add(path)
        if not path.is_file():
            gaps.append(dict(kind='missing_shell_source', path=str(path))); continue
        add(path, 'local_sources' if path.is_relative_to(root) else 'runtime_and_extension_assets')
        children, dynamic = _literal_shell_sources(path); gaps.extend(dynamic); queue.extend(children)
    # Permission and camera ledgers are execution inputs; no future data is read.
    pointers = list(calibration_inputs)
    if mutable_state_permission is not None:pointers.append(mutable_state_permission)
    if import_edge_review is not None:pointers.append(import_edge_review)
    if isinstance(config.get('development_reset_profile'), dict):
        pointers.append(config['development_reset_profile'])
    if isinstance(config.get('initial_condition_tolerance_profile'), dict):
        pointers.append(config['initial_condition_tolerance_profile'])
    declared_checksums = {}; historical_metadata=[]
    actual_permission_paths={(root/item['path']).resolve() for item in (
        *(config.get(key) for key in CURRENT_EXECUTION_PERMISSION_FIELDS),mutable_state_permission,import_edge_review)
        if isinstance(item,dict) and isinstance(item.get('path'),str)}
    for item in pointers:
        path = item['path'] if isinstance(item, dict) else item
        add(path, 'calibration_inputs')
        if isinstance(item, dict): declared_checksums[str((root/path).resolve())] = item['sha256']
        # The shared reset loader reads its diagnostic permission and validates
        # exact native source hashes. Bind those direct inputs without opening
        # diagnostic HDF5 motion/expert payloads or inferring a new permission.
        actual = (root/path).resolve()
        if actual.is_file() and actual.suffix == '.json':
            try:
                permission = json.loads(actual.read_text())
            except (UnicodeError, ValueError):
                continue
            if actual not in actual_permission_paths:
                if permission.get('inputs_sha256') or permission.get('native_inputs_sha256'):
                    historical_metadata.append(dict(path=str(actual),sha256=_sha(actual),
                        schema=permission.get('schema',permission.get('version')),
                        reason='retained JSON bytes are frozen; historical producer input maps are not current baseline execution permissions'))
                continue
            for field, base in (('native_inputs_sha256', repo), ('inputs_sha256', root),
                                ('source_bindings_sha256',root)):
                for name, checksum in permission.get(field, {}).items():
                    dependency = (base/name).resolve()
                    add(dependency, 'native_sources_and_scenes' if field.startswith('native') else 'calibration_inputs')
                    declared_checksums[str(dependency)] = checksum
            for field in ('diagnostic_report', 'TRAIN_evidence','evidence_report','evidence'):
                value = permission.get(field)
                if isinstance(value, dict) and {'path','sha256'} <= set(value):
                    dependency = (root/value['path']).resolve()
                    add(dependency, 'calibration_inputs')
                    declared_checksums[str(dependency)] = value['sha256']

    # Keep only maximal roots to avoid duplicate huge environment inventories.
    runtime_directories={path for path in runtime_directories
        if not any(path!=parent and path.is_relative_to(parent) for parent in runtime_directories)}
    runtime_sets={path:_paths(path)-excluded for path in runtime_directories}
    paths = sorted(set().union(*groups.values()), key=str)
    if edge_review is not None:
        excluded_paths={root.joinpath(*name.split('.')).with_suffix('.py').resolve() for name in import_scope.PRUNED_MODULES}
        _require(not (excluded_paths & set(paths)), 'an excluded ours module remains reachable through another actual import/input edge')
    stats = {str(path):dict(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        for path in paths}
    hashes = {} if dry_run else {str(path):_sha(path) for path in paths}
    if not dry_run:
        for name, expected in declared_checksums.items():
            if hashes.get(name) != expected:
                gaps.append(dict(kind='declared_pointer_SHA_differs', path=name))
        for path in paths:
            actual = path.stat()
            _require(stats[str(path)] == dict(size_bytes=actual.st_size, mtime_ns=actual.st_mtime_ns),
                'dependency changed while inventory was being prepared: '+str(path))
    dependency_groups = {name:{str(path):hashes[str(path)] for path in sorted(files, key=str)}
        if not dry_run else {} for name, files in groups.items()}
    root_inventories = [dict(root=item['root'], files_sha256={str(path):hashes[str(path)]
        for path in sorted(item['paths'], key=str)} if not dry_run else {},
        file_count=len(item['paths'])) for item in inventories]
    runtime_files = dependency_groups['runtime_and_extension_assets']
    runtime = dict(schema=RUNTIME_SCHEMA, complete=False, CUDA_and_native_extensions_bound=False,
        prepared_files_hashed=not dry_run, files_sha256=runtime_files,
        actual_native_runtime_executed=False, approved=False, review_required=True,
        interpreter=str(executable), python_metadata=metadata,
        launcher_programs=launcher_programs,
        environment_at_metadata_preparation={name:os.environ.get(name) for name in ENVIRONMENT_KEYS},
        final_launcher_environment_requires_review=True,
        complete_directory_inventories=[dict(root=str(path),files_sha256={str(item):hashes[str(item)]
            for item in sorted(files,key=str)} if not dry_run else {},file_count=len(files))
            for path,files in sorted(runtime_sets.items(),key=lambda pair:str(pair[0]))],
        mutable_state_permission=mutable_state_permission,
        immutable_directory_claim='all enumerated files except the one exact declared crash-bookkeeping state file'
            if excluded else 'all enumerated non-cache files immutable',
        excluded_state_paths=sorted(str(path) for path in excluded))
    archive = None
    if archive_directory is not None:
        _require(not dry_run, 'dry inventory cannot publish an executable source archive')
        directory = (root/archive_directory).resolve()
        _require(directory.is_relative_to(root) and not directory.exists(), 'fresh owned archive directory required')
        directory.mkdir(parents=True)
        entries = []
        for name, checksum in sorted(hashes.items()):
            source = Path(name)
            if source.suffix not in CODE_EXTENSIONS: continue
            target = directory/'sources'/checksum/source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): shutil.copyfile(source, target)
            _require(_sha(target) == checksum and _sha(source) == checksum,
                'source changed before/after archival: '+name)
            entries.append(dict(original_path=name, path=str(target.relative_to(root)), sha256=checksum))
        for name, checksum in hashes.items():
            _require(_sha(name) == checksum, 'dependency changed while executable sources were archived: '+name)
        archive = dict(schema=ARCHIVE_SCHEMA, all_inputs_verified_before_and_after=True,
            entries=entries, prepared_only=True, native_or_GPU_launched=False)
        target = directory/'source_archive.json'
        target.write_text(json.dumps(archive, ensure_ascii=False, indent=2)+'\n')
        archive = dict(path=str(target.relative_to(root)), sha256=_sha(target))
    for directory,expected in runtime_sets.items():
        _require(_paths(directory)-excluded==expected,'runtime directory file set changed: '+str(directory))
    state_after=None
    if state_permission is not None:
        state_after=state.snapshot(state_permission['resolved_path'])
    return dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        prepared_only=True, approved=False, dry_run=bool(dry_run),
        native_or_GPU_launched=False, model_fitting_performed=False,
        dependency_groups=dependency_groups, complete_source_root_inventories=root_inventories,
        runtime_inventory=runtime, source_archive=archive, python_metadata=metadata,
        inventory_files=stats, total_files=len(paths), total_bytes=sum(row['size_bytes'] for row in stats.values()),
        gaps=gaps, review_required=True,
        retained_historical_metadata_not_replayed=historical_metadata,
        current_execution_permission_scope=dict(config_fields=list(CURRENT_EXECUTION_PERMISSION_FIELDS),
            separate_pointers=['mutable_state_permission','import_edge_review'],
            paths=sorted(str(path) for path in actual_permission_paths)),
        reviewed_unreachable_import_edges=None if edge_review is None else dict(
            pointer=import_edge_review,excluded_import_nodes=edge_review['excluded_import_nodes'],
            excluded_modules=edge_review['excluded_modules'],unexpected_actual_imports_rejected=True),
        mutable_state_observations=None if state_permission is None else dict(
            permission=mutable_state_permission,before=state_before,after=state_after,
            limitations=state.LIMITATIONS),
        limitations=['Static inventory is not execution qualification or a formal VAL release.',
            'Every actual late import and mapped native library must remain bound by the production witness.',
            'The reviewed launcher must freeze environment and native/system/driver roots before execution.'])
