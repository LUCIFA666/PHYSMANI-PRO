"""Independent early-VAL PhysMani component proof; no existing release gate.

An explicitly projected backend prevents ours-only artifact fields reaching
the simulator, policy or online WM. Raw cases retain their original complete
config/lock/queue identity. Reuse proves equality of a separately frozen
execution dependency projection; it does not relabel those historical cases.
No TEST permission, model coverage certificate or five-method release is
issued here. Missing actual dependency/reset witnesses fail closed.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType

from .evaluation import (digest, file_sha, case_identity, validate_completed_case)
from .specs import TASK_SPECS
from . import baseline_runtime_state as runtime_state
from . import baseline_import_scope as import_scope
from . import baseline_jit_artifacts as jit_artifacts

VERSION = 'r10_explicit_PhysMani_execution_projection_v1'
CONTRACT = 'r10_frozen_PhysMani_execution_dependency_contract_v1'
SOURCE_ARCHIVE = 'r10_frozen_PhysMani_actual_execution_sources_v1'
COMPONENT = 'r10_verified_early_PhysMani_VAL_component_v1'
ROOT = Path(__file__).resolve().parents[2]
# Exact branch-reviewed exclusions. No wildcard metadata or name heuristic.
OURS_ROOTS = ('context_checkpoint', 'continuation_identities')
OURS_LEAVES = ('cv_value', 'context_value', 'interaction', 'colour', 'rigid_layout')
METADATA = ('version', 'created_utc', 'formal_comparison_ready',
    'full_source_training_started', 'full_source_training_parents_completed',
    'full_source_collection_started')
BASELINE_READ_KEYS = ('baseline_repo', 'baseline_checkpoint', 'baseline_decision_ticks',
    'baseline_executor', 'baseline_output', 'seed', 'local_servo_lead', 'dt',
    'simulation_seconds', 'gripper_velocity', 'global_joint_acceleration',
    'current_instance_masks', 'information_setting', 'map_access',
    'shared_semantic_instance_binding', 'semantic_binding_available_to_baseline',
    'shared_TRAIN_CAD_geometry', 'development_reset_profile', 'r09_completed_release')
REVIEW_SOURCE_FILES = (
    'scripts/evaluate_multitask_case_r10.py', 'our_mani/multitask/protocol.py',
    'our_mani/multitask/maps.py',
    'our_mani/multitask/simulation.py', 'our_mani/multitask/reset_parity.py',
    'our_mani/aligned/physmani.py', 'our_mani/ablations/physmani_executor_v4.py',
    'our_mani/sensor_alignment/physmani_executor_v5.py',
    'our_mani/sensor_alignment/physmani_executor_v6.py',
    'our_mani/multitask/physmani_executor_v7.py',
    'our_mani/multitask/development_reset_recovery.py',
    'our_mani/multitask/baseline_runtime_state.py',
    'our_mani/multitask/baseline_import_scope.py',
    'our_mani/multitask/baseline_jit_artifacts.py',
    'our_mani/multitask/baseline_component.py')
GROUPS = ('local_sources', 'native_sources_and_scenes', 'policy_and_WM_sources',
    'model_assets', 'runtime_and_extension_assets', 'calibration_inputs')
_VERIFIED_DEPENDENCIES = {}
TORCH_VIRTUAL_OWNER_SCHEMA = 'r10_exact_torch_virtual_namespace_owner_v1'
TORCH_VIRTUAL_OWNERS = MappingProxyType({
    'torch.ops': ('torch._ops', '_Ops', '_ops.py', 'ops'),
    'torch.classes': ('torch._classes', '_Classes', '_classes.py', 'classes')})


def require(value, message):
    if not value: raise ValueError('baseline component: '+message)


def read(path): return json.loads(Path(path).read_text())


def absolute(root, path): return (Path(root)/path).resolve()


def pointer(root, value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'exact path/SHA pointer required')
    path = absolute(root, value['path'])
    require(path.is_relative_to(Path(root).resolve()) and path.is_file()
        and file_sha(path) == value['sha256'], 'immutable project pointer changed: '+str(path))
    return path


def normalized(root, files):
    result = {}
    require(isinstance(files, dict) and files, 'nonempty dependency map required')
    for name, checksum in files.items():
        key = str(absolute(root, name))
        require(isinstance(checksum, str) and len(checksum) == 64
            and all(c in '0123456789abcdef' for c in checksum), 'invalid dependency SHA')
        require(key not in result or result[key] == checksum, 'conflicting path aliases')
        result[key] = checksum
    return result


def observer_dependencies(root, contract_pointer):
    """Reuse only a fully verified immutable map; rehash the contract each call.

    Minimal injected CPU fixtures may use an uncached map. That path never
    populates the verification cache or grants native execution permission.
    """
    root = Path(root).resolve()
    path = pointer(root, contract_pointer)
    key = (str(root), contract_pointer['sha256'])
    cached = _VERIFIED_DEPENDENCIES.get(key)
    if cached is not None:
        return cached
    return {name: checksum for values in read(path)['dependency_groups'].values()
            for name, checksum in normalized(root, values).items()}


def torch_virtual_owner(registry_key, module):
    """Resolve only the two stock registry objects, never arbitrary relatives."""
    spec = TORCH_VIRTUAL_OWNERS.get(registry_key)
    if spec is None:
        return None
    owner_name, type_name, literal, attribute = spec
    owner = sys.modules.get(owner_name)
    torch = sys.modules.get('torch')
    require(owner is not None and torch is not None
        and sys.modules.get(registry_key) is module
        and vars(owner).get(attribute) is module and vars(torch).get(attribute) is module,
        'torch virtual namespace registry identity differs: '+registry_key)
    stock_type = vars(owner).get(type_name)
    require(stock_type is not None and type(module) is stock_type
        and stock_type.__module__ == owner_name and stock_type.__name__ == type_name
        and getattr(module, '__name__', None) == registry_key
        and getattr(module, '__file__', None) == literal,
        'torch virtual namespace stock type or placeholder differs: '+registry_key)
    name = vars(owner).get('__file__')
    require(isinstance(name, str) and Path(name).is_absolute(),
        'torch virtual namespace owner has no real absolute file')
    path = Path(name).resolve()
    origin = getattr(vars(owner).get('__spec__'), 'origin', None)
    require(path.is_file() and path.name == literal and path.parent.name == 'torch'
        and isinstance(origin, str) and Path(origin).resolve() == path
        and Path(stock_type.__init__.__code__.co_filename).resolve() == path,
        'torch virtual namespace owner origin differs: '+registry_key)
    return path, dict(schema=TORCH_VIRTUAL_OWNER_SCHEMA, registry_key=registry_key,
        module_name=registry_key, owner_module=owner_name, stock_type=owner_name+'.'+type_name,
        literal_placeholder=literal, owner_attribute=attribute,
        registry_identity_verified=True, stock_type_identity_verified=True)


def validate_virtual_owner_bindings(value, dependencies, imports):
    require(isinstance(value, dict) and not set(value)-set(TORCH_VIRTUAL_OWNERS),
            'unknown torch virtual namespace owner witness')
    # A real installed torch tree in this closure requires both stock aliases.
    required = {key for key, (_, _, literal, _) in TORCH_VIRTUAL_OWNERS.items()
                if any(Path(name).name == literal and Path(name).parent.name == 'torch'
                       for name in dependencies)}
    require(set(value) == required, 'missing torch virtual namespace owner witness')
    for key, row in value.items():
        owner_name, type_name, literal, attribute = TORCH_VIRTUAL_OWNERS[key]
        expected = dict(schema=TORCH_VIRTUAL_OWNER_SCHEMA, registry_key=key,
            module_name=key, owner_module=owner_name, stock_type=owner_name+'.'+type_name,
            literal_placeholder=literal, owner_attribute=attribute,
            registry_identity_verified=True, stock_type_identity_verified=True)
        require(isinstance(row, dict) and set(row) == set(expected)|{'owner_file'}
            and {name: row[name] for name in expected} == expected,
            'torch virtual namespace witness type or identity differs')
        owner = row['owner_file']
        require(isinstance(owner, dict) and set(owner) == {'path', 'sha256'}
            and Path(owner['path']).is_absolute() and Path(owner['path']).name == literal
            and Path(owner['path']).parent.name == 'torch'
            and dependencies.get(owner['path']) == owner['sha256']
            and imports.get(owner['path']) == owner['sha256'],
            'torch virtual namespace real owner bytes were not observed/bound')


def project_baseline_config(config):
    """All unreviewed config fields remain, so any unknown difference fails."""
    require(isinstance(config, dict), 'config must be a dictionary')
    result = deepcopy(config)
    for key in OURS_ROOTS+METADATA: result.pop(key, None)
    artifacts = result.get('learned_artifacts')
    if artifacts is not None:
        require(isinstance(artifacts, dict) and not set(artifacts)-set(TASK_SPECS), 'unknown artifact task')
        for values in artifacts.values():
            require(isinstance(values, dict), 'task artifact table must be a dictionary')
            for key in OURS_LEAVES: values.pop(key, None)
    require(set(BASELINE_READ_KEYS).isdisjoint(set(OURS_ROOTS+OURS_LEAVES+METADATA)),
            'a reviewed baseline read became an exclusion')
    return result


def compare_config_dependencies(early, final):
    left, right = project_baseline_config(early), project_baseline_config(final)
    require(left == right, 'baseline execution config changed (including unknown fields)')
    return dict(projected_config=left, projection_sha256=digest(left),
        early_full_config_sha256=digest(early), final_full_config_sha256=digest(final),
        full_config_identities_claimed_equal=False)


def canonical_VAL_roster(value):
    rows = [row for row in value.get('episodes', []) if row.get('task') in TASK_SPECS]
    expected = dict.fromkeys(TASK_SPECS, 20)
    expected.update(insert_onto_rotating_peg=18, insert_onto_rotating_peg_high_speed=19,
                    put_rubbish_in_moving_bin_high_speed=18)
    counts = {task: sum(row['task'] == task for row in rows) for task in TASK_SPECS}
    require(value.get('split') == 'validation' and len(rows) == 275 and counts == expected
        and len({(row['task'], row['raw_episode_id']) for row in rows}) == 275
        and all(isinstance(row.get('path'), str) and row['path'] for row in rows),
        'exact canonical 275-source fourteen-task VAL roster required')
    return rows


def calibration_dependency(row):
    """Bind the ONLY HDF5 fields the native development camera loader reads.

    Do not hash/re-read future images, motion or expert arrays as if baseline
    used them. The complete canonical source row and original file stat remain
    bound alongside exact dtype/shape/bytes of all sixteen camera datasets.
    """
    import h5py
    path = Path(row['path']).resolve(); stat = path.stat(); fields = {}
    with h5py.File(path, 'r') as source:
        for camera in ('front', '8', '16', '36'):
            for key in ('extrinsics', 'intrinsics', 'near', 'far'):
                name = 'cam_params/'+camera+'/'+key; value = source[name][...]
                fields[name] = dict(dtype=str(value.dtype), shape=list(value.shape),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return dict(source_row=deepcopy(row), source_path=str(path),
        source_file_stat=dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, device=stat.st_dev),
        camera_dataset_dependencies=fields, future_motion_images_or_expert_arrays_read=False)


def verify_execution_contract(root, value, config, lock, roster, *, require_approved=True):
    """Fail closed until the explicitly reviewed full execution closure exists.

    Complete source-root inventories and actual runtime/import witnesses are
    required, rather than treating an empty guessed repository glob as proof.
    The contract must be frozen before the first early VAL execution.
    """
    root = Path(root).resolve(); contract_path = pointer(root, value)
    contract = read(contract_path); files = normalized(root, lock['files'])
    require(files.get(str(contract_path)) == value['sha256'], 'execution contract itself is not in the actual lock')
    require(contract.get('schema') == CONTRACT and contract.get('version') == VERSION
        and contract.get('approved_before_VAL_execution') is require_approved
        and contract.get('projected_backend_required') is True
        and contract.get('actual_dependency_closure_reviewed') is require_approved,
        'missing pre-execution approved dependency closure/projected backend contract')
    require(contract.get('projected_config_sha256') == digest(project_baseline_config(config))
        and contract.get('canonical_VAL_roster_sha256') == digest(roster)
        and contract.get('shared_development_reset_profile') == config.get('development_reset_profile'),
        'frozen config/roster/shared reset projection differs')
    require(isinstance(config.get('development_reset_profile'), dict), 'actual shared TRAIN/VAL reset profile is required')
    groups = contract.get('dependency_groups', {})
    require(set(groups) == set(GROUPS) and all(isinstance(groups[g], dict) and groups[g] for g in GROUPS),
            'every baseline source/WM/model/runtime/calibration dependency group is required')
    dependencies = {}
    for group in GROUPS:
        for name, checksum in normalized(root, groups[group]).items():
            require(name not in dependencies or dependencies[name] == checksum, 'conflicting dependency groups')
            dependencies[name] = checksum
    for name, checksum in dependencies.items():
        require(files.get(name) == checksum, 'dependency is not bound by the actual input lock: '+name)
        require(Path(name).is_file() and file_sha(name) == checksum, 'execution dependency bytes changed: '+name)
    if contract.get('import_edge_review') is not None:
        edge=import_scope.load_review(root,contract['import_edge_review'],require_reviewed=require_approved)
        require(dependencies.get(str(pointer(root,contract['import_edge_review'])))==contract['import_edge_review']['sha256']
            and all(dependencies.get(name)==checksum for name,checksum in edge['source_bindings_sha256'].items())
            and dependencies.get(str(pointer(root,edge['evidence'])))==edge['evidence']['sha256'],
            'exact function-edge review/origin/evidence is not in execution closure')
        targets={str(root.joinpath(*name.split('.')).with_suffix('.py')) for name in edge['excluded_modules']}
        require(not (targets & set(dependencies)) and not (targets & set(files)),
                'ours-only source remains in the supposedly independent baseline fingerprint')
    require(str(pointer(root, config['development_reset_profile'])) in dependencies,
            'shared reset permission is not an execution dependency')
    repo = Path(config['baseline_repo']).resolve()
    for path in (absolute(root, config['baseline_checkpoint']),
            root/'runtime/framework/clip/RN50.pt',
            repo/'tasks/18_peract_tasks_location_bounds.json',
            repo/'instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl'):
        require(str(path.resolve()) in dependencies, 'actual baseline model/instruction/bounds asset is missing: '+str(path))
    scene_root = repo/'third_party/RLBench/rlbench'
    for path in [scene_root/'task_design.ttt']+[scene_root/'task_ttms'/(task+'.ttm') for task in TASK_SPECS]:
        require(str(path.resolve()) in dependencies, 'actual native task/base scene dependency missing: '+str(path))
    ledger_path = pointer(root, contract['calibration_ledger']); ledger = read(ledger_path)
    require(dependencies.get(str(ledger_path)) == contract['calibration_ledger']['sha256']
        and ledger.get('schema') == 'r10_canonical_VAL_camera_dependency_ledger_v1'
        and ledger.get('entries') == [calibration_dependency(row) for row in roster],
        'canonical source camera dependencies changed or are not bound')
    review = read(pointer(root, contract['config_access_review']))
    require(review.get('schema') == 'r10_PhysMani_config_access_code_review_v1'
        and review.get('approved') is require_approved and review.get('branch') == 'physmani_TRAIN_VAL_only'
        and review.get('baseline_read_keys') == list(BASELINE_READ_KEYS)
        and review.get('excluded_roots') == list(OURS_ROOTS)
        and review.get('excluded_artifact_leaves') == list(OURS_LEAVES)
        and review.get('excluded_metadata') == list(METADATA), 'exact baseline access review is missing or changed')
    reviewed = normalized(root, review.get('sources_sha256', {}))
    require(set(reviewed) == {str(root/name) for name in REVIEW_SOURCE_FILES}
        and all(dependencies.get(name) == checksum for name, checksum in reviewed.items()),
        'all actual config-reading sources and projection helper must be frozen/reviewed')
    entrypoints = contract.get('execution_entrypoints', {})
    require(set(entrypoints) == {'parent_source', 'child_source', 'launcher_source'}
        and review.get('execution_entrypoints') == entrypoints,
        'actual queue/child/launcher entrypoints require explicit code review')
    for item in entrypoints.values():
        path = pointer(root, item)
        require(dependencies.get(str(path)) == item['sha256'], 'reviewed executable entrypoint is outside frozen closure')
    inventories = contract.get('complete_source_root_inventories', [])
    require(isinstance(inventories, list) and inventories, 'complete actual source roots are missing')
    covered = set()
    for inventory in inventories:
        directory = Path(inventory['root']).resolve()
        require(directory.is_dir(), 'actual source root does not exist: '+str(directory))
        actual = {str(path.resolve()) for path in directory.rglob('*') if path.is_file()
            and '__pycache__' not in path.parts and '.git' not in path.parts and path.suffix != '.pyc'}
        declared = normalized(root, inventory.get('files_sha256', {}))
        require(actual == set(declared) and all(dependencies.get(name) == checksum for name, checksum in declared.items()),
                'actual source root inventory is empty, incomplete or not locked: '+str(directory))
        covered.update(actual)
    for package in ('physmani_policy', 'physmani_wm', 'physmani_bench'):
        actual_root = repo/'packages'/package/package
        require(any(Path(row['root']).resolve() == actual_root for row in inventories)
            and str(actual_root/'__init__.py') in covered, 'actual installed package source root missing: '+package)
    runtime = read(pointer(root, contract['runtime_inventory']))
    require(runtime.get('schema') == 'r10_PhysMani_actual_runtime_dependency_inventory_v1'
        and runtime.get('complete') is require_approved
        and runtime.get('CUDA_and_native_extensions_bound') is require_approved,
        'runtime/native-extension inventory is incomplete')
    runtime_files = normalized(root, runtime.get('files_sha256', {}))
    require(all(dependencies.get(name) == checksum for name, checksum in runtime_files.items()),
            'actual runtime/imported binary dependency is not locked')
    jit_pointer=contract.get('jit_artifact_profile');jit_profile=None
    if jit_pointer is not None:
        require(dependencies.get(str(pointer(root,jit_pointer)))==jit_pointer['sha256']
            and runtime.get('jit_artifact_profile')==jit_pointer,'JIT profile is outside actual lock/runtime inventory')
        jit_profile=jit_artifacts.load_profile(root,jit_pointer,dependencies,require_reviewed=require_approved)
    else:require(runtime.get('jit_artifact_profile') is None,'undeclared JIT artifact profile')
    state_pointer=contract.get('mutable_state_permission');state_permission=None;excluded=set()
    if state_pointer is not None:
        state_permission=runtime_state.load_permission(root,state_pointer,config,require_reviewed=require_approved)
        excluded={state_permission['resolved_path']}
        require(dependencies.get(str(pointer(root,state_pointer)))==state_pointer['sha256']
            and runtime.get('mutable_state_permission')==state_pointer
            and runtime.get('excluded_state_paths')==sorted(excluded)
            and not (excluded & set(dependencies)) and not (excluded & set(files)),
            'exact mutable bookkeeping permission is unbound or widened')
        require(all(dependencies.get(name)==checksum for name,checksum in state_permission['source_bindings_sha256'].items())
            and dependencies.get(str(pointer(root,state_permission['evidence_report'])))==state_permission['evidence_report']['sha256'],
            'bookkeeping observer/native API/review evidence is outside frozen closure')
        require(runtime.get('complete_directory_inventories'), 'runtime directory integrity is required for bookkeeping exception')
        launcher_pointer=contract.get('launcher_environment_review')
        launcher=read(pointer(root,launcher_pointer))
        require(dependencies.get(str(pointer(root,launcher_pointer)))==launcher_pointer['sha256']
            and launcher.get('schema')=='r10_PhysMani_actual_launcher_source_environment_review_v5'
            and launcher.get('preparation_environment_not_substituted_for_worker_evidence') is True
            and launcher.get('actual_case_observer',{}).get('source')==dict(
                path=str(Path(runtime_state.__file__).resolve()),sha256=file_sha(runtime_state.__file__))
            and all(dependencies.get(str(Path(item['path']).resolve()))==item['sha256']
                for item in launcher.get('source_chain',[])) and launcher.get('source_chain'),
            'reviewed real launcher/conda/Xvfb source chain or actual-child observer is missing')
    else:
        require(not runtime.get('excluded_state_paths') and runtime.get('mutable_state_permission') is None,
                'undeclared mutable runtime state exclusion')
    state_inside_directory=False
    for inventory in runtime.get('complete_directory_inventories',[]):
        directory=Path(inventory['root']).resolve()
        require(directory.is_dir(),'runtime directory vanished')
        actual={str(path.resolve()) for path in directory.rglob('*') if path.is_file()
            and '__pycache__' not in path.parts and '.git' not in path.parts and path.suffix!='.pyc'}-excluded
        declared=normalized(root,inventory.get('files_sha256',{}))
        require(actual==set(declared) and inventory.get('file_count')==len(declared)
            and all(runtime_files.get(name)==checksum for name,checksum in declared.items()),
            'runtime directory has unknown/missing/unbound files: '+str(directory))
        if state_pointer is not None and Path(state_permission['resolved_path']).is_relative_to(directory):
            state_inside_directory=True
    require(state_pointer is None or state_inside_directory,'bookkeeping state is outside audited runtime directories')
    archive = read(pointer(root, contract['source_archive']))
    require(archive.get('schema') == SOURCE_ARCHIVE and archive.get('all_inputs_verified_before_and_after') is True,
            'actual source archive proof missing')
    expected_sources = {name: checksum for name, checksum in dependencies.items()
        if Path(name).suffix in ('.py', '.sh', '.c', '.cpp', '.cu', '.cuh', '.h', '.hpp')}
    archived = {}
    for row in archive.get('entries', []):
        name = str(Path(row['original_path']).resolve())
        require(name in expected_sources and name not in archived and row['sha256'] == expected_sources[name],
                'duplicate/unknown/mismatched archived actual source')
        pointer(root, dict(path=row['path'], sha256=row['sha256'])); archived[name] = row['sha256']
    require(archived == expected_sources, 'every actual executable source must have an immutable archive')
    for field in ('config_access_review', 'runtime_inventory', 'source_archive'):
        path = pointer(root, contract[field])
        require(files.get(str(path)) == contract[field]['sha256'], 'execution proof is outside actual lock: '+field)
    if require_approved:
        _VERIFIED_DEPENDENCIES[(str(root), value['sha256'])] = MappingProxyType(dict(dependencies))
    return dict(contract=contract, dependencies=dependencies, source_archive=archive,
        jit_artifact_profile=jit_profile,
        mutable_state_permission=state_permission,
        fingerprint=dict(version=VERSION, projected_config=project_baseline_config(config),
            canonical_VAL_roster=roster, dependencies=dependencies,
            config_access_review=contract['config_access_review'], runtime_inventory=contract['runtime_inventory']))


class ProjectedBaselineBackend:
    """Use existing evaluator/policy unchanged with an explicit config boundary.

    root/inner are injected for CPU tests. Production uses the existing native
    backend and the already owned shared reset subclass. No simulator launches
    until environment() is explicitly called by the original evaluator.
    """
    def __init__(self, *, execution_contract, root=ROOT, inner=None, simulation_class=None):
        self.root, self.execution_contract = Path(root).resolve(), deepcopy(execution_contract)
        if inner is None:
            from scripts.evaluate_multitask_case_r10 import NativeBackend
            inner = NativeBackend()
        self.inner, self.simulation_class, self.witness = inner, simulation_class, None
        self._import_hash_cache = {}
        self._runtime_state_initial=None;self._runtime_state_permission=None
        self._runtime_state_before=None;self._runtime_state_pointer=None
        self.jit_session=None;self.jit_profile=None;self.jit_receipt=None

    def start_jit_observer(self,prepared):
        contract=read(pointer(self.root,self.execution_contract));profile_pointer=contract.get('jit_artifact_profile')
        if profile_pointer is None:return
        require(prepared['args'].split=='validation' and prepared['args'].method=='physmani',
            'JIT derived-artifact profile is VAL PhysMani only')
        dependencies=observer_dependencies(self.root,self.execution_contract)
        self.jit_profile=jit_artifacts.load_profile(self.root,profile_pointer,dependencies)
        require('torch' not in sys.modules and not any(key.startswith('triton.') for key in sys.modules),
            'explicit JIT cache initialization must precede first Torch/Triton initialization')
        initialization=jit_artifacts.initialize_cache_environment(self.root,self.jit_profile)
        self.jit_session=jit_artifacts.ArtifactSession(self.root,prepared['out'],self.jit_profile,dependencies,
            identity=prepared['identity'])
        self.jit_session.initialization=initialization
        self.jit_session.initialized_before_Torch_Triton_import=True
        try:self.jit_session.install()
        except Exception as error:
            self.jit_receipt=self.jit_session.close(error);raise

    def close_jit_observer(self,error=None):
        if self.jit_session is None:return None
        self.jit_receipt=self.jit_session.close(error)
        if error is None:
            jit_artifacts.validate_execution(self.root,self.jit_receipt,self.jit_profile,
                observer_dependencies(self.root,self.execution_contract),identity=self.jit_session.identity)
        return self.jit_receipt

    def __getattr__(self, name): return getattr(self.inner, name)

    def projected(self, prepared):
        require(prepared['args'].method == 'physmani', 'projected backend is only for PhysMani')
        result = dict(prepared); result['config'] = project_baseline_config(prepared['config'])
        return result

    def environment(self, prepared):
        projected = self.projected(prepared); args, config, row = projected['args'], projected['config'], projected['row']
        if args.split == 'test':
            # This adapter does not grant TEST permission or alter archive reset.
            require(read(pointer(self.root,self.execution_contract)).get('import_edge_review') is None,
                    'VAL-only unreachable-import review cannot authorize TEST')
            return self.inner.environment(projected)
        require(args.split in ('train', 'validation') and isinstance(config.get('development_reset_profile'), dict),
                'explicit shared development reset profile required')
        contract = read(pointer(self.root, self.execution_contract))
        require(contract.get('import_edge_review') is None or args.split=='validation',
                'VAL-only unreachable-import review cannot authorize TRAIN')
        require(contract.get('schema') == CONTRACT and contract.get('version') == VERSION
            and contract.get('approved_before_VAL_execution') is True and contract.get('projected_backend_required') is True
            and contract.get('projected_config_sha256') == digest(config),
            'backend actual config differs from its approved pre-execution projection')
        state_pointer=contract.get('mutable_state_permission');state_before=None;launcher_environment=None
        if state_pointer is not None:
            self._runtime_state_pointer=deepcopy(state_pointer)
            self._runtime_state_permission=runtime_state.load_permission(self.root,state_pointer,config)
            state_before=runtime_state.snapshot(self._runtime_state_permission['resolved_path'])
            self._runtime_state_before=deepcopy(state_before)
            runtime=read(pointer(self.root,contract['runtime_inventory']))
            dependencies={name:checksum for values in contract['dependency_groups'].values()
                for name,checksum in normalized(self.root,values).items()}
            launcher_environment=runtime_state.observe_launcher_environment(dependencies,self.root,config,runtime['interpreter'],
                jit_profile=self.jit_profile)
        binding = calibration_dependency(row)
        ledger = read(pointer(self.root, contract['calibration_ledger']))
        require(binding in ledger['entries'], 'actual reset camera source differs from frozen calibration ledger')
        if self.simulation_class is None:
            from .development_reset_recovery import SharedPhysicalDevelopmentSimulation
            simulation_class = SharedPhysicalDevelopmentSimulation
        else: simulation_class = self.simulation_class
        source = dict(task=args.task, raw_episode_id=row['raw_episode_id'], path=row['path'])
        if 'variation' in row: source['variation'] = row['variation']
        env = simulation_class(args.task, row['path'], source=source, split=args.split,
            protocol=config, seed=config['seed'], replica=args.replica, servo_lead=config.get('local_servo_lead', 1.4))
        try:
            if state_pointer is not None:
                self._runtime_state_initial=dict(permission=deepcopy(state_pointer),before=state_before,
                    headless=runtime_state.observe_headless())
            descriptions, _ = env.reset_development(variation=row.get('variation'))
        except Exception:
            env.close(); raise
        self.witness = dict(version=VERSION, execution_contract=deepcopy(self.execution_contract),
            projected_config_sha256=digest(config), full_config_identity_preserved=True,
            projection_helper_sha256=file_sha(Path(__file__)), split=args.split, method='physmani',
            shared_reset_receipt=deepcopy(env.reset_recovery_receipt),
            shared_reset_permission=deepcopy(config['development_reset_profile']),
            calibration_binding_sha256=digest(binding),
            complete=False, imported_dependency_files_sha256={}, loaded_native_files_sha256={})
        if self._runtime_state_initial is not None:
            self.witness['mutable_runtime_state_initial']=deepcopy(self._runtime_state_initial)
            self.witness['actual_launcher_environment']=launcher_environment
        if self.jit_session is not None:
            self.witness['jit_artifact_profile']=deepcopy(contract['jit_artifact_profile'])
        return env, descriptions

    def finalize_runtime_state(self):
        if self._runtime_state_initial is None:return None
        return dict(schema=runtime_state.WITNESS,**deepcopy(self._runtime_state_initial),
            after=runtime_state.snapshot(self._runtime_state_permission['resolved_path']),
            limitations=deepcopy(runtime_state.LIMITATIONS))

    def infrastructure_runtime_state(self):
        if self._runtime_state_pointer is None:return None
        return dict(schema='r10_failed_case_runtime_bookkeeping_observations_v1',complete=False,
            permission=deepcopy(self._runtime_state_pointer),before=deepcopy(self._runtime_state_before),
            initial_headless=None if self._runtime_state_initial is None else deepcopy(self._runtime_state_initial['headless']),
            after=runtime_state.diagnostic_snapshot(),counted_as_policy_failure=False,
            limitations=deepcopy(runtime_state.LIMITATIONS))

    def initial_conditions(self, env, observation, robot):
        initial = self.inner.initial_conditions(env, observation, robot)
        if self.witness is not None: initial['baseline_execution_dependency_witness'] = self.witness
        return initial

    def _imports(self):
        dependencies = observer_dependencies(self.root, self.execution_contract)
        # Actual imported files must be included; no unbound late-import source
        # or extension is silently inferred from a familiar module name.
        observed = {}
        virtual = {}
        for registry_key, module in list(sys.modules.items()):
            alias = torch_virtual_owner(registry_key, module)
            if alias is not None:
                path, binding = alias
            else:
                name = getattr(module, '__file__', None)
                if not name: continue
                require(Path(name).is_absolute(), 'actual imported module has an unreviewed relative file: '+str(name))
                path = Path(name).resolve()
            if path.suffix == '.pyc':
                try:source=Path(importlib.util.source_from_cache(str(path)))
                except ValueError:source=path.with_suffix('.py')
                if source.is_file():path=source.resolve()
            if not path.is_file():
                archive=next((parent for parent in path.parents
                    if parent.suffix in ('.zip','.egg','.whl') and parent.is_file()),None)
                require(archive is not None,'actual imported module file is missing or unbound: '+str(path))
                path=archive.resolve()
            stat = path.stat(); marker = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
            cached = self._import_hash_cache.get(str(path))
            checksum = cached[1] if cached and cached[0] == marker else file_sha(path)
            self._import_hash_cache[str(path)] = (marker, checksum)
            if dependencies.get(str(path)) != checksum:
                require(self.jit_session is not None,'actual imported module is outside frozen closure: '+str(path))
                if '.so' in path.name:
                    binding=self.jit_session.authorize_native(path,checksum)
                    self.witness.setdefault('generated_native_file_bindings',{})[str(path)]=binding
                else:
                    binding=self.jit_session.authorize_python(registry_key,module,path,checksum)
                    self.witness.setdefault('generated_python_module_bindings',{})[registry_key]=binding
            observed[str(path)] = checksum
            if alias is not None:
                binding['owner_file'] = dict(path=str(path), sha256=checksum)
                virtual[registry_key] = binding
        self.witness['imported_dependency_files_sha256'].update(observed)
        self.witness.setdefault('virtual_module_owner_bindings', {}).update(virtual)
        # CoppeliaSim/CUDA libraries loaded through dlopen are not necessarily
        # Python modules. Bind the actual interpreter and mapped native files.
        native = {Path(sys.executable).resolve()}
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith('/') and '.so' in Path(fields[5]).name:
                path = Path(fields[5]).resolve()
                require(path.is_file(), 'mapped native library has no immutable readable file: '+str(path))
                native.add(path)
        for path in native:
            stat = path.stat(); marker = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
            cached = self._import_hash_cache.get(str(path))
            checksum = cached[1] if cached and cached[0] == marker else file_sha(path)
            self._import_hash_cache[str(path)] = (marker, checksum)
            if dependencies.get(str(path)) != checksum:
                require(self.jit_session is not None,'actual interpreter/native library is outside frozen closure: '+str(path))
                binding=self.jit_session.authorize_native(path,checksum)
                self.witness.setdefault('generated_native_file_bindings',{})[str(path)]=binding
            self.witness['loaded_native_files_sha256'][str(path)] = checksum
        self.witness['actual_python_executable'] = str(Path(sys.executable).resolve())

    def baseline(self, prepared, env, observation, robot):
        adapter, model = self.inner.baseline(self.projected(prepared), env, observation, robot)
        if self.witness is None: return adapter, model  # official archive TEST is unchanged
        backend = self
        self._imports()
        # A native-terminal reset can legitimately finish with zero actions.
        self.witness['complete'] = True
        class WitnessedAdapter:
            def __getattr__(self, name): return getattr(adapter, name)
            def predict(self, *args, **kwargs):
                result = adapter.predict(*args, **kwargs); backend._imports(); return result
            def execute(self, *args, **kwargs):
                result = adapter.execute(*args, **kwargs); backend._imports()
                backend.witness['complete'] = True
                return result
        return WitnessedAdapter(), model


def validate_baseline_component(root, early_plan_pointer, execution_contract, final_config_pointer,
                                final_lock_pointer, canonical_roster_pointer):
    """Audit all 275 original raw cases; issue only a reusable VAL component."""
    root = Path(root).resolve(); plan_path = pointer(root, early_plan_pointer); plan = read(plan_path)
    require(plan.get('version') == 'r10_fresh_policy_queue_v1'
        and plan.get('sha256') == digest({k: v for k, v in plan.items() if k != 'sha256'})
        and plan.get('split') == 'validation' and plan.get('methods') == ['physmani']
        and plan.get('replicas') == 1 and plan.get('source_episodes') == plan.get('expected_cases') == 275
        and plan.get('official_TEST_used_for_fitting') is False
        and plan.get('case_isolation') == 'fresh_simulator_process_per_episode',
        'an immutable full 275-case baseline-only VAL plan is required')
    early_config = read(pointer(root, plan['config'])); early_lock = read(pointer(root, plan['frozen_inputs']))
    final_config = read(pointer(root, final_config_pointer)); final_lock = read(pointer(root, final_lock_pointer))
    roster = canonical_VAL_roster(read(pointer(root, canonical_roster_pointer)))
    require(canonical_VAL_roster(read(pointer(root, plan['manifest']))) == roster,
            'early baseline canonical VAL source rows changed')
    projection = compare_config_dependencies(early_config, final_config)
    early_files = normalized(root, early_lock['files'])
    for field in ('config', 'manifest', 'parent_source', 'child_source'):
        require(early_files.get(str(absolute(root, plan[field]['path']))) == plan[field]['sha256'],
                'actual early config/roster/evaluator/queue source was not frozen: '+field)
    closure = verify_execution_contract(root, execution_contract, early_config, early_lock, roster)
    entrypoints = closure['contract']['execution_entrypoints']
    require(all(plan[field] == entrypoints[field] for field in ('parent_source', 'child_source'))
        and str(absolute(root, entrypoints['launcher_source']['path'])) == str(root/'scripts/run_aligned_gpu.sh'),
        'actual early plan/evaluator/launcher differs from its reviewed execution entrypoints')
    require(datetime.fromisoformat(closure['contract']['frozen_utc']) <= datetime.fromisoformat(plan['created_utc']),
            'execution dependency contract was not frozen before the early queue')
    final_files = normalized(root, final_lock['files'])
    for name, checksum in closure['dependencies'].items():
        require(final_files.get(name) == checksum and file_sha(name) == checksum,
                'final baseline execution closure changed: '+name)
    output = absolute(root, plan['output'])
    require(output.is_relative_to(root/'artifacts/framework/multitask_r10'), 'baseline output escaped project')
    status = read(output/'status.json')
    require(status.get('complete') is True and status.get('plan_sha256') == plan['sha256']
        and status.get('completed') == status.get('expected') == 275 and not status.get('errors')
        and status.get('stop_requested') is False, 'baseline VAL queue is not clean/complete')
    jobs = plan.get('jobs', []); by_source = {(row['task'], row['raw_episode_id']): row for row in roster}
    seen = set(); completed = []; archive = closure['dependencies']
    require(len(jobs) == 275 and {p.name for p in (output/'results').glob('*.json')} ==
        {job['job_id']+'.json' for job in jobs}, 'missing/foreign/duplicate baseline canonical output')
    for job in jobs:
        row = job['row']; key = (row['task'], row['raw_episode_id'])
        require(key in by_source and row == by_source[key] and key not in seen
            and job.get('source_raw_episode_id') == row['raw_episode_id']
            and job.get('replica') == 0 and job.get('exploration_seed') is None and job.get('record_demo') is False,
            'baseline VAL job roster/replica/demo differs')
        expected = case_identity(split='validation', task=key[0], raw_episode_id=key[1]+'__policy_aug_r0',
            method='physmani', source_row=row, config_sha256=plan['config']['sha256'],
            inputs_sha256=plan['frozen_inputs']['sha256'], parent_raw_episode_id=key[1], replica=0)
        require(job['identity'] == expected and job['job_id'] == expected['sha256'][:20], 'actual early job identity changed')
        case_path = output/'results'/(job['job_id']+'.json'); record = read(case_path)
        validate_completed_case(record, expected); parent = record.get('parent_case_process_receipt', {})
        child = absolute(root, parent['child_path'])
        require(child.name == 'case.json' and child.is_relative_to(output/'cases'/job['job_id']) and file_sha(child) == parent['child_sha256']
            and read(child) == {k: v for k, v in record.items() if k != 'parent_case_process_receipt'}
            and parent.get('plan_sha256') == plan['sha256']
            and parent.get('parent_source_sha256') == plan['parent_source']['sha256']
            and parent.get('child_process_id') == record['process_id']
            and type(parent.get('launcher_process_id')) is int and parent['launcher_process_id'] > 0,
            'raw child/parent/actual plan chain changed')
        command = parent.get('command', [])
        require(command.count('--config') == command.count('--frozen-inputs') == command.count('--manifest') == 1
            and command[command.index('--config')+1] == plan['config']['path']
            and command[command.index('--frozen-inputs')+1] == plan['frozen_inputs']['path']
            and command[command.index('--manifest')+1] == plan['manifest']['path'], 'actual historical command/config/lock changed')
        expected_command = ['bash', 'scripts/run_aligned_gpu.sh', plan['child_source']['path'],
            '--config', plan['config']['path'], '--manifest', plan['manifest']['path'],
            '--frozen-inputs', plan['frozen_inputs']['path'], '--split', 'validation',
            '--task', key[0], '--raw-episode-id', key[1], '--method', 'physmani', '--replica', '0',
            '--out', str(child.parent.relative_to(root))]
        if 'baseline_execution_contract' in plan:
            expected_command.insert(3, '--collect-case')
            require(plan['baseline_execution_contract'] == execution_contract,
                    'actual wrapper plan contract differs from the component contract')
            expected_command += ['--execution-contract', execution_contract['path']]
        require(command == expected_command, 'actual baseline command differs from its exact planned argv')
        for name in ('parent_source', 'child_source'):
            require(archive.get(str(absolute(root, plan[name]['path']))) == plan[name]['sha256'],
                    'actual evaluator/queue source is outside frozen execution archive')
        require(file_sha(absolute(root, parent['log_path'])) == parent['log_sha256'], 'original native log changed')
        require(record.get('video') is None and record.get('demo_only') is False
            and record.get('cleanup_completed') is True and record.get('formal_test_used') is False
            and record.get('baseline_executor') == early_config['baseline_executor']
            and record.get('checkpoint_sha256') == archive[str(absolute(root, early_config['baseline_checkpoint']))],
            'baseline VAL execution/checkpoint/demo/cleanup evidence differs')
        require(all(isinstance(record.get(name), (int, float)) and math.isfinite(record[name]) and record[name] >= 0
            for name in ('compute_seconds', 'wall_seconds', 'simulation_seconds'))
            and abs(record['simulation_seconds']-record['counted_clock']['seconds']) <= 2e-5
            and record.get('evaluation_process_pid') == record['process_id']
            and record.get('source_row_sha256') == expected['source_row_sha256'],
            'actual baseline runtime/source/pid measurement differs')
        witness = record['recorded_initial_conditions'].get('baseline_execution_dependency_witness', {})
        require(witness.get('version') == VERSION and witness.get('execution_contract') == execution_contract
            and witness.get('projected_config_sha256') == projection['projection_sha256']
            and witness.get('full_config_identity_preserved') is True and witness.get('complete') is True
            and witness.get('projection_helper_sha256') == archive[str(root/'our_mani/multitask/baseline_component.py')]
            and witness.get('calibration_binding_sha256') == digest(calibration_dependency(row))
            and witness.get('split') == 'validation' and witness.get('method') == 'physmani',
            'missing actual projected backend execution witness')
        reset = record['recorded_initial_conditions']['reset_receipt'].get('reset_recovery_receipt')
        require(isinstance(reset, dict) and reset == witness.get('shared_reset_receipt')
            and reset.get('complete') is True and reset.get('permission') == early_config['development_reset_profile']
            and reset.get('source_resampled') is False and reset.get('official_TEST_allowed') is False,
            'missing actual shared same-source development reset witness')
        require(record.get('config_sha256') == plan['config']['sha256']
            and record.get('manifest_sha256') == plan['manifest']['sha256']
            and record.get('frozen_input_lock') == dict(path=str(absolute(root, plan['frozen_inputs']['path'])),
                sha256=plan['frozen_inputs']['sha256'])
            and record.get('entrypoint_sha256') == (archive.get(str(root/'scripts/evaluate_multitask_case_r10.py'))
                if 'baseline_execution_contract' in plan else plan['child_source']['sha256']),
            'actual early full config/lock/source identity was rewritten')
        if 'baseline_execution_contract' in plan:
            wrapper = record.get('baseline_projected_wrapper_execution', {})
            require(wrapper.get('version') == 'r10_actual_projected_PhysMani_VAL_wrapper_v1'
                and wrapper.get('execution_contract') == execution_contract
                and wrapper.get('actual_wrapper') == plan['child_source']
                and wrapper.get('evaluator_source') == dict(path='scripts/evaluate_multitask_case_r10.py',
                    sha256=archive.get(str(root/'scripts/evaluate_multitask_case_r10.py')))
                and wrapper.get('full_config_identity_preserved') is True
                and wrapper.get('original_record_fields_rewritten') is False
                and wrapper.get('projected_config_sha256') == projection['projection_sha256'],
                'actual projected wrapper and original evaluator identities are missing or rewritten')
            base = pointer(root, wrapper['base_run_case_receipt'])
            inputs_path = pointer(root, wrapper['base_run_case_inputs'])
            require(base == child.parent/'evaluate_run_case_receipt.json' and inputs_path == child.parent/'inputs.json'
                and read(base) == {k:v for k,v in record.items()
                    if k not in ('parent_case_process_receipt','baseline_projected_wrapper_execution')},
                'wrapper changes original evaluator receipt or its actual input provenance')
            inputs = read(inputs_path)
            require(inputs.get('identity') == expected and inputs.get('process_id') == record['process_id']
                and inputs.get('frozen_input_lock') == record['frozen_input_lock']
                and inputs.get('entrypoint_sha256') == record['entrypoint_sha256']
                and inputs.get('manifest_sha256') == plan['manifest']['sha256']
                and inputs.get('original_manifest_row_sha256') == digest(job['row'])
                and normalized(root, inputs.get('files', {})) == normalized(root, early_lock['files'])
                and inputs.get('argv') == [plan['child_source']['path'], *command[3:]],
                'wrapper original evaluator inputs do not bind actual full config/lock/argv/process')
            final_witness = wrapper.get('dependency_witness_after_cleanup', {})
            dynamic = {'imported_dependency_files_sha256','loaded_native_files_sha256','actual_python_executable',
                       'virtual_module_owner_bindings','generated_python_module_bindings','generated_native_file_bindings'}
            require({k:v for k,v in final_witness.items() if k not in dynamic}
                == {k:v for k,v in witness.items() if k not in dynamic}
                and final_witness.get('actual_python_executable') == witness.get('actual_python_executable'),
                'wrapper post-cleanup dependency observer changed reset or execution evidence')
            generated={}
            profile_pointer=closure['contract'].get('jit_artifact_profile')
            if profile_pointer is not None:
                require(witness.get('jit_artifact_profile')==profile_pointer
                    and final_witness.get('jit_artifact_profile')==profile_pointer,'actual JIT profile differs from contract')
                profile=jit_artifacts.load_profile(root,profile_pointer,archive)
                execution=wrapper.get('jit_artifact_execution')
                require((root/execution['path']).resolve()==child.parent/'jit_artifacts/execution.json',
                    'JIT execution receipt escaped actual child namespace')
                generated=jit_artifacts.validate_observed_bindings(root,execution,profile,archive,final_witness,
                    identity=expected,process_id=record['process_id'])
                jit_artifacts.validate_observed_bindings(root,execution,profile,archive,witness,
                    identity=expected,process_id=record['process_id'])
                for field in ('generated_python_module_bindings','generated_native_file_bindings'):
                    require(all(final_witness.get(field,{}).get(key)==value for key,value in witness.get(field,{}).items()),
                        'generated loader binding changed during cleanup')
            else:
                require(wrapper.get('jit_artifact_execution') is None
                    and not witness.get('generated_python_module_bindings') and not witness.get('generated_native_file_bindings')
                    and not final_witness.get('generated_python_module_bindings') and not final_witness.get('generated_native_file_bindings'),
                    'case claims undeclared generated code/native authority')
            for field in ('imported_dependency_files_sha256','loaded_native_files_sha256'):
                final_files = normalized(root, final_witness.get(field, {}))
                require(all(final_files.get(name) == checksum for name,checksum in witness.get(field, {}).items())
                    and all(archive.get(name) == checksum or generated.get(name)==checksum for name,checksum in final_files.items()),
                    'wrapper actual post-cleanup import/native closure differs or is unbound')
            validate_virtual_owner_bindings(final_witness.get('virtual_module_owner_bindings', {}), archive,
                final_witness.get('imported_dependency_files_sha256', {}))
            require(all(final_witness.get('virtual_module_owner_bindings', {}).get(key) == value
                for key, value in witness.get('virtual_module_owner_bindings', {}).items()),
                'wrapper post-cleanup virtual namespace owners changed')
            if closure['contract'].get('mutable_state_permission') is not None:
                runtime_state.validate_case_witness(wrapper.get('mutable_runtime_state_execution'),
                    closure['mutable_state_permission'],closure['contract']['mutable_state_permission'],
                    witness.get('mutable_runtime_state_initial'))
                runtime=read(pointer(root,closure['contract']['runtime_inventory']))
                runtime_state.validate_launcher_environment(witness.get('actual_launcher_environment'),archive,root,
                    early_config,runtime['interpreter'],jit_profile=profile if profile_pointer is not None else None)
            else:
                require(wrapper.get('mutable_runtime_state_execution') is None,
                        'case claims an undeclared mutable runtime exception')
        imports = normalized(root, witness.get('imported_dependency_files_sha256', {}))
        validate_virtual_owner_bindings(witness.get('virtual_module_owner_bindings', {}), archive, imports)
        generated=generated if 'baseline_execution_contract' in plan else {}
        require(all(archive.get(name) == checksum or generated.get(name)==checksum for name, checksum in imports.items()),
            'actual imported module was unbound')
        natives = normalized(root, witness.get('loaded_native_files_sha256', {}))
        require(witness.get('actual_python_executable') in natives
            and all(archive.get(name) == checksum or generated.get(name)==checksum for name, checksum in natives.items()),
            'actual interpreter/native mapped library was unbound')
        if closure['mutable_state_permission'] is not None:
            binary=str(Path(closure['mutable_state_permission']['resolved_path']).parent.parent/'libcoppeliaSim.so')
            require(natives.get(binary)==closure['mutable_state_permission']['source_bindings_sha256'][binary],
                    'actual loaded CoppeliaSim binary does not match the exact bookkeeping runtime')
        completed.append(dict(task=key[0], raw_episode_id=key[1], actual_identity=expected,
            canonical_receipt=dict(path=str(case_path.relative_to(root)), sha256=file_sha(case_path)),
            raw_child_receipt=dict(path=str(child.relative_to(root)), sha256=parent['child_sha256'])))
        seen.add(key)
    require(seen == set(by_source), 'baseline VAL source coverage incomplete')
    return dict(schema=COMPONENT, complete=True, baseline_component_reusable=True,
        actual_early_queue_plan=deepcopy(early_plan_pointer), actual_early_config=plan['config'],
        actual_early_lock=plan['frozen_inputs'], final_config=deepcopy(final_config_pointer),
        final_lock=deepcopy(final_lock_pointer), execution_contract=deepcopy(execution_contract),
        projection_sha256=projection['projection_sha256'], execution_dependency_fingerprint=digest(closure['fingerprint']),
        sources=275, completed_cases=completed, raw_identity_rewritten=False, raw_receipts_rewritten=False,
        single_full_config_identity_claimed=False, requires_complete_five_method_VAL_cases=1375,
        requires_final_model_coverage_and_initial_condition_parity=True,
        formal_validation_release_issued=False, formal_TEST_permitted=False, actual_TEST_read=False)
