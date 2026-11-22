"""Independent task TEST authorization around frozen PhysMani algorithms.

VAL-only source-edge permission is never used for TEST. The old JIT profile
is an exact stock-source/cache descriptor here, not a grant of TEST access.
The new contract and complete-three-method task release grant this scope.
"""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType

from . import baseline_component as component
from . import baseline_jit_artifacts as jit
from . import baseline_runtime_state as state
from .evaluation import atomic_json,case_identity,digest,file_sha,verify_inputs
from .protocol import validate_protocol
from .specs import TASK_SPECS

VERSION='r10_PhysMani_official_task_TEST_wrapper_v1'
CONTRACT='r10_PhysMani_official_task_TEST_runtime_contract_v1'
NORMAL_BASKETBALL='moving_basketball_in_hoop'
OFFICIAL_MANIFEST='artifacts/framework/multitask_r10/official_test_manifest.json'
RESOURCE_KEY='TEST_GPU_execution_resource_budget'
METHODS=('cv_geometry','context_geometry','physmani')
RELEASE_SOURCE='our_mani/multitask/geometry_task_release.py'
ROOT=Path(__file__).resolve().parents[2]
require=component.require


def read(path):return json.loads(Path(path).read_text())
def absolute(root,path):return (Path(root)/path).resolve()


def official_task_rows(manifest,task):
    require(task in TASK_SPECS,'unknown official task')
    rows=manifest.get('episodes',[])
    counts=Counter(row.get('task') for row in rows)
    require(manifest.get('split')=='test' and manifest.get('mapping_status')=='verified'
        and manifest.get('episode_count')==len(rows)==560
        and manifest.get('training_calibration_or_parameter_selection_used') is False
        and set(counts)==set(TASK_SPECS) and set(counts.values())=={40}
        and len({(row['task'],row['raw_episode_id']) for row in rows})==560,
        'official TEST must retain all560 original rows and exactly40 per task')
    selected=[row for row in rows if row['task']==task]
    require(len(selected)==40 and all(row.get('mapping_status')=='verified' for row in selected),
        'entire original official task40 required')
    return selected


def structural_release(config,task,files,*,root=ROOT):
    require(config.get('formal_comparison_ready') is True,'TEST requires completed-three-method task VAL first')
    reference=config.get('task_validation_release')
    require(isinstance(reference,dict),'TEST requires completed-three-method task VAL first')
    path=component.pointer(root,reference);normalized=component.normalized(root,files)
    require(normalized.get(str(path))==reference['sha256'],'task VAL release is not in actual frozen lock')
    report=read(path)
    require(report.get('schema')=='r10_task_complete_three_method_geometry_validation_release_v1'
        and report.get('task')==task and report.get('complete') is True
        and report.get('formal_test_permitted') is True and report.get('protocol_acceptance_passed') is True
        and report.get('split')=='validation' and report.get('TEST_used') is False
        and report.get('methods')==list(METHODS) and report.get('Value_required') is False
        and report.get('Value_artifacts_loaded') is False and report.get('expected_source_cases')==20
        and report.get('completed_method_cases')==report.get('expected_method_cases')
        ==3*report.get('expected_source_cases',-1),
        'TEST requires actual complete-three-method task VAL release, never VAL-only baseline permission')
    return reference


def metadata_preflight(args,*,root=ROOT):
    """No Torch/native imports; typed TEST-source and release header gate."""
    from scripts import evaluate_multitask_case_r10 as evaluator
    root=Path(root).resolve()
    require(args.split=='test' and args.method=='physmani' and args.task==NORMAL_BASKETBALL,
        'new official baseline entrypoint is normal basketball TEST only')
    require(args.replica==0 and args.exploration_seed is None and not args.record_demo and not args.demo_only,
        'official TEST cannot use replicas/exploration/demo or training labels')
    config_path,manifest_path,lock_path=[absolute(root,value) for value in (args.config,args.manifest,args.frozen_inputs)]
    config,manifest,lock=map(read,(config_path,manifest_path,lock_path));validate_protocol(config)
    files=lock['files'];verify_inputs(root,files)
    normalized=component.normalized(root,files)
    for path in (config_path,manifest_path,Path(__file__),root/'scripts/evaluate_PhysMani_task_TEST_r10.py',
                 Path(evaluator.__file__),root/'our_mani/multitask/geometry_task_release.py',root/'our_mani/multitask/gpu_budget.py'):
        require(normalized.get(str(path.resolve()))==file_sha(path),'TEST executable/config/source not frozen: '+str(path))
    require(manifest_path==root/OFFICIAL_MANIFEST,'official full560 manifest path changed')
    rows=official_task_rows(manifest,args.task)
    chosen=[row for row in rows if row['raw_episode_id']==args.raw_episode_id]
    require(len(chosen)==1,'TEST source not in original task40')
    row=chosen[0]
    require(Path(row['path']).is_dir(),'official reset alias missing')
    release=structural_release(config,args.task,files,root=root)
    budget=config.get(RESOURCE_KEY)
    require(isinstance(budget,dict) and set(budget)=={'profile','environment_contract'},
        'TEST-specific96SM profile must be predeclared before VAL; do not repurpose VALargv')
    for ref in budget.values():component.pointer(root,ref)
    out=absolute(root,args.out)
    require(out.is_relative_to(root/'artifacts/framework/multitask_r10')
        and (not out.exists() or not any(out.iterdir())),'fresh TEST output required; no retry/overwrite')
    identity=case_identity(split='test',task=args.task,raw_episode_id=row['raw_episode_id'],method='physmani',
        source_row=row,config_sha256=file_sha(config_path),inputs_sha256=file_sha(lock_path),
        parent_raw_episode_id=None,replica=0,demo_only=False,exploration_seed=None)
    return dict(args=args,config=config,config_path=config_path,manifest_path=manifest_path,
        manifest_sha256=file_sha(manifest_path),row=deepcopy(row),original_row=deepcopy(row),out=out,
        identity=identity,files=files,lock_path=lock_path,lock_sha256=file_sha(lock_path),
        r09_release=evaluator.r09_release(config,files),r10_validation_release=None,
        expected_task_validation_release=release,record_demo=False)


def authorize_task_TEST(prepared,*,root=ROOT,checker=None):
    # This CPU model/provenance checker can import Torch. Call it only after
    # the stock JIT observer has been installed, always before any native reset.
    if checker is None:
        from .geometry_task_release import require_task_test_release
        checker=require_task_test_release
    result=checker(root,prepared['args'].task,prepared['expected_task_validation_release'],
        prepared['config'],prepared['files'],official_test_manifest=prepared['manifest_path'])
    require(result.get('passed') is True and result.get('VAL_method_cases')==3*result.get('VAL_source_cases',-1)
        and result.get('TEST_source_cases')==40,'actual three-method task VAL/model proof did not pass')
    prepared['r10_validation_release']=result
    return result


def validate_ROOT_review(root,reference,config_path,manifest_path,release,original_VAL_plan,files):
    path=component.pointer(root,reference);review=read(path)
    ref=lambda value:dict(path=str(Path(value).resolve().relative_to(root)),sha256=file_sha(value))
    require(component.normalized(root,files).get(str(path))==reference['sha256']
        and review.get('schema')=='r10_ROOT_PhysMani_normal_official_TEST_runtime_approval_v1'
        and review.get('approved') is True and review.get('config')==ref(config_path)
        and review.get('manifest')==ref(manifest_path) and review.get('task_validation_release')==release
        and review.get('original_VAL_runtime_plan')==ref(original_VAL_plan)
        and review.get('full560_task40_original_sources') is True
        and review.get('VAL_import_edge_permission_used_for_TEST') is False,
        'exact independently bound new TEST runtime ROOT review is required')
    return review


def observe_TEST_resource_environment(root,config):
    """Bind exact launch exports; the separate budget audit proves the SM set."""
    from . import gpu_budget
    budget=config[RESOURCE_KEY]
    profile=read(component.pointer(root,budget['profile']))
    gpu_budget.validate_profile(profile,root)
    contract=read(component.pointer(root,budget['environment_contract']))
    require(contract.get('schema')==gpu_budget.CONTRACT_SCHEMA
        and contract.get('budget_profile')==gpu_budget.pointer(component.pointer(root,budget['profile'])),
        'TEST budget environment contract/profile differ')
    actual={name:os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES',)+gpu_budget.MPS_KEYS}
    namespace=Path(profile['namespace'])
    require(actual['CUDA_VISIBLE_DEVICES']=='0'
        and actual['CUDA_MPS_PIPE_DIRECTORY']==str(namespace/'p')
        and actual['CUDA_MPS_LOG_DIRECTORY']==str(namespace/'l')
        and isinstance(actual['CUDA_MPS_SM_PARTITION'],str)
        and re.fullmatch(re.escape(profile['GPU']['uuid'])+r'/[A-Za-z0-9+/=]+',actual['CUDA_MPS_SM_PARTITION']),
        'TEST must inherit its exact predeclared owned MPS environment')
    return dict(profile=deepcopy(budget['profile']),environment_contract=deepcopy(budget['environment_contract']),
        actual_exports=actual,scope='environment identity only; actual96SM and clients require budget execution audit',
        actual_SM_partition_qualification_claimed=False)


def prepare_TEST_runtime_contract(root,config_path,manifest_path,common_union_lock,original_VAL_plan,*,
                                  output,ROOT_review=None):
    """Fresh metadata only, callable after the actual three-VAL release exists.

    The supplied common union already binds all actual TEST-preflight sources.
    No VAL source-edge pruning is applied to this new TEST execution closure.
    """
    from scripts.evaluate_physmani_projected_VAL_r10 import verify_plan
    root=Path(root).resolve();output=absolute(root,output)
    require(output.is_relative_to(root) and not output.exists(),'fresh TEST contract namespace required')
    config_path,manifest_path,common_union_lock,original_VAL_plan=[absolute(root,path)
        for path in (config_path,manifest_path,common_union_lock,original_VAL_plan)]
    config,manifest,lock=map(read,(config_path,manifest_path,common_union_lock))
    validate_protocol(config)
    require(manifest_path==root/OFFICIAL_MANIFEST,'new TEST contract must retain original full560 manifest path')
    rows=official_task_rows(manifest,NORMAL_BASKETBALL)
    release=structural_release(config,NORMAL_BASKETBALL,lock['files'],root=root)
    old_plan=verify_plan(original_VAL_plan)
    original=read(component.pointer(root,old_plan['baseline_execution_contract']))
    files=component.normalized(root,lock['files']);verify_inputs(root,files)
    for path in (config_path,manifest_path,Path(__file__),root/'scripts/evaluate_PhysMani_task_TEST_r10.py',
                 root/'our_mani/multitask/geometry_task_release.py',root/'our_mani/multitask/gpu_budget.py'):
        require(files.get(str(path.resolve()))==file_sha(path),'TEST-preflight source/config must enter common union lock')
    budget=config.get(RESOURCE_KEY)
    require(isinstance(budget,dict) and set(budget)=={'profile','environment_contract'},'predeclared TEST resource scope missing')
    for ref in budget.values():component.pointer(root,ref)
    approved=ROOT_review is not None
    def ref(path):return dict(path=str(Path(path).resolve().relative_to(root)),sha256=file_sha(path))
    if approved:
        review_path=absolute(root,ROOT_review)
        files[str(review_path)]=file_sha(review_path)
        validate_ROOT_review(root,ref(review_path),config_path,manifest_path,release,original_VAL_plan,files)
    output.mkdir(parents=True)
    groups=deepcopy(original['dependency_groups']);groups['calibration_inputs'].update(files)
    groups['calibration_inputs'][str(original_VAL_plan)]=file_sha(original_VAL_plan)
    for refvalue in budget.values():
        p=component.pointer(root,refvalue);groups['calibration_inputs'][str(p)]=refvalue['sha256']
    dependencies={name:checksum for values in groups.values() for name,checksum in component.normalized(root,values).items()}
    expected={name:checksum for name,checksum in dependencies.items()
        if Path(name).suffix in ('.py','.sh','.c','.cpp','.cu','.cuh','.h','.hpp')}
    old_archive=read(component.pointer(root,original['source_archive']))
    entries=[deepcopy(row) for row in old_archive['entries'] if row['original_path'] in expected]
    known={row['original_path']:row['sha256'] for row in entries}
    require(all(expected[name]==checksum for name,checksum in known.items()),'original policy/runtime source changed')
    for name in sorted(set(expected)-set(known)):
        source=Path(name);checksum=expected[name];require(file_sha(source)==checksum,'new TEST source changed before archive')
        stored=output/'source_bytes'/checksum/source.name;stored.parent.mkdir(parents=True,exist_ok=True)
        stored.write_bytes(source.read_bytes());entries.append(dict(original_path=name,path=str(stored.relative_to(root)),sha256=checksum))
    for row in entries:component.pointer(root,dict(path=row['path'],sha256=row['sha256']))
    verify_inputs(root,dependencies)
    archive_path=output/'source_archive.json'
    atomic_json(archive_path,dict(schema=component.SOURCE_ARCHIVE,entries=entries,all_inputs_verified_before_and_after=True,
        original_immutable_runtime_archive_reused=original['source_archive'],scope='new TEST full source closure; no VAL edge permission'))
    groups['calibration_inputs'][str(archive_path)]=file_sha(archive_path)
    contract=dict(schema=CONTRACT,approved_before_TEST_execution=approved,split='test',task=NORMAL_BASKETBALL,
        full_official_TEST_sources=560,selected_original_task_sources=40,official_TEST_manifest=ref(manifest_path),
        original_task_rows_sha256=digest(rows),projected_config_sha256=digest(component.project_baseline_config(config)),
        task_validation_release=release,original_verified_VAL_runtime_plan=ref(original_VAL_plan),
        observer_stock_cache_descriptor=original['jit_artifact_profile'],old_VAL_scope_is_not_TEST_authorization=True,
        TEST_GPU_execution_resource_budget=deepcopy(budget),
        import_edge_review=None,VAL_source_edge_permission_used_for_TEST=False,development_reset_hook_used=False,
        mutable_state_permission=original['mutable_state_permission'],runtime_inventory=original['runtime_inventory'],
        dependency_groups=groups,source_archive=ref(archive_path),actual_native_or_GPU_execution=False,
        ROOT_review=None if not approved else ref(review_path))
    contract_path=output/'execution_contract.json';atomic_json(contract_path,contract)
    actual_lock=dict(lock);actual_lock['files']={**dependencies,str(archive_path):file_sha(archive_path),
        str(contract_path):file_sha(contract_path)}
    lock_path=output/'frozen_inputs.json';atomic_json(lock_path,actual_lock)
    report=dict(schema='r10_PhysMani_normal_task_TEST_runtime_metadata_preparation_v1',approved=approved,
        config=ref(config_path),manifest=ref(manifest_path),contract=ref(contract_path),lock=ref(lock_path),
        original_VAL_runtime_plan=ref(original_VAL_plan),source_archive=ref(archive_path),original_sources=560,selected_sources=40,
        TEST_policy_parameters_fitted=False,VAL_import_edge_permission_extended=False,native_GPU_started=False)
    atomic_json(output/'preparation_report.json',report)
    return report


def load_TEST_contract(root,reference,prepared,*,verify_original=None):
    """Full original runtime closure plus explicit TEST-source extensions."""
    root=Path(root).resolve();path=component.pointer(root,reference);value=read(path)
    files=component.normalized(root,prepared['files']);args=prepared['args'];config=prepared['config']
    require(files.get(str(path))==reference['sha256'] and value.get('schema')==CONTRACT
        and value.get('approved_before_TEST_execution') is True and value.get('split')=='test'
        and value.get('task')==args.task and value.get('full_official_TEST_sources')==560
        and value.get('selected_original_task_sources')==40
        and value.get('official_TEST_manifest')==dict(path=str(prepared['manifest_path'].relative_to(root)),
            sha256=prepared['manifest_sha256'])
        and value.get('original_task_rows_sha256')==digest(official_task_rows(read(prepared['manifest_path']),args.task))
        and value.get('projected_config_sha256')==digest(component.project_baseline_config(config))
        and value.get('task_validation_release')==prepared['expected_task_validation_release']
        and value.get('import_edge_review') is None
        and value.get('VAL_source_edge_permission_used_for_TEST') is False
        and value.get('development_reset_hook_used') is False,
        'new exact approved task TEST contract/full560/task40/release is required')
    original=value['original_verified_VAL_runtime_plan'];old_path=component.pointer(root,original)
    validate_ROOT_review(root,value.get('ROOT_review'),prepared['config_path'],prepared['manifest_path'],
        prepared['expected_task_validation_release'],old_path,files)
    require(value.get(RESOURCE_KEY)==config.get(RESOURCE_KEY),'TEST-specific resource profile differs from common config')
    for ref in value[RESOURCE_KEY].values():
        actual=component.pointer(root,ref)
        require(files.get(str(actual))==ref['sha256'],'TEST resource permission is not frozen')
    if verify_original is None:
        from scripts.evaluate_physmani_projected_VAL_r10 import verify_plan
        verify_original=verify_plan
    old_plan=verify_original(old_path)
    old_config=read(component.pointer(root,old_plan['config']))
    original_contract=read(component.pointer(root,old_plan['baseline_execution_contract']))
    require(value['observer_stock_cache_descriptor']==original_contract['jit_artifact_profile']
        and value.get('old_VAL_scope_is_not_TEST_authorization') is True,
        'TEST must explicitly distinguish stock cache descriptor from VAL execution permission')
    require(Path(config['baseline_checkpoint']).resolve()==Path(old_config['baseline_checkpoint']).resolve()
        and config['baseline_executor']==old_config['baseline_executor']
        and config['baseline_decision_ticks']==old_config['baseline_decision_ticks']
        and config['baseline_reference_filter']==old_config['baseline_reference_filter'],
        'TEST baseline checkpoint/executor differs from original validated policy')
    groups=value.get('dependency_groups',{})
    require(set(groups)==set(component.GROUPS) and all(groups.values()),'complete TEST runtime/source dependency groups required')
    dependencies={}
    for group in component.GROUPS:
        for name,checksum in component.normalized(root,groups[group]).items():
            require(name not in dependencies or dependencies[name]==checksum,'conflicting TEST dependencies')
            require(files.get(name)==checksum and Path(name).is_file() and file_sha(name)==checksum,
                'actual TEST source/runtime dependency changed/unbound: '+name)
            dependencies[name]=checksum
    for group in original_contract['dependency_groups'].values():
        for name,checksum in component.normalized(root,group).items():
            require(dependencies.get(name)==checksum,'TEST omitted/changed original approved baseline dependency')
    archive=read(component.pointer(root,value['source_archive']))
    require(archive.get('schema')==component.SOURCE_ARCHIVE
        and archive.get('all_inputs_verified_before_and_after') is True,'TEST actual source archive required')
    expected={name:checksum for name,checksum in dependencies.items()
        if Path(name).suffix in ('.py','.sh','.c','.cpp','.cu','.cuh','.h','.hpp')}
    actual={}
    for row in archive['entries']:
        name=str(Path(row['original_path']).resolve())
        require(name not in actual and expected.get(name)==row['sha256'],'TEST source archive unknown/duplicate/mismatched')
        component.pointer(root,dict(path=row['path'],sha256=row['sha256']));actual[name]=row['sha256']
    require(actual==expected,'TEST source closure is not fully archived')
    for source in (Path(__file__),root/'scripts/evaluate_PhysMani_task_TEST_r10.py',
                   root/'our_mani/multitask/geometry_task_release.py',root/'our_mani/multitask/gpu_budget.py'):
        require(dependencies.get(str(source.resolve()))==file_sha(source),'actual TEST entrypoint/checker is outside closure')
    require(value['mutable_state_permission']==original_contract['mutable_state_permission']
        and value['runtime_inventory']==original_contract['runtime_inventory'],'shared runtime/state policy changed')
    # Same immutable-cache optimization after this full new-scope verification.
    component._VERIFIED_DEPENDENCIES[(str(root),reference['sha256'])]=MappingProxyType(dict(dependencies))
    return dict(contract=value,dependencies=dependencies,source_archive=archive,
        original_runtime_scope_verified=True,TEST_permission_still_requires_full_task_release=True)


class TestProjectedBaselineBackend(component.ProjectedBaselineBackend):
    """New TEST authorization; original policy/executor/observer algorithms."""
    def __init__(self,*,verified,execution_contract,root=ROOT,inner=None):
        super().__init__(execution_contract=execution_contract,root=root,inner=inner)
        self.verified=verified

    def start_jit_observer(self,prepared):
        require(prepared['args'].split=='test' and prepared['args'].method=='physmani','new observer adapter is official TEST only')
        descriptor=self.verified['contract']['observer_stock_cache_descriptor']
        self.jit_profile=jit.load_profile(self.root,descriptor,self.verified['dependencies'])
        require('torch' not in sys.modules and not any(name.startswith('triton.') for name in sys.modules),
            'TEST stock observer must precede Torch/Triton and CPU model-proof imports')
        initialization=jit.initialize_cache_environment(self.root,self.jit_profile)
        self.jit_session=jit.ArtifactSession(self.root,prepared['out'],self.jit_profile,self.verified['dependencies'],identity=prepared['identity'])
        self.jit_session.initialization=initialization;self.jit_session.initialized_before_Torch_Triton_import=True
        try:self.jit_session.install()
        except Exception as error:self.jit_receipt=self.jit_session.close(error);raise

    def environment(self,prepared):
        require(prepared['r10_validation_release'] is not None,'no native TEST before full three-method task VAL verification')
        projected=self.projected(prepared);config=projected['config'];contract=self.verified['contract']
        budget_environment=observe_TEST_resource_environment(self.root,prepared['config'])
        permission=contract['mutable_state_permission']
        self._runtime_state_pointer=deepcopy(permission)
        self._runtime_state_permission=state.load_permission(self.root,permission,config)
        self._runtime_state_before=state.snapshot(self._runtime_state_permission['resolved_path'])
        runtime=read(component.pointer(self.root,contract['runtime_inventory']))
        launcher=state.observe_launcher_environment(self.verified['dependencies'],self.root,config,runtime['interpreter'],jit_profile=self.jit_profile)
        # NativeBackend.environment dispatches to AlignedOfficialSimulation and
        # reset_official. No SharedPhysicalDevelopmentSimulation/reset hook.
        env,descriptions=self.inner.environment(projected)
        try:
            require(env.reset_receipt.get('reset_kind')=='released_official_TEST_family_dispatch'
                and env.reset_receipt.get('original_source_replayed') is True,'official TEST reset dispatch changed')
            self._runtime_state_initial=dict(permission=deepcopy(permission),before=self._runtime_state_before,
                headless=state.observe_headless())
            self.witness=dict(version=VERSION,split='test',method='physmani',complete=False,
                execution_contract=deepcopy(self.execution_contract),projected_config_sha256=digest(config),
                full_config_identity_preserved=True,projection_helper_sha256=file_sha(component.__file__),
                official_reset_receipt=deepcopy(env.reset_receipt),development_reset_hook_used=False,
                full_three_method_task_VAL_release=deepcopy(prepared['r10_validation_release']),
                TEST_resource_environment=budget_environment,
                observer_stock_cache_descriptor=deepcopy(contract['observer_stock_cache_descriptor']),
                old_VAL_profile_used_as_TEST_permission=False,actual_launcher_environment=launcher,
                mutable_runtime_state_initial=deepcopy(self._runtime_state_initial),
                imported_dependency_files_sha256={},loaded_native_files_sha256={})
            return env,descriptions
        except Exception:env.close();raise
