"""Prepare a reviewable, unapproved full PhysMani VAL dependency contract.

Reads only canonical metadata/camera calibration and hashes installed runtime
files. Never starts the policy, CUDA, simulator, native extension or GPU queue.
ROOT reviews new immutable approval artifacts before any native execution.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask import baseline_component as component
from our_mani.multitask import baseline_runtime_state as runtime_state
from our_mani.multitask import baseline_import_scope as import_scope
from our_mani.multitask import baseline_jit_artifacts as jit_artifacts
from our_mani.multitask.baseline_dependencies import inventory_baseline_dependencies
from our_mani.multitask.evaluation import atomic_json,digest,file_sha
from scripts import evaluate_multitask_case_r10 as evaluator
from scripts import evaluate_physmani_projected_VAL_r10 as wrapper

VERSION='r10_projected_PhysMani_VAL_contract_preparation_v1'
DEFAULT_PYTHON='/home/ubuntu/physmani-conda/envs/physmani5090/bin/python'
DEFAULT_RUNTIME_ROOTS=(
 '/home/ubuntu/physmani-conda/envs/physmani5090',
 '/home/ubuntu/projects/physmani_reproduction/runtime/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04',
 '/home/ubuntu/projects/physmani_reproduction/runtime/usr', '/usr/lib/x86_64-linux-gnu',
 '/usr/lib64', '/usr/share/X11', '/etc/ld.so.cache', '/usr/bin',
 '/home/ubuntu/physmani-conda/bin','/home/ubuntu/physmani-conda/lib',
 '/home/ubuntu/physmani-conda/condabin','/home/ubuntu/physmani-conda/etc',
 '/home/ubuntu/physmani-conda/ssl','/home/ubuntu/physmani-conda/.condarc')


def read(path):return json.loads(Path(path).read_text())


def source_access_inventory(root):
    entries=[]
    for name in component.REVIEW_SOURCE_FILES:
        path=Path(root)/name
        for node in ast.walk(ast.parse(path.read_text(),filename=str(path))):
            if isinstance(node,ast.Subscript):
                rendered=ast.unparse(node.value)
                if rendered in ('config','protocol','self.config','self.protocol','env.protocol'):
                    entries.append(dict(path=name,line=node.lineno,access=ast.unparse(node)))
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='get':
                if ast.unparse(node.func.value) in ('config','protocol','self.config','self.protocol','env.protocol'):
                    entries.append(dict(path=name,line=node.lineno,access=ast.unparse(node)))
    return entries


def retained_pointers(config,root):
    """Only exact project pointers; preserve retained shared prior evidence."""
    result={}
    def visit(value):
        if isinstance(value,dict):
            if set(value)>={'path','sha256'} and isinstance(value['path'],str):
                path=(Path(root)/value['path']).resolve()
                component.require(path.is_file() and file_sha(path)==value['sha256'],'retained config dependency changed: '+str(path))
                result[str(path)]=dict(path=str(path),sha256=value['sha256'])
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(component.project_baseline_config(config))
    return list(result.values())


def access_review(root,entrypoints):
    return dict(schema='r10_PhysMani_config_access_code_review_v1',approved=False,branch='physmani_TRAIN_VAL_only',
        baseline_read_keys=list(component.BASELINE_READ_KEYS),excluded_roots=list(component.OURS_ROOTS),
        excluded_artifact_leaves=list(component.OURS_LEAVES),excluded_metadata=list(component.METADATA),
        sources_sha256={str((Path(root)/name).resolve()):file_sha(Path(root)/name) for name in component.REVIEW_SOURCE_FILES},
        execution_entrypoints=entrypoints,AST_access_inventory=source_access_inventory(root),
        review_status='actual wrapper and source dependency draft; ROOT approval still required before GPU/native',
        non_baseline_branch_evidence=[
            dict(source='scripts/evaluate_multitask_case_r10.py',branch='preflight method!=physmani / NativeBackend.actor / run_case actor else',
                 excludes=['interaction','colour','rigid_layout','cv_value','context_value','context_checkpoint','continuation_identities']),
            dict(source='our_mani/multitask/factory.py',branch='create_actor only; never called by physmani',
                 consumed_ours_artifacts=['interaction','colour','rigid_layout']),
            dict(source='our_mani/multitask/baseline_component.py',branch='projected() before environment and baseline',
                 absent_from_actual_baseline_config=list(component.OURS_LEAVES)+list(component.OURS_ROOTS)),
            dict(source='our_mani/multitask/maps.py',branch='SharedMultiObjectMap',
                 evidence='native static mesh/roles; no learned_artifacts reads'),
            dict(source='our_mani/multitask/reset_parity.py',branch='capture_recorded_initial_conditions',
                 evidence='fixed PERMISSIONS and physical clock/gripper; no learned_artifacts reads'),
            dict(source='our_mani/aligned/physmani.py',branch='build_policy and adapter/WorldModel',
                 evidence='baseline checkpoint/bounds/instructions/output and fixed WM constructor; no ours learned fields')],
        geometry_retained=True,unknown_config_fields_retained_and_compared=True,
        raw_full_config_and_original_evaluator_identity_preserved=True)


def derive_config(shared, released, geometry_manifest,root=ROOT):
    config=deepcopy(shared)
    config.update(baseline_checkpoint=released['baseline_checkpoint'],baseline_decision_ticks=10,
        baseline_executor=evaluator.EXECUTOR,baseline_reference_filter='author_direct_setpoint',
        baseline_output=str(Path(root)/'runtime/framework/disabled_baseline_wm_logger'),
        formal_comparison_ready=False,r09_completed_release='artifacts/framework/aligned_r09/final_report.json')
    component.require(set(geometry_manifest['assets'])==set(component.TASK_SPECS),'fixed14 TRAIN CAD map required')
    table=config.setdefault('learned_artifacts',{})
    for task,item in geometry_manifest['assets'].items():
        geometry=dict(path=item['path'],sha256=item['sha256'])
        values=table.setdefault(task,{})
        component.require('geometry' not in values or values['geometry']==geometry,'shared geometry pointer changed')
        values['geometry']=geometry
        for field in component.OURS_LEAVES:values.setdefault(field,None)
    component.require(isinstance(config.get('development_reset_profile'),dict),'use current accepted shared reset profile')
    return config


def prepare(config,*,output,manifest_path=None,python_executable=DEFAULT_PYTHON,
            runtime_roots=DEFAULT_RUNTIME_ROOTS,extra_source_roots=(),dry_run=False,inventory_builder=None,
            bookkeeping_evidence=None,import_edge_evidence=None,launcher_evidence=None,
            jit_derived_artifacts=False,jit_evidence=()):
    output=wrapper.inside(output);component.require(not output.exists(),'fresh dependency preparation namespace required')
    output.mkdir(parents=True)
    manifest_path=Path(manifest_path or ROOT/'artifacts/framework/protocol/validation_manifest.json').resolve()
    manifest=read(manifest_path);roster=component.canonical_VAL_roster(manifest)
    config_path=output/'early_config.json';atomic_json(config_path,config)
    evidence=Path(bookkeeping_evidence or ROOT/'artifacts/framework/multitask_r10/'
        'PhysMani_settings_dat_state_review_v1/headless_api_review_v2.json')
    permission_path=output/'mutable_bookkeeping_permission_draft.json'
    atomic_json(permission_path,runtime_state.draft_permission(ROOT,config,wrapper.ptr(evidence)))
    state_pointer=wrapper.ptr(permission_path)
    jit_pointer=None
    if jit_derived_artifacts:
        jit_path=output/'jit_artifact_profile_draft.json'
        atomic_json(jit_path,jit_artifacts.draft_profile(ROOT,python_executable,jit_evidence))
        jit_pointer=wrapper.ptr(jit_path)
    edge_evidence=Path(import_edge_evidence or ROOT/'artifacts/framework/multitask_r10/'
        'PhysMani_VAL_ours_import_edge_ROOT_review_v3/report.json')
    edge_path=output/'unreachable_import_edge_review_draft.json'
    atomic_json(edge_path,import_scope.draft_review(ROOT,wrapper.ptr(edge_evidence)))
    edge_pointer=wrapper.ptr(edge_path)
    launcher_evidence=Path(launcher_evidence or ROOT/'artifacts/framework/multitask_r10/'
        'PhysMani_launcher_environment_review_v6/report.json')
    ledger_path=output/'calibration_ledger.json'
    atomic_json(ledger_path,dict(schema='r10_canonical_VAL_camera_dependency_ledger_v1',
        entries=[component.calibration_dependency(row) for row in roster],future_motion_images_or_expert_arrays_read=False))
    entrypoints=dict(parent_source=wrapper.ptr(Path(wrapper.__file__)),child_source=wrapper.ptr(Path(wrapper.__file__)),
                     launcher_source=wrapper.ptr(ROOT/'scripts/run_aligned_gpu.sh'))
    local_sources=[*evaluator.required_local_sources('physmani',split='validation'),Path(wrapper.__file__),Path(__file__),
        ROOT/'our_mani/multitask/baseline_component.py',ROOT/'our_mani/multitask/baseline_dependencies.py',
        ROOT/'our_mani/multitask/development_reset_recovery.py',ROOT/'scripts/run_multitask_fresh_queue_r10.py']
    calibration=[wrapper.ptr(ledger_path),wrapper.ptr(config_path),wrapper.ptr(manifest_path),
        wrapper.ptr(ROOT/config.get('r09_completed_release','artifacts/framework/aligned_r09/final_report.json')),
        wrapper.ptr(evidence),*read(evidence).get('evidence_dependencies',[]),
        wrapper.ptr(edge_evidence),edge_pointer,
        wrapper.ptr(launcher_evidence),
        *retained_pointers(config,ROOT)]
    if jit_pointer is not None:calibration.extend([jit_pointer,*jit_evidence])
    builder=inventory_baseline_dependencies if inventory_builder is None else inventory_builder
    inventory=builder(ROOT,config,python_executable=python_executable,runtime_roots=runtime_roots,
        extra_source_roots=extra_source_roots,local_sources=local_sources,calibration_inputs=calibration,
        dry_run=dry_run,archive_directory=None if dry_run else output/'source_archive',
        mutable_state_permission=state_pointer,import_edge_review=edge_pointer)
    if jit_pointer is not None:inventory['runtime_inventory']['jit_artifact_profile']=jit_pointer
    inventory_path=output/'dependency_inventory.json';atomic_json(inventory_path,inventory)
    if dry_run:
        return dict(version=VERSION,native_started=False,GPU_started=False,approved=False,
            dry_run=True,inventory=wrapper.ptr(inventory_path),gaps=inventory['gaps'])
    runtime_path=output/'runtime_inventory.json';atomic_json(runtime_path,inventory['runtime_inventory'])
    review_path=output/'config_access_review_draft_v3.json';atomic_json(review_path,access_review(ROOT,entrypoints))
    groups=deepcopy(inventory['dependency_groups'])
    for path in (runtime_path,review_path,ROOT/inventory['source_archive']['path']):
        groups['calibration_inputs'][str(path.resolve())]=file_sha(path)
    contract=dict(schema=component.CONTRACT,version=component.VERSION,frozen_utc=datetime.now(timezone.utc).isoformat(),
        approved_before_VAL_execution=False,actual_dependency_closure_reviewed=False,projected_backend_required=True,
        projected_config_sha256=digest(component.project_baseline_config(config)),canonical_VAL_roster_sha256=digest(roster),
        shared_development_reset_profile=config['development_reset_profile'],dependency_groups=groups,
        calibration_ledger=wrapper.ptr(ledger_path),config_access_review=wrapper.ptr(review_path),
        execution_entrypoints=entrypoints,complete_source_root_inventories=inventory['complete_source_root_inventories'],
        mutable_state_permission=state_pointer,
        import_edge_review=edge_pointer,
        launcher_environment_review=wrapper.ptr(launcher_evidence),
        runtime_inventory=wrapper.ptr(runtime_path),source_archive=inventory['source_archive'],
        preparation_scope='unapproved metadata-only draft; actual late imports/native maps still fail closed',
        unresolved_review_items=inventory['gaps'],static_inventory_reviewed=False,actual_native_or_GPU_execution=False)
    if jit_pointer is not None:contract['jit_artifact_profile']=jit_pointer
    contract_path=output/'execution_contract_draft.json';atomic_json(contract_path,contract)
    files={name:sha for values in groups.values() for name,sha in values.items()}
    files[str(contract_path)]=file_sha(contract_path)
    lock_path=output/'frozen_inputs_draft.json';atomic_json(lock_path,dict(files=files,approved_for_execution=False,
        prepared_only=True,native_started=False,GPU_started=False,formal_TEST_permission=False))
    component.verify_execution_contract(ROOT,wrapper.ptr(contract_path),config,{'files':files},roster,require_approved=False)
    plan_path=output/'queue_plan_draft.json'
    wrapper.prepare_plan(config_path,manifest_path,lock_path,contract_path,plan_path=plan_path,output=output/'executions')
    wrapper.verify_plan(plan_path,approved=False)
    result=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),approved=False,
        native_started=False,GPU_started=False,formal_TEST_permission=False,baseline_component_qualified=False,
        plan=wrapper.ptr(plan_path),config=wrapper.ptr(config_path),lock=wrapper.ptr(lock_path),contract=wrapper.ptr(contract_path),
        inventory=wrapper.ptr(inventory_path),files=inventory['total_files'],bytes=inventory['total_bytes'],
        unresolved_review_items=inventory['gaps'],source_archive=inventory['source_archive'],
        prepared_VAL_cases=275,complete_five_method_VAL_requirement=1375,
        needs_ROOT_review_and_new_approved_contract_lock_plan_before_execution=True)
    atomic_json(output/'preparation_report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--config',type=Path);source.add_argument('--shared-config',type=Path)
    p.add_argument('--released-config',type=Path,default=ROOT/'config/framework/aligned_protocol_r09_baseline_v6.json')
    p.add_argument('--geometry-manifest',type=Path,default=ROOT/'artifacts/framework/multitask_r10/frozen_TRAIN_geometry_manifest_v11.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path)
    p.add_argument('--python',default=DEFAULT_PYTHON);p.add_argument('--runtime-root',action='append')
    p.add_argument('--extra-source-root',action='append',default=[]);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--bookkeeping-evidence',type=Path)
    p.add_argument('--import-edge-evidence',type=Path)
    p.add_argument('--launcher-evidence',type=Path)
    p.add_argument('--jit-derived-artifacts',action='store_true')
    p.add_argument('--jit-evidence',type=Path,action='append',default=[])
    args=p.parse_args()
    config=(read(args.config) if args.config else derive_config(read(args.shared_config),read(args.released_config),read(args.geometry_manifest)))
    result=prepare(config,output=args.out,manifest_path=args.manifest,python_executable=args.python,
        runtime_roots=DEFAULT_RUNTIME_ROOTS if args.runtime_root is None else tuple(args.runtime_root),
        extra_source_roots=args.extra_source_root,dry_run=args.dry_run,bookkeeping_evidence=args.bookkeeping_evidence,
        import_edge_evidence=args.import_edge_evidence,launcher_evidence=args.launcher_evidence,
        jit_derived_artifacts=args.jit_derived_artifacts,jit_evidence=[wrapper.ptr(path) for path in args.jit_evidence])
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
