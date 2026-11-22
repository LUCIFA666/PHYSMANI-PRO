"""Freeze exact TRAIN CAD feature permissions and mathematical AST parity."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.cup_features import (VERSION,PERMISSION_SCHEMA,CUP_TASKS,CUP_ROLES,FeatureConfig,
    features_from_frozen_cad,validate_frozen_cup_cad,load_frozen_cup_features)

BASE=ROOT/'artifacts/framework/multitask_r10'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('all cup feature artifacts must be in our-mani')
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def mathematical_ast_parity(original_path,production_path):
    original=ast.parse(Path(original_path).read_text());production=ast.parse(Path(production_path).read_text())
    kinds=(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)
    before={node.name:ast.dump(node,include_attributes=False) for node in original.body if isinstance(node,kinds)}
    after={node.name:ast.dump(node,include_attributes=False) for node in production.body if isinstance(node,kinds)}
    if not before or any(name not in after or dump!=after[name] for name,dump in before.items()):
        raise ValueError('production mathematical function/class AST differs from frozen experimental prototype')
    return dict(all_original_functions_and_classes_AST_identical=True,count=len(before),
        mathematical_function_and_class_AST_sha256={name:hashlib.sha256(dump.encode()).hexdigest() for name,dump in before.items()},
        newly_added_functions=sorted(set(after)-set(before)))


def run(output,geometry_map):
    output=Path(output).resolve();geometry_map=Path(geometry_map).resolve()
    if not output.is_relative_to(BASE) or output.exists():raise ValueError('fresh R10 permission directory required')
    original=ROOT/'scripts/experimental_cup_features_r10.py';production=ROOT/'our_mani/multitask/cup_features.py'
    plan_path=BASE/'TRAIN_cup_feature_replay_plan_v1.json';report_path=BASE/'cup_feature_replay_v1/report.json'
    audit_path=BASE/'cup_feature_replay_v1/independent_parent_and_input_audit.json'
    plan,report,audit=read(plan_path),read(report_path),read(audit_path)
    if not (report['complete'] is True and report['TRAIN_only'] is True and report['VAL_used'] is False
            and report['TEST_used'] is False and report['production_policy_authorized'] is False
            and report['overall']['body_frames']==336 and report['plan']['sha256']==sha(plan_path)
            and audit['complete'] is True and audit['all_plan_input_hashes_verified'] is True
            and audit['all_prediction_hashes_verified'] is True and audit['predictions_finished_before_GT_metrics'] is True
            and audit['four_independent_parent_summary']['body_frames']==192
            and plan['inputs_sha256']['scripts/experimental_cup_features_r10.py']==sha(original)
            and plan['configuration']==asdict(FeatureConfig()) and plan['version']==VERSION):
        raise ValueError('immutable fixed TRAIN experimental evidence differs')
    parity=mathematical_ast_parity(original,production)
    source_paths=(original,production,Path(__file__),ROOT/'tests/test_cup_feature_permissions_r10.py',
        ROOT/'our_mani/contracts.py',ROOT/'our_mani/perception.py',ROOT/'our_mani/multitask/depth_pose_consistency.py')
    inputs={pointer(path)['path']:sha(path) for path in (*source_paths,plan_path,report_path,audit_path,geometry_map)}
    canonical_pointer=plan['canonical_CAD'];canonical_path=ROOT/canonical_pointer['path']
    if sha(canonical_path)!=canonical_pointer['sha256']:raise ValueError('validated original canonical CAD changed')
    canonical_asset=read(canonical_path);validate_frozen_cup_cad(canonical_asset,task='place_cups_on_rotating_frame',allow_historical_canonical=True)
    inputs[canonical_pointer['path']]=canonical_pointer['sha256'];mapping=read(geometry_map)
    output.mkdir(parents=True)
    parity_report=dict(version='r10_cup_feature_production_math_AST_parity_v1',complete=True,
        fixed_configuration=asdict(FeatureConfig()),algorithm_VERSION=VERSION,
        experimental_source=pointer(original),production_source=pointer(production),**parity,
        GT_fit_or_bias_correction_applied=False,parameters_changed=False,
        source_inputs_sha256={pointer(path)['path']:sha(path) for path in source_paths})
    parity_path=output/'mathematical_AST_parity.json';write(parity_path,parity_report)
    inputs[str(parity_path.relative_to(ROOT))]=sha(parity_path)
    permissions={};tasks={}
    for task in CUP_TASKS:
        task_path=Path(mapping[task]).resolve();task_pointer=pointer(task_path);asset=read(task_path)
        validate_frozen_cup_cad(asset,task=task)
        models={role:features_from_frozen_cad(canonical_asset,asset,role) for role in CUP_ROLES}
        task_inputs={**inputs,task_pointer['path']:task_pointer['sha256']}
        manifest=dict(schema=PERMISSION_SCHEMA,task=task,TRAIN_only=True,VAL_used=False,TEST_used=False,
            roles=list(CUP_ROLES),configuration=asdict(FeatureConfig()),canonical_CAD=canonical_pointer,
            TRAIN_CAD=task_pointer,registered_TRAIN_CAD_internal_sha256=asset['asset_sha256'],
            CAD_feature_receipts={role:model.receipt for role,model in models.items()},
            production_helper_sha256=sha(production),mathematical_AST_parity=pointer(parity_path),
            evidence_report=pointer(report_path),evidence_audit=pointer(audit_path),
            world_pose_or_GT_seed_in_model=False,GT_bias_correction_applied=False,
            source_model_migration_allowed=False,parameters_changed_from_fixed_experiment=False,
            static_permission_authorizes_current_orientation=False,
            runtime_readiness_required='current causal points, unambiguous feature proposal, cylinder RMS and existing inlier gate',
            annotation_reconstruction_numeric_tolerance_m=1e-10,
            annotation_numeric_tolerance_reason='IEEE/BLAS roundoff across Python environments; not a geometric or GT bias tolerance',
            inputs_sha256=task_inputs)
        path=output/(task+'.json');write(path,manifest)
        permission=dict(enabled=True,**pointer(path),canonical_CAD=canonical_pointer,TRAIN_CAD=task_pointer,
            roles=list(CUP_ROLES),configuration=asdict(FeatureConfig()),
            evidence_report=pointer(report_path),evidence_audit=pointer(audit_path))
        reconstructed=load_frozen_cup_features(permission,asset)
        if set(reconstructed)!=set(CUP_ROLES):raise ValueError('strict model loader roundtrip differs')
        permissions[task]=permission;tasks[task]=dict(manifest=pointer(path),TRAIN_CAD=task_pointer,
            CAD_feature_receipts=manifest['CAD_feature_receipts'],strict_loader_roundtrip_verified=True)
    write(output/'cup_feature_priors.json',permissions)
    result=dict(version='r10_frozen_TRAIN_cup_feature_permission_build_v1',complete=True,TRAIN_only=True,
        VAL_used=False,TEST_used=False,native_started=False,GPU_used=False,
        production_algorithm_math_AST_unchanged=True,configuration_unchanged=True,GT_bias_correction_applied=False,
        current_orientation_observability_authorized_by_manifest=False,
        production_helper=pointer(production),mathematical_AST_parity=pointer(parity_path),
        evidence_report=pointer(report_path),evidence_audit=pointer(audit_path),
        cup_feature_priors=permissions,tasks=tasks,inputs_sha256=inputs)
    write(output/'report.json',result)
    print(json.dumps(dict(complete=True,tasks=len(tasks),parity_count=parity['count'],report=pointer(output/'report.json'))))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(BASE/'cup_feature_permissions_v1'))
    parser.add_argument('--geometry-map',default=str(BASE/'frozen_TRAIN_geometry_map_v8.json'))
    args=parser.parse_args();run(args.out,args.geometry_map)
