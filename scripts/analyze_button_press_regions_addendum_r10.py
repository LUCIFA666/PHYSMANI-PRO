"""Source/CAD and one-tick timing counterchecks for actual200 press diagnosis."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,release_preflight as release
from scripts.diagnose_button_press_native_success200_r10 import heldout_endpoints,distribution,write
BASE=ROOT/'artifacts/framework/multitask_r10'


def run(output):
    out=a.local(ROOT,output);a.require(not out.exists(),'fresh button addendum required');out.mkdir(parents=True)
    inputs=release._Inputs(ROOT);p=BASE/'button_press_native_success200_endpoint_regions_diagnostic_v2/report.json'
    report=inputs.read(str(p),a.file_sha(p));rows=inputs.read_pointer(report['all_source_endpoint_audit'])
    timelines={row['source_id']:row['frames'] for row in inputs.read_pointer(report['terminal_timelines'])}
    inputs.source_snapshot(report['frozen22_core_source_snapshot'])
    for path in (Path(__file__).resolve(),ROOT/'scripts/diagnose_button_press_native_success200_r10.py'):
        inputs.file(str(path),a.file_sha(path))
    geometry_map_path=BASE/'frozen_TRAIN_geometry_map_v8.json';geometry_map=inputs.read(str(geometry_map_path),a.file_sha(geometry_map_path))
    CAD={};source={}
    for task in ('push_moving_button','push_moving_button_high_speed'):
        asset_path=inputs.file(geometry_map[task]);asset=a.read(asset_path);bounds={}
        for role,body in asset['bodies'].items():
            points=np.concatenate([np.asarray(component['points_body']) for component in body['components']])
            bounds[role]=dict(minimum=points.min(axis=0).tolist(),maximum=points.max(axis=0).tolist(),
                              size=np.ptp(points,axis=0).tolist())
        CAD[task]=dict(asset=a.pointer(ROOT,asset_path),internal_asset_sha256=asset['asset_sha256'],bounds=bounds)
        task_path=BASE/f'command_goal_reference_rule_v5_translation_revision3_setup/readonly_source/{task}.py'
        text=inputs.file(str(task_path)).read_text();tree=ast.parse(text)
        expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
        descent=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GripperDescendingCondition')
        native_task=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('PushMovingButton','PushMovingButtonHighSpeed'))
        episode=next(n for n in native_task.body if isinstance(n,ast.FunctionDef) and n.name=='init_episode')
        step=next(n for n in native_task.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        source[task]=dict(actual_task_source=a.pointer(ROOT,task_path),expert_AST=ast.unparse(expert),
            descending_condition_AST=ast.unparse(descent),episode_success_conditions_AST=ast.unparse(episode),
            step_AST=ast.unparse(step),press_axis_world='downward z; fixed source waypoint1 z minus .01m',
            native_success_has_explicit_TCP_to_plate_XY_distance_gate=False)
    conditions=ROOT.parent/'phsimani/PhysMani/third_party/RLBench/rlbench/backend/conditions.py'
    copied=out/'readonly_current_source/conditions.py';copied.parent.mkdir();shutil.copyfile(conditions,copied)
    checksum=a.file_sha(conditions);inputs.file(str(copied),checksum)
    tree=ast.parse(copied.read_text());joint=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointCondition')
    source['joint_condition']=dict(current_readonly_source=a.pointer(ROOT,copied),source_original_path=str(conditions),
        historical_execution_dependency_claimed=False,AST=ast.unparse(joint),condition_is_absolute_displacement=True,
        condition_is_contact_sensor=False,condition_checks_TCP_to_plate_XY=False)
    timing={};tails={};proxy={}
    for task in CAD:
        success=[deepcopy(row) for row in rows if row['task']==task and row['native_success']]
        for row in success:
            trace=timelines[row['source_id']]
            end=trace[-1]
            for ticks in (1,2):
                prior=trace[-1-ticks]
                button=np.asarray(prior['tcp_xyzw'][:3])-prior['tcp_relative_current_button']
                plate=np.asarray(prior['tcp_xyzw'][:3])-prior['tcp_relative_current_plate']
                row['relative'][f'button_previous{ticks}_native_tick']=(np.asarray(end['tcp_xyzw'][:3])-button).tolist()
                row['relative'][f'plate_previous{ticks}_native_tick']=(np.asarray(end['tcp_xyzw'][:3])-plate).tolist()
        timing[task]={basis:heldout_endpoints(success,basis) for basis in success[0]['relative'] if 'previous' in basis}
        tails[task]=sorted(success,key=lambda row:np.linalg.norm(row['relative']['committed_world_goal']),reverse=True)[:8]
        selected=[row for row in rows if row['task']==task]
        proxy[task]=dict(success_count=len(success),
            success_depression_below_initial3mm=sum(row['plate_relative_button_change_from_initial_m'][2]<-.003 for row in success),
            success_rise_above_initial3mm=[row['parent'] for row in success if row['plate_relative_button_change_from_initial_m'][2]>.003],
            native_success_and_simple_plate_descent_proxy_agreement=sum(row['native_success']==(
                abs(row['plate_relative_button_change_from_initial_m'][2])>.003 and
                row['native_last10_TCP_z_descent_including_terminal_m']>.03) for row in selected),
            total_preserved=len(selected),proxy_is_not_verified_native_joint_or_runtime_completion=True)
    a.require(a.file_sha(conditions)==checksum,'readonly condition source changed during diagnosis');inputs.rehash_all()
    proposal=dict(schema='r10_button_press_actual200_minimal_repair_proposal_v1',TRAIN_only=True,
        current_body_to_plate_reference_swap_rejected_by_actual_crossfit=True,
        one_or_two_native_tick_reference_shift_is_countercheck_not_runtime_rule=True,
        contact_cluster_oracle_not_admitted_as_selector=True,
        proposed_next_branch=[
            dict(task='push_moving_button_high_speed',action='Independent source-backed press-intention candidate: original .25s forecast plus original boundary clip .03, preserving raw current metrics/events/censoring; actual100 full-curve and consumer verification before any core admission.',
                 actual_endpoint_position_cap_m=report['per_task']['push_moving_button_high_speed']['independent_parent_K1_crossfits']['committed_world_goal']['position_tolerance_m'],full_curve_pass_established=False),
            dict(task='push_moving_button',action='Keep press pending: original .5s intention endpoint still fails .03; audit all tail sources and native absolute joint/descent outcome separately. Do not switch reference or drop tails to pass.',
                 actual_endpoint_position_cap_m=report['per_task']['push_moving_button']['independent_parent_K1_crossfits']['committed_world_goal']['position_tolerance_m']),
            dict(task='both',action='Keep independent native task-success evidence distinct from actual contact/command completion. A later fresh RGBD plate deformation + known downward motion verifier needs separate physical calibration; raw native boolean and grasp-association flag are insufficient.',
                 native_success_can_be_public_completion=False,public_contact_manifold_or_terminal_joint_recorded=False)],
        original106_success_and94_censored_preserved=True,minimum8_position03_orientation15_unchanged=True,
        frozen22_core_or_production_artifacts_modified=False,native_GPU_new_collection_or_formal_model_fit_started=False)
    write(out/'minimal_repair_proposal.json',proposal)
    result=dict(schema='r10_button_press_actual200_source_CAD_timing_addendum_v1',actual200_endpoint_report=a.pointer(ROOT,p),
        source_and_CAD_evidence=source,CAD=CAD,one_two_native_tick_reference_crossfits=timing,
        all_tail_rows_retained_worst8_for_inspection=tails,native_joint_depression_proxy_counterchecks=proxy,
        minimal_repair_proposal=a.pointer(ROOT,out/'minimal_repair_proposal.json'),verified_files_sha256=inputs.files,
        final_direct_input_rehash_passed=True,frozen22_source_snapshot=report['frozen22_core_source_snapshot'])
    write(out/'report.json',result);return a.pointer(ROOT,out/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(run(args.out)))
