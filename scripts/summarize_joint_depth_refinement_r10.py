"""Descriptive fixed TRAIN joint-refinement metrics; no selection or fitting."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c


def summarize(directory):
    directory=Path(directory);report=common.read(directory/'report.json');plan=common.read(report['plan']['path'])
    common.require(c.sha(report['plan']['path'])==report['plan']['sha256'],'diagnostic plan file changed')
    common.require(report['complete'] and report['all_predictions_completed_before_GT_metrics'],'incomplete/GT boundary missing')
    rows=[];all_metrics=[];optimizer=Counter();calls=0;extra=0
    for case in report['cases']:
        predictions=common.read(directory/(case['case_id']+'.json'))
        for role in ('cup0','cup1','cup2'):
            metrics=[m for m in case['metrics'] if m['role']==role]
            valid=[m for m in metrics if m['status']=='refined_fixed_original_candidates']
            before=np.asarray([[m['original_pose']['position_error_m'],m['original_pose']['full_SO3_error_deg']] for m in valid])
            after=np.asarray([[m['joint_pose']['position_error_m'],m['joint_pose']['full_SO3_error_deg']] for m in valid])
            metric=dict(case_id=case['case_id'],parent_source_group=case['parent_source_group'],role=role,frames=len(metrics),refined_frames=len(valid))
            if len(valid):
                metric.update(original_position_error_m=common.scalar_stats(before[:,0]),joint_position_error_m=common.scalar_stats(after[:,0]),
                    original_SO3_error_deg=common.scalar_stats(before[:,1]),joint_SO3_error_deg=common.scalar_stats(after[:,1]),
                    position_degraded_frames=int((after[:,0]>before[:,0]+1e-9).sum()),orientation_degraded_frames=int((after[:,1]>before[:,1]+1e-6).sum()),
                    original_under10deg_frames=int((before[:,1]<10).sum()),joint_under10deg_frames=int((after[:,1]<10).sum()),
                    joint_objective_decreased_frames=sum(m['joint_objective_after']<m['joint_objective_before'] for m in valid),
                    refinement_wall_seconds=sum(m['runtime_seconds'] for m in valid))
            rows.append(metric);all_metrics+=valid
        for frame in predictions:
            for body in frame['bodies'].values():
                if body['status']!='refined_fixed_original_candidates':continue
                extra+=body.get('original_extra_candidate_not_optimized',0)
                for candidate in body['refinement']['candidates']:
                    optimizer[str(candidate['status'])]+=1;calls+=candidate['residual_calls']
    before=np.asarray([[m['original_pose']['position_error_m'],m['original_pose']['full_SO3_error_deg']] for m in all_metrics])
    after=np.asarray([[m['joint_pose']['position_error_m'],m['joint_pose']['full_SO3_error_deg']] for m in all_metrics])
    result=dict(version='r10_joint_depth_refinement_descriptive_summary_v1',complete=True,report=common.pointer(directory/'report.json'),
        plan=report['plan'],summary_source=common.pointer(__file__),cases=len(report['cases']),
        unique_raw_TRAIN_parent_sources=len({case['parent_source_group'] for case in report['cases']}),
        expected_body_frames=report['expected_body_frames'],refined_body_frames=len(all_metrics),
        fixed_parameters=plan['refinement_config'],fixed_optimized_candidates=4,
        extra_original_history_candidates_omitted=extra,optimizer_status_counts=dict(optimizer),residual_calls=calls,
        total_refinement_wall_seconds=sum(m['runtime_seconds'] for m in all_metrics),
        per_pose_refinement_wall_seconds=common.scalar_stats([m['runtime_seconds'] for m in all_metrics]),
        original_position_error_m=common.scalar_stats(before[:,0]),joint_position_error_m=common.scalar_stats(after[:,0]),
        original_SO3_error_deg=common.scalar_stats(before[:,1]),joint_SO3_error_deg=common.scalar_stats(after[:,1]),
        original_under10deg_frames=int((before[:,1]<10).sum()),joint_under10deg_frames=int((after[:,1]<10).sum()),
        position_degraded_frames=int((after[:,0]>before[:,0]+1e-9).sum()),orientation_degraded_frames=int((after[:,1]>before[:,1]+1e-6).sum()),
        joint_objective_decreased_frames=sum(m['joint_objective_after']<m['joint_objective_before'] for m in all_metrics),
        rows=rows,pose_promotion_supported=False,core_perception_modified=False,native_GPU_run=False,
        TRAIN_only=True,VAL_TEST_used=False,thresholds_or_weights_searched=False,GT_only_final_metrics=True,
        model_or_pipeline_SR_evaluated=False,software=dict(numpy=np.__version__,scipy=scipy.__version__),
        limitations=['fixed original four candidate refinement and original-scorer history; not integrated joint-refined history tracker',
            'joint objective descent need not improve physical pose; no GT/outcome-based filtering, bounds or candidate seeds',
            'clipped residual has zero gradient in saturated gaps; nearest CAD correspondence, front-surface ownership and visibility boundaries remain nonsmooth',
            'bilinear depth interpolation can blend foreground/background at depth edges; all four support pixels finite does not certify common surface identity',
            'true foreground occlusion/invalid depth/frustum-out supplies no penalty or complete orientation evidence',
            'short TRAIN hold sequences and repeated parent sources/frames are correlated; no independent CI or generalization claim'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();report=summarize(args.directory);c.write_json(args.directory/'summary.json',report)
    lines=[f"固定TRAIN联合SE3优化：{report['refined_body_frames']}/{report['expected_body_frames']}bodyframes，4候选×max_nfev30；只有{report['unique_raw_TRAIN_parent_sources']}个不同原始父源。",
        f"完整SO3<10°：{report['original_under10deg_frames']}→{report['joint_under10deg_frames']}；位置退化{report['position_degraded_frames']}帧；朝向退化{report['orientation_degraded_frames']}帧。",
        f"joint目标下降{report['joint_objective_decreased_frames']}帧；refinement累计wall={report['total_refinement_wall_seconds']:.2f}s，单pose均值={report['per_pose_refinement_wall_seconds']['mean']:.3f}s。",
        '仅描述固定候选离线优化效果，不支持cup完整朝向升级；不改core、不native、不VAL/TEST、不搜权重阈值。',
        'case | cup | SO3 p50原→新(deg) | position p95原→新(mm) | degraded position frames']
    for row in report['rows']:
        if row['refined_frames']:
            lines.append(f"{row['case_id']} | {row['role']} | {row['original_SO3_error_deg']['median']:.2f}→{row['joint_SO3_error_deg']['median']:.2f} | {row['original_position_error_m']['p95']*1000:.2f}→{row['joint_position_error_m']['p95']*1000:.2f} | {row['position_degraded_frames']}/{row['frames']}")
    lines+=['限制：'+item for item in report['limitations']]
    (args.directory/'summary.txt').write_text('\n'.join(lines)+'\n')
    print({key:report[key] for key in ('refined_body_frames','original_under10deg_frames','joint_under10deg_frames','position_degraded_frames','total_refinement_wall_seconds')})


if __name__=='__main__':main()
