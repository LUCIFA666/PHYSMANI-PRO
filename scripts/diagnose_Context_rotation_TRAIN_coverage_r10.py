"""Read actual Context TRAIN rotation coverage without fitting or GPU work."""
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from our_mani.multitask.context_se3 import MultiBodyContextDynamics
from our_mani.multitask.track_dataset import read_tracks,examples


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base=ROOT/'artifacts/framework/multitask_r10'
    out=base/'Context_rotation_TRAIN_coverage_diagnosis_v2'
    if out.exists():raise ValueError('fresh rotation diagnostic output required')
    checkpoint=base/'context_se3_full_v6/dynamics.pt'
    manifest=base/'context_training_view_full_v6/manifest.json'
    prior=base/'context_se3_full_v6_body_VAL_diagnosis_v2/report.json'
    files={str(path):sha(path) for path in (checkpoint,manifest,prior,Path(__file__),
        ROOT/'our_mani/multitask/context_se3.py',ROOT/'our_mani/multitask/fitting.py',
        ROOT/'our_mani/multitask/track_dataset.py')}
    torch.set_num_threads(2)
    model=MultiBodyContextDynamics(checkpoint).model
    _,episodes=read_tracks(ROOT,manifest)
    train=[episode for episode in episodes if episode.entry['split']=='train']
    if len(train)!=1400:raise ValueError('actual original1400 TRAIN hold sources required')
    for episode in train:files[str(ROOT/episode.entry['cache'])]=episode.entry['cache_sha256']
    samples,coverage=examples(train,model,steps=22)
    if len(samples)!=162269 or len(coverage)!=1400:raise ValueError('completed training window coverage differs')
    groups=defaultdict(list)
    for row in samples:groups[row['source_key']].append(row)
    summary={};total_rotation_parents=0;expected_rotation_draws=0.
    for task in sorted({key[1] for key in groups}):
        parents={key:rows for key,rows in groups.items() if key[1]==task}
        windows=[row for rows in parents.values() for row in rows]
        rotation=[row for row in windows if row['rotation_mask'].any()]
        speeds=[];accelerations=[];by_body={}
        for body in sorted({row['body_id'] for row in windows}):
            subset=[row for row in windows if row['body_id']==body]
            by_body[body]=dict(windows=len(subset),rotation_supervised_windows=int(sum(row['rotation_mask'].any() for row in subset)))
        for row in rotation:
            mask=row['rotation_mask'];w=row['targets']['angular_velocities']
            speeds.extend(np.linalg.norm(w[mask],axis=-1).tolist())
            adjacent=mask[1:]&mask[:-1]
            accelerations.extend(np.linalg.norm(np.diff(w,axis=0)[adjacent]/.05,axis=-1).tolist())
        parent_counts=[int(sum(row['rotation_mask'].any() for row in rows)) for rows in parents.values()]
        expected=sum(64*count/len(rows) for count,rows in zip(parent_counts,parents.values()))
        total_rotation_parents+=sum(count>0 for count in parent_counts);expected_rotation_draws+=expected
        stats=lambda values:None if not values else dict(mean=float(np.mean(values)),median=float(np.median(values)),q95=float(np.quantile(values,.95)))
        summary[task]=dict(parent_sources=len(parents),windows=len(windows),
            rotation_supervised_parent_sources=sum(count>0 for count in parent_counts),
            rotation_supervised_windows=len(rotation),rotation_full1p1_windows=int(sum(row['rotation_mask'][-1] for row in windows)),
            expected_rotation_supervised_draws_per_epoch_under_original_sampler=expected,
            angular_speed_rad_s=stats(speeds),angular_acceleration_rad_s2=stats(accelerations),
            repeated_overlapping_window_label_values_are_not_independent_samples=True,by_body=by_body)
    diagnostic=json.loads(prior.read_text());body_errors=[]
    lookup={(row['method'],row['task'],row['body_id']):row for row in diagnostic['by_body'] if row['horizon_seconds']==1.1}
    for (method,task,body),row in lookup.items():
        if method!='context' or row['parent_macro_angle_rad'] is None:continue
        cv=lookup[('CV',task,body)]
        body_errors.append(dict(task=task,body_id=body,Context_angle_rad=row['parent_macro_angle_rad'],
            CV_angle_rad=cv['parent_macro_angle_rad'],difference_rad=row['parent_macro_angle_rad']-cv['parent_macro_angle_rad']))
    body_errors.sort(key=lambda row:row['difference_rad'],reverse=True)
    if any(sha(path)!=checksum for path,checksum in files.items()):raise ValueError('diagnostic input changed')
    report=dict(schema='r10_actual_Context_TRAIN_rotation_coverage_and_existing_VAL_errors_v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        TRAIN_hold_sources=1400,TRAIN_windows=len(samples),rotation_supervised_TRAIN_parent_sources=total_rotation_parents,
        expected_optimizer_draws_per_epoch=1400*64,
        expected_rotation_supervised_draws_per_epoch=expected_rotation_draws,
        current_loss_coefficients=dict(normalized_position=1.,normalized_linear_velocity=.05,
            normalized_rotation_matrix=.1,normalized_angular_velocity=.01,reconstruction=.01),
        coefficient_comparison_is_not_a_gradient_magnitude_measurement=True,
        current_architecture='6D orientation input and SO3 exponential angular-acceleration integration',
        by_task=summary,existing_VAL_by_body_rotation_error=body_errors,
        verified_files_sha256=files,model_fitted_or_selected=False,native_or_GPU_started=False,
        TEST_used=False,new_training_data_or_checkpoint_created=False)
    out.mkdir()
    p=out/'report.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(p.relative_to(ROOT)),sha256=sha(p),rotation_supervised_TRAIN_parent_sources=total_rotation_parents,
        expected_rotation_supervised_draws_per_epoch=expected_rotation_draws),ensure_ascii=False))


if __name__=='__main__':main()
