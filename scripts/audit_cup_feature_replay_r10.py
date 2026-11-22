"""Post-run provenance and four-parent summary; no predictions are rerun."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experimental_cup_features_r10 import features_from_frozen_cad
from scripts.replay_cup_feature_prototype_r10 import verify,statistics

BASE=ROOT/'artifacts/framework/multitask_r10'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def run():
    plan_path=BASE/'TRAIN_cup_feature_replay_plan_v1.json';plan=verify(plan_path)
    folder=BASE/'cup_feature_replay_v1';report_path=folder/'report.json';report=read(report_path)
    receipt_path=ROOT/report['predictions_complete']['path'];receipt=read(receipt_path)
    assert sha(receipt_path)==report['predictions_complete']['sha256'] and receipt['actual_body_frames']==336
    assert datetime.fromisoformat(report['metrics_started_utc'])>=datetime.fromisoformat(receipt['completed_utc'])
    for row in receipt['cases']:
        assert sha(ROOT/row['predictions']['path'])==row['predictions']['sha256']
    first={}
    for case in plan['cases']:first.setdefault(case['parent_source_group'],case)
    assert len(first)==4
    chosen={case['case_id'] for case in first.values()}
    rows=[row for row in report['per_frame'] if row['case_id'] in chosen]
    assert len(rows)==192 and len({(r['parent_source_group'],r['body_id'],r['frame']) for r in rows})==192
    good=[r for r in rows if r['pose_present']]
    independent=dict(body_frames=192,independent_TRAIN_parents=4,pose_present=len(good),
        ambiguous=sum(r['ambiguous'] for r in rows),full_SO3_error_degrees=statistics([r['SO3_error_degrees'] for r in good]),
        position_error_m=statistics([r['position_error_m'] for r in good]),
        SO3_below_10_degrees=sum(r['SO3_error_degrees']<10 for r in good),
        position_below_5mm=sum(r['position_error_m']<.005 for r in good),
        prediction_seconds=statistics([r['prediction_seconds'] for r in rows]),
        selection='first fixed plan-order cache per parent, independent of predictions/outcomes',
        selected_case_ids=sorted(chosen))
    canonical=read(ROOT/plan['canonical_CAD']['path']);bias={}
    # These are post-prediction metrics only, never estimates or adjustments.
    for case in first.values():
        asset=read(ROOT/case['geometry']['path']);native=read(ROOT/case['metric_report']['path'])
        predictions=[json.loads(line) for line in (folder/(case['case_id']+'.predictions.jsonl')).read_text().splitlines()]
        for role in ('cup0','cup1','cup2'):
            model=features_from_frozen_cad(canonical,asset,role);axial=[];transverse=[]
            for row in predictions:
                if row['body_id']!=role or row['prediction']['pose_xyzw'] is None:continue
                metric=native['frames'][row['frame']]['metric_only'];j=metric['body_ids'].index(role)
                gt=metric['native_current_ground_truth'];target=np.asarray(gt['positions'][j]);R=np.asarray(gt['rotations'][j])
                error=np.asarray(row['prediction']['pose_xyzw'][:3])-target
                axis=R@model.canonical_to_role.rotation@np.array([0.,0.,1.]);signed=float(error@axis)
                axial.append(signed);transverse.append(float(np.linalg.norm(error-signed*axis)))
            bias[case['case_id']+'__'+role]=dict(signed_axial_bias_median_m=float(np.median(axial)),
                transverse_error_m=statistics(transverse),posthoc_GT_used_for_diagnosis_only=True,
                bias_correction_applied=False)
    inputs={str(plan_path.relative_to(ROOT)):sha(plan_path),str(report_path.relative_to(ROOT)):sha(report_path),
        str(receipt_path.relative_to(ROOT)):sha(receipt_path),str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    result=dict(version='r10_fixed_cup_feature_postrun_input_and_independent_parent_audit_v1',
        complete=True,original_replay_report_unchanged=True,all_plan_input_hashes_verified=True,
        all_prediction_hashes_verified=True,predictions_finished_before_GT_metrics=True,
        original_all_cache_summary=report['overall'],four_independent_parent_summary=independent,
        posthoc_position_bias_by_sequence=bias,
        inputs_sha256=inputs,core_modified=False,native_started=False,parameters_changed=False,
        predictions_rerun=False,policy_authorized=False,
        interpretation=['All192 distinct parent/cup/frame cases have SO3 error below10deg.',
            'Four parent trajectories remain a small correlated TRAIN diagnostic, not unseen-task validation.',
            'Remaining position error includes radial/occlusion bias, especially hanging remove-cup views; no GT bias correction was applied.',
            'Three cups need about3 times the reported per-cup wall time; a20Hz real-time deployment is not demonstrated.'])
    path=folder/'independent_parent_and_input_audit.json';assert not path.exists()
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (folder/'independent_parent_and_input_audit.txt').write_text(
        'All frozen inputs / completed prediction hashes verified; no replay or parameter change.\n'+
        'First fixed cache per parent, selected independently of outcomes:\n'+json.dumps(independent,ensure_ascii=False,indent=2)+'\n'+
        'No policy authorization; no20Hz real-time claim; no GT position bias correction.\n')
    print(json.dumps(dict(complete=True,independent=independent,audit_sha256=sha(path))))


if __name__=='__main__':run()
