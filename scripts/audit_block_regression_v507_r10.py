"""Separate completed R09 results from subsequent TRAIN adapter regressions."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.three_method_release16 import measured_motion


def summarize(path):
    d = json.loads(path.read_text())
    if d['split'] != 'train' or d.get('formal_test_used'):
        raise ValueError('adapter diagnosis uses TRAIN only')
    steps = d['steps']
    return dict(task=d['task'], raw=d['raw_episode_id'], method=d['method'],
        native_success=d['official_task_success'], motion_pass=measured_motion(d),
        checkpoint=(d.get('shared_context_receipt') or {}).get('checkpoint_sha256'),
        captured=d.get('captured'),
        stages=dict(Counter(s.get('stage') for s in steps)),
        commands=[dict(command=key[0], reason=key[1], count=n) for key, n in
                  Counter((s.get('decision', {}).get('command'),
                           s.get('decision', {}).get('reason')) for s in steps).items()],
        first_public_robot_pose=steps[0]['robot_pose'],
        source=dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path)))


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    historical_path = ROOT/'artifacts/framework/aligned_r09/final_report.json'
    historical = json.loads(historical_path.read_text())
    if historical['complete'] is not True or historical['protocol_acceptance_passed'] is not True:
        raise ValueError('completed historical evaluation required')
    cohorts = {}
    for name in ('joint16_block_support_v318', 'block_remaining_sources_v363',
                 'block_fresh_tail_v451', 'block_receding_bridge_v491'):
        rows = [summarize(p) for p in sorted((base/name).rglob('case.json'))]
        cohorts[name] = dict(cases=rows, completed=len(rows),
                            native_success=sum(r['native_success'] for r in rows),
                            motion_pass=sum(r['motion_pass'] is True for r in rows))
    def first(name):
        return next(r for r in cohorts[name]['cases'] if r['task'] == 'pick_moving_target_on_the_table'
                    and r['raw'] == 'episode0' and r['method'] == 'context_geometry')
    before, after = first('block_fresh_tail_v451'), first('block_receding_bridge_v491')
    regression = dict(before=before, after=after,
        same_shared_checkpoint=before['checkpoint'] == after['checkpoint'],
        same_initial_public_robot_pose=before['first_public_robot_pose'] == after['first_public_robot_pose'],
        success_lost=before['native_success'] and not after['native_success'],
        latest_never_entered_local_capture='local' not in after['stages'],
        checkpoint_replacement_alone_does_not_explain_this_regression=True,
        exact_causal_change_not_isolated=True)
    if not (regression['same_shared_checkpoint'] and regression['success_lost']):
        raise ValueError('claimed shared-model TRAIN regression must be supported')
    atomic_json(base/'block_regression_v507/report.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        historical_completed_evaluation=dict(
            source=str(historical_path.relative_to(ROOT)), sha256=file_sha(historical_path),
            complete=True, protocol_acceptance_passed=True,
            authorized_three_method_results={m: historical['variants'][m]
                for m in ('cv_geometry', 'context_geometry', 'physmani')},
            historical_results_not_revoked=True,
            historical_checkpoint_is_not_the_shared_joint16_checkpoint=True),
        TRAIN_cohorts=cohorts, same_shared_model_regression=regression,
        disposition=dict(latest491_not_a_functionally_qualified_replacement=True,
            preserve_historical_sources_models_and_results=True,
            no_automatic_rerun_of_completed_TEST=True,
            diagnosis_and_new_changes_USE_TRAIN_only=True,
            historical_scores_not_relabelled_as_current_joint16_scores=True),
        TEST_case_or_manifest_read=False, new_native_or_GPU_started=False,
        Value_NGFF_FASTER_restored=False, source_sha256=file_sha(__file__)))
    print(json.dumps(dict(complete=True, success_lost=regression['success_lost'],
        same_shared_checkpoint=regression['same_shared_checkpoint'],
        cohorts={k:{a:b for a,b in v.items() if a!='cases'} for k,v in cohorts.items()})))


if __name__ == '__main__':
    main()
