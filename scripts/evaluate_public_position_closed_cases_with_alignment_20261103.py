"""Original after-exit development report plus first50ms close alignment A.

A measures alignment to the selected local point at the same GT object pose;
it is not unique-correct-grasp quality or free-motion forecast accuracy.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_public_position_closed_cases_20261103 as base


def alignment_from_records(initial,ticks,decisions):
    previous=initial;edge=None
    for state in ticks:
        if previous['requested_opening']>=.5 and state['requested_opening']<.5:
            edge=(previous,state);break
        previous=state
    if edge is None:return dict(available=False,reason='no_recorded_open_to_close_edge',first_close_observation_time=None)
    before,after=edge
    matched=[r for r in decisions if abs(r['time']-before['time'])<=1e-8]
    header=dict(first_close_dispatch_time=before['time'],first_close_observation_time=after['time'],
        Native_held_at_first_observation=after['held'],first_observation_not_contact_time_proof=True)
    if len(matched)!=1:return dict(**header,available=False,reason='no_unique_same_time_dispatch')
    row=matched[0];decision=row['decision'];action=decision.get('action')
    if not isinstance(action,dict):return dict(**header,available=False,reason='no_selected_action_at_close')
    candidates=decision.get('candidate_pool',{}).get('candidates',[])
    points=[c for c in candidates if c.get('id')==action.get('candidate_id')]
    if len(points)!=1:return dict(**header,available=False,reason='no_unique_selected_local_candidate')
    candidate=points[0];point=np.asarray(candidate.get('object_tcp_position'),float)
    if point.shape!=(3,)or not np.isfinite(point).all():return dict(**header,available=False,reason='local_candidate_point_missing_or_invalid')
    errors=[]
    for name,state in (('legacy_same_time_dispatch',before),('first_completed50ms_close_tick',after)):
        R=Rotation.from_quat(state['object_quaternion']).as_matrix()
        target=np.asarray(state['object_position'])+R@point;difference=np.asarray(state['tcp_position'])-target
        errors.append(dict(definition=name,time=state['time'],TCP_minus_same_time_transformed_selected_local_point_m=difference.tolist(),distance_m=float(np.linalg.norm(difference))))
    return dict(**header,available=True,candidate_id=candidate['id'],object_relative_TCP_point_m=point.tolist(),
        fixed_deadline=candidate.get('deadline'),primary_distance_m=errors[1]['distance_m'],comparisons=errors,
        error_is_control_alignment_to_selected_point_only=True,GT_used_only_posthoc=True)


def first_close_alignment(case):
    case=Path(case).resolve(strict=True);base._non_test(case)
    owner=base.read(case.with_name(case.name+'.owner.actual_exit.json'));worker=base.read(case/'worker_exit.json');result=base.read(case/'result.json')
    if owner.get('actual_process_returncode')!=0 or owner.get('actual_wait_completed')is not True or worker.get('native_closed')is not True or result.get('complete')is not True:
        raise ValueError('Actual closed0 case required before alignment GT')
    logs=list(case.glob('public_candidate_position_v*_decisions.jsonl'))
    if len(logs)!=1:raise ValueError('Unique actual public decision log required')
    return alignment_from_records(base.read(case/'initial_snapshot.json')['state'],list(base.lines(case/'native_ticks.jsonl')),base.lines(logs[0]))


def evaluate_with_alignment(plan,output):
    report=base.evaluate_plan(plan,output)
    rows=[]
    for item in report['cases']:
        A=first_close_alignment(item['case']);m=item['original_native_task_and_hold']
        rows.append(dict(id=item['id'],expected_negative_no_proxy=item['expected_negative_no_proxy'],
            actual_goal_any=m['native_success_any'],strict_current_hold_final=m['strict_current_joint_condition']['final'],
            maximum_lift_m=m['p1_hold_metric']['maximum_lift_m'],drops=m['p1_hold_metric']['drops'],
            public=item['public'],first_close_alignment_A=A,cycle_wall=item['cycle_wall']))
    available=[x['first_close_alignment_A']['primary_distance_m']for x in rows if x['first_close_alignment_A']['available']]
    values=np.asarray(available,float)
    summary=dict(schema='closed_public_position_original_judge_and_firstclose_alignment_v1',
        original_report=base.pointer(Path(output)/'report.json'),cases=rows,aggregate=report['aggregate'],
        firstclose_A=dict(entire_declared_cases=len(rows),available=len(values),missing=len(rows)-len(values),
            mean_m=None if not len(values)else float(values.mean()),p95_m=None if not len(values)else float(np.quantile(values,.95)),
            max_m=None if not len(values)else float(values.max()),over3cm_count=int((values>.03).sum()),over5cm_count=int((values>.05).sum()),
            missing_or_failed_attempts_not_removed_from_declared_cohort=True,not_network_unique_correct_grasp_error=True),
        pooled_cycle_wall=report['pooled_all_cycle_wall'],GT_only_posthoc=True,Native_started=False,fit_started=False,TEST_read=False,
        helper=base.pointer(__file__))
    with(Path(output)/'summary_with_alignment.json').open('x')as stream:json.dump(summary,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(summary=base.pointer(Path(output)/'summary_with_alignment.json'),aggregate=summary['aggregate'],firstclose_A=summary['firstclose_A'])))
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    evaluate_with_alignment(a.plan,a.output);return 0
if __name__=='__main__':raise SystemExit(main())
