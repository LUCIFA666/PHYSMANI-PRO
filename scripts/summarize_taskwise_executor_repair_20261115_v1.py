"""Reconcile taskwise executor evidence without changing historical results."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/framework/taskwise_executor_repair_20261115_v1'
FRAME=ROOT/'artifacts/framework'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pointer(p):return dict(path=str(p),sha256=sha(p))
def atomic(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    old_path=FRAME/'executor_diagnostics_20261113_v2/repair_report.json';old=read(old_path)
    checks=[]
    for prior in old['frozen_baseline_audit']:
        actual=sha(prior['path']);checks.append(dict(prior,actual=actual))
        if actual!=prior['expected']:raise RuntimeError('Frozen baseline changed: '+prior['path'])
    cohorts=[]
    batches=[
      ('release_order_comparison','executor_release_order_proposal_20261113_v19/native_validation/batch_receipt.json'),
      ('guided_approach_regression','executor_guided_approach_20261113_v20/native_validation/batch_receipt.json'),
      ('reachable_hover_regression','executor_reachable_approach_20261113_v21/native_validation/batch_receipt.json'),
      ('continuous_IK_regression','executor_continuous_posture_20261113_v22/native_validation/batch_receipt.json'),
      ('high_speed_peg_continuous_IK','executor_release_continuous_ik_20261113_v23/native_validation/batch_receipt.json'),
      ('cup_strict_release_regression','executor_continuous_posture_20261113_v22/release_alignment/batch_receipt.json'),
      ('cup_live_pose_during_gripper_ACK','executor_gripper_goal_refresh_20261113_v24/native_validation/batch_receipt.json')]
    for version in (1,2,3):
        batches.append((f'cup_independent_stability_v{version}',f'taskwise_executor_validation_20261113_v{version}/place_cups_on_rotating_frame/stability_report.json'))
    for name,rel in batches:
        path=FRAME/rel
        if not path.exists():
            if args.final:raise RuntimeError('Required cohort unfinished: '+rel)
            cohorts.append(dict(name=name,complete=False,expected_receipt=str(path)));continue
        d=read(path)
        if not d['complete']:raise RuntimeError('Incomplete cohort receipt')
        rows=[]
        for row in d['rows']:
            r=row['result'];life=row['process_lifecycle']
            if not(life['complete']and life['actual_wait_completed']and not life['unreaped_descendants']):
                raise RuntimeError('Native lifecycle not complete')
            if not(r['NN_forward_calls']==0 and r['optimizer_steps']==0 and not r['formal_VAL_TEST']and r['source_unchanged']):
                raise RuntimeError('Diagnostic provenance changed')
            rows.append(dict(task=r['task'],seed=r['seed'],profile=r['profile'],
                original_native_success=r['original_native_success'],actual_seconds=r['actual_seconds'],
                stop_reason=r['stop_reason'],ever_held=r['ever_held'],rejected_pose_chunks=r['rejected_pose_chunks'],
                reference_limits_passed=r['issued_reference_audit']['all_reference_differences_within_limits'],
                native_bounds_passed=r['issued_reference_audit']['all_position_commands_within_native_bounds'],
                prohibited_robot_table_contact_frames=r['hardware_contact_audit']['prohibited_robot_table_contact_frames'],
                actual_owner_wait_and_reap_completed=True))
        cohorts.append(dict(name=name,complete=True,receipt=pointer(path),rows=rows,
            successes=sum(r['original_native_success']for r in rows),total=len(rows)))
    cup=read(FRAME/'taskwise_executor_validation_20261113_v3/place_cups_on_rotating_frame/stability_report.json')
    peg=next(c for c in cohorts if c['name']=='high_speed_peg_continuous_IK')
    tasks=[]
    for task in old['task_types_with_any_qualified_oracle_success']:
        state=dict(task=task,phase='executor_repair',complete_stability_small_and_full_cycle_accepted=False,
            any_qualified_TRAIN_oracle_success=True,new_formal_VAL_TEST_started=False,new_optimizer_steps=0)
        if task==cup['task']:
            state.update(latest_independent_stability=dict(successes=cup['successes'],total=cup['total'],
                passed=cup['stability_passed'],all_mechanical_checks_passed=cup['all_mechanical_checks_passed'],
                report=pointer(FRAME/'taskwise_executor_validation_20261113_v3/place_cups_on_rotating_frame/stability_report.json')),
                current_repair='same-stage pose refresh during committed gripper event')
        elif task=='insert_onto_rotating_peg_high_speed':
            state.update(latest_development_regression=dict(successes=peg['successes'],total=peg['total'],
                independent_stability_started=False,receipt=peg['receipt']))
        tasks.append(state)
    units=FRAME/'executor_gripper_goal_refresh_20261113_v24/unit_report.json'
    report=dict(updated_utc=datetime.now(timezone.utc).isoformat(),checkpoint_complete=args.final,
        user_objective_complete=False,all_tasks_fixed=False,task_time_limit_seconds=60,
        previously_completed_cube_types=2,new_other_task_types_fully_accepted=0,
        cube_TEST=dict(ordinary=dict(successes=39,total=40),high_speed=dict(successes=27,total=40),
            report=pointer(FRAME/'cube_full_reuse_20261112_v1/report.json'),
            VAL_uses_generated_scenes_per_source=True,TEST_uses_all_80_archived_resets=True,
            paper_benchmark_comparable=False),
        taskwise_workflow='Each task: repair, independent stability, small TRAIN/VAL/TEST, full TRAIN/VAL/TEST; Native at most one.',
        other_tasks=tasks,previous_report=pointer(old_path),previous_Native_case_count=old['new_native_case_count'],
        completed_Native_cases_since_previous_report=sum(c.get('total',0)for c in cohorts if c['complete']),
        cumulative_new_Native_diagnostic_cases=old['new_native_case_count']+sum(c.get('total',0)for c in cohorts if c['complete']),
        cohorts=cohorts,unit_tests=dict(**read(units),artifact=pointer(units)),
        frozen_baseline_audit=checks,NN_forward_calls_this_repair=0,optimizer_steps_this_repair=0,
        new_formal_VAL_TEST_started=False,model_success_rate_claimed=False,
        default_runtime_changed=False,model_weights_changed=False,
        original_Native_physics_and_Task_success_preserved=True,
        legacy_pose8_compatibility_preserved=True,execution_protocol_envelope_version=2,
        control_strategy_authorization='User approved open fingers before release; unified report with strategy fields.',
        active_service=None if args.final else'executor_gripper_goal_refresh_20261113_v24',
        pending_user_decision=None)
    target=OUT/('report.json'if args.final else'progress_checkpoint.json')
    if args.final:
        with target.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2)
    else:atomic(target,report)
    index=dict(state='taskwise_executor_repair_checkpoint'if args.final else'taskwise_executor_repair_running',
        current_checkpoint=pointer(target),active_service=report['active_service'],remaining_user_decision=None,
        all_tasks_fixed=False,default_runtime_changed=False,model_weights_changed=False,
        new_VAL_TEST_started=False,protocol_version=2,legacy_pose8_compatibility_preserved=True,
        unit_test_report=pointer(units))
    for name in ['latest_executor_continuation_20261113.json','latest_executor_contract_repair_20261112.json','latest_executor_diagnostics_20261113_v2.json']:
        path=ROOT/'runtime/framework'/name;backup=path.with_name(path.stem+'_before_taskwise_20261115_v1.json')
        if path.exists()and not backup.exists():shutil.copy2(path,backup)
        atomic(path,index)
    print(json.dumps(dict(report=str(target),completed_diagnostic_cases=report['cumulative_new_Native_diagnostic_cases'],unit_tests=read(units)['tests_run'],checkpoint_complete=args.final),ensure_ascii=False))

if __name__=='__main__':main()
