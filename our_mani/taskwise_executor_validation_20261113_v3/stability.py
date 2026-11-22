"""Run an already-qualified task profile while other task repairs progress."""
import json,time
from pathlib import Path
from .common import ROOT,OUT,PLAN,read,write,append,pointer,sha,sources,verify,stopped
from .owner import run_case

def status(**value):
    path=OUT/'status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def main():
    plan=read(PLAN);frozen=sources()
    frozen.update({name:sha(name)for name in plan['cases']})
    write(OUT/'request.json',dict(plan=pointer(PLAN),source_sha256=frozen,
        user_instruction=pointer(OUT/'user_taskwise_instruction.json')))
    dependency=Path(plan['Native_slot_dependency'])
    status(task=plan['task'],phase='waiting_for_Native_slot',dependency=str(dependency),NN_forward_calls=0)
    while True:
        stopped()
        if dependency.exists():
            try:ready=read(dependency)
            except json.JSONDecodeError:ready=None
            if ready is not None:
                if not ready['complete']or not all(row['process_lifecycle']['complete']and row['process_lifecycle']['actual_wait_completed']and not row['process_lifecycle']['unreaped_descendants']for row in ready['rows']):
                    raise RuntimeError('Previous Native queue did not release a qualified slot')
                break
        time.sleep(.25)
    write(OUT/'Native_slot_dependency_receipt.json',dict(dependency=pointer(dependency),previous_owners_waited_and_reaped=True))
    rows=[]
    for name in plan['cases']:
        stopped();verify(frozen);case=Path(name)
        status(task=plan['task'],phase='stability',finished=len(rows),total=len(plan['cases']),active_case=name)
        print('START',case.stem,flush=True)
        result,life=run_case(case,OUT/'stability'/case.stem,mode='online_teacher')
        row=dict(case=pointer(case),result=result,process_lifecycle=life);rows.append(row)
        append(OUT/'stability/batch_progress.jsonl',row)
        print('DONE',case.stem,None if result is None else dict(success=result['original_native_success'],seconds=result['actual_seconds'],stop=result['stop_reason']),flush=True)
        if not life['complete']:raise RuntimeError('Incomplete Native lifecycle during task stability')
    verify(frozen)
    successes=sum(r['result']['original_native_success']for r in rows)
    mechanical=all(r['result']['issued_reference_audit']['all_reference_differences_within_limits']
        and r['result']['issued_reference_audit']['all_position_commands_within_native_bounds']
        and r['result']['hardware_contact_audit']['prohibited_robot_table_contact_frames']==0
        and r['result']['stop_reason']!='hardware_model_diverged'for r in rows)
    passed=bool(successes>=plan['qualification']['minimum_original_native_successes']and mechanical)
    report=dict(complete=True,task=plan['task'],phase='stability',successes=successes,total=len(rows),
        stability_passed=passed,all_mechanical_checks_passed=mechanical,rows=rows,
        next_phase='small_train'if passed else'executor_repair',NN_forward_calls=0,optimizer_steps=0,
        formal_VAL_TEST=False,source_unchanged=True,model_success_rate_claimed=False,
        full_recorded_source_counts=plan['full_recorded_source_counts'])
    write(OUT/'stability_report.json',report)
    status(task=plan['task'],phase='stability_complete',successes=successes,total=len(rows),stability_passed=passed,next_phase=report['next_phase'])
    print('STABILITY_REPORT',dict(task=plan['task'],successes=successes,total=len(rows),passed=passed,next_phase=report['next_phase']),flush=True)

if __name__=='__main__':main()
