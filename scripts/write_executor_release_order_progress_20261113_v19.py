#!/usr/bin/env python3
import datetime,json
from pathlib import Path
from summarize_executor_continuation_20261113_v1 import ROOT,read,pointer

OUT=ROOT/'artifacts/framework/executor_release_order_proposal_20261113_v19'

def replace_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def main():
    rows=[]
    for name in read(OUT/'predeclared_plan.json')['cases']:
        case=read(name);base=OUT/'native_validation'/Path(name).stem
        result=read(base/'worker/result.json')if(base/'worker/result.json').exists()else None
        life=read(base/'actual_exit.json')if(base/'actual_exit.json').exists()else None
        status='not_started'
        if base.exists():status='running'
        if life is not None:status='finished'if life['complete']else'process_failed'
        rows.append(dict(case=name,task=case['task'],seed=case['seed'],control_strategy=case['gripper_release_order'],
            status=status,success=None if result is None else result['original_native_success'],
            physical_seconds=None if result is None else result['actual_seconds']))
    value=dict(state='release_control_strategy_comparison_running',remaining_user_decision=None,
        user_decision=pointer(OUT/'user_control_strategy_decision.json'),
        updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total=len(rows),finished=sum(r['status']=='finished'for r in rows),rows=rows,
        Native_at_most=1,NN_forward_calls=0,optimizer_steps=0,formal_VAL_TEST=False,
        default_runtime_changed=False,reporting_policy='unified_executor_report_with_control_strategy_field')
    replace_json(OUT/'progress_checkpoint.json',value)
    index=dict(state=value['state'],remaining_user_decision=None,
        previous_report=pointer(ROOT/'artifacts/framework/executor_diagnostics_20261113_v2/repair_report.json'),
        current_checkpoint=pointer(OUT/'progress_checkpoint.json'),
        active_service='executor_release_order_proposal_20261113_v19',default_runtime_changed=False,
        all_tasks_fixed=False,model_weights_changed=False,new_VAL_TEST_started=False,
        protocol_version=2,legacy_pose8_compatibility_preserved=True,
        unit_test_report=pointer(OUT/'unit_test_report.json'))
    for name in ['latest_executor_continuation_20261113.json','latest_executor_contract_repair_20261112.json','latest_executor_diagnostics_20261113_v2.json']:
        replace_json(ROOT/'runtime/framework'/name,index)
    print(json.dumps(dict(total=value['total'],finished=value['finished'],
        running=[r['case']for r in rows if r['status']=='running']),ensure_ascii=False))

if __name__=='__main__':main()
