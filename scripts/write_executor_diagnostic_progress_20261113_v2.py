#!/usr/bin/env python3
"""Publish a truthful checkpoint without claiming unfinished Native checks passed."""
import datetime,json
from pathlib import Path
from summarize_executor_diagnostics_20261113_v2 import ROOT,OUT,GROUPS,read,pointer

def replace_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)

def main():
    rows=[]
    for label,namespace,folder,plan in GROUPS:
        base=ROOT/'artifacts/framework'/namespace
        for case in read(base/plan)['cases']:
            output=base/folder/Path(case).stem
            path=output/'worker/result.json';result=read(path)if path.exists()else None
            lifecycle=read(output/'actual_exit.json')if(output/'actual_exit.json').exists()else None
            qualified=(result is not None and lifecycle is not None and lifecycle['complete']
                and lifecycle['actual_wait_completed']and not lifecycle['unreaped_descendants'])
            status='finished'if qualified else'failed_initialization'if lifecycle is not None else'running'if output.exists()else'not_started'
            rows.append(dict(group=label,case=case,status=status,
                success=None if result is None else result['original_native_success'],
                stop_reason=None if result is None else result['stop_reason']))
    value=dict(state='executor_repairs_in_progress',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        case_count=len(rows),finished=sum(r['status']=='finished'for r in rows),
        running=[r['case']for r in rows if r['status']=='running'],
        not_started=sum(r['status']=='not_started'for r in rows),rows=rows,
        protocol_version=2,legacy_pose8_preserved=True,unit_tests=read(OUT/'unit_test_report.json')['tests_run'],
        model_weights_changed=False,formal_VAL_TEST=False,default_runtime_changed=False,
        initial_launcher_failure_retained=True,cup_common_profile_successes=4,cup_common_profile_total=4,
        all_tasks_fixed=False,cube_existing_TEST={'normal':[39,40],'high_speed':[27,40]})
    path=OUT/'progress_checkpoint.json';replace_json(path,value)
    replace_json(ROOT/'runtime/framework/latest_executor_diagnostics_20261113_v2.json',
        dict(state=value['state'],checkpoint=pointer(path)))
    print(json.dumps({key:value[key]for key in('state','case_count','finished','running','not_started')},ensure_ascii=False))

if __name__=='__main__':main()
