"""Audit all preregistered bin TRAIN cases without fitting or source selection."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from scripts.run_bin_TRAIN_v16_r10 import SOURCES,METHODS,SETUP,OUTPUT,TASK


def main():
    parent=json.loads((OUTPUT/'parent_summary.json').read_text())
    assert parent['all_children_terminal'] and len(parent['children'])==8
    lock=json.loads((SETUP/'frozen_inputs.json').read_text())
    verify_inputs(ROOT,lock['files'])
    rows=[]
    for raw in SOURCES:
        for method in METHODS:
            path=OUTPUT/'development_demo_only'/raw/method/'case.json'
            if not path.is_file():
                rows.append(dict(source=raw,method=method,complete=False,infrastructure_pending=True))
                continue
            d=json.loads(path.read_text());a=d.get('actor_metrics',{})
            reference=a.get('continuous_reference_feedforward',{})
            dispatch=a.get('stopped_release_dispatch_v16',{}).get('events',[])
            assert d['task']==TASK and d['source_raw_episode_id']==raw and d['method']==method
            bad=[{'tick':e.get('tick_after'),'velocity':e.get('actual_window_velocity_within_limits'),
                'acceleration':e.get('actual_window_acceleration_within_limits')} for e in dispatch
                if e.get('actual_window_velocity_within_limits') is False or e.get('actual_window_acceleration_within_limits') is False]
            q=dict(source=raw,method=method,case=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)),
                complete=d['complete'],infrastructure_error=d['infrastructure_error'],cleanup_completed=d['cleanup_completed'],
                official_task_success=d['official_task_success'],simulation_seconds=d['simulation_seconds'],counted_ticks=len(dispatch),
                max_acceleration_rad_s2=max((abs(v) for e in dispatch for v in (e.get('backward_window_acceleration') or [])),default=None),
                motion_violations=bad,reference_motion_failed=reference.get('observed_motion_qualification_failed'),
                observable_skills=[dict(phase=e['phase'],time=e['time']) for e in a.get('observable_skill_events',[])],
                selections=[dict(phase=e['phase'],time=e['time']) for e in a.get('candidate_selection_events',[])],
                final_public_diagnostics=a.get('last_diagnostics'),formal_metric=False)
            rows.append(q)
    complete=all(r['complete'] and not r.get('infrastructure_error') and r.get('cleanup_completed') for r in rows)
    motion=complete and all(not r['motion_violations'] and r['reference_motion_failed'] is False for r in rows)
    successes={m:sum(r.get('official_task_success') is True for r in rows if r['method']==m) for m in METHODS}
    passed=complete and motion and all(n>0 for n in successes.values())
    report=dict(schema='r10_bin_fixed_first4_TRAIN_qualification_v16',created_utc=datetime.now(timezone.utc).isoformat(),
        task=TASK,methods=list(METHODS),fixed_sources=list(SOURCES),expected_cases=8,cases=rows,
        complete=complete,all_original_motion_checks_passed=motion,successes=successes,
        closed_loop_task_success_each_geometry_method=all(n>0 for n in successes.values()),
        TRAIN_executor_qualification_passed=passed,formal_VAL_TEST_release=False,
        baseline_native_sanity_checked=False,full_model_source_audit_is_separate=True,
        all_frozen_inputs_unchanged=True,models_retrained=False,TEST_read=False,source_replacement=False)
    atomic_json(OUTPUT/'actual_TRAIN_acceptance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','fixed_sources']},ensure_ascii=False))


if __name__=='__main__':main()
