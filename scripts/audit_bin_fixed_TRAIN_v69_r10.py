"""Preserve all fixed4x2 TRAIN results, including timeout policy failures."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
B=ROOT/'artifacts/framework/multitask_r10';OUT=B/'bin_fixed_four_sources_v69/report.json'


def main():
    if OUT.exists():raise RuntimeError('preserve immutable acceptance report')
    lock=B/'task_executor_smoke_v69_setup/frozen_inputs.json';files=json.loads(lock.read_text())['files'];verify_inputs(ROOT,files)
    rows=[]
    for raw in ('episode105','episode106','episode119','episode123'):
        for method in ('cv_geometry','context_geometry'):
            p=B/'task_executor_smoke_v69/development_demo_only/put_rubbish_in_moving_bin'/raw/method/'case.json'
            d=json.loads(p.read_text());a=d['actor_metrics'];m=a['continuous_reference_feedforward']
            assert d['complete'] and d['infrastructure_error'] is None and d['cleanup_completed']
            assert d['split']=='train' and not d['formal_test_used'] and d['demo_only']
            assert d['counted_clock']['dt']==.05 and d['counted_clock']['ticks']<=200
            events=m['observed_events'];maxa=max(abs(x) for e in events for x in (e['backward_window_acceleration'] or []))
            maxv=max(abs(x) for e in events for x in (e['backward_window_velocity'] or []))
            rows.append(dict(fixed_source=raw,method=method,case=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p)),
                native_task_success=d['official_task_success'],counted_clock=d['counted_clock'],complete=True,infrastructure_error=None,cleanup_completed=True,
                motion_qualified=not m['observed_motion_qualification_failed'],maximum_actual_backward_window_acceleration_rad_s2=maxa,
                maximum_actual_backward_window_speed_rad_s=maxv,observed_phase_indices=sorted({e['phase_index'] for e in a['executor_observed_state_trace']}),
                native_end_reason=d['native_end_reason'],close_gate=a['close_completion_gate_v68']))
    summaries={method:dict(native_successes=sum(r['native_task_success'] for r in rows if r['method']==method),
        fixed_TRAIN_denominator=4,completed_cases=4,infrastructure_errors=0,motion_passes=sum(r['motion_qualified'] for r in rows if r['method']==method))
        for method in ('cv_geometry','context_geometry')}
    report=dict(schema='r10_ordinary_bin_all_fixed4_TRAIN_runtime_qualification_v69',created_utc=datetime.now(timezone.utc).isoformat(),
        task='put_rubbish_in_moving_bin',rows=rows,summary=summaries,
        executor_runtime_smoke_passed=all(r['motion_qualified'] for r in rows),benchmark_policy_success_all_sources=all(r['native_task_success'] for r in rows),
        policy_failures_remain_in_denominator=True,formal_VAL_TEST_success_rate=False,all16_executors_qualified=False,
        joint16_training_started=False,formal_release=False,frozen_inputs_rehashed=len(files),
        frozen_lock=dict(path=str(lock.relative_to(ROOT)),sha256=file_sha(lock)),source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),
        next='Continue remaining task executor repair, joint16 training, then fresh complete smoke and unified VAL/TEST.')
    atomic_json(OUT,report);print(json.dumps(dict(summary=summaries,executor_runtime_smoke_passed=report['executor_runtime_smoke_passed'],all_sources_succeeded=report['benchmark_policy_success_all_sources'])))

if __name__=='__main__':main()
