"""Audit a completed v8 pair, including actual motion and public collision witnesses."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
B=ROOT/'artifacts/framework/multitask_r10/geometry_spherical_reference_v8'
S=ROOT/'artifacts/framework/multitask_r10/geometry_spherical_reference_v8_setup'
def ptr(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
def main():
 target=B/'actual_smoke_report.json'
 if target.exists():raise RuntimeError('Preserve audit')
 lock=json.loads((S/'frozen_inputs.json').read_text());verify_inputs(ROOT,lock['files']);parent=json.loads((B/'parent_summary.json').read_text());assert parent['all_children_terminal'] and parent['original_frozen_inputs_verified_after'];rows={}
 for child in parent['children']:
  p=ROOT/child['case'];c=json.loads(p.read_text());m=c['actor_metrics'];d=m['continuous_direct_counted_setpoint_v5'];assert child['returncode']==0 and c['complete'] and c['cleanup_completed'] and c['infrastructure_error'] is None
  ev=m['public_collision_query_diagnostics_v6']['events'];counts=Counter(w['obstacle']['label'] for e in ev for w in e['held_world'])
  rows[child['method']]=dict(case=ptr(p),policy_outcome=c['official_task_success'],actual_motion_qualification=d['actual_observed_window_limits_passed'],maximum_actual_window_acceleration_rad_s2=max(float(np.max(np.abs(e['backward_window_acceleration']))) for e in d['events'] if e.get('backward_window_acceleration') is not None),observable_skill_events=m['observable_skill_events'],first_collision=ev[:1],colliding_obstacles=dict(counts),counted_clock=c['counted_clock'])
 report=dict(schema='r10_TRAIN_spherical_reference_v8_actual_audit',created_utc=datetime.now(timezone.utc).isoformat(),methods=rows,parent_summary=ptr(B/'parent_summary.json'),frozen_inputs=ptr(S/'frozen_inputs.json'),all256_inputs_preserved=True,executor_fully_solved=False,formal_metric=False,formal_VAL_or_TEST_started=False,Value_or_NGFF_or_FASTER_started=False)
 atomic_json(target,report);p=ROOT/'runtime/framework/handoff_runtime_latest.json';s=json.loads(p.read_text());s.update(status='native_pair_v8_terminal_audited',updated_utc=report['created_utc'],actual_report=ptr(target),parent_summary=report['parent_summary'],native_processes_all_terminal=True,native_policy_result_available=True,executor_fully_solved=False,next_required_steps=['Use fresh public RGBD attachment updates and scene-constrained sphere localization in a new version; retain all physical collision checks.']);atomic_json(p,s,overwrite=True);print(json.dumps(rows))
if __name__=='__main__':main()
