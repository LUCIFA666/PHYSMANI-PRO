"""Read-only audit of the completed v6 collision witness pair."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
B=ROOT/'artifacts/framework/multitask_r10/geometry_public_collision_diagnostic_v6'
S=ROOT/'artifacts/framework/multitask_r10/geometry_public_collision_diagnostic_v6_setup'
def ptr(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
def main():
 target=B/'actual_smoke_report.json'
 if target.exists():raise RuntimeError('Preserve audit')
 lock=json.loads((S/'frozen_inputs.json').read_text());verify_inputs(ROOT,lock['files'])
 parent=json.loads((B/'parent_summary.json').read_text());assert parent['all_children_terminal'] and parent['original_frozen_inputs_verified_after']
 rows={}
 for child in parent['children']:
  p=ROOT/child['case'];c=json.loads(p.read_text());m=c['actor_metrics'];d=m['continuous_direct_counted_setpoint_v5']
  assert child['returncode']==0 and not child['timed_out'] and c['complete'] and c['cleanup_completed'] and c['infrastructure_error'] is None
  ev=m['public_collision_query_diagnostics_v6']['events'];counts=Counter((w['held']['label'],w['obstacle']['label']) for e in ev for w in e['held_world'])
  rows[child['method']]=dict(case=ptr(p),policy_outcome=c['official_task_success'],actual_motion_qualification=d['actual_observed_window_limits_passed'],observable_skill_events=m['observable_skill_events'],first_collision=ev[0] if ev else None,collision_pair_counts=[dict(held=k[0],obstacle=k[1],count=v) for k,v in counts.items()],diagnostic_record_count=len(ev),counted_clock=c['counted_clock'],all_children_terminal=True)
 report=dict(schema='r10_public_collision_v6_actual_report',created_utc=datetime.now(timezone.utc).isoformat(),methods=rows,parent_summary=ptr(B/'parent_summary.json'),frozen_inputs=ptr(S/'frozen_inputs.json'),all232_frozen_inputs_verified_after=True,executor_fully_solved=False,formal_metric=False,Value_or_NGFF_or_FASTER_started=False,interpretation='All recorded lift collisions are held ball vs static geometry, not arm collision. CV first witness has circumsphere/AABB separation; Context first witness still intersects tabletop under a sphere enclosure. No collision exemption is justified.')
 atomic_json(target,report)
 p=ROOT/'runtime/framework/handoff_runtime_latest.json';status=json.loads(p.read_text())
 for row in status.get('native_process_ids',[]):
  try:row['same_process_live']=int(Path(f"/proc/{row['pid']}/stat").read_text().rsplit(')',1)[1].split()[19])==row['start_ticks']
  except OSError:row['same_process_live']=False
 assert not any(x['same_process_live'] for x in status['native_process_ids'])
 status.update(status='native_pair_v6_terminal_audited',updated_utc=report['created_utc'],actual_report=ptr(target),parent_summary=report['parent_summary'],native_processes_all_terminal=True,native_policy_result_available=True,prepared_entrypoint=None,next_required_steps=['Implement independently bound conservative spherical enclosure plus explicit existing TRAIN-CAD sphere RGBD registration; preserve all v6/previous sources and gates.'])
 atomic_json(p,status,overwrite=True);print(json.dumps({k:dict(outcome=v['policy_outcome'],motion=v['actual_motion_qualification'],pairs=v['collision_pair_counts']) for k,v in rows.items()}))
if __name__=='__main__':main()
