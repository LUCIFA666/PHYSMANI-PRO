"""Read-only causal/label/budget audit of completed candidate collection records."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from our_mani.benchmark_protocol import verify_lock,file_sha256
from our_mani.risk_v2.selector import attempt_controller_hash

def main():
    base=ROOT/'artifacts/framework/aligned_r06'
    config=json.loads((ROOT/'config/framework/aligned_protocol_r06.json').read_text())
    verify_lock(ROOT,json.loads((base/'collection_inputs.json').read_text()))
    problems=[];counts={};success_without_positive=[]
    for predictor in ('cv','context'):
        expected_controller=attempt_controller_hash(ROOT,config,predictor)
        for split in ('train','validation'):
            episodes=[];attempts=[]
            for p in sorted(base.glob(f'collection_{predictor}_{split}_*/pick*__episode*__*.json')):
                r=json.loads(p.read_text());episodes.append(r)
                if r['formal_test_used'] or r['split']!=split or r['attempt_controller_sha256']!=expected_controller:
                    problems.append(dict(file=str(p.relative_to(ROOT)),error='split/controller'))
                positive=0
                for a in r['attempts']:
                    attempts.append(a);positive+=bool(a['official_task_success'])
                    feature=np.asarray(a['features'])
                    if feature.shape!=((87,) if predictor=='context' else (71,)) or not np.isfinite(feature).all():
                        problems.append(dict(file=p.name,error='feature_shape_or_finite'))
                    if not a['complete'] or not a['selected_at']<=a['outcome_observed_until']+1e-8:
                        problems.append(dict(file=p.name,error='censored_label_or_clock'))
                    if abs(a['remaining_episode_budget_seconds']-(10.-a['selected_at']))>1e-6:
                        problems.append(dict(file=p.name,error='remaining_budget'))
                    if abs(a['capture_transport_budget_after_arrival']-(10.-a['arrival_time']))>1e-6:
                        problems.append(dict(file=p.name,error='arrival_budget'))
                    if predictor=='context' and not np.allclose(feature[-16:].reshape(2,8).sum(1),1.,atol=1e-5):
                        problems.append(dict(file=p.name,error='causal_SimNorm_context'))
                if positive>1 or positive>int(r['official_task_success']):problems.append(dict(file=p.name,error='incorrect_success_inheritance'))
                if r['official_task_success'] and not positive:
                    success_without_positive.append(p.name)
                    problems.append(dict(file=p.name,error='native_success_without_positive_attempt'))
                if r.get('alignment_receipt'):
                    clock=r['alignment_receipt']['clock']
                    if clock['ticks']>200:problems.append(dict(file=p.name,error='physics_budget'))
            counts[predictor+'_'+split]=dict(completed_rollouts=len(episodes),candidate_attempts=len(attempts),
                positives=sum(a['official_task_success'] for a in attempts),
                source_episodes=len({(r['task'],r['source_episode']) for r in episodes}))
    extra={str(p.relative_to(ROOT)):file_sha256(p) for p in [ROOT/'our_mani/ablations/physmani_executor_v3.py',ROOT/'scripts/evaluate_physmani_aligned_r05_v3.py',base/'acceptance_tolerances.json']}
    path=base/'baseline_extra_frozen_inputs.json'
    if path.exists():verify_lock(ROOT,json.loads(path.read_text()))
    else:path.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),files=extra),indent=2)+'\n')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not problems,counts=counts,
        errors=problems,native_success_without_active_candidate=success_without_positive,formal_test_used=False)
    (base/'data_quality_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if problems:raise RuntimeError('candidate data quality failed')

if __name__=='__main__':main()
