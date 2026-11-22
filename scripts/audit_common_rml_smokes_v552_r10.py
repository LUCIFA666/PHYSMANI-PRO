"""Read-only first-source native runtime/metric audit for the common RML pilots."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release16 import measured_motion


def main():
    base=ROOT/'artifacts/framework/multitask_r10';roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    owner=json.loads((ROOT/'runtime/framework/native_resource_owner_v401/status.json').read_text())
    rows=[];profile_shas=set()
    for setup_name in ('physmani_skills_v537_setup','physmani_blocks_v547_setup','physmani_basketball_v553_setup'):
        setup=base/setup_name
        if not (setup/'CPU_acceptance.json').exists():continue
        acceptance=json.loads((setup/'CPU_acceptance.json').read_text())
        if acceptance.get('complete') is not True:raise ValueError('only complete CPU admission belongs in the native audit')
        for plan in acceptance['actual_argv_preflights']:
            if setup_name=='physmani_skills_v537_setup' and plan['task'].startswith('moving_basketball_in_hoop'):
                continue  # frozen constructor-failure version retained separately; new548 replaces it
            argv=plan['argv'];out=ROOT/argv[argv.index('--out')+1]
            config=json.loads((ROOT/argv[argv.index('--config')+1]).read_text())
            profile_sha=fingerprint(config['common_author_RML_profile']);profile_shas.add(profile_sha)
            task,raw,method=plan['task'],plan['raw'],plan['method']
            first=next(r for r in roster if r['task']==task)
            if raw!=first['raw_episode_id'] or method not in ('cv_geometry','context_geometry'):
                raise ValueError('first original source and authorized geometry methods required')
            result=dict(task=task,raw=raw,method=method,setup=setup_name,out=str(out.relative_to(ROOT)),
                status='pending',original_source_row_sha256=digest(first),common_RML_profile_sha256=profile_sha)
            if (out/'infrastructure_error.json').exists():
                error=json.loads((out/'infrastructure_error.json').read_text())
                result.update(status='infrastructure_error',error_type=error['type'],error=error['message'])
            elif (out/'case.json').exists():
                path=out/'case.json';case=json.loads(path.read_text());validate_completed_case(case,case['identity'])
                if (case.get('cleanup_completed') is not True or case.get('TEST_used_for_fitting') is not False
                    or case.get('source_raw_episode_id')!=raw or case['task']!=task or case['method']!=method):
                    raise ValueError('incomplete cleanup or changed source/method')
                require_attestation(case.get('common_author_RML_parameters',{}),profile_sha)
                motion=(case.get('all_stage_counted_joint_motion') or
                        case['alignment_receipt'].get('all_stage_counted_joint_motion_v191') or {})
                if len(motion.get('events',[]))!=case['counted_clock']['ticks'] or type(measured_motion(case)) is not bool:
                    raise ValueError('every counted physical tick needs an explicit observed motion window')
                intents=case['actor_metrics']['geometry_intents']
                result.update(status='completed',native_success=case['official_task_success'],
                    ticks=case['counted_clock']['ticks'],motion_pass=measured_motion(case),
                    peak_window_velocity=max([abs(v) for e in motion['events'] for v in e['backward_window_velocity']],default=0.),
                    peak_window_acceleration=motion['peak_actual_window_acceleration'],
                    phase=intents.get('current_phase',intents.get('phase')),
                    actual_RML_actions=len(case['actor_metrics']['physmani_geometry_executor']['actions']),
                    geometry_rejections=sum(not e['accepted'] for e in case['actor_metrics']['physmani_geometry_executor']['generated_target_admissions']),
                    physical_close_edges=sum(bool(e.get('own_close_edge')) for e in intents.get('gripper_events',[])),
                    case=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)))
            rows.append(result)
    if len(profile_shas)>1:raise ValueError('all16 adapters must share the identical author RML profile')
    counts=Counter(r['status'] for r in rows)
    result=dict(schema='r10_first16_two_geometry_common_author_RML_smoke_audit_v552',
        generated_utc=datetime.now(timezone.utc).isoformat(),expected_cases=32,planned_cases=len(rows),
        statuses=dict(counts),native_successes=sum(r.get('native_success',False) for r in rows),
        complete=len(rows)==32 and counts['completed']==32,
        common_RML_profile_sha256=next(iter(profile_shas),None),cases=rows,
        active_jobs=owner['active_jobs'],owner_updated_utc=owner['updated_utc'],
        motion_is_diagnostic_only=True,all_failures_retained=True,
        previous_constructor_failures_retained_in='physmani_skills_v536',
        previous_block_receipt_failures_retained_in=['physmani_blocks_v540','physmani_blocks_v543'],
        full_fixed_four_readiness_granted=False,formal_VAL_TEST_started=False,
        TEST_read=False,Value_NGFF_FASTER_used=False,source_sha256=file_sha(__file__))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=base/'common_author_RML_audits_v552'/(stamp+'.json');atomic_json(path,result)
    print(json.dumps(dict(report=str(path.relative_to(ROOT)),planned=len(rows),statuses=dict(counts),
        native_successes=result['native_successes'],active_jobs=owner['active_jobs'])),flush=True)


if __name__=='__main__':main()
