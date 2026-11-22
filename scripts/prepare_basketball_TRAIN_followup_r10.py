"""Read-only committed basketball evidence and nonadaptive supplement proposal.

Writes new metadata only; no production plan, fitting threshold or raw case is
changed. Proposed additional TRAIN cases are not executable acquisition plans.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.acquisition_lineage import expected_reset, file_sha, pointer

BASE = ROOT / 'artifacts/framework/multitask_r10'
INPUT = BASE / 'basketball_TRAIN_completed_skill_diagnostic_v1/report.json'
OUT = BASE / 'basketball_TRAIN_followup_v3'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def stats(values):
    values = np.asarray(values, float)
    require(values.ndim == 1 and len(values) and np.isfinite(values).all(), 'invalid metric rows')
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
        quantile95=float(np.quantile(values, .95)), maximum=float(values.max()))


def main():
    require(not OUT.exists(), 'fresh evidence/proposal output directory required')
    evidence = read(INPUT)
    require(file_sha(INPUT) == '0b81e62c5ebd8c0e92037e66e857986659d5018c21a1346e8b3b10055606d7a6',
        'actual task100 audit changed')
    registrations = read(ROOT / evidence['actual_registrations']['path'])
    base = read(ROOT / registrations[0]['plan']['path'])
    snapshot = read(ROOT / registrations[0]['source_snapshot']['path'])
    sources = {Path(row['original_path']).name: row for row in snapshot['entries']}
    task_reports = {}
    proposed = []
    for task, report in evidence['tasks'].items():
        require(report['full_100_teacher_observed'], 'full original task100 required')
        author = sources[task + '.py']
        author_path = ROOT / author['path']
        require(file_sha(author_path) == author['sha256'], 'archived author source changed')
        text = author_path.read_text()
        require('th_w0 = 0.3' in text and 'eepose = w0_pose' in text,
            'archived basketball coarse approach code differs')
        endpoint_rows = []
        rare = []
        for row in report['case_receipts_and_adapter_diagnostics']:
            case_path = ROOT / row['canonical_receipt']['path']
            require(file_sha(case_path) == row['canonical_receipt']['sha256'], 'audited canonical changed')
            record = read(case_path)
            command_name = record['entry']['teacher_command_receipt']
            command_path = ROOT / command_name
            require(file_sha(command_path) == record['files_sha256'][command_name], 'actual command ledger changed')
            ledger = read(command_path)
            commands = {command['command_id']: command for command in ledger['commands']}
            for event in ledger['phase_completion_events']:
                command = commands[event['origin_command_id']]
                if event['phase'] == 'approach':
                    terminal = np.asarray(event['current_measured_tcp_xyzw'][:3])
                    current_reference = np.asarray(event['current_reference_native_pose_xyzw'][:3])
                    issue_reference = np.asarray(command['issued_reference_native_pose_xyzw'][:3])
                    committed = np.asarray(event['committed_world_waypoint_xyzw'][:3])
                    endpoint_rows.append(dict(parent=row['parent_source_group'], case_id=row['case_id'],
                        command_id=event['origin_command_id'], start=event['attempt_start_time'],
                        end=event['attempt_end_time'],
                        committed_world_arrival_error_m=float(np.linalg.norm(terminal - committed)),
                        reference_issue_to_completion_displacement_m=float(np.linalg.norm(current_reference - issue_reference)),
                        committed_issue_reference_offset_m=(committed - issue_reference).tolist(),
                        terminal_current_reference_offset_m=(terminal - current_reference).tolist(),
                        evidence=event['evidence'], actual_command=pointer(ROOT, command_path)))
                if event['phase'] == 'acquire' and event['origin_reference_motion_mode'] == 'attached':
                    rare.append(dict(parent=row['parent_source_group'], case_id=row['case_id'],
                        actual_command=pointer(ROOT, command_path), origin_command=command, completion_event=event,
                        mode_provenance='public RGBD/proprioception/contact association; no native held ID',
                        current_reference_pose_provenance='private TRAIN metric label; never actor observation'))
        require(len(endpoint_rows) == report['phase_usable_lag0_support']['approach']['completed_skill_attempts'],
            'completed approach evidence differs from audited support')
        fit = report['real_CPU_fit']['diagnostics']['phase_fits']
        approach = next(value for key, value in fit.items() if key.startswith('approach|'))
        selected = next(row for row in approach['candidates'] if row['lag'] == approach['selected_lag'])
        task_reports[task] = dict(full_100_original_teachers_observed=True,
            archived_author_source=pointer(ROOT, author_path),
            author_semantics=dict(approach_stage='wp0', distance_threshold_m=.3,
                raw_waypoint='current native expertwp0 pose committed at command issue',
                acquire_stage='wp1', acquire_distance_threshold_m=.03),
            committed_world_arrival_error_m=stats([r['committed_world_arrival_error_m'] for r in endpoint_rows]),
            reference_issue_to_completion_displacement_m=stats([
                r['reference_issue_to_completion_displacement_m'] for r in endpoint_rows]),
            parent_crossfit_current_reference_position_error_m=stats([
                v['position'] for v in approach['terminal_parent_residuals'].values()]),
            selected_lag_seconds=approach['selected_lag'],
            lag_candidates=[dict(lag=c['lag'], valid=c['valid'], grouped_TRAIN_CV_error=c.get('grouped_TRAIN_CV_error'))
                for c in approach['candidates']],
            endpoint_rows=endpoint_rows, rare_attached_acquire=rare,
            diagnostic_inference='coarse approach has varying current-ball-relative endpoints; a single relative endpoint'
                ' with 3cm tolerance is mismatched to these observations. More data alone does not establish a fix.',
            raw_label_or_sensor_corruption_demonstrated=False, thresholds_changed=False,
            completed_deposit_original_distinct_parents=report['phase_usable_lag0_support']['deposit']['distinct_completed_TRAIN_parents'])
        original = [c for c in base['cases'] if c['task'] == task and c['teacher_used']]
        require(len(original) == 100, 'original fixed100 teacher roster differs')
        for prior in original:
            case = dict(prior)
            case.update(replica=1, behavior_replica=1,
                raw_episode_id=prior['parent_raw_episode_id'] + '__TRAIN_teacher_committed_waypoint_r1')
            case['case_id'] = digest(dict(split='train', task=task, raw=prior['parent_raw_episode_id'],
                replica=1, behavior=case['behavior']))[:16]
            reset = expected_reset(case, base['protocol'])
            proposed.append(dict(original_K1_case_id=prior['case_id'], proposed_case=case, proposed_reset=reset))
    require(len(proposed) == 200 and len({row['proposed_case']['case_id'] for row in proposed}) == 200,
        'fixed all100 x two-task supplement identity differs')
    require(not {row['proposed_case']['case_id'] for row in proposed} & {c['case_id'] for c in base['cases']},
        'supplement must not replace an original K1 case')
    latest_protocol_plan = read(ROOT / registrations[-1]['plan']['path'])
    proposal = dict(schema='r10_fixed_all100_basketball_teacher_replica1_proposal_v1', proposal_only=True,
        created_utc=datetime.now(timezone.utc).isoformat(), source_audit=pointer(ROOT, INPUT),
        original_plan=registrations[0]['plan'], actual_registrations=evidence['actual_registrations'],
        shared_profile_reference_plan=registrations[-1]['plan'],
        shared_development_reset_profile=latest_protocol_plan['protocol']['development_reset_profile'],
        shared_RML_factory_hook=latest_protocol_plan['protocol']['RML_factory_hook'],
        supplement_scope=dict(tasks=list(task_reports), original_parents_per_task=100, extra_teacher_replicas_per_parent=1,
            replica_indices=[1], total_extra_cases=200, maximum_ticks_per_case=200, maximum_extra_ticks=40000,
            maximum_extra_simulation_seconds=2000, extra_hold_cases=0, context_primary_view_changed=False),
        source_selection='all original100 TRAIN parents in each basketball task, frozen before new outcomes; no successful-parent selection',
        reset_seed_rule='unchanged source-derived r10 native reset identity with base_seed=original protocol.seed and replica=1; optimizer seed1018 is unrelated',
        execution_rule='one semantic episode per fixed identity; no scene/source/outcome resampling, no retry-until-success; genuine terminal/failed/pathless cases retained',
        infrastructure_rule='any infra continuation requires explicit independent registration and raw previous-failure chain; no overwrite',
        parent_support_rule='replicas share parent_source_group and CV fold; repeat success of one parent still counts as one parent',
        preservation=dict(original_K1_2800_unmodified=True, original_TRAIN_1400_unchanged=True,
            original_VAL_275_unchanged=True, original_full_composite_gate_unchanged=True,
            minimum_completed_distinct_parents=8, maximum_position_tolerance_m=.03, maximum_angle_tolerance_rad=.15,
            official_TEST_unmodified=True),
        comparison_rule='separate declared TRAIN-budget amendment and append-only namespace; all candidate pipelines receive the same resulting library; disclose extra TRAIN budget in baseline comparison',
        consumer_status='current frozen originalK1 collectors/lineage deliberately reject replica1; needs a separately reviewed supplement schema/consumer and source freeze before execution',
        fit_status='no outcome-based coverage guarantee; extra replicas may improve high-speed deposit support but cannot be presented as fixing coarse approach or singleton attached partitions',
        required_model_followup='resolve TRAIN-only phase/reference/endpoint representation semantics under unchanged min8/.03m/.15rad gates before release; do not relabel old events or drop rare partitions silently',
        proposed_cases=proposed, native_or_GPU_launched=False, production_plan_created=False)
    OUT.mkdir()
    write(OUT / 'approach_endpoint_evidence.json', dict(schema='r10_basketball_current_reference_endpoint_diagnostic_v1',
        source_audit=pointer(ROOT, INPUT), TRAIN_only=True, VAL_TEST_read=False, tasks=task_reports,
        numerical_threshold_search=False, algorithm_or_production_source_changed=False,
        interpretation_is_diagnostic_not_policy_performance=True))
    write(OUT / 'fixed_all100_teacher_replica1_proposal.json', proposal)
    launch_input = BASE / 'basketball_TRAIN_completed_skill_diagnostic_v1/launch_plan_v3_registration_binding.json'
    launch = read(launch_input)
    old_snapshot_fields=('latest_reviewed_supervision_snapshot','latest_reviewed_snapshot_created_utc',
        'latest_reviewed_snapshot_canonical_count','snapshot_is_observed_so_far_not_final',
        'required_phases_observed_below8','concrete_supervision_risk')
    launch['prior_supervision_snapshot_metadata']={key:launch.pop(key) for key in old_snapshot_fields if key in launch}
    launch.update(prior_registration_update=pointer(ROOT, launch_input),
        latest_basketball_task100_diagnostic=pointer(ROOT, INPUT),
        supersedes_basketball_reviewed_observed_so_far_snapshot=True,
        basketball_interaction_artifact_currently_pending=True,
        current_pending_is_not_permission_to_relax_release_gates=True,
        final_original_basketball_teacher_count_per_task=100,
        final_original_basketball_deposit_support=dict(normal=25,high_speed=6),
        concrete_supervision_risk='Original basketball teachers100/100 per task audited; high-speed deposit6<8; both approach caps and rare attached-acquire support remain pending.')
    write(OUT / 'launch_plan_v3_seven_registrations.json', launch)
    lines = ['Metadata only; no full training launched. Working directory: ' + str(ROOT),
        'All2800 TRAIN+275VAL raw parent commits, stable source/archive audits, and final consumer source freeze required.',
        'Basketball task100 audit is complete but interaction fitter remains pending; production gates unchanged.',
        'The registration chain is actualv3 seven plans; preserve v11 plus every failed/recovered execution.', '']
    for job in launch['command_jobs']:
        environment=' '.join(key+'='+shlex.quote(value) for key,value in job.get('environment',{}).items())
        lines.extend([job['id'], (environment+' ' if environment else '') + shlex.join(job['argv']), ''])
    (OUT / 'launch_commands_v3.txt').write_text('\n'.join(lines))
    (OUT / 'report.txt').write_text(
        'Both basketball task100 original teacher rosters are complete and audited.\n'
        'Normal deposit25 distinct completed parents; high-speed deposit6 final originalK1 parents (<8).\n'
        'Approach coarse endpoint dispersion fails fixed3cm cap despite100/99 supported parents.\n'
        'Acquire attached partitions have only1 parent each. No events dropped or tolerance relaxed.\n'
        'A proposal for all100 parents x2tasks xreplica1 (200 extra teachers) is metadata only.\n'
        'It does not guarantee a remedy for approach representation or rare attachment modes.\n'
        'Launch metadata binds actualv3 seven registrations. No native/GPU or production plan created.\n')
    print(json.dumps(dict(evidence=pointer(ROOT, OUT / 'approach_endpoint_evidence.json'),
        supplement_proposal=pointer(ROOT, OUT / 'fixed_all100_teacher_replica1_proposal.json'),
        launch=pointer(ROOT, OUT / 'launch_plan_v3_seven_registrations.json'))))


if __name__ == '__main__':
    main()
