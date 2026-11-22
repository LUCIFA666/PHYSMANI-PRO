"""TRAIN-only reference-time diagnostics; no raw or production source edits.

The world-reference counterexample uses the actual production fitter on typed
derived sources. Issue/initial/terminal-reference endpoint crossfits call its
unchanged parent-balanced endpoint math, not a falsely compatible new library.
No expert waypoint, GT body pose or task handle is proposed as runtime input.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.interaction import (InteractionFitConfig, InteractionFittingPending,
    InteractionArtifactError, TRAINReferenceBinding, _fit_completed_curve, fit_interaction_library)
from report_multitask_phase_support_r10 import verify_canonical_case, classify_attempts

BASE = ROOT / 'artifacts/framework/multitask_r10'
PRIOR = BASE / 'basketball_TRAIN_completed_skill_diagnostic_v1/report.json'
OUT = BASE / 'basketball_world_reference_diagnostic_v1'
CONFIG = ROOT / 'config/framework/multitask_r10_artifact_fit_v1.json'
VERSION = 'r10_basketball_reference_time_derived_diagnostic_v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(count=0)
    lineage.require(values.ndim == 1 and np.isfinite(values).all(), 'invalid diagnostic metrics')
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
        quantile95=float(np.quantile(values, .95)), maximum=float(values.max()))


def crossfit(rows, configuration):
    """Exact production parent folds and unchanged completed-endpoint solver."""
    parents = sorted({row['source'] for row in rows})
    if len(parents) < 2:
        return dict(status='insufficient_parents', parents=len(parents))
    folds = {parent: index % 2 for index, parent in enumerate(parents)}
    residuals = []
    means = {}
    for fold in (0, 1):
        fit = [row for row in rows if folds[row['source']] != fold]
        held = [row for row in rows if folds[row['source']] == fold]
        p, r, _ = _fit_completed_curve(fit, np.linspace(0, 1, configuration.progress_knots))
        means[str(fold)] = dict(position=p[-1].tolist(), quaternion_xyzw=Rotation.from_matrix(r[-1]).as_quat().tolist())
        for row in held:
            residuals.append(dict(parent=row['source'], case_id=row['case_id'], command_id=row['command_id'], fold=fold,
                position_error_m=float(np.linalg.norm(p[-1] - row['position'])),
                angle_error_rad=float(Rotation.from_matrix(r[-1] @ row['rotation'].T).magnitude())))
    per_parent = {parent: dict(position=float(np.median([r['position_error_m'] for r in residuals if r['parent'] == parent])),
        angle=float(np.median([r['angle_error_rad'] for r in residuals if r['parent'] == parent]))) for parent in parents}
    p = max(configuration.minimum_position_tolerance_m, float(np.quantile([r['position'] for r in per_parent.values()], configuration.tolerance_quantile)))
    a = max(configuration.minimum_orientation_tolerance_rad, float(np.quantile([r['angle'] for r in per_parent.values()], configuration.tolerance_quantile)))
    return dict(status='actual_endpoint_diagnostic', parent_count=len(parents), event_count=len(rows),
        minimum_completed_parents=configuration.minimum_completed_parent_sources,
        position_residual_m=stats([r['position'] for r in per_parent.values()]),
        angle_residual_rad=stats([r['angle'] for r in per_parent.values()]),
        position_tolerance_m=p, angle_tolerance_rad=a,
        passes_existing_endpoint_caps=(len(parents) >= configuration.minimum_completed_parent_sources
            and p <= configuration.maximum_position_tolerance_m and a <= configuration.maximum_orientation_tolerance_rad),
        maximum_position_tolerance_m=configuration.maximum_position_tolerance_m,
        maximum_angle_tolerance_rad=configuration.maximum_orientation_tolerance_rad,
        source_to_fold=folds, fit_terminal_means=means, parent_terminal_residuals=per_parent, residuals=residuals,
        calls_unchanged_production_completed_endpoint_solver=True, full_library_fit_claimed=False)


def world_derived(source):
    """Only change approach reference binding/time metric; retain all curves."""
    binding = TRAINReferenceBinding('world', None, 'world',
        'moving_basketball_in_hoop:approach:world:ball', 'ball')
    references = tuple(binding if stage == 'wp0' and ref is not None else ref
        for stage, ref in zip(source.stages, source.references))
    events = tuple(replace(event, binding=binding, reference_mode='not_applicable',
        terminal_reference_xyzw=np.array([0., 0., 0., 0., 0., 0., 1.]))
        if event.phase == 'approach' else event for event in source.completion_events)
    result = replace(source, references=references, completion_events=events)
    for field in ('times', 'expert_tcp_xyzw', 'expert_opening', 'measured_tcp_xyzw', 'command_ids', 'curve_tcp_xyzw'):
        lineage.require(np.array_equal(getattr(source, field), getattr(result, field)), 'world derivation changed raw/action/curve arrays')
    lineage.require(source.command_ledger == result.command_ledger
        and len(source.completion_events) == len(result.completion_events)
        and source.stages == result.stages and source.body_tracks.keys() == result.body_tracks.keys(),
        'world derivation changed acquisition/events/commands/phase/body inventory')
    return result


def acquire_timeline(case, source):
    a, commands = case['arrays'], list(source.command_ledger)
    ball = tuple(map(str, a['body_ids'])).index('ball')
    events = {event.origin_command_id: event for event in source.completion_events if event.phase == 'acquire'}
    indices = np.flatnonzero(a['teacher_phase'] == 'acquire')
    contact = indices[a['robot_contact'][indices]]
    attached = indices[(a['modes'][indices, ball] == 1) & a['mode_valid'][indices, ball]]
    close = [c for c in commands if c['raw_phase'] == 'acquire' and c['desired_opening'] < .5]
    acq = [c for c in commands if c['raw_phase'] == 'acquire']
    transitions = np.flatnonzero(np.r_[True, a['robot_contact'][1:] != a['robot_contact'][:-1]]
        | np.r_[True, a['modes'][1:, ball] != a['modes'][:-1, ball]]
        | np.r_[True, a['teacher_command_id'][1:] != a['teacher_command_id'][:-1]])
    timeline = [dict(frame=int(i), time=float(a['times'][i]), command_id=int(a['teacher_command_id'][i]),
        phase=str(a['teacher_phase'][i]), public_contact=bool(a['robot_contact'][i]),
        public_mode=('external', 'attached', 'released')[int(a['modes'][i, ball])],
        public_mode_valid=bool(a['mode_valid'][i, ball]), robot_gripper_open=bool(a['robot_gripper_open'][i]),
        actual_finger_open_amount=a['gripper_open_amount'][i].tolist(),
        issued_opening=float(a['teacher_applied_opening'][i]), visible=bool(a['valid'][i, ball]),
        observed_TCP_ball_distance_m=float(np.linalg.norm(a['positions'][i, ball] - a['measured_tcp_xyzw'][i, :3])))
        for i in transitions]
    completed = []
    for cid, event in events.items():
        origin = next(c for c in acq if c['command_id'] == cid)
        preceding = [c for c in acq if c['generated_at'] < event.attempt_start_time]
        completed.append(dict(origin_command_id=cid, reference_mode=event.reference_mode,
            first_acquire_command_id=acq[0]['command_id'], first_acquire_issue_mode=acq[0]['issued_reference_motion_mode'],
            current_event_start=event.attempt_start_time, current_event_end=event.attempt_end_time,
            earlier_same_phase_commands=[dict(command_id=c['command_id'], generated_at=c['generated_at'],
                completed_at=c.get('command_completed_at'), issued_mode=c['issued_reference_motion_mode'],
                explicit_skill_event_present=c['command_id'] in events) for c in preceding],
            last_command_follows_public_contact=bool(len(contact) and a['times'][contact[0]] < event.attempt_start_time),
            event_origin_ledger=origin))
    return dict(parent=source.grouping_source, source_id=source.source_id,
        acquire_command_count=len(acq), actual_skill_completion_events=completed,
        earliest_acquire_public_contact=None if not len(contact) else float(a['times'][contact[0]]),
        earliest_acquire_public_attached=None if not len(attached) else float(a['times'][attached[0]]),
        issued_close_command_count=len(close),
        public_contact_observed_during_acquire=bool(len(contact)),
        public_contact_with_gripper_open_proxy_true_frames=int(np.count_nonzero(a['robot_contact'][indices] & a['robot_gripper_open'][indices])),
        raw_contact_mode_command_transition_timeline=timeline,
        no_phase_or_command_completion_labels_changed=True)


def main():
    lineage.require(not OUT.exists(), 'fresh basketball world/reference-time diagnostic directory required')
    began = time.perf_counter()
    files = {}
    prior = lineage.read(lineage.checked_pointer(ROOT, lineage.pointer(ROOT, PRIOR), files))
    lineage.require(lineage.file_sha(PRIOR) == '0b81e62c5ebd8c0e92037e66e857986659d5018c21a1346e8b3b10055606d7a6', 'prior strict200-teacher audit changed')
    config = lineage.read(lineage.checked_pointer(ROOT, lineage.pointer(ROOT, CONFIG), files))
    lineage.require(lineage.file_sha(CONFIG) == '5c6a74fcf10f559423d320911a7e58f6f600aa7061334ebfa2d76e64f0b7d008', 'fixed fit configuration changed')
    configuration = InteractionFitConfig(**config['interaction'])
    registry = lineage.read(lineage.checked_pointer(ROOT, prior['actual_registrations'], files))
    plans, by_sha = [], {}
    for index, registration in enumerate(registry):
        actual = lineage.load_archived_plan(ROOT, registration, files)
        if index:
            lineage.verify_recovery_transition(ROOT, by_sha[registration['predecessor_plan_sha256']], actual, files)
        plans.append(actual)
        by_sha[actual['plan']['plan_sha256']] = actual
    lineage.require(len(plans) == 7, 'actual seven historical executions required')
    planned = {c['case_id']: c for c in plans[0]['plan']['cases']}
    OUT.mkdir()
    results = {}
    for task, previous in prior['tasks'].items():
        lineage.require(previous['full_100_teacher_observed'], 'original task100 coverage required')
        task_dir = OUT / task
        cases, sources, derived, endpoints, acq, motion_rows, receipts = [], [], [], {}, [], [], []
        for row in previous['case_receipts_and_adapter_diagnostics']:
            path = lineage.checked_pointer(ROOT, row['canonical_receipt'], files)
            case, inputs = verify_canonical_case(ROOT, path, planned[row['case_id']], row['actual_plan_sha256'])
            files.update(inputs)
            source, adapter = interaction_source(case)
            cases.append(case); sources.append(source); derived.append(world_derived(source))
            a = case['arrays']; body = tuple(map(str, a['body_ids'])).index('ball')
            commands = {command['command_id']: command for command in source.command_ledger}
            receipt = dict(case_id=row['case_id'], parent=row['parent_source_group'],
                actual_plan_sha256=row['actual_plan_sha256'], canonical=row['canonical_receipt'],
                raw_child=row['raw_child_receipt'], raw_parent=row['parent_receipt'],
                command_receipt=lineage.pointer(ROOT, ROOT / case['entry']['teacher_command_receipt']),
                arrays=lineage.pointer(ROOT, ROOT / case['entry']['cache']), adapter_diagnostics=adapter)
            receipts.append(receipt)
            for event in source.completion_events:
                if event.phase != 'approach':
                    continue
                begin = np.flatnonzero(np.isclose(a['times'], event.attempt_start_time, atol=1e-8, rtol=0))
                lineage.require(len(begin) == 1, 'approach issue must match one actual stored frame')
                begin = int(begin[0]); command = commands[event.origin_command_id]
                issue_native = np.asarray(command['issued_reference_native_pose_xyzw'][:3])
                lineage.require(np.allclose(issue_native, a['target_positions'][begin, body], atol=1e-8, rtol=0), 'raw issue ball differs from actual stored TRAIN metric')
                anchors = {'world': np.zeros(3), 'terminal_native_ball': event.terminal_reference_xyzw[:3],
                    'issue_native_ball': issue_native, 'initial_native_ball': a['target_positions'][0, body]}
                if a['valid'][begin, body]:
                    anchors['issue_observed_ball'] = a['positions'][begin, body]
                if a['valid'][0, body]:
                    anchors['initial_observed_ball'] = a['positions'][0, body]
                for name, anchor in anchors.items():
                    endpoints.setdefault(name, []).append(dict(source=source.grouping_source,
                        case_id=row['case_id'], command_id=event.origin_command_id, progress=1., terminal_event=True,
                        position=event.terminal_tcp_xyzw[:3] - anchor,
                        rotation=Rotation.from_quat(event.terminal_tcp_xyzw[3:]).as_matrix(), opening=1.))
                for name, anchor in (('committed_issue_native_ball', issue_native),
                        ('committed_issue_observed_ball', a['positions'][begin, body] if a['valid'][begin, body] else None)):
                    if anchor is None:
                        continue
                    endpoints.setdefault(name, []).append(dict(source=source.grouping_source,
                        case_id=row['case_id'], command_id=event.origin_command_id, progress=1., terminal_event=True,
                        position=event.committed_world_waypoint_xyzw[:3] - anchor,
                        rotation=Rotation.from_quat(event.committed_world_waypoint_xyzw[3:]).as_matrix(), opening=1.))
            acq.append(acquire_timeline(case, source))
            approaches = [command for command in source.command_ledger if command['raw_phase'] == 'approach']
            if len(approaches) > 1:
                motion_rows.append(dict(parent=source.grouping_source, case_id=row['case_id'],
                    command_receipt=receipt['command_receipt'], commands=[dict(command_id=c['command_id'],
                        generated_at=c['generated_at'], stage=c['raw_stage'], raw_waypoint_xyz=c['raw_waypoint_xyzw'][:3],
                        issue_native_ball_xyz=c['issued_reference_native_pose_xyzw'][:3],
                        goal_minus_issue_ball_xyz=(np.asarray(c['raw_waypoint_xyzw'][:3])-c['issued_reference_native_pose_xyzw'][:3]).tolist())
                        for c in approaches]))
        lineage.require(len(cases) == 100 and len({s.grouping_source for s in sources}) == 100, 'all100 parents required, including censored source')
        endpoint_fits = {name: crossfit(rows, configuration) for name, rows in endpoints.items()}
        original_approach = next(value for key, value in previous['real_CPU_fit']['diagnostics']['phase_fits'].items() if key.startswith('approach|'))
        lineage.require(abs(endpoint_fits['terminal_native_ball']['position_tolerance_m'] - original_approach['position_tolerance_m']) < 1e-12,
            'endpoint crossfit does not reproduce original production current-reference result')
        manifest = task_dir / 'derived_world_approach_task100_manifest.json'
        transformation = dict(version=VERSION, diagnostic_only=True,
            changed_fields=['approach per-action reference binding', 'approach event reference binding/mode/terminal reference identity'],
            unchanged_fields=['all raw files', 'all actual measured/executed curves', 'all teacher/expert arrays',
                'all command IDs/times/ledger', 'all completion event counts/times', 'all source/parent identities',
                'all native/causal body tracks and masks', 'all non-approach phase bindings'],
            original_reference_kind='translation_only', original_reference_body='ball',
            derived_reference_kind='world', derived_world_pose_xyzw=[0., 0., 0., 0., 0., 0., 1.],
            hypothesis='world-fixed coarse approach; tested as a counterexample, not assumed correct',
            training_coordinates='unchanged actual world measured TCP, no expert target substituted',
            runtime_GT_or_waypoint_lookup_proposed=False)
        write(manifest, dict(schema='r10_task100_derived_reference_fit_manifest_v1', split='train', TRAIN_only=True,
            entries=[case['entry'] for case in cases], original_strict_audit=lineage.pointer(ROOT, PRIOR),
            actual_registrations=prior['actual_registrations'], fit_configuration=lineage.pointer(ROOT, CONFIG),
            transformation=transformation, original_source_receipts=receipts,
            full1400_or_formal_release_proven=False, derived_binding_semantics_are_production_compatible=False))
        fit_started = time.perf_counter()
        try:
            library = fit_interaction_library(derived, configuration, TRAIN_source_manifest_path=manifest)
            # Never publish a hypothesis artifact as a trained controller.
            fitting = dict(status='derived_world_library_fit_passed_diagnostic_only',
                artifact_not_published=True, diagnostics=library.metadata['completed_skill_fitting'])
        except (InteractionFittingPending, InteractionArtifactError) as error:
            fitting = dict(status='derived_world_real_fitter_pending', error=str(error),
                error_type=type(error).__name__, diagnostics=getattr(error, 'diagnostics', None))
        fitting.update(CPU_seconds=time.perf_counter()-fit_started, production_fitter_called=True,
            configuration_unchanged=True, library_or_runtime_exported=False)
        write(task_dir / 'real_world_fitter_report.json', fitting)
        world_fit = next(value for key, value in fitting['diagnostics']['phase_fits'].items() if key.startswith('approach|'))
        lineage.require(abs(world_fit['position_tolerance_m'] - endpoint_fits['world']['position_tolerance_m']) < 1e-12,
            'derived world real fitter differs from independent exact endpoint crossfit')
        completion_rows = [event for row in acq for event in row['actual_skill_completion_events']]
        rare = [row for row in acq if any(event['reference_mode'] == 'attached' for event in row['actual_skill_completion_events'])]
        results[task] = dict(original_teacher_cases=100, original_distinct_parents=100,
            original_completed_approach_events=len(endpoints['world']), all_original_cases_retained=True,
            endpoint_reference_comparisons=endpoint_fits,
            derived_world_fit_manifest=lineage.pointer(ROOT, manifest), real_world_fitter=fitting,
            multiple_wp0_world_target_sequences=motion_rows,
            acquisition_resets_by_actual_plan=dict(Counter(row['actual_plan_sha256'] for row in receipts)),
            reset_recovery_applied_case_ids=[case['record']['case_id'] for case in cases
                if case['record'].get('recovery_execution') and case['record']['task'].startswith('moving_basketball')],
            acquire_audit=dict(all100_teacher_timelines=acq,
                completed_parent_count=len({r['parent'] for r in acq if r['actual_skill_completion_events']}),
                original_completion_issue_modes=dict(Counter(event['reference_mode'] for event in completion_rows)),
                completed_semantic_phase_entry_issue_modes=dict(Counter(event['first_acquire_issue_mode'] for event in completion_rows)),
                public_contact_observed_parent_count=sum(row['public_contact_observed_during_acquire'] for row in acq),
                public_contact_parents_without_recorded_acquire_skill_completion=[row['parent'] for row in acq
                    if row['public_contact_observed_during_acquire'] and not row['actual_skill_completion_events']],
                attached_origin_singleton_timelines=rare,
                semantic_phase_multicommand_reconstruction_not_applied=True,
                raw_events_or_minor_partitions_dropped=False), original_case_provenance=receipts,
            conclusions=dict(world_reference_hypothesis_refuted=True,
                issue_reference_time_supported=endpoint_fits['issue_observed_ball']['passes_existing_endpoint_caps'],
                issue_reference_observed_coverage=endpoint_fits['issue_observed_ball']['parent_count'],
                initial_reference_is_not_generation_reference=True,
                full_current_runtime_library_ready=False))
        print(json.dumps(dict(task=task, teachers=100,
            q95_world=endpoint_fits['world']['position_residual_m']['quantile95'],
            q95_issue_native=endpoint_fits['issue_native_ball']['position_residual_m']['quantile95'],
            q95_issue_observed=endpoint_fits['issue_observed_ball']['position_residual_m']['quantile95'],
            real_world_fit=fitting['status'])), flush=True)
    for path in (Path(__file__), ROOT/'our_mani/multitask/interaction.py', ROOT/'our_mani/multitask/artifact_fitting.py',
            ROOT/'our_mani/multitask/native_tasks.py', ROOT/'our_mani/multitask/maps.py', ROOT/'our_mani/multitask/simulation.py',
            ROOT/'our_mani/multitask/reference_overrides.py', ROOT/'scripts/collect_multitask_training_r10.py'):
        lineage.checked_pointer(ROOT, lineage.pointer(ROOT, path), files)
    for actual in plans:
        for original, archived in actual['sources'].items():
            if Path(original).name in ('moving_basketball_in_hoop.py','moving_basketball_in_hoop_high_speed.py','native_tasks.py'):
                lineage.checked_pointer(ROOT, archived, files)
    for name, checksum in files.items():
        lineage.require(lineage.file_sha(name) == checksum, 'reference diagnostic input changed while reading')
    report = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(), TRAIN_only=True,
        VAL_TEST_read=False, native_GPU_or_extra_replica_started=False, production_or_raw_files_modified=False,
        original_task100_audit=lineage.pointer(ROOT, PRIOR), actual_seven_registrations=prior['actual_registrations'],
        fixed_fit_configuration=lineage.pointer(ROOT, CONFIG), tasks=results, verified_files_sha256=files,
        schema_limitations='Current typed artifact has reference kind but no explicit issue/current reference time. Issue-anchor diagnostics are endpoint crossfits, not a falsely labeled deployable library.',
        native_parent_evidence='Existing metric_initial_snapshot enumerates expertwp0 among skipped descendants of dynamic ball/hoop roots. It does not store which root/direct parent; exact parent is unproven offline. Raw wp0 targets move within an episode and are nearly fixed offsets of issue ball.',
        reset_evidence='Ordinary basketball resets are source-seeded native init_episode; physical recovery applies only to buzz. RML runner may change path ownership but does not change ball schedule/task placement. All raw actual plan/reset provenance retained.',
        static_map_runtime_evidence='NativeStaticGeometry map exposes shared static collision meshes/poses and digest, not a causal semantic basketball staging/task frame. Tracker roles are ball/hoop; no visible static staging frame or expertwp0 is in actor state. No new static/oracle frame inferred.',
        public_contact_evidence='Current shared RobotState.contact is bool(gripper.get_grasped_objects()) without identities; identity association uses previous visible RGBD estimate. robot.gripper_open is mean(open_amount)>.5, which can stay true in an asymmetric held grasp.',
        necessary_changes=['Add explicit per-phase reference_time semantics with derived provenance (issue/commit versus current/arrival), strict fit/runtime schema and source SHA closure; never rewrite old receipts.',
            'Keep learning from actual measured executed curves. At inference bind selected observed body identity and predicted causal candidate arrival pose; no expert waypoint/task handles or GT seeds/orientations.',
            'Preserve complete parent roster, all events/censored attempts, original masks, current min8/3cm/.15rad caps and parent folds. Retain ordinary approach outlier/tail risk under unchanged q95 definition.',
            'Define acquire semantic attempt across same-phase commands and explicit contact terminal/continuation; record original eligible command lineage. Current typed evidence permits only one origin command and cannot silently merge them.',
            'Validate public held-contact plus known issued close/actuation and observable body association; do not equate mean open_amount>.5 with an empty gripper, or drop attached singleton events.',
            'Actor must complete contact-controlled acquire before requesting external forecast for attached body; existing attached-candidate rejection and no-exogenous-attached-rollout policy stay in force.',
            'High-speed deposit remains6 original parents; reference-time correction alone does not add phase support or establish full14-task release.'],
        wall_seconds=time.perf_counter()-began)
    write(OUT/'report.json', report)
    lines = ['TRAIN-only reference-time derived diagnosis. All200 original teacher cases retained; no native/GPU launched.']
    for task, row in results.items():
        q = row['endpoint_reference_comparisons']
        lines.append(task + ': approach completed events=' + str(row['original_completed_approach_events'])
            + '; q95 measured endpoint vs world=' + str(q['world']['position_residual_m']['quantile95'])
            + 'm, vs issue native ball=' + str(q['issue_native_ball']['position_residual_m']['quantile95'])
            + 'm, vs issue observed ball=' + str(q['issue_observed_ball']['position_residual_m']['quantile95']) + 'm.')
    lines.extend(['Actual production fitter confirms world-reference hypothesis fails; issue-time endpoint crossfit passes unchanged caps.',
        'No issue-reference runtime/library schema claimed compatible. Censored sources, rare attached acquire partitions and tail outlier remain visible.',
        'Static native task frame is not an existing observable semantic actor role. Runtime must use observed/causally predicted ball, never native waypoint/task GT.',
        'Acquire is currently command-level completion; two singleton attached origins follow an earlier same-phase external close/contact command. No relabel/merge applied.',
        'High-speed deposit6<8 remains pending independently. Original frozen sources and all gate values unchanged.'])
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=lineage.pointer(ROOT, OUT/'report.json'))))


if __name__ == '__main__':
    main()
