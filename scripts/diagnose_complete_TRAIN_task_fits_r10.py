"""Strict CPU-only phase fitting for tasks with all100 original teachers.

This is an in-progress diagnostic. It never constructs a complete TRAIN
manifest, launches a simulator, selects parents by outcome, or releases a model.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import artifact_fitting, interaction, semantic_acquire
from our_mani.multitask.reference_time import load_rule as load_reference_rule
from our_mani.multitask.training_view import validate_arrays


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def phase_summary(diagnostics):
    return {key: {name: value.get(name) for name in (
        'passed', 'completed_parent_count', 'reason', 'position_tolerance_m',
        'orientation_tolerance_rad', 'duration_seconds', 'duration_budget_seconds')}
        for key, value in diagnostics.get('phase_fits', {}).items()}


def run(registrations_path, configuration_path, reused_basketball_report, output):
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    output = a.local(ROOT, output)
    a.require(not output.exists(), 'fresh diagnostic output required')
    output.mkdir(parents=True)
    files = {}
    def checked(path):
        return a.checked_pointer(ROOT, a.pointer(ROOT, a.local(ROOT, path)), files)
    registrations_path = checked(registrations_path)
    registrations = a.read(registrations_path)
    a.require(len(registrations) == 8, 'actual eight-registration TRAIN chain required')
    plans = {}
    for registration in registrations:
        current = a.load_archived_plan(ROOT, registration, files)
        key = current['plan']['plan_sha256']
        a.require(key not in plans, 'duplicate actual execution plan')
        if 'predecessor_plan_sha256' in registration:
            a.verify_recovery_transition(ROOT, plans[registration['predecessor_plan_sha256']], current, files)
        plans[key] = current
    original = next(iter(plans.values()))
    roster = {case['case_id']: case for case in original['plan']['cases']}
    a.require(len(roster) == 2800 and all(case['split'] == 'train' for case in roster.values()),
              'unchanged original2800 TRAIN roster required')
    a.require(all(plan['plan']['cases'] == original['plan']['cases'] for plan in plans.values()),
              'actual recovery roster differs from original fixed source/budget')
    expected_by_task = {}
    for case in roster.values():
        if case['teacher_used'] is True:
            expected_by_task.setdefault(case['task'], []).append(case)
    a.require(len(expected_by_task) == 14 and all(len(rows) == 100 and
        len({row['parent_source_group'] for row in rows}) == 100 and
        all(row['replica'] == row['behavior_replica'] == 0 for row in rows)
        for rows in expected_by_task.values()), 'original100 K1 teacher parents per task required')
    configuration_path = checked(configuration_path)
    configuration = a.read(configuration_path)
    a.require(configuration['fit_split'] == 'train' and configuration['VAL_used'] is False
        and configuration['TEST_used'] is False, 'TRAIN-only fitting configuration required')
    config = interaction.InteractionFitConfig(**configuration['interaction'])
    a.require(config.minimum_completed_parent_sources == 8 and config.maximum_position_tolerance_m == .03
        and config.maximum_orientation_tolerance_rad == .15 and config.development_only is False,
        'existing8/.03/.15 production fitting requirements cannot change')
    reference = load_reference_rule(ROOT, configuration['reference_time_derivation_rule'])
    semantic = semantic_acquire.load_rule(ROOT, configuration['semantic_acquire_derivation_rule'])
    files.update(reference.verified_files_sha256)
    files.update(semantic.verified_files_sha256)
    source_paths = [Path(__file__), Path(artifact_fitting.__file__), Path(interaction.__file__),
                    Path(semantic_acquire.__file__), ROOT/'our_mani/multitask/specs.py',
                    ROOT/'our_mani/multitask/training_view.py', Path(a.__file__)]
    source_before = {str(path): a.file_sha(path) for path in source_paths}
    files.update(source_before)
    # Freeze this inventory once, before fitting. Later live queue completions
    # cannot make an incomplete task appear complete within this diagnostic.
    inventory = {}
    uncommitted = []
    for key, current in plans.items():
        for path in sorted((current['collection']/'cases/train').glob('*/*/case.json')):
            record = a.read(path)
            cid = record.get('case_id')
            a.require(cid in roster and record.get('plan_sha256') == key,
                      'unknown or wrongly planned canonical acquisition')
            a.require(cid not in inventory, 'duplicate/replaced canonical across actual plans')
            parent = Path(a.command_argument(record['parent_process_receipt']['command'], '--output')).resolve().parent/'parent_receipt.json'
            if not parent.is_file() or a.read(parent).get('status') != 'completed':
                uncommitted.append(dict(case_id=cid, canonical_receipt=a.pointer(ROOT, path)))
                continue
            inventory[cid] = (current, path, parent)
    inventory_rows = [dict(case_id=cid, actual_plan_sha256=current['plan']['plan_sha256'],
        canonical_receipt=a.pointer(ROOT, path), parent_receipt=a.pointer(ROOT, parent))
        for cid, (current, path, parent) in sorted(inventory.items())]
    inventory_path = output/'snapshot_canonical_inventory.json'
    write(inventory_path, dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
        actual_registrations=a.pointer(ROOT, registrations_path), committed_canonical_candidates=inventory_rows,
        parent_commits_not_yet_complete=uncommitted,
        incomplete_task_raw_arrays_not_audited_or_fitted=True))
    reused_path = checked(reused_basketball_report)
    reused = a.read(reused_path)
    a.require(reused['schema'] == 'r10_actual_original100x2_semantic_acquire_CPU_refit_v1'
        and reused['all200_original_teacher_sources_retained'] is True
        and reused['formal_model_or_policy_release'] is False
        and reused['thresholds_or_parent_support_relaxed'] is False,
        'only prior complete actual original100x2 TRAIN diagnostic can be reused')
    a.require(reused['fit_config'] == a.pointer(ROOT, configuration_path),
              'reused task diagnostic used another frozen fit configuration')
    for path, checksum in reused['source_sha256'].items():
        a.require(a.file_sha(path) == checksum, 'reused diagnostic source changed')
    tasks = {}
    for task, expected in sorted(expected_by_task.items()):
        missing = [case['case_id'] for case in expected if case['case_id'] not in inventory]
        row = dict(expected_original_teacher_parents=100, snapshot_committed_teacher_candidates=100-len(missing),
            missing_teacher_case_ids=missing, all_original_sources_retained=True,
            task_success_does_not_select_fit_sources=True, diagnostic_only=True, formal_model_release=False)
        if missing:
            row.update(status='not_fitted_incomplete_original100_teacher_snapshot', real_fit_performed=False)
            tasks[task] = row
            continue
        if task in semantic_acquire.TASKS:
            prior = reused['tasks'][task]
            a.checked_pointer(ROOT, prior['fit_manifest'], files)
            row.update(status='reused_existing_actual_original100_CPU_diagnostic',
                prior_report=a.pointer(ROOT, reused_path), actual_fit_status=prior['status'],
                fit_manifest=prior['fit_manifest'], phase_fits=phase_summary(prior['completed_skill_fitting']),
                original_sources_verified_by_prior_strict_raw_diagnostic=True,
                fitting_repeated=False, real_fit_performed=False)
            tasks[task] = row
            continue
        cases = []
        sources = []
        case_proofs = []
        errors = []
        for expected_case in expected:
            cid = expected_case['case_id']
            current, path, parent = inventory[cid]
            try:
                completion = a.verify_completed(ROOT, dict(canonical_receipt=a.pointer(ROOT, path),
                    parent_receipt=a.pointer(ROOT, parent)), current, expected_case, files)
                if 'recovery_input' in current['plan']['protocol']:
                    previous = plans[current['registration']['predecessor_plan_sha256']]
                    recovery = a.read(a.checked_pointer(ROOT, current['plan']['protocol']['recovery_input'], files))
                    preceding = [dict(actual_plan_sha256=previous['plan']['plan_sha256'], parent_receipt=pointer)
                        for pointer in recovery['failed_parent_receipts']
                        if a.read(a.local(ROOT, pointer['path']))['case_id'] == cid]
                    a.verify_recovery_execution(ROOT, completion, current, previous, preceding, files)
                record = a.read(path)
                entry = record['entry']
                cache = a.local(ROOT, entry['cache'])
                command_path = a.local(ROOT, entry['teacher_command_receipt'])
                a.require(entry['teacher_command_receipt'] == entry['phase_completion_events_receipt']
                    and record['files_sha256'][entry['teacher_command_receipt']] == a.file_sha(command_path),
                    'raw teacher command/completion ledger differs')
                with np.load(cache, allow_pickle=False) as stored:
                    arrays = {key: stored[key].copy() for key in stored.files}
                stats = validate_arrays(arrays, entry, record)
                case = dict(path=path, record=record, entry=entry, arrays=arrays,
                    command_receipt=a.read(command_path), command_receipt_verified=True,
                    command_receipt_sha256=record['files_sha256'][entry['teacher_command_receipt']])
                source, audit = artifact_fitting.interaction_source(case, reference_time_rule=reference,
                    semantic_acquire_rule=semantic)
                cases.append(case)
                sources.append(source)
                case_proofs.append(dict(case_id=cid, source_id=source.source_id,
                    parent_source_group=source.grouping_source, actual_plan_sha256=current['plan']['plan_sha256'],
                    canonical_receipt=a.pointer(ROOT, path), parent_receipt=a.pointer(ROOT, parent),
                    raw_child_receipt=completion['child_receipt'], cache=a.pointer(ROOT, cache),
                    command_receipt=a.pointer(ROOT, command_path), full_arrays_validation=stats,
                    raw_command_count=len(source.command_ledger),
                    original_completion_event_count=len(source.completion_events),
                    excluded_completion_event_count=len(source.excluded_completion_events),
                    teacher_action_rows=int(arrays['teacher_action_valid'].sum()),
                    raw_censored_and_zero_action_sources_retained=True, adapter_audit=audit))
            except (ValueError, KeyError, TypeError, OSError, interaction.InteractionArtifactError) as error:
                errors.append(dict(case_id=cid, error_type=type(error).__name__, error=str(error),
                    original_source_preserved=True))
        row.update(verified_original_teacher_cases=len(cases),
            verified_distinct_original_parents=len({source.grouping_source for source in sources}),
            raw_or_adapter_errors=errors)
        if errors or len(sources) != 100 or row['verified_distinct_original_parents'] != 100:
            row.update(status='not_fitted_original100_raw_or_adapter_audit_failed', real_fit_performed=False)
        else:
            manifest = output/task/'TRAIN_task_teacher_fit_manifest.json'
            write(manifest, dict(schema='r10_actual_complete_task100_CPU_diagnostic_manifest_v1',
                fit_split='train', VAL_used=False, TEST_used=False, diagnostic_only=True,
                complete_task_original100_teacher_coverage=True,
                configuration=a.pointer(ROOT, configuration_path), actual_registrations=a.pointer(ROOT, registrations_path),
                snapshot_inventory=a.pointer(ROOT, inventory_path), source_sha256=source_before,
                entries=[case['entry'] for case in cases], case_proofs=case_proofs,
                full1400_TRAIN_or_formal_policy_release=False))
            row['fit_manifest'] = a.pointer(ROOT, manifest)
            fit_started = time.perf_counter()
            try:
                library = interaction.fit_interaction_library(sources, config, TRAIN_source_manifest_path=manifest)
                diagnostics = library.metadata['completed_skill_fitting']
                library.payload['metadata'].update(diagnostic_only=True,
                    full1400_TRAIN_or_formal_policy_release=False, complete_task_original_teacher_sources=100)
                artifact = output/task/'interaction_diagnostic.json'
                interaction.InteractionLibrary(library.payload).save(artifact)
                fitting = dict(status='actual_complete_task100_CPU_fit_passed_diagnostic_only',
                    artifact=a.pointer(ROOT, artifact), completed_skill_fitting=diagnostics,
                    formal_model_or_policy_release=False)
            except (interaction.InteractionFittingPending, interaction.InteractionArtifactError, ValueError) as error:
                diagnostics = getattr(error, 'diagnostics', {}) or {}
                fitting = dict(status='actual_complete_task100_CPU_fit_pending', error_type=type(error).__name__,
                    error=str(error), completed_skill_fitting=diagnostics,
                    thresholds_or_parent_support_relaxed=False, formal_model_or_policy_release=False)
            fitting['CPU_seconds'] = time.perf_counter()-fit_started
            details = output/task/'real_CPU_fit_report.json'
            write(details, fitting)
            row.update(status=fitting['status'], real_fit_performed=True,
                detailed_fit_report=a.pointer(ROOT, details), phase_fits=phase_summary(diagnostics),
                required_phase_failures=diagnostics.get('required_phase_failures', []),
                failed_fit_keys=diagnostics.get('failed_fit_keys', []))
        tasks[task] = row
        write(output/'progress.json', dict(tasks=tasks, complete=False, native_or_GPU_started=False))
    for path, checksum in files.items():
        a.require(a.file_sha(path) == checksum, 'verified committed diagnostic input changed: '+path)
    report = dict(schema='r10_complete_original100_teacher_tasks_CPU_fit_snapshot_v1',
        started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        actual_registrations=a.pointer(ROOT, registrations_path), actual_registration_count=8,
        configuration=a.pointer(ROOT, configuration_path), inventory=a.pointer(ROOT, inventory_path),
        tasks=tasks, actual_new_task_fits=sum(row.get('real_fit_performed') is True for row in tasks.values()),
        complete_original100_teacher_tasks=sum(not row['missing_teacher_case_ids'] for row in tasks.values()),
        source_sha256=source_before, verified_files_sha256=files,
        TRAIN_only=True, VAL_or_TEST_used=False, partial_tasks_not_fitted=True,
        outcomes_do_not_select_sources=True, raw_masks_events_censor_and_parent_folds_unchanged=True,
        minimum_independent_completed_parents=8, maximum_position_tolerance_m=.03,
        maximum_orientation_tolerance_rad=.15, thresholds_or_parent_support_relaxed=False,
        native_GPU_or_auxiliary_started=False, full_composite_or_Context_view_constructed=False,
        formal_model_or_policy_release=False, wall_seconds=time.perf_counter()-started)
    write(output/'report.json', report)
    write(output/'progress.json', dict(tasks=tasks, complete=True, native_or_GPU_started=False))
    lines = ['Actual complete original100 TRAIN teacher task CPU fitting diagnostic; no native/GPU, no formal release.']
    for task, row in tasks.items():
        lines.append(task+': teachers='+str(row['snapshot_committed_teacher_candidates'])+'/100; '+row['status'])
        for partition, phase in row.get('phase_fits', {}).items():
            if phase['passed'] is not True:
                lines.append('  PENDING '+partition+' '+str(phase))
        if row.get('raw_or_adapter_errors'):
            lines.append('  All original sources retained; no fit because raw/adapter errors are present.')
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    return dict(report=a.pointer(ROOT, output/'report.json'),
        complete_original100_teacher_tasks=report['complete_original100_teacher_tasks'],
        actual_new_task_fits=report['actual_new_task_fits'],
        tasks={task:dict(status=row['status'], phase_fits=row.get('phase_fits'),
            errors=row.get('raw_or_adapter_errors', [])) for task,row in tasks.items()},
        native_or_GPU_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registrations', required=True)
    parser.add_argument('--fit-config', required=True)
    parser.add_argument('--reuse-basketball-report', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.registrations, args.fit_config, args.reuse_basketball_report, args.out), ensure_ascii=False))
