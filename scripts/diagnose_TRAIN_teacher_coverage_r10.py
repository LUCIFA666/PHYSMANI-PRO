"""Read actual TRAIN ledgers to distinguish planning failure from truncation.

This diagnostic compares the fixed native teachers with the small arrays
already extracted from the corresponding original TRAIN demonstrations. Stage
presence in those recordings is never treated as completed skill evidence.
No simulator, GPU, VAL or TEST data is used and no release gate is changed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as audit


def run(prior_report, output):
    output = audit.local(ROOT, output)
    audit.require(not output.exists(), 'fresh teacher coverage diagnostic required')
    files = {}
    prior_pointer = audit.pointer(ROOT, audit.local(ROOT, prior_report))
    prior = audit.read(audit.checked_pointer(ROOT, prior_pointer, files))
    tasks = {}
    selected = ('place_cups_on_rotating_frame',
                'place_cups_on_rotating_frame_high_speed',
                'insert_onto_rotating_peg_high_speed')
    for task in selected:
        manifest_pointer = prior['tasks'][task]['fit_manifest']
        manifest = audit.read(audit.checked_pointer(ROOT, manifest_pointer, files))
        rows = manifest['case_proofs']
        audit.require(len(rows) == 100 and len({r['parent_source_group'] for r in rows}) == 100,
                      'each diagnostic requires all100 distinct original TRAIN parents')
        phase_counts = defaultdict(Counter)
        phase_parents = defaultdict(lambda: defaultdict(set))
        endings = Counter()
        original_stage_parents = defaultdict(set)
        parents = []
        for row in rows:
            record = audit.read(audit.checked_pointer(ROOT, row['canonical_receipt'], files))
            ledger = audit.read(audit.checked_pointer(ROOT, row['command_receipt'], files))
            entry = record['entry']
            group = row['parent_source_group']
            audit.require(entry['task'] == task and entry['parent_source_group'] == group
                and entry['split'] == 'train' and record['teacher_actions_used'] is True
                and record['files_sha256'][row['command_receipt']['path']] == row['command_receipt']['sha256'],
                'fixed source or raw command ledger binding differs')
            original = entry['source_demonstration']
            original_path = audit.checked_pointer(ROOT, original, files)
            with np.load(original_path, allow_pickle=False) as arrays:
                stages = tuple(map(str, arrays['expert_stage']))
                frames = len(arrays['tcp_pose'])
            audit.require(frames == original['frames'], 'original TRAIN cached frame count differs')
            for stage in set(stages):
                original_stage_parents[stage].add(group)
            commands = ledger['commands']
            for command in commands:
                status = ('completed_world_waypoint' if 'command_completed_at' in command else
                          'planning_failure' if 'planning_failure' in command else 'incomplete')
                phase = command['raw_phase']
                phase_counts[phase][status] += 1
                phase_parents[phase][status].add(group)
            last = commands[-1] if commands else {}
            ending = ('planning_failure' if 'planning_failure' in last else
                      last.get('cutoff_reason', 'completed_world_waypoint' if
                               'command_completed_at' in last else 'unclassified'))
            endings[(last.get('raw_phase', 'no_command'), ending)] += 1
            parents.append(dict(
                parent_source_group=group,
                canonical_receipt=row['canonical_receipt'],
                command_receipt=row['command_receipt'],
                original_TRAIN_cached_recording=original,
                original_recorded_frames=frames,
                original_stages=sorted(set(stages)),
                native_ticks=record['alignment']['clock']['ticks'],
                native_success_latched=record['alignment']['native_success_latched'],
                last_command={k: v for k, v in last.items() if k in (
                    'command_id', 'generated_at', 'raw_stage', 'raw_phase',
                    'executed_tick_count', 'command_completed_at', 'planning_failure',
                    'cutoff_reason', 'cutoff_at', 'measured_arrival_at', 'gripper_completed_at')},
                ending=ending))
        tasks[task] = dict(
            all100_original_parents_retained=True,
            source_manifest=manifest_pointer,
            raw_command_counts_by_phase={k: dict(v) for k, v in phase_counts.items()},
            distinct_parent_counts_by_phase={phase: {status: len(groups)
                for status, groups in values.items()} for phase, values in phase_parents.items()},
            terminal_command_reasons=[dict(phase=phase, reason=reason, parents=count)
                for (phase, reason), count in sorted(endings.items())],
            original_recorded_stage_parent_counts={stage: len(groups)
                for stage, groups in sorted(original_stage_parents.items())},
            original_stage_presence_is_completed_skill_evidence=False,
            parents=parents)
    follower = ROOT / 'our_mani/multitask/train_teacher.py'
    files[str(follower)] = audit.file_sha(follower)
    script = Path(__file__).resolve()
    files[str(script)] = audit.file_sha(script)
    audit.require(all(audit.file_sha(name) == checksum for name, checksum in files.items()),
                  'teacher coverage diagnostic inputs changed')
    report = dict(
        schema='r10_actual_TRAIN_teacher_failure_and_recorded_stage_coverage_diagnostic_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        prior_production_fit=prior_pointer,
        tasks=tasks,
        verified_files_sha256=files,
        interpretation={
            'raw_completion': 'Measured committed world-waypoint completion; not native phase or task success.',
            'planning_failure': 'Current follower enters failed_no_resample and holds for remaining ticks; extending time alone cannot retry or repair it.',
            'recorded_stage': 'Original TRAIN recording contains the expert stage; no completion event, persistent identity or reference pose is inferred from presence.',
            'next_work': 'Inspect actual planning goals and recorded public RGBD support before authorizing a new fixed teacher budget; do not repeat the same failed planner or fabricate hang labels.'},
        native_or_GPU_started=False,
        VAL_or_TEST_used=False,
        full_policy_success_or_model_release_claimed=False,
        thresholds_masks_raw_cases_or_protocol_changed=False)
    output.mkdir(parents=True)
    path = output / 'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return dict(report=audit.pointer(ROOT, path), tasks={task: {
        'terminal_command_reasons': row['terminal_command_reasons'],
        'original_recorded_stage_parent_counts': row['original_recorded_stage_parent_counts']}
        for task, row in tasks.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-report', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.prior_report, args.out), ensure_ascii=False))
