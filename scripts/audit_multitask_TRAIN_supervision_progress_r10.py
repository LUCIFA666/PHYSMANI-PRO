"""Read-only completed TRAIN supervision inventory; never a fitting release.

Snapshot only canonical cases observed at start. New concurrent completions
are deliberately deferred to the next snapshot. No policy/VAL/TEST outcomes
are read, and no failed, censored or zero-command teacher is replaced.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.specs import TASK_SPECS, get_task_spec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('TRAIN diagnostic input escaped desktop project')
    return path


def inventory(collections):
    paths = [path for collection in collections
             for path in sorted(inside(collection).glob('cases/train/*/*/case.json'))]
    seen, inputs, details, errors = set(), {}, [], []
    phase_parents, partition_parents = defaultdict(set), defaultdict(set)
    counts, windows = defaultdict(Counter), defaultdict(Counter)
    for path in paths:
        try:
            record = json.loads(path.read_text())
            entry = record['entry']
            cid, task = record['case_id'], entry['task']
            if cid in seen or record.get('status') != 'completed' or entry['split'] != 'train':
                raise ValueError('duplicate or noncompleted TRAIN case')
            if task not in TASK_SPECS:
                raise ValueError('unknown task')
            seen.add(cid)
            process = record['parent_process_receipt']
            child = inside(process['child_case_path'])
            if sha(child) != process['child_case_sha256'] or json.loads(child.read_text()) != {
                    key: value for key, value in record.items() if key != 'parent_process_receipt'}:
                raise ValueError('canonical/raw child differs')
            attempt = next(parent for parent in child.parents if parent.name == 'output').parent
            parent_path = attempt / 'parent_receipt.json'
            parent = json.loads(parent_path.read_text())
            if (parent.get('status') != 'completed' or parent.get('returncode') != 0
                    or parent.get('process_id') != process['process_id']
                    or parent.get('case_id') != cid or parent.get('command') != process['command']):
                raise ValueError('parent not committed; defer this observation')
            inputs[str(path.relative_to(ROOT))] = sha(path)
            inputs[str(parent_path.relative_to(ROOT))] = sha(parent_path)
            for name, checksum in record['files_sha256'].items():
                if sha(inside(name)) != checksum:
                    raise ValueError('changed TRAIN output: ' + name)
            teacher = entry['teacher_used']
            counts[task]['teacher' if teacher else 'hold'] += 1
            row = dict(case_id=cid, task=task, parent_source_group=entry['parent_source_group'],
                       actual_plan_sha256=record['plan_sha256'], teacher=teacher)
            if not teacher:
                eligibility = entry['context_window_eligibility']
                for key in ('usable_context_windows', 'full_horizon_windows'):
                    windows[task][key] += eligibility[key]
                counts[task]['hold_with_usable_windows'] += eligibility['usable_context_windows'] > 0
                counts[task]['hold_with_full_horizon'] += eligibility['full_horizon_windows'] > 0
                row['context_window_eligibility'] = eligibility
            else:
                with np.load(inside(entry['cache']), allow_pickle=False) as stored:
                    arrays = {name: stored[name].copy() for name in stored.files}
                command_path = entry['teacher_command_receipt']
                checksum = record['files_sha256'][command_path]
                case = dict(record=record, entry=entry, arrays=arrays,
                            command_receipt=json.loads(inside(command_path).read_text()),
                            command_receipt_sha256=checksum, command_receipt_verified=True)
                source, audit = interaction_source(case)
                commands = int(np.count_nonzero(arrays['teacher_action_valid']))
                counts[task]['actual_teacher_action_rows'] += commands
                counts[task]['zero_action_teacher'] += commands == 0
                counts[task]['teacher_with_completion_events'] += bool(source.completion_events)
                row.update(actual_teacher_action_rows=commands,
                           phase_completion_events=len(source.completion_events), adapter_audit=audit)
                for event in source.completion_events:
                    binding = event.binding
                    phase_parents[(task, event.phase)].add(source.grouping_source)
                    partition = (task, event.phase, binding.anchor_id, binding.reference_role,
                                 binding.reference_kind, binding.manipulated_role, event.reference_mode)
                    partition_parents[partition].add(source.grouping_source)
            details.append(row)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            errors.append(dict(path=str(path.relative_to(ROOT)), error=repr(error)))
    by_task = {}
    for task in TASK_SPECS:
        spec = get_task_spec(task)
        by_task[task] = dict(counts[task], context_windows=dict(windows[task]),
            phases={phase.name: dict(completed_parent_count=len(phase_parents[(task, phase.name)]),
                                    optional_after_native_completion=phase.optional_after_native_completion)
                    for phase in spec.phase_skills})
    return dict(schema='r10_read_only_TRAIN_supervision_progress_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), collections=collections,
        snapshot_canonical_count=len(paths), verified_cases=len(details), per_task=by_task,
        completed_event_partitions=[dict(task=key[0], phase=key[1], anchor_id=key[2],
            reference_role=key[3], reference_kind=key[4], manipulated_role=key[5],
            reference_mode=key[6], parent_count=len(value))
            for key, value in sorted(partition_parents.items())],
        case_details=details, errors=errors, canonical_parent_inputs_sha256=inputs,
        fitting_module_sha256=sha(ROOT / 'our_mani/multitask/artifact_fitting.py'),
        diagnostic_script_sha256=sha(__file__),
        partial_TRAIN_inventory_only=True, complete_lineage_or_phase_curve_fit_proven=False,
        minimum_eight_parent_support_not_relaxed=True, held_out_splits_read=False,
        fit_or_optimizer_started=False, policy_success_rate_measured=False,
        failed_censored_and_zero_action_teachers_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = inside(args.out)
    if output.exists():
        raise ValueError('fresh output required; preserve prior diagnostic')
    report = inventory(args.collection)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified_cases=report['verified_cases'], errors=report['errors'],
                         per_task=report['per_task']), ensure_ascii=False))


if __name__ == '__main__':
    main()
