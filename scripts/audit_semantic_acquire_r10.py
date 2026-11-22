"""CPU-only original K1 basketball200 semantic acquire draft audit."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask import semantic_acquire as semantic
from our_mani.multitask.training_view import validate_arrays


def audit(registrations, output):
    output = lineage.local(ROOT, output)
    lineage.require(not output.exists(), 'fresh semantic audit output required')
    module_source = lineage.pointer(ROOT, Path(semantic.__file__))
    script_source = lineage.pointer(ROOT, Path(__file__))
    files = {}; contexts = []; by_sha = {}
    regpath = lineage.local(ROOT, registrations)
    registrations = lineage.read(lineage.checked_pointer(ROOT, lineage.pointer(ROOT, regpath), files))
    for row in registrations:
        context = lineage.load_archived_plan(ROOT, row, files)
        key = context['plan']['plan_sha256']
        lineage.require(key not in by_sha, 'duplicate actual plan registration')
        if 'predecessor_plan_sha256' in row:
            lineage.verify_recovery_transition(ROOT, by_sha[row['predecessor_plan_sha256']], context, files)
        by_sha[key] = context; contexts.append(context)
    original = contexts[0]
    expected = {case['case_id']: case for case in original['plan']['cases']
                if case['task'] in semantic.TASKS and case['teacher_used'] is True}
    lineage.require(len(expected) == 200 and all(case['behavior_replica'] == 0 for case in expected.values()),
                    'only fixed original K1 two-task teacher200 accepted')
    derived = []; negative_checks = []
    for cid, case in sorted(expected.items()):
        found = [(context, context['collection']/'cases/train'/case['task']/cid/'case.json')
                 for context in contexts
                 if (context['collection']/'cases/train'/case['task']/cid/'case.json').is_file()]
        lineage.require(len(found) == 1, 'every original teacher must have one actual canonical completion: '+cid)
        context, path = found[0]; record = lineage.read(path)
        output_path = Path(lineage.command_argument(record['parent_process_receipt']['command'], '--output')).resolve()
        row = dict(canonical_receipt=lineage.pointer(ROOT, path),
                   parent_receipt=lineage.pointer(ROOT, output_path.parent/'parent_receipt.json'))
        completion = lineage.verify_completed(ROOT, row, context, case, files)
        previous = None; preceding = []
        if 'recovery_input' in context['plan']['protocol']:
            previous = by_sha[context['registration']['predecessor_plan_sha256']]
            payload = lineage.read(lineage.checked_pointer(ROOT, context['plan']['protocol']['recovery_input'], files))
            preceding = [dict(actual_plan_sha256=previous['plan']['plan_sha256'], parent_receipt=ptr)
                         for ptr in payload['failed_parent_receipts']
                         if lineage.read(lineage.local(ROOT, ptr['path']))['case_id'] == cid]
            lineage.verify_recovery_execution(ROOT, completion, context, previous, preceding, files)
        entry = record['entry']; cache = lineage.local(ROOT, entry['cache'])
        command_path = lineage.local(ROOT, entry['teacher_command_receipt'])
        lineage.require(entry['teacher_command_receipt'] == entry['phase_completion_events_receipt']
                        and record['files_sha256'][entry['teacher_command_receipt']] == lineage.file_sha(command_path),
                        'raw command/completion ledger changed')
        with np.load(cache, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        validate_arrays(arrays, entry, record)
        result = semantic.derive_acquire(entry, arrays, lineage.read(command_path),
            raw_bindings=dict(canonical_receipt=lineage.pointer(ROOT, path),
                cache=lineage.pointer(ROOT, cache), commands=lineage.pointer(ROOT, command_path)))
        result['actual_plan_sha256'] = context['plan']['plan_sha256']
        result['original_source_archive'] = context['registration']['source_snapshot']
        semantic.verify_derived_receipt(ROOT, result, registered=context, expected_case=case,
                                        predecessor=previous, preceding=preceding)
        if result['physical_acquire_evidence'] and not negative_checks:
            for mutation in ('frames', 'mask_fingerprint', 'parent', 'actual_plan'):
                bad = deepcopy(result)
                if mutation == 'frames': bad['mode_segments'][1]['frame_indices'] = (0,)
                elif mutation == 'mask_fingerprint': bad['original_masks_sha256']['target_valid'] = '0'*64
                elif mutation == 'parent': bad['parent_source_group'] += ':unbound'
                else: bad['actual_plan_sha256'] = '0'*64
                try:
                    semantic.verify_derived_receipt(ROOT, bad, registered=context, expected_case=case,
                                                    predecessor=previous, preceding=preceding)
                except ValueError:
                    negative_checks.append(dict(mutation=mutation, actual_raw_receipt_unchanged=True, rejected=True))
                else:
                    raise ValueError('altered actual-derived receipt was accepted: '+mutation)
        derived.append(result)
    report = dict(schema='r10_original_K1_200_semantic_acquire_CPU_audit_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), actual_registration_list=lineage.pointer(ROOT, regpath),
        verified_original_teacher_acquisitions=200, all_original_parents_and_raw_events_preserved=True,
        original_full2800_or_Context_view_changed=False, auxiliary_teacher_replicas_included=False,
        summary=semantic.summarize(derived), per_source=derived,
        existing_fitter_interface=semantic.legacy_fitter_interface(),
        actual200_derivation_recompute_roundtrip_passed=True, actual_raw_negative_checks=negative_checks,
        derivation_module=module_source, audit_script=script_source,
        verified_files_sha256=files, native_GPU_or_fitting_started=False,
        released_semantic_or_interaction_model=False)
    lineage.require(module_source == lineage.pointer(ROOT, Path(semantic.__file__))
                    and script_source == lineage.pointer(ROOT, Path(__file__)),
                    'semantic diagnostic source changed during execution')
    output.mkdir(parents=True)
    p = output/'report.json'; p.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return dict(report=lineage.pointer(ROOT, p), verified_original_teachers=200,
        summary={task:dict(statuses=value['statuses'], external_parents=len(value['external_parent_sources']),
            attached_parents=len(value['attached_parent_sources'])) for task, value in report['summary'].items()},
        existing_fitter_compatible=False, fitting_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registrations', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.registrations, args.out), ensure_ascii=False))
