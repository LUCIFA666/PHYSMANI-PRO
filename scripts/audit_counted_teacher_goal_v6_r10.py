"""Real whole100 goal-v6 producer/consumer and 3100 source-budget proof.

Uses the actual complete3100 acquisition audit supplied by the independent
auditor; reconstructs the newly admitted100 sources with production functions.
It does not fit an interaction library or release a policy.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import artifact_fitting as adapter, command_goal_reference as goal
from our_mani.multitask import interaction, release_preflight as release


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(interaction._json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(preparation_path, output):
    began = time.perf_counter()
    out = a.local(ROOT, output)
    a.require(not out.exists(), 'fresh v6 actual100/3100 consumer proof directory required')
    out.mkdir(parents=True)
    inputs = release._Inputs(ROOT)
    prep_path = a.local(ROOT, preparation_path)
    prep = inputs.read(str(prep_path), a.file_sha(prep_path))
    a.require(prep['mode'] == 'final' and prep['production_enabled'] is True
              and prep['actual100_complete_claimed'] is True, 'only the final audited v6 preparation can be consumed')
    config = inputs.read_pointer(prep['configuration'])
    for field in ('fitting_core_source_snapshot', 'fitting_producer_source_snapshot'):
        inputs.source_snapshot(config[field])
    inputs.file(__file__, a.file_sha(__file__))
    raw3100 = inputs.read_pointer(prep['bundle_audit'])
    proof3100 = raw3100.get('audit', raw3100)
    projection = proof3100['acquisition_projection']
    a.require(proof3100['audit_passed'] is True and proof3100['complete'] is True
              and projection['expected'] == projection['completed'] == len(projection['entries']) == 3100
              and projection['teacher_rollouts'] == 1700 and projection['TRAIN_parent_sources'] == 1400
              and projection['original_Context_K1_hold_rollouts'] == 1400
              and projection['primary_Context_view_unchanged'] is True,
              'complete actual3100 source/parent/teacher/hold audit required')
    write(out / 'progress.json', dict(stage='loading_real_v6_whole100_and_aux200_auditors', complete=False))
    rule = goal.load_rule(ROOT, config['command_goal_reference_rule'], purpose='production_fitting')
    inputs.include(rule.verified_files_sha256)
    a.require(len(rule.contexts) == 9 and len(rule.auxiliary_contexts) == 1
              and len(rule.counted_teacher_contexts) == 1, 'exact primary9, basketball1 and counted1 contexts required')
    descriptor = next(iter(rule.counted_teacher_contexts.values()))
    context = descriptor['context']
    manifest = inputs.read_pointer(descriptor['manifest'])
    expected = {row['case_id']: row for row in context['plan']['cases']}
    combined_map = proof3100['per_case_actual_plan_sha256']
    a.require(set(combined_map) == {r['case_id'] for r in projection['case_receipts']}
              and all(combined_map[cid] == context['plan']['plan_sha256'] for cid in expected),
              'actual3100 proof omits or substitutes a counted source plan')
    cases, rows, per_source, originals = [], [], {}, {}
    for row in manifest['case_receipts']:
        path = inputs.file(row['path'], row['sha256'])
        record = a.read(path)
        entry = record['entry']
        cid = record['case_id']
        a.require(cid in expected and entry['parent_source_group'] == expected[cid]['parent_source_group'],
                  'actual counted case or original parent differs')
        cache = inputs.file(entry['cache'], record['files_sha256'][entry['cache']])
        command_path = inputs.file(entry['teacher_command_receipt'], record['files_sha256'][entry['teacher_command_receipt']])
        with np.load(cache, allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        case = dict(path=path, record=record, entry=entry, arrays=arrays, command_receipt=a.read(command_path),
                    command_receipt_verified=True, command_receipt_sha256=a.file_sha(command_path))
        original = adapter.interaction_source(case)[0]
        source, audit = adapter.interaction_source(case, command_goal_rule=rule)
        sid = source.source_id
        a.require(sid not in per_source, 'counted source duplicate')
        evidence = source.command_goal_reference_evidence
        p = goal.evidence_proof(evidence)
        a.require(p['counted_teacher_acquisition']['whole100_complete'] is True
                  and p['counted_teacher_acquisition']['registration'] == descriptor['pointer']
                  and p['counted_teacher_acquisition']['is_infrastructure_recovery'] is False
                  and p['counted_teacher_acquisition']['is_basketball_teacher_supplement'] is False,
                  'counted execution proof aliases another data kind')
        a.require(source.references == original.references and source.command_ledger == original.command_ledger
                  and np.array_equal(source.command_ids, original.command_ids)
                  and len(source.completion_events) == len(original.completion_events)
                  and all(x.command_completed == y.command_completed
                      and np.array_equal(x.terminal_reference_xyzw, y.terminal_reference_xyzw)
                      and np.array_equal(x.terminal_tcp_xyzw, y.terminal_tcp_xyzw)
                      for x, y in zip(source.completion_events, original.completion_events))
                  and p['original_masks_sha256'] == goal.fingerprint(arrays),
                  'derived context changed original raw command/event/metric/mask semantics')
        per_source[sid] = p
        originals[sid] = original
        inputs.include(audit['command_goal_reference_derivation']['verified_files_sha256'])
        rows.append(dict(case_id=cid, source_id=sid, parent_source_group=source.grouping_source,
                         canonical_receipt=a.pointer(ROOT, path), cache=a.pointer(ROOT, cache),
                         command_receipt=a.pointer(ROOT, command_path),
                         references=len(evidence.references), pending_commands=len(evidence.pending_commands),
                         goal_proof=p))
        cases.append(case)
        if len(cases) % 10 == 0:
            write(out / 'progress.json', dict(stage='actual100_original_raw_goal_derivation', completed=len(cases), expected=100))
    a.require(len(cases) == len(per_source) == len({row['parent_source_group'] for row in rows}) == 100,
              'whole100 source and independent parent coverage required')
    metadata = dict(rule=rule.pointer, helper=dict(path=str(Path(goal.__file__).resolve()), sha256=a.file_sha(goal.__file__)),
                    derivation_schema=goal.SCHEMA, original_raw_metric_events_and_masks_preserved=True,
                    per_source=per_source, verified_files_sha256=rule.verified_files_sha256)
    frozen = dict(schema='r10_actual100_counted_goal_v6_prefit_evidence_v1', fit_split='train', VAL_used=False, TEST_used=False,
                  fitting_configuration=prep['configuration'], entries=[case['entry'] for case in cases], case_proofs=rows,
                  command_goal_reference_derivation=metadata, all_failed_censored_sources_retained=True,
                  fitting_or_policy_release_claimed=False)
    proof_path = out / 'TRAIN_counted100_goal_derivation_before_fit.json'
    write(proof_path, frozen)
    frozen_sha = a.file_sha(proof_path)
    write(out / 'progress.json', dict(stage='real_production_consumer_reconstructing100', completed=100, expected=100))
    consumed_rule, sources = release._command_goal_sources(inputs, config, frozen, cases, None, {})
    a.require(len(sources) == 100 and consumed_rule.pointer == rule.pointer
              and {sid: goal.evidence_proof(s.command_goal_reference_evidence) for sid, s in sources.items()} == per_source,
              'production consumer differs from frozen actual100 producer evidence')
    negatives = []
    def rejects(name, callback):
        try:
            callback()
        except (ValueError, KeyError, TypeError):
            negatives.append(name)
            return
        raise ValueError('negative unexpectedly accepted: ' + name)
    # Metadata validation is a real production consumer.  No fake fit/library
    # is promoted: this is only the exact per-source derivation contract.
    library_meta = dict(command_goal_reference_derivation=dict(schema=goal.SCHEMA,
        rule_sha256_values=[rule.sha256], helper_sha256=a.file_sha(goal.__file__),
        original_raw_completion_metric_preserved=True, per_source=per_source), public_completion_verifier_calibrated=False)
    release._command_goal_library_metadata(library_meta, list(sources.values()), consumed_rule)
    first = next(iter(per_source))
    for mutation in ('missing_source', 'counted_kind', 'manifest_SHA', 'raw_masks', 'rule_SHA', 'calibrated'):
        changed = deepcopy(library_meta)
        body = changed['command_goal_reference_derivation']
        if mutation == 'missing_source':
            body['per_source'].pop(first)
        elif mutation == 'counted_kind':
            body['per_source'][first]['counted_teacher_acquisition']['is_infrastructure_recovery'] = True
        elif mutation == 'manifest_SHA':
            body['per_source'][first]['counted_teacher_acquisition']['manifest']['sha256'] = 'f' * 64
        elif mutation == 'raw_masks':
            body['per_source'][first]['original_masks_sha256']['valid'] = 'f' * 64
        elif mutation == 'rule_SHA':
            body['rule_sha256_values'] = ['f' * 64]
        else:
            changed['public_completion_verifier_calibrated'] = True
        rejects(mutation, lambda changed=changed: release._command_goal_library_metadata(changed, list(sources.values()), consumed_rule))
    original_only = goal.load_rule(ROOT, a.pointer(ROOT, ROOT / 'artifacts/framework/multitask_r10/command_goal_reference_rule_v5_translation_revision3_setup/command_goal_rule_original_only.json'), purpose='production_fitting')
    rejects('old_v5_unknown_counted_plan', lambda: goal.verify_original_source(cases[0], originals[first], original_only))
    changed_case = deepcopy(cases[0])
    actual_terminal_tick = changed_case['record']['alignment']['native_terminal_tick']
    changed_case['record']['alignment']['native_terminal_tick'] = 1 if actual_terminal_tick is None else actual_terminal_tick + 1
    rejects('changed_inmemory_canonical_json', lambda: adapter.interaction_source(changed_case, command_goal_rule=rule))
    a.require(a.file_sha(proof_path) == frozen_sha, 'actual100 prefit evidence changed')
    inputs.rehash_all()
    result = dict(schema='r10_actual_counted100_goal_v6_3100_source_budget_consumer_v1',
        preparation=a.pointer(ROOT, prep_path), configuration=prep['configuration'], rule=rule.pointer,
        complete3100_acquisition_audit=prep['bundle_audit'], complete100_acquisition_audit=prep['complete100_audit'],
        actual3100_acquisition_proof_reused_with_actual_identity_not_rewritten=True,
        newly_admitted_counted_sources_reconstructed=100, counted_parent_groups=100,
        complete3100_budget=dict(total=3100, primary=2800, basketball_teacher200=200, counted_high_cup_teacher100=100,
                                independent_parents=1400, teacher=1700, Context_K1_hold=1400),
        per_source_goal_proof=a.pointer(ROOT, proof_path), producer_and_real_consumer_match=True,
        raw_commands_events_masks_and_censoring_preserved=True, lower_auditor_or_consumer_mocked=False,
        negative_boundaries=negatives, final_direct_all_input_rehash_passed=True,
        verified_files_sha256=inputs.files, wall_seconds=time.perf_counter() - began,
        full3100_interaction_fit_started=False, native_GPU_started=False, formal_policy_VAL_TEST_or_SR_release=False)
    write(out / 'report.json', result)
    write(out / 'progress.json', dict(stage='actual_counted100_goal_v6_consumer_complete', complete=True))
    return a.pointer(ROOT, out / 'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.preparation, args.out)))
