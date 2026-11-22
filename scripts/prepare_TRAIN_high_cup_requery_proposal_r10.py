"""Derive a separate all100-parent high-cup TRAIN requery augmentation proposal.

This creates no execution registration, collector, permission or native worker.
The failed original data and original Context K1 view remain unchanged.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
from our_mani.multitask.train_teacher_requery_hook import TASK, BEHAVIOR
from our_mani.multitask.simulation import development_seed
from our_mani.multitask.requery_acquisition import PROPOSAL_VERSION, RAW_AUDIT_VERSION

BASE = ROOT / 'artifacts/framework/multitask_r10'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ptr(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('proposal inputs/outputs must stay inside our-mani')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def prepare(output):
    output = (ROOT / output).resolve()
    if not output.is_relative_to(BASE) or output.exists():
        raise ValueError('fresh owned proposal namespace required')
    registrations_path = BASE / 'TRAIN_recovery_v20_final_remaining_setup/registrations_with_final_recovery.json'
    registrations = json.loads(registrations_path.read_text())
    original_pointer = registrations[0]['plan']
    original_path = ROOT / original_pointer['path']
    if sha(original_path) != original_pointer['sha256']:
        raise ValueError('actual original2800 plan differs')
    original = json.loads(original_path.read_text())
    lineage_path = BASE / 'TRAIN_full_lineage_release_v6/report.json'
    lineage = json.loads(lineage_path.read_text())
    if not lineage.get('complete') or not lineage.get('audit_passed') or len(original['cases']) != 2800:
        raise ValueError('actual complete original primary acquisition required')
    diagnostic_path = BASE / 'teacher_phase_coverage_actual300_20261018T0747Z/report.json'
    diagnostic = json.loads(diagnostic_path.read_text())['tasks'][TASK]
    if not diagnostic['all100_original_parents_retained']:
        raise ValueError('all100 failed/censored and successful original parents required')
    before = [row for row in original['cases'] if row['task'] == TASK and row['behavior'] == BEHAVIOR]
    if len(before) != 100 or len({row['parent_source_group'] for row in before}) != 100:
        raise ValueError('exact all100 high-cup original teacher parents required')
    proposed = []
    for row in before:
        if (row['split'] != 'train' or row['replica'] != 0 or row['behavior_replica'] != 0
                or row['maximum_collection_ticks'] != 200 or row['teacher_used'] is not True
                or row['context_primary_view'] is not False or row['interaction_eligible'] is not True):
            raise ValueError('original teacher scope differs')
        case = deepcopy(row)
        raw = row['parent_raw_episode_id']
        case.update(replica=1, behavior_replica=1, raw_episode_id=raw + '__' + BEHAVIOR + '_r1')
        case['case_id'] = canonical(dict(split='train', task=TASK, raw=raw, replica=1, behavior=BEHAVIOR))[:16]
        seed, identity = development_seed(task=TASK, source=case['source'], split='train',
                                          seed=original['protocol']['seed'], replica=1)
        proposed.append(dict(original_K1_case_id=row['case_id'], proposed_case=case,
                             proposed_reset=dict(seed=seed, seed_identity=identity,
                                                 variation=case['source']['variation'])))
    if (len({r['proposed_case']['case_id'] for r in proposed}) != 100 or
            {r['proposed_case']['case_id'] for r in proposed} & {c['case_id'] for c in original['cases']}):
        raise ValueError('new identities overlap original cases')
    paths = [Path(__file__).resolve(), ROOT / 'our_mani/multitask/train_teacher_requery.py',
             ROOT / 'our_mani/multitask/requery_acquisition.py',
             ROOT / 'scripts/collect_TRAIN_high_cup_requery_r10.py',
             ROOT / 'our_mani/multitask/train_teacher_requery_hook.py',
             ROOT / 'our_mani/multitask/train_teacher.py',
             ROOT / 'tests/test_multitask_teacher_requery.py',
             ROOT / 'tests/test_multitask_teacher_requery_hook.py',
             ROOT / 'tests/test_multitask_requery_acquisition_v2.py']
    output.mkdir(parents=True)
    archives = []
    for path in paths:
        checksum = sha(path); target = output / 'sources' / checksum / path.name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        if sha(target) != checksum or sha(path) != checksum:
            raise ValueError('proposal source changed during archival')
        archives.append(dict(original_path=str(path.relative_to(ROOT)), **ptr(target)))
    snapshot = output / 'source_snapshot.json'
    snapshot.write_text(json.dumps(dict(schema='r10_unexecuted_high_cup_requery_proposal_sources_v2',
        entries=archives, execution_permission=False), ensure_ascii=False, indent=2) + '\n')
    proposal = dict(schema=PROPOSAL_VERSION, raw_audit_version=RAW_AUDIT_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(), proposal_only=True,
        native_or_GPU_started=False, execution_registration_created=False,
        original_plan=ptr(original_path), complete_primary_lineage=ptr(lineage_path),
        actual_coverage_diagnosis=ptr(diagnostic_path), primary_registrations=ptr(registrations_path),
        source_snapshot=ptr(snapshot), task=TASK, proposed_cases=proposed,
        first_fixed_case_id=proposed[0]['proposed_case']['case_id'],
        augmentation_budget=dict(parent_sources=100, teacher_rollouts=100, replica=1,
            maximum_ticks_per_case=200, maximum_total_ticks=20000, hold_rollouts=0,
            Context_primary_view=False, parent_folds='same original TRAIN parent'),
        requery_configuration=asdict(PlanningRequeryConfig()),
        teacher_policy='same original committed teacher, fresh query after ten actual failed-planning hold ticks; at most20 fresh queries; same evolving episode',
        comparison_to_author='new explicit TRAIN augmentation inspired by periodic replanning; not exact author every10ticks unfinished-path/gripper schedule',
        all100_outcomes_retained=True, outcome_selected=False, infrastructure_retry=False,
        original_primary_cases_receipts_or_raw_labels_modified=False,
        original_Context_primary_view_changed=False,
        preservation=dict(minimum_completed_parent_sources=8,
            maximum_position_tolerance_m=.03, maximum_orientation_tolerance_rad=.15,
            failed_commands_never_marked_completed=True),
        preserved_unexecuted_v1_candidate=dict(
            proposal=ptr(BASE/'TRAIN_high_cup_requery_all100_proposal_v1/proposal.json'),
            plan=ptr(BASE/'TRAIN_high_cup_requery_all100_candidate_v1_setup/plan.json'),
            execution_registration=ptr(BASE/'TRAIN_high_cup_requery_all100_candidate_v1_setup/execution_registration_candidate.json'),
            source_snapshot=ptr(BASE/'TRAIN_high_cup_requery_all100_candidate_v1_setup/source_snapshot.json'),
            audit_counterexamples=ptr(BASE/'TRAIN_high_cup_requery_candidate_v1_independent_peer_v1/report.json')),
        limitations=['Extra TRAIN data budget; does not guarantee sufficient completed align/hang parents.',
                     'Candidate hook CPU tests prove scope/ownership/counting only, not real native operation.',
                     'Requires a separate hash-bound worker, raw hook receipts, actual source closure, and a new typed manifest/consumer before execution.'])
    path = output / 'proposal.json'
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return dict(proposal=ptr(path), source_snapshot=ptr(snapshot), cases=100,
                first_fixed_case_id=proposal['first_fixed_case_id'], native_or_GPU_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), ensure_ascii=False))
