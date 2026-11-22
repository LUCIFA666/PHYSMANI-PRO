"""Freeze an unapproved, separate all-parent TRAIN teacher supplement draft.

The primary 2800-case K1 acquisition and Context view are unchanged. This
script creates no native worker or execution permission. A separately reviewed
wrapper/consumer is required before an executable registration may be made.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import collect_multitask_training_r10 as collection
from our_mani.multitask.simulation import development_seed

SCHEMA = 'r10_fixed_all_parent_TRAIN_teacher_supplement_draft_v1'
TASKS = ('moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed')
BEHAVIOR = 'TRAIN_teacher_committed_waypoint'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def owned(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT), 'supplement metadata must stay inside our-mani')
    return path


def pointer(path):
    path = owned(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def checked(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'exact pointer required')
    path = owned(value['path'])
    require(path.is_file() and sha(path) == value['sha256'], 'supplement input SHA differs')
    return path


def expected_cases(original, proposal):
    """Derive every replica1 case from every original parent, without outcomes."""
    require(len(original['cases']) == 2800 and original['split'] == 'train',
            'the complete original K1 roster is required')
    parents = [case for case in original['cases']
               if case['task'] in TASKS and case.get('behavior') == BEHAVIOR]
    require(len(parents) == 200 and all(sum(case['task'] == task for case in parents) == 100
                                      for task in TASKS), 'both complete100-parent tasks are required')
    require(len({(case['task'], case['parent_raw_episode_id']) for case in parents}) == 200,
            'original parent uniqueness differs')
    expected = []
    for before in parents:
        require(before['replica'] == before['behavior_replica'] == 0
                and before['maximum_collection_ticks'] == 200 and before['teacher_used'] is True
                and before['context_primary_view'] is False and before['interaction_eligible'] is True,
                'the original teacher contract differs')
        after = deepcopy(before)
        raw = before['parent_raw_episode_id']
        after.update(replica=1, behavior_replica=1, raw_episode_id=raw + '__' + BEHAVIOR + '_r1')
        after['case_id'] = collection.canonical_sha(dict(split='train', task=before['task'],
            raw=raw, replica=1, behavior=BEHAVIOR))[:16]
        seed, identity = development_seed(task=after['task'], source=after['source'], split='train',
            seed=original['protocol']['seed'], replica=1)
        expected.append(dict(original_K1_case_id=before['case_id'], proposed_case=after,
            proposed_reset=dict(seed=seed, seed_identity=identity, variation=after['source']['variation'])))
    require(expected == proposal['proposed_cases'], 'proposal cases/seed/source/budget changed from original roster')
    require(len({row['proposed_case']['case_id'] for row in expected}) == 200
            and not ({row['proposed_case']['case_id'] for row in expected}
                     & {case['case_id'] for case in original['cases']}), 'supplement identities overlap original K1')
    return expected


def prepare(proposal_path, output, collection_output):
    proposal_path, output, collection_output = map(owned, (proposal_path, output, collection_output))
    require(not output.exists() and not collection_output.exists(), 'fresh draft and future collection required')
    proposal = read(proposal_path)
    require(proposal.get('schema') == 'r10_fixed_all100_basketball_teacher_replica1_proposal_v1'
            and proposal.get('proposal_only') is True and proposal.get('native_or_GPU_launched') is False,
            'the fixed TRAIN-only proposal is required')
    original_path = checked(proposal['original_plan'])
    original = collection.verify_plan(original_path)
    reviewed_path = checked(proposal['shared_profile_reference_plan'])
    reviewed = collection.verify_plan(reviewed_path)
    require(reviewed['cases'] == original['cases'], 'shared-profile plan changed original K1 roster')
    checked(proposal['source_audit'])
    cases = expected_cases(original, proposal)
    require(proposal['preservation']['minimum_completed_distinct_parents'] == 8
            and proposal['preservation']['maximum_position_tolerance_m'] == .03
            and proposal['preservation']['maximum_angle_tolerance_rad'] == .15,
            'the original parent support/terminal caps remain required')
    require(reviewed['protocol']['development_reset_profile'] == proposal['shared_development_reset_profile']
            and reviewed['protocol']['RML_factory_hook'] == proposal['shared_RML_factory_hook'],
            'use exactly the accepted shared physical initialization and owned RML hook')
    protocol = deepcopy(reviewed['protocol'])
    protocol.pop('recovery_input', None)
    protocol.update(draft=True, native_execution_authorized=False)
    output.mkdir(parents=True)
    protocol_path = output / 'protocol_draft.json'
    collection.write_json(protocol_path, protocol)
    inputs = dict(reviewed['inputs'])
    for path in (proposal_path, original_path, reviewed_path, protocol_path, Path(__file__).resolve()):
        inputs[str(path)] = sha(path)
    plan = deepcopy(reviewed)
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), cases=[row['proposed_case'] for row in cases],
        protocol=protocol, protocol_path=str(protocol_path), inputs=inputs,
        auxiliary_teacher_scope=SCHEMA, primary_K1_plan=pointer(original_path),
        primary_K1_roster_unmodified=True, primary_Context_hold_view_unmodified=True,
        native_execution_ready=False, no_native_or_GPU_work_performed=True,
        augmentation_budget=dict(version='r10_separate_fixed_TRAIN_teacher_supplement_budget_v1',
            parents=200, parents_per_task=100, extra_teacher_replicas_per_parent=1,
            replica_indices=[1], expected_cases=200, maximum_ticks_per_case=200,
            maximum_total_ticks=40000, extra_hold_cases=0, context_primary_view=False,
            original_primary_budget_unchanged=True),
        supplement_proposal=pointer(proposal_path))
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = collection.canonical_sha(plan)
    plan_path = output / 'execution_plan_draft.json'
    collection.write_json(plan_path, plan)
    collection.verify_plan(plan_path)
    archive_rows = []
    for name, checksum in sorted(inputs.items()):
        source = Path(name)
        if source.suffix not in ('.py', '.sh'):
            continue
        raw = source.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == checksum, 'source changed before archival')
        target = output / 'sources' / checksum / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        archive_rows.append(dict(original_path=str(source), **pointer(target)))
    collection.verify_plan(plan_path)
    archive_path = output / 'source_archive.json'
    collection.write_json(archive_path, dict(schema='r10_unapproved_teacher_supplement_source_archive_v1',
        entries=archive_rows, all_inputs_verified_before_and_after=True, native_started=False))
    report = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        approved_for_execution=False, production_worker_bound=False, production_consumer_bound=False,
        plan=pointer(plan_path), source_archive=pointer(archive_path), proposal=pointer(proposal_path),
        primary_plan=pointer(original_path), fixed_cases=200, fixed_parents=200,
        tasks=list(TASKS), replicas=[1], maximum_extra_ticks=40000,
        collection=str(collection_output.relative_to(ROOT)),
        all_original_cases_preserved=True, outcome_selection=False, automatic_retry=False,
        original_Context_primary_view_changed=False, strict_primary_full2800_gate_changed=False,
        parent_grouping_unchanged=True, min8_position03_angle015_gates_changed=False,
        native_started=False, GPU_started=False, source_files_modified=False,
        required_before_execution=['A separately hash-bound native wrapper and supplement-only consumer',
            'Independent CPU scope/identity/receipt/seed tests and ROOT review',
            'A fresh approved plan/registration; this draft explicitly refuses native execution'],
        additional_data_guarantees_model_support=False, is_recovery_of_old_failed_cases=False)
    collection.write_json(output / 'preparation_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.proposal, args.out, args.collection), ensure_ascii=False))
