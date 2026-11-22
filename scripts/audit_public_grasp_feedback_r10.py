"""Read-only diagnostic of recorded public grasp feedback, never a release.

Snapshot completed TRAIN teacher receipts only. This does not launch native
simulation, change observations, select successful tasks, or fit a policy.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/multitask_r10'
COLLECTIONS = ('TRAIN_full_collection_v11', 'TRAIN_recovery_v15_initial_pair',
               'TRAIN_recovery_v16_known_failures',
               'TRAIN_recovery_v17_additional_known_failures')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def run(output):
    output = (ROOT / output).resolve()
    require(output.is_relative_to(BASE) and not output.exists(), 'fresh R10 output required')
    snapshot = [(name, path) for name in COLLECTIONS
                for path in sorted((BASE / name / 'cases').rglob('case.json'))]
    counts = defaultdict(Counter)
    entries = []
    seen = set()
    for collection, path in snapshot:
        raw_bytes = path.read_bytes()
        record = json.loads(raw_bytes)
        entry = record['entry']
        if entry.get('source_kind') != 'TRAIN_native_generated_augmentation':
            continue
        require(record.get('status') == 'completed' and record.get('split') == 'train',
                'only completed TRAIN acquisitions are diagnostic inputs')
        require(record['case_id'] not in seen, 'duplicate accepted original case')
        seen.add(record['case_id'])
        parent = record['parent_process_receipt']
        child_path = ROOT / parent['child_case_path']
        require(sha(child_path) == parent['child_case_sha256'] and
                json.loads(child_path.read_text()) ==
                {k: v for k, v in record.items() if k != 'parent_process_receipt'},
                'raw child differs from canonical acquisition')
        cache = ROOT / entry['cache']
        require(sha(cache) == entry['cache_sha256'] == record['files_sha256'][entry['cache']],
                'actual cache differs from raw collector SHA')
        with np.load(cache, allow_pickle=False) as data:
            t = data['times'].copy()
            contact = data['robot_contact'].copy()
            opened = data['robot_gripper_open'].copy()
            amount = data['gripper_open_amount'].copy()
            applied = data['teacher_applied_opening'].copy()
            action_valid = data['teacher_action_valid'].copy()
        n = len(t)
        require(contact.shape == opened.shape == applied.shape == action_valid.shape == (n,)
                and contact.dtype == opened.dtype == action_valid.dtype == np.bool_
                and amount.shape == (n, 2) and np.isfinite(amount).all(),
                'public feedback arrays are malformed')
        # Each observation precedes that row's command. Use only prior commands
        # when determining whether a close edge has already been issued.
        prior_close = np.r_[False, np.maximum.accumulate(
            (applied[:-1] < .5) & action_valid[:-1])]
        conflict = contact & opened
        observed_mean = np.mean(amount, axis=1)
        require(np.array_equal(opened, observed_mean > .5),
                'diagnostic source no longer uses the reviewed public mean > .5 definition')
        task_counts = counts[entry['task']]
        task_counts.update(teachers=1, frames=n, public_grasp_frames=int(contact.sum()),
                           public_grasp_and_open_frames=int(conflict.sum()),
                           public_grasp_and_open_parents=int(conflict.any()),
                           conflict_after_prior_issued_close_frames=int((conflict & prior_close).sum()))
        entries.append(dict(collection=collection, case_id=record['case_id'],
            source_group=entry['parent_source_group'], task=entry['task'],
            receipt_path=str(path.relative_to(ROOT)),
            receipt_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            cache_path=entry['cache'], cache_sha256=entry['cache_sha256'],
            public_grasp_frames=int(contact.sum()), public_grasp_and_open_frames=int(conflict.sum()),
            examples=[dict(time=float(t[i]), finger_open_amount=amount[i].tolist(),
                           public_contact=bool(contact[i]), public_gripper_open=bool(opened[i]),
                           prior_issued_close=bool(prior_close[i]))
                      for i in np.flatnonzero(conflict)[:3]]))
    report = dict(schema='r10_partial_TRAIN_public_grasp_feedback_diagnostic_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=sha(__file__),
        simulation_source_sha256=sha(ROOT / 'our_mani/multitask/simulation.py'),
        completed_teacher_receipt_snapshot=len(entries), original_expected_teachers=1400,
        full_TRAIN_coverage_claimed=False, canonical_receipts_in_snapshot=len(snapshot),
        per_task={task: dict(value) for task, value in sorted(counts.items())}, entries=entries,
        contact_contract='shared public bool(gripper.get_grasped_objects()); object identities not exposed',
        gripper_open_contract='shared public mean(gripper.get_open_amount()) > 0.5',
        interpretation='Recorded public holding and open signals can coexist. This diagnoses a potential controller completion mismatch, not task success or a validated replacement detector.',
        task_outcome_filtering=False, failed_or_censored_tasks_excluded=False,
        all_snapshot_teacher_cases_preserved=True, source_lineage_release_claimed=False,
        raw_arrays_modified=False, runtime_observations_changed=False,
        policy_fitted=False, native_started=False, GPU_started=False)
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return {k: report[k] for k in ('completed_teacher_receipt_snapshot', 'per_task')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out), ensure_ascii=False))
