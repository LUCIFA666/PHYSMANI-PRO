"""Read-only TRAIN audit of cup RGB-D pose flips and attached-pose continuity.

Native poses are offline metrics only. No threshold here changes acquisition,
training masks, policy behavior, or the reported task outcomes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('all audit inputs/outputs must stay in our-mani')
    return path


def summary(values):
    a = np.asarray(values, float)
    if not len(a):
        return dict(n=0, median=None, p95=None, maximum=None)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite offline audit statistic')
    return dict(n=len(a), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), maximum=float(np.max(a)))


def audit(collection):
    collection = inside(collection)
    rows, cases, files, errors = [], [], {}, []
    frozen_sources = (
        'our_mani/multitask/forecasting.py',
        'our_mani/multitask/interaction.py',
        'our_mani/multitask/context_se3.py',
        'our_mani/multitask/track_dataset.py',
    )
    for name in frozen_sources:
        if inside(name).exists():
            files[name] = sha(inside(name))
    for path in sorted(collection.glob('cases/train/*cups*/*/case.json')):
        try:
            record = json.loads(path.read_text())
            entry = record['entry']
            if (record['split'] != 'train' or entry['split'] != 'train'
                    or record['status'] != 'completed'
                    or not path.parent.name == record['case_id']):
                raise ValueError('noncanonical or non-TRAIN completion')
            files[str(path.relative_to(ROOT))] = sha(path)
            cache = inside(entry['cache'])
            if sha(cache) != entry['cache_sha256']:
                raise ValueError('raw body-track SHA changed')
            files[str(cache.relative_to(ROOT))] = sha(cache)
            with np.load(cache, allow_pickle=False) as arrays:
                a = {key: arrays[key] for key in arrays.files}
            times, names = a['times'], list(map(str, a['body_ids']))
            tcp_r = Rotation.from_quat(a['measured_tcp_xyzw'][:, 3:]).as_matrix()
            tcp_p = a['measured_tcp_xyzw'][:, :3]
            case_rows = []
            for body, name in enumerate(names):
                if not name.startswith('cup'):
                    continue
                local_r = np.einsum('tji,tjk->tik', tcp_r, a['rotations'][:, body])
                local_p = np.einsum('tji,tj->ti', tcp_r, a['positions'][:, body] - tcp_p)
                gt_local_r = np.einsum('tji,tjk->tik', tcp_r, a['target_rotations'][:, body])
                gt_local_p = np.einsum('tji,tj->ti', tcp_r, a['target_positions'][:, body] - tcp_p)
                # Mode 1 is the declared public attached mode, not native grasp.
                attached = ((a['modes'][:, body] == 1) & a['mode_valid'][:, body]
                            & a['robot_contact'] & ~a['robot_gripper_open'])
                registered = a['valid'][:, body] & a['rotation_observable'][:, body]
                gt_valid = a['native_pose_valid'][:, body]
                error = np.rad2deg(Rotation.from_matrix(
                    a['rotations'][:, body] @ np.swapaxes(a['target_rotations'][:, body], -1, -2)
                ).magnitude())
                for index, time in enumerate(times):
                    row = dict(case_id=record['case_id'], task=entry['task'],
                        parent_source_group=entry['parent_source_group'],
                        context_primary_view=entry['context_primary_view'],
                        behavior=entry['behavior'], body=name, frame=index, time=float(time),
                        public_attached_with_contact=bool(attached[index]),
                        registered_full_R=bool(registered[index]),
                        native_pose_metric_available=bool(gt_valid[index]),
                        teacher_phase=str(a['teacher_phase'][index]),
                        context_external_dynamics_frame_eligible=bool(
                            entry['context_primary_view'] and registered[index]
                            and a['modes'][index, body] != 1),
                        orientation_error_deg=float(error[index]) if gt_valid[index] else None,
                        public_local_rotation_step_deg=None, public_local_position_step_m=None,
                        native_local_rotation_step_deg=None, native_local_position_step_m=None)
                    if (index and attached[index] and attached[index-1]
                            and registered[index] and registered[index-1]
                            and abs(float(times[index]-times[index-1])-.05) < 1e-6):
                        row['public_local_rotation_step_deg'] = float(np.rad2deg(
                            Rotation.from_matrix(local_r[index] @ local_r[index-1].T).magnitude()))
                        row['public_local_position_step_m'] = float(np.linalg.norm(local_p[index]-local_p[index-1]))
                        if gt_valid[index] and gt_valid[index-1]:
                            row['native_local_rotation_step_deg'] = float(np.rad2deg(
                                Rotation.from_matrix(gt_local_r[index] @ gt_local_r[index-1].T).magnitude()))
                            row['native_local_position_step_m'] = float(np.linalg.norm(gt_local_p[index]-gt_local_p[index-1]))
                    case_rows.append(row)
            rows.extend(case_rows)
            cases.append(dict(case_id=record['case_id'], task=entry['task'],
                parent_source_group=entry['parent_source_group'], behavior=entry['behavior'],
                context_primary_view=entry['context_primary_view'], body_frames=len(case_rows)))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(path=str(path.relative_to(ROOT)), error=repr(error)))
    large = [row for row in rows if row['registered_full_R']
             and row['native_pose_metric_available'] and row['orientation_error_deg'] > 30.]
    continuity = [row for row in rows if row['public_local_rotation_step_deg'] is not None]
    return dict(version='r10_TRAIN_attached_cup_pose_readonly_audit_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        collection=str(collection.relative_to(ROOT)), partial_collection_snapshot=True,
        source_parent_count=len({case['parent_source_group'] for case in cases}),
        completed_cup_cases=len(cases), cup_body_frames=len(rows), errors=errors, cases=cases,
        diagnostic_threshold_deg=30., threshold_changes_policy_or_training=False,
        confident_large_pose_errors=len(large),
        large_pose_errors_in_context_primary_external_frames=sum(
            row['context_external_dynamics_frame_eligible'] for row in large),
        large_pose_errors_in_public_attached_mode=sum(
            row['public_attached_with_contact'] for row in large),
        large_pose_error_examples=large,
        attached_continuity={key: summary([row[key] for row in continuity if row[key] is not None])
            for key in ('public_local_rotation_step_deg', 'public_local_position_step_m',
                        'native_local_rotation_step_deg', 'native_local_position_step_m')},
        discontinuity_examples=[row for row in continuity if row['public_local_rotation_step_deg'] > 30.],
        interpretation=dict(
            context_primary='Only declared hold primary views feed Context; attached mode is excluded from external-body dynamics.',
            controller_attachment='Runtime can preserve a transform measured at public acquire contact using current robot FK.',
            remaining_risk='A bad pose when measuring that acquire transform can still corrupt control; this TRAIN audit does not prove actual learned-policy captures were correct.',
            TRAIN_teacher_not_learned_policy=True, native_GT_used_only_for_offline_metrics=True,
            no_frame_or_case_removed=True, no_training_or_policy_gate_added=True),
        files_read_sha256=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = inside(args.out)
    if output.exists():
        raise ValueError('audit output must be a fresh path')
    value = audit(args.collection)
    value['audit_source_sha256'] = sha(__file__)
    for filename, expected in value['files_read_sha256'].items():
        if sha(inside(filename)) != expected:
            raise ValueError('audited input changed while reading: ' + filename)
    value['all_inputs_sha_rechecked_at_end'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value[key] for key in ('completed_cup_cases', 'cup_body_frames',
        'confident_large_pose_errors', 'large_pose_errors_in_context_primary_external_frames',
        'large_pose_errors_in_public_attached_mode', 'errors')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
