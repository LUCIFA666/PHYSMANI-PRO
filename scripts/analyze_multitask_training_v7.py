"""Read-only v7 reference, rigid SE3 and collision-uncertainty smoke audit.

Uses only completed TRAIN receipts. Native poses are metric-side comparisons,
never corrections to stored tracker histories or labels. This does not fit a
model, promote an annotation, resample a failed source, or run native code.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_multitask_training_v6 as v6
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c
from our_mani.multitask.collision_geometry import SCHEMA, collision_geometry_for_orientation
from our_mani.multitask.reference_overrides import structural_overrides
from our_mani.multitask.specs import get_task_spec


def stats(values):
    return common.scalar_stats(values)


def extra_case(path, plan, case):
    """Validate new v7 contracts after the shared v6 clock/label audit."""
    record = common.read(path)
    entry = record['entry']
    with np.load(ROOT / entry['cache'], allow_pickle=False) as stored:
        a = {key: stored[key].copy() for key in stored.files}
    ids = list(map(str, a['body_ids']))
    files = record['files_sha256']
    diagnostic_path = next(ROOT / name for name in files if Path(name).name == 'tracker_diagnostics.json')
    frames = common.read(diagnostic_path)['frames']
    asset_path = next(ROOT / name for name in files if Path(name).name == 'TRAIN_geometry.json')
    asset = common.read(asset_path)
    annotations = plan['protocol']['body_orientation_observability'][case['task']]
    receipts = record.get('TRAIN_collision_geometry_receipt', {})
    common.require(set(receipts) == set(ids), 'per-body collision uncertainty receipt missing')
    for body in ids:
        actual, shape, annotation = receipts[body], asset['bodies'][body], annotations[body]
        common.require(actual['schema'] == SCHEMA and actual['orientation_observability'] == annotation,
                       'collision receipt annotation/schema differs: ' + body)
        common.require(actual['TRAIN_asset_sha256'] == asset['asset_sha256']
                       and actual['native_pose_or_future_state_used'] is False
                       and actual['TRAIN_asset_unchanged'] is True, 'collision provenance differs: ' + body)
        if shape['empty_collision_role']:
            common.require(actual['collision_geometry_kind'] == 'declared_noncolliding'
                           and actual['vertex_count'] == actual['triangle_count'] == 0,
                           'empty collision role fabricated mesh: ' + body)
        else:
            expected = collision_geometry_for_orientation(shape['collision_vertices_body'],
                shape['collision_triangles'], orientation_observability=annotation).receipt
            common.require(actual == dict(expected, TRAIN_asset_sha256=asset['asset_sha256']),
                           'registered uncertain/full geometry differs mathematically: ' + body)
    approved = structural_overrides(plan['protocol']).get(case['task'], {})
    common.require(entry['TRAIN_reference_kind_overrides_for_task'] == approved,
                   'entry reference override does not match frozen protocol')
    reference_rows = Counter()
    spec = get_task_spec(case['task'])
    for i in np.flatnonzero(a['teacher_action_valid']):
        phase = str(a['teacher_phase'][i])
        selected = {'selected_reference': str(a['teacher_reference_body'][i]),
                    'manipulated_body': str(a['teacher_manipulated_body'][i])}
        original = c.reference_for_phase(spec, phase, selected)
        frozen = c.reference_for_phase(spec, phase, selected, plan['protocol'])
        common.require({k: v for k, v in original.items() if k != 'reference_kind'} ==
                       {k: v for k, v in frozen.items() if k != 'reference_kind'},
                       'override changed physical/semantic role or anchor')
        for name in ('reference_body', 'reference_role', 'reference_kind', 'manipulated_body', 'manipulated_role'):
            common.require(str(a['teacher_' + name][i]) == str(frozen[name]),
                           'held teacher binding differs from explicit protocol: ' + name)
        reference_rows[phase + ':' + frozen['reference_kind']] += 1
    body_rows = {}
    for j, body in enumerate(ids):
        evidence = [frame['bodies'][body] for frame in frames]
        registered = np.asarray([row.get('registration_valid') is True for row in evidence])
        direct = np.asarray([row.get('directly_visible') is True for row in evidence])
        fresh = a['valid'][:, j]
        native = a['native_pose_valid'][:, j]
        pose_mask = registered & native
        indirect = pose_mask & ~direct
        orientation = np.asarray([row.get('orientation_observable') is True for row in evidence])
        rotation_mask = pose_mask & orientation
        position_error = np.linalg.norm(a['positions'][:, j] - a['target_positions'][:, j], axis=1)
        rotation_error = Rotation.from_matrix(a['rotations'][:, j] @
            np.swapaxes(a['target_rotations'][:, j], -1, -2)).magnitude()
        common.require(not np.any(fresh & ~direct), 'indirect pose was marked fresh visible: ' + body)
        if annotations[body] == 'none':
            common.require(not orientation.any() and not a['rotation_observable'][:, j].any(),
                           'unknown body rotation falsely promoted: ' + body)
        if asset['bodies'][body]['empty_visual_role']:
            common.require(not direct.any() and not fresh.any(), 'empty visual body fabricated visibility: ' + body)
        body_rows[body] = dict(annotation=annotations[body], direct_visible_rows=int(direct.sum()),
            direct_valid_history_rows=int(fresh.sum()), current_registration_rows=int(registered.sum()),
            indirect_registered_rows=int(indirect.sum()),
            indirect_pose_source_counts=dict(Counter(evidence[i].get('pose_source') for i in np.flatnonzero(indirect))),
            indirect_never_fresh=True, position_error_m=stats(position_error[pose_mask]),
            direct_valid_position_error_m=stats(position_error[fresh & native]),
            indirect_position_error_m=stats(position_error[indirect]),
            observed_rotation_error_rad=stats(rotation_error[rotation_mask]),
            indirect_observed_rotation_error_rad=stats(rotation_error[rotation_mask & ~direct]),
            rotation_metric_rows=int(rotation_mask.sum()),
            full_annotation_rotation_unobservable_rows=int((registered & ~orientation).sum()) if annotations[body] == 'full' else None,
            unknown_rotation_not_scored_as_true_spin=annotations[body] != 'full')
    groups = Counter()
    for frame in frames:
        for group in frame.get('rigid_groups', []):
            if group['registration_valid']:
                groups[group['root']] += 1
    constraints = c.load_rigid_constraints(case['rigid_layout'], case['task']) if case['rigid_layout'] else ()
    references = {}
    for i in np.flatnonzero(a['teacher_action_valid']):
        phase, body, kind = (str(a[name][i]) for name in
            ('teacher_phase', 'teacher_reference_body', 'teacher_reference_kind'))
        key = phase + ':' + str(a['teacher_reference_role'][i]) + ':' + kind
        summary = references.setdefault(key, dict(action_rows=0, direct_required_reference_rows=0,
            indirect_verified_group_reference_rows=0, unavailable_reference_rows=0))
        summary['action_rows'] += 1
        if kind in ('world', 'phase_origin'):
            summary['direct_required_reference_rows'] += 1
            continue
        j = ids.index(body)
        diagnostic = frames[i]['bodies'][body]
        rotation_ok = kind != 'observed_se3' or diagnostic.get('orientation_observable') is True
        if a['valid'][i, j] and rotation_ok:
            summary['direct_required_reference_rows'] += 1
            continue
        root = diagnostic.get('rigid_root')
        indirect_ok = (not diagnostic.get('directly_visible') and root in ids
            and diagnostic.get('registration_valid') is True and rotation_ok
            and a['mode_valid'][i, j] and a['dynamics_segment_valid'][i, j]
            and a['valid'][i, ids.index(root)] and a['rotation_observable'][i, ids.index(root)]
            and any(constraint.child_role == body and constraint.parent_role == root
                    for constraint in constraints))
        summary['indirect_verified_group_reference_rows' if indirect_ok else 'unavailable_reference_rows'] += 1
    return dict(reference_kind_rows=dict(reference_rows), TRAIN_reference_override_binding_checked=True,
        collision_receipts_mathematically_verified=True, collision_geometry=receipts,
        current_group_registration_rows=dict(groups), pose_metrics=body_rows,
        current_reference_evidence_by_phase=references,
        reference_availability_scope='current direct estimate or verified group-derived estimate; does not establish five-frame warmup, forecast, confidence, reachability or collision feasibility',
        pose_metric_namespace='TRAIN native metric-only; no correction to policy history or target labels',
        empty_visual_indirect_estimates_are_not_fresh_history=True)


def analyze(plan_path, queue_path):
    report = v6.analyze(plan_path, queue_path)
    plan = c.verify_plan(plan_path)
    cases = {case['case_id']: case for case in plan['cases']}
    verified = []
    for row in report['cases']:
        case = cases[row['case_id']]
        path = Path(queue_path) / 'cases' / case['split'] / case['task'] / case['case_id'] / 'case.json'
        try:
            row['v7'] = extra_case(path, plan, case)
            verified.append(row)
        except Exception as error:
            report['invalid_cases'].append(dict(case_id=case['case_id'], task=case['task'],
                behavior=case['behavior'], error=type(error).__name__ + ': ' + str(error)))
    report['cases'] = verified
    report['verified'] = len(verified)
    report['invalid'] = len(report['invalid_cases'])
    report['complete'] = len(verified) == 28 and not report['invalid']
    teachers = [row for row in verified if row['teacher'] is not None]
    holds = [row for row in verified if row['context_primary_view']]
    report.update(version='r10_offline_fixed_teacher_hold_smoke_v7', analysis_source=common.pointer(__file__),
        teacher_cases=len(teachers), teacher_native_successes=sum(row['native_success'] for row in teachers),
        hold_parents=len(holds), hold_full_horizon_parents=sum(row['parent_has_1p1_second_horizon'] for row in holds),
        teacher_completed_commands=sum(row['teacher']['completed_commands'] for row in teachers),
        explicit_phase_completion_events=sum(len(row['teacher']['phase_completion_events']) for row in teachers),
        allnone_orientation_is_not_full_SE3_observability=False,
        configured_orientation_is_not_proof_of_actual_observability=True,
        interpretation='fixed TRAIN interface acquisition regression; native teacher success is not pipeline SR; one parent/task cannot establish formal skill support; current measured orientation and visibility audited individually',
        reference_override_tasks=len(plan['protocol']['TRAIN_reference_kind_overrides']),
        reference_override_phase_pairs=sum(map(len, plan['protocol']['TRAIN_reference_kind_overrides'].values())),
        unknown_rotation_collision_cavities_filled=True,
        known_policy_limitation='unknown-yaw bin uses conservative solid all-rotation cube, so bin cavity is filled; this smoke is not a ranking or policy-benefit evaluation')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=c.BASE / 'TRAIN_smoke_collection_plan_v7.json')
    parser.add_argument('--queue', type=Path, default=c.BASE / 'TRAIN_smoke_collection_v7')
    parser.add_argument('--report', type=Path, default=c.BASE / 'TRAIN_smoke_analysis_v7.json')
    args = parser.parse_args()
    report = analyze(args.plan, args.queue)
    c.write_json(args.report, report, overwrite=True)
    lines = [f"R10 v7固定TRAIN smoke：{report['verified']}/28验收；invalid={report['invalid']} infra={report['infrastructure']} pending={report['missing']}",
        f"hold完整1.1秒窗口 {report['hold_full_horizon_parents']}/{report['hold_parents']}父源；teacher原生成功 {report['teacher_native_successes']}/{report['teacher_cases']}（采集诊断，不是pipeline SR）",
        f"world-waypoint完成={report['teacher_completed_commands']}；独立phase完成事件={report['explicit_phase_completion_events']}",
        '参考kind只采用批准的TRAIN映射；刚体间接pose不伪造fresh可见；none朝向不评分真实旋转；未知朝向碰撞cube填孔为已知限制。',
        '原生GT仅作离线误差核验，无label覆写，无真实fit或全1400/VAL/TEST。']
    for row in report['cases']:
        lines.append(row['task'] + ' | ' + row['behavior'] + f" | ticks={row['ticks']} native={row['native_success']} fullwin={row['context_window_eligibility']['full_horizon_windows']}")
        for body, metric in row['v7']['pose_metrics'].items():
            if metric['annotation'] == 'full' or metric['indirect_registered_rows']:
                lines.append(f"  {body}: direct={metric['direct_visible_rows']} indirect={metric['indirect_registered_rows']} position_p95_m={metric['position_error_m']['p95']} rotation_p95_rad={metric['observed_rotation_error_rad']['p95']}")
    lines += ['INVALID ' + str(row) for row in report['invalid_cases']]
    Path(args.report).with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print({key: report[key] for key in ('verified', 'invalid', 'infrastructure', 'missing', 'hold_parents', 'hold_full_horizon_parents', 'teacher_cases', 'teacher_native_successes')})


if __name__ == '__main__':
    main()
