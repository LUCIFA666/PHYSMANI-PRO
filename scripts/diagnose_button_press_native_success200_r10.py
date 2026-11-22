"""TRAIN-only endpoint/contact-region diagnosis for all original200 buttons.

Native success is an archived outcome, not a public completion capability.
TCP at success is not an engine contact-manifold point.  All94 censored cases
remain in the audit; only the106 original success endpoints have endpoint
completion targets.  Region nearest-centre assignment is explicitly oracle.
"""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as a, artifact_fitting as adapter
from our_mani.multitask import command_goal_reference as goal, interaction, native_task_success_labels as labels
from our_mani.multitask import release_preflight as release, translation_command_goal_reference as translation

BASE = ROOT / 'artifacts/framework/multitask_r10'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(interaction._json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def distribution(values):
    x = np.asarray(values, float)
    if len(x) == 0:
        return dict(count=0)
    return dict(count=len(x), minimum=np.min(x, axis=0).tolist(), q05=np.quantile(x, .05, axis=0).tolist(),
                median=np.median(x, axis=0).tolist(), q95=np.quantile(x, .95, axis=0).tolist(),
                maximum=np.max(x, axis=0).tolist())


def regions(points, k):
    """Fixed deterministic train-fold-only Lloyd clusters, no parameter search."""
    x = np.asarray(points, float)
    centers = [x[np.argmin(np.linalg.norm(x - x.mean(axis=0), axis=1))]]
    while len(centers) < k:
        distance = np.min(np.linalg.norm(x[:, None, :] - np.asarray(centers)[None, :, :], axis=2), axis=1)
        centers.append(x[np.argmax(distance)])
    centers = np.asarray(centers)
    for _ in range(100):
        selected = np.argmin(np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2), axis=1)
        updated = np.asarray([x[selected == j].mean(axis=0) if np.any(selected == j) else centers[j] for j in range(k)])
        if np.max(np.abs(updated - centers)) <= 1e-12:
            centers = updated
            break
        centers = updated
    selected = np.argmin(np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2), axis=1)
    return centers, selected


def heldout_endpoints(rows, basis, *, k=1, conditional=False):
    parents = sorted({row['parent'] for row in rows})
    folds = {parent: index % 2 for index, parent in enumerate(parents)}
    residuals, fold_reports = [], []
    for fold in (0, 1):
        train = [row for row in rows if folds[row['parent']] != fold]
        held = [row for row in rows if folds[row['parent']] == fold]
        target = np.asarray([row['relative'][basis] for row in train])
        held_target = np.asarray([row['relative'][basis] for row in held])
        if conditional:
            x = np.asarray([row['issue_features'] for row in train])
            h = np.asarray([row['issue_features'] for row in held])
            mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
            design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
            hd = np.column_stack((np.ones(len(h)), (h - mean) / scale))
            penalty = np.eye(design.shape[1]) * .001
            penalty[0, 0] = 0
            weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
            predictions = hd @ weights
            assignment = np.zeros(len(train), dtype=int)
            centers = np.asarray([target.mean(axis=0)])
        else:
            centers, assignment = regions(target, k)
            # This uses held-out terminal targets: an optimistic lower bound,
            # never a causal region selector and never a policy Value target.
            held_assignment = np.argmin(np.linalg.norm(held_target[:, None, :] - centers[None, :, :], axis=2), axis=1)
            predictions = centers[held_assignment]
        counts = [len({row['parent'] for row, c in zip(train, assignment) if c == j}) for j in range(k)]
        rotation = Rotation.from_quat(np.asarray([row['tcp_xyzw'][3:] for row in train])).mean().as_matrix()
        for row, predicted in zip(held, predictions):
            residuals.append(dict(parent=row['parent'], fold=fold, predicted=predicted.tolist(),
                position=float(np.linalg.norm(predicted - row['relative'][basis])),
                orientation=float(Rotation.from_matrix(rotation @ Rotation.from_quat(row['tcp_xyzw'][3:]).as_matrix().T).magnitude())))
        fold_reports.append(dict(heldout_fold=fold, train_parents=len(train), heldout_parents=len(held),
                                 train_region_parent_counts=counts, centers=centers.tolist(),
                                 every_region_has8_train_parents=min(counts) >= 8))
    position = max(.005, float(np.quantile([row['position'] for row in residuals], .95)))
    orientation = max(.03, float(np.quantile([row['orientation'] for row in residuals], .95)))
    return dict(basis=basis, k=k, conditional=conditional, parent_count=len(parents), source_to_fold=folds,
                folds=fold_reports, parent_residuals=residuals, position_tolerance_m=position,
                orientation_tolerance_rad=orientation, unchanged_caps_pass=position <= .03 and orientation <= .15,
                every_train_region_has8parents=all(row['every_region_has8_train_parents'] for row in fold_reports),
                oracle_nearest_region_assignment=not conditional and k > 1,
                causal_public_region_selector_qualified=False, endpoint_diagnostic_is_full_curve_fit=False)


def run(output):
    began = time.perf_counter()
    out = a.local(ROOT, output)
    a.require(not out.exists(), 'fresh actual200 press diagnosis namespace required')
    out.mkdir(parents=True)
    inputs = release._Inputs(ROOT)
    reference_path = BASE / 'button_native_success_independent_labels_actual200_v2/report.json'
    reference = inputs.read(str(reference_path), a.file_sha(reference_path))
    existing = inputs.read_pointer(reference['labels'])
    a.require(len(existing) == len({row['parent_source_group'] for row in existing}) == 200,
              'all original200 source/parent labels required')
    rule = goal.load_rule(ROOT, a.pointer(ROOT, BASE / 'command_goal_reference_rule_v5_translation_revision3_setup/command_goal_rule_original_only.json'), purpose='production_fitting')
    inputs.include(rule.verified_files_sha256)
    for module in (goal, adapter, labels, translation, interaction, release):
        inputs.file(module.__file__, a.file_sha(module.__file__))
    inputs.file(__file__, a.file_sha(__file__))
    snapshot_pointer = a.read(BASE / 'command_goal_reference_rule_v6_counted_final_setup_v1/preparation.json')['source_snapshot']
    snapshot = inputs.source_snapshot(snapshot_pointer)
    original_label_map = {row['source_id']: row for row in existing}
    source_evidence = {}
    native_tasks = ROOT.parent / 'phsimani/PhysMani/third_party/RLBench/rlbench/tasks'
    high_helper = native_tasks / 'reach_single_moving_target_on_the_table_high_speed.py'
    helper_copy = out/'readonly_current_source'/high_helper.name
    helper_copy.parent.mkdir(parents=True)
    shutil.copyfile(high_helper, helper_copy)
    source_evidence['high_speed_boundary_helper'] = dict(original_path=str(high_helper),
        current_derivation_snapshot=a.pointer(ROOT,inputs.file(str(helper_copy))), sha256=a.file_sha(high_helper),
        historical_execution_dependency_claimed=False)
    tree = ast.parse(helper_copy.read_text())
    boundary_function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='handle_boundary')
    boundary_namespace = dict(np=np)
    exec(compile(ast.Module(body=[boundary_function],type_ignores=[]),'<new-readonly-TRAIN-button-boundary>', 'exec'),boundary_namespace)
    rows, trajectories, replay_errors, formula_errors = [], [], [], []
    for old in existing:
        binding = old['raw_bindings']
        path = inputs.pointer(binding['canonical_receipt'])
        record = a.read(path)
        with np.load(inputs.pointer(binding['cache']), allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        commands_path = inputs.pointer(binding['commands'])
        command_receipt = a.read(commands_path)
        case = dict(path=path, record=record, entry=record['entry'], arrays=arrays,
                    command_receipt=command_receipt, command_receipt_verified=True,
                    command_receipt_sha256=binding['commands']['sha256'])
        source = adapter.interaction_source(case)[0]
        value = labels.extract(case, source, rule)
        a.require(interaction._json(asdict(value)) == old, 'cached native success label differs from actual raw reconstruction')
        inputs.include(value.derivation_proof['verified_files_sha256'])
        states, kernel, replay = translation.replay_states(case, rule.translation_permission)
        high = value.task_name.endswith('_high_speed')
        delay = .25 if high else .5
        _, source_area = translation._kernel(rule.translation_permission,value.task_name)
        replay_errors.append(replay['maximum_body_replay_error_m'])
        ids = list(arrays['body_ids'])
        native = arrays['target_positions']
        tcp = arrays['measured_tcp_xyzw']
        times = arrays['times']
        final = len(times) - 1
        body = ids.index('button')
        plate = ids.index('plate')
        wraps = ids.index('wrap')
        ledger = command_receipt['commands']
        last = next((c for c in ledger if c['command_id'] == int(arrays['teacher_command_id'][final])), ledger[-1])
        press = [c for c in ledger if c['raw_phase'] == 'press']
        predictions = {}
        for command in press:
            found = np.flatnonzero(np.isclose(times, command['generated_at'], rtol=0, atol=1e-8))
            a.require(len(found) == 1, 'one actual issue frame required')
            i = int(found[0])
            tt, state = states[i]
            forecast = np.asarray(kernel['compute_target_position'](t=tt + delay, t0=state['t0'],
                x0=state['x'], v0=state['v'], a0=state['a'], dt=float(np.float32(.05))))
            unclipped = forecast.copy()
            if high:
                forecast = boundary_namespace['handle_boundary'](forecast.copy(),source_area)
            raw_goal = np.asarray(command['raw_waypoint_xyzw'], float)
            error = float(np.linalg.norm(raw_goal[:2] - forecast[:2]))
            a.require(error <= 1e-6, 'source-specific press forecast differs from committed actual waypoint')
            formula_errors.append(error)
            predictions[command['command_id']] = dict(forecast=forecast, issue=i,
                canonical_world_goal_offset=raw_goal[:3] - forecast,
                raw_unclipped_forecast=unclipped, source_delay=delay,
                boundary_clip_distance=float(np.linalg.norm(forecast[:2]-unclipped[:2])))
        relative = {role + '_current': (tcp[final, :3] - native[final, ids.index(role)]).tolist()
                    for role in ('button', 'plate', 'wrap')}
        relative['committed_world_goal'] = (tcp[final, :3] - np.asarray(last['raw_waypoint_xyzw'])[:3]).tolist()
        original_initial = record['native_reset_receipt']['native_initial_snapshot']
        row = dict(task=value.task_name, source_id=value.source_id, parent=value.parent_source_group,
            status=value.status, native_success=value.raw_task_success, budget_censored=value.budget_censored,
            raw_bindings=binding, raw_command_count=len(ledger), raw_press_command_count=len(press),
            terminal_frame=final, terminal_time=float(times[final]), terminal_phase=last['raw_phase'],
            terminal_command_id=last['command_id'], terminal_command_completed='command_completed_at' in last,
            terminal_TCP_is_not_recorded_contact_manifold=True, tcp_xyzw=tcp[final].tolist(), relative=relative,
            button_plate_terminal_difference_m=(native[final, plate] - native[final, body]).tolist(),
            plate_relative_button_change_from_initial_m=((native[final, plate] - native[final, body]) - (native[0, plate] - native[0, body])).tolist(),
            public_robot_contact_at_terminal=bool(arrays['robot_contact'][final]),
            public_robot_contact_is_grasp_association_not_force_sensor=True,
            native_last10_TCP_z_descent_including_terminal_m=float(tcp[max(0, final - 9), 2] - tcp[final, 2]),
            native_last10_TCP_z_descent_before_terminal_m=float(tcp[max(0, final - 10), 2] - tcp[max(0, final - 1), 2]),
            terminal_finger_open=arrays['gripper_open_amount_after'][final].tolist(),
            original_native_initial_joint=original_initial['bodies']['press_joint']['joint_position_native_units'],
            terminal_press_joint_not_recorded=True)
        if last['command_id'] in predictions:
            prediction = predictions[last['command_id']]
            issue = prediction['issue']
            issued = native[issue, body]
            row['relative'].update(button_issue=(tcp[final, :3] - issued).tolist(),
                original_press_forecast=(tcp[final, :3] - prediction['forecast']).tolist())
            row.update(command_issued_at=float(last['generated_at']), terminal_command_age=float(times[final] - last['generated_at']),
                issue_frame=issue, original_press_forecast=prediction['forecast'].tolist(),
                canonical_world_goal_offset=prediction['canonical_world_goal_offset'].tolist(),
                original_source_forecast_delay=prediction['source_delay'],
                original_high_speed_boundary_clip_distance=prediction['boundary_clip_distance'],
                button_issue_to_terminal_displacement_m=(native[final, body] - issued).tolist(),
                issue_features=np.concatenate((tcp[issue, :3] - issued, arrays['target_linear_velocities'][issue, body, :2],
                    prediction['forecast'][:2] - issued[:2])).tolist(),
                issue_features_are_causal_TRAIN_native_labels_not_qualified_public_inputs=True)
        if value.raw_task_success:
            a.require(last['raw_phase'] == 'press' and not row['terminal_command_completed']
                and value.success_completion.terminal_frame == final, 'success must remain actual incomplete final press')
        rows.append(row)
        trajectories.append(dict(source_id=value.source_id, parent=value.parent_source_group, native_success=value.raw_task_success,
            frames=[dict(frame=i,time=float(times[i]),phase=str(arrays['teacher_phase'][i]),
                command_id=int(arrays['teacher_command_id'][i]), tcp_xyzw=tcp[i].tolist(),
                tcp_relative_current_button=(tcp[i,:3]-native[i,body]).tolist(),
                tcp_relative_current_plate=(tcp[i,:3]-native[i,plate]).tolist(),
                plate_relative_button=(native[i,plate]-native[i,body]).tolist(),
                public_robot_contact=bool(arrays['robot_contact'][i]),
                gripper_open=arrays['gripper_open_amount_after'][i].tolist())
                for i in range(max(0, final - 20), final + 1)]))
    a.require(len(rows) == 200 and sum(row['native_success'] for row in rows) == 106
              and sum(row['budget_censored'] for row in rows) == 94, 'all106 success and94 censored must remain')
    results = {}
    for task in labels.TASKS:
        selected = [row for row in rows if row['task'] == task]
        success = [row for row in selected if row['native_success']]
        failed = [row for row in selected if not row['native_success']]
        endpoint = {basis:heldout_endpoints(success,basis) for basis in success[0]['relative']}
        previous_path = BASE / f'translation_goal_v5_actual300_production_CPU_fit_v2/{task}/real_production_fit_report.json'
        previous = inputs.read(str(previous_path), a.file_sha(previous_path))
        production = next(row for key, row in previous['phase_fits'].items() if key.startswith('press|'))
        a.require(abs(endpoint['button_current']['position_tolerance_m'] - production['position_tolerance_m']) < 1e-12
            and abs(endpoint['button_current']['orientation_tolerance_rad'] - production['orientation_tolerance_rad']) < 1e-12,
            'independent K1 endpoint crossfit must reproduce actual production press cap')
        results[task] = dict(all_original100_retained=len(selected), success_endpoints=len(success), budget_censored=len(failed),
            raw_terminal_TCP_distributions={status:{basis:dict(xyz=distribution([row['relative'][basis] for row in group if basis in row['relative']]),
                    norm=distribution([np.linalg.norm(row['relative'][basis]) for row in group if basis in row['relative']]))
                    for basis in success[0]['relative']} for status,group in (('success',success),('budget_censored',failed))},
            independent_parent_K1_crossfits=endpoint,
            oracle_nearest_contact_region_K1_to4={str(k):heldout_endpoints(success,'button_current',k=k) for k in (1,2,3,4)},
            issue_conditioned_TRAIN_native_feature_ridge=heldout_endpoints(success,'button_current',conditional=True),
            source_press_forecast_command_count=sum(row['raw_press_command_count'] for row in selected),
            current_plate_minus_button=distribution([row['button_plate_terminal_difference_m'] for row in selected]),
            success_plate_deformation_proxy=distribution([row['plate_relative_button_change_from_initial_m'] for row in success]),
            success_descent_including_terminal=distribution([row['native_last10_TCP_z_descent_including_terminal_m'] for row in success]),
            success_descent_before_terminal=distribution([row['native_last10_TCP_z_descent_before_terminal_m'] for row in success]),
            success_terminal_command_age=distribution([row['terminal_command_age'] for row in success]),
            success_button_issue_to_terminal_displacement=distribution([np.linalg.norm(row['button_issue_to_terminal_displacement_m']) for row in success]),
            success_TCP_to_raw_committed_goal=distribution([np.linalg.norm(row['relative']['committed_world_goal']) for row in success]),
            success_public_robot_contact_count=sum(row['public_robot_contact_at_terminal'] for row in success),
            success_terminal_finger_open=distribution([row['terminal_finger_open'] for row in success]))
    row_path, trace_path = out/'all200_source_endpoint_audit.json', out/'all200_terminal20frame_timelines.json'
    write(row_path, rows);write(trace_path, trajectories)
    a.require(a.file_sha(high_helper)==source_evidence['high_speed_boundary_helper']['sha256'],
              'readonly current boundary helper changed during diagnosis')
    inputs.rehash_all()
    summary = dict(schema='r10_actual200_button_press_native_success_endpoint_region_diagnosis_v1',
        TRAIN_only=True, VAL_used=False, TEST_used=False, native_success_reference_report=a.pointer(ROOT,reference_path),
        labels_reconstructed_exactly_from_actual_raw=True, actual_sources=200, independent_parents=200,
        original_success=106, original_budget_censored=94, all_source_endpoint_audit=a.pointer(ROOT,row_path),
        terminal_timelines=a.pointer(ROOT,trace_path), per_task=results,
        source_replay_maximum_error_m=max(replay_errors), original_press_forecast_goal_xy_maximum_error_m=max(formula_errors),
        source_specific_press_forecast_semantics=dict(normal_delay=.5,high_speed_delay=.25,high_speed_boundary_clip_buffer=.03),
        new_readonly_source_evidence=source_evidence,
        frozen22_core_source_snapshot=snapshot_pointer, all22_current_archived_core_sources_unchanged=True,
        original_masks_commands_events_and_censoring_not_modified=True,
        endpoint_oracle_regions_not_public_selector_or_Value_labels=True,
        actual_contact_manifold_or_terminal_press_joint_observations_recorded=False,
        native_success_not_promoted_to_runtime_completion=True, production_library_or_thresholds_modified=False,
        native_GPU_new_collection_or_formal_model_fit_started=False, final_direct_input_rehash_passed=True,
        verified_files_sha256=inputs.files, seconds=time.perf_counter()-began)
    write(out/'report.json',summary)
    return a.pointer(ROOT,out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    print(json.dumps(run(args.out)))
