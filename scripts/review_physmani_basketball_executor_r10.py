"""Read-only source comparison using the existing fixed TRAIN pair only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/multitask_r10'
OUT = BASE / 'physmani_basketball_executor_reference_review_v1'


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path, line=None):
    path = Path(path).resolve()
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if line is not None:
        result['line'] = line
    return result


def main():
    if OUT.exists():
        raise RuntimeError('Preserve previous reference review')
    config = read(BASE / 'geometry_causal_safety_brake_v12_setup/configuration.json')
    repo = Path(config['baseline_repo'])
    smoke = read(BASE / 'geometry_causal_safety_brake_v12/actual_smoke_report.json')
    sources = {
        'released_actioner': (repo / 'packages/physmani_policy/physmani_policy/actioner.py', 77),
        'released_arm_then_gripper': (repo / 'packages/physmani_bench/physmani_bench/env.py', 449),
        'released_path_modification': (repo / 'packages/physmani_bench/physmani_bench/env.py', 516),
        'released_RML': (repo / 'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py', 109),
        'basketball_TRAIN_expert': (repo / 'third_party/RLBench/rlbench/tasks/moving_basketball_in_hoop.py', 70),
        'basketball_native_success': (repo / 'third_party/RLBench/rlbench/tasks/moving_basketball_in_hoop.py', 192),
        'installed_neural_WM_adapter': (ROOT / 'our_mani/aligned/physmani.py', 121),
        'installed_counted_baseline_executor': (ROOT / 'our_mani/multitask/physmani_executor_v7.py', 32),
        'installed_baseline_direct_dispatch': (ROOT / 'our_mani/sensor_alignment/physmani_executor_v6.py', 25),
        'ours_candidate_arrival_gate': (ROOT / 'our_mani/multitask/interaction.py', 1534),
        'ours_current_fixed_executor': (ROOT / 'our_mani/multitask/causal_safety_brake_reference_v12.py', 266),
        'existing_TRAIN_hoop_intention': (ROOT / 'our_mani/multitask/basketball_command_goal_reference.py', 172),
        'review_generator': (Path(__file__), 1),
    }
    evidence = {}
    for method, folder in [('cv_geometry', 'cv_normal'), ('context_geometry', 'context_normal')]:
        path = BASE / 'geometry_causal_safety_brake_v12/development_demo_only' / folder / 'case.json'
        actual = pointer(path)
        assert actual['sha256'] == smoke['methods'][method]['case']['sha256']
        case = read(path)
        assert case['split'] == 'train' and not case['formal_test_used']
        actor = case['actor_metrics']
        trace = actor['executor_observed_state_trace']
        selections = [{k: event.get(k) for k in ('time', 'phase', 'candidate_count')}
                      for event in actor['candidate_selection_events']]
        first_search = {}
        for row in trace:
            phase = row['phase_index']
            search = row.get('controller_diagnostics', {}).get('candidate_search', {})
            # At the exact completion frame diagnostics can still describe the
            # preceding phase; select an uncommitted subsequent phase frame.
            if row['active_candidate_id'] is None and search.get('feasible_candidates') == 0:
                first_search.setdefault(str(phase), dict(time=row['time'], search=search))
        evidence[method] = dict(
            case=actual, task_success=case['official_task_success'],
            actual_motion_qualification=smoke['methods'][method]['actual_motion_qualification'],
            selected_phase_counts=dict(Counter(e['phase'] for e in selections)),
            selections=selections, observable_skill_events=actor['observable_skill_events'],
            first_uncommitted_rejected_search_by_phase=first_search,
            global_counted_clock=case['counted_clock'],
            search_snapshot_counts_are_not_unique_candidate_counts=True,
        )
    library = Path(config['learned_artifacts']['moving_basketball_in_hoop']['interaction']['path'])
    templates = [{k: row.get(k) for k in ('phase', 'reference_mode', 'reference_time_basis',
                  'duration_seconds', 'duration_budget_seconds')}
                 for row in read(library)['templates']]
    report = dict(
        schema='r10_readonly_PhysMani_basketball_executor_comparison_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_bindings={key: pointer(path, line) for key, (path, line) in sources.items()},
        TRAIN_only_behavior_evidence=evidence, interaction_library=pointer(library), templates=templates,
        verified_mechanisms={
            'neural_policy': 'Released model emits next TCP pose plus gripper state; generic actuator has no basketball-specific stage controller.',
            'prediction': 'Installed released WM predicts ten 50ms frames; neural action is refreshed after a counted ten-tick actuator chunk.',
            'execution': 'RRTConnect path is converted to RML joint references; execute up to nine arm ticks then apply new gripper state on the tenth. The endpoint need not be reached in this chunk.',
            'TRAIN_expert': 'Close toward a t+0.5s ball position, lift TCP by0.1m, carry closed toward t+0.5s hoop waypoint, then open when current hoop waypoint distance is at most0.03m.',
            'success': 'Native sensor detects ball, nothing remains grasped, gripper is open, and ball has previously been picked.',
            'ours_existing_goal_evidence': 'TRAIN deposit already uses the source-declared t+0.5 hoop intention; a missing throwing model is not established by this evidence.',
        },
        diagnosis={
            'cv': 'Acquire completes at4.85s, lift is selected at6.85s and completes at7.8s. No deposit candidate is selected. First deposit search rejects short-arrival IK/speed; later the remaining full-continuation budget is exhausted.',
            'context': 'First approach is selected at6.05s and completes at7.95s; acquire is never selected. The ball has not been grasped, so release timing is not the first demonstrated failure.',
            'mechanism_to_investigate': 'A bounded collision-checked transit segment can make partial progress toward a distant moving goal, as the released RML chunk does, before an arrival-qualified manipulation candidate is available.',
            'causal_status': 'Proposed remedy is not implemented or validated by this read-only review.',
        },
        implementation_constraints={
            'use_public_RGBD_forecast_and_TRAIN_templates': True,
            'retain_collision_SE3_joint_limits_and_global_10s': True,
            'retain_existing_phase_deadlines': True,
            'new_controller_requires_separate_version_and_TRAIN_acceptance': True,
            'active_v12_formal_policy_unchanged': True,
            'released_expert_driver_not_an_online_policy_input': True,
        },
        native_or_GPU_started_by_review=False, model_fitting_performed=False,
        formal_VAL_or_TEST_actor_results_read=False, formal_TEST_roster_read=False,
        prohibited_methods_started=False,
    )
    OUT.mkdir()
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(complete=True, report=str(OUT / 'report.json'), methods=list(evidence))))


if __name__ == '__main__':
    main()
