"""Audit completed author-OPENGL TRAIN/VAL rollouts without changing old inputs.

Zero or partial collections are valid: all checks apply to records already
committed atomically by the sampler. Coverage becomes mandatory when a
predictor/split reaches its preregistered total.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/aligned_r07'
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.benchmark_protocol import verify_lock, file_sha256
from our_mani.risk_v2.selector import attempt_controller_hash

TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
SENSOR_PROTOCOL = 'shared_author_opengl_rgbd_r07'
CAMERAS = {'front', '8', '16', '36'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.audit.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def audit_record(record, *, predictor, split, expected_controller, expected_sources,
                 expected_collection_files, seconds=10., dt=.05):
    """Return errors for one completed rollout; never repairs or relabels it."""
    errors = []
    def check(condition, error):
        if not condition:
            errors.append(error)

    source = (record.get('task'), record.get('source_episode'))
    check(source in expected_sources, 'unknown_source_or_wrong_manifest')
    check(record.get('raw_episode_id') == record.get('source_episode'), 'source_identity')
    check(record.get('predictor') == predictor and record.get('split') == split, 'split_or_predictor')
    check(record.get('formal_test_used') is False, 'formal_TEST_used')
    check(record.get('attempt_controller_sha256') == expected_controller, 'continuation_controller')
    check(record.get('sensor_domain') == 'author_OPENGL', 'sensor_domain')
    replica = record.get('replica')
    replicas = 1 if split == 'train' else 3
    check(type(replica) is int and 0 <= replica < replicas, 'replica_range')
    if type(replica) is int and 0 <= replica < replicas:
        check(record.get('exploration') == ('geometry', 'softmax', 'uniform')[replica], 'replica_exploration')

    sources = record.get('collection_sources', {})
    check(sources == expected_collection_files, 'collection_source_hashes')
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    check(record.get('inputs_sha256') == identity, 'collection_identity')
    conditions = record.get('public_initial_conditions', {})
    protocol = conditions.get('sensor_protocol', {})
    check(protocol.get('version') == SENSOR_PROTOCOL, 'sensor_protocol_version')
    modes = protocol.get('camera_render_modes', {})
    check(set(modes) == CAMERAS and all(mode == 'OPENGL' for mode in modes.values()), 'camera_render_modes')
    cameras = conditions.get('cameras', {})
    check(set(cameras) == CAMERAS and all(c.get('render_mode') == 'OPENGL' for c in cameras.values()),
          'public_camera_render_modes')
    rgb = protocol.get('rgb_processing', {})
    check(set(rgb.get('observation/raw_rgb_255', {})) == CAMERAS, 'shared_raw_RGB_spec')

    receipt = record.get('alignment_receipt', {})
    clock = receipt.get('clock', {})
    ticks = clock.get('ticks')
    check(type(ticks) is int and 0 <= ticks <= round(seconds / dt), 'physics_tick_budget')
    end = float(clock.get('simulation_seconds', float('nan')))
    check(np.isfinite(end) and -.000001 <= end <= seconds + .000001, 'simulation_clock_budget')
    if type(ticks) is int:
        check(abs(end - ticks * dt) <= 1e-6, 'clock_tick_consistency')
    check(abs(float(clock.get('dt', float('nan'))) - dt) <= 1e-9, 'physics_timestep')
    check(abs(float(clock.get('budget_seconds', float('nan'))) - seconds) <= 1e-9, 'declared_budget')
    for key in ('actual_physics_seconds', 'actual_task_seconds'):
        measured = float(receipt.get(key, float('nan')))
        check(np.isfinite(measured) and abs(measured - end) <= 1e-4, key + '_ledger_mismatch')

    attempts = record.get('attempts', [])
    positive_indices = []
    previous_end = None
    goals = set()
    for index, attempt in enumerate(attempts):
        prefix = 'attempt_' + str(index) + ':'
        def local(condition, error):
            check(condition, prefix + error)
        local(attempt.get('attempt_id') == index, 'attempt_sequence')
        goal = attempt.get('goal_id')
        local(goal is not None and goal not in goals, 'unique_goal')
        goals.add(goal)
        try:
            features = np.asarray(attempt.get('features'), dtype=float)
        except (TypeError, ValueError):
            features = np.asarray([])
        correct_features = features.shape == ((87,) if predictor == 'context' else (71,))
        local(correct_features and np.isfinite(features).all(), 'feature_shape_or_finite')
        selected = float(attempt.get('selected_at', float('nan')))
        arrival = float(attempt.get('arrival_time', float('nan')))
        observed = float(attempt.get('outcome_observed_until', float('nan')))
        local(attempt.get('complete') is True, 'censored_label')
        local(np.isfinite([selected, arrival, observed]).all() and
              0 <= selected <= observed + 1e-8 and selected < arrival <= seconds + 1e-8 and observed <= end + 1e-6,
              'attempt_clock_or_arrival')
        if previous_end is not None:
            local(previous_end <= selected + 1e-8, 'overlapping_attempts')
        previous_end = observed
        remaining = float(attempt.get('remaining_episode_budget_seconds', float('nan')))
        transport = float(attempt.get('capture_transport_budget_after_arrival', float('nan')))
        local(abs(remaining - (seconds - selected)) <= 1e-6, 'remaining_budget')
        local(abs(transport - (seconds - arrival)) <= 1e-6, 'arrival_budget')
        if correct_features:
            local(np.allclose(features[61:64], [arrival-selected, remaining, transport], atol=1e-5, rtol=0),
                  'feature_budget_matches_record')
        if predictor == 'context':
            local(attempt.get('context_source') == 'current_past_observations' and
                  bool(attempt.get('context_encoder_sha256')), 'causal_context_provenance')
            if correct_features:
                context = features[-16:].reshape(2, 8)
                local((context >= -1e-7).all() and (context <= 1 + 1e-7).all() and
                      np.allclose(context.sum(1), 1., atol=1e-5), 'SimNorm_2x8_simplex')
        else:
            local(attempt.get('context_source') == 'CV_no_learned_context' and
                  attempt.get('context_encoder_sha256') is None, 'CV_context_provenance')
        if attempt.get('official_task_success') is True:
            positive_indices.append(index)
            local(attempt.get('outcome_reason') == 'native_task_success', 'success_outcome_reason')
            local(abs(observed-end) <= 1e-6, 'success_must_be_terminal_active_attempt')
        else:
            local(attempt.get('official_task_success') is False, 'binary_label')
    native = record.get('official_task_success')
    check(type(native) is bool, 'binary_native_success')
    check(positive_indices == ([len(attempts)-1] if native and attempts else []), 'incorrect_success_inheritance')
    check(not native or bool(attempts), 'native_success_without_active_candidate')
    check(receipt.get('native_success_latched') == native, 'native_success_receipt')
    return errors


def main():
    problems = []
    counts = {}
    verified_locks = []
    def problem(file, error):
        problems.append(dict(file=str(file), error=error))
    config = json.loads((ROOT / 'config/framework/aligned_protocol_r07.json').read_text())
    for lock_path in (BASE / 'collection_inputs.json', ROOT / 'artifacts/framework/aligned_r06/collection_inputs.json'):
        try:
            verify_lock(ROOT, json.loads(lock_path.read_text()))
            verified_locks.append(str(lock_path.relative_to(ROOT)))
        except (OSError, ValueError, KeyError) as error:
            problem(lock_path.relative_to(ROOT), 'frozen_inputs:' + str(error))
    expected = {}
    for split in ('train', 'validation'):
        manifest = json.loads((ROOT / f'artifacts/framework/protocol/{split}_manifest.json').read_text())
        selected = [(r['task'], r['raw_episode_id']) for r in manifest['episodes'] if r['task'] in TASKS]
        expected[split] = set(selected)
        if len(selected) != len(expected[split]):
            problem(split + '_manifest', 'duplicate_source')
        if len(selected) != (200 if split == 'train' else 40):
            problem(split + '_manifest', 'unexpected_source_coverage')
    if expected['train'] & expected['validation']:
        problem('TRAIN_VAL_manifests', 'overlapping_raw_sources')
    source_paths = ('scripts/collect_candidate_success_r07.py', 'our_mani/risk_v2/selector.py',
                    'scripts/evaluate_aligned_r05_v2.py', 'config/framework/aligned_protocol_r07.json',
                    'our_mani/sensor_alignment/author_rgbd.py')
    expected_files = {name: file_sha256(ROOT / name) for name in source_paths}
    for predictor in ('cv', 'context'):
        controller = attempt_controller_hash(ROOT, config, predictor)
        for split in ('train', 'validation'):
            replica_count = 1 if split == 'train' else 3
            required = {(*source, replica) for source in expected[split] for replica in range(replica_count)}
            # Snapshot shard reports before records, so a completed report can
            # never race ahead of the record glob checked below.
            shard_reports = []
            for path in sorted(BASE.glob(f'collection_{predictor}_{split}_*/report.json')):
                try:
                    shard_reports.append(json.loads(path.read_text()))
                except (OSError, ValueError) as error:
                    problem(path.relative_to(ROOT), 'malformed_shard_report:' + str(error))
            all_shards_declared_complete = (len(shard_reports) == (6 if split == 'train' else 2)
                and all(report.get('complete') is True for report in shard_reports))
            seen = Counter()
            attempts = []
            episodes = []
            per_task = {task: dict(completed=0, candidate_attempts=0, positives=0, native_successes=0) for task in TASKS}
            for path in sorted(BASE.glob(f'collection_{predictor}_{split}_*/pick*__episode*__*.json')):
                try:
                    record = json.loads(path.read_text())
                    errors = audit_record(record, predictor=predictor, split=split, expected_controller=controller,
                                          expected_sources=expected[split], expected_collection_files=expected_files,
                                          seconds=config['simulation_seconds'], dt=config['dt'])
                    for error in errors:
                        problem(path.relative_to(ROOT), error)
                    key = (record.get('task'), record.get('source_episode'), record.get('replica'))
                    seen[key] += 1
                    episodes.append(record)
                    attempts.extend(record.get('attempts', []))
                    if record.get('task') in per_task:
                        row = per_task[record['task']]
                        row['completed'] += 1
                        row['candidate_attempts'] += len(record.get('attempts', []))
                        row['positives'] += sum(a.get('official_task_success') is True for a in record.get('attempts', []))
                        row['native_successes'] += record.get('official_task_success') is True
                except (OSError, ValueError, TypeError, KeyError) as error:
                    problem(path.relative_to(ROOT), 'malformed_record:' + str(error))
            duplicates = [key for key, n in seen.items() if n > 1]
            if duplicates:
                problem(predictor + '_' + split, 'duplicate_source_replica:' + str(duplicates))
            if set(seen) - required:
                problem(predictor + '_' + split, 'unrecognized_source_replica')
            if (len(episodes) >= len(required) or all_shards_declared_complete) and set(seen) != required:
                problem(predictor + '_' + split, 'complete_collection_source_replica_coverage')
            if all_shards_declared_complete and (sum(r.get('completed', 0) for r in shard_reports) != len(required)
                    or sum(r.get('expected', 0) for r in shard_reports) != len(required)):
                problem(predictor + '_' + split, 'complete_shard_report_coverage')
            counts[predictor + '_' + split] = dict(completed_rollouts=len(episodes), expected_rollouts=len(required),
                complete=set(seen) == required and not duplicates, candidate_attempts=len(attempts),
                positives=sum(a.get('official_task_success') is True for a in attempts),
                native_successes=sum(r.get('official_task_success') is True for r in episodes),
                source_episodes=len({(r.get('task'), r.get('source_episode')) for r in episodes}), per_task=per_task)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=not problems, counts=counts,
                  errors=problems, verified_locks=verified_locks, formal_test_used=False,
                  sensor_domain='author_OPENGL', expected_rollouts=640,
                  complete=all(row['complete'] for row in counts.values()),
                  native_success_without_active_candidate=[p['file'] for p in problems
                      if 'native_success_without_active_candidate' in p['error']])
    write(BASE / 'data_quality_audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if problems:
        raise RuntimeError('r07 candidate data quality failed')


if __name__ == '__main__':
    main()
