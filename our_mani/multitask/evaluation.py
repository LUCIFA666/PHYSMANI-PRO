"""Fresh-case input identities and metric-only completion checks for R10."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import MultitaskProtocolError

METHODS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value', 'physmani')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def comparison_protocol_sha256(config):
    """Bind all policy/external conditions without a release SHA cycle.

    VAL and TEST use distinct frozen config files. Only the completed-VAL
    release pointer and its readiness flag are added after VAL; every other
    field, including model/source pointers, participates in this identity.
    """
    return digest({key: value for key, value in config.items()
                   if key not in ('r10_validation_release', 'formal_comparison_ready')})


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError('completed records cannot be silently overwritten: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_inputs(root, declared):
    """File receipts are explicit, not globbed differently between workers."""
    if not declared or not isinstance(declared, dict):
        raise MultitaskProtocolError('a frozen input lock is required')
    verified = {}
    for relative, expected in declared.items():
        path = Path(root) / relative
        if (not isinstance(expected, str) or len(expected) != 64
                or not path.is_file() or file_sha(path) != expected):
            raise MultitaskProtocolError('frozen input changed or missing: ' + str(path))
        verified[relative] = expected
    return verified


def case_identity(*, split, task, raw_episode_id, method, source_row,
                  config_sha256, inputs_sha256, parent_raw_episode_id=None, replica=0,
                  demo_only=False, exploration_seed=None):
    if split not in ('train', 'validation', 'test') or method not in METHODS:
        raise MultitaskProtocolError('unknown split/method')
    source_raw = source_row.get('raw_episode_id')
    if source_row.get('task') != task or not isinstance(source_raw, str):
        raise MultitaskProtocolError('exact source task/raw episode identity required')
    if split == 'test':
        if (parent_raw_episode_id is not None or replica != 0 or raw_episode_id != source_raw
                or exploration_seed is not None):
            raise MultitaskProtocolError('official TEST is not synthetic augmentation')
    elif parent_raw_episode_id != source_raw or raw_episode_id != f'{source_raw}__policy_aug_r{replica}':
        raise MultitaskProtocolError('development rollout needs an explicit source parent and unique augmentation ID')
    if exploration_seed is not None and (split != 'train' or not method.endswith('_geometry')):
        raise MultitaskProtocolError('first candidate exploration is TRAIN geometry only')
    value = dict(version='r10_fresh_case_identity_v1', split=split, task=task,
        raw_episode_id=raw_episode_id, parent_raw_episode_id=parent_raw_episode_id,
        source_raw_episode_id=source_raw, exploration_seed=exploration_seed,
        method=method, source_row_sha256=digest(source_row), config_sha256=config_sha256,
        frozen_inputs_sha256=inputs_sha256, replica=int(replica), demo_only=bool(demo_only))
    return dict(value, sha256=digest(value))


def validate_completed_case(record, expected_identity):
    """Protocol failures and censored infrastructure never become SR negatives."""
    if (record.get('identity') != expected_identity or record.get('complete') is not True
            or record.get('case_isolation') != 'fresh_simulator_process_per_episode'
            or not isinstance(record.get('process_id'), int) or record['process_id'] <= 0
            or record.get('infrastructure_error') is not None):
        raise MultitaskProtocolError('case is incomplete, censored or has different inputs')
    for field in ('task', 'raw_episode_id', 'parent_raw_episode_id', 'split', 'method'):
        if record.get(field) != expected_identity[field]:
            raise MultitaskProtocolError('top-level case identity differs: ' + field)
    if type(record.get('official_task_success')) is not bool or type(record.get('native_terminal_failure')) is not bool:
        raise MultitaskProtocolError('independent native boolean outcome required')
    if record['official_task_success'] and record['native_terminal_failure']:
        raise MultitaskProtocolError('first native terminal cannot be success and fatal')
    if record.get('formal_test_used') != (expected_identity['split'] == 'test' and not expected_identity['demo_only']):
        raise MultitaskProtocolError('demo-only replay cannot enter formal TEST metrics')
    if record.get('TEST_used_for_fitting') is not False:
        raise MultitaskProtocolError('offline TEST fitting is forbidden')
    clock = record.get('counted_clock', {})
    ticks, dt, seconds = clock.get('ticks'), clock.get('dt'), clock.get('seconds')
    if (type(ticks) is not int or not 0 <= ticks <= 200 or dt != .05
            or not isinstance(seconds, (int, float)) or not np.isfinite(seconds)
            or abs(seconds - ticks * dt) > 1e-6):
        raise MultitaskProtocolError('counted physical clock differs from declared budget')
    alignment = record.get('alignment_receipt', {})
    ledger = alignment.get('clock', {})
    if (ledger.get('ticks') != ticks or ledger.get('dt') != dt or ledger.get('budget_seconds') != 10.
            or not isinstance(ledger.get('simulation_seconds'), (int, float))
            or not np.isfinite(ledger['simulation_seconds'])
            or abs(ledger['simulation_seconds'] - seconds) > 1e-6):
        raise MultitaskProtocolError('native ledger differs from counted record')
    for field in ('actual_physics_seconds', 'actual_task_seconds'):
        value = alignment.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or abs(value - seconds) > 2e-5:
            raise MultitaskProtocolError('actual native clock differs: ' + field)
    if (alignment.get('native_success_latched') is not record['official_task_success']
            or alignment.get('native_terminal_failure_latched') is not record['native_terminal_failure']):
        raise MultitaskProtocolError('independent native outcome receipt differs')
    initial = record.get('recorded_initial_conditions', {})
    if initial.get('metric_only') is not True or initial.get('policy_use_allowed') is not False:
        raise MultitaskProtocolError('recorded parent-only reset conditions required')
    if expected_identity['split'] != 'test' and record.get('original_source_motion_replayed') is not False:
        raise MultitaskProtocolError('development generated resets must disclose non-replay')
    return dict(passed=True, identity_sha256=expected_identity['sha256'], ticks=ticks,
                demo_only=expected_identity['demo_only'], official_task_success=record['official_task_success'])


def native_completion_record(identity, *, process_id, success, fatal, ticks,
                             initial_conditions, alignment_receipt, policy_failure=None):
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
        task=identity['task'], split=identity['split'], raw_episode_id=identity['raw_episode_id'],
        parent_raw_episode_id=identity['parent_raw_episode_id'], method=identity['method'],
        process_id=int(process_id), complete=True, infrastructure_error=None,
        official_task_success=bool(success), native_terminal_failure=bool(fatal),
        formal_test_used=identity['split'] == 'test' and not identity['demo_only'],
        demo_only=identity['demo_only'], TEST_used_for_fitting=False,
        case_isolation='fresh_simulator_process_per_episode',
        original_source_motion_replayed=None if identity['split'] == 'test' else False,
        counted_clock=dict(ticks=int(ticks), dt=.05, seconds=float(ticks * .05)),
        recorded_initial_conditions=initial_conditions, alignment_receipt=alignment_receipt,
        policy_failure=policy_failure)
    validate_completed_case(record, identity)
    return record
