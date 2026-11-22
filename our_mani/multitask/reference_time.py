"""Explicit TRAIN reference-time derivation and causal committed references.

Offline evidence validation is separate from runtime: a selected command-issue
template uses its already selected causal forecast pose, with no task/GT read.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from ..contracts import Pose

SCHEMA = 'r10_explicit_reference_time_basis_v1'
RULE_SCHEMA = 'r10_TRAIN_basketball_approach_command_issue_rule_v1'
CURRENT = 'current'
COMMAND_ISSUE = 'command_issue'
COMMAND_GOAL = 'command_goal_reference'
ALLOWED = {task: {'approach': {'reference_role': 'ball', 'reference_kind': 'translation_only'}}
    for task in ('moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_basis(basis, kind, rule_sha256=None):
    require(basis in (CURRENT, COMMAND_ISSUE), 'unknown explicit reference_time_basis')
    if basis == COMMAND_ISSUE:
        require(kind == 'translation_only' and valid_sha(rule_sha256),
            'command_issue requires translation reference and bound TRAIN derivation rule SHA')
    else:
        require(rule_sha256 is None, 'current reference cannot carry a command_issue rule SHA')


def validate_template_basis(basis,kind,rule_sha256=None):
    # Raw TRAIN bindings/events keep the original stricter legacy contract.
    if basis==COMMAND_GOAL:
        require(kind in ('observed_se3','translation_only') and valid_sha(rule_sha256),
            'command-goal template needs physical geometry and explicit TRAIN rule SHA')
        return
    validate_basis(basis,kind,rule_sha256)


def selected_reference(candidate, rolling_reference):
    """Causal immutable reference after selection; current templates still roll."""
    template = candidate.template
    validate_template_basis(template.reference_time_basis, template.reference_kind,
        template.reference_time_rule_sha256)
    if template.reference_time_basis in (COMMAND_ISSUE,COMMAND_GOAL):
        require(isinstance(candidate.reference_pose, Pose), 'selected command_issue reference pose missing')
        return candidate.reference_pose, np.zeros(3), np.zeros(3)
    return rolling_reference()


@dataclass(frozen=True)
class VerifiedTRAINReferenceTimeRule:
    path: str
    sha256: str
    tasks: dict
    evidence_report: dict
    verified_files_sha256: dict


def load_rule(root, pointer):
    """Bind historical CPU evidence to archived code, immutable data to bytes."""
    root = Path(root).resolve()
    files = {}
    def checked(ptr):
        require(isinstance(ptr, dict) and set(ptr) == {'path', 'sha256'} and valid_sha(ptr['sha256']), 'strict reference-time evidence pointer required')
        path = (root / ptr['path']).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path) == ptr['sha256'], 'reference-time evidence bytes/path changed')
        files[str(path)] = ptr['sha256']
        return path
    path = checked(pointer)
    rule = json.loads(path.read_text())
    require(set(rule) == {'schema', 'TRAIN_only', 'VAL_used', 'TEST_used', 'tasks', 'evidence_report', 'source_snapshot'},
        'unexpected reference-time rule keys')
    require(rule['schema'] == RULE_SCHEMA and rule['TRAIN_only'] is True and rule['VAL_used'] is False
        and rule['TEST_used'] is False and rule['tasks'] == ALLOWED, 'unsupported TRAIN-only reference-time derivation')
    evidence_path = checked(rule['evidence_report'])
    evidence = json.loads(evidence_path.read_text())
    require(evidence.get('version') == 'r10_basketball_reference_time_derived_diagnostic_v1'
        and evidence.get('TRAIN_only') is True and evidence.get('VAL_TEST_read') is False
        and evidence.get('native_GPU_or_extra_replica_started') is False and set(evidence['tasks']) == set(ALLOWED),
        'incomplete or forbidden reference-time design evidence')
    archive = json.loads(checked(rule['source_snapshot']).read_text())
    require(archive.get('schema') == 'r10_CPU_reference_time_evidence_source_snapshot_v1'
        and archive.get('evidence_report') == rule['evidence_report'], 'historical source snapshot report differs')
    sources = {}
    for entry in archive['entries']:
        original = str((root / entry['original_path']).resolve())
        require(original not in sources, 'duplicate historical reference-time source')
        sources[original] = checked({'path': entry['path'], 'sha256': entry['sha256']})
    for filename, checksum in evidence['verified_files_sha256'].items():
        original = (root / filename).resolve()
        require(valid_sha(checksum), 'invalid reference-time evidence source SHA')
        actual = sources.get(str(original), original)
        require(actual.is_relative_to(root) and actual.is_file() and sha(actual) == checksum,
            'reference-time design input not preserved in actual historical source/data: ' + str(original))
        files[str(actual)] = checksum
    for task, row in evidence['tasks'].items():
        require(row['original_teacher_cases'] == row['original_distinct_parents'] == 100
            and row['all_original_cases_retained'] is True, 'reference-time rule needs complete original task100 TRAIN coverage')
        metric = row['endpoint_reference_comparisons']['issue_observed_ball']
        require(metric['parent_count'] >= 8 and metric['passes_existing_endpoint_caps'] is True
            and metric['maximum_position_tolerance_m'] == .03 and metric['maximum_angle_tolerance_rad'] == .15,
            'issue-time evidence does not pass unchanged TRAIN endpoint gates')
    return VerifiedTRAINReferenceTimeRule(str(path), pointer['sha256'], rule['tasks'], rule['evidence_report'], files)


def derive_source(source, case, rule):
    """Derive issue semantics after strict original source/ledger validation.

    Raw terminal_reference_xyzw remains the actual completion-time metric pose.
    The separate issued_reference_xyzw is bound to the original ledger/array.
    """
    from .interaction import TRAINReferenceBinding
    require(isinstance(rule, VerifiedTRAINReferenceTimeRule), 'verified TRAIN reference-time rule required')
    phases = rule.tasks.get(source.task_name, {})
    if not phases:
        return source, {'reference_time_basis': CURRENT, 'rule_applied': False}
    require(case.get('command_receipt_verified') is True and valid_sha(case.get('command_receipt_sha256')),
        'reference-time derivation needs SHA-verified original command receipt')
    require(source.source_id == case['entry']['source_id'] and source.grouping_source == case['entry']['parent_source_group'],
        'reference-time derivation source/case identity differs')
    commands = {row['command_id']: row for row in source.command_ledger}
    ids = tuple(map(str, case['arrays']['body_ids']))
    references = []
    for stage, binding in zip(source.stages, source.references):
        if stage == 'wp0' and binding is not None:
            require(binding.reference_role == 'ball' and binding.reference_kind == 'translation_only'
                and binding.reference_time_basis == CURRENT, 'original approach reference binding differs')
            binding = replace(binding, reference_time_basis=COMMAND_ISSUE, reference_time_rule_sha256=rule.sha256)
        references.append(binding)
    events = []
    changes = []
    for event in source.completion_events:
        if event.phase not in phases:
            events.append(event)
            continue
        require(event.binding.reference_time_basis == CURRENT and event.binding.reference_role == 'ball'
            and event.binding.reference_kind == 'translation_only', 'original completion binding differs')
        command = commands[event.origin_command_id]
        value = np.asarray(command.get('issued_reference_native_pose_xyzw'), float)
        require(value.shape == (7,) and np.isfinite(value).all() and abs(np.linalg.norm(value[3:]) - 1.) < 1e-6,
            'missing or invalid original issued reference pose')
        begin = np.flatnonzero(np.isclose(source.times, event.attempt_start_time, atol=1e-8, rtol=0))
        require(len(begin) == 1 and command['generated_at'] == event.attempt_start_time, 'issued reference has no exact original acquired clock')
        j = ids.index(event.binding.reference_id)
        require(case['arrays']['native_pose_valid'][int(begin[0]), j]
            and np.allclose(value[:3], case['arrays']['target_positions'][int(begin[0]), j], atol=1e-8, rtol=0),
            'issued reference differs from original TRAIN issue metric label')
        # Translation-only orientation is an explicit identity gauge, not GT R.
        issued = np.r_[value[:3], [0., 0., 0., 1.]]
        proof = dict(schema=SCHEMA, rule_sha256=rule.sha256, original_source_id=source.source_id,
            original_parent_source_group=source.grouping_source, command_id=event.origin_command_id,
            issued_at=event.attempt_start_time, command_receipt_sha256=case['command_receipt_sha256'],
            original_field='commands[command_id].issued_reference_native_pose_xyzw',
            original_terminal_metric_pose_preserved=True, source_masks_preserved=True)
        changed = replace(event, binding=replace(event.binding, reference_time_basis=COMMAND_ISSUE,
            reference_time_rule_sha256=rule.sha256), issued_reference_xyzw=issued,
            reference_time_provenance=proof)
        require(np.array_equal(changed.terminal_reference_xyzw, event.terminal_reference_xyzw), 'raw terminal metric label changed')
        events.append(changed)
        changes.append(proof)
    result = replace(source, references=tuple(references), completion_events=tuple(events))
    for field in ('times', 'expert_tcp_xyzw', 'expert_opening', 'measured_tcp_xyzw', 'command_ids', 'curve_tcp_xyzw'):
        require(np.array_equal(getattr(result, field), getattr(source, field)), 'reference derivation changed source arrays/masks')
    require(result.command_ledger == source.command_ledger and len(result.completion_events) == len(source.completion_events),
        'reference derivation changed raw ledger/completion count')
    return result, dict(schema=SCHEMA, reference_time_basis=COMMAND_ISSUE, rule_applied=True,
        rule=dict(path=rule.path, sha256=rule.sha256), derived_events=changes, original_raw_labels_and_masks_unchanged=True)
