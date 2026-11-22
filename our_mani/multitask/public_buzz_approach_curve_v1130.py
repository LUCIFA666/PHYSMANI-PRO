"""Candidate policy: retain the frozen TRAIN Buzz approach curve.

Only approach pose selection changes. A cursor passes a knot only when the
current public TCP is inside the original author's approach distance guard.
There is no time-based replay, new numerical guard, forecast call, native
condition input, gripper change or claim that this candidate solves Buzz.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import ast
import hashlib
import json

import numpy as np

from ..contracts import Pose, RobotState
from .interaction import InteractionLibrary, PhaseTemplate, COMMAND_GOAL, _world_target
from .physmani_skill_policy_v548 import APPROACH_DISTANCE
from .physmani_terminal_opening_v620 import BuzzCurveIntents
from .physmani_buzz_curve_v602 import binding as original_curve_binding
from .specs import TASK_SPECS

SCHEMA = 'r10_candidate_public_TRAIN_Buzz_approach_curve_v1130'
SOURCE = 'our_mani/multitask/public_buzz_approach_curve_v1130.py'
TASKS = ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')
PINS = {
    TASKS[0]: dict(
        interaction='711fe5b2c4ef6f915a01dec801416eb05b491ea4bbc4c44a84bf567227ce8ec8',
        geometry='a50d00d94ee581236b2239b743d18bffdbb805884672ef8b0750e27857962c02'),
    TASKS[1]: dict(
        interaction='fe448d7afb66cf2813cd72bb2e89d44995c6cff38bf65bf118d8deb75a1ea3fa',
        geometry='454565024081fcbf0ce9be13a122a35f9c6ddd0f4904092da0328d7b1afe5833'),
}
ORIGINAL_SOURCES = tuple('our_mani/multitask/' + name for name in (
    'interaction.py', 'specs.py', 'physmani_skill_policy_v548.py',
    'physmani_geometry_goals_v593.py', 'physmani_buzz_curve_v602.py',
    'physmani_terminal_opening_v620.py', 'physmani_buzz_frontend_v586.py'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _template_sha(template):
    fields = dict(vars(template))
    fields = {key: value.tolist() if isinstance(value, np.ndarray) else value
              for key, value in fields.items()}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _approach_templates(library, task):
    if (type(library) is not InteractionLibrary or library.task_spec.task_name != task
            or library.metadata.get('synthetic') is not False
            or library.metadata.get('fit_split') != 'train'
            or library.metadata.get('TEST_used_for_fitting') is not False):
        raise ValueError('the unchanged completed real TRAIN Buzz library is required')
    templates = library.for_phase('approach')
    if not templates:
        raise ValueError('frozen TRAIN approach templates required')
    for t in templates:
        if (type(t) is not PhaseTemplate or len(t.progress) != 21
                or not np.array_equal(t.progress, np.linspace(0., 1., 21))
                or t.reference_kind != 'observed_se3' or t.reference_role != 'wand'
                or t.manipulated_role != 'wand' or t.reference_mode != 'external'
                or t.reference_time_basis != COMMAND_GOAL or t.lag_seconds != 0.
                or not np.array_equal(t.opening, np.ones(21))):
            raise ValueError('unaltered 21-knot open-gripper CAD-frame approach required')
    return templates


def binding(root, config, task):
    """Return a JSON rule bound to existing TRAIN bytes and CAD-frame inputs."""
    if task not in TASKS:
        raise ValueError('explicit two Buzz tasks only')
    root = Path(root)
    pointers = {}
    for kind in ('interaction', 'geometry'):
        pointer = config['learned_artifacts'][task][kind]
        path = (root / pointer['path']).resolve()
        path.relative_to(root.resolve())
        if pointer['sha256'] != PINS[task][kind] or _sha(path) != pointer['sha256']:
            raise ValueError('frozen TRAIN artifact bytes changed: ' + kind)
        pointers[kind] = deepcopy(pointer)
    registration = config['public_buzz_frontend_v586'][task]['original_conditional_registration']
    if registration['geometry'] != pointers['geometry']:
        raise ValueError('TRAIN interaction and public registration must keep the same CAD frame')
    original_rule = original_curve_binding(root, config['baseline_repo'], task)
    if config['public_Buzz_curve_v602'][task] != original_rule:
        raise ValueError('original follow/opening rule changed')
    native = Path(config['baseline_repo']) / TASK_SPECS[task].native_source_relative_path
    if _sha(native) != TASK_SPECS[task].native_source_sha256:
        raise ValueError('original author task source changed')
    expert = next(n for n in ast.parse(native.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'get_expert_info')
    guards = [ast.literal_eval(n.value) for n in ast.walk(expert)
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'th_wp0']
    guard = APPROACH_DISTANCE[TASK_SPECS[task].family]
    if guards != [guard] or original_rule['approach_lookahead_seconds'] != 0.:
        raise ValueError('the original author approach distance and zero lookahead are required')
    library = InteractionLibrary.load(root / pointers['interaction']['path'])
    templates = _approach_templates(library, task)
    return dict(schema=SCHEMA, task=task, candidate_not_validated_as_solution=True,
        source_sha256=_sha(root / SOURCE), artifacts=pointers,
        library_canonical_sha256=library.sha256,
        templates_sha256={t.anchor_id: _template_sha(t) for t in templates},
        native_task_sha256=_sha(native),
        original_sources_sha256={name: _sha(root / name) for name in ORIGINAL_SOURCES},
        current_public_progress_distance_m=guard, approach_lookahead_seconds=0.,
        number_of_TRAIN_knots=21, new_numeric_thresholds=False,
        cursor_key=['anchor_id', 'reference_id', 'manipulated_ids'],
        progress_requires_current_public_TCP=True, elapsed_time_advances_progress=False,
        selected_reference_and_CAD_frame_changed=False, nonapproach_exact_delegate=True,
        original_approach_completion_guard_retained=True,
        original_opening_preclose_association_actuator_and_outcome_unchanged=True,
        extra_prediction_or_native_getter_or_condition_calls=0,
        global_budget_extended=False, formal_VAL_TEST_qualification=False)


def _check_live(policy, rule):
    if not isinstance(policy, BuzzCurveIntents) or policy.task not in TASKS:
        raise TypeError('one original terminal-opening Buzz policy instance required')
    library = policy.frontend.actor.controller.library
    templates = _approach_templates(library, policy.task)
    if (rule.get('schema') != SCHEMA or rule.get('task') != policy.task
            or rule.get('library_canonical_sha256') != library.sha256
            or rule.get('templates_sha256') != {t.anchor_id: _template_sha(t) for t in templates}
            or rule.get('current_public_progress_distance_m') != APPROACH_DISTANCE['beat_the_rotating_buzz']
            or rule.get('approach_lookahead_seconds') != 0.
            or rule.get('number_of_TRAIN_knots') != 21
            or any(rule.get('artifacts', {}).get(k, {}).get('sha256') != PINS[policy.task][k]
                   for k in ('interaction', 'geometry'))):
        raise ValueError('live policy differs from the source-bound TRAIN approach rule')


def derive_buzz_approach_rule(root, config, task, source_policy):
    rule = binding(root, config, task)
    _check_live(source_policy, rule)
    return rule


@contextmanager
def owned_buzz_approach_scope(source_policy, rule, *, events=None, max_events=1000):
    """Adapt only one policy's approach, restoring all owned methods on exit."""
    _check_live(source_policy, rule)
    rule = deepcopy(rule)
    if type(max_events) is not int or not 1 <= max_events <= 10000:
        raise ValueError('bounded approach trace required')
    marker = '_owned_buzz_approach_scope_v1130'
    if marker in vars(source_policy):
        raise RuntimeError('Buzz approach scopes cannot nest')
    original_candidates = source_policy.candidates
    original_done, original_receipt = source_policy._done, source_policy.receipt
    cursors, trace = {}, []
    counts = dict(approach_calls=0, nonapproach_candidate_delegations=0,
                  nonapproach_done_delegations=0, dropped_events=0)

    def append(row):
        if len(trace) < max_events:
            trace.append(row)
        else:
            counts['dropped_events'] += 1

    def candidates(owned, phase):
        rows = original_candidates(phase)
        if phase != 'approach':
            counts['nonapproach_candidate_delegations'] += 1
            return rows
        counts['approach_calls'] += 1
        if not rows:
            return rows
        f = owned.frontend
        if type(f.state.robot) is not RobotState or type(f.state.robot.pose) is not Pose:
            raise TypeError('existing strict public TCP DTO required')
        result = []
        for row in rows:
            t = row['template']
            if (type(t) is not PhaseTemplate or _template_sha(t) != rule['templates_sha256'].get(t.anchor_id)
                    or row['predicted_reference_time'] != row['selected_at']
                    or row['selected_at'] != float(f.state.robot.timestamp)
                    or row['lookahead_seconds'] != 0. or row['opening'] != 1.):
                raise ValueError('original zero-lookahead open approach candidate required')
            reference = f.actor.controller.generator.observed_reference(
                f.state, row['reference_id'], t, owned.origin)
            key = (t.anchor_id, row['reference_id'], tuple(row['manipulated_ids']))
            index = cursors.get(key, 0)
            before = index
            while index < len(t.progress) - 1:
                present = _world_target(reference.pose, reference.linear_velocity,
                    reference.angular_velocity, t, float(t.progress[index]))[0]
                if float(np.linalg.norm(f.state.robot.pose.position - present.position)) > rule['current_public_progress_distance_m']:
                    break
                index += 1
            cursors[key] = index
            progress = float(t.progress[index])
            target = _world_target(reference.pose, reference.linear_velocity,
                reference.angular_velocity, t, progress)[0]
            distance = float(np.linalg.norm(f.state.robot.pose.position - target.position))
            result.append(dict(row, pose=target, present_pose=target,
                curve_index=index, curve_progress=progress,
                score=-distance))
            append(dict(event='current_public_TRAIN_approach_knot',
                time=float(f.state.robot.timestamp), anchor_id=t.anchor_id,
                reference_id=row['reference_id'], manipulated_ids=list(row['manipulated_ids']),
                cursor_before=before, curve_index=index, curve_progress=progress,
                number_of_TRAIN_knots=len(t.progress),
                current_public_distance_to_selected_knot_m=distance,
                original_author_distance_guard_m=rule['current_public_progress_distance_m'],
                selected_current_public_pose_xyzw=target.xyzw().tolist(),
                native_condition_or_future_observation_used=False))
        return sorted(result, key=lambda r: (-r['score'], r['reference_id'],
            r['manipulated_ids'], r['template'].anchor_id))

    def done(owned, goal):
        if goal['phase'] != 'approach':
            counts['nonapproach_done_delegations'] += 1
            return original_done(goal)
        key = (goal['template'].anchor_id, goal['reference_id'], tuple(goal['manipulated_ids']))
        if (goal.get('curve_index') != 20 or cursors.get(key) != 20):
            return False
        return original_done(goal)

    def own_receipt():
        return dict(schema=SCHEMA, rule=deepcopy(rule), events=deepcopy(trace), **counts,
            cursors=[dict(anchor_id=k[0], reference_id=k[1], manipulated_ids=list(k[2]),
                          curve_index=v) for k, v in sorted(cursors.items())],
            native_success_claimed=False, extra_prediction_or_native_getter_or_condition_calls=0,
            global_budget_extended=False, candidate_not_validated_as_solution=True)

    def receipt(owned):
        result = dict(original_receipt())
        result['public_buzz_approach_curve_v1130'] = own_receipt()
        return result

    try:
        with ExitStack() as scopes:
            scopes.enter_context(patch.object(source_policy, marker, True, create=True))
            for name, method in (('candidates', candidates), ('_done', done), ('receipt', receipt)):
                scopes.enter_context(patch.object(source_policy, name, MethodType(method, source_policy)))
            yield source_policy
    finally:
        restored = (source_policy.candidates == original_candidates
            and source_policy._done == original_done and source_policy.receipt == original_receipt
            and marker not in vars(source_policy))
        if events is not None:
            events.update(schema=SCHEMA, restored=restored, approach_receipt=own_receipt())
