"""Local author-wp4 adapter: submit the existing canonical terminal command.

The measured TRAIN curve and its command-goal completion geometry are distinct
contracts. This adapter only aligns ring insert/wp4-open terminal setpoints
with590's declared canonical continuation. It makes no universal library-bug,
physical seating, prediction-accuracy or Native-success claim.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ..contracts import Pose
from .interaction import PhaseTemplate, COMMAND_GOAL
from .grasp_frame_compensation_native_compat import CorrectedTemplate, compose, inverse
from . import author_ring_release_integration_v1140 as parent

SCHEMA = 'r10_author_ring_terminal_canonical_goal_integration_v1156'
SOURCE = 'our_mani/multitask/author_ring_terminal_goal_integration_v1156.py'
CHANGED_FACTOR = 'author_ring_terminal_canonical_waypoint'
RING_TASKS = parent.RING_TASKS
PHASES = ('insert', 'release')
COMPARISON_LOCKS = {
    'artifacts/framework/multitask_r10/TRAIN_author_release_full16_v1141_setup/source_overlay_lock.json':
        'cbb0f900d27270b2c41974827224ce105dff0f48d7714e5f65837b98dc3a26c8',
    'artifacts/framework/multitask_r10/TRAIN_block_preclose_full16_v1154_setup/source_overlay_lock.json':
        '0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9',
}
SOURCES = tuple(dict.fromkeys((SOURCE, *parent.SOURCES,
    'our_mani/contracts.py', 'our_mani/multitask/interaction.py',
    'our_mani/multitask/command_goal_reference.py',
    'our_mani/multitask/grasp_frame_compensation_native_compat.py',
    'our_mani/multitask/physmani_geometry_goals_v529.py',
    'our_mani/multitask/physmani_geometry_goals_v593.py')))


def _sha(root, name):
    return hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _template_digest(template):
    return _digest({f.name:(getattr(template,f.name).tolist()
        if isinstance(getattr(template,f.name),np.ndarray) else getattr(template,f.name))
        for f in fields(PhaseTemplate)})


def _comparisons(root):
    for name, expected in COMPARISON_LOCKS.items():
        if _sha(root,name)!=expected:raise ValueError('unchanged comparison source locks required')
    return [dict(path=name,sha256=sha) for name,sha in COMPARISON_LOCKS.items()]


def binding(root, native_repo):
    return dict(schema=SCHEMA,source_sha256=_sha(root,SOURCE),changed_factor=CHANGED_FACTOR,
        sources_sha256={name:_sha(root,name) for name in SOURCES},
        comparison_source_overlay_locks=_comparisons(root),parent=parent.binding(root,native_repo),
        changed_tasks=list(RING_TASKS),changed_phases=list(PHASES),
        target='original_row_pose compose inverse(actual_template_curve_end) compose actual_template_completion_geometry',
        compensation='both actual curve end and canonical already include the original measured delta',
        source_semantics='local590_declared_same_author_wp4_canonical_continuation_adapter',
        universal_COMMAND_GOAL_contract_bug_claimed=False,
        source_TRAIN_curve_or_weights_modified=False,original_lookahead_seconds=.5,
        original_author_position_guard_m=.03,new_numeric_thresholds=False,
        original_phase_done_qualify_ready_collision_and_Native_loop_unchanged=True,
        additional_observed_reference_predict_observe_Native_getter_or_success_calls=0,
        extra_drop_steps=0,unknown_or_unsupported_candidates='omitted_without_inventing_a_target',
        Native_success_or_physical_seating_guaranteed=False,formal_VAL_TEST_qualification=False)


def ring_rule_binding(root, config, task):
    if task not in RING_TASKS:raise ValueError('two original ring tasks required')
    inherited=parent.ring_rule_binding(root,config,task)
    pointer=deepcopy(config['learned_artifacts'][task]['interaction'])
    path=Path(root)/pointer['path']
    if (Path(pointer['path']).is_absolute() or '..' in Path(pointer['path']).parts
            or _sha(root,pointer['path'])!=pointer['sha256']):
        raise ValueError('unchanged project TRAIN interaction pointer required')
    payload=json.loads(path.read_text());metadata=payload['metadata']
    if (metadata.get('task_name')!=task or metadata.get('training_complete') is not True
            or metadata.get('fit_split')!='train' or metadata.get('selection_split')!='train'
            or metadata.get('TEST_used_for_fitting') is not False or metadata.get('synthetic') is not False):
        raise ValueError('completed original TRAIN interaction required')
    inserts=[PhaseTemplate(**row) for row in payload['templates'] if row['phase']=='insert']
    if (not inserts or len({t.anchor_id for t in inserts})!=len(inserts)
            or any(t.reference_kind!='observed_se3' or t.reference_time_basis!=COMMAND_GOAL
                or t.lag_seconds!=0. for t in inserts)):
        raise ValueError('explicit rigid original command-goal insert templates required')
    return dict(schema=SCHEMA,task=task,source_sha256=_sha(root,SOURCE),changed_factor=CHANGED_FACTOR,
        comparison_source_overlay_locks=_comparisons(root),interaction=pointer,
        original_insert_template_sha256={t.anchor_id:_template_digest(t) for t in inserts},
        inherited_author_stage_binding=inherited['inherited_author_stage_binding'],
        inherited_author_release_binding=inherited['inherited_author_release_binding'],
        inherited_author_release_and_semantic_rule=inherited,
        local_declared_canonical_continuation_source='our_mani/multitask/physmani_ring_release_v590.py',
        target_phases=list(PHASES),actual_template_reference_kind='observed_se3',
        actual_template_reference_time_basis=COMMAND_GOAL,original_lookahead_seconds=.5,
        original_author_position_guard_m=.03,original_delta_retained_once=True,
        source_curve_weights_thresholds_qualify_done_and_Native_execution_unchanged=True,
        additional_observed_reference_predict_or_Native_calls=0,extra_drop_steps=0,
        universal_COMMAND_GOAL_bug_or_success_guarantee=False)


def canonical_terminal_row(source, row, rule):
    """Pure already-public rigid-frame conversion; no reference/predictor query."""
    if type(row) is not dict or row.get('phase') not in PHASES:
        raise ValueError('original ring insert or wp4-open candidate required')
    template=row.get('template')
    if type(template) not in (PhaseTemplate,CorrectedTemplate):
        raise ValueError('known original or measured-compensated template required')
    base=template.source_template if type(template) is CorrectedTemplate else template
    if type(base) is not PhaseTemplate or (type(template) is CorrectedTemplate and type(template.tool_correction) is not Pose):
        raise ValueError('known single measured template correction required')
    if (source.task!=rule['task'] or template.phase!='insert' or base.phase!='insert'
            or template.reference_kind!='observed_se3' or template.reference_time_basis!=COMMAND_GOAL
            or template.lag_seconds!=0. or template.anchor_id!=base.anchor_id
            or _template_digest(base)!=rule['original_insert_template_sha256'].get(base.anchor_id)):
        raise ValueError('original frozen rigid command-goal insert template required')
    library=source.frontend.actor.controller.library
    if (not any(template is t for t in library.templates)
            or not any(base is t for t in source.original_templates)):
        raise ValueError('owned current and original TRAIN template membership required')
    if (type(row.get('pose')) is not Pose or type(row.get('present_pose')) is not Pose
            or type(source.frontend.state.robot.pose) is not Pose):
        raise ValueError('typed existing public candidate and robot poses required')
    if (row.get('lookahead_seconds')!=.5 or type(row.get('selected_at')) not in (float,int)
            or type(row.get('predicted_reference_time')) not in (float,int)
            or not np.isfinite([row['selected_at'],row['predicted_reference_time']]).all()
            or abs(row['predicted_reference_time']-row['selected_at']-.5)>1e-8):
        raise ValueError('unchanged original half-second zero-lag intent required')
    if (not isinstance(row.get('reference_id'),str) or not row['reference_id']
            or row.get('manipulated_ids')!=('ring',)):
        raise ValueError('original selected peg and single ring identity required')
    # at() is a pure local interpolation. For CorrectedTemplate this reads the
    # original local curve and composes its existing delta exactly once.
    local=template.at(1.)
    if type(local) is not tuple or len(local)!=4 or type(local[0]) is not Pose:
        raise ValueError('known local endpoint pose/velocity/angular/opening contract required')
    curve=local[0];canonical=Pose.from_xyzw(template.completion_geometry_xyzw)
    offset=compose(inverse(curve),canonical)
    target=compose(row['pose'],offset);present=compose(row['present_pose'],offset)
    score=-float(np.linalg.norm(target.position-source.frontend.state.robot.pose.position))
    return dict(row,pose=target,present_pose=present,score=score), dict(
        phase=row['phase'],anchor_id=template.anchor_id,reference_id=row['reference_id'],
        selected_at=float(row['selected_at']),predicted_reference_time=float(row['predicted_reference_time']),
        original_pose_xyzw=row['pose'].xyzw().tolist(),terminal_pose_xyzw=target.xyzw().tolist(),
        original_present_pose_xyzw=row['present_pose'].xyzw().tolist(),
        terminal_present_pose_xyzw=present.xyzw().tolist(),
        actual_curve_end_xyzw=curve.xyzw().tolist(),actual_canonical_xyzw=canonical.xyzw().tolist(),
        measured_compensation_already_in_template=type(template) is CorrectedTemplate,
        score=score,additional_reference_or_prediction_queries=0)


@contextmanager
def owned_terminal_goal_scope(source, rule, *, events=None, max_events=1000):
    if type(max_events) is not int or max_events<=0:raise ValueError('positive bounded event capacity required')
    marker='_author_ring_terminal_goal_v1156'
    if marker in vars(source):raise RuntimeError('owned terminal-goal scopes cannot nest')
    original_candidates=source.candidates;original_receipt=source.receipt
    trace=dict(schema=SCHEMA,changed_factor=CHANGED_FACTOR,rule=deepcopy(rule),events=[],
        candidate_calls=0,terminal_candidate_calls=0,original_exceptions=0,
        converted_candidates=0,unknown_candidates=0,dropped_events=0,restored=False,
        additional_reference_predict_observe_Native_getter_or_success_calls=0)
    def record(event):
        if len(trace['events'])<max_events:trace['events'].append(event)
        else:trace['dropped_events']+=1
    def candidates(phase):
        trace['candidate_calls']+=1
        try:rows=original_candidates(phase)
        except BaseException:
            trace['original_exceptions']+=1;raise
        if phase not in PHASES:return rows
        trace['terminal_candidate_calls']+=1
        if type(rows) is not list:
            trace['unknown_candidates']+=1
            record(dict(phase=phase,applied=False,reason='nonstandard_original_candidates_return'))
            return rows
        converted=[]
        for row in rows:
            try:
                if type(row) is not dict or row.get('phase')!=phase:
                    raise ValueError('candidate phase must match original dispatch phase')
                new,evidence=canonical_terminal_row(source,row,rule)
                converted.append(new);trace['converted_candidates']+=1
                record(dict(applied=True,**evidence))
            except (ValueError,KeyError,TypeError,AttributeError,IndexError) as error:
                trace['unknown_candidates']+=1
                record(dict(phase=phase,applied=False,reason=str(error),target_invented=False))
        return sorted(converted,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))
    def receipt():
        return dict(original_receipt(),author_ring_terminal_goal_integration_v1156=deepcopy(trace))
    original_error=None
    try:
        with patch.object(source,marker,trace,create=True),patch.object(source,'candidates',candidates), \
             patch.object(source,'receipt',receipt):
            yield source
    except BaseException as error:
        original_error=error;raise
    finally:
        trace['restored']=(source.candidates==original_candidates and source.receipt==original_receipt
            and marker not in vars(source))
        if events is not None:events.update(deepcopy(trace))
        if not trace['restored'] and original_error is None:
            raise RuntimeError('owned original candidates and receipt were not restored')


@contextmanager
def integrated_components(root, config, task, actor, *, events=None, max_events=1000):
    if task not in RING_TASKS:
        with parent.integrated_components(root,config,task,actor,events=events,max_events=max_events) as triple:
            yield triple
        return
    parent_events={};terminal_events={}
    try:
        with parent.integrated_components(root,config,task,actor,events=parent_events,max_events=max_events) as triple:
            frontend,dispatched,executor=triple
            if executor.skill_policy is not dispatched:raise RuntimeError('owned final1140 dispatch required')
            source=parent._source(dispatched)
            rule=ring_rule_binding(root,config,task)
            with owned_terminal_goal_scope(source,rule,events=terminal_events,max_events=max_events):
                yield triple
    finally:
        if events is not None:events.update(schema=SCHEMA,parent_scope=parent_events,
            terminal_goal=terminal_events,
            restored=bool(parent_events.get('restored') and terminal_events.get('restored')))
