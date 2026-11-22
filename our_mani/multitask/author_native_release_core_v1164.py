"""Retain558/602 author first-release commands in the full1140 scope.

The parent has already called the raw qualifier exactly once.  Only a closed
to requested-open deposit/release command uses that same result instead of1114's
requested-opening passthrough.  Zero means keep closed, never a new close edge.
No target, distance, grasp identity, prediction or physical API is queried here.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

from . import author_ring_release_integration_v1140 as parent
from . import physmani_basketball_release_v558 as ball
from . import physmani_buzz_curve_v602 as buzz
from .author_aligned_geometry_executor_v1114 import ProtocolInvariantError, _plain
from .retained_preclose_policy_integration_v1121 import RetainedPrecloseWitness

SCHEMA='r10_retained_author_native_release_core_integration_v1164'
CHANGED_FACTOR='retained_original_author_first_release_qualification'
SOURCE='our_mani/multitask/author_native_release_core_v1164.py'
TEST_SOURCE='tests/test_author_native_release_core_v1164.py'
BASKETBALL_TASKS=('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
BUZZ_TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
TASKS=(*BASKETBALL_TASKS,*BUZZ_TASKS)
BALL_SOURCE='our_mani/multitask/physmani_basketball_release_v558.py'
BUZZ_SOURCE='our_mani/multitask/physmani_buzz_curve_v602.py'
PINS={parent.SOURCE:'1ece8e678c59363ff4d5085ddf2659a7ba5dce3810dcc2ee771974ce09a30486',
      BALL_SOURCE:'a4db097924c6fbdc33620a89eb4c9410e430ccb677c9a232df99a0b59919b10f',
      BUZZ_SOURCE:'2775bcbf49967d4989598302e2f38562e793a59c30f6635001526368bedc1867',
      'our_mani/multitask/retained_preclose_policy_integration_v1121.py':
          '37b7e1989485262db2d4d3e1c9a2468b64fb057fe73df86a8e3edb5fe020f6e7'}
SOURCES=tuple(dict.fromkeys((SOURCE,*parent.SOURCES,*PINS,
    'our_mani/multitask/author_aligned_geometry_executor_v1114.py')))


def _sha(root,name):return hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()


def _pins(root):
    if any(_sha(root,name)!=expected for name,expected in PINS.items()):
        raise ProtocolInvariantError('frozen1140/1121/558/602 sources required')


def binding(root,native_repo):
    _pins(root)
    return dict(schema=SCHEMA,source_sha256=_sha(root,SOURCE),changed_factor=CHANGED_FACTOR,
        sources_sha256={name:_sha(root,name) for name in SOURCES},parent=parent.binding(root,native_repo),
        changed_tasks=list(TASKS),changed_edge='author_first_deposit_or_release_closed_to_requested_open_only',
        original_author_rules={task:(ball if task in BASKETBALL_TASKS else buzz).binding(root,native_repo,task) for task in TASKS},
        original_qualifier_calls_per_counted_command=1,new_numeric_thresholds=False,
        existing_current_public3cm_qualified_result_retained=True,
        original_acquire_close_ring_semantics_Buzz_approach_follow_and_other_routes_preserved=True,
        ring590_second_release_gate_not_restored=True,
        full_parent1140_construction_and_lifetime_retained=True,
        Native_RRT_RML_gripper_actuator_physics_and_outcome_sampling_unchanged=True,
        additional_native_getters_observe_predict_predicate_calls=0,
        Native_grasp_identity_verified=False,release_success_guaranteed=False,
        formal_VAL_TEST_qualification=False)


def task_rule_binding(root,config,task):
    _pins(root)
    if task not in TASKS:raise ValueError('four original first-release tasks only')
    basketball=task in BASKETBALL_TASKS
    module,key,phase=((ball,'author_basketball_release_v558','deposit') if basketball
                      else (buzz,'public_Buzz_curve_v602','release'))
    rule=module.binding(root,config['baseline_repo'],task)
    if config.get(key,{}).get(task)!=rule:
        raise ProtocolInvariantError('unchanged configured source558/602 first-release rule required')
    return dict(schema=SCHEMA,task=task,source_sha256=_sha(root,SOURCE),changed_factor=CHANGED_FACTOR,
        inherited_author_release_binding=rule,controlled_phase=phase,
        source_family='basketball' if basketball else 'Buzz',
        original_current_public_qualification_reused=True,new_numeric_thresholds=False,
        no_extra_target_or_distance_reads=True,original_acquire_close_rule_unchanged=True,
        actual_open_required_for_release_notification=True,Native_capture_or_success_guaranteed=False)


def basketball_rule_binding(root,config,task):
    if task not in BASKETBALL_TASKS:raise ValueError('two basketball tasks only')
    return task_rule_binding(root,config,task)


def buzz_rule_binding(root,config,task):
    if task not in BUZZ_TASKS:raise ValueError('two Buzz tasks only')
    return task_rule_binding(root,config,task)


def ring_rule_binding(root,config,task):
    return parent.ring_rule_binding(root,config,task)


def _binary(value):
    return type(value) in (float,int,bool) and math.isfinite(value) and value in (0,1)


class RetainedAuthorReleaseWitness:
    def __init__(self,dispatched,rule,*,max_events=1000):
        if not isinstance(dispatched,RetainedPrecloseWitness):
            raise ProtocolInvariantError('existing1121 final dispatch witness required')
        source=dispatched.source_policy
        basketball=source.task in BASKETBALL_TASKS
        expected_type=ball.AuthorBasketballReleaseIntents if basketball else buzz.BuzzCurveIntents
        expected_source=BALL_SOURCE if basketball else BUZZ_SOURCE
        phase='deposit' if basketball else 'release'
        if not isinstance(source,expected_type) or source.task not in TASKS:
            raise ProtocolInvariantError('owned original558/602 policy required')
        inherited=rule.get('inherited_author_release_binding') if type(rule) is dict else None
        threshold_key='release_distance_m' if basketball else 'author_terminal_release_distance_m'
        if (type(rule) is not dict or rule.get('schema')!=SCHEMA or rule.get('task')!=source.task
                or rule.get('changed_factor')!=CHANGED_FACTOR or type(inherited) is not dict
                or rule.get('controlled_phase')!=phase
                or inherited.get('task')!=source.task or inherited.get(threshold_key)!=.03
                or inherited.get('source_sha256')!=PINS[expected_source]):
            raise ProtocolInvariantError('source-bound1164 first-release rule required')
        if type(max_events) is not int or not 1<=max_events<=10000:
            raise ValueError('bounded qualification receipt required')
        self.parent_policy=dispatched;self.source_policy=source;self.rule=deepcopy(rule)
        self.phase=phase
        self.reasons=(('original_author_current_public3cm_release_guard','defer_release_until_author_current_public3cm_guard')
                      if basketball else ('original_author_Buzz_current_public3cm_release','defer_Buzz_current_public3cm_release'))
        self.max_events=max_events;self.events=[];self.dropped_events=0

    def __getattr__(self,name):return getattr(self.parent_policy,name)

    def qualify_opening(self,requested,current_requested):
        selected=self.source_policy.selected
        retain=(type(selected) is dict and selected.get('phase')==self.phase
                and requested>=.5 and current_requested<.5)
        # The actual1121 parent owns the sole original qualifier call. It keeps
        # the firstacquire close gate and returns its proposal unchanged there.
        command,proof,proposed=self.parent_policy.qualify_opening(requested,current_requested)
        if retain:
            legacy=proof.get('legacy_qualification') if type(proof) is dict else None
            raw=legacy.get('proof') if type(legacy) is dict else None
            qualified=legacy.get('qualified_opening') if type(legacy) is dict else None
            if (not _binary(qualified) or type(raw) is not dict
                or raw.get('phase')!=self.phase or raw.get('qualified_opening')!=qualified
                or raw.get('native_truth_read') is not False
                or raw.get('reason') not in self.reasons
                or proposed is not None):
                raise ProtocolInvariantError('same-call original558/602 release command/proof required')
            command=float(qualified)
            proof=dict(_plain(raw),legacy_qualification=deepcopy(legacy),
                dispatch_variant=SCHEMA,original_public_release_qualification_retained=True,
                legacy_geometric_qualification_is_diagnostic_only=False,
                requested_opening=float(requested),qualified_opening=command,
                native_capture_claimed=False,Native_grasp_identity_verified=False)
        event=dict(retained_original_release=bool(retain),requested_opening=float(requested),
            current_requested_opening=float(current_requested),qualified_opening=float(command),
            proof=_plain(proof),proposed=_plain(proposed),
            evidence_kind='qualification_only_not_applied_or_confirmed')
        if len(self.events)<self.max_events:self.events.append(event)
        else:self.dropped_events+=1
        return command,proof,proposed

    def confirm_opening(self,opening,proof,proposed,*,confirmed_after):
        # Called by unchanged1114 only after original Native apply succeeds.
        # The parent callback sees actual0 on a deferredopen, never requested1.
        return self.parent_policy.confirm_opening(opening,proof,proposed,confirmed_after=confirmed_after)

    def receipt(self):
        result=dict(self.parent_policy.receipt())
        physical=dict(result.get('physical_grasp_gate',{}))
        physical['release_and_other_openings']='author_first_deposit_or_release_original558_or602_qualification; other_edges_exact_parent1121'
        result['physical_grasp_gate']=physical
        result['author_native_release_core_v1164']=dict(schema=SCHEMA,rule=deepcopy(self.rule),
            changed_factor=CHANGED_FACTOR,qualification_events=deepcopy(self.events),
            dropped_qualifications=self.dropped_events,
            actual_parent_qualifier_calls_per_counted_command=1,
            actual_confirm_events_owned_by_retained_preclose_dispatch_v1121=True,
            qualification_does_not_confirm_dispatch_or_release=True,
            confirmed_witness_only_after_original_apply=True,
            deferred_zero_keeps_closed_without_new_close_proposal=True,
            already_open_never_reclosed_by_this_rule=True,additional_native_calls=0,
            Native_grasp_identity_verified=False,release_success_guaranteed=False)
        return result


@contextmanager
def integrated_components(root,config,task,actor,*,events=None,max_events=1000):
    # Other routes receive the exact same triple and receipts. The
    # parent retains its complete semantic/ring/Buzz scope until our exit.
    if events is not None and type(events) is not dict:raise TypeError('plain optional receipt sink required')
    with parent.integrated_components(root,config,task,actor,events=events,max_events=max_events) as triple:
        if task not in TASKS:
            yield triple
            return
        frontend,dispatched,executor=triple
        if executor.skill_policy is not dispatched:
            raise ProtocolInvariantError('owned1140 final executor policy required')
        rule=task_rule_binding(root,config,task)
        active=RetainedAuthorReleaseWitness(dispatched,rule,max_events=max_events)
        original_error=None
        try:
            with patch.object(executor,'skill_policy',active):
                yield frontend,active,executor
        except BaseException as error:
            original_error=error;raise
        finally:
            restored=executor.skill_policy is dispatched
            if events is not None:
                events.update(schema=SCHEMA,restored=restored,
                    qualification_events=deepcopy(active.events),dropped_qualifications=active.dropped_events)
            if not restored and original_error is None:
                raise ProtocolInvariantError('owned1164 executor dispatch did not restore')
