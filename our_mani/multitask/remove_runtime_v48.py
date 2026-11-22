"""Source-qualified remove operation model and current-public nearest roles."""
from pathlib import Path
from types import MethodType
import json
import numpy as np
from .evaluation import file_sha
from .pipeline import _fresh_direct
from .protocol import MultitaskProtocolError

SCHEMA='r10_source_qualified_remove_runtime_v48'
TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')


def checked(root,pointer):
    path=(Path(root)/pointer['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or file_sha(path)!=pointer['sha256']:raise MultitaskProtocolError('remove proof pointer changed')
    return path


def binding(root,protocol,task):
    if task not in TASKS:raise MultitaskProtocolError('remove runtime supports only the two declared tasks')
    settings=protocol['remove_runtime_v48'][task];path=checked(root,settings['qualification']);proof=json.loads(path.read_text())
    if not (proof['schema']==SCHEMA and proof['task']==task and proof['source_goals_reconstructed_passed'] is True
            and proof['original_parents']==100 and proof['all_numeric_partitions_passed'] is True
            and proof['all_original_censored_failed_parents_preserved'] is True):raise MultitaskProtocolError('remove qualification incomplete')
    pointer=protocol['learned_artifacts'][task]['interaction']
    if pointer!=proof['interaction_library']:raise MultitaskProtocolError('remove runtime/library source mismatch')
    checked(root,pointer)
    for key in ('source_goal_audit','current_full_curve_fit','current_prefit_proof'):
        checked(root,proof[key])
    return dict(schema=SCHEMA,task=task,qualification=settings['qualification'],interaction_library=pointer,
        source=dict(path=str(Path(__file__).resolve().relative_to(Path(root).resolve())),sha256=file_sha(__file__)),
        public_binding='nearest fresh visible cup to measured TCP, then nearest fresh visible spoke to that cup; committed cup identity retained',
        native_target_or_waypoint_read=False,full_orientation_promoted=False,original_gates_changed=False)


def nearest_roles(state,bindings,committed,maximum_age):
    result=dict(bindings);result.pop('selected_cup',None);result.pop('selected_spoke',None)
    fresh={body for body in state.bodies if _fresh_direct(state,body,maximum_age)}
    cups=sorted(body for body in fresh if body.startswith('cup') and body[3:].isdigit())
    spokes=sorted(body for body in fresh if body.startswith('spoke') and body[5:].isdigit())
    selected=committed.get('selected_cup',())
    if selected:
        if len(selected)!=1 or selected[0] not in state.bodies:return result
        cup=selected[0]
    elif cups:
        cup=min(cups,key=lambda name:(np.linalg.norm(state.bodies[name].pose.position-state.robot.pose.position),name))
    else:return result
    result['selected_cup']=(cup,)
    if spokes:
        spoke=min(spokes,key=lambda name:(np.linalg.norm(state.bodies[name].pose.position-state.bodies[cup].pose.position),name))
        result['selected_spoke']=(spoke,)
    return result


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol['remove_runtime_v48'][task].get('binding')!=expected:raise MultitaskProtocolError('freeze explicit public remove role adapter')
    previous=actor._roles
    def roles(instance,state):
        return nearest_roles(state,previous(state),getattr(instance.controller,'committed_roles',{}),
            instance.controller.generator.config.maximum_observation_age)
    actor._roles=MethodType(roles,actor)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),remove_runtime_v48=expected)
    actor.metrics_report=MethodType(metrics,actor)
