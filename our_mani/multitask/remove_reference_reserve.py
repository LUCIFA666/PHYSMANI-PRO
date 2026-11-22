"""Actor-local migration to the existing four-unit reference planning service.

The manager, dispatch ledger, confirmation callbacks, phase state and native
six-unit model retain their identities. Only the reference proof operations
are replaced, before execution, with the already used v86 implementation.
"""
from pathlib import Path
from copy import deepcopy
from types import MethodType
import numpy as np
from .task_reference_v86 import (reserved_reference_gate,ContinuousReferenceManager,
    _DualProofReachability)
from .reference_stop_guard_v274 import NextReferenceStopGuard
from .continuous_transit import ConstraintContract,install_transit
from .evaluation import file_sha
TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')
SOURCES=('our_mani/multitask/remove_reference_reserve.py','our_mani/multitask/task_reference_v86.py',
    'our_mani/multitask/reference_constraint_authority.py','our_mani/multitask/continuous_transit.py',
    'our_mani/multitask/bounded_task_transit.py')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit remove reference service')
    return dict(schema='r10_actor_local_existing_reference4_service_upgrade_v1',task=task,
        phases=['approach','acquire','detach','transport','place'],replan_seconds=.2,
        previous_reference_acceleration_limit=6.,actual_reference_acceleration_limit=4.,actual_native_acceleration_limit=6.,
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        reference_full_proof_next_step_and_stopping_service_share_same_bound=True,
        actual_motor_parameters_collision_grasp_phase_completion_and10s_unchanged=True,
        no_class_global_factory_or_active_job_mutation=True,formal_VAL_TEST=False)


def install(actor,config,root,task):
    expected=binding(root,config,task)
    if config.get('remove_reference_reserve')!=expected:raise ValueError('freeze reserve service upgrade')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations or m.plan is not None:
        raise ValueError('upgrade before first execution')
    if c.library.task_spec.task_name!=task:raise ValueError('actual task library differs')
    if hasattr(m,'reference_reserve_gate') or not np.allclose(m.native_gate.model.acceleration_limits,6.):
        raise ValueError('expected original reference6/native6 manager')
    function=getattr(m.make_brake,'__func__',m.make_brake)
    guards=[cell.cell_contents for cell in (getattr(function,'__closure__',None)or())
        if isinstance(cell.cell_contents,NextReferenceStopGuard)]
    if len(guards)!=1 or not hasattr(guards[0],'constraint_contract'):
        raise ValueError('actual consistent actor-local stop service required')
    guard=guards[0]
    if guard.events or guard.pending_stop or guard.confirmed_stop:raise ValueError('stopping service already active')
    m.reference_reserve_gate=reserved_reference_gate(m.native_gate)
    m.dual_gate=_DualProofReachability(m)
    c.original_reachability=c.generator.reachability=m.dual_gate
    # v37 and v86 have identical state/boundary/rolling methods. This replaces
    # the step proof, then reconnects the same stop service to that proof.
    m.check_stream=MethodType(ContinuousReferenceManager.check_stream,m)
    guard.constraint_contract=ConstraintContract.from_manager(m)
    guard.install_stream()
    install_transit(actor,phases=tuple(expected['phases']),replan_seconds=expected['replan_seconds'])
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),remove_reference_reserve=dict(binding=deepcopy(expected),
        actual_contract=deepcopy(guard.constraint_contract.__dict__),
        actual_confirmation_service_reused=True,actual_manager_object_reused=True,
        previous_reference6_binding_explicitly_superseded=True))
    actor.metrics_report=MethodType(metrics,actor)
    return guard
