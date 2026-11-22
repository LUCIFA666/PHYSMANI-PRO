"""Optional TRAIN-only private true-clock labels; never policy observations.

Installation is explicit. Reading label bodies or grasp state occurs only in
an actual TRAIN backend after Root starts Native. This module does not launch,
step, restore, or construct a model. Static CF zero-clock captures are not
accepted as motion ledgers. Unknown contact force is never invented.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np

from ..local_mpc_20261102.executor import ShortHorizonExecutor
from ..multitask.specs import get_task_spec
from ..multitask.native_tasks import resolve_native_roles
from .contracts import vector, retained_gripper_command


def resolve_training_bodies(task, task_name, scene_object_resolver):
    """TaskSpec role identity for labels only, no expert-selected role lookup."""
    if task_name in ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed'):
        return {'target_block': task.target_block}, dict(dynamic_roots=['target_block'],
            manipulated_roles=['target_block'],selected_role_observed=False)
    spec = get_task_spec(task_name)
    names = tuple(dict.fromkeys(spec.dynamic_roots + spec.primary_manipulated_roles))
    roles = tuple(r for r in spec.roles if r.name in names)
    if {r.name for r in roles} != set(names) or any(r.kind != 'body' for r in roles):
        raise ValueError('All TRAIN dynamic/manipulated body identities must be explicit')
    selected = replace(spec, roles=roles)
    return resolve_native_roles(task, selected, scene_object_resolver=scene_object_resolver), dict(
        dynamic_roots=list(spec.dynamic_roots),manipulated_roles=list(spec.primary_manipulated_roles),
        selected_role_observed=False)


def _plain(value):
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, dict): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): raise ValueError('Finite TRAIN label required')
    if value is None or isinstance(value,(str,int,float,bool)): return value
    raise TypeError('TRAIN labels must be JSON serializable, no Native handles/objects')


class TRAINLabelRecorder:
    """One initial row then exactly one row after each committed real tick.

    The sink is a dedicated private label stream owned by the future collector.
    A failed sink/clock/identity check permanently faults this recorder; no
    silent retry or missing-row repair is allowed.
    """
    def __init__(self, backend, sink):
        if getattr(backend,'dataset_use',None) != 'train' or not callable(sink):
            raise ValueError('Explicit TRAIN-only backend and separate private label sink required')
        self.backend,self.sink=backend,sink
        self._last=None;self._fault=None;self.rows=0

    def record_initial(self, actuator_h):
        if self._last is not None: raise ValueError('Initial TRAIN row is single-use')
        return self._record(actuator_h, initial=True)

    def record_after_commit(self, actuator_h):
        if self._last is None: raise ValueError('Record actual initial row before any physics')
        return self._record(actuator_h, initial=False)

    def _record(self, actuator_h, *, initial):
        if self._fault: raise RuntimeError('TRAIN recorder permanently faulted: '+self._fault)
        try:
            snapshot=self.backend._training_label_snapshot()
            tick=snapshot['physics_tick'];clock=snapshot['actual_relative_physics_timestamp']
            if type(tick) is not int or tick < 0 or not np.isfinite(clock):
                raise ValueError('Actual Native timestamp and integer physics tick required')
            if not isinstance(actuator_h,dict): raise ValueError('Actual committed actuator h required')
            retained_gripper_command({'actuator_state':actuator_h})
            transition=snapshot['last_physical_tick'];issued=snapshot['issued_reference']
            if initial:
                if tick != 0 or transition is not None or issued is not None:
                    raise ValueError('Initial label must precede the first Native tick')
            else:
                old_tick,old_clock=self._last
                if tick != old_tick+1 or clock <= old_clock:
                    raise ValueError('No duplicate, skipped, stale or fabricated TRAIN physics row')
                if (not isinstance(transition,dict) or not isinstance(issued,dict)
                        or transition['tick_after'] != tick or transition['tick_before'] != old_tick
                        or abs(transition['physics_time_before']-snapshot['physics_time_origin']-old_clock)>1e-9
                        or abs(transition['physics_time_after']-snapshot['physics_time_origin']-clock)>1e-9
                        or abs((clock-old_clock)-.05)>2e-5):
                    raise ValueError('TRAIN motion time must come from measured before/after Native clocks')
                if (actuator_h.get('reference_valid') is not True
                        or actuator_h.get('reference_clock') is None
                        or abs(actuator_h['reference_clock']-clock)>1e-9
                        or issued['physics_tick_before'] != old_tick
                        or abs(issued['actual_clock_before']-old_clock)>1e-9):
                    raise ValueError('Record only after original executor commits the same physical tick')
                for field,ref in (('reference_positions','q_ref'),('reference_velocities','dq_ref'),
                                  ('reference_accelerations','ddq_ref')):
                    if not np.array_equal(np.asarray(actuator_h[field]),np.asarray(issued[ref])):
                        raise ValueError('Committed actuator reference differs from the actual issued reference')
                if actuator_h['gripper_target'] != issued['gripper_target']:
                    raise ValueError('Actual issued gripper target differs from committed h')
            row=_plain(dict(schema='TRAIN_private_true_clock_robot_multibody_label_v1',
                dataset_use='train',policy_use_allowed=False,label_only=True,
                **snapshot,actuator_h=deepcopy(actuator_h),initial_row=initial,
                observed_dt=None if initial else clock-self._last[1],
                time_basis='Native_simGetSimulationTime_and_task_t_readbacks_not_tick_times_DT',
                physical_timestamp_observed=True,contact_force_observed=False,
                Native_grasped_list_is_not_contact_force=True,CF_zero_clock_is_motion_data=False))
            self.sink(deepcopy(row));self._last=(tick,clock);self.rows+=1
            return row
        except BaseException as error:
            self._fault=type(error).__name__+': '+str(error)
            raise


class TRAINRecordingExecutor(ShortHorizonExecutor):
    """Original execution math; only observe issue and post-commit boundaries."""
    def __init__(self, *args, recorder, **kwargs):
        if type(recorder) is not TRAINLabelRecorder:
            raise TypeError('Exact optional TRAIN recorder required')
        self._train_recorder=recorder
        super().__init__(*args, **kwargs)

    def _next_reference(self):
        q,v,a=super()._next_reference()
        self._train_recorder.backend._bind_issued_training_reference(dict(
            q_ref=q,dq_ref=v,ddq_ref=a,gripper_target=self.gripper_target,
            command_id=self._command_id,command_goal=self._goal,
            actuator_h_before=self.state()))
        return q,v,a

    def _state(self):
        result=super()._state()
        recorder=self._train_recorder
        if (recorder._last is not None and recorder.backend._ticks > recorder._last[0]
                and self._reference_clock is not None
                and abs(self._reference_clock-self._clock())<=1e-9):
            recorder.record_after_commit(self.state())
        return result


def run_hold_teacher(backend, executor, *, log_private):
    """Keep initial measured q/command; no NN reference, body pose or GT input."""
    if getattr(backend,'dataset_use',None) != 'train':
        raise ValueError('HoldTeacher is explicitly TRAIN-only')
    q0=vector(backend.get_joint_positions(),7,'actual initial hold q')
    opening=retained_gripper_command({'actuator_state':executor.state()})
    cycles=0
    while not backend.terminal():
        receipt=executor.execute_joint_target(q0,opening=opening,phase='TRAIN_initial_actual_q_hold_teacher')
        log_private(receipt.to_dict());cycles+=1
    return dict(cycles=cycles,initial_actual_q=q0.tolist(),gripper_target=opening,
        model_forward_calls=0,teacher_uses_task_expert=False,Native_labels_are_policy_inputs=False)
