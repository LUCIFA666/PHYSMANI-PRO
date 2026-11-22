"""Read-only native actuator/contact instrumentation around the frozen v5 actor.

Diagnostic values are never supplied to the actor, forecast, gates or commands.
Queries do not advance physics. Force/contact observations are engine-dependent;
last-step contact absence and force magnitude alone establish no causal claim.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np

from .evaluation import file_sha
from .capture_stop_actor_backend_v11 import (
    ContinuousReferenceNativeBackend, persistent_reference_actor_model_preflight,
)
from .protocol import MultitaskProtocolError

SCHEMA = 'r10_native_actuator_readonly_diagnostics_v11'
NATIVE_RELATIVE = (
    'third_party/PyRep/pyrep/robots/robot_component.py',
    'third_party/PyRep/pyrep/objects/joint.py',
    'third_party/PyRep/pyrep/objects/object.py',
    'third_party/PyRep/pyrep/backend/sim.py',
)


def diagnostic_binding(root, native_repo):
    root, repo = Path(root).resolve(), Path(native_repo).resolve()
    return dict(schema=SCHEMA, source=dict(path=str(Path(__file__).resolve().relative_to(root)),
        sha256=file_sha(__file__)), native_API_sources_sha256={
            str((repo / name).resolve()): file_sha(repo / name) for name in NATIVE_RELATIVE},
        policy_inputs_augmented=False, policy_commands_modified=False,
        additional_physics_or_camera_calls=False,
        query_stages=['before_policy', 'after_policy', 'before_dispatch', 'after_dispatch'],
        source_kind='native_metric_diagnostics_only',
        retained_policy='v9 same-frame public RGBD support-constrained sphere and attachment updates; unchanged v5 continuous motion gates', formal_metric=False)


def diagnostic_actor_model_preflight(root, prepared):
    result = persistent_reference_actor_model_preflight(root, prepared)
    expected = diagnostic_binding(root, prepared['config']['baseline_repo'])
    if prepared['config'].get('native_metric_diagnostics') != expected:
        raise MultitaskProtocolError('native read-only diagnostic binding changed')
    from scripts.evaluate_multitask_case_r10 import locked_digest
    sources = {str(Path(root) / expected['source']['path']): expected['source']['sha256'],
        **expected['native_API_sources_sha256']}
    for name, checksum in sources.items():
        if locked_digest(Path(name), prepared['files']) != checksum:
            raise MultitaskProtocolError('native diagnostic API/source omitted from frozen lock')
    return dict(result, native_metric_diagnostics=expected)


def _clock(env):
    return dict(ticks=int(env.ledger.ticks), task_time=float(env.task.t),
        physics_time=float(env.physics_time()), policy_time=float(env.time))


def _vector(value, count):
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError('nonfinite or wrong-shaped native diagnostic joint vector')
    return result.tolist()


def snapshot(env, stage):
    """Read native diagnostics, annotate unsupported API, assert no clock step."""
    before = _clock(env)
    count = len(env.arm.joints)
    row = dict(stage=stage, clock_before=before, diagnostic_only=True, queries={})
    queries = row['queries']
    def capture(key, callback):
        try:
            queries[key] = dict(available=True, value=callback())
        except Exception as error:
            queries[key] = dict(available=False, error_type=type(error).__name__, message=str(error))
    for key, method in (
        ('joint_positions', 'get_joint_positions'),
        ('native_reported_joint_velocities', 'get_joint_velocities'),
        ('joint_target_positions', 'get_joint_target_positions'),
        ('joint_target_velocities', 'get_joint_target_velocities'),
        ('joint_forces_or_torques', 'get_joint_forces'),
    ):
        capture(key, lambda method=method: _vector(getattr(env.arm, method)(), count))
    capture('joint_modes', lambda: [dict(name=getattr(mode, 'name', str(mode)),
        value=int(getattr(mode, 'value', mode))) for mode in env.arm.get_joint_modes()])
    capture('joint_control_loop_enabled', lambda: [int(j.is_control_loop_enabled()) for j in env.arm.joints])
    capture('joint_motor_enabled', lambda: [int(j.is_motor_enabled()) for j in env.arm.joints])
    def contacts():
        rows=[]
        for obj in env.task.get_graspable_objects():
            actual=[]
            for contact in obj.get_contact(get_contact_normal=True):
                handles=contact['contact_handles']
                if len(handles)!=2: raise ValueError('unexpected native contact handles')
                data=np.asarray(contact['contact'],float)
                if data.ndim!=1 or not np.isfinite(data).all():
                    raise ValueError('unexpected native contact payload')
                actual.append(dict(contact_handles=[int(h) for h in handles],
                    native_contact_payload=data.tolist()))
            rows.append(dict(object_name=obj.get_name(), object_handle=int(obj.get_handle()),
                contacts=actual))
        return rows
    capture('graspable_object_last_physics_step_contacts', contacts)
    row['clock_after'] = _clock(env)
    row['clock_unchanged'] = before == row['clock_after']
    if not row['clock_unchanged']:
        raise MultitaskProtocolError('read-only native diagnostic query changed the simulation clock')
    return row


class NativeActuatorDiagnosticBackend(ContinuousReferenceNativeBackend):
    def __init__(self):
        super().__init__()
        self.diagnostic_events=[]

    def _capture(self, env, stage):
        row=snapshot(env, stage)
        self.diagnostic_events.append(row)
        return row

    def instrument_actor(self, actor, env):
        previous_act=actor.act
        def observed_act(instance, *args, **kwargs):
            self._capture(env, 'before_policy')
            try:
                return previous_act(*args, **kwargs)
            finally:
                self._capture(env, 'after_policy')
        actor.act=MethodType(observed_act, actor)
        previous_metrics=actor.metrics_report
        def observed_metrics(instance):
            return dict(previous_metrics(), native_metric_diagnostics=self.receipt())
        actor.metrics_report=MethodType(observed_metrics, actor)
        return actor

    def actor(self, prepared, env, descriptions):
        actor, identity=super().actor(prepared, env, descriptions)
        from .public_collision_diagnostic_v6 import CollisionObserver
        observer = CollisionObserver(actor, env)
        before_metrics = actor.metrics_report
        def collision_metrics(instance):
            return dict(before_metrics(), public_collision_query_diagnostics_v6=observer.receipt())
        actor.metrics_report = MethodType(collision_metrics, actor)
        self.instrument_actor(actor, env)
        return actor, identity

    def dispatch(self, env, command):
        self._capture(env, 'before_dispatch')
        try:
            return super().dispatch(env, command)
        finally:
            self._capture(env, 'after_dispatch')

    def receipt(self):
        return dict(schema=SCHEMA, events=deepcopy(self.diagnostic_events),
            diagnostic_values_supplied_to_policy=False, commands_modified=False,
            no_additional_physics_or_camera_calls=True,
            force_semantics='native engine-dependent joint-axis force/torque, not configured force capacity or saturation proof',
            contact_semantics='native last-physics-step graspable-object contact pairs; no continuous contact or causal proof',
            velocity_semantics='native reported proxy separately recorded; no instantaneous derivative claim',
            motor_or_dynamics_setters_called_by_observer=False, formal_metric=False,
            all_read_queries_clock_unchanged=all(r['clock_unchanged'] for r in self.diagnostic_events))
