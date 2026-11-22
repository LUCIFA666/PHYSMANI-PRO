"""Public-observation bridge for equally warmed CV and learned SE3 forecasts.

No environment, native task, target schedule, teacher or metric receipt enters
this module. RobotState is measured proprioception. Attached geometry remains
in the controller/map attachment path rather than an external-body forecast.
"""
from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Forecast, ObjectState, Pose, RobotState
from .perception import RigidChildConstraint


class ForecastingError(ValueError):
    pass


@dataclass(frozen=True)
class ForecastConfig:
    dt: float = .05
    minimum_history_frames: int = 5
    maximum_history_frames: int = 20
    maximum_forecast_seconds: float = 1.1
    observation_time_tolerance: float = 1e-7

    def __post_init__(self):
        if (not np.isclose(self.dt, .05) or self.minimum_history_frames < 5
                or self.maximum_history_frames < self.minimum_history_frames
                or not np.isfinite([self.maximum_forecast_seconds, self.observation_time_tolerance]).all()
                or self.maximum_forecast_seconds <= 0 or self.observation_time_tolerance <= 0):
            raise ForecastingError('declare common 50ms clocks and at least five causal warmup frames')


@dataclass(frozen=True)
class ObservedAttachment:
    object_id: str
    tool_to_body: Pose
    observed_at: float
    provenance: str = 'current_rgbd_robot_proprioception'

    def __post_init__(self):
        if (not self.object_id or not isinstance(self.tool_to_body, Pose)
                or not np.isfinite(self.observed_at) or self.observed_at < 0
                or self.provenance != 'current_rgbd_robot_proprioception'):
            raise ForecastingError('attachments require a causal RGB-D/proprioceptive tool-to-body estimate')


def _compose(a, b):
    return Pose(a.apply(b.position), a.rotation @ b.rotation)


def _cv(objects, times):
    count = len(times)
    positions = np.zeros((count, len(objects), 3))
    rotations = np.zeros((count, len(objects), 3, 3))
    velocity = np.zeros_like(positions)
    angular = np.zeros_like(positions)
    for i, obj in enumerate(objects):
        elapsed = times-times[0]
        positions[:, i] = obj.pose.position+elapsed[:, None]*obj.linear_velocity
        rotations[:, i] = Rotation.from_rotvec(elapsed[:, None]*obj.angular_velocity).as_matrix() @ obj.pose.rotation
        velocity[:, i], angular[:, i] = obj.linear_velocity, obj.angular_velocity
    return positions, rotations, velocity, angular


class MultiBodyForecastBridge:
    """Maintain per-root causal readiness and preserve verified rigid layouts.

    A context engine is an already validated MultiBodyContextDynamics object;
    the CV branch never constructs a neural network or imports torch. Context
    histories are cleared per body under that engine's public history lock.
    Rigid children inherit their root prediction/context, not an independent
    GRU rollout. The observed ObjectState instances are never modified.
    """
    def __init__(self, body_ids, *, method='cv', context=None, rigid_children=(),
                 task_spec=None, config=ForecastConfig()):
        self.body_ids = tuple(body_ids)
        if (not self.body_ids or len(set(self.body_ids)) != len(self.body_ids)
                or any(not isinstance(body, str) or not body for body in self.body_ids)
                or method not in ('cv', 'context')):
            raise ForecastingError('unique declared physical body IDs and cv/context method required')
        if (method == 'cv' and context is not None) or (method == 'context' and context is None):
            raise ForecastingError('only the context branch accepts its completed fitted engine')
        if context is not None:
            metadata = getattr(context, 'metadata', {})
            if (metadata.get('training_complete') is not True
                    or metadata.get('TEST_used_for_fitting') is not False
                    or not isinstance(getattr(context, 'checkpoint_sha256', None), str)
                    or not hasattr(context, 'history') or not hasattr(context, 'lock')):
                raise ForecastingError('pass a completed provenance-validated TRAIN context engine')
        if task_spec is not None:
            permitted = {role.name for role in task_spec.roles if role.kind == 'body'}
            if set(self.body_ids)-permitted or set(self.body_ids)&set(task_spec.audit_only_goal_roles):
                raise ForecastingError('forecast only declared physical TaskSpec bodies')
        self.method, self.context, self.config = method, context, config
        self.constraints, self.parent = tuple(rigid_children), {}
        for edge in self.constraints:
            if not isinstance(edge, RigidChildConstraint) or {edge.parent_role, edge.child_role}-set(self.body_ids):
                raise ForecastingError('rigid forecasts need explicit TRAIN RigidChildConstraint objects')
            safe = (edge.parent_role == 'frame' and edge.child_role in
                    ('peg0','peg1','peg2','spoke0','spoke1','spoke2')
                    or edge.parent_role == 'button' and edge.child_role == 'wrap')
            if not safe or edge.child_role in self.parent:
                raise ForecastingError('attachable cups/articulated plate cannot be permanently bound to a frame')
            self.parent[edge.child_role] = edge
        self.root, self.local = {}, {}
        for body in self.body_ids:
            current, transform, seen = body, Pose(np.zeros(3)), set()
            while current in self.parent:
                if current in seen:
                    raise ForecastingError('cyclic rigid layout')
                seen.add(current)
                edge = self.parent[current]
                transform = _compose(edge.child_to_parent, transform)
                current = edge.parent_role
            self.root[body], self.local[body] = current, transform
        self.reset()

    def reset(self):
        self.history = {body:deque(maxlen=self.config.maximum_history_frames) for body in self.body_ids}
        self.current, self.modes, self.orientation = {}, {}, {}
        self.attachments, self.missing, self.groups = {}, {}, {}
        self.last_time, self.episode_start = None, None
        self.last_forecast = None
        if self.context is not None:
            self.context.reset()

    def _forget(self, body):
        self.history[body].clear()
        self.current.pop(body, None)
        if self.context is not None:
            with self.context.lock:
                self.context.history.pop(body, None)

    def _observed_state(self, obj, diagnostic, metadata):
        if not isinstance(diagnostic.get('orientation_observable'), (bool, np.bool_)):
            raise ForecastingError('tracker must explicitly report every body orientation observability')
        full = bool(diagnostic['orientation_observable'])
        declared = diagnostic.get('orientation_observability')
        if declared not in ('full', 'axis', 'none'):
            raise ForecastingError('tracker must declare full/axis/none orientation interpretation')
        if full and declared != 'full':
            raise ForecastingError('symmetric/gauge orientation cannot be declared fully observable')
        angular = obj.angular_velocity.copy()
        if not full:
            if declared == 'axis':
                axis = np.asarray(metadata.get('template_priors', {}).get(obj.object_id, {}).get('symmetry_axis', (0.,0.,1.)),float)
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
                    raise ForecastingError('invalid declared symmetry axis')
                axis = obj.pose.rotation @ (axis/np.linalg.norm(axis))
                angular -= axis*np.dot(angular,axis)
            else:
                angular[:] = 0.
        measured = ObjectState(obj.object_id,obj.timestamp,obj.pose,obj.points_world,
            obj.linear_velocity,angular,obj.confidence,obj.visible)
        return measured, full, declared

    def observe(self, states, tracker_metadata, robot, *, modes, invalid_mode_ids=(),
                non_exogenous_ids=(), attachments=None):
        """Accept one current sensor frame; invalidity clears both methods equally.

        Existing measured attachments can survive body occlusion while current
        robot contact and attached mode remain known. This is controller FK
        evidence, never a newly visible body measurement or an external root.
        """
        if not isinstance(robot, RobotState) or not isinstance(tracker_metadata, Mapping):
            raise ForecastingError('data-only RobotState and tracker metadata required')
        if tracker_metadata.get('metric_only') is True or any(key in tracker_metadata for key in
                ('native_result','active_motion_schedule','task_internal_fields','reset_receipt')):
            raise ForecastingError('privileged state is not a forecasting observation')
        now = float(robot.timestamp)
        if self.last_time is not None and now < self.last_time-self.config.observation_time_tolerance:
            raise ForecastingError('reset bridge at episode boundaries')
        if self.episode_start is None:
            self.episode_start = now
        objects = {state.object_id:state for state in states}
        if len(objects) != len(states) or set(objects)-set(self.body_ids):
            raise ForecastingError('unknown/duplicate observed body ID')
        if not isinstance(modes, Mapping) or set(modes)-set(self.body_ids):
            raise ForecastingError('controller modes must name declared body IDs')
        invalid, controlled = set(invalid_mode_ids), set(non_exogenous_ids)
        if (invalid|controlled)-set(self.body_ids):
            raise ForecastingError('unknown mode/command body identity')
        attachments = {} if attachments is None else dict(attachments)
        if set(attachments)-set(self.body_ids):
            raise ForecastingError('unknown attachment body identity')
        self.groups = {group['root']:group for group in tracker_metadata.get('rigid_groups', ())}
        self.missing, self.orientation = {}, {}
        direct = {body for body,obj in objects.items() if obj.visible
            and tracker_metadata.get('bodies',{}).get(body,{}).get('visible') is True
            and tracker_metadata.get('bodies',{}).get(body,{}).get('registration_valid') is True
            and np.isclose(obj.timestamp,now,atol=self.config.observation_time_tolerance,rtol=0)}
        context_inputs = []
        for body in self.body_ids:
            mode = modes.get(body, 'unknown')
            if mode not in ('external','attached','released') or body in invalid:
                self._forget(body); self.attachments.pop(body,None)
                self.missing[body] = 'unknown_controller_motion_mode'
                continue
            if self.modes.get(body) != mode:
                self._forget(body)
            self.modes[body] = mode
            if body in controlled:
                self._forget(body); self.attachments.pop(body,None)
                self.missing[body] = 'non_exogenous_public_command_segment'
                continue
            if mode == 'attached':
                self._forget(body)
                candidate = attachments.get(body,self.attachments.get(body))
                if not robot.contact or not isinstance(candidate,ObservedAttachment) or candidate.object_id != body:
                    self.attachments.pop(body,None)
                    self.missing[body] = 'attached_without_current_contact_and_measured_transform'
                    continue
                if not self.episode_start-self.config.observation_time_tolerance <= candidate.observed_at <= now+self.config.observation_time_tolerance:
                    raise ForecastingError('attachment estimate is future or from a previous episode')
                self.attachments[body] = candidate
                # Absence/occlusion is still recorded below; it never becomes
                # a fresh ObjectState observation for either predictor.
                continue
            self.attachments.pop(body,None)
            obj = objects.get(body)
            diagnostic = tracker_metadata.get('bodies',{}).get(body,{})
            if (obj is None or not obj.visible or diagnostic.get('visible') is not True
                    or diagnostic.get('registration_valid') is not True
                    or not np.isclose(obj.timestamp,now,atol=self.config.observation_time_tolerance,rtol=0)):
                self._forget(body)
                self.missing[body] = 'missing_occluded_or_stale_current_RGBD_observation'
                continue
            measured, full, declared = self._observed_state(obj,diagnostic,tracker_metadata)
            self.orientation[body] = dict(full=full,declared=declared)
            direct.add(body)
            history = self.history[body]
            if history and now-history[-1][0] > self.config.observation_time_tolerance and not np.isclose(
                    now-history[-1][0],self.config.dt,atol=self.config.observation_time_tolerance,rtol=0):
                self._forget(body); history = self.history[body]
            if history and abs(now-history[-1][0]) <= self.config.observation_time_tolerance:
                previous = history[-1][1]
                if (not np.allclose(previous.pose.xyzw(),measured.pose.xyzw(),atol=1e-8)
                        or not np.allclose(previous.linear_velocity,measured.linear_velocity,atol=1e-8)):
                    raise ForecastingError('one timestamp cannot provide different body measurements')
            else:
                history.append((now,measured))
            self.current[body] = measured
            # Declared children are composed from their verified root only.
            if self.root[body] == body and self.context is not None:
                context_inputs.append(measured)
        if self.context is not None and context_inputs:
            self.context.observe(context_inputs,
                modes={state.object_id:self.modes[state.object_id] for state in context_inputs},
                orientation_observability={state.object_id:self.orientation[state.object_id]['full'] for state in context_inputs})
        self.last_time = now
        self.direct_body_ids = direct
        self.last_forecast = None
        return dict(observed_until=now,history_frames={body:len(rows) for body,rows in self.history.items()},
            attached_body_ids=sorted(self.attachments),directly_visible_body_ids=sorted(direct),
            unavailable_body_reasons=dict(self.missing),common_warmup_frames=self.config.minimum_history_frames)

    def attachment_updates(self):
        """Actor synchronizes this complete map and clears no-longer-held IDs."""
        return {body:value.tool_to_body for body,value in self.attachments.items()}

    def _rigid_evidence(self, body):
        root = self.root[body]
        group = self.groups.get(root,{})
        if group.get('registration_valid') is not True or body not in group.get('members', ()):
            return False
        edge = self.parent[body]
        return any(row.get('parent') == edge.parent_role and row.get('child') == edge.child_role
                   and row.get('evidence_sha256') == edge.provenance.evidence_sha256
                   for row in group.get('TRAIN_constraints', ()))

    def predict(self, robot, horizon=1.1, dt=.05):
        if (not isinstance(robot,RobotState) or self.last_time is None
                or not np.isclose(robot.timestamp,self.last_time,atol=self.config.observation_time_tolerance,rtol=0)
                or not np.isclose(dt,self.config.dt) or not np.isfinite(horizon) or horizon <= 0
                or horizon > self.config.maximum_forecast_seconds+1e-8
                or not np.isclose(horizon/dt,round(horizon/dt))):
            raise ForecastingError('predict only from the current robot clock at a declared 50ms horizon')
        if self.context is not None and horizon > self.context.maximum_validated_horizon+1e-8:
            raise ForecastingError('context horizon exceeds completed TRAIN validation')
        times = self.last_time+np.arange(int(round(horizon/dt))+1)*dt
        missing = dict(self.missing)
        roots = []
        for body in self.body_ids:
            if self.root[body] != body or body in self.attachments or body in missing:
                continue
            if len(self.history[body]) < self.config.minimum_history_frames:
                missing[body] = 'common_five_frame_warmup'
            else:
                roots.append(self.current[body])
        contexts = {}
        source_metadata = dict(checkpoint_sha256=self.context.checkpoint_sha256,
            context_encoder_sha256=getattr(self.context,'encoder_sha256',None),
            state_schema=self.context.metadata.get('state_schema')) if self.context is not None else {}
        if self.context is not None and roots:
            prediction = self.context.predict(roots,horizon,dt)
            if (set(prediction.object_ids) != {state.object_id for state in roots}
                    or not np.allclose(prediction.times,times,atol=1e-7)):
                raise ForecastingError('context engine returned a different body set or clock')
            columns = {body:(prediction.positions[:,i],prediction.rotations[:,i],prediction.linear_velocities[:,i],
                             prediction.angular_velocities[:,i]) for i,body in enumerate(prediction.object_ids)}
            contexts = dict(prediction.metadata.get('contexts',{}))
        else:
            p,r,v,w = _cv(roots,times)
            columns = {state.object_id:(p[:,i],r[:,i],v[:,i],w[:,i]) for i,state in enumerate(roots)}
        provenance, visible, full, history_frames = {}, {}, {}, {}
        for state in roots:
            body = state.object_id
            provenance[body] = dict(source='current_visible_causal_root_'+self.method,history_owner=body,
                declared_rotation=self.orientation[body]['declared'],angular_velocity_is_true_spin=self.orientation[body]['full'])
            visible[body], full[body], history_frames[body] = True,self.orientation[body]['full'],len(self.history[body])
        for body in self.body_ids:
            root = self.root[body]
            if root == body or body in self.attachments:
                continue
            if (self.modes.get(body) not in ('external','released') or body in self.missing and
                    self.missing[body] in ('unknown_controller_motion_mode','non_exogenous_public_command_segment')):
                continue
            if root not in columns:
                missing[body] = 'rigid_root_missing_occluded_or_warming_up'
                continue
            if not self.orientation[root]['full']:
                missing[body] = 'rigid_root_full_orientation_unobservable'
                continue
            if not self._rigid_evidence(body):
                missing[body] = 'rigid_TRAIN_constraint_or_current_group_evidence_invalid'
                continue
            p,r,v,w = columns[root]
            transform = self.local[body]
            offset = np.einsum('tij,j->ti',r,transform.position)
            columns[body] = (p+offset,r@transform.rotation,v+np.cross(w,offset),w.copy())
            edge = self.parent[body]
            provenance[body] = dict(source='TRAIN_verified_rigid_parent_forecast',history_owner=root,
                parent_role=edge.parent_role,TRAIN_evidence_sha256=edge.provenance.evidence_sha256,
                constraint_source=edge.provenance.source_reference,
                context_copied_from=root if root in contexts else None,
                current_rigidity_consistency_tested=self.groups[root].get('rigidity_consistency_tested') is True)
            if root in contexts:
                contexts[body] = list(contexts[root])
            visible[body] = body in self.direct_body_ids
            full[body],history_frames[body] = True,len(self.history[root])
            missing.pop(body,None)
        selected = tuple(body for body in self.body_ids if body in columns)
        count = len(times)
        stack = lambda slot,shape: np.stack([columns[body][slot] for body in selected],axis=1) if selected else np.zeros((count,0,*shape))
        result = Forecast(selected,times,stack(0,(3,)),stack(1,(3,3)),stack(2,(3,)),stack(3,(3,)),
            metadata=dict(method=self.method,**source_metadata,contexts=contexts,
                required_body_ids=list(self.body_ids),missing_body_ids=sorted(missing),missing_body_reasons=missing,
                attached_body_ids=sorted(self.attachments),
                attachment_forecast_source='caller measured tool-to-body transform and candidate robot FK; not external rollout',
                directly_visible=visible,not_directly_observed_body_ids=sorted(set(self.body_ids)-self.direct_body_ids),
                full_orientation_observable=full,history_frames=history_frames,body_provenance=provenance,
                common_warmup_frames=self.config.minimum_history_frames,observed_until=self.last_time,
                unknown_rotation_is_gauge=True,learned_true_spin_for_unobservable_bodies=False,
                missing_geometry_requires_collision_fail_closed=True,root_occlusion_can_be_virtual_measurement=False))
        self.last_forecast = result
        return result


ForecastBridge = MultiBodyForecastBridge
