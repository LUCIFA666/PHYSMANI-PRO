"""Metric-only collision witnesses around the unchanged v5 policy and map."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import itertools
import numpy as np

from .evaluation import file_sha
from .native_actuator_diagnostic_v5 import NativeActuatorDiagnosticBackend, diagnostic_actor_model_preflight
from .protocol import MultitaskProtocolError

SCHEMA = 'r10_all_phase_public_collision_metric_only_diagnostic_v17'
ROOT = Path(__file__).resolve().parents[2]


def binding(root, native_repo):
    root, native_repo = Path(root).resolve(), Path(native_repo).resolve()
    api = native_repo / 'third_party/PyRep/pyrep/robots/arms/arm.py'
    return dict(schema=SCHEMA, source=dict(path=str(Path(__file__).resolve().relative_to(root)), sha256=file_sha(__file__)),
        arm_API=dict(path=str(api), sha256=file_sha(api)), policy_commands_modified=False,
        diagnostic_values_supplied_to_policy=False, physics_or_camera_calls_added=False,
        diagnostic_window='first 512 collision rejections across every manipulation phase', maximum_rejection_records=512)


def preflight(root, prepared):
    result = diagnostic_actor_model_preflight(root, prepared)
    expected = binding(root, prepared['config']['baseline_repo'])
    if prepared['config'].get('public_collision_query_diagnostics') != expected:
        raise MultitaskProtocolError('public collision diagnostic binding changed')
    from scripts.evaluate_multitask_case_r10 import locked_digest
    for pointer in (expected['source'], expected['arm_API']):
        if locked_digest(Path(root) / pointer['path'], prepared['files']) != pointer['sha256']:
            raise MultitaskProtocolError('actual collision diagnostic source is not frozen')
    return dict(result, public_collision_query_diagnostics=expected)


class CollisionObserver:
    def __init__(self, actor, env):
        self.actor, self.env = actor, env
        self.service = env.map_service
        self.events = []
        self.current = None
        self.sample = None
        self.extra_pair_queries = 0
        self._instrument()

    def label(self, ghost):
        for index, value in enumerate(self.service.static_ghosts):
            if value is ghost:
                return 'static:' + self.service.map[index]['name']
        for identity, value in self.service.estimated_bodies.items():
            if value.ghost is ghost:
                return 'estimated:' + identity + ':primary'
            if value.fallback_ghost is ghost:
                return 'estimated:' + identity + ':rotation_uncertainty'
        return 'unidentified_ghost'

    def geometry(self, ghost):
        try:
            bounds = np.asarray(ghost.get_bounding_box(), float).reshape(3, 2)
            local = np.asarray(list(itertools.product(*bounds)))
            matrix = np.asarray(ghost.get_matrix(), float)
            world = local @ matrix[:3, :3].T + matrix[:3, 3]
            return dict(label=self.label(ghost), world_AABB_min=world.min(axis=0).tolist(),
                world_AABB_max=world.max(axis=0).tolist(), estimated_or_static_ghost_only=True)
        except Exception as error:
            return dict(label=self.label(ghost), geometry_unavailable=type(error).__name__)

    def _instrument(self):
        service = self.service
        original_check = service.check
        original_collision = self.actor.robot_model.collision
        original_arm = self.env.arm.check_arm_collision
        ghosts = list(service.static_ghosts)
        for body in service.estimated_bodies.values():
            ghosts.extend(body.ghosts())

        def sample(instance, q, offset):
            previous = self.sample
            self.sample = dict(future_offset=float(offset), sampled_joints=np.asarray(q).tolist())
            try:
                return original_collision(q, offset)
            finally:
                self.sample = previous
        self.actor.robot_model.collision = MethodType(sample, self.actor.robot_model)

        def check(instance, *args, **kwargs):
            port = self.actor.collision_port
            record = None
            if (port.bound is not None and len(self.events) < 512):
                record = dict(time=float(port.frame[0].robot.timestamp), sample=deepcopy(self.sample),
                    binding=deepcopy(port.bound), attachments=sorted(service.estimated_attachments),
                    geometry_modes=deepcopy(kwargs.get('geometry_modes')), arm=None, held_world=[])
            previous = self.current
            self.current = record
            try:
                result = original_check(*args, **kwargs)
                if record is not None and bool(result):
                    record['collision'] = True
                    self.events.append(record)
                return result
            finally:
                self.current = previous
        service.check = MethodType(check, service)

        def arm(*args, **kwargs):
            result = original_arm(*args, **kwargs)
            record = self.current
            if record is not None:
                record['arm'] = dict(collision=bool(result), colliding_ghosts=[])
                if result:
                    clock = service._clock()
                    for ghost in ghosts:
                        if ghost.is_collidable():
                            self.extra_pair_queries += 1
                            try:
                                if original_arm(ghost):
                                    record['arm']['colliding_ghosts'].append(self.geometry(ghost))
                            except Exception as error:
                                record['arm'].setdefault('unavailable_pair_queries', []).append(dict(
                                    label=self.label(ghost), error_type=type(error).__name__))
                    service._clock_check(clock, 'read-only metric collision partner diagnostic')
            return result
        self.env.arm.check_arm_collision = arm

        for ghost in ghosts:
            original = ghost.check_collision
            def pair(other=None, original=original, ghost=ghost):
                result = original(other)
                if self.current is not None and bool(result):
                    self.current['held_world'].append(dict(held=self.geometry(ghost), obstacle=self.geometry(other)))
                return result
            ghost.check_collision = pair

    def receipt(self):
        return dict(schema=SCHEMA, events=deepcopy(self.events),
            extra_read_only_arm_ghost_queries=self.extra_pair_queries,
            policy_commands_modified=False, diagnostic_values_supplied_to_policy=False,
            added_physics_ticks=0, added_camera_queries=0,
            collision_parameters_or_permissions_modified=False,
            dynamic_native_truth_read=False, formal_metric=False)


class NativePublicCollisionDiagnosticBackend(NativeActuatorDiagnosticBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        observer = CollisionObserver(actor, env)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), public_collision_query_diagnostics_v6=observer.receipt())
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
