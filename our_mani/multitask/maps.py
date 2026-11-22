"""Immutable static map plus caller-estimated dynamic collision ghosts.

Native role bindings partition the scene only. Dynamic meshes and poses used
in a policy query must come from current RGB-D or TRAIN geometry; no live
native dynamic pose is read or temporarily teleported during planning.
"""
from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Pose
from ..reachability import PyRepRobotModel
from .native_tasks import resolve_native_roles
from .protocol import MultitaskProtocolError, MultitaskInfrastructureError
from .safe_native_mesh import read_native_mesh


def _matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = pose.rotation, pose.position
    return matrix


def _shape_tree(obj, shape_type):
    values = list(obj.get_objects_in_tree(object_type=shape_type))
    if hasattr(obj, 'get_mesh_data'):
        values.append(obj)
    return {int(x.get_handle()): x for x in values}


@dataclass
class EstimatedBody:
    ghost: object
    creation_matrix: np.ndarray
    geometry_sha256: str
    provenance: str
    source_sha256: str | None = None
    collision_geometry_receipt: dict | None = None
    fallback_ghost: object | None = None
    fallback_creation_matrix: np.ndarray | None = None
    fallback_geometry_sha256: str | None = None
    fallback_geometry_receipt: dict | None = None

    def ghosts(self):
        return [self.ghost] + ([] if self.fallback_ghost is None else [self.fallback_ghost])


def _geometry_receipt(value, source_sha256, geometry_sha256=None):
    if value is None:
        return None
    try:
        copied = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (ValueError, TypeError) as error:
        raise MultitaskProtocolError('collision geometry receipt must be finite JSON data') from error
    if (not isinstance(copied, dict) or copied.get('TRAIN_asset_sha256') != source_sha256
            or geometry_sha256 is not None and copied.get('registered_mesh_sha256') != geometry_sha256):
        raise MultitaskProtocolError('collision geometry receipt differs from its TRAIN source or registered mesh')
    return copied


class SharedMultiObjectMap:
    def __init__(self, env, *, shape_type=None, ghost_factory=None):
        if shape_type is None or ghost_factory is None:
            from pyrep.const import ObjectType
            from pyrep.objects.shape import Shape
            shape_type = ObjectType.SHAPE if shape_type is None else shape_type
            ghost_factory = Shape.create_mesh if ghost_factory is None else ghost_factory
        self.env, self._ghost_factory = env, ghost_factory
        self.queries = self.path_queries = 0
        self.robot_contact_queries, self.allowed_robot_contact_ids_seen = 0, set()
        self.geometry_mode_queries = Counter()
        self.static_ghosts, self.estimated_bodies, self.estimated_attachments = [], {}, {}
        self.noncolliding_geometry = {}
        clock = self._clock()
        roles = resolve_native_roles(env.task, env.spec, scene_object_resolver=env.resolve_scene_object)
        robot = _shape_tree(env.arm, shape_type)
        robot.update(_shape_tree(env.gripper, shape_type))
        excluded = dict(robot)
        roots = set(env.spec.dynamic_roots)
        roots.update(role.name for role in env.spec.roles if role.kind in ('body', 'joint'))
        for name in roots:
            excluded.update(_shape_tree(roles[name], shape_type))
        for obj in env.task.get_graspable_objects():
            excluded.update(_shape_tree(obj, shape_type))
        scene = {int(x.get_handle()): x for x in env.pyrep.get_objects_in_tree(object_type=shape_type)}
        self._live_nonrobot = [obj for handle, obj in scene.items() if handle not in robot]
        self._dynamic_excluded = sorted(set(excluded) - set(robot))
        self.required_dynamic_role_ids = tuple(sorted(role.name for role in env.spec.roles
            if role.kind == 'body' and role.name not in env.spec.audit_only_goal_roles))
        static_flags = {handle: dict(collidable=bool(obj.is_collidable()),
                                    respondable=bool(obj.is_respondable()))
                        for handle, obj in scene.items() if handle not in excluded}
        self.static = [obj for handle, obj in scene.items() if handle in static_flags
                       and any(static_flags[handle].values())]
        self.map = []
        try:
            for shape in self.static:
                vertices, triangles, _ = read_native_mesh(shape)
                vertices, triangles = np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)
                digest = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
                self.map.append(dict(name=shape.get_name(), pose=list(shape.get_pose()),
                                     physical_component_flags=static_flags[int(shape.get_handle())],
                                     bounding_box=list(shape.get_bounding_box()), mesh_sha256=digest))
                ghost = self._new_ghost(vertices, triangles)
                self.static_ghosts.append(ghost)
                ghost.set_matrix(shape.get_matrix() @ ghost.get_matrix())
        except Exception:
            self.close()
            raise
        self.sha256 = hashlib.sha256(json.dumps(self.map, sort_keys=True).encode()).hexdigest()
        self._clock_check(clock, 'static map construction')

    def _clock(self):
        return self.env.physics_time(), float(self.env.task.t)

    def _clock_check(self, clock, label):
        if abs(self.env.physics_time() - clock[0]) > 1e-6 or float(self.env.task.t) != clock[1]:
            raise MultitaskProtocolError(label + ' advanced physics/task clock')

    def _new_ghost(self, vertices, triangles):
        ghost = self._ghost_factory(np.asarray(vertices).reshape(-1).tolist(),
                                    np.asarray(triangles).reshape(-1).tolist())
        try:
            ghost.set_dynamic(False)
            ghost.set_respondable(False)
            ghost.set_renderable(False)
            ghost.set_collidable(False)
        except Exception:
            ghost.remove()
            raise
        return ghost

    def register_estimated_geometry(self, object_id, vertices_local, triangles, *, provenance, source_sha256=None,
                                    collision_geometry_receipt=None, fallback_mesh=None):
        if object_id in self.noncolliding_geometry:
            raise MultitaskProtocolError('a frozen noncolliding role cannot acquire a collision mesh')
        if provenance not in ('current_rgbd', 'TRAIN_geometry_model', 'shared_CAD_template'):
            raise MultitaskProtocolError('dynamic collision geometry must be observed or TRAIN-derived')
        if provenance != 'current_rgbd' and (not isinstance(source_sha256, str) or len(source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_sha256)):
            raise MultitaskProtocolError('bind the shared TRAIN/CAD template source SHA256')
        vertices = np.asarray(vertices_local, dtype=np.float32)
        indices = np.asarray(triangles)
        if (not isinstance(object_id, str) or not object_id or vertices.ndim != 2 or vertices.shape[1] != 3
                or len(vertices) < 3 or not np.isfinite(vertices).all() or indices.ndim != 2
                or indices.shape[1] != 3 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= len(vertices)):
            raise MultitaskProtocolError('invalid caller-estimated collision mesh')
        clock = self._clock()
        indices = indices.astype(np.int32)
        digest = hashlib.sha256(vertices.tobytes() + indices.tobytes()).hexdigest()
        receipt = _geometry_receipt(collision_geometry_receipt, source_sha256, digest)
        fallback = None
        if fallback_mesh is not None:
            if (provenance != 'TRAIN_geometry_model' or not isinstance(fallback_mesh, dict)
                    or set(fallback_mesh) != {'vertices', 'triangles', 'collision_geometry_receipt'}):
                raise MultitaskProtocolError('fallback mesh requires an explicit immutable TRAIN rotation bound')
            fv, ft = np.asarray(fallback_mesh['vertices']), np.asarray(fallback_mesh['triangles'])
            if (fv.ndim != 2 or fv.shape[1] != 3 or len(fv) < 3 or not np.isfinite(fv).all()
                    or ft.ndim != 2 or ft.shape[1] != 3 or not len(ft)
                    or not np.issubdtype(ft.dtype, np.integer) or ft.min() < 0 or ft.max() >= len(fv)):
                raise MultitaskProtocolError('invalid TRAIN fallback collision mesh')
            with np.errstate(over='ignore', invalid='ignore'):
                fv, ft = fv.astype(np.float32), ft.astype(np.int32)
            if not np.isfinite(fv).all():
                raise MultitaskProtocolError('TRAIN fallback mesh must remain finite in native float32 storage')
            fdigest = hashlib.sha256(fv.tobytes() + ft.tobytes()).hexdigest()
            freceipt = _geometry_receipt(fallback_mesh['collision_geometry_receipt'], source_sha256, fdigest)
            if (freceipt is None or freceipt.get('collision_geometry_kind') != 'rotation_uncertainty_cube'
                    or freceipt.get('native_pose_or_future_state_used') is not False):
                raise MultitaskProtocolError('fallback must bind the fixed TRAIN all-rotation cube')
            fallback = fv, ft, fdigest, freceipt
        existing = self.estimated_bodies.get(object_id)
        if existing and existing.collision_geometry_receipt != receipt:
            raise MultitaskProtocolError('registered collision geometry uncertainty declaration changed')
        if existing and (existing.fallback_geometry_sha256 != (fallback[2] if fallback else None)
                or existing.fallback_geometry_receipt != (fallback[3] if fallback else None)):
            raise MultitaskProtocolError('registered fallback collision geometry changed')
        if existing and existing.geometry_sha256 == digest and existing.provenance == provenance and existing.source_sha256 == source_sha256:
            return
        ghost = self._new_ghost(vertices, indices)
        fallback_ghost = None
        try:
            creation = ghost.get_matrix().copy()
            if fallback:
                fallback_ghost = self._new_ghost(fallback[0], fallback[1])
                fallback_creation = fallback_ghost.get_matrix().copy()
        except Exception:
            ghost.remove()
            if fallback_ghost is not None:
                fallback_ghost.remove()
            raise
        if existing:
            for old in existing.ghosts():
                old.remove()
        self.estimated_bodies[object_id] = EstimatedBody(ghost, creation, digest, provenance, source_sha256, receipt,
            fallback_ghost, fallback_creation if fallback else None, fallback[2] if fallback else None,
            fallback[3] if fallback else None)
        self._clock_check(clock, 'estimated mesh construction')

    def declare_noncolliding_geometry(self, object_id, *, provenance='TRAIN_geometry_model', source_sha256,
                                     collision_geometry_receipt=None):
        if (not isinstance(object_id, str) or not object_id or provenance != 'TRAIN_geometry_model'
                or not isinstance(source_sha256, str) or len(source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_sha256)
                or object_id in self.estimated_bodies):
            raise MultitaskProtocolError('noncolliding geometry requires an explicit frozen TRAIN asset')
        record = dict(provenance=provenance, source_sha256=source_sha256,
                      geometry_rule='no owned collidable OR respondable component in TRAIN asset')
        receipt = _geometry_receipt(collision_geometry_receipt, source_sha256)
        if receipt is not None:
            record['collision_geometry_receipt'] = receipt
        if object_id in self.noncolliding_geometry and self.noncolliding_geometry[object_id] != record:
            raise MultitaskProtocolError('noncolliding role provenance changed')
        self.noncolliding_geometry[object_id] = record

    def bind_estimated_attachment(self, object_id, tool_to_body, *, provenance):
        """Use only a controller's RGB-D/proprioceptive attachment estimate.

        The transform maps body-local points into the observed tool frame. It
        is never inferred from native grasp parents or live object transforms.
        Passing None clears an estimated attachment after observed release.
        """
        if provenance != 'current_rgbd_robot_proprioception':
            raise MultitaskProtocolError('held-body transforms require current RGB-D/proprioception estimates')
        if tool_to_body is None:
            self.estimated_attachments.pop(object_id, None)
            return
        if object_id not in self.estimated_bodies or not isinstance(tool_to_body, Pose):
            raise MultitaskProtocolError('register held geometry and provide an estimated tool-to-body Pose')
        self.estimated_attachments[object_id] = tool_to_body

    @contextmanager
    def public_scene(self, poses=None, *, ignored_object_ids=(), include_attachments=True, geometry_modes=None):
        poses = {} if poses is None else dict(poses)
        ignored = set(ignored_object_ids)
        if include_attachments and self.estimated_attachments:
            tool = _matrix(Pose.from_xyzw(self.env.arm.get_tip().get_pose()))
            for identity, attachment in self.estimated_attachments.items():
                world = tool @ _matrix(attachment)
                poses[identity] = Pose(world[:3, 3], world[:3, :3])
                # A permitted robot/body grasp contact does not excuse the
                # held body's collision against the rest of the scene.
                ignored.discard(identity)
        if any(not isinstance(pose, Pose) for pose in poses.values()):
            raise MultitaskProtocolError('collision queries require caller Pose estimates')
        missing = set(poses) - set(self.estimated_bodies) - set(self.noncolliding_geometry) - ignored
        if missing:
            raise MultitaskProtocolError('estimated collision geometry missing: ' + str(sorted(missing)))
        modes = {} if geometry_modes is None else dict(geometry_modes)
        if set(modes) - set(self.estimated_bodies) or any(
                mode not in ('primary', 'rotation_uncertainty') for mode in modes.values()):
            raise MultitaskProtocolError('geometry modes must name registered bodies and fixed variants')
        active_ghosts = {}
        clock = self._clock()
        flags = [(shape, shape.is_collidable()) for shape in self._live_nonrobot]
        try:
            for shape, _ in flags:
                shape.set_collidable(False)
            for ghost in self.static_ghosts:
                ghost.set_collidable(True)
            for identity, body in self.estimated_bodies.items():
                active = identity in poses and identity not in ignored
                mode = modes.get(identity, 'rotation_uncertainty' if body.fallback_ghost is not None else 'primary')
                if mode == 'rotation_uncertainty' and body.fallback_ghost is None:
                    raise MultitaskProtocolError('body has no registered rotation uncertainty fallback: ' + identity)
                ghost, creation = (body.ghost, body.creation_matrix) if mode == 'primary' else (
                    body.fallback_ghost, body.fallback_creation_matrix)
                for variant in body.ghosts():
                    variant.set_collidable(False)
                if active:
                    ghost.set_matrix(_matrix(poses[identity]) @ creation)
                    active_ghosts[identity] = ghost
                    self.geometry_mode_queries[mode] += 1
                ghost.set_collidable(active)
            yield active_ghosts
        finally:
            errors = []
            for ghost in self.static_ghosts + [ghost for body in self.estimated_bodies.values() for ghost in body.ghosts()]:
                try:
                    ghost.set_collidable(False)
                except Exception as exc:
                    errors.append(repr(exc))
            for shape, flag in flags:
                try:
                    shape.set_collidable(flag)
                except Exception as exc:
                    errors.append(repr(exc))
            self._clock_check(clock, 'planning/collision query')
            if errors:
                raise MultitaskInfrastructureError('collision flag restore failed: ' + '; '.join(errors))

    def check(self, poses=None, *, ignored_object_ids=(), allowed_robot_contact_ids=(), geometry_modes=None):
        self.queries += 1
        allowed = set(allowed_robot_contact_ids)
        if allowed - set(self.estimated_bodies) - set(self.noncolliding_geometry):
            return True
        if (allowed & set(ignored_object_ids)) - set(self.estimated_attachments):
            raise MultitaskProtocolError('robot contact permission cannot remove that obstacle from the held/world scene')
        if allowed:
            self.robot_contact_queries += 1
            self.allowed_robot_contact_ids_seen.update(allowed)
        try:
            with self.public_scene(poses, ignored_object_ids=ignored_object_ids, geometry_modes=geometry_modes) as active:
                held = [active[key] for key in self.estimated_attachments]
                # Robot contact permission only exempts the arm collection.
                # Restoring the target before held/world checks keeps e.g. a
                # selected peg solid to the held ring throughout insertion.
                # Native PyRep objects define __eq__ and are unhashable.
                exempt = {int(ghost.get_handle()): ghost for ghost in held}
                exempt.update({int(active[key].get_handle()): active[key] for key in allowed if key in active})
                flags = [(ghost, ghost.is_collidable()) for ghost in exempt.values()]
                try:
                    for ghost, _ in flags:
                        ghost.set_collidable(False)
                    arm_collision = bool(self.env.arm.check_arm_collision())
                finally:
                    for ghost, flag in flags:
                        ghost.set_collidable(flag)
                if arm_collision:
                    return True
                obstacles = self.static_ghosts + list(active.values())
                return any(ghost.check_collision(other) for ghost in held for other in obstacles
                           if other is not ghost and other.is_collidable())
        except (MultitaskProtocolError, MultitaskInfrastructureError):
            raise
        except Exception as exc:
            raise MultitaskInfrastructureError('public ghost collision backend failed') from exc

    def plan_path(self, position, quaternion, **kwargs):
        self.path_queries += 1
        with self.public_scene(include_attachments=False):
            return self.env.arm.get_path(position, quaternion=quaternion, **kwargs)

    def receipt(self):
        return dict(access='shared_static_snapshot_and_caller_estimated_multiobject_poses',
                    map_sha256=self.sha256, static_shapes=len(self.static), queries=self.queries,
                    static_geometry_rule='collidable OR respondable at episode-start snapshot',
                    noncolliding_geometry=dict(self.noncolliding_geometry),
                    path_queries=self.path_queries, excluded_native_dynamic_shapes=len(self._dynamic_excluded),
                    robot_contact_queries=self.robot_contact_queries,
                    allowed_robot_contact_ids_seen=sorted(self.allowed_robot_contact_ids_seen),
                    geometry_mode_queries=dict(self.geometry_mode_queries),
                    registered_fallback_default='rotation_uncertainty; primary requires explicit caller evidence',
                    robot_contact_permissions_do_not_exempt_held_world_collisions=True,
                    required_dynamic_role_ids=list(self.required_dynamic_role_ids),
                    map_is_episode_start_snapshot=True, dynamic_ground_truth_returned=False,
                    live_dynamic_poses_read_during_query=False, live_dynamic_pose_setters_called=False,
                    attached_native_subtree_may_follow_kinematic_robot_probes=True,
                    kinematic_probe_restores='robot joints, servo targets, IK target; physics/task clock checked',
                    queries_advance_physics=False,
                    held_collision='caller RGB-D/proprioception attachment and robot FK; swept samples',
                    estimated_attachments={key: pose.xyzw().tolist()
                                           for key, pose in self.estimated_attachments.items()},
                    estimated_geometry={key: dict(sha256=value.geometry_sha256, provenance=value.provenance,
                                                 source_sha256=value.source_sha256,
                                                 collision_geometry_receipt=value.collision_geometry_receipt,
                                                 fallback_sha256=value.fallback_geometry_sha256,
                                                 fallback_collision_geometry_receipt=value.fallback_geometry_receipt)
                                        for key, value in self.estimated_bodies.items()})

    def close(self):
        errors = []
        for ghost in self.static_ghosts + [ghost for body in self.estimated_bodies.values() for ghost in body.ghosts()]:
            try:
                ghost.remove()
            except Exception as exc:
                errors.append(repr(exc))
        self.static_ghosts.clear()
        self.estimated_bodies.clear()
        self.estimated_attachments.clear()
        self.noncolliding_geometry.clear()
        if errors:
            raise MultitaskInfrastructureError('collision ghost cleanup failed: ' + '; '.join(errors))


class AlignedRobotModel(PyRepRobotModel):
    def __init__(self, arm, service):
        super().__init__(arm)
        self.service = service
        self.collision_scene = 'r10_shared_static_and_caller_multiobject_forecast'
        self.joint_limit_probe_rejections = 0

    def _outside_current_interval(self, joints=None):
        measured = np.asarray(self.arm.get_joint_positions() if joints is None else joints, dtype=float)
        if (measured.shape != self.lower.shape or not np.isfinite(measured).all()):
            raise MultitaskInfrastructureError('invalid measured robot configuration')
        outside = (measured < self.lower) | (measured > self.upper)
        if outside.any():
            self.joint_limit_probe_rejections = getattr(self, 'joint_limit_probe_rejections', 0) + 1
        return bool(outside.any())

    def _probe(self, joints, fn):
        clock = self.service._clock()
        before, targets = self.arm.get_joint_positions(), self.arm.get_joint_target_positions()
        # Physical contacts can put a measured joint slightly beyond its hard
        # interval. simSetJointPosition clamps it, including on restoration.
        # Reject the query before any mutation; do not relax the restore check.
        if self._outside_current_interval(before):
            raise MultitaskProtocolError('kinematic probe unavailable while actual joints exceed their intervals')
        ik = self.arm._ik_target.get_pose()
        try:
            self.arm.set_joint_positions(np.asarray(joints).tolist())
            return fn()
        except (MultitaskProtocolError, MultitaskInfrastructureError):
            raise
        except Exception as exc:
            raise MultitaskInfrastructureError('kinematic backend probe failed') from exc
        finally:
            try:
                self.arm.set_joint_positions(before)
                self.arm.set_joint_target_positions(targets)
                self.arm._ik_target.set_pose(ik)
                restored_ik = np.asarray(self.arm._ik_target.get_pose(), dtype=float)
                original_ik = np.asarray(ik, dtype=float)
                valid_poses = (restored_ik.shape == original_ik.shape == (7,)
                    and np.isfinite(restored_ik).all() and np.isfinite(original_ik).all()
                    and abs(np.linalg.norm(restored_ik[3:])-1.) <= 1e-6
                    and abs(np.linalg.norm(original_ik[3:])-1.) <= 1e-6)
                # Native set_pose can return -q for the same orientation.
                # Keep the existing tolerance and compare physical SE(3).
                restored_pose = (valid_poses
                    and np.allclose(restored_ik[:3], original_ik[:3], rtol=0, atol=1e-6)
                    and (Rotation.from_quat(restored_ik[3:]).inv()
                         * Rotation.from_quat(original_ik[3:])).magnitude() <= 1e-6)
                if (not np.allclose(self.arm.get_joint_positions(), before, rtol=0, atol=1e-6)
                        or not np.allclose(self.arm.get_joint_target_positions(), targets, rtol=0, atol=1e-6)
                        or not restored_pose):
                    raise MultitaskProtocolError('kinematic probe failed to restore robot state')
                self.service._clock_check(clock, 'kinematic probe')
            except (MultitaskProtocolError, MultitaskInfrastructureError):
                raise
            except Exception as exc:
                raise MultitaskInfrastructureError('kinematic robot state cleanup failed') from exc

    def inverse(self, pose, seed):
        if self._outside_current_interval():
            return None
        with self.service.public_scene(include_attachments=False):
            return super().inverse(pose, seed)

    def forward(self, joints):
        current = np.asarray(self.arm.get_joint_positions(), dtype=float)
        if self._outside_current_interval(current) and np.array_equal(np.asarray(joints), current):
            return Pose.from_xyzw(self.arm.get_tip().get_pose())
        return super().forward(joints)

    def collision(self, joints, future_offset):
        if self._outside_current_interval():
            return True
        return self._probe(joints, lambda: bool(self.collision_query(future_offset)
            if self.collision_query else self.service.check()))


class KnownMapForecastCollision:
    def __init__(self, service):
        self.service, self.forecast, self.origin = service, None, None
        self.body_ids, self.ignored_object_ids = (), ()
        self.allowed_robot_contact_ids = ()

    def bind(self, forecast, robot_time, *, body_ids=None, ignored_object_ids=(), allowed_robot_contact_ids=()):
        self.forecast, self.origin = forecast, float(robot_time)
        supplied = forecast.object_ids if body_ids is None else body_ids
        noncolliding = set(getattr(self.service, 'noncolliding_geometry', {}))
        self.body_ids = tuple(sorted((set(self.service.required_dynamic_role_ids) | set(supplied)) - noncolliding))
        self.ignored_object_ids = tuple(ignored_object_ids)
        self.allowed_robot_contact_ids = tuple(allowed_robot_contact_ids)

    def check(self, future_offset):
        if (self.forecast is None or not np.isfinite([self.origin, future_offset]).all()
                or future_offset < 0):
            return True
        try:
            poses = {key: self.forecast.at(key, self.origin + future_offset)[0] for key in self.body_ids
                     if key not in self.ignored_object_ids and key not in self.service.estimated_attachments}
        except (IndexError, ValueError):
            return True
        if set(poses) - set(self.service.estimated_bodies):
            return True
        return self.service.check(poses, ignored_object_ids=self.ignored_object_ids,
                                  allowed_robot_contact_ids=self.allowed_robot_contact_ids)
