"""Shared r05 actuator and public-map collision service for both methods.

Reset truth and success conditions belong to the environment. The policy gets
RGB-D, proprioception and the explicitly declared static map. Dynamic-object
collision probes use a caller-provided estimate/forecast, never its live pose.
"""
from contextlib import contextmanager
import hashlib
import json
import numpy as np

from ..official_simulation import OfficialTestSimulation
from ..pyrep_env import LocalSimulation
from ..reachability import PyRepRobotModel
from .protocol import TickLedger

def physics_time():
    from pyrep.backend import sim
    return float(sim.simGetSimulationTime())



class SharedKnownMap:
    def __init__(self, env):
        from pyrep.const import ObjectType
        self.env, self.target = env, env.task.target_block
        self.queries = 0
        self.path_queries = 0
        robot_shapes = env.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
        robot_shapes += env.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
        excluded = {shape.get_handle() for shape in robot_shapes} | {
            env.arm.get_handle(),env.gripper.get_handle(),self.target.get_handle()}
        self.static = [shape for shape in env.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)
                       if shape.get_handle() not in excluded and shape.is_collidable()]
        self.map = []
        for shape in self.static:
            vertices,triangles,_ = shape.get_mesh_data()
            mesh_hash = hashlib.sha256(np.asarray(vertices,dtype=np.float32).tobytes()
                +np.asarray(triangles,dtype=np.int32).tobytes()).hexdigest()
            self.map.append(dict(name=shape.get_name(),pose=list(shape.get_pose()),
                                 bounding_box=list(shape.get_bounding_box()),mesh_sha256=mesh_hash))
        self.sha256 = hashlib.sha256(json.dumps(self.map,sort_keys=True).encode()).hexdigest()
        # Query an immutable map snapshot, not later hidden live-scene changes.
        self.ghosts = []
        from pyrep.objects.shape import Shape
        for shape in self.static:
            vertices,triangles,_ = shape.get_mesh_data()
            # Object.copy can recursively duplicate a model hierarchy. A raw
            # mesh creates exactly one inert collision object, with no children.
            ghost = Shape.create_mesh(np.asarray(vertices).reshape(-1).tolist(),
                                      np.asarray(triangles).reshape(-1).tolist())
            source_matrix,created_matrix = shape.get_matrix(),ghost.get_matrix()
            ghost.set_matrix(source_matrix@created_matrix)
            ghost.set_dynamic(False)
            ghost.set_respondable(False)
            ghost.set_renderable(False)
            ghost.set_collidable(False)
            self.ghosts.append(ghost)

    def close(self):
        for shape in self.ghosts:
            shape.remove()
        self.ghosts.clear()

    @contextmanager
    def estimated_target(self, pose=None):
        """Live pose is saved only for restoration and never returned to policy."""
        old_pose, collidable = self.target.get_pose(), self.target.is_collidable()
        live_flags = [shape.is_collidable() for shape in self.static]
        clock = physics_time()
        try:
            for shape in self.static:shape.set_collidable(False)
            for shape in self.ghosts:shape.set_collidable(True)
            if pose is None:
                self.target.set_collidable(False)
            else:
                self.target.set_pose(pose.xyzw().tolist(), reset_dynamics=False)
            yield
        finally:
            for shape in self.ghosts:shape.set_collidable(False)
            for shape,flag in zip(self.static,live_flags):shape.set_collidable(flag)
            self.target.set_pose(old_pose, reset_dynamics=False)
            self.target.set_collidable(collidable)
            if abs(physics_time()-clock) > 1e-6:
                raise RuntimeError('planning/collision query advanced physics')

    def check(self, pose=None):
        self.queries += 1
        with self.estimated_target(pose):
            return bool(self.env.arm.check_arm_collision())

    def plan_path(self,position,quaternion,**kwargs):
        self.path_queries += 1
        with self.estimated_target():
            return self.env.arm.get_path(position,quaternion=quaternion,**kwargs)

    def receipt(self):
        return dict(access='shared_static_map_and_caller_estimated_dynamic_target',
                    map_sha256=self.sha256, static_shapes=len(self.static), queries=self.queries,
                    path_queries=self.path_queries,
                    map_is_episode_start_snapshot=True,
                    dynamic_ground_truth_returned=False, queries_advance_physics=False)


class AlignedRobotModel(PyRepRobotModel):
    def __init__(self, arm, service):
        super().__init__(arm)
        self.service = service
        self.collision_scene = 'shared_public_map_and_method_forecast'

    def inverse(self, pose, seed):
        # IK may use the public static map, not the target's unobserved live pose.
        with self.service.estimated_target():
            return super().inverse(pose,seed)

    def collision(self, joints, future_offset):
        return self._probe(joints, lambda: bool(self.collision_query(future_offset)
            if self.collision_query else self.service.check()))


class KnownMapForecastCollision:
    def __init__(self, service):
        self.service = service
        self.forecast, self.origin = None, None

    def bind(self, forecast, robot_time):
        self.forecast, self.origin = forecast, float(robot_time)

    def check(self, future_offset):
        if self.forecast is None:
            return True
        try:
            pose = self.forecast.at('target',self.origin+future_offset)[0]
        except (IndexError,ValueError):
            return True
        return self.service.check(pose)


class AlignedActuator:
    """All commands use this nonblocking arm/gripper interface and counted tick."""
    def _start_alignment(self):
        if hasattr(self,'map_service'):
            self.map_service.close()
        self.ledger = TickLedger(dt=self.protocol['dt'],budget=self.protocol['simulation_seconds'])
        self.time = 0.
        self.previous_robot = None
        self._physics_origin = physics_time()
        self._task_origin = float(self.task.t)
        self._requested_opening = float(np.mean(self.gripper.get_open_amount()) > .5)
        self._native_success_latched = False
        self.map_service = SharedKnownMap(self)
        self.robot_model = AlignedRobotModel(self.arm,self.map_service)
        if self.protocol.get('current_instance_masks',False):
            from pyrep.objects.vision_sensor import VisionSensor
            from pyrep.const import RenderMode
            if not hasattr(self,'mask_cameras'):
                self.mask_cameras = {name:VisionSensor.create([256,256],explicit_handling=True,
                    render_mode=RenderMode.OPENGL_COLOR_CODED) for name in self.cameras}
            for name,camera in self.mask_cameras.items():
                source = self.cameras[name]
                camera.set_matrix(source.get_matrix())
                camera.set_perspective_angle(source.get_perspective_angle())
                camera.set_near_clipping_plane(source.get_near_clipping_plane())
                camera.set_far_clipping_plane(source.get_far_clipping_plane())

    def observe(self):
        observation,robot = super().observe()
        if self.protocol.get('current_instance_masks',False) and hasattr(self,'mask_cameras'):
            masks = {}
            for name,camera in self.mask_cameras.items():
                camera.handle_explicitly()
                masks[name] = np.asarray(camera.capture_rgb(),dtype=np.float32)
            observation['observation/instance_masks'] = masks
        return observation,robot

    def _tick(self, phase):
        self.ledger.before()
        physics_before = physics_time()
        task_before = float(self.task.t)
        # CoppeliaSim 4.1 exposes its absolute clock as float32. Long-lived
        # workers need an ULP-based tolerance, still far below a whole tick.
        tolerance = max(2e-5,2.*float(np.spacing(np.float32(max(1.,physics_before)))))
        if abs(physics_before-self._physics_origin-self.ledger.time) > tolerance:
            raise RuntimeError('physics advanced outside the shared tick')
        self.scene.step()
        self.ledger.after(physics_time()-physics_before,
                          float(self.task.t)-task_before,phase,physics_tolerance=tolerance)
        self.time = self.ledger.time
        success,_ = self.task.success()
        self._native_success_latched |= bool(success)

    def command_joints(self, joints, *, opening=None, phase='arm'):
        self.ledger.before()
        current = np.asarray(self.arm.get_joint_positions())
        requested = np.asarray(joints,dtype=float)
        if requested.shape != current.shape or not np.isfinite(requested).all():
            raise ValueError('invalid joint command')
        delta = np.clip(requested-current,-self.robot_model.velocity_limits*self.ledger.dt,
                        self.robot_model.velocity_limits*self.ledger.dt)
        bounded = np.clip(current+delta,self.robot_model.lower,self.robot_model.upper)
        self.arm.set_joint_target_positions(bounded.tolist())
        if opening is not None:
            requested_opening = float(opening >= .5)
            if requested_opening != self._requested_opening:
                if requested_opening:
                    self.gripper.release(); self.ledger.release_requests += 1
                else:
                    # One identical native grasp attempt per close transition.
                    for obj in self.task.get_graspable_objects():
                        self.gripper.grasp(obj)
                    self.ledger.grasp_requests += 1
                self._requested_opening = requested_opening
            self.gripper.actuate(requested_opening,velocity=self.protocol['gripper_velocity'])
        self._tick(phase)
        return True

    def advance(self, pose, *, opening=1.):
        current = np.asarray(self.arm.get_joint_positions())
        joints = self.robot_model.inverse(pose,current)
        if joints is None:
            self.command_joints(current,opening=opening,phase='ik_rejected')
            return False
        target = current+self.servo_lead*(np.asarray(joints)-current)
        return self.command_joints(target,opening=opening,phase='local_capture')

    def hold_tick(self):
        return self.command_joints(self.arm.get_joint_positions(),opening=self._requested_opening,phase='hold')

    def alignment_receipt(self):
        return dict(clock=self.ledger.receipt(),map=self.map_service.receipt(),
                    actual_physics_seconds=physics_time()-self._physics_origin,
                    actual_task_seconds=float(self.task.t)-self._task_origin,
                    native_success_latched=self._native_success_latched)


class AlignedOfficialSimulation(AlignedActuator,OfficialTestSimulation):
    def __init__(self, task_name, calibration_episode, *, protocol, seed=0, servo_lead=1.4):
        self.protocol = protocol
        super().__init__(task_name,calibration_episode,seed=seed,servo_lead=servo_lead)

    def reset_official(self, episode):
        result = super().reset_official(episode)
        self._start_alignment()
        return result


class AlignedLocalSimulation(AlignedActuator,LocalSimulation):
    def __init__(self, task_name, calibration_episode, *, protocol, seed=0, servo_lead=1.4):
        self.protocol = protocol
        super().__init__(task_name,calibration_episode,seed=seed,servo_lead=servo_lead)

    def reset(self, **kwargs):
        # Local training reset uses source kinematics and an ideal handover pose.
        # It is never represented as an official TEST reset or full-pipeline Value.
        self.robot_model = PyRepRobotModel(self.arm)
        result = super().reset(**kwargs)
        if result:
            self._start_alignment()
        return result

    def finish_home_reset(self):
        """Place the robot at the shared home state before the episode starts.

        PyRep's disable_dynamics setter advances two internal physics steps.
        These belong to reset staging; never call this during a rollout.
        """
        if self.ledger.ticks:
            raise RuntimeError('cannot reset robot after counted rollout has started')
        self.arm.set_joint_positions(self.home_joints.tolist(),disable_dynamics=True)
        self.arm.set_joint_target_positions(self.home_joints.tolist())
        self._start_alignment()
