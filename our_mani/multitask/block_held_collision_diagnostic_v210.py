"""Bounded read-only pair diagnostics; original collision decision is returned exactly."""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
from ..contracts import Pose
from .block_exact_reference_v201 import SmoothBlockEnvironment as OriginalBoundary, public_decision_scope


class SmoothBlockEnvironment(OriginalBoundary):
    def __init__(self, env, context=None):
        super().__init__(env, context)
        self.held_collision_diagnostics = []
        self.held_collision_diagnostic_candidates = 0

    def _diagnose(self, offset):
        from pyrep.const import ObjectType
        service, robot, body = self.env.map_service, self.public_robot, self.public_body
        tool = Pose.from_xyzw(self.env.arm.get_tip().get_pose())
        local_p = robot.pose.rotation.T @ (body.pose.position-robot.pose.position)
        local_r = robot.pose.rotation.T @ body.pose.rotation
        predicted = Pose(tool.position+tool.rotation@local_p, tool.rotation@local_r)
        with service.estimated_target(predicted):
            arm = [shape.get_name() for shape in self.env.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
                if service.target.check_collision(shape)]
            static = [info['name'] for shape,info in zip(service.ghosts,service.map)
                if service.target.check_collision(shape)]
            return dict(time=float(self.env.time), offset=float(offset),
                public_contact=True, public_body_id=body.object_id,
                predicted_held_body_pose=predicted.xyzw().tolist(),
                target_robot_colliding_shape_names=arm, target_static_colliding_shape_names=static,
                diagnostic_only=True, native_dynamic_pose_read_as_policy_input=False)

    @contextmanager
    def _collision_scope(self):
        with super()._collision_scope():
            model = self.env.robot_model
            original = model.collision_query
            def query(offset):
                result = original(offset)
                if result and self.public_robot.contact:
                    self.held_collision_diagnostic_candidates += 1
                    if len(self.held_collision_diagnostics) < 16:
                        before = (self.env.ledger.ticks,float(self.env.time))
                        try: row = self._diagnose(offset)
                        except Exception as error:
                            row = dict(time=float(self.env.time),offset=float(offset),diagnostic_unavailable=repr(error))
                        after = (self.env.ledger.ticks,float(self.env.time))
                        if before != after: raise RuntimeError('read-only held collision diagnostic changed clock')
                        self.held_collision_diagnostics.append(dict(row,original_collision_result=bool(result),
                            policy_result_changed=False,original_clock_unchanged=True))
                return result
            model.collision_query = query
            try: yield
            finally: model.collision_query = original

    def receipt(self):
        return dict(super().receipt(), held_collision_diagnostic_v210=dict(
            events=deepcopy(self.held_collision_diagnostics), candidate_queries=self.held_collision_diagnostic_candidates,
            maximum_retained_events=16, events_are_not_all_queries=True,
            collision_or_action_decisions_changed=False, native_physics_or_camera_calls_added=False,
            private_diagnostic_pairs_supplied_to_policy=False))
