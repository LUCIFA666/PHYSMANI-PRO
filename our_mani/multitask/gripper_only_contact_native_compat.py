"""Restrict existing public target contact exemptions to gripper-owned shapes.

The original complete collision check still runs. This adds target-versus-arm
pairs it previously exempted, using the same predicted public ghost scene and
the same temporary robot FK preview. No dynamic native pose is read.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from .evaluation import file_sha


def binding(root):
    return dict(schema='r10_gripper_subtree_only_public_target_contact_v1',
        sources_sha256={p:file_sha(Path(root)/p) for p in ('our_mani/multitask/gripper_only_contact_native_compat.py',
            'our_mani/multitask/maps.py')},
        existing_collision_checks_retained=True,additional_target_to_non_gripper_arm_checks=True,
        dynamic_native_pose_read=False,extra_physics_or_camera_calls=False,formal_VAL_TEST=False)


def install(actor,env,*,shape_type=None):
    if actor.controller.feedback_step_calls:raise ValueError('install before execution')
    if shape_type is None:
        from pyrep.const import ObjectType
        shape_type=ObjectType.SHAPE
    service=actor.collision_port.map
    gripper=env.gripper.get_objects_in_tree(object_type=shape_type,exclude_base=False)
    gripper_handles={int(g.get_handle()) for g in gripper}
    arm=[s for s in env.arm.get_objects_in_tree(object_type=shape_type,exclude_base=False)
        if int(s.get_handle()) not in gripper_handles and s.is_collidable()]
    if not gripper_handles or not arm:raise ValueError('explicit complete robot and gripper shape ownership required')
    arm_names={int(s.get_handle()):s.get_name() for s in arm}
    original=service.check;events=[];counts=dict(checks=0,pairs=0,rejected=0)
    def check(instance,poses=None,*,ignored_object_ids=(),allowed_robot_contact_ids=(),geometry_modes=None):
        result=original(poses,ignored_object_ids=ignored_object_ids,
            allowed_robot_contact_ids=allowed_robot_contact_ids,geometry_modes=geometry_modes)
        if result:return True
        exempt=set(allowed_robot_contact_ids)|set(instance.estimated_attachments)
        if not exempt:return False
        counts['checks']+=1
        with instance.public_scene(poses,ignored_object_ids=ignored_object_ids,geometry_modes=geometry_modes) as active:
            for identity in sorted(exempt):
                target=active.get(identity)
                if target is None:continue
                for shape in arm:
                    counts['pairs']+=1
                    if target.check_collision(shape):
                        counts['rejected']+=1
                        if len(events)<128:events.append(dict(body=identity,robot_shape=arm_names[int(shape.get_handle())],
                            gripper_exemption_not_applied_to_arm=True,public_ghost_scene=True))
                        return True
        return False
    service.check=MethodType(check,service)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),gripper_only_contact=dict(binding=binding(Path(__file__).resolve().parents[2]),
        gripper_shape_handles=sorted(gripper_handles),non_gripper_arm_shapes=list(arm_names.values()),
        counts=deepcopy(counts),events=deepcopy(events),existing_held_world_and_arm_world_checks_retained=True))
    actor.metrics_report=MethodType(metrics,actor)
    return counts,events
