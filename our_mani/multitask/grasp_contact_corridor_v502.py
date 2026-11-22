"""Only qualified acquisition-corridor finger contacts may touch a target.

Use current public pose evidence, the same predicted ghost poses and restored
robot FK previews. All original checks run first. Palm, table and unselected
body contacts remain solid. Native contact metrics never enter this policy.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .interaction import _world_target
from .evaluation import file_sha


def binding(root,native_repo=None):
 api={}
 if native_repo is not None:
  for name in ('shape.py','object.py','joint.py'):
   path=(Path(native_repo)/'third_party/PyRep/pyrep/objects'/name).resolve();api[str(path)]=file_sha(path)
  for name in ('robots/robot_component.py','robots/end_effectors/gripper.py','backend/sim.py'):
   path=(Path(native_repo)/'third_party/PyRep/pyrep'/name).resolve();api[str(path)]=file_sha(path)
 return dict(native_static_API_sources_sha256=api,schema='r10_public_fitted_contact_corridor_finger_only_v502',
  sources_sha256={p:file_sha(Path(root)/p) for p in (
   'our_mani/multitask/grasp_contact_corridor_v502.py','our_mani/multitask/maps.py')},
  additional_complete_gripper_to_static_and_dynamic_pairs=True,
  contact_guard_is_an_additional_engineering_corridor_not_a_fitted_contact_classifier=True,
  target_contact='finger-owned physical shapes only; original TRAIN fit upper caps3cm/.15rad for contact only; original2cm/.1rad close unchanged; current full-R evidence',
  held_contact='own finger shapes only after existing public attachment admission',
  native_grasp_motion_success_and_original_collision_checks_unchanged=True,
  dynamic_native_pose_or_contacts_used=False,extra_physics_or_camera_queries=False,formal_VAL_TEST=False)


def qualified(actor,identity,poses,modes,tool):
 port=actor.collision_port;bound=port.bound;frame=port.frame
 if bound is None or frame is None or bound['phase']!='acquire' or identity not in bound['manipulated_ids']:
  return False,dict(reason='not_current_selected_acquire_target')
 state,_=frame;body=state.bodies.get(identity);meta=state.perception_metadata.get('bodies',{}).get(identity,{})
 stamp=meta.get('pose_evidence_until');now=float(state.robot.timestamp)
 fresh=(body is not None and meta.get('registration_valid') is True and meta.get('orientation_observable') is True
  and isinstance(stamp,(int,float)) and np.isfinite(stamp) and abs(float(stamp)-float(body.timestamp))<=1e-7
  and -1e-8<=now-float(stamp)<=port.maximum_observation_age)
 if not fresh or modes.get(identity)!='primary':return False,dict(reason='current_full_orientation_grasp_geometry_unavailable')
 reference=poses.get(identity)
 if not isinstance(reference,Pose):return False,dict(reason='predicted_public_target_pose_missing')
 c=actor.controller;geometry=[]
 fit=c.library.metadata['fit_configuration'];contact_position=float(fit['maximum_position_tolerance_m']);contact_angle=float(fit['maximum_orientation_tolerance_rad'])
 for t in c.library.for_phase('acquire'):
  if t.reference_role!=identity or t.manipulated_role!=identity or t.reference_kind!='observed_se3' or t.reference_time_basis!='current':continue
  target=_world_target(reference,np.zeros(3),np.zeros(3),t,1.)[0]
  distance=float(np.linalg.norm(tool.position-target.position))
  angle=float(Rotation.from_matrix(np.array(tool.rotation@target.rotation.T,copy=True)).magnitude())
  geometry.append(dict(distance_m=distance,angle_rad=angle))
  if distance<=contact_position and angle<=contact_angle:
   return True,dict(reason='within_original_fit_cap_contact_corridor',distance_m=distance,angle_rad=angle,contact_position_cap=contact_position,contact_orientation_cap=contact_angle)
 return False,dict(reason='outside_original_fit_cap_contact_corridor',evaluated_geometry=geometry)


def install(actor,env,*,shape_type=None,native_repo=None):
 c=actor.controller
 if c.feedback_step_calls:raise ValueError('install before counted task control')
 if (abs(c.gripper_timing_config.close_position_tolerance_m-.02)>1e-12
  or abs(c.gripper_timing_config.close_orientation_tolerance_rad-.1)>1e-12):
  raise ValueError('retain actual original physical2cm/.1rad grasp gate')
 fit=c.library.metadata.get('fit_configuration',{})
 if (fit.get('maximum_position_tolerance_m')!=.03 or fit.get('maximum_orientation_tolerance_rad')!=.15):
  raise ValueError('preserve original TRAIN3cm/.15rad fitting limits; no new tolerance fit')
 if shape_type is None:
  if native_repo is None:raise ValueError('freeze exact native static shape/finger APIs')
  from pyrep.const import ObjectType
  shape_type=ObjectType.SHAPE
 def physical(s):return bool(s.is_collidable() or s.is_respondable())
 shapes=[s for s in env.gripper.get_objects_in_tree(object_type=shape_type,exclude_base=False) if physical(s)]
 fingers={int(s.get_handle()) for joint in env.gripper.joints
  for s in joint.get_objects_in_tree(object_type=shape_type,exclude_base=False) if physical(s)}
 handles={int(s.get_handle()) for s in shapes}
 if not shapes or not fingers or not fingers<=handles:raise ValueError('complete native gripper and finger shape ownership required')
 expected=binding(Path(__file__).resolve().parents[2],native_repo)
 names={int(s.get_handle()):s.get_name() for s in shapes};service=actor.collision_port.map
 original=service.check;events=[];counts=dict(checks=0,pairs=0,rejected=0,qualified_target_permissions=0)
 def check(instance,poses=None,*,ignored_object_ids=(),allowed_robot_contact_ids=(),geometry_modes=None):
  result=original(poses,ignored_object_ids=ignored_object_ids,allowed_robot_contact_ids=allowed_robot_contact_ids,geometry_modes=geometry_modes)
  if result:return True
  poses={} if poses is None else dict(poses);modes={} if geometry_modes is None else dict(geometry_modes)
  counts['checks']+=1
  tool=Pose.from_xyzw(env.arm.get_tip().get_pose()) # same restoring FK preview, no dynamic native body pose
  with instance.public_scene(poses,ignored_object_ids=ignored_object_ids,geometry_modes=geometry_modes) as active:
   for ghost in instance.static_ghosts:
    for shape in shapes:
     counts['pairs']+=1
     if ghost.check_collision(shape):
      counts['rejected']+=1
      if len(events)<256:events.append(dict(body='static_world',gripper_shape=names[int(shape.get_handle())],reason='gripper_static_collision'))
      return True
   for identity,target in active.items():
    held=identity in instance.estimated_attachments
    permission,evidence=(False,dict(reason='unselected_target_contact_forbidden'))
    if held:permission=True;evidence=dict(reason='existing_public_attachment_finger_contact')
    elif identity in allowed_robot_contact_ids:permission,evidence=qualified(actor,identity,poses,modes,tool)
    counts['qualified_target_permissions']+=int(permission and not held)
    for shape in shapes:
     handle=int(shape.get_handle())
     if permission and handle in fingers:continue
     counts['pairs']+=1
     if target.check_collision(shape):
      counts['rejected']+=1
      if len(events)<256:events.append(dict(body=identity,gripper_shape=names[handle],finger_shape=handle in fingers,
       held=held,grasp_region=deepcopy(evidence),current_public_evidence_only=True))
      return True
  return False
 service.check=MethodType(check,service)
 previous=actor.metrics_report
 def metrics(instance):return dict(previous(),grasp_contact_corridor_v502=dict(binding=deepcopy(expected),
  physical_gripper_shapes=list(names.values()),finger_shape_handles=sorted(fingers),counts=deepcopy(counts),events=deepcopy(events),
  all_original_arm_world_and_held_world_checks_retained=True))
 actor.metrics_report=MethodType(metrics,actor)
 return counts,events
