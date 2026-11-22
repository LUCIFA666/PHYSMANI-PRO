"""Support-constrained held cup transform from current public RGB-D only.

The existing external tracker result and collision meshes are unchanged. At
unambiguous full-R public contact, the original registration quality gates
also verify a pose constrained by the shared finite static workspace plane.
Only the held tool-to-cup transform uses that measurement.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import json
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .forecasting import ObservedAttachment
from .pipeline import _fresh_direct
from .safe_native_mesh import read_native_mesh
from .supported_sphere_attachment_v10 import horizontal_rectangle
from .public_held_mesh_v33 import quality,supported_pose,registration_role_valid

TASKS=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')
ROLE='cup0'


def binding(root,protocol,task):
    root=Path(root).resolve()
    if task not in TASKS:raise ValueError('only the two full-R cup placement tasks')
    if protocol['body_orientation_observability'][task][ROLE]!='full':raise ValueError('original full-R cup capability required')
    pointer=protocol['learned_artifacts'][task]['geometry'];path=root/pointer['path']
    if file_sha(path)!=pointer['sha256']:raise ValueError('cup TRAIN geometry changed')
    asset=json.loads(path.read_text())
    if ROLE not in asset['bodies']:raise ValueError('original cup0 mesh missing')
    return dict(schema='r10_current_public_fullR_held_cup_static_support_v148',task=task,role=ROLE,
        TRAIN_geometry=pointer,TRAIN_asset_sha256=asset['asset_sha256'],
        sources_sha256={name:file_sha(root/name) for name in (
            'our_mani/multitask/cup_public_support_v148.py','our_mani/multitask/public_held_mesh_v33.py',
            'our_mani/multitask/safe_native_mesh.py','our_mani/multitask/supported_sphere_attachment_v10.py')},
        measurement='same-current unique full-R RGBD cup fit; original component RMS/inlier gates after finite static support constraint',
        held_only=True,public_contact_and_fresh_direct_measurement_required=True,
        external_history_or_original_body_estimate_modified=False,
        collision_meshes_or_permissions_modified=False,native_dynamic_pose_used=False,
        additional_camera_or_physics_calls=False,calibrated_pose_error_guarantee=False,formal_release=False)


def admissible(state,modes,invalid,evidence,maximum_age):
    if not (state.robot.contact and modes.get(ROLE)=='attached' and ROLE not in invalid):return False
    if not evidence or not evidence.get('support_quality_passed'):return False
    if not _fresh_direct(state,ROLE,maximum_age):return False
    meta=state.perception_metadata.get('bodies',{}).get(ROLE,{})
    if meta.get('orientation_observable') is not True:return False
    age=float(state.robot.timestamp)-float(evidence['timestamp'])
    return bool(-1e-8<=age<=maximum_age)


def update_attachment(state,modes,invalid,current,active,attachments,maximum_age):
    if not (state.robot.contact and modes.get(ROLE)=='attached' and ROLE not in invalid):
        active.clear();return None
    if admissible(state,modes,invalid,current,maximum_age):
        pose=current['pose'];tcp=state.robot.pose
        local=Pose(tcp.rotation.T@(pose.position-tcp.position),tcp.rotation.T@pose.rotation)
        item=ObservedAttachment(ROLE,local,float(state.robot.timestamp))
        attachments[ROLE]=item;active['attachment']=item
        return item
    if active and ROLE in attachments:attachments[ROLE]=active['attachment']
    return None


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('cup_public_support_v148',{}).get(task)!=expected:raise ValueError('freeze cup public support binding')
    if actor.controller.feedback_step_calls or actor.continuous_reference_manager.confirmations:
        raise ValueError('install support measurement before first step')
    service=actor.map;matches=[g for info,g in zip(service.map,service.static_ghosts) if info['name']=='workspace']
    if len(matches)!=1:raise ValueError('one original shared static workspace required')
    ghost=matches[0];v,f,_=read_native_mesh(ghost);plane=horizontal_rectangle(v,f,ghost.get_matrix())
    asset=json.loads((Path(root)/expected['TRAIN_geometry']['path']).read_text())
    vertices=np.asarray(asset['bodies'][ROLE]['collision_vertices_body'],float)
    tracker=actor.tracker;previous_registration=tracker._registration;current={};active={};events=[];attached=[]
    def registration(instance,members,clouds,previous=None):
        result=previous_registration(members,clouds,previous)
        if ROLE in members:
            current.clear();row=dict(time=float(instance._current_timestamp),valid=False)
            if result is not None and result['valid'] and not result['rotation_ambiguous'] and registration_role_valid(result,ROLE):
                local=members[ROLE];base=result['pose']
                raw=Pose(base.position+base.rotation@local.position,base.rotation@local.rotation)
                pose,support=supported_pose(raw,clouds[ROLE],instance.templates[ROLE],vertices,plane,instance.config)
                q=quality(pose,clouds[ROLE],instance.templates[ROLE],instance.config)
                accepted=bool(support['accepted'] and q['valid'])
                row.update(valid=accepted,support=support,quality=q,measured_xyzw=pose.xyzw().tolist())
                if accepted:current.update(timestamp=float(instance._current_timestamp),pose=pose,support_quality_passed=True)
            events.append(row)
        return result
    tracker._registration=MethodType(registration,tracker)
    previous_attachments=actor._attachments
    def attachments(instance,state,modes,invalid):
        previous_attachments(state,modes,invalid)
        item=update_attachment(state,modes,invalid,current,active,instance.attachments,
            instance.controller.generator.config.maximum_observation_age)
        if item is not None:attached.append(dict(time=float(state.robot.timestamp),local_xyzw=item.tool_to_body.xyzw().tolist(),
            source='same-current unique full-R RGBD and static support constrained quality, under public contact'))
    actor._attachments=MethodType(attachments,actor)
    before=actor.metrics_report
    def metrics(instance):return dict(before(),cup_public_support_v148=dict(binding=expected,
        shared_static_plane=plane,registration_events=deepcopy(events),attachment_events=deepcopy(attached)))
    actor.metrics_report=MethodType(metrics,actor)
