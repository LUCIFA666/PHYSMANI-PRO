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
from .cup_public_support_v148 import binding as original_binding
from .safe_native_mesh import read_native_mesh
from .supported_sphere_attachment_v10 import horizontal_rectangle
from .public_held_mesh_v33 import quality,supported_pose,registration_role_valid

TASKS=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')
ROLE='cup0'


def binding(root,protocol,task):
    original=original_binding(root,protocol,task)
    if protocol.get('cup_public_support_v148',{}).get(task)!=original:
        raise ValueError('retain the exact original TRAIN support measurement binding')
    return dict(schema='r10_direct_RML_current_public_held_cup_support_v695',task=task,
        original_public_measurement=original,source_sha256=file_sha(Path(root)/'our_mani/multitask/rml_cup_support_v695.py'),
        old_reference_manager_required=False,original_registration_support_and_attachment_math_unchanged=True,
        observed_RGBD_and_static_workspace_only=True,additional_sensor_or_physics_calls=False)


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
    if protocol.get('RML_cup_public_support_v695',{}).get(task)!=expected:raise ValueError('freeze cup public support binding')
    if actor.controller.feedback_step_calls or actor.last_time is not None or actor.previous_states:
        raise ValueError('install support measurement before first step')
    service=actor.map;matches=[g for info,g in zip(service.map,service.static_ghosts) if info['name']=='workspace']
    if len(matches)!=1:raise ValueError('one original shared static workspace required')
    ghost=matches[0];v,f,_=read_native_mesh(ghost);plane=horizontal_rectangle(v,f,ghost.get_matrix())
    asset=json.loads((Path(root)/expected['original_public_measurement']['TRAIN_geometry']['path']).read_text())
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
    def metrics(instance):return dict(before(),RML_cup_public_support_v695=dict(binding=expected,
        shared_static_plane=plane,registration_events=deepcopy(events),attachment_events=deepcopy(attached)))
    actor.metrics_report=MethodType(metrics,actor)
