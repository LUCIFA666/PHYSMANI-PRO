"""Measured rigid held geometry without changing exogenous tracking histories.

Only an unambiguous current RGB-D registration at public contact admits the
TRAIN mesh. Missing evidence retains the original all-rotation enclosure.
The mesh pose may be constrained by the existing finite static support, with
the original component registration quality gates checked again. No native
dynamic object, task waypoint, extra camera or physics query is used.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
import json
import numpy as np
from scipy.spatial import cKDTree

from ..contracts import Pose
from .collision_geometry import collision_geometry_for_orientation,mesh_sha256
from .evaluation import file_sha
from .forecasting import ObservedAttachment
from .perception import _sample
from .pipeline import _fresh_direct
from .protocol import MultitaskProtocolError
from .safe_native_mesh import read_native_mesh
from .supported_sphere_attachment_v10 import horizontal_rectangle

SCHEMA='r10_current_public_rigid_held_TRAIN_mesh_v33'
TASKS=('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')
ROLE='rubbish'


def require(value,message):
    if not value:raise MultitaskProtocolError(message)


def binding(root,protocol,task):
    root=Path(root).resolve();require(task in TASKS,'held mesh supports only the two rubbish TRAIN assets')
    pointer=protocol['learned_artifacts'][task]['geometry'];path=root/pointer['path']
    require(file_sha(path)==pointer['sha256'],'held TRAIN CAD bytes changed')
    asset=json.loads(path.read_text());body=asset['bodies'][ROLE]
    exact=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='full')
    fallback=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='none')
    return dict(schema=SCHEMA,task=task,role=ROLE,source=dict(path=str(Path(__file__).relative_to(root)),sha256=file_sha(__file__)),
        TRAIN_geometry=pointer,TRAIN_asset_sha256=asset['asset_sha256'],
        primary_mesh_sha256=exact.receipt['registered_mesh_sha256'],fallback_mesh_sha256=fallback.receipt['registered_mesh_sha256'],
        admission='same-current-RGBD valid unambiguous original registration plus public attached contact; rigid transform persists only under continuous public contact',
        support_constraint='finite static workspace footprint and current surface above support; original component RMS and inlier gates rechecked',
        global_orientation_observability_changed=False,model_history_changed=False,
        collision_permissions_changed=False,native_dynamic_pose_used=False,extra_camera_or_physics_calls=False,
        calibrated_pose_error_guarantee=False,formal_task_admission=False)


@contextmanager
def install_registration(service,protocol,root,task):
    expected=binding(root,protocol,task)
    require(protocol.get('public_held_mesh_v33',{}).get(task)==expected,'freeze exact held geometry binding')
    asset=json.loads((Path(root)/expected['TRAIN_geometry']['path']).read_text());body=asset['bodies'][ROLE]
    exact=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='full')
    fallback=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],orientation_observability='none')
    original=service.register_estimated_geometry;seen=[]
    def register(instance,object_id,vertices,triangles,**kwargs):
        if object_id==ROLE:
            require(object_id not in instance.estimated_bodies and kwargs.get('source_sha256')==asset['asset_sha256']
                and mesh_sha256(vertices,triangles)==expected['fallback_mesh_sha256'] and kwargs.get('fallback_mesh') is None,
                'replace only the fresh frozen rubbish all-rotation registration')
            kwargs=dict(kwargs,collision_geometry_receipt=dict(exact.receipt,TRAIN_asset_sha256=asset['asset_sha256']),
                fallback_mesh=dict(vertices=fallback.vertices,triangles=fallback.triangles,
                    collision_geometry_receipt=dict(fallback.receipt,TRAIN_asset_sha256=asset['asset_sha256'])))
            vertices,triangles=exact.vertices,exact.triangles;seen.append(object_id)
        return original(object_id,vertices,triangles,**kwargs)
    service.register_estimated_geometry=MethodType(register,service)
    try:
        yield expected
        require(seen==[ROLE],'held mesh must register exactly once')
    finally:service.register_estimated_geometry=original


def quality(pose,clouds,template,config):
    distances=[]
    for component in template.components:
        observed=np.asarray(clouds[component.name],float)
        if len(observed)<3:continue
        model=_sample(component.points_body,config.maximum_points_per_component)
        distances.extend(cKDTree(model).query(pose.inverse_apply(observed))[0].tolist())
    d=np.asarray(distances);inliers=d<=2*config.maximum_registration_rms_m
    rms=float(np.sqrt(np.mean(d[inliers]**2))) if inliers.any() else float('inf')
    fraction=float(inliers.mean()) if len(d) else 0.
    return dict(rms_m=rms,inlier_fraction=fraction,point_count=len(d),
        valid=bool(len(d)>=config.minimum_visible_points and rms<=config.maximum_registration_rms_m and fraction>=config.minimum_inlier_fraction))


def supported_pose(pose,clouds,template,vertices,plane,config):
    points=np.concatenate([np.asarray(v,float) for v in clouds.values() if len(v)],axis=0)
    world=pose.apply(vertices);lo=np.asarray(plane['xy_min']);hi=np.asarray(plane['xy_max']);height=plane['height']
    row=dict(accepted=False,original_xyzw=pose.xyzw().tolist(),source='same_current_RGBD_and_shared_static_mesh')
    if (len(points)<config.minimum_visible_points or not np.isfinite(points).all()
            or np.mean(points[:,2]>=height)<config.minimum_inlier_fraction
            or np.any(world[:,:2].min(0)<lo) or np.any(world[:,:2].max(0)>hi)):
        return pose,dict(row,reason='current_surface_or_mesh_footprint_does_not_establish_support')
    margin=16*float(np.spacing(np.float32(max(abs(height),1.))))
    shift=max(0.,height+margin-float(world[:,2].min()))
    candidate=Pose(pose.position+np.array([0.,0.,shift]),pose.rotation)
    q=quality(candidate,clouds,template,config)
    row.update(accepted=q['valid'],correction_m=shift,quality=q,constrained_xyzw=candidate.xyzw().tolist(),
        reason='original_component_registration_quality_passed' if q['valid'] else 'original_registration_quality_rejected_support_correction')
    return (candidate if q['valid'] else pose),row


def registration_role_valid(result,role):
    value=result['per_role'].get(role,{}).get('valid')
    return isinstance(value,(bool,np.bool_)) and bool(value)


def held_admissible(state,modes,invalid,evidence,maximum_age):
    if not (state.robot.contact and modes.get(ROLE)=='attached' and ROLE not in invalid):return False
    if evidence is None:return False
    age=float(state.robot.timestamp)-evidence['timestamp']
    return bool(-1e-8<=age<=maximum_age and _fresh_direct(state,ROLE,maximum_age))


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    require(protocol.get('public_held_mesh_v33',{}).get(task)==expected,'public held mesh binding changed')
    service=actor.map;matches=[g for info,g in zip(service.map,service.static_ghosts) if info['name']=='workspace']
    require(len(matches)==1,'one shared finite workspace plane required')
    ghost=matches[0];v,f,_=read_native_mesh(ghost);plane=horizontal_rectangle(v,f,ghost.get_matrix())
    asset=json.loads((Path(root)/expected['TRAIN_geometry']['path']).read_text());vertices=np.asarray(asset['bodies'][ROLE]['collision_vertices_body'],float)
    tracker=actor.tracker;previous_registration=tracker._registration;current={};active={};events=[];attachment_events=[]
    def registration(instance,members,clouds,previous=None):
        result=previous_registration(members,clouds,previous)
        if ROLE in members:
            current.clear();row=dict(time=float(instance._current_timestamp),valid=False)
            if (result is not None and result['valid'] and not result['rotation_ambiguous']
                    and registration_role_valid(result,ROLE)):
                local=members[ROLE];base=result['pose'];raw=Pose(base.position+base.rotation@local.position,base.rotation@local.rotation)
                pose,support=supported_pose(raw,clouds[ROLE],instance.templates[ROLE],vertices,plane,instance.config)
                q=quality(pose,clouds[ROLE],instance.templates[ROLE],instance.config)
                row.update(valid=q['valid'],quality=q,support=support,measured_xyzw=pose.xyzw().tolist())
                if q['valid']:current.update(timestamp=float(instance._current_timestamp),pose=pose)
            events.append(row)
        return result  # Preserve exactly the original exogenous tracker result.
    tracker._registration=MethodType(registration,tracker)
    previous_attachments=actor._attachments
    def attachments(instance,state,modes,invalid):
        previous_attachments(state,modes,invalid)
        if not (state.robot.contact and modes.get(ROLE)=='attached' and ROLE not in invalid):
            active.clear();return
        maximum_age=instance.controller.generator.config.maximum_observation_age
        if held_admissible(state,modes,invalid,current or None,maximum_age):
            pose=current['pose'];tcp=state.robot.pose;local=Pose(tcp.rotation.T@(pose.position-tcp.position),tcp.rotation.T@pose.rotation)
            item=ObservedAttachment(ROLE,local,float(state.robot.timestamp));instance.attachments[ROLE]=item
            active.update(attachment=item)
            attachment_events.append(dict(time=float(state.robot.timestamp),local_xyzw=local.xyzw().tolist(),source='current_RGBD_unique_registration_and_public_contact'))
        elif active and ROLE in instance.attachments:
            instance.attachments[ROLE]=active['attachment']
    actor._attachments=MethodType(attachments,actor)
    previous_modes=actor.collision_port._geometry_modes
    def geometry_modes(instance):
        modes,reasons=previous_modes();state,_=instance.frame
        if (active and state.robot.contact and ROLE in instance.attachments
                and ROLE in service.estimated_attachments):
            modes[ROLE]='primary';reasons[ROLE]=dict(reason='current-contact rigid attachment from unique current public RGBD fit',
                primary_geometry_kind='exact_TRAIN_triangles',full_orientation_promoted=False,
                held_only=True,observed_at=active['attachment'].observed_at)
        return modes,reasons
    actor.collision_port._geometry_modes=MethodType(geometry_modes,actor.collision_port)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),public_held_mesh_v33=dict(binding=expected,
        shared_static_plane=plane,registration_events=deepcopy(events),attachment_events=deepcopy(attachment_events)))
    actor.metrics_report=MethodType(metrics,actor)
