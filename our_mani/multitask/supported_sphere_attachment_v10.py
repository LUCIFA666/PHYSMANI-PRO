"""Current RGBD localization constrained by the declared static support plane.

The sphere radius and conservative collision mesh are TRAIN-derived. A plane
constraint can be used only when current measured surface points are above the
finite shared static plane and the fitted sphere lies inside its footprint.
Original registration residual/inlier gates remain mandatory. Fresh contact
measurements refresh tool-to-body transforms; native object poses are never read.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from scipy.optimize import least_squares
from ..contracts import Pose
from .evaluation import file_sha
from .forecasting import ObservedAttachment
from .pipeline import _fresh_direct
from .protocol import MultitaskProtocolError
from .safe_native_mesh import read_native_mesh
ROOT=Path(__file__).resolve().parents[2]
SCHEMA='r10_current_RGBD_static_support_and_fresh_attachment_v10'

def binding(root):
    return dict(schema=SCHEMA,source=dict(path=str(Path(__file__).relative_to(root)),sha256=file_sha(__file__)),
        support_role='workspace',sphere_role='ball',support_constraint='current measured surface satisfies original minimum-inlier fraction above finite rectangular horizontal static plane; center above all-rotation mesh outer radius',
        quality_gates='original RegistrationConfig maximum_registration_rms_m/minimum_inlier_fraction',
        clearance='16 float32 ulps at support height, numerical storage only',
        attachment_update='each fresh directly observed RGBD body pose at public contact in attached mode',
        native_dynamic_pose_used=False,collision_permissions_changed=False,extra_camera_or_physics_calls=False)

def horizontal_rectangle(vertices,faces,matrix):
    v=np.asarray(vertices,float)@np.asarray(matrix)[:3,:3].T+np.asarray(matrix)[:3,3]
    f=np.asarray(faces,int);lo,hi=v.min(axis=0),v.max(axis=0)
    tol=32*np.finfo(np.float32).eps*max(1.,np.max(np.abs(v)))
    if hi[2]-lo[2]>tol or np.any(hi[:2]-lo[:2]<=0):raise MultitaskProtocolError('support mesh is not a horizontal plane')
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).sum()/2
    if abs(area-np.prod(hi[:2]-lo[:2]))>tol or not all(np.any(np.linalg.norm(v[:,:2]-corner,axis=1)<tol) for corner in ((lo[0],lo[1]),(lo[0],hi[1]),(hi[0],lo[1]),(hi[0],hi[1]))):
        raise MultitaskProtocolError('support plane lacks complete rectangular footprint')
    return dict(height=float(hi[2]),xy_min=lo[:2].tolist(),xy_max=hi[:2].tolist())

def constrained_fit(points,pose,radius,outer_radius,plane,config):
    points=np.asarray(points,float);height=plane['height'];lo=np.asarray(plane['xy_min']);hi=np.asarray(plane['xy_max'])
    margin=16*float(np.spacing(np.float32(max(abs(height),1.))))
    floor=height+outer_radius+margin
    row=dict(unconstrained_center=pose.position.tolist(),constraint_floor=floor,current_points=len(points),accepted=False)
    if (len(points)<config.minimum_visible_points or not np.isfinite(points).all() or
            float(np.mean(points[:,2]>=height))<config.minimum_inlier_fraction or pose.position[2]<=height or
            np.any(pose.position[:2]-outer_radius<lo) or np.any(pose.position[:2]+outer_radius>hi)):
        return pose,dict(row,reason='current_surface_or_footprint_does_not_establish_above_support')
    initial=pose.position.copy();initial[2]=max(initial[2],floor)
    fit=least_squares(lambda c:np.linalg.norm(points-c,axis=1)-radius,initial,
        bounds=([lo[0]+outer_radius,lo[1]+outer_radius,floor],[hi[0]-outer_radius,hi[1]-outer_radius,np.inf]),loss='soft_l1',f_scale=.003)
    residual=np.abs(np.linalg.norm(points-fit.x,axis=1)-radius);mask=residual<=2*config.maximum_registration_rms_m
    rms=float(np.sqrt(np.mean(residual[mask]**2))) if mask.any() else float('inf');inliers=float(mask.mean())
    accepted=bool(fit.success and np.isfinite(fit.x).all() and rms<=config.maximum_registration_rms_m and inliers>=config.minimum_inlier_fraction)
    row.update(above_support_current_point_fraction=float(np.mean(points[:,2]>=height)),accepted=accepted,rms_m=rms,inlier_fraction=inliers,correction_m=float(np.linalg.norm(fit.x-pose.position)),constrained_center=fit.x.tolist(),reason='original_registration_quality_passed' if accepted else 'original_registration_quality_rejected')
    return (Pose(fit.x,pose.rotation) if accepted else pose),row

def install(actor,protocol,root):
    if protocol.get('public_support_attachment_v10')!=binding(root):raise MultitaskProtocolError('v9 public measurement binding differs')
    service=actor.map;matches=[ghost for info,ghost in zip(service.map,service.static_ghosts) if info['name']=='workspace']
    if len(matches)!=1:raise MultitaskProtocolError('exact declared shared workspace plane required')
    ghost=matches[0];v,f,_=read_native_mesh(ghost);plane=horizontal_rectangle(v,f,ghost.get_matrix())
    entry=service.estimated_bodies['ball'];receipt=entry.collision_geometry_receipt
    if receipt.get('all_rotation_inclusion_proved') is not True:raise MultitaskProtocolError('v9 needs the verified TRAIN all-rotation mesh')
    outer=receipt['maximum_vertex_radius_m'];tracker=actor.tracker;old_pose=tracker._symmetry_pose;fit_events=[];attachment_events=[]
    def symmetry(instance,role,pose,points,previous):
        original=old_pose(role,pose,points,previous)
        if role!='ball':return original
        fitted,row=constrained_fit(points,original,instance.templates[role].sphere_radius,outer,plane,instance.config)
        fit_events.append(dict(row,time=float(instance._current_timestamp),role=role,source='same_current_RGBD_and_shared_static_mesh'))
        return fitted
    tracker._symmetry_pose=MethodType(symmetry,tracker)
    previous_attach=actor._attachments
    def attachments(instance,state,modes,invalid):
        previous_attach(state,modes,invalid)
        body_id='ball'
        if (modes.get(body_id)=='attached' and body_id not in invalid and state.robot.contact and
                _fresh_direct(state,body_id,instance.controller.generator.config.maximum_observation_age)):
            body=state.bodies[body_id];tcp=state.robot.pose
            local=Pose(tcp.rotation.T@(body.pose.position-tcp.position),tcp.rotation.T@body.pose.rotation)
            old=instance.attachments.get(body_id);instance.attachments[body_id]=ObservedAttachment(body_id,local,float(state.robot.timestamp))
            attachment_events.append(dict(time=float(state.robot.timestamp),body=body_id,old_observed_at=None if old is None else old.observed_at,
                local_pose_xyzw=local.xyzw().tolist(),source='same_current_RGBD_robot_proprioception_public_contact',native_dynamic_pose_used=False))
    actor._attachments=MethodType(attachments,actor)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),public_support_attachment_v10=dict(binding=binding(root),shared_static_plane=deepcopy(plane),sphere_fit_events=deepcopy(fit_events),attachment_updates=deepcopy(attachment_events)))
    actor.metrics_report=MethodType(metrics,actor)
