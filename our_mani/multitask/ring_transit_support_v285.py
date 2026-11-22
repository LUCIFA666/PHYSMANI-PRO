"""Fully checked3D ring transit and support-consistent public held geometry.

External tracker observations/histories and current D4 qualification stay
unchanged. Only a qualified current RGBD/contact measurement can establish a
support-constrained tool-to-ring transform; original fit quality is rechecked.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import json
import numpy as np
from scipy.spatial import cKDTree
from ..contracts import Pose
from .bounded_transit_v180 import BoundedApproach,SCHEMA as TRANSIT_SCHEMA
from .public_held_mesh_v33 import supported_pose
from .supported_sphere_attachment_v10 import horizontal_rectangle
from .safe_native_mesh import read_native_mesh
from .forecasting import ObservedAttachment
from .perception import _sample
from .pipeline import _fresh_direct
from .rotation_equivalence import CurrentRotationEquivalence
from .depth_pose_consistency import current_depth_pose_cost,FreeSpaceConfig
from .evaluation import file_sha

TASK='insert_onto_rotating_peg_high_speed'


def binding(root,protocol,task):
    if task!=TASK:raise ValueError('explicit full100 highpeg capability required')
    geometry=protocol['learned_artifacts'][task]['geometry'];path=Path(root)/geometry['path']
    if file_sha(path)!=geometry['sha256']:raise ValueError('actual TRAIN ring CAD differs')
    if task not in protocol.get('high_peg_ring_runtime_v243',{}):raise ValueError('verified current finite ring capability required')
    if protocol['mpc_config']['maximum_linear_speed']!=.8 or protocol['mpc_config']['maximum_angular_speed']!=2.:
        raise ValueError('original additional Cartesian prefix caps required')
    return dict(schema='r10_highpeg_fully_checked3D_prefix_and_public_support_attachment_v285',task=task,
        sources_sha256={f'our_mani/multitask/{name}':file_sha(Path(__file__).with_name(name)) for name in
            ('ring_transit_support_v285.py','bounded_transit_v180.py','public_held_mesh_v33.py','depth_pose_consistency.py')},
        TRAIN_geometry=geometry,full100_D4_permission=protocol['ring_affordances'][task],
        original_step_heuristic_m=.08,actual_maximum_proposal_step_m=.2,segment_seconds=.5,
        original_BoundedApproach180_actual_reference_cartesian_collision_offset_brake_gates_retained=True,
        held_support='unique public held ring; fresh current qualified D4; same current RGBD clouds; finite static workspace; original point fit and enabled depth cost rechecked',
        external_tracker_result_and_history_unchanged=True,global_full_orientation_not_promoted=True,
        collision_meshes_permissions_grasp_physics_success_and10s_unchanged=True,
        calibrated_pose_accuracy_guarantee=False,formal_VAL_TEST=False)


def admissible(actor,state,modes,invalid,clouds):
    now=float(state.robot.timestamp);role='ring'
    attached={key for key,value in modes.items() if value=='attached'}
    if (not state.robot.contact or attached!={role} or role in invalid or not clouds
            or abs(clouds['timestamp']-now)>1e-8
            or not _fresh_direct(state,role,actor.controller.generator.config.maximum_observation_age)):return False
    info=state.perception_metadata.get('bodies',{}).get(role,{})
    if info.get('mode_provenance')!='caller_controller_touch_action_proprioception':return False
    evidence=info.get('rotation_equivalence_evidence')
    if not isinstance(evidence,CurrentRotationEquivalence):return False
    body=state.bodies[role];capability=actor.collision_port.ring_affordance
    return capability.permits_geometry(evidence,body_id=role,observed_pose=body.pose,
        observed_timestamp=body.timestamp,now=now,maximum_age=actor.controller.generator.config.maximum_observation_age)


def combined_quality(pose,clouds,template,config,views):
    errors=[]
    for component in template.components:
        points=np.asarray(clouds[component.name],float)
        if len(points)<3:continue
        distances=cKDTree(_sample(component.points_body,config.maximum_points_per_component)).query(pose.inverse_apply(points))[0]
        errors.append(float(np.mean(np.minimum(distances,config.maximum_correspondence_m)**2)))
    cost=float(np.mean(errors)) if errors else float('inf')
    depth=None
    if config.free_space_weight>0:
        depth=current_depth_pose_cost({c.name:c.points_body for c in template.components},pose,views,
            config=FreeSpaceConfig(tolerance_m=config.free_space_tolerance_m,
                maximum_gap_m=config.free_space_maximum_gap_m,weight=config.free_space_weight))
        cost+=depth['cost_m2']
    return dict(combined_current_cost_RMS_m=float(np.sqrt(cost)),current_depth_cost=depth,
        valid=bool(np.sqrt(cost)<=config.maximum_registration_rms_m))


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('ring_transit_support_v285',{}).get(task)!=expected:raise ValueError('freeze actual ring transit/support policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first control step')
    raw=deepcopy(m.approach_transit.normalized)
    if raw['segment_seconds']!=.5 or raw['maximum_TCP_step_m']!=.08:raise ValueError('expected original highpeg prefix config')
    raw.update(version=TRANSIT_SCHEMA,maximum_TCP_step_m=.2)
    m.approach_transit=BoundedApproach(m,raw)
    service=actor.map;matches=[shape for info,shape in zip(service.map,service.static_ghosts) if info['name']=='workspace']
    if len(matches)!=1:raise ValueError('one original finite static workspace required')
    vertices,triangles,_=read_native_mesh(matches[0]);plane=horizontal_rectangle(vertices,triangles,matches[0].get_matrix())
    asset=json.loads((Path(root)/expected['TRAIN_geometry']['path']).read_text())
    ring_vertices=np.asarray(asset['bodies']['ring']['collision_vertices_body'],float)
    tracker=actor.tracker;original_registration=tracker._registration;current={};active={};events=[]
    def registration(instance,members,clouds,previous=None):
        result=original_registration(members,clouds,previous)
        if 'ring' in members:
            current.clear()
            if result is not None and result['valid']:
                current.update(timestamp=float(instance._current_timestamp),
                    clouds={key:np.asarray(value).copy() for key,value in clouds['ring'].items()})
        return result
    tracker._registration=MethodType(registration,tracker)
    previous_attachments=actor._attachments
    def attachments(instance,state,modes,invalid):
        previous_attachments(state,modes,invalid)
        if not state.robot.contact or modes.get('ring')!='attached' or 'ring' in invalid or {k for k,v in modes.items() if v=='attached'}!={'ring'}:
            active.clear();return
        if admissible(instance,state,modes,invalid,current):
            pose,row=supported_pose(state.bodies['ring'].pose,current['clouds'],tracker.templates['ring'],ring_vertices,plane,tracker.config)
            quality=combined_quality(pose,current['clouds'],tracker.templates['ring'],tracker.config,tracker._current_views)
            if row['accepted'] and quality['valid']:
                tcp=state.robot.pose;local=Pose(tcp.rotation.T@(pose.position-tcp.position),tcp.rotation.T@pose.rotation)
                item=ObservedAttachment('ring',local,float(state.robot.timestamp));instance.attachments['ring']=item;active.update(attachment=item)
                events.append(dict(time=float(state.robot.timestamp),event='current_D4_RGBD_support_attachment_admitted',
                    support=row,combined_quality=quality,local_xyzw=local.xyzw().tolist(),full_orientation_promoted=False))
            else:events.append(dict(time=float(state.robot.timestamp),event='support_attachment_original_quality_rejected',support=row,combined_quality=quality))
        if active and 'ring' in instance.attachments:instance.attachments['ring']=active['attachment']
    actor._attachments=MethodType(attachments,actor)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),ring_transit_support_v285=dict(binding=expected,
        actual_prefix_configuration=deepcopy(m.approach_transit.normalized),shared_static_plane=plane,events=deepcopy(events),
        raw_current_tracker_result_and_history_not_replaced=True,original_collision_permissions_not_changed=True))
    actor.metrics_report=MethodType(metrics,actor)
