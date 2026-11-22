"""Fixed robot-asset layouts for independent static V9 calibration images.

Only the robot-only precontrol mesh audit, preassigned TRAIN q0, initial public
Sphere and frozen PoE chain are used. Static placement is never an attachment
or grasp. This module imports no simulator and starts no renderer or physics.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np
from . import depth_owner_full32 as fixed

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'artifacts/framework/local_mpc_20261102/P3_finger_contact_geometry_v1/robot_geometry_calibration.bytecopy.json'
ASSET_SHA256='9bf49962896a9824234e26424d4e83bc4f1e01e55e38dad37d98e8fc0bcbfb55'
SHOT_IDS=tuple('shot%02d'%i for i in range(10))
KINEMATIC_SCOPE='static_independent_scene_not_attached_not_grasp_not_execution_transition'
PALM_CLEARANCE_M=.002  # Existing frozen DLS position tolerance; fixed before capture.


def robot_asset_geometry():
    if hashlib.sha256(ASSET.read_bytes()).hexdigest()!=ASSET_SHA256:raise ValueError('Exact robot-only mesh asset audit required')
    data=json.loads(ASSET.read_text())
    if (data.get('queried_robot_only')is not True or data.get('Object_GT_read')is not False
            or data.get('held_or_grasp_membership_read')is not False or data.get('task_success_read')is not False
            or data['clocks_before']!=data['clocks_after']or data['controlled_physics_ticks_added']!=0):
        raise ValueError('Only closed zero-control-tick robot geometry, never task observations')
    joints=data['finger_joints'];axes=np.asarray([j['positive_intrinsic_Z_axis_TCP']for j in joints],float)
    if len(joints)!=2 or any(j['joint_type']['name']!='PRISMATIC'or j['cyclic']for j in joints):raise ValueError('Exact two Panda prismatic fingers required')
    axes/=np.linalg.norm(axes,axis=1)[:,None];axis=axes[1]-axes[0];axis/=np.linalg.norm(axis)
    if axes[0]@axes[1]>-.99999 or axis@np.array([0.,1.,0.])<.99999:raise ValueError('Known antiparallel Panda finger axes required')
    shapes={s['name']:s for s in data['shapes']};faces=[]
    for i,name in enumerate(('Panda_leftfinger_force_contact','Panda_rightfinger_force_contact')):
        shape=shapes[name]
        if shape.get('mesh_readable')is not True:raise ValueError('Readable exact contact-pad mesh required')
        vertices=np.asarray(shape['vertices_in_TCP'],float);choices=[]
        for tri in shape['triangle_indices']:
            points=vertices[tri];normal=np.cross(points[1]-points[0],points[2]-points[0]);size=np.linalg.norm(normal)
            if size<1e-14:continue
            normal/=size
            if abs(normal@axis)>.999:choices.append((float(points.mean(0)@axis),points,normal))
        if not choices:raise ValueError('Current asset has no closing-axis pad face')
        at,points,normal=max(choices,key=lambda x:x[0])if i==0 else min(choices,key=lambda x:x[0])
        polygon=vertices[np.abs((vertices-points.mean(0))@normal)<1e-7]
        qcal=float(joints[i]['measured_coordinate']);slope=float(axis@axes[i])
        faces.append(dict(shape=name,centre_TCP_at_audit=polygon.mean(0).tolist(),
            plane_intercept_m=at-slope*qcal,slope=slope,q_at_audit_m=qcal,
            prismatic_axis_TCP=axes[i].tolist(),polygon_TCP_at_audit=polygon.tolist()))
    palm=shapes['Panda_gripper_visual'];pv=np.asarray(palm['vertices_in_TCP'],float)
    if palm.get('mesh_readable')is not True or not np.isfinite(pv).all():raise ValueError('Actual palm visual mesh required')
    limits=np.asarray([[j['minimum'],j['maximum']]for j in joints],float)
    return dict(schema='V9_robot_asset_pad_layout_v1',source=dict(path=str(ASSET.relative_to(ROOT)),sha256=ASSET_SHA256),
        closing_axis_TCP=axis.tolist(),faces=faces,joint_intervals_m=limits.tolist(),
        palm_shape='Panda_gripper_visual',palm_vertices_TCP=pv.tolist(),
        palm_front_coordinate_TCP_z_m=float(pv[:,2].max()),palm_bounds_TCP=[pv.min(0).tolist(),pv.max(0).tolist()],
        gap_scope='inward force_contact_patch planes; not respondable carrier or visual silhouette',
        calibration_robot_q_used_only_for_geometric_intercept_removal=True,task_pose_or_VAL_q_or_r0_used=False)


def finger_layout(geometry,kind):
    if kind not in ('full_open_asset_max','gap060','gap050'):raise ValueError('One fixed asset gap required')
    limits=np.asarray(geometry['joint_intervals_m']);left,right=geometry['faces']
    intercept=right['plane_intercept_m']-left['plane_intercept_m'];slope=right['slope']-left['slope']
    requested={'full_open_asset_max':.08,'gap060':.06,'gap050':.05}[kind]
    if kind=='full_open_asset_max':q=limits[:,1].copy()
    else:q=np.full(2,(requested-intercept)/slope)
    if not np.isfinite(q).all()or np.any(q<limits[:,0])or np.any(q>limits[:,1]):raise ValueError('Fixed finger target outside real asset limits; no clipping')
    planes=[f['plane_intercept_m']+f['slope']*v for f,v in zip((left,right),q)]
    centres=[np.asarray(f['centre_TCP_at_audit'])+np.asarray(f['prismatic_axis_TCP'])*(v-f['q_at_audit_m'])for f,v in zip((left,right),q)]
    return dict(kind=kind,requested_gap_m=requested,actual_force_pad_gap_m=float(planes[1]-planes[0]),
        finger_positions_m=q.tolist(),pad_midpoint_TCP_m=np.mean(centres,axis=0).tolist(),
        requested80_is_explicit_asset_maximum_exception=kind=='full_open_asset_max',joint_targets_clipped=False,
        gap_scope=geometry['gap_scope'],not_contact_or_grasp_proof=True)


def shot_specs_for_parent(ordinal):
    if type(ordinal)is not int or not 0<=ordinal<8:raise ValueError('Fixed eight octants required')
    rows=[('sphere_only','parked','negative_outside_0_12'),('cube_far','nearhand','negative_outside_0_12'),
        ('near_goal','parked','near_0_08'),('near_goal','static_inhand_full_open_asset_max','near_0_08'),
        ('near_goal','static_inhand_gap060','near_0_08'),('near_goal','static_inhand_gap050','near_0_08'),
        ('near_goal','palm_Z_occlusion_gap060','near_0_08'),('near_goal','annulus090_open','annulus_08_10'),
        ('near_goal','annulus110_nearhand','annulus_10_12'),('sphere_only','same_palm_occlusion_nearhand','negative_outside_0_12')]
    return [dict(shot_id=SHOT_IDS[i],condition=c,condition_detail=detail+f'_dir{ordinal:02d}_{fixed.DIRECTION_NAMES[ordinal]}',
        position_band=band,near_direction_ordinal=ordinal,near_direction_unit_world=fixed.DIRECTIONS[ordinal].tolist(),
        independent_scene=True,static_layout_not_attached_or_grasp=True)for i,(c,detail,band)in enumerate(rows)]


def materialize_v9_shots(plan,parent,sphere,native_cube_pose,bbox,cameras,kinematics,joint_limits):
    """One fixed CPU materialization before any of the10 augmented images.

    Native Cube pose is deliberately unused. Missing/failed poses are retained
    and abort the complete parent; no source/target/seed replacement is made.
    """
    if PALM_CLEARANCE_M!=fixed.DLS['position_tolerance']:raise ValueError('Declared palm clearance reference changed')
    ordinal=parent.get('near_direction_ordinal')
    if (type(ordinal)is not int or not 0<=ordinal<8
            or parent.get('shots')!=shot_specs_for_parent(ordinal)
            or not np.array_equal(parent.get('near_direction_unit_world'),fixed.DIRECTIONS[ordinal])):
        raise ValueError('Exact preassigned octant and all ten recipes required')
    if plan.get('DLS')!=fixed.DLS or plan.get('tool_quaternion_xyzw')!=[1.,0.,0.,0.]:
        raise ValueError('Original fixed solver and declared down-facing tool convention required')
    geom=robot_asset_geometry();layouts={k:finger_layout(geom,k)for k in ('full_open_asset_max','gap060','gap050')}
    S=fixed.b.vector(sphere['center'],3,'initial public Sphere');direction=np.asarray(parent['near_direction_unit_world'],float)
    C=S+.06*direction;seed=fixed.b.vector(parent['TRAIN_q_source']['joint_positions'],7,'preassigned TRAIN q0')
    lower=np.asarray(joint_limits['lower']);upper=np.asarray(joint_limits['upper'])
    if np.any(seed<lower)or np.any(seed>upper):raise ValueError('TRAIN q0 cannot be clipped')
    R=np.diag([1.,-1.,-1.]);pad=np.asarray(layouts['gap060']['pad_midpoint_TCP_m'])
    palm=pad.copy();palm[2]=geom['palm_front_coordinate_TCP_z_m']+.025+PALM_CLEARANCE_M
    targets=np.stack([C-R@pad,C-R@palm]);quat=np.tile([1.,0.,0.,0.],(2,1))
    adapter=fixed.BatchPoseJointAdapter(kinematics,type('Limits',(),dict(lower=lower,upper=upper))(),fixed.DLSConfig(**fixed.DLS))
    solved=fixed.solve_fixed_starts(adapter,seed,targets,quat);failures=[]
    if not np.all(solved['joint_goal_feasible']):failures.append('fixed9start_asset_hand_DLS_infeasible_no_retry')
    qhand,qpalm=solved['joint_goal'];Thand=kinematics(qhand);Tpalm=kinematics(qpalm)
    outside=np.asarray(plan['world_outside_position_m'],float)
    proof=fixed.outside_frusta_proof(outside,[0,0,0,1],bbox,cameras)
    if not proof['verified']:failures.append('fixed_outside_cube_not_outside_all_camera_frusta')
    def world(T,p):return T[:3,:3]@np.asarray(p)+T[:3,3]
    poses=[outside,S+.20*direction,C,world(Thand,layouts['full_open_asset_max']['pad_midpoint_TCP_m']),
        world(Thand,pad),world(Thand,layouts['gap050']['pad_midpoint_TCP_m']),world(Tpalm,palm),
        S+.09*direction,S+.11*direction,outside]
    qs=[seed,qhand,seed,qhand,qhand,qhand,qpalm,seed,qhand,qpalm]
    fs=['full_open_asset_max','gap060','full_open_asset_max','full_open_asset_max','gap060','gap050','gap060','full_open_asset_max','gap060','gap060']
    shots=[]
    for i,(spec,p,q,f)in enumerate(zip(parent['shots'],poses,qs,fs)):
        distance=float(np.linalg.norm(p-S));s=deepcopy(spec)
        if s['position_band']=='near_0_08'and not distance<.08:failures.append(s['shot_id']+'_near_band_invalid')
        if i==7 and abs(distance-.09)>1e-12:failures.append('annulus090_not_fixed_public_Sphere_distance')
        if i==8 and abs(distance-.11)>1e-12:failures.append('annulus110_not_fixed_public_Sphere_distance')
        if i==1 and not distance>.12:failures.append('cube_far_not_outside012')
        s.update(cube_mode='public_sphere_offset',offset_world_m=(p-S).tolist(),quaternion_xyzw=[0.,0.,0.,1.],
            materialized_cube_pose_world=np.r_[p,[0.,0.,0.,1.]].tolist(),joint_positions=q.tolist(),
            finger_positions_m=layouts[f]['finger_positions_m'],finger_layout=layouts[f],
            cube_scene_present=True,absence_from_all_camera_frusta_by_construction=i in (0,9),
            distance_from_initial_public_sphere_m=distance,kinematic_layout_scope=KINEMATIC_SCOPE,
            pose_source='fixed robot-only asset+frozen PoE+initial public Sphere; never VAL q/r0 or Native task labels',
            annulus_is_not_claimed_static_inhand=i in (7,8),Native_robot_pose_readback_still_required=True)
        if np.any(q<lower)or np.any(q>upper):failures.append(s['shot_id']+'_joint_target_bounds_no_clip')
        shots.append(s)
    return dict(complete=not failures,failures=failures,parent_id=parent['parent_id'],seed=parent['seed'],
        public_Sphere_center=S.tolist(),shots=shots,robot_asset_geometry=geom,finger_layouts=layouts,
        outside_frusta_proof=proof,derived_before_any_shot=True,no_adaptive_retry=True,not_a_continuous_trajectory=True,
        DLS=dict(config=fixed.DLS,target_positions=targets.tolist(),target_quaternions=quat.tolist(),
            results=fixed.b.plain(solved),Native_IK_called=False,physical_steps=0,collision_or_reachability_qualified=False),
        static_layout_not_grasp=True,robot_geometry_ideal_PoE_frame_not_Native_pose_accuracy_certificate=True,
        bootstrap_initial_public_Sphere_reused_without_refit=True)
