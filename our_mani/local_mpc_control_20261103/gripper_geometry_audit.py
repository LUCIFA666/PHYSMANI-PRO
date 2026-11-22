"""Read-only Panda hardware geometry export; this module never starts Native.

The caller owns home staging and the single Native session. Only robot
objects, their hierarchy/meshes, and physics/counting clocks are read. Task
objects, grasp-membership/held, state snapshots, contacts and success are not
queried. Unknown-name descendants are excluded before pose/mesh reads.
"""
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


class GeometryClockMutation(RuntimeError):
    pass


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Enum):return dict(name=value.name,value=value.value)
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


def _matrix(value):
    result=np.asarray(value,dtype=float)
    if (result.shape!=(4,4) or not np.isfinite(result).all()
            or not np.allclose(result[3],[0,0,0,1],atol=1e-8,rtol=0)
            or not np.allclose(result[:3,:3].T@result[:3,:3],np.eye(3),atol=1e-4,rtol=0)):
        raise ValueError('finite rigid world matrix[4,4] required')
    return result.copy()


def audit_gripper_geometry(backend,output,*,hardware_name_prefixes=('Panda',),require_zero_control_ticks=True):
    """Write robot_gripper_geometry.json using only existing robot getters.

    No PyRep import/session is created here. ``backend`` supplies arm/gripper,
    clock(), _physics_clock(), and (for an independent count) raw.ledger.
    Panda is a declared hardware-name allowlist, not a task-object lookup.
    Excluded descendants make coverage explicitly incomplete; their geometry
    and identity are not written into the hardware calibration file.
    """
    if (not hardware_name_prefixes or any(not isinstance(p,str) or not p for p in hardware_name_prefixes)
            or type(require_zero_control_ticks) is not bool):
        raise ValueError('explicit nonempty hardware prefixes and zero-tick policy required')
    directory=Path(output)
    if (directory/'robot_gripper_geometry.json').exists():
        raise FileExistsError(directory/'robot_gripper_geometry.json')
    if not callable(getattr(backend,'clock',None)) or not callable(getattr(backend,'_physics_clock',None)):
        raise ValueError('independent control and physical clock readers required')
    counts={};source_files={};excluded=0

    def clocks():
        ledger=getattr(getattr(backend,'raw',None),'ledger',None)
        result=dict(control_time=float(backend.clock()),physics_time=float(backend._physics_clock()),
            ledger_ticks=None if ledger is None else int(ledger.ticks))
        if not np.isfinite([result['control_time'],result['physics_time']]).all():
            raise ValueError('finite audit clocks required')
        return result

    def call(obj,name,*args,**kwargs):
        before=clocks();method=getattr(obj,name)
        counts[name]=counts.get(name,0)+1
        try:
            path=inspect.getsourcefile(method)
            if path is not None:
                path=Path(path).resolve()
                if str(path) not in source_files:source_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            return method(*args,**kwargs)
        finally:
            if clocks()!=before:
                raise GeometryClockMutation('robot getter advanced a physics/control clock: '+name)

    def allowed(name):return any(name.startswith(prefix) for prefix in hardware_name_prefixes)

    def identity(obj):
        name=str(call(obj,'get_name'))
        if not allowed(name):return None
        return dict(name=name,handle=int(call(obj,'get_handle')),object_type=_plain(call(obj,'get_type')))

    def parent_chain(obj):
        chain=[];seen=set();parent=call(obj,'get_parent');outside=False
        while parent is not None:
            meta=identity(parent)
            if meta is None:outside=True;break
            if meta['handle'] in seen:raise ValueError('cycle in robot parent hierarchy')
            seen.add(meta['handle']);chain.append(meta);parent=call(parent,'get_parent')
        return dict(hardware_ancestors=chain,stopped_at_nonhardware_parent=outside)

    before=clocks()
    if require_zero_control_ticks and (abs(before['control_time'])>1e-9 or before['ledger_ticks'] not in (None,0)):
        raise ValueError('hardware calibration must precede all controlled physics ticks')
    arm,gripper=backend.arm,backend.gripper
    tip=call(arm,'get_tip');tip_meta=identity(tip);gripper_meta=identity(gripper)
    if tip_meta is None or gripper_meta is None:raise ValueError('declared Panda robot tip/gripper required')
    tcp_world=_matrix(call(tip,'get_matrix'))
    tcp_inverse=np.linalg.inv(tcp_world)
    gripper_world=_matrix(call(gripper,'get_matrix'))
    joints=list(gripper.joints)
    if len(joints)!=2:raise ValueError('this Panda audit requires exactly two finger joints')
    opening=np.asarray(call(gripper,'get_open_amount'),float)
    if opening.shape!=(2,) or not np.isfinite(opening).all() or np.any((opening<0)|(opening>1)):
        raise ValueError('two finite normalized measured finger openings required')
    joint_records=[];objects={};memberships={}
    roots=[('gripper',gripper)]+[('finger_joint_'+str(i+1),joint) for i,joint in enumerate(joints)]
    for label,root in roots:
        for obj in call(root,'get_objects_in_tree',exclude_base=False,first_generation_only=False):
            meta=identity(obj)
            if meta is None:excluded+=1;continue
            objects[meta['handle']]=(obj,meta)
            memberships.setdefault(meta['handle'],[]).append(label)
    for i,joint in enumerate(joints):
        meta=identity(joint)
        if meta is None:raise ValueError('finger joint is outside declared hardware allowlist')
        world=_matrix(call(joint,'get_matrix'))
        cyclic,interval=call(joint,'get_joint_interval')
        interval=np.asarray(interval,float)
        if interval.shape!=(2,) or not np.isfinite(interval).all():raise ValueError('finite native joint minimum/range required')
        q=float(call(joint,'get_joint_position'))
        if not np.isfinite(q):raise ValueError('finite measured finger coordinate required')
        kind=call(joint,'get_joint_type')
        joint_records.append(dict(**meta,index=i,joint_type=_plain(kind),cyclic=bool(cyclic),
            interval_minimum_and_range=interval,minimum=interval[0],range=interval[1],maximum=interval.sum(),
            measured_coordinate=q,normalized_open_amount=opening[i],coordinate_units='m_if_PRISMATIC_rad_if_REVOLUTE',
            pyrep_normalization_denominator_interval1_minus_interval0=interval[1]-interval[0],
            matrix_world=world,matrix_in_TCP=tcp_inverse@world,
            positive_intrinsic_Z_axis_world=world[:3,2],
            positive_intrinsic_Z_axis_TCP=tcp_world[:3,:3].T@world[:3,2],
            axis_scope='Coppelia joint intrinsic +Z; verify PRISMATIC type before treating as linear opening axis',
            parent_chain=parent_chain(joint)))
    shapes=[];other_nodes=[]
    for handle,(obj,meta) in sorted(objects.items()):
        kind=meta['object_type']
        is_shape=isinstance(kind,dict) and kind.get('name')=='SHAPE' or kind==0
        if not is_shape:
            other_nodes.append(dict(**meta,membership_roots=memberships[handle],parent_chain=parent_chain(obj)))
            continue
        world=_matrix(call(obj,'get_matrix'));bbox=np.asarray(call(obj,'get_bounding_box'),float)
        if bbox.shape!=(6,) or not np.isfinite(bbox).all():raise ValueError('finite local shape bbox required')
        record=dict(**meta,membership_roots=memberships[handle],matrix_world=world,matrix_in_TCP=tcp_inverse@world,
            local_bbox_xmin_xmax_ymin_ymax_zmin_zmax=bbox,parent_chain=parent_chain(obj))
        try:
            vertices,indices,normals=call(obj,'get_mesh_data')
            vertices=np.asarray(vertices,float);indices=np.asarray(indices,dtype=np.int64);normals=np.asarray(normals,float)
            if (vertices.ndim!=2 or vertices.shape[1]!=3 or indices.ndim!=2 or indices.shape[1]!=3
                    or normals.ndim!=2 or normals.shape[1]!=3
                    or not np.isfinite(vertices).all() or not np.isfinite(normals).all()
                    or (indices.size and (indices.min()<0 or indices.max()>=len(vertices)))):
                raise ValueError('invalid readable mesh vertices/triangle indices')
            record.update(mesh_readable=True,vertices_local=vertices,triangle_indices=indices,normals_local=normals,
                vertices_world=vertices@world[:3,:3].T+world[:3,3],
                vertices_in_TCP=vertices@(tcp_inverse@world)[:3,:3].T+(tcp_inverse@world)[:3,3])
        except GeometryClockMutation:raise
        except (RuntimeError,ValueError,TypeError,AttributeError) as error:
            record.update(mesh_readable=False,mesh_error=type(error).__name__+': '+str(error))
        shapes.append(record)
    after=clocks()
    if after!=before:raise GeometryClockMutation('hardware audit changed physical/control time')
    staging=getattr(backend,'staging_receipt',{})
    staging_keys=('excluded_from_control_budget','clock_before','clock_after','raw_physics_seconds',
        'controlled_ticks','reset_home_joints','disable_dynamics_used_only_during_staging','extra_raw_settle_steps')
    report=dict(schema='Panda_readonly_gripper_geometry_audit_20261103_v1',
        tcp=dict(**tip_meta,matrix_world=tcp_world),gripper=dict(**gripper_meta,matrix_world=gripper_world,matrix_in_TCP=tcp_inverse@gripper_world),
        finger_joints=joint_records,shapes=shapes,other_hardware_nodes=other_nodes,
        queried_robot_only=True,Object_GT_read=False,held_or_grasp_membership_read=False,task_success_read=False,
        setters_called=False,step_called=False,dynamics_changed=False,clocks_before=before,clocks_after=after,
        controlled_physics_ticks_added=0,home_staging_accounting_separate={k:staging[k] for k in staging_keys if k in staging},
        hardware_name_prefixes=list(hardware_name_prefixes),excluded_nonhardware_descendant_occurrences=excluded,
        tree_coverage_complete_under_declared_hardware_allowlist=excluded==0,
        all_included_meshes_readable=all(r['mesh_readable'] for r in shapes),
        inner_pad_planes_calibrated=False,geometry_export_is_not_contact_or_grasp_proof=True,
        getter_call_counts=counts,getter_source_sha256=source_files)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'robot_gripper_geometry.json').open('x',encoding='utf-8') as stream:
        json.dump(_plain(report),stream,indent=2,allow_nan=False);stream.write('\n')
    return _plain(report)
