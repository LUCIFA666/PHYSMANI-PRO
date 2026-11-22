"""Read-only robot mesh/FK export and conservative static tabletop clearance.

This is an execution-side static geometry model. No task-body pose, ID mask,
held state or future state is used. It is not a full-scene collision planner.
Finger contact is permitted; distal arm links and the gripper housing must
remain above the solid tabletop footprint plus an explicit planning margin.
"""
import numpy as np
from ..local_mpc_p2_20261103.kinematics import export_native_kinematic_snapshot,PoEKinematics,twist_exp


def tabletop_frame(native_matrix,native_bounds):
    """An object's local axis names do not define the gravity/up direction."""
    native_matrix=np.asarray(native_matrix,float);b=np.asarray(native_bounds,float)
    up=np.array([0.,0.,1.]);rotation=native_matrix[:3,:3]
    horizontal=int(np.argmin(abs(rotation[2])))
    x=rotation[:,horizontal]-up*np.dot(up,rotation[:,horizontal]);x/=np.linalg.norm(x)
    y=np.cross(up,x)
    frame=np.eye(4);frame[:3,:3]=np.stack((x,y,up),axis=1);frame[:3,3]=native_matrix[:3,3]
    corners=np.array([[x,y,z]for x in b[:2]for y in b[2:4]for z in b[4:6]])
    world=corners@rotation.T+native_matrix[:3,3]
    local=(world-frame[:3,3])@frame[:3,:3]
    return frame,np.stack((local.min(0),local.max(0)),axis=1).reshape(6)


class HardwareTableClearance:
    names=tuple(f'Panda_link{i}_respondable'for i in range(4,8))+('Panda_gripper',)

    def __init__(self,backend,*,margin=.05):
        from pyrep.objects.shape import Shape
        before=backend._clock_tuple()
        self.snapshot=export_native_kinematic_snapshot(backend.arm,clock=backend._clock_tuple)
        self.fk=PoEKinematics(self.snapshot);self.margin=float(margin)
        table=Shape('diningTable')
        self.native_table_matrix=np.asarray(table.get_matrix(),float)
        self.native_table_bounds=np.asarray(table.get_bounding_box(),float)
        self.table_matrix,self.bounds=tabletop_frame(self.native_table_matrix,self.native_table_bounds)
        self.table_inverse=np.linalg.inv(self.table_matrix)
        self.bodies=[];joint_ids={j.get_handle():i for i,j in enumerate(backend.arm.joints)}
        for name in self.names:
            shape=Shape(name);parent=shape.get_parent();ancestors=[]
            while parent is not None:
                if parent.get_handle()in joint_ids:ancestors.append(joint_ids[parent.get_handle()])
                parent=parent.get_parent()
            depth=max(ancestors)+1 if ancestors else 0
            vertices=np.asarray(shape.get_mesh_data()[0],float).reshape(-1,3)
            if not depth or not len(vertices):raise ValueError('Robot rigid-link mesh ancestry required')
            self.bodies.append(dict(name=name,depth=depth,vertices=vertices,
                                    reference_matrix=np.asarray(shape.get_matrix(),float)))
        # The housing is closer to intended fingertip contact; its smaller
        # radius/observed model uncertainty permits a smaller planning margin.
        self.body_margins=np.array([.02 if b['name']=='Panda_gripper'else self.margin for b in self.bodies])
        if backend._clock_tuple()!=before:raise RuntimeError('Geometry getters advanced physics')

    def refresh(self,backend):
        """Re-anchor the rigid model to current public robot poses, without writes.

        Dynamic joint constraints have measurable small link-pose residuals;
        a snapshot from home is not a permanent exact model under load.
        Static mesh vertices and the fixed table remain unchanged.
        """
        from pyrep.objects.shape import Shape
        before=backend._clock_tuple()
        self.snapshot=export_native_kinematic_snapshot(backend.arm,clock=backend._clock_tuple)
        self.fk=PoEKinematics(self.snapshot)
        for body in self.bodies:body['reference_matrix']=np.asarray(Shape(body['name']).get_matrix(),float)
        if backend._clock_tuple()!=before:raise RuntimeError('Robot geometry refresh advanced physics')

    def calibration(self):
        return dict(snapshot=self.snapshot,body_reference_matrices={b['name']:b['reference_matrix'].tolist()for b in self.bodies})

    def _prefixes(self,q):
        q=np.asarray(q,float).reshape(-1,7)
        if not np.isfinite(q).all():raise ValueError('Finite robot joint samples required')
        value=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();out=[value]
        for i in range(7):
            value=value@twist_exp(self.fk.screws[i],q[:,i]-self.fk.q_ref[i]);out.append(value)
        return out

    def depths(self,q):
        prefixes=self._prefixes(q);results=[]
        for body in self.bodies:
            transform=self.table_inverse@prefixes[body['depth']]@body['reference_matrix']
            points=np.einsum('nij,vj->nvi',transform[:,:3,:3],body['vertices'])+transform[:,:3,3,None].transpose(0,2,1)
            low=points.min(axis=1);high=points.max(axis=1);b=self.bounds
            overlap=(high[:,0]>=b[0])&(low[:,0]<=b[1])&(high[:,1]>=b[2])&(low[:,1]<=b[3])
            margin=getattr(self,'body_margins',np.full(len(self.bodies),self.margin))[len(results)]
            results.append(np.where(overlap,np.maximum(b[5]+margin-low[:,2],0),0))
        return np.stack(results,axis=1)

    def safe(self,q):return np.all(self.depths(q)<=1e-7,axis=1)

    def native_pose_errors(self,backend):
        return {name:value['matrix_max_abs']for name,value in self.native_pose_residuals(backend).items()}

    def native_pose_residuals(self,backend):
        from pyrep.objects.shape import Shape
        from scipy.spatial.transform import Rotation
        before=backend._clock_tuple();prefixes=self._prefixes(backend.get_joint_positions())
        errors={}
        for body in self.bodies:
            predicted=prefixes[body['depth']][0]@body['reference_matrix']
            actual=np.asarray(Shape(body['name']).get_matrix(),float)
            errors[body['name']]=dict(matrix_max_abs=float(np.max(abs(predicted-actual))),
                position_m=float(np.linalg.norm(predicted[:3,3]-actual[:3,3])),
                rotation_rad=float((Rotation.from_matrix(predicted[:3,:3]).inv()*Rotation.from_matrix(actual[:3,:3])).magnitude()))
        if backend._clock_tuple()!=before:raise RuntimeError('Hardware FK validation advanced physics')
        return errors

    def export(self):
        return dict(snapshot=self.snapshot,table_matrix=self.table_matrix.tolist(),table_local_bounds=self.bounds.tolist(),
            native_table_matrix=self.native_table_matrix.tolist(),native_table_bounds=self.native_table_bounds.tolist(),
            bodies=[{k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in b.items()}for b in self.bodies],
            planning_margin_m=self.margin,body_planning_margins_m=self.body_margins.tolist(),
            tabletop_up_axis='world_positive_z_not_native_object_local_z',
            geometry_scope='distal_robot_hardware_vs_solid_tabletop_footprint',
            finger_contact_permitted=True,full_scene_collision_certificate=False,Native_task_body_poses_used=False)
