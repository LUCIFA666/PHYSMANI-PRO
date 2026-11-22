"""Ephemeral scene graph for oracle math; native objects expose getters only.

Some original expert label functions move waypoints/frames even with path=False.
Those writes are evaluated here in NumPy, never against the simulator scene.
"""
import numpy as np
from scipy.spatial.transform import Rotation

class Scene:
    def __init__(self,object_type,rotation_from_euler,euler_from_matrix):
        self.object_type=object_type;self.rotation_from_euler=rotation_from_euler
        self.euler_from_matrix=euler_from_matrix;self.nodes={};self.virtual_writes=0

    def wrap(self,value):
        if isinstance(value,self.object_type):
            handle=value.get_handle()
            if handle not in self.nodes:
                node=object.__new__(Node);self.nodes[handle]=node
                node.scene=self;node.native=value;node.initial=np.asarray(value.get_matrix(),float).copy()
                node.parent=self.wrap(value.get_parent())
                parent=np.eye(4)if node.parent is None else node.parent.initial
                node.local=np.linalg.solve(parent,node.initial)
            return self.nodes[handle]
        if isinstance(value,np.ndarray):return value.copy()
        if isinstance(value,list):return[self.wrap(v)for v in value]
        if isinstance(value,tuple):return tuple(self.wrap(v)for v in value)
        if isinstance(value,dict):return{k:self.wrap(v)for k,v in value.items()}
        return value

    def verify(self):
        errors={n.native.get_handle():float(np.max(abs(np.asarray(n.native.get_matrix())-n.initial)))for n in self.nodes.values()}
        if any(e>1e-9 for e in errors.values()):raise RuntimeError('Oracle changed original Native object matrices')
        return dict(Native_objects_checked=len(errors),maximum_matrix_error=max(errors.values(),default=0.),
                    virtual_writes=self.virtual_writes,Native_setter_calls=0)

class Node:
    def world(self):return self.local.copy()if self.parent is None else self.parent.world()@self.local
    def get_handle(self):return self.native.get_handle()
    def get_parent(self):return self.parent
    def get_matrix(self,relative_to=None):
        world=self.world()
        return world if relative_to is None else np.linalg.solve(relative_to.world(),world)
    def get_pose(self,relative_to=None):
        matrix=self.get_matrix(relative_to)
        return np.r_[matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat()]
    def get_position(self,relative_to=None):return self.get_matrix(relative_to)[:3,3].copy()
    def get_orientation(self,relative_to=None):return np.asarray(self.scene.euler_from_matrix(self.get_matrix(relative_to)))
    def set_matrix(self,matrix,relative_to=None):
        world=np.asarray(matrix,float).copy()
        if relative_to is not None:world=relative_to.world()@world
        self.local=world if self.parent is None else np.linalg.solve(self.parent.world(),world)
        self.scene.virtual_writes+=1
    def set_pose(self,pose,relative_to=None,reset_dynamics=True):
        matrix=np.eye(4);matrix[:3,3]=pose[:3];matrix[:3,:3]=Rotation.from_quat(pose[3:7]).as_matrix()
        self.set_matrix(matrix,relative_to)
    def set_position(self,position,relative_to=None,reset_dynamics=True):
        matrix=self.get_matrix(relative_to);matrix[:3,3]=position;self.set_matrix(matrix,relative_to)
    def set_orientation(self,orientation,relative_to=None,reset_dynamics=True):
        matrix=self.get_matrix(relative_to);matrix[:3,:3]=self.scene.rotation_from_euler(orientation)
        self.set_matrix(matrix,relative_to)
    def set_parent(self,parent_object,keep_in_place=True):
        world=self.world();self.parent=parent_object
        if keep_in_place:self.local=world if self.parent is None else np.linalg.solve(self.parent.world(),world)
        self.scene.virtual_writes+=1
    def rotate(self,rotation):
        matrix=self.world();x,y,z=rotation
        matrix[:3,:3]=matrix[:3,:3]@Rotation.from_rotvec([x,0,0]).as_matrix()@Rotation.from_rotvec([0,y,0]).as_matrix()@Rotation.from_rotvec([0,0,z]).as_matrix()
        self.set_matrix(matrix)
    def __getattr__(self,name):
        if name.startswith('get_')or name in ('still_exists','is_detected'):
            return getattr(self.native,name)
        raise AttributeError('Native object mutation unavailable to oracle: '+name)

class TaskProxy:
    def __init__(self,task,scene_factory):
        self._task=task;self._factory=scene_factory;self.stage=task.stage;self.scene=None
    def begin(self):self.scene=self._factory()
    def __getattr__(self,name):
        if name=='get_path':raise RuntimeError('Native path/physics preview forbidden in oracle query')
        return self.scene.wrap(getattr(self._task,name))
