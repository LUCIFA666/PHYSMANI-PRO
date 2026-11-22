"""Map-source ablation using initial RGB-D and the shared current robot masks.

Only occupied visible voxels are known; unobserved space is not reconstructed.
Native fixture handles below are used solely to disable true collision geometry
inside the sandboxed query, not to construct or return the perceived map.
"""
import hashlib
import numpy as np
from ..benchmark_perception import RedBoxTracker
from .simulation import (SharedKnownMap,AlignedRobotModel,
                         AlignedOfficialSimulation,AlignedLocalSimulation)


class ObservedVoxelMap(SharedKnownMap):
    def __init__(self,env,observation,previous,voxel_size=.025):
        from pyrep.const import ObjectType
        from pyrep.objects.octree import Octree
        self.env,self.target=env,previous.target
        self.static=previous.static
        self.queries,self.path_queries=0,0
        self.voxel_size=voxel_size
        robot_shapes=env.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)+env.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
        robot_ids={x.get_handle() for x in robot_shapes}|{env.arm.get_handle(),env.gripper.get_handle()}
        try:target,_=RedBoxTracker().observe(observation)
        except ValueError:target=None
        clouds=[]
        for name,depth in observation['observation/depths'].items():
            calibration=observation['observation/calibrations'][name]
            cloud=calibration.backproject(depth,np.ones(depth.shape,dtype=bool)).reshape(*depth.shape,3)
            encoded=np.rint(observation['observation/instance_masks'][name]*255).astype(np.int64)
            identities=encoded[...,0]+256*encoded[...,1]+65536*encoded[...,2]
            keep=~np.isin(identities,list(robot_ids))
            keep &= ((cloud>=[-.7,-.85,.1])&(cloud<=[.85,.85,1.7])).all(-1)
            if target is not None:keep &= (np.abs(cloud-target.pose.position)>.045).any(-1)
            clouds.append(cloud[keep])
        points=np.concatenate(clouds)
        if len(points)<10:raise ValueError('RGB-D obstacle map has insufficient visible points')
        cells=np.unique(np.floor(points/voxel_size).astype(np.int32),axis=0)
        centers=(cells+.5)*voxel_size
        self.sha256=hashlib.sha256(cells.tobytes()).hexdigest()
        self.map=[dict(name='observed_rgbd_octree',pose=[0.,0.,0.,0.,0.,0.,1.],
                       bounding_box=[float(centers[:,0].min()),float(centers[:,0].max()),float(centers[:,1].min()),float(centers[:,1].max()),float(centers[:,2].min()),float(centers[:,2].max())],mesh_sha256=self.sha256)]
        previous.close()
        octree=Octree.create(voxel_size)
        octree.insert_voxels(centers.reshape(-1).tolist())
        octree.set_renderable(False);octree.set_collidable(False)
        self.ghosts=[octree]
        self.voxels=len(cells)

    def receipt(self):
        return dict(access='initial_RGBD_occupied_voxels_and_shared_current_robot_instance_masks',
                    map_sha256=self.sha256,voxel_size_m=self.voxel_size,occupied_voxels=self.voxels,
                    queries=self.queries,path_queries=self.path_queries,map_is_episode_start_snapshot=True,
                    dynamic_ground_truth_returned=False,queries_advance_physics=False,
                    unseen_space='no occupied evidence; not claimed collision-free',
                    target_removed_using='causal_RGBD_known_red_box_estimate')


class ObservedMapMixin:
    def _start_alignment(self):
        super()._start_alignment()
        observation,_=self.observe()
        self.map_service=ObservedVoxelMap(self,observation,self.map_service)
        self.robot_model=AlignedRobotModel(self.arm,self.map_service)


class ObservedMapOfficialSimulation(ObservedMapMixin,AlignedOfficialSimulation):pass
class ObservedMapLocalSimulation(ObservedMapMixin,AlignedLocalSimulation):pass
