"""Reconstruct only recorded public RGBD, calibration and initial visual fit."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from ..contracts import Pose
from ..perception import CameraCalibration
from ..benchmark_perception import BoxPrior
from ..local_mpc_control_20261103.goal_aware_perception import VisualGoalSphere,filter_observed_sphere_surface
from ..local_mpc_control_20261103.pixel_centre_far_tracker_prototype_20261103 import PixelCentreFarCalibration


def load_observation(case,index):
    case=Path(case);values=json.loads((case/'public_observer/camera_calibration.json').read_text())
    calibrations={c:CameraCalibration(np.asarray(v['intrinsics']),Pose.from_xyzw(v['camera_to_world'])) for c,v in values.items()}
    with np.load(case/'public_observer/rgbd'/f'{index:04d}.npz',allow_pickle=False) as frame:
        observation={'timestamp':float(frame['timestamp']),'observation/images':{c:frame['images_'+c].copy() for c in values},
            'observation/depths':{c:frame['depths_'+c].copy() for c in values},'observation/calibrations':calibrations}
    sphere=VisualGoalSphere(**json.loads((case/'public_sphere.json').read_text()))
    return observation,sphere


def public_red_components(observation,sphere,*,workspace_min=(-.05,-.6,.74),workspace_max=(.5,.6,1.3)):
    prior=BoxPrior(minimum_red=100,workspace_min=workspace_min,workspace_max=workspace_max)
    private=dict(observation)
    private['observation/calibrations']={c:PixelCentreFarCalibration(v.intrinsics,v.camera_to_world) for c,v in observation['observation/calibrations'].items()}
    filtered,trace=filter_observed_sphere_surface(private,sphere,prior)
    clouds=[];views=[];colors=[]
    for c,rgb in filtered['observation/images'].items():
        image=rgb.astype(float)
        mask=(image[...,0]>=100)&(image[...,0]>3.5*image[...,1])&(image[...,0]>3.5*image[...,2])
        depth=filtered['observation/depths'][c];valid=np.isfinite(depth)&(depth>0)
        cloud=private['observation/calibrations'][c].backproject(depth,mask)
        keep=((cloud>=prior.workspace_min)&(cloud<=prior.workspace_max)).all(1)
        clouds.append(cloud[keep]);views.extend([c]*int(keep.sum()));colors.append(image[mask&valid][keep])
    cloud=np.concatenate(clouds);views=np.asarray(views);colors=np.concatenate(colors)
    if not len(cloud):return [],trace
    graph=cKDTree(cloud).sparse_distance_matrix(cKDTree(cloud),.028,output_type='coo_matrix')
    n,labels=connected_components(graph,directed=False);components=[]
    for i in range(n):
        points=cloud[labels==i]
        if len(points)<12:continue
        low,hi=np.quantile(points,[.01,.99],axis=0);center=(low+hi)/2;center[2]=np.quantile(points[:,2],.98)-.025
        components.append(dict(points=points,views=views[labels==i],colors=colors[labels==i],n=len(points),
            visible_views=len(set(views[labels==i])),center=center,extent=hi-low,
            initial_height_gate=bool(abs(center[2]-.775)<=.08)))
    return components,trace
