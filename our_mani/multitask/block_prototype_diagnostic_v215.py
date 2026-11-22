"""Read-only collision-prototype and support geometry on the unchanged policy."""
import hashlib
import numpy as np
from ..contracts import Pose
from .block_held_collision_diagnostic_v210 import SmoothBlockEnvironment as OriginalBoundary, public_decision_scope


def box_corners(bounds):
    b=np.asarray(bounds,float)
    if b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[::2]>=b[1::2]):
        raise ValueError('finite nonempty local collision bounding box required')
    return np.array([[x,y,z] for x in b[:2] for y in b[2:4] for z in b[4:6]])


def world_bounds(points,matrix):
    p=np.asarray(points,float);m=np.asarray(matrix,float)
    if p.ndim!=2 or p.shape[1]!=3 or m.shape!=(4,4) or not np.isfinite(np.r_[p.ravel(),m.ravel()]).all():
        raise ValueError('finite local vertices and original static transform required')
    world=p@m[:3,:3].T+m[:3,3]
    return dict(min=world.min(0).tolist(),max=world.max(0).tolist())


class SmoothBlockEnvironment(OriginalBoundary):
    def _diagnose(self,offset):
        row=super()._diagnose(offset)
        service=self.env.map_service
        pose=Pose.from_xyzw(row['predicted_held_body_pose'])
        matrix=np.eye(4);matrix[:3,:3]=pose.rotation;matrix[:3,3]=pose.position
        bounds=service.target.get_bounding_box()
        vertices,triangles,_=service.target.get_mesh_data()
        vertices=np.asarray(vertices,float).reshape(-1,3)
        points=np.asarray(self.public_body.points_world,float)
        row.update(target_local_bbox=list(bounds),
            target_bbox_world_at_public_estimate=world_bounds(box_corners(bounds),matrix),
            target_mesh_world_at_public_estimate=world_bounds(vertices,matrix),
            target_mesh_local_bounds=world_bounds(vertices,np.eye(4)),
            target_local_mesh_sha256=hashlib.sha256(vertices.tobytes()+np.asarray(triangles).tobytes()).hexdigest(),
            public_box_points_world_bounds=world_bounds(points,np.eye(4)),
            collision_prototype_geometry_is_metric_only=True)
        workspace=[]
        for ghost,info in zip(service.ghosts,service.map):
            if info['name']!='workspace':continue
            matrix=np.asarray(ghost.get_matrix(),float)
            bounds=ghost.get_bounding_box();vertices,_,_=ghost.get_mesh_data()
            workspace.append(dict(name=info['name'],local_bbox=list(bounds),world_matrix=matrix.tolist(),
                bbox_world=world_bounds(box_corners(bounds),matrix),
                mesh_world=world_bounds(np.asarray(vertices).reshape(-1,3),matrix),
                original_shared_map_descriptor=info))
        row['workspace_static_collision_geometry']=workspace
        return row
