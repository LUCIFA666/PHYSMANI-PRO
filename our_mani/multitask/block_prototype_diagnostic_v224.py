"""Read-only collision-prototype and support geometry on the unchanged policy."""
import hashlib
import numpy as np
from ..contracts import Pose
from .block_held_collision_diagnostic_v210 import SmoothBlockEnvironment as OriginalBoundary, public_decision_scope


from .geometry_bounds_v223 import box_corners,world_bounds


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
