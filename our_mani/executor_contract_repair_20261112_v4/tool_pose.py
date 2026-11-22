"""Rigid TCP-to-housing geometry can reject a pose before any IK search."""
import numpy as np
from scipy.spatial.transform import Rotation


def housing_clearance(position,quaternion_xyzw,*,tcp_reference,housing_reference,vertices,
                      table_matrix,table_bounds,margin):
    target=np.eye(4);target[:3,:3]=Rotation.from_quat(quaternion_xyzw).as_matrix();target[:3,3]=position
    tool_to_housing=np.linalg.inv(tcp_reference)@housing_reference
    transform=np.linalg.inv(table_matrix)@target@tool_to_housing
    points=np.asarray(vertices)@transform[:3,:3].T+transform[:3,3]
    low=points.min(0);high=points.max(0);b=np.asarray(table_bounds)
    overlap=bool(high[0]>=b[0]and low[0]<=b[1]and high[1]>=b[2]and low[1]<=b[3])
    clearance=float(low[2]-b[5])if overlap else None
    violation=max(float(margin)-clearance,0.)if overlap else 0.
    return dict(admissible=violation<=1e-7,tool_housing_clearance_m=clearance,
                planning_margin_violation_m=violation,Native_IK_called=False,
                scope='fixed_TCP_pose_fixes_gripper_housing_for_every_IK_branch',
                physical_collision_certificate=False)
