"""Original expert branch logic, with future rotation computed without setters."""
import types
import numpy as np
from scipy.spatial.transform import Rotation

def future_pose(frame_base,waypoint,t_delay,yaw_speed):
    base=np.asarray(frame_base.get_matrix(),float)
    waypoint_matrix=np.asarray(waypoint.get_matrix(),float)
    relative=np.linalg.solve(base,waypoint_matrix)
    rotation=np.eye(4)
    rotation[:3,:3]=Rotation.from_rotvec([0.,0.,np.deg2rad(yaw_speed)*t_delay]).as_matrix()
    transform=base@rotation@relative
    return np.r_[transform[:3,3],Rotation.from_matrix(transform[:3,:3]).as_quat()]

def readonly_expert(original):
    namespace=dict(original.__globals__)
    if 'compute_target_pose'in namespace:namespace['compute_target_pose']=future_pose
    return types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
