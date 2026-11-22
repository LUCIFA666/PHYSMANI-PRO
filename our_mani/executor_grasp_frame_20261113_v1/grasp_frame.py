import numpy as np
from scipy.spatial.transform import Rotation

def matrix(position,quaternion):
    result=np.eye(4);result[:3,3]=position;result[:3,:3]=Rotation.from_quat(quaternion).as_matrix();return result

def capture_delta(nominal_position,nominal_quaternion,actual_pose):
    return np.linalg.solve(matrix(nominal_position,nominal_quaternion),matrix(actual_pose[:3],actual_pose[3:7]))

def apply_delta(position,quaternion,twist,delta):
    nominal=matrix(position,quaternion);actual=nominal@delta
    offset=nominal[:3,:3]@delta[:3,3]
    corrected=np.r_[np.asarray(twist[:3])+np.cross(twist[3:],offset),twist[3:]]
    return actual[:3,3],Rotation.from_matrix(actual[:3,:3]).as_quat(),corrected
