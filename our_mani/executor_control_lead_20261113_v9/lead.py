import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

def lead_pose(position,quaternion,twist,seconds):
    p=np.asarray(position,float);twist=np.asarray(twist,float)
    angular=twist[3:];linear=twist[:3]-np.cross(angular,p)
    x,y,z=angular;generator=np.zeros((4,4));generator[:3,:3]=[[0,-z,y],[z,0,-x],[-y,x,0]]
    generator[:3,3]=linear;transform=np.eye(4)
    transform[:3,3]=p;transform[:3,:3]=Rotation.from_quat(quaternion).as_matrix()
    projected=expm(seconds*generator)@transform
    p_next=projected[:3,3];q_next=Rotation.from_matrix(projected[:3,:3]).as_quat()
    v_next=np.r_[linear+np.cross(angular,p_next),angular]
    return p_next,q_next,v_next
