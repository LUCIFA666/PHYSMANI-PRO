import numpy as np
from scipy.linalg import logm,expm
from scipy.spatial.transform import Rotation
from ..executor_capture_repair_20261112_v1.goal_motion import current_pose

def predict(history,horizon):
    if horizon>=0:return current_pose(history,horizon)
    p,q,receipt=current_pose(history,0.)
    if not receipt['valid']:return p,q,receipt
    previous_p,previous_q,_=current_pose(history,.001)
    rotation=Rotation.from_quat(q);previous_rotation=Rotation.from_quat(previous_q)
    if receipt['model']=='constant_world_rigid_body_twist':
        transform=np.eye(4);transform[:3,3]=p;transform[:3,:3]=rotation.as_matrix()
        previous=np.eye(4);previous[:3,3]=previous_p;previous[:3,:3]=previous_rotation.as_matrix()
        twist=logm(transform@np.linalg.inv(previous)).real/.001
        predicted=expm(-horizon*twist)@transform
        p=predicted[:3,3];q=Rotation.from_matrix(predicted[:3,:3]).as_quat()
    else:
        p=p-horizon*(p-previous_p)/.001
        angular=(rotation*previous_rotation.inv()).as_rotvec()/.001
        q=(Rotation.from_rotvec(-horizon*angular)*rotation).as_quat()
    return p,q,dict(receipt,forecast_from_last_observation_seconds=-horizon)

def sample(history,lookahead,elapsed):
    horizon=lookahead-elapsed
    p,q,receipt=predict(history,horizon)
    if not receipt['valid']:return np.asarray(history[-1][1]),np.asarray(history[-1][2]),np.zeros(6),receipt
    future_p,future_q,_=predict(history,horizon-.001)
    twist=np.r_[(future_p-p)/.001,(Rotation.from_quat(future_q)*Rotation.from_quat(q).inv()).as_rotvec()/.001]
    return p,q,twist,receipt
