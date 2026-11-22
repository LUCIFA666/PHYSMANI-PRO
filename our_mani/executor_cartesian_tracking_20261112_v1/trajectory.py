import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_capture_repair_20261112_v1.goal_motion import current_pose

def sample(history,lookahead,elapsed):
    horizon=lookahead-elapsed
    if horizon<.002:raise ValueError('Timed trajectory must be refreshed before lookahead expires')
    p,q,receipt=current_pose(history,horizon)
    if not receipt['valid']:
        # Until motion can be inferred, approach the explicitly supplied
        # destination with feedback; do not invent target velocity.
        return np.asarray(history[-1][1]),np.asarray(history[-1][2]),np.zeros(6),receipt
    future_p,future_q,_=current_pose(history,horizon-.001)
    twist=np.r_[(future_p-p)/.001,(Rotation.from_quat(future_q)*Rotation.from_quat(q).inv()).as_rotvec()/.001]
    return p,q,twist,receipt
