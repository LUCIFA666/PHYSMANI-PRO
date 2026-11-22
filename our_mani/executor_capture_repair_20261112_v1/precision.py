"""Optional read-only local IK refinement; never changes a world destination."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def refine(fk,seed,position,quaternion,lower,upper):
    seed=np.asarray(seed,float);target=np.asarray(position,float);rotation=Rotation.from_quat(quaternion)
    def errors(q):
        transform=fk(q)
        return transform[:3,3]-target,(rotation.inv()*Rotation.from_matrix(transform[:3,:3])).as_rotvec()
    def residual(q):
        p,r=errors(q)
        return np.r_[10*p,r,1e-4*(q-seed)]
    before=errors(seed)
    result=least_squares(residual,seed,bounds=(np.asarray(lower),np.asarray(upper)),
                         xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=40)
    after=errors(result.x)
    receipt=dict(before_position_m=float(np.linalg.norm(before[0])),before_rotation_rad=float(np.linalg.norm(before[1])),
        after_position_m=float(np.linalg.norm(after[0])),after_rotation_rad=float(np.linalg.norm(after[1])),
        joint_change_norm_rad=float(np.linalg.norm(result.x-seed)),function_evaluations=result.nfev,
        native_setters_or_physics_calls=0,world_destination_modified=False)
    return result.x,receipt
