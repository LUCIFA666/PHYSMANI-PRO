"""Robot-only IK seeds from TRAIN robot joint states; no scene/object queries."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_capture_repair_20261112_v1.precision import refine

def solve_posture(fk,bank,current,position,quaternion,lower,upper,*,safe=lambda q:True):
    bank=np.asarray(bank,float);current=np.asarray(current,float)
    transforms=fk(bank);desired=Rotation.from_quat(quaternion)
    pose_score=10*np.linalg.norm(transforms[:,:3,3]-position,axis=1)+(
        desired*Rotation.from_matrix(transforms[:,:3,:3]).inv()).magnitude()
    seeds=[current,*bank[np.argsort(pose_score)[:3]]];unique=[]
    lower=np.asarray(lower,float);upper=np.asarray(upper,float);trials=[];valid=[]
    for seed in seeds:
        if any(np.linalg.norm(seed-old)<.1 for old in unique):continue
        unique.append(seed)
        # This bounds an internal optimization seed, not robot sensing or a world goal.
        initial=np.minimum(upper-.05,np.maximum(lower+.05,seed))
        q,receipt=refine(fk,initial,position,quaternion,lower+.05,upper-.05)
        margin=np.minimum(q-lower,upper-q)
        accepted=receipt['after_position_m']<=.002 and receipt['after_rotation_rad']<=.02 and safe(q)
        score=.05*np.dot(q-current,q-current)+.05*np.sum(1/(margin+.05)**2)
        trials.append(dict(**receipt,accepted=bool(accepted),minimum_joint_margin_rad=float(np.min(margin)),
            posture_cost=float(score),joint_distance_from_reference=float(np.linalg.norm(q-current))))
        if accepted:valid.append((float(score),q))
    chosen=None if not valid else min(valid,key=lambda item:item[0])[1].copy()
    return chosen,dict(available=chosen is not None,trials=trials,
        selected_posture=None if chosen is None else chosen.tolist(),
        source='TRAIN_robot_joint_states_and_current_robot_FK',task_body_pose_used=False,
        Native_setters_called=False,reference_state_reseeded=False,
        endpoint_clearance_is_not_a_global_path_certificate=True)

class RobotPosturePrior:
    def __init__(self,path,task,*,period=.5):
        with np.load(path)as source:self.bank=source[task+'_q'].astype(float)
        if self.bank.ndim!=2 or not len(self.bank)or not np.isfinite(self.bank).all():
            raise ValueError('Finite TRAIN robot posture bank required')
        self.last_time=None;self.selected=None;self.receipt=None;self.period=period

    def update(self,clearance,limits,current,position,quaternion,clock):
        if self.last_time is None or clock-self.last_time>=self.period-1e-6:
            self.selected,self.receipt=solve_posture(clearance.fk,self.bank,current,position,quaternion,
                limits.lower,limits.upper,safe=lambda q:clearance.safe(q)[0])
            self.last_time=clock
        return self.selected,self.receipt
