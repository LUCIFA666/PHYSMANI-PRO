"""Continuity-preserving robot-only IK with a bounded global-seed fallback."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from ..executor_capture_repair_20261112_v1.precision import refine

def diverse_bank_seeds(fk,bank,position,quaternion,*,maximum=6,separation=.5):
    transforms=fk(bank);desired=Rotation.from_quat(quaternion)
    score=10*np.linalg.norm(transforms[:,:3,3]-position,axis=1)+(
        desired*Rotation.from_matrix(transforms[:,:3,:3]).inv()).magnitude()
    chosen=[]
    for index in np.argsort(score):
        q=bank[index]
        if any(np.linalg.norm(q-old)<separation for old in chosen):continue
        chosen.append(q.copy())
        if len(chosen)>=maximum:break
    return chosen

class ContinuousRobotPosturePrior:
    def __init__(self,bank,*,period=.1):
        self.bank=np.asarray(bank,float).copy();self.period=period
        self.selected=None;self.receipt=None;self.last_time=None;self.last_pose=None
        self.generic_unit_seeds=qmc.Sobol(d=7,scramble=True,seed=20261113).random_base2(4)

    def update(self,clearance,limits,current,position,quaternion,clock):
        position=np.asarray(position,float);desired=Rotation.from_quat(quaternion)
        if self.last_pose is not None:
            movement=float(np.linalg.norm(position-self.last_pose[0]))
            turn=float((desired*Rotation.from_quat(self.last_pose[1]).inv()).magnitude())
            if self.selected is not None and movement<=.0001 and turn<=.002:
                pose=clearance.fk(self.selected)
                if (np.linalg.norm(pose[:3,3]-position)<=.002
                        and(desired*Rotation.from_matrix(pose[:3,:3]).inv()).magnitude()<=.02
                        and bool(clearance.safe(self.selected)[0])):
                    return self.selected,dict(self.receipt,unchanged_goal_branch_retained=True)
            if (self.last_time is not None and clock-self.last_time<self.period-1e-6
                    and movement<.03 and turn<.2):return self.selected,self.receipt
        previous=None if self.selected is None else self.selected.copy()
        lo=np.asarray(limits.lower)+.05;hi=np.asarray(limits.upper)-.05
        seeds=[] if previous is None else[previous]
        seeds.append(np.asarray(current,float))
        seeds.extend(diverse_bank_seeds(clearance.fk,self.bank,position,quaternion))
        trials=[];valid=[];seen=[]
        def attempt(seed,kind):
            if any(np.linalg.norm(seed-old)<.1 for old in seen):return False
            seen.append(np.asarray(seed,float).copy())
            initial=np.minimum(hi,np.maximum(lo,seed))
            q,receipt=refine(clearance.fk,initial,position,quaternion,lo,hi)
            accepted=bool(receipt['after_position_m']<=.002 and receipt['after_rotation_rad']<=.02
                and clearance.safe(q)[0])
            margin=np.minimum(q-np.asarray(limits.lower),np.asarray(limits.upper)-q)
            cost=.05*np.dot(q-current,q-current)+.05*np.sum(1/(margin+.05)**2)
            if previous is not None:cost+=.5*np.dot(q-previous,q-previous)
            trials.append(dict(receipt,accepted=accepted,seed_kind=kind,posture_cost=float(cost),
                minimum_joint_margin_rad=float(np.min(margin))))
            if accepted:valid.append((float(cost),q.copy()))
            return accepted
        for i,seed in enumerate(seeds):
            accepted=attempt(seed,'previous_solution'if previous is not None and i==0 else'robot_state_or_diverse_TRAIN')
            if (accepted and previous is not None and i==0
                    and np.linalg.norm(valid[-1][1]-previous)<.35):break
        fallback=not valid
        if fallback:
            for unit in self.generic_unit_seeds:
                attempt(lo+unit*(hi-lo),'generic_internal_joint_seed')
                if len(valid)>=2:break
        self.selected=None if not valid else min(valid,key=lambda row:row[0])[1].copy()
        self.receipt=dict(available=self.selected is not None,trials=trials,
            selected_posture=None if self.selected is None else self.selected.tolist(),
            generic_fallback_used=fallback,source='robot_FK_diverse_TRAIN_joint_states_and_internal_joint_seeds',
            previous_solution_used_for_continuity=previous is not None,task_body_pose_used=False,
            Native_setters_called=False,reference_state_reseeded=False,
            endpoint_clearance_is_not_a_global_path_certificate=True)
        self.last_time=clock;self.last_pose=(position.copy(),np.asarray(quaternion,float).copy())
        return self.selected,self.receipt
