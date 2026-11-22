"""Prefer a response margin; every fallback retains the original exact gates."""
import numpy as np
from .collision_guard import CollisionStoppingGuard
from .collision_guard_v2 import RobustCollisionStoppingGuard
from .tracking_v4 import reference_window


class RecoverableCollisionStoppingGuard(RobustCollisionStoppingGuard):
    def select(self,state,proposed,opening):
        try:
            action,receipt=super().select(state,proposed,opening)
            receipt['additional_response_margin_qualified']=True
            return action,receipt
        except RuntimeError as error:
            margin_failure=str(error)
        # A calibration-error margin is an extra diagnostic hypothesis. It
        # cannot make the original collision oracle unavailable. The original
        # endpoint, seven path samples and whole braking prefix still pass.
        p=self.planner;r,v,a=p._history(state)
        lo,hi=reference_window(r,v,a,p.limits,margin=p.config.joint_limit_margin)
        brake=np.minimum(hi,np.maximum(lo,r));candidates=[]
        if proposed is not None:
            candidates.extend((w,brake+w*(np.asarray(proposed['joint_goal'])-brake)) for w in (1.,.75,.5,.25))
        candidates.append((0.,brake))
        for joint in range(7):
            for bound in (lo,hi):
                ref=brake.copy();ref[joint]=bound[joint];candidates.append((0.,ref))
        attempts=[]
        for weight,ref in candidates:
            allowed,receipt=CollisionStoppingGuard.certify(self,state,ref,opening)
            attempts.append(dict(proposal_weight=weight,allowed=allowed,receipt=receipt))
            if allowed:
                action=dict(proposed) if weight==1. else self.action(ref,opening)
                return action,dict(proposal_weight=weight,attempts=attempts,stopping_continuation_qualified=True,
                    additional_response_margin_qualified=False,margin_failure=margin_failure,
                    original_collision_gates_preserved=True,physical_steps=0)
        raise RuntimeError('No original-collision-qualified bounded stopping continuation: '+str([(x['proposal_weight'],x['receipt']['reason']) for x in attempts]))
