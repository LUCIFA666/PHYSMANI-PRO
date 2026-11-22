import numpy as np

def joint_recovery(reference,actual,lower,upper,*,reserve=.05):
    reference=np.asarray(reference,float);actual=np.asarray(actual,float)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    low=actual<lower;high=actual>upper;target=reference.copy()
    target[low]=lower[low]+reserve;target[high]=upper[high]-reserve
    return target,dict(active=bool(np.any(low|high)),violating_joints=np.flatnonzero(low|high).tolist(),
        original_measured_positions=actual.tolist(),explicit_recovery_reference_goal=target.tolist(),
        actual_state_clipped=False,original_world_target_changed=False,reserve_rad=reserve)
