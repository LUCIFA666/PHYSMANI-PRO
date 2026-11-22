import numpy as np

class LiveReferenceUnload:
    """Change the reference velocity toward current sensing; never reseed q/v/a."""
    def __init__(self,enter=.10,exit=.025,dwell=.10,gain=6.,speed=.8,reserve=.05):
        self.enter=enter;self.exit=exit;self.dwell=dwell;self.gain=gain;self.speed=speed;self.reserve=reserve
        self.active=False;self.since=None

    def update(self,reference,actual,lower,upper,clock):
        reference=np.asarray(reference,float);actual=np.asarray(actual,float)
        lower=np.asarray(lower,float);upper=np.asarray(upper,float)
        low=actual<lower;high=actual>upper;outside=bool(np.any(low|high))
        error=float(np.max(abs(reference-actual)))
        if not self.active:
            if outside:self.active=True
            elif error>self.enter:
                if self.since is None:self.since=clock
                if clock-self.since>=self.dwell-1e-6:self.active=True
            else:self.since=None
        elif error<self.exit and not outside:self.active=False;self.since=None
        target=actual.copy();target[low]=lower[low]+self.reserve;target[high]=upper[high]-self.reserve
        velocity=self.gain*(target-reference)
        velocity*=min(1.,self.speed/max(float(np.linalg.norm(velocity)),1e-12))
        receipt=dict(active=self.active,maximum_tracking_error_rad=error,
            measured_joint_positions=actual.tolist(),explicit_reference_unload_goal=target.tolist(),
            velocity_norm_cap_rad_s=self.speed,reference_reseeded=False,actual_state_clipped=False,
            target_tracks_current_robot_sensing=True,task_body_state_used=False,
            measured_bound_violating_joints=np.flatnonzero(low|high).tolist())
        return velocity if self.active else None,receipt
