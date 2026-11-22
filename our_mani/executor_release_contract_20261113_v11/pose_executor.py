import math
from ..executor_held_target_20261113_v7.pose_executor import PoseGoalExecutor as CurrentExecutor,WorldPoseGoal

def release_decision(receipt,policy,window_m):
    if policy=='terminal_pose':return dict(receipt,explicit_release_policy=policy)
    if policy=='caller_intent':allowed=True
    elif policy=='position_window':
        allowed=receipt['motion_compensation']['valid']and receipt['measured_TCP_position_error_m']<=window_m
    else:raise ValueError('Unsupported release policy')
    return dict(receipt,allowed=bool(allowed),reason='explicit_'+policy+'_release',
        explicit_release_policy=policy,position_window_m=window_m if policy=='position_window'else None,
        previous_terminal_pose_gate_allowed=receipt['allowed'],
        task_success_claimed=False,Native_task_success_predicate_unchanged=True)

class PoseGoalExecutor(CurrentExecutor):
    def __init__(self,*args,release_policy='terminal_pose',release_window_m=.03,release_delay_ticks=1,**kwargs):
        super().__init__(*args,**kwargs)
        if release_policy not in ('terminal_pose','caller_intent','position_window'):raise ValueError('Explicit release policy required')
        if not math.isfinite(release_window_m)or not 0<release_window_m<=.05:raise ValueError('Release window must be in (0,5 cm]')
        if type(release_delay_ticks)is not int or not 1<=release_delay_ticks<=20:raise ValueError('Counted release delay 1..20 ticks required')
        self.release_policy=release_policy;self.release_window_m=release_window_m;self.release_delay_ticks=release_delay_ticks

    def _release_receipt(self):
        return release_decision(super()._release_receipt(),self.release_policy,self.release_window_m)

    def _dispatch(self,*args,opening=None,retained=False,**kwargs):
        delay=self.schedule.delay_ticks
        if not retained and(opening==1. or self.commitment.release_latched)and self.executor.gripper_target!=1.:
            self.schedule.delay_ticks=self.release_delay_ticks
        try:
            result=super()._dispatch(*args,opening=opening,retained=retained,**kwargs)
            result['release_contract']=dict(policy=self.release_policy,release_delay_physical_ticks=self.release_delay_ticks,
                Native_success_predicate_changed=False)
            return result
        finally:self.schedule.delay_ticks=delay
