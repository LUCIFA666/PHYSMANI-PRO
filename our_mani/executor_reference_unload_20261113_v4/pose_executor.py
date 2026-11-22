from ..executor_contact_frame_20261113_v3.pose_executor import PoseGoalExecutor as ContactExecutor,WorldPoseGoal
from ..executor_grasp_frame_20261113_v1.reference import step
from .unload import LiveReferenceUnload

class PoseGoalExecutor(ContactExecutor):
    def __init__(self,*args,live_reference_unload=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.live_unload=LiveReferenceUnload()if live_reference_unload else None

    def _next(self,state,goal,actual,A,b,*,retained):
        diagnostic=None
        if self.live_unload is not None:
            velocity,diagnostic=self.live_unload.update(state.position,actual,
                self.executor.limits.lower,self.executor.limits.upper,self.backend.clock())
            if velocity is not None:
                next_state,receipt=step(state,velocity,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                    speed_norm=self.speed_norm,joint_margin=self.joint_margin)
                receipt['live_reference_unload']=diagnostic
                return next_state,receipt
        next_state,receipt=super()._next(state,goal,actual,A,b,retained=retained)
        if diagnostic is not None:receipt['live_reference_unload']=diagnostic
        return next_state,receipt

    def _dispatch(self,*args,**kwargs):
        receipt=super()._dispatch(*args,**kwargs)
        receipt['scheduled_gripper_delay_ticks']=self.schedule.delay_ticks
        return receipt
