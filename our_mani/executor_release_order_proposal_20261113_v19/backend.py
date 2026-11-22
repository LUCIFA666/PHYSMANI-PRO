"""Selectable gripper release order using the original Native actuator APIs."""
import numpy as np
from ..executor_release_settle_20261113_v18.backend import TaskExecutionBackend as OriginalBackend

class TaskExecutionBackend(OriginalBackend):
    def __init__(self,*args,release_order='native_detach_before_open',**kwargs):
        if release_order not in ('native_detach_before_open','open_fingers_then_native_release'):
            raise ValueError('Explicit reviewed gripper release ordering required')
        self.release_order=release_order
        self.release_order_events=[]
        super().__init__(*args,**kwargs)
        self.provenance.update(native_grasp_and_release_API_implementation_unchanged=True,
            native_author_release_order_preserved=release_order=='native_detach_before_open',
            release_order=release_order,control_strategy_configurable=True)

    def _record_release_order(self,held_before,native_done,completed,release_called,amounts,rearmed=False):
        self.last_release_order_receipt=dict(order=self.release_order,clock=self.clock(),
            requested_opening=self.requested_opening,native_actuate_done=bool(native_done),
            physical_finger_open_amounts=list(map(float,amounts)),physical_opening_completed=bool(completed),
            Native_held_before=bool(held_before),Native_held_after=bool(self.gripper.get_grasped_objects()),
            Native_release_called=bool(release_called),reversal_rearmed_without_physics=bool(rearmed),
            physics_steps=0,task_success_claimed=False)
        if release_called:self.release_order_events.append(dict(self.last_release_order_receipt))

    def actuate_gripper(self,opening):
        if self.release_order=='native_detach_before_open':
            held=bool(self.gripper.get_grasped_objects());done=super().actuate_gripper(opening)
            amounts=self.gripper.get_open_amount()
            self._record_release_order(held,done,done and all(x>=.95 for x in amounts),
                held and not self.gripper.get_grasped_objects(),amounts)
            return done
        if not np.isfinite(opening)or not 0<=opening<=1:raise ValueError('Finite opening intent in 0..1 required')
        if opening<.5:return super().actuate_gripper(opening)
        self._can_command();before=self._clock_tuple()
        previous=self.requested_opening;held=bool(self.gripper.get_grasped_objects());self.requested_opening=1.
        done=bool(self.gripper.actuate(1.,velocity=self.gripper_velocity))
        amounts=np.asarray(self.gripper.get_open_amount(),float)
        rearmed=False
        if previous!=1. and done and not np.all(amounts>=.95):
            # Native actuate can return True on velocity reversal while its
            # fingers are still closed. Re-arm once without advancing time.
            done=bool(self.gripper.actuate(1.,velocity=self.gripper_velocity));rearmed=True
            amounts=np.asarray(self.gripper.get_open_amount(),float)
        if not np.isfinite(amounts).all():raise ValueError('Finite measured finger aperture required')
        completed=bool(done and np.all(amounts>=.95))
        release_called=bool(completed and self.gripper.get_grasped_objects())
        if release_called:self.gripper.release()
        self._assert_unchanged(before,'proposed open-fingers-then-release ordering')
        self._record_release_order(held,done,completed,release_called,amounts,rearmed)
        return completed
