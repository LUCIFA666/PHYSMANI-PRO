"""World-pose GOAL -> bounded persistent joint goal + scheduled gripper.

The input is a destination, not a sampled trajectory. Existing sampled joint
trajectories retain their separate unchanged API. A chunk completing means
physical steps completed, never actual pose arrival, grasp or task success.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from ..execution_interface_20261111_v1.commands import vector,opening,timestamp
from ..local_mpc_p2_20261103.reference import reference_step


@dataclass(frozen=True)
class WorldPoseGoal:
    timestamp: float
    position: object
    quaternion_xyzw: object
    opening: float
    frame: str='world'

    def __post_init__(self):
        at=timestamp(self.timestamp)
        if at is None or self.frame!='world':raise ValueError('Explicit current timestamp and world frame required')
        p=vector(self.position,'world TCP position',3)
        q=vector(self.quaternion_xyzw,'world TCP quaternion xyzw',4)
        if abs(np.linalg.norm(q)-1.)>1e-5:raise ValueError('Unit xyzw quaternion required')
        op=opening(self.opening)
        if op is None:raise ValueError('Explicit binary opening intent required')
        object.__setattr__(self,'timestamp',at)
        object.__setattr__(self,'position',p)
        object.__setattr__(self,'quaternion_xyzw',q)
        object.__setattr__(self,'opening',op)


class PoseGoalExecutor:
    """N9 arm stage + scheduled opening on tick10; no full-path wait.

    This adopts PhysMani's fixed-step dispatch idea with our counted,
    nonblocking actuator protocol. Gripper completion persists across chunks;
    no hidden physics or dynamics-reset path-preview helpers are invoked.
    """
    def __init__(self,backend,executor):
        if executor.backend is not backend:raise ValueError('Same backend required')
        self.backend,self.executor=backend,executor
        self._cached_pose=None
        self._cached_joint_goal=None
        self._fault=None

    def _decode(self,action):
        before=float(self.backend.clock())
        q=np.asarray(self.backend.get_joint_positions(),float)
        dq=np.asarray(self.backend.get_joint_velocities(),float)
        servo_reader=getattr(self.backend,'get_joint_target_positions',None)
        if servo_reader is None and hasattr(self.backend,'arm'):
            servo_reader=self.backend.arm.get_joint_target_positions
        servo=None if servo_reader is None else np.asarray(servo_reader(),float)
        try:
            return self.backend.solve_ik(action.position,action.quaternion_xyzw)
        finally:
            errors=dict(clock=abs(float(self.backend.clock())-before),
                        q=float(np.max(np.abs(np.asarray(self.backend.get_joint_positions())-q))),
                        dq=float(np.max(np.abs(np.asarray(self.backend.get_joint_velocities())-dq))))
            if servo is not None:
                errors['servo']=float(np.max(np.abs(np.asarray(servo_reader())-servo)))
            if (errors['clock']>1e-9 or errors['q']>1e-6 or errors['dq']>1e-9 or errors.get('servo',0)>1e-6):
                self._fault='IK query changed robot state or physics: '+repr(errors)
                raise RuntimeError(self._fault)

    def execute(self,action,*,stop_requested=lambda:False):
        if self._fault is not None:raise RuntimeError('Pose executor faulted: '+self._fault)
        if type(action)is not WorldPoseGoal:raise TypeError('Typed world-pose goal required')
        now=float(self.backend.clock())
        if abs(action.timestamp-now)>self.executor.config.clock_tolerance:
            raise ValueError('Stale/future pose goal; no implicit retiming')
        if self.backend.terminal()or self.backend.remaining_ticks()==0:
            return dict(status='terminal_or_budget',executed_ticks=0,execution=[],gripper_intent_issued=False)
        actual=np.asarray(self.backend.get_joint_positions(),float)
        if np.any(actual<self.executor.limits.lower)or np.any(actual>self.executor.limits.upper):
            return dict(status='measured_start_outside_bounds',executed_ticks=0,execution=[],
                        gripper_intent_issued=False,measured_positions=actual.tolist(),measured_q_clipped=False)
        same=False
        if self._cached_pose is not None:
            p,q=self._cached_pose
            same=(np.allclose(action.position,p,atol=1e-9,rtol=0)and
                  abs(abs(float(np.dot(action.quaternion_xyzw,q)))-1.)<=1e-8)
        before=float(self.backend.clock())
        if same:
            goal=self._cached_joint_goal.copy()
        else:
            goal=self._decode(action)
            if goal is None:
                return dict(status='IK_failed',executed_ticks=0,execution=[],gripper_intent_issued=False,
                            measured_arrival_claimed=False)
            goal=np.asarray(goal,float)
            self.executor._validate_positions(goal)
            # Acceptance is atomic: a failed retarget must not overwrite the
            # last accepted destination or the real reference state. Preview
            # the unchanged kernel to include its braking/settling tail, so
            # retaining an accepted target remains possible after rejection.
            h=deepcopy(self.executor._h)if self.executor._mode=='joint_goal'else{}
            dq=np.asarray(self.backend.get_joint_velocities(),float)
            for step in range(200):
                preview=reference_step(actual,dq,goal,h,self.executor.limits,
                                       frequency=self.executor.config.reference_frequency)
                if not preview['feasible'][0]or not preview['applied'][0]:
                    return dict(status='reference_rejected',executed_ticks=0,execution=[],
                                gripper_intent_issued=False,failure_code=int(preview['failure_code'][0]),
                                failed_reference_preview_tick=step+1,reference_preview_physics_steps=0,
                                previous_accepted_goal_preserved=True)
                h=preview['next_h']
            self._cached_pose=(np.asarray(action.position),np.asarray(action.quaternion_xyzw))
            self._cached_joint_goal=goal.copy()
        if abs(self.backend.clock()-before)>1e-9:raise RuntimeError('Pose decoding advanced physics')
        receipts=[]
        issued=False
        for tick in range(10):
            if stop_requested()or self.backend.terminal()or self.backend.remaining_ticks()==0:break
            # Keep the actual persistent request for nine arm ticks. Send the
            # new binary intent on the tenth, independently of arrival error.
            op=action.opening if tick==9 else None
            receipt=self.executor.execute_joint_target(goal,opening=op,ticks=1,
                phase='pose_goal_arm'if tick<9 else'pose_goal_scheduled_gripper',expected_time=self.backend.clock())
            receipts.append(receipt.to_dict())
            if tick==9 and receipt.executed_ticks:issued=True
            if receipt.executed_ticks==0:break
        count=sum(r['executed_ticks']for r in receipts)
        return dict(status='chunk_complete'if count==10 else'truncated',requested_ticks=10,executed_ticks=count,
                    start_clock=now,end_clock=float(self.backend.clock()),execution=receipts,
                    IK_reused=same,joint_goal=goal.tolist(),gripper_intent_issued=issued,
                    scheduled_gripper_tick=10,gripper_done=self.executor.gripper_done,
                    actual_persistent_opening=self.executor.gripper_target,
                    whole_RML_path_waited=False,measured_arrival_claimed=False,
                    grasp_or_task_success_claimed=False,reference_mode='bounded_joint_goal_not_sampled_RML',
                    author_fixed_step_idea=True,exact_unmodified_author_executor=False,
                    reference_acceptance_preview_ticks=200,reference_preview_predicts_actual_robot=False,
                    all_physics_and_task_steps_counted=True)

    def advance_retained_goal(self,*,stop_requested=lambda:False):
        """Explicit caller-selected response to a rejected destination.

        Continue the last accepted bounded joint destination for at most 0.5s,
        retaining the real gripper request. With no previously accepted goal,
        use a measured initial robot-joint hold. No failed target is projected,
        no new gripper intent is latched, and each tick still samples success.
        """
        if self._fault is not None:raise RuntimeError('Pose executor faulted: '+self._fault)
        initial=self._cached_joint_goal is None
        goal=(np.asarray(self.backend.get_joint_positions(),float)if initial else self._cached_joint_goal.copy())
        self.executor._validate_positions(goal)
        receipts=[]
        start=float(self.backend.clock())
        for _ in range(10):
            if stop_requested()or self.backend.terminal()or self.backend.remaining_ticks()==0:break
            r=self.executor.execute_joint_target(goal,opening=None,ticks=1,
                phase='rejected_pose_retain_accepted_control',expected_time=self.backend.clock())
            receipts.append(r.to_dict())
            if r.executed_ticks==0:break
        count=sum(r['executed_ticks']for r in receipts)
        return dict(status='retained_goal_chunk_complete'if count==10 else'truncated',
                    executed_ticks=count,requested_ticks=10,start_clock=start,end_clock=float(self.backend.clock()),
                    execution=receipts,gripper_intent_issued=False,
                    retained_goal_source='measured_initial_joint_hold'if initial else'previously_accepted_joint_goal',
                    joint_goal=goal.tolist(),rejected_target_modified=False)
