"""Fixed-step pose-goal dispatch through continuous, checked joint samples."""
import numpy as np
from ..executor_contract_repair_20261112_v1.pose_executor import WorldPoseGoal
from ..executor_contract_repair_20261112_v1.pose_executor import PoseGoalExecutor as PreviousPoseExecutor
from ..execution_interface_20261111_v1.commands import JointTrajectory
from .reference import ReferenceState,step,preview


class PoseGoalExecutor(PreviousPoseExecutor):
    def _state(self):
        prior=self.executor._last_direct
        if prior is None:
            return ReferenceState(self.backend.get_joint_positions(),self.backend.get_joint_velocities(),
                                  np.zeros(len(self.executor.limits.lower)))
        if abs(prior[0]-self.backend.clock())>self.executor.config.clock_tolerance:
            raise RuntimeError('Stale dispatched reference state')
        return ReferenceState(*prior[1:])

    def _dispatch(self,goal,*,opening=None,stop_requested=lambda:False,retained=False):
        receipts=[];issued=False;start=float(self.backend.clock())
        for tick in range(10):
            if stop_requested()or self.backend.terminal()or self.backend.remaining_ticks()==0:break
            next_ref=step(self._state(),goal,self.executor.limits,frequency=self.executor.config.reference_frequency)
            op=opening if tick==9 else None
            command=JointTrajectory(start_time=self.backend.clock(),positions=[next_ref.position],
                                    times_from_start=[.05],opening=op)
            receipt=self.executor.execute_command(command,phase='retained_continuous_goal'if retained else
                'continuous_pose_goal_scheduled_gripper'if tick==9 else'continuous_pose_goal_arm',
                planner_contract=dict(reference_source='discrete_bounded_continuation',
                                      reseed_on_retarget=False,actual_robot_prediction=False))
            receipts.append(receipt.to_dict())
            if tick==9 and receipt.executed_ticks and opening is not None:issued=True
            if not receipt.executed_ticks:break
        count=sum(r['executed_ticks']for r in receipts)
        return dict(status=('retained_goal_chunk_complete'if retained else'chunk_complete')if count==10 else'truncated',
                    requested_ticks=10,executed_ticks=count,start_clock=start,end_clock=float(self.backend.clock()),
                    execution=receipts,gripper_intent_issued=issued,scheduled_gripper_tick=10,
                    joint_goal=np.asarray(goal).tolist(),gripper_done=self.executor.gripper_done,
                    actual_persistent_opening=self.executor.gripper_target,
                    whole_RML_path_waited=False,measured_arrival_claimed=False,grasp_or_task_success_claimed=False,
                    reference_mode='continuous_discrete_joint_samples_to_destination',
                    reference_reseeded_on_target_change=False,reference_difference_limits_checked=True,
                    exact_unmodified_author_executor=False,all_physics_and_task_steps_counted=True)

    def execute(self,action,*,stop_requested=lambda:False):
        if self._fault is not None:raise RuntimeError('Pose executor faulted: '+self._fault)
        if type(action)is not WorldPoseGoal:raise TypeError('Typed world-pose goal required')
        if abs(action.timestamp-self.backend.clock())>self.executor.config.clock_tolerance:
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
            same=np.allclose(action.position,p,atol=1e-9,rtol=0)and abs(abs(float(np.dot(action.quaternion_xyzw,q)))-1.)<=1e-8
        if same:goal=self._cached_joint_goal.copy()
        else:
            goal=self._decode(action)
            if goal is None:return dict(status='IK_failed',executed_ticks=0,execution=[],gripper_intent_issued=False)
            goal=np.asarray(goal,float);self.executor._validate_positions(goal)
            acceptance=preview(self._state(),goal,self.executor.limits,frequency=self.executor.config.reference_frequency)
            if not acceptance['feasible']:
                return dict(status='reference_rejected',executed_ticks=0,execution=[],
                            gripper_intent_issued=False,previous_accepted_goal_preserved=True,**acceptance)
            self._cached_pose=(np.asarray(action.position),np.asarray(action.quaternion_xyzw))
            self._cached_joint_goal=goal.copy()
        result=self._dispatch(goal,opening=action.opening,stop_requested=stop_requested)
        result['IK_reused']=bool(same)
        return result

    def advance_retained_goal(self,*,stop_requested=lambda:False):
        if self._fault is not None:raise RuntimeError('Pose executor faulted: '+self._fault)
        initial=self._cached_joint_goal is None
        goal=self._state().position if initial else self._cached_joint_goal
        result=self._dispatch(goal,stop_requested=stop_requested,retained=True)
        result.update(retained_goal_source='measured_initial_joint_hold'if initial else'previously_accepted_joint_goal',
                      rejected_target_modified=False)
        return result
