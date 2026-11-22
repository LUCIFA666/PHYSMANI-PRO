import numpy as np
from ..executor_contract_repair_20261112_v4.pose_executor import PoseGoalExecutor as PreviousExecutor,WorldPoseGoal
from ..executor_contract_repair_20261112_v4.reference import step,ReferenceInfeasible
from ..execution_interface_20261111_v1.commands import JointTrajectory
from .gripper_schedule import GripperSchedule

class PoseGoalExecutor(PreviousExecutor):
    def __init__(self,backend,executor,*,control_ticks=2,barrier_rate=5.):
        if type(control_ticks)is not int or control_ticks not in (2,10):raise ValueError('Qualified cadence: 2 or 10 ticks')
        if barrier_rate not in (1.,5.):raise ValueError('Explicit qualified barrier profile required')
        super().__init__(backend,executor)
        self.control_ticks=control_ticks
        self.barrier_rate=barrier_rate
        self.schedule=GripperSchedule(10)

    def _dispatch(self,goal,*,opening=None,stop_requested=lambda:False,retained=False):
        receipts=[];issued=False;start=float(self.backend.clock());safety=None;safety_diagnostic=None
        if retained:self.schedule.cancel()
        elif opening is not None:self.schedule.request(opening,self.executor.gripper_target,self.backend._ticks)
        for _ in range(self.control_ticks):
            if stop_requested()or self.backend.terminal()or self.backend.remaining_ticks()==0:break
            self.clearance.refresh(self.backend)
            actual=np.asarray(self.backend.get_joint_positions(),float)
            if (np.all(actual>=self.executor.limits.lower)and np.all(actual<=self.executor.limits.upper)
                    and self.clearance.safe(actual)[0]):self._last_safe_actual=actual.copy()
            state=self._state();A,b,barriers=self._barriers(state,actual)
            try:
                next_ref,governor=step(state,goal,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                                     frequency=self.executor.config.reference_frequency)
                if not self.clearance.safe(next_ref.position)[0]:
                    raise ReferenceInfeasible('Next governed reference violates exact static hardware clearance')
            except ReferenceInfeasible as error:
                safety=str(error);safety_diagnostic=getattr(error,'diagnostic',None);self.schedule.cancel();break
            op=None if retained else self.schedule.due(self.backend._ticks)
            command=JointTrajectory(start_time=self.backend.clock(),positions=[next_ref.position],times_from_start=[.05],opening=op)
            receipt=self.executor.execute_command(command,phase='feedback_brake_and_observe'if retained else'feedback_governed_pose',
                planner_contract=dict(reference_source='coupled_measured_feedback_governor',
                    target_and_measured_state_unmodified=True,governor=governor,barriers=barriers,
                    barrier_acceleration_matrix=A.tolist(),barrier_lower_bound=b.tolist(),
                    gripper_schedule_uses_physical_ticks=True))
            receipts.append(receipt.to_dict())
            if receipt.executed_ticks and op is not None:self.schedule.acknowledge(op);issued=True
            if not receipt.executed_ticks:break
        count=sum(r['executed_ticks']for r in receipts)
        result=dict(status=('retained_goal_chunk_complete'if retained else'chunk_complete')if count==self.control_ticks else
                    'control_barrier_infeasible'if safety else'truncated',safety_stop=safety,safety_diagnostic=safety_diagnostic,
            requested_ticks=self.control_ticks,executed_ticks=count,start_clock=start,end_clock=float(self.backend.clock()),execution=receipts,
            gripper_intent_issued=issued,scheduled_gripper_delay_ticks=10,gripper_pending=self.schedule.pending,
            joint_goal=np.asarray(goal).tolist(),gripper_done=self.executor.gripper_done,
            actual_persistent_opening=self.executor.gripper_target,whole_RML_path_waited=False,measured_arrival_claimed=False,
            grasp_or_task_success_claimed=False,reference_mode='coupled_actual_feedback_discrete_joint_references',
            reference_reseeded_on_target_change=False,reference_difference_limits_checked=True,
            exact_unmodified_author_executor=False,all_physics_and_task_steps_counted=True)
        self.last_dispatch_receipt=result
        return result
