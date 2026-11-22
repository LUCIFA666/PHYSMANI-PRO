import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_contract_repair_20261112_v3.pose_executor import PoseGoalExecutor as ClearanceExecutor,WorldPoseGoal
from ..execution_interface_20261111_v1.commands import JointTrajectory
from .reference import step,ReferenceInfeasible,DT
from .tool_pose import housing_clearance


class PoseGoalExecutor(ClearanceExecutor):
    def __init__(self,backend,executor):
        super().__init__(backend,executor)
        self._last_safe_actual=np.asarray(backend.get_joint_positions(),float)
        self.barrier_rate=1.

    def _decode(self,action):
        housing=next(b for b in self.clearance.bodies if b['name']=='Panda_gripper')
        index=self.clearance.bodies.index(housing)
        check=housing_clearance(action.position,action.quaternion_xyzw,
            tcp_reference=self.clearance.fk.reference_tcp,housing_reference=housing['reference_matrix'],
            vertices=housing['vertices'],table_matrix=self.clearance.table_matrix,
            table_bounds=self.clearance.bounds,margin=self.clearance.body_margins[index])
        if not check['admissible']:
            self.last_clearance_diagnostic=dict(accepted=False,alternative_IK=False,checks=[dict(
                source='rigid_tool_pose_precheck',reason='hardware_table_intersection',body='Panda_gripper',
                reference_index=0,**check)])
            self.backend.last_ik_receipt=dict(success=False,reason='tool_housing_pose_fails_static_table_constraint',
                                              Native_IK_called=False,physics_steps=0)
            return None
        return super()._decode(action)

    def _qualify(self,goal,action):
        if goal is None:return False,dict(reason='IK_no_solution')
        goal=np.asarray(goal,float)
        try:self.executor._validate_positions(goal)
        except ValueError as error:return False,dict(reason='IK_solution_outside_native_contract',error=str(error))
        transform=self.clearance.fk(goal)
        position=float(np.linalg.norm(transform[:3,3]-action.position))
        angle=float((Rotation.from_matrix(transform[:3,:3]).inv()*Rotation.from_quat(action.quaternion_xyzw)).magnitude())
        if position>.002 or angle>.02:return False,dict(reason='IK_pose_residual',position_m=position,rotation_rad=angle)
        depths=self.clearance.depths(goal)[0]
        if np.any(depths>1e-7):
            body=int(np.argmax(depths))
            return False,dict(reason='hardware_table_intersection',body=self.clearance.bodies[body]['name'],
                reference_index=0,planning_margin_violation_m=float(depths[body]))
        state=self._state();actual=np.asarray(self.backend.get_joint_positions(),float)
        A,b,_=self._barriers(state,actual)
        try:
            first,_=step(state,goal,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                         frequency=self.executor.config.reference_frequency)
        except ReferenceInfeasible:return False,dict(reason='first_governed_reference_infeasible')
        if not self.clearance.safe(first.position)[0]:return False,dict(reason='first_governed_reference_table_intersection')
        return True,dict(reason='endpoint_and_first_governed_reference_clear',
            qualification_scope='exact_first_feedback_action_plus_static_endpoint_not_open_loop_tail',
            position_residual_m=position,rotation_residual_rad=angle,whole_path_certificate=False)

    def _barriers(self,state,actual):
        A=[];bounds=[];diagnostics=[]
        up=np.array([0.,0.,1.]);top=float(self.clearance.table_matrix[2,3]+self.clearance.bounds[5])
        screws=self.clearance.fk.screws
        for body,margin in zip(self.clearance.bodies,self.clearance.body_margins):
            transform=body['reference_matrix']
            points=body['vertices']@transform[:3,:3].T+transform[:3,3]
            local=(points-self.clearance.table_matrix[:3,3])@self.clearance.table_matrix[:3,:3]
            low=local.min(0);high=local.max(0);box=self.clearance.bounds
            if not(high[0]>=box[0]and low[0]<=box[1]and high[1]>=box[2]and low[1]<=box[3]):continue
            point=points[np.argmin(points[:,2])];gap=float(point[2]-top)
            jac=np.zeros(7)
            for joint in range(body['depth']):
                jac[joint]=(screws[joint,:3]+np.cross(screws[joint,3:],point))@up
            # Next commanded configuration must preserve the measured body's
            # clearance exponentially toward a safe margin. The 5mm reserve
            # covers local linearization, before exact next-sample checking.
            reference_gap=gap+float(jac@(state.position-actual))
            required_velocity=self.barrier_rate*(margin+.005-min(gap,reference_gap))
            # Govern approach velocity from both measured and commanded
            # clearance; do not demand a discontinuous jump back to actual q.
            rhs=(required_velocity-jac@state.velocity)/DT
            A.append(jac);bounds.append(rhs)
            diagnostics.append(dict(body=body['name'],measured_clearance_m=gap,required_margin_m=float(margin)))
        return np.asarray(A).reshape(-1,7),np.asarray(bounds),diagnostics

    def _dispatch(self,goal,*,opening=None,stop_requested=lambda:False,retained=False):
        receipts=[];issued=False;start=float(self.backend.clock());safety=None;safety_diagnostic=None
        for tick in range(10):
            if stop_requested()or self.backend.terminal()or self.backend.remaining_ticks()==0:break
            self.clearance.refresh(self.backend)
            actual=np.asarray(self.backend.get_joint_positions(),float)
            if (np.all(actual>=self.executor.limits.lower)and np.all(actual<=self.executor.limits.upper)
                    and self.clearance.safe(actual)[0]):self._last_safe_actual=actual.copy()
            state=self._state();A,b,barriers=self._barriers(state,actual)
            try:
                next_ref,governor=step(state,goal,self.executor.limits,actual_q=actual,barrier_A=A,barrier_b=b,
                                       frequency=self.executor.config.reference_frequency)
                # The exact nonlinear mesh check supplements the local barrier.
                if not self.clearance.safe(next_ref.position)[0]:
                    raise ReferenceInfeasible('Next governed reference violates exact static hardware clearance')
            except ReferenceInfeasible as error:
                safety=str(error);safety_diagnostic=getattr(error,'diagnostic',None);break
            op=opening if tick==9 else None
            command=JointTrajectory(start_time=self.backend.clock(),positions=[next_ref.position],times_from_start=[.05],opening=op)
            receipt=self.executor.execute_command(command,phase='feedback_brake_and_observe'if retained else'feedback_governed_pose',
                planner_contract=dict(reference_source='coupled_measured_feedback_governor',
                    target_and_measured_state_unmodified=True,governor=governor,barriers=barriers,
                    barrier_acceleration_matrix=A.tolist(),barrier_lower_bound=b.tolist()))
            receipts.append(receipt.to_dict())
            if tick==9 and receipt.executed_ticks and opening is not None:issued=True
            if not receipt.executed_ticks:break
        count=sum(r['executed_ticks']for r in receipts)
        result=dict(status=('retained_goal_chunk_complete'if retained else'chunk_complete')if count==10 else
                    'control_barrier_infeasible'if safety else'truncated',safety_stop=safety,safety_diagnostic=safety_diagnostic,
            requested_ticks=10,executed_ticks=count,start_clock=start,end_clock=float(self.backend.clock()),execution=receipts,
            gripper_intent_issued=issued,scheduled_gripper_tick=10,joint_goal=np.asarray(goal).tolist(),
            gripper_done=self.executor.gripper_done,actual_persistent_opening=self.executor.gripper_target,
            whole_RML_path_waited=False,measured_arrival_claimed=False,grasp_or_task_success_claimed=False,
            reference_mode='coupled_actual_feedback_discrete_joint_references',reference_reseeded_on_target_change=False,
            reference_difference_limits_checked=True,exact_unmodified_author_executor=False,
            all_physics_and_task_steps_counted=True)
        self.last_dispatch_receipt=result
        return result

    def advance_retained_goal(self,*,stop_requested=lambda:False):
        # A rejected moving-world command must not drive toward an indefinitely
        # stale destination. Brake toward a recent measured safe configuration.
        result=self._dispatch(self._last_safe_actual,stop_requested=stop_requested,retained=True)
        result.update(retained_goal_source='recent_safe_measured_joint_brake',rejected_target_modified=False)
        self._validate_hardware(result)
        return result
