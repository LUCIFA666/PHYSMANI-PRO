import numpy as np
from scipy.spatial.transform import Rotation
from ..executor_contract_repair_20261112_v2.pose_executor import PoseGoalExecutor as ContinuousPoseExecutor,WorldPoseGoal
from ..executor_contract_repair_20261112_v2.reference import step,ReferenceInfeasible
from .clearance import HardwareTableClearance


class PoseGoalExecutor(ContinuousPoseExecutor):
    def __init__(self,backend,executor):
        super().__init__(backend,executor)
        self.clearance=HardwareTableClearance(backend)
        self.last_clearance_diagnostic=None
        initial=self.clearance.depths(backend.get_joint_positions())[0]
        if np.any(initial>1e-7):raise ValueError('Original robot home violates static tabletop clearance: '+str(initial))
        self.hardware_FK_max_error=0.

    def _qualify(self,goal,action):
        if goal is None:return False,dict(reason='IK_no_solution')
        goal=np.asarray(goal,float)
        try:self.executor._validate_positions(goal)
        except ValueError as error:
            return False,dict(reason='IK_solution_outside_native_contract',error=str(error))
        transform=self.clearance.fk(goal)
        pos=float(np.linalg.norm(transform[:3,3]-action.position))
        angle=float((Rotation.from_matrix(transform[:3,:3]).inv()*Rotation.from_quat(action.quaternion_xyzw)).magnitude())
        if pos>.002 or angle>.02:return False,dict(reason='IK_pose_residual',position_m=pos,rotation_rad=angle)
        current=self._state();points=[]
        for _ in range(200):
            try:current=step(current,goal,self.executor.limits,frequency=self.executor.config.reference_frequency)
            except ReferenceInfeasible:return False,dict(reason='reference_not_bounded')
            points.append(current.position)
        depths=self.clearance.depths(np.vstack((goal,np.asarray(points))))
        if np.any(depths>1e-7):
            index,body=np.unravel_index(np.argmax(depths),depths.shape)
            return False,dict(reason='hardware_table_intersection',body=self.clearance.bodies[body]['name'],
                              reference_index=int(index),planning_margin_violation_m=float(depths[index,body]))
        return True,dict(reason='bounded_and_table_clear',reference_samples=200,position_residual_m=pos,rotation_residual_rad=angle)

    def _sample(self,action):
        backend=self.backend;before=backend._clock_tuple()
        q=np.asarray(backend.get_joint_positions());dq=np.asarray(backend.get_joint_velocities())
        servo=np.asarray(backend.arm.get_joint_target_positions());ik=backend.arm._ik_target.get_pose()
        try:
            try:
                return backend.arm.solve_ik_via_sampling(action.position,quaternion=action.quaternion_xyzw,
                    ignore_collisions=True,trials=80,max_configs=8,max_time_ms=5)
            except backend._ik_errors:return []
        finally:
            if not np.allclose(backend.arm.get_joint_positions(),q,atol=1e-6,rtol=0):
                backend.arm.set_joint_positions(q.tolist(),disable_dynamics=False)
            backend.arm.set_joint_target_positions(servo.tolist());backend.arm._ik_target.set_pose(ik)
            errors=(np.max(abs(np.asarray(backend.get_joint_positions())-q)),
                    np.max(abs(np.asarray(backend.get_joint_velocities())-dq)),
                    np.max(abs(np.asarray(backend.arm.get_joint_target_positions())-servo)))
            if backend._clock_tuple()!=before or errors[0]>1e-6 or errors[1]>1e-9 or errors[2]>1e-6:
                self._fault='Alternative IK query state/clock changed';raise RuntimeError(self._fault)

    def _decode(self,action):
        primary=super()._decode(action)
        ok,receipt=self._qualify(primary,action)
        checks=[dict(source='native_primary',**receipt)]
        if ok:
            self.last_clearance_diagnostic=dict(accepted=True,alternative_IK=False,checks=checks)
            return primary
        alternatives=self._sample(action)
        current=np.asarray(self.backend.get_joint_positions())
        ordered=sorted(alternatives,key=lambda q:np.linalg.norm(np.asarray(q)-current))
        for goal in ordered:
            ok,receipt=self._qualify(goal,action);checks.append(dict(source='native_alternative',**receipt))
            if ok:
                self.last_clearance_diagnostic=dict(accepted=True,alternative_IK=True,checks=checks)
                return np.asarray(goal,float)
        self.last_clearance_diagnostic=dict(accepted=False,alternative_IK=False,checks=checks)
        return None

    def execute(self,action,**kwargs):
        self.clearance.refresh(self.backend)
        # A previous IK solution was qualified against a previous measured
        # robot calibration. Retain it as fallback, but solve the requested
        # Cartesian goal again without resetting the dispatched references.
        self._cached_pose=None
        result=super().execute(action,**kwargs)
        result['hardware_clearance']=self.last_clearance_diagnostic
        if result['status']=='IK_failed'and self.last_clearance_diagnostic and any(
                c['reason']=='hardware_table_intersection'for c in self.last_clearance_diagnostic['checks']):
            result['status']='hardware_clearance_rejected'
        self._validate_hardware(result)
        return result

    def _validate_hardware(self,result):
        errors=self.clearance.native_pose_residuals(self.backend)
        maximum=max(v['matrix_max_abs']for v in errors.values())
        self.hardware_FK_max_error=max(self.hardware_FK_max_error,maximum)
        # Allocate at most half the planning margin to rigid-model residuals.
        # Translation + radius*angle bounds vertex displacement; this avoids
        # comparing metres with raw matrix entries or requiring bitwise FK.
        uncertainty={b['name']:errors[b['name']]['position_m']+
            float(np.linalg.norm(b['vertices'],axis=1).max())*errors[b['name']]['rotation_rad']for b in self.clearance.bodies}
        if any(uncertainty[b['name']]>margin/2 for b,margin in zip(self.clearance.bodies,self.clearance.body_margins)):
            self._fault='Robot body FK disagrees with Native geometry: '+repr(errors);raise RuntimeError(self._fault)
        result['hardware_FK_max_matrix_error']=maximum
        result['hardware_FK_residuals']=errors
        result['hardware_vertex_uncertainty_bound_m']=uncertainty
        result['hardware_kinematic_calibration']=self.clearance.calibration()
        depths=self.clearance.depths(self.backend.get_joint_positions())[0]
        result['actual_hardware_table_margin_violations_m']={b['name']:float(d)for b,d in zip(self.clearance.bodies,depths)}

    def advance_retained_goal(self,**kwargs):
        result=super().advance_retained_goal(**kwargs)
        self._validate_hardware(result)
        return result
