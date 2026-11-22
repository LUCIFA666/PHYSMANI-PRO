"""Unwired model-only hold nominal study; no Native/collision certificates.

The existing MPC cost is not replaced or called. The optimizer below chooses
only a feedforward nominal by a fixed DLS endpoint-tracking objective. Every
trial starts from the same current virtual state AND its matching virtual h;
only a chosen virtual prediction propagates to the next horizon step.
"""
from copy import deepcopy
import numpy as np
from .hold_orientation_guard_prototype_20261103 import _identity
from .joint_goal_cem import JointCEMConfig
from ..local_mpc_20261102.batch_mpc import norm,qlog,qmul,qinv
from ..local_mpc_p2_20261103.action_adapter import DLSConfig

VERSION='unwired_timeindexed_hold_nominal_bounded_response_DLS_20261103_v1'
HORIZON=8
DT=.1
STEP_M=.012
MICRO_ANGLE_RAD=.04
NUMERICAL_DIFFERENCE_RAD=1e-5


def hold_reference_schedule(actual,*,progressive):
    """Original12mm first reference; optional fixed-time incremental horizon."""
    _identity(actual)
    if type(progressive)is not bool:raise ValueError('Explicit static/progressive research arm required')
    if (actual.get('held')is not True or actual.get('grasp_motion_mode')!='ATTACHED_PUBLIC'
        or any(actual.get(k)is not False for k in ('Native_object_state_used','Native_held_used','Native_success_used'))):
        raise ValueError('Existing publicly confirmed hold only; no Native labels')
    tcp=np.asarray(actual['tcp_position'],float);body=np.asarray(actual['object_position'],float)
    goal=np.asarray(actual['phase_state']['lift_goal_position'],float)
    rotation=np.asarray(actual['phase_state']['hold_tcp_quaternion'],float)
    if any(v.shape!=(3,)or not np.isfinite(v).all()for v in (tcp,body,goal)):
        raise ValueError('Finite current public TCP/object/global goal required')
    delta=goal-body;distance=float(np.linalg.norm(delta));direction=delta/distance if distance else np.zeros(3)
    fractions=np.arange(1,HORIZON+1)if progressive else np.ones(HORIZON)
    positions=tcp+np.minimum(distance,STEP_M*fractions)[:,None]*direction
    return dict(positions=positions,quaternions=np.broadcast_to(rotation,(HORIZON,4)).copy(),
        times=float(actual['time'])+DT*np.arange(1,HORIZON+1),global_object_goal=goal.copy(),
        current_object=body.copy(),current_TCP=tcp.copy(),progressive=progressive,
        first_increment_m=float(np.linalg.norm(positions[0]-tcp)),
        successive_increments_m=np.linalg.norm(np.diff(np.vstack((tcp,positions)),axis=0),axis=1),
        source='one_current_public_feedback_no_GT_or_future_observations')


def _batch(value,n):
    if isinstance(value,np.ndarray):
        return np.repeat(value,n,axis=0)if value.ndim and value.shape[0]==1 else value.copy()
    if isinstance(value,dict):return {k:_batch(v,n)for k,v in value.items()}
    if isinstance(value,list):return [_batch(v,n)for v in value]
    if isinstance(value,tuple):return tuple(_batch(v,n)for v in value)
    return deepcopy(value)


def _row(value,index,n):
    if isinstance(value,np.ndarray):
        return value[index:index+1].copy()if value.ndim and value.shape[0]==n else value.copy()
    if isinstance(value,dict):return {k:_row(v,index,n)for k,v in value.items()}
    if isinstance(value,list):return [_row(v,index,n)for v in value]
    if isinstance(value,tuple):return tuple(_row(v,index,n)for v in value)
    return deepcopy(value)


class ModelOnlyHoldResponse:
    """Actual frozen execution model, with explicit model-only admissibility."""
    def __init__(self,execution,helper,actual):
        self.execution,self.helper=execution,helper
        self.feedback_sha256=_identity(actual)
        self.holdR=np.asarray(actual['phase_state']['hold_tcp_quaternion'],float).copy()
        self.model_batch_calls=0;self.modeled_macro_rows=0;self.modeled_micro_rows=0

    def initialize(self,actual):
        if _identity(actual)!=self.feedback_sha256:raise ValueError('Fresh same actual q/dq/h/holdR required')
        return self.execution.initialize_batch(actual,1)

    def evaluate(self,x,h,joint_goals):
        goals=np.asarray(joint_goals,float)
        if goals.ndim!=2 or goals.shape[1]!=7 or not np.isfinite(goals).all():raise ValueError('Finite trial qgoal[N,7] required')
        n=len(goals);bounds=self.execution.limits
        command=self.execution.fk_batch(goals.copy())
        action=dict(joint_goal=goals.copy(),opening=np.zeros(n),phase_index=np.full(n,3,np.int64),
            tcp_position=command['tcp_position'].copy(),tcp_quaternion=command['tcp_quaternion'].copy(),
            tcp_velocity_reference=np.zeros((n,3)),angular_velocity_reference=np.zeros((n,3)))
        nxt,hn,info=self.execution.step_batch(_batch(x,n),action,_batch(h,n),DT)
        self.model_batch_calls+=1;self.modeled_macro_rows+=n;self.modeled_micro_rows+=2*n
        q=np.asarray(info['predicted_actual_q'],float)
        if q.shape!=(n,2,7)or info['microsteps']!=2:raise ValueError('Both50ms actual-q predictions required')
        fk=self.execution.fk_batch(q.reshape(-1,7));angles=norm(qlog(qmul(fk['tcp_quaternion'],qinv(self.holdR)))).reshape(n,2)
        feasible=np.asarray(info['feasible'])
        if feasible.shape!=(n,)or feasible.dtype!=np.bool_:raise ValueError('Original model feasible[N] bool required')
        qbounds=np.all((goals>=bounds.lower-1e-8)&(goals<=bounds.upper+1e-8),axis=1)
        actualbounds=np.all((q>=bounds.lower-1e-8)&(q<=bounds.upper+1e-8),axis=(1,2))
        q_inputs=np.asarray(info['actual_q_inputs'],float)
        if q_inputs.shape!=q.shape:raise ValueError('Both microstep actual-q inputs required')
        velocity=np.all(abs((q-q_inputs)/.05)<=bounds.velocity+1e-8,axis=(1,2))
        angle_ok=np.all(angles<=MICRO_ANGLE_RAD+1e-10,axis=1)
        workspace=np.all((nxt['tcp_position']>=self.helper.config.lower)&(nxt['tcp_position']<=self.helper.config.upper),axis=1)
        command_workspace=np.all((command['tcp_position']>=self.helper.config.lower)&(command['tcp_position']<=self.helper.config.upper),axis=1)
        ok=feasible&qbounds&actualbounds&velocity&angle_ok&workspace&command_workspace
        return dict(x=nxt,h=hn,feasible=ok,original_model_feasible=feasible,micro_angles_rad=angles,
            q_bounds=qbounds,actual_q_bounds=actualbounds,actual_micro_velocity_bounds=velocity,
            workspace=workspace&command_workspace,joint_goals=goals.copy(),command_TCP=command['tcp_position'],
            Native_collision_qualified=False,physical_execution_qualified=False)


def _pose_residual(prediction,position,quaternion,rotation_weight):
    s=prediction['x']
    return np.c_[np.asarray(position)-s['tcp_position'],
        rotation_weight*qlog(qmul(np.asarray(quaternion),qinv(s['tcp_quaternion'])))]


def inverse_response_nominal(model,x,h,nominal_q,position,quaternion):
    """Fixed12-iteration damped response inverse, bounded by old search domain.

    DLS numerical constants come from the original generic DLS defaults. Search
    never grants a Native certificate. An infeasible best diagnostic is kept
    only to propagate/report a failed model-only study, not as a policy action.
    """
    # Original generic DLS defaults were predeclared for this new numerical
    # inverse. The real pose helper's separate24/.01 configuration is untouched.
    cfg=DLSConfig();nominal=np.asarray(nominal_q,float).copy()
    radius=JointCEMConfig().max_perturbation_rad
    lower=np.maximum(model.execution.limits.lower,nominal-radius)
    upper=np.minimum(model.execution.limits.upper,nominal+radius)
    q=np.minimum(np.maximum(nominal,lower),upper)
    best=None;diagnostic=None;trace=[]
    for iteration in range(cfg.iterations):
        trials=np.broadcast_to(q,(15,7)).copy()
        for j in range(7):
            trials[1+2*j,j]=min(upper[j],q[j]+NUMERICAL_DIFFERENCE_RAD)
            trials[2+2*j,j]=max(lower[j],q[j]-NUMERICAL_DIFFERENCE_RAD)
        pred=model.evaluate(x,h,trials)
        error=_pose_residual(pred,position,quaternion,cfg.rotation_weight)
        losses=np.sum(error*error,axis=1)
        for i in range(len(trials)):
            item=dict(loss=float(losses[i]),joint_goal=trials[i].copy(),prediction=_row(pred,i,len(trials)))
            if diagnostic is None or item['loss']<diagnostic['loss']:diagnostic=item
            if pred['feasible'][i]and(best is None or item['loss']<best['loss']):best=item
        trace.append(dict(iteration=iteration,current_loss=float(losses[0]),current_model_feasible=bool(pred['feasible'][0]),
            feasible_trials=int(pred['feasible'].sum())))
        jac=np.zeros((6,7))
        for j in range(7):
            width=trials[1+2*j,j]-trials[2+2*j,j]
            if width:jac[:,j]=(error[1+2*j]-error[2+2*j])/width
        if np.linalg.norm(jac)<1e-12:break
        step=-jac.T@np.linalg.solve(jac@jac.T+cfg.damping**2*np.eye(6),error[0])
        length=float(np.linalg.norm(step))
        if length>cfg.max_joint_step:step*=cfg.max_joint_step/length
        qnext=np.minimum(np.maximum(q+step,lower),upper)
        if np.linalg.norm(qnext-q)<1e-12:break
        q=qnext
    selected=best if best is not None else diagnostic
    return dict(selected=selected,model_feasible_proposal=best is not None,iterations=trace,
        numerical_constants=dict(iterations=cfg.iterations,damping=cfg.damping,rotation_weight=cfg.rotation_weight,
            max_joint_step=cfg.max_joint_step,max_deviation_from_IK_nominal_rad=radius,finite_difference_rad=NUMERICAL_DIFFERENCE_RAD),
        clipping_scope='projected numerical optimizer iterate only; no posthoc clipping of a selected MPC action',
        Native_collision_qualified=False)


def rollout_nominal_study(execution,helper,actual,*,progressive,response_aware):
    if type(response_aware)is not bool:raise ValueError('Explicit fixed research arm required')
    before=_identity(actual);schedule=hold_reference_schedule(actual,progressive=progressive)
    ik=helper.adapter.solve(np.broadcast_to(np.asarray(actual['q'],float),(8,7)).copy(),
        schedule['positions'],schedule['quaternions'])
    model=ModelOnlyHoldResponse(execution,helper,actual);x,h=model.initialize(actual)
    ledger=[];prefix=True
    for k in range(8):
        q=ik['joint_goal'][k]
        if response_aware:
            inverse=inverse_response_nominal(model,x,h,q,schedule['positions'][k],schedule['quaternions'][k])
            chosen=inverse['selected'];pred=chosen['prediction'];q=chosen['joint_goal'];trace=inverse['iterations']
        else:
            pred=model.evaluate(x,h,q[None]);trace=[]
        feasible=bool(pred['feasible'][0]);prefix&=feasible
        x,h=pred['x'],pred['h']
        ledger.append(dict(step=k+1,reference_TCP=schedule['positions'][k].copy(),reference_R=schedule['quaternions'][k].copy(),
            algebraic_IK_feasible=bool(ik['joint_goal_feasible'][k]),IK_nominal_q=ik['joint_goal'][k].copy(),
            proposed_q=q.copy(),model_feasible=feasible,prefix_model_feasible=prefix,
            predicted_TCP=x['tcp_position'][0].copy(),predicted_object=x['object_position'][0].copy(),
            predicted_q=x['q'][0].copy(),predicted_dq=x['dq'][0].copy(),
            reference_tracking_error_m=float(np.linalg.norm(x['tcp_position'][0]-schedule['positions'][k])),
            world_object_goal_distance_m=float(np.linalg.norm(x['object_position'][0]-schedule['global_object_goal'])),
            micro_angles_rad=pred['micro_angles_rad'][0].copy(),
            q_goal_and_actual_bounds=bool(pred['q_bounds'][0]&pred['actual_q_bounds'][0]),
            actual_micro_velocity_bounds=bool(pred['actual_micro_velocity_bounds'][0]),
            inverse_iterations=trace,Native_collision_qualified=False))
    if _identity(actual)!=before:raise ValueError('Original actual q/dq/h/holdR was changed')
    return dict(version=VERSION,schedule=schedule,steps=ledger,model_feasible_horizon=prefix,
        model_calls=dict(batch_calls=model.model_batch_calls,macro_rows=model.modeled_macro_rows,micro_rows=model.modeled_micro_rows),
        proposed_q_sequence=[r['proposed_q']for r in ledger]if prefix else None,
        production_integrated=False,Native_collision_qualified=False,Native_steps=0,
        actual_or_continuous_motion_guaranteed=False,virtual_h_reused_as_next_actual_feedback=False)
