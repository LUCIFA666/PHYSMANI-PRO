"""Pure copies of Native free-motion schedule and boundary state transitions.

This is privileged TRAIN data-collection assistance, never a learned predictor
or public policy. The forecast is exact conditional on the target remaining
ungrasped and the original task driver continuing, not after contact/release.
No Native module is imported, stepped, written or asked to plan by this code.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

from ..contracts import Forecast
from ..local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel


def _xyz(value,name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('finite xyz required: '+name)
    return a.copy()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
        default=lambda v:np.asarray(v).tolist()).encode()).hexdigest()


def simulate_free_future(segment,*,task_time,area,current_position,current_velocity,
                         anchor_time,steps=16,dt=.05,task_dt=None):
    """Snapshot index0 +16 actual tick-end positions from original task rules.

    task.step computes at current task.t THEN increments task.t. Thus future
    index1 uses task_time, not task_time+dt. Crossing flips the instantaneous
    velocity but keeps acceleration, and retains boundary overshoot unchanged.
    """
    if not isinstance(segment,dict) or set(segment)!={'x','v','a','t0'}:
        raise ValueError('Exact copied original last motion segment required')
    if (type(steps)is not int or steps!=16 or isinstance(dt,bool)or dt!=.05
            or any(isinstance(v,(bool,np.bool_))or not np.isscalar(v)or not np.isfinite(v)or v<0 for v in (task_time,anchor_time))):
        raise ValueError('Measured nonnegative clocks, H17 and actual50ms timestep required')
    if isinstance(segment['t0'],(bool,np.bool_))or not np.isfinite(segment['t0'])or segment['t0']>task_time+1e-9:
        raise ValueError('Causal original segment start time required')
    task_dt=float(dt)if task_dt is None else task_dt
    if isinstance(task_dt,(bool,np.bool_))or not np.isfinite(task_dt)or abs(task_dt-.05)>1e-6:
        raise ValueError('Measured original task driver must have a50ms timestep')
    bounds=np.asarray(area,dtype=np.float64)
    if bounds.shape!=(6,)or not np.isfinite(bounds).all()or np.any(bounds[:2]>=bounds[3:5]):
        raise ValueError('Original finite Native area bounds required')
    copied=deepcopy(segment);before=_digest(segment)
    state=dict(x=_xyz(copied['x'],'segment x'),v=_xyz(copied['v'],'segment v'),
        a=_xyz(copied['a'],'segment a'),t0=float(copied['t0']))
    position=_xyz(current_position,'measured current position')
    velocity=_xyz(current_velocity,'measured current causal FD velocity')
    positions=[position.copy()];velocities=[velocity.copy()];trace=[]
    simulated_task_time=float(task_time)
    for index in range(1,17):
        at=simulated_task_time
        elapsed=at-state['t0']
        nxt=state['x']+state['v']*elapsed+.5*state['a']*elapsed**2
        analytic_velocity=state['v']+state['a']*elapsed
        axes=[]
        # Exact Native cross_boundary inequalities, only world X/Y.
        if ((position[0]>=bounds[0]and nxt[0]<=bounds[0])or
                (position[0]<=bounds[3]and nxt[0]>=bounds[3])):axes.append(0)
        if ((position[1]>=bounds[1]and nxt[1]<=bounds[1])or
                (position[1]<=bounds[4]and nxt[1]>=bounds[4])):axes.append(1)
        previous=position.copy();position=nxt.copy()
        fd_velocity=(position-previous)/dt
        if axes:
            reflected=analytic_velocity.copy()
            reflected[axes]=-reflected[axes]
            state=dict(t0=at,x=position.copy(),v=reflected,a=state['a'].copy())
        positions.append(position.copy());velocities.append(fd_velocity)
        trace.append(dict(index=index,physical_query_time=float(anchor_time)+index*dt,
            original_task_evaluation_time=at,boundary_axes=axes,position=position.tolist(),
            analytic_driver_velocity_before_reflection=analytic_velocity.tolist(),
            nominal50ms_difference_velocity=fd_velocity.tolist(),clamped=False))
        simulated_task_time+=task_dt
    if _digest(segment)!=before:raise RuntimeError('Oracle mutated caller motion segment')
    forecast=Forecast(('target',),float(anchor_time)+np.arange(17)*dt,
        np.asarray(positions)[:,None,:],np.tile(np.eye(3),(17,1,1,1)),
        np.asarray(velocities)[:,None,:],np.zeros((17,1,3)),
        metadata=dict(method='PRIVATE_TRAIN_exact_conditional_Native_schedule',
            Native_inputs_used=True,future_observations_used=False,public_policy_prediction=False,
            conditional_on_target_remaining_ungrasped=True,not_GRU=True,
            task_time=float(task_time),observed_anchor_time=float(anchor_time),
            canonical_planning_rotation='world_identity_prior_only_not_Native_body_rotation'))
    return forecast,dict(privileged_TRAIN_teacher_only=True,public_bridge_verified=False,
        GRU_used=False,conditional_on_target_remaining_ungrasped=True,
        original_segment_sha256=before,original_segment=deepcopy(copied),original_area=bounds.tolist(),
        measured_task_driver_dt_seconds=float(task_dt),nominal_Forecast_query_dt_seconds=dt,
        driver_dt_not_replaced_by_nominal=True,
        index1_evaluates_task_time_before_increment=True,current_index0_is_actual_state=True,
        future_velocity_source='conditional tick position differences / nominal .05s Forecast grid, not Native Object.get_velocity certificate',
        trajectory_trace=trace,Native_steps_added=0,Native_writes=0,RNG_calls=0)


def forecast_from_private_task(task,state):
    if state.get('held')is not False:
        raise ValueError('Private free schedule is not an attached/release predictor')
    if getattr(task,'_bool_expert',None)is not False:
        raise ValueError('Original task expert stepping must be disabled')
    if not task.target_state_list:raise ValueError('Original Native motion segment is absent')
    now=float(task.t)
    if abs(float(state['task_time'])-now)>1e-9:
        raise ValueError('Private task driver and actual state task clocks disagree')
    task_dt=float(task.pyrep.get_simulation_timestep())
    snapshot=deepcopy(task.target_state_list[-1]);area=deepcopy(task.area)
    forecast,receipt=simulate_free_future(snapshot,task_time=now,area=area,
        current_position=state['object_position'],current_velocity=state['object_linear_velocity'],
        anchor_time=state['time'],task_dt=task_dt)
    if _digest(task.target_state_list[-1])!=_digest(snapshot)or task.t!=now or list(task.area)!=list(area):
        raise RuntimeError('Private oracle reads changed task state')
    return forecast,receipt


class PrivateExactFutureExecutionBinding(BoundForecastExecutionModel):
    """Original binding mathematics, with truthful private oracle receipt."""
    def __init__(self,execution_model,forecast,*,anchor_time,oracle_receipt):
        if forecast.metadata.get('public_policy_prediction')is not False or oracle_receipt.get('privileged_TRAIN_teacher_only')is not True:
            raise ValueError('Explicit private oracle Forecast required, never public authority')
        self.oracle_receipt=deepcopy(oracle_receipt)
        super().__init__(execution_model,forecast,anchor_time=anchor_time,
            orientation_observable=False,held_source='diagnostic_actual_anchor',public_bridge_verified=False)

    def receipt(self):
        result=super().receipt()
        result.update(rotation_source='private_canonical_Cube_I_not_observed_attitude',
            prediction_source='PRIVATE_TRAIN_exact_conditional_Native_schedule',
            public_bridge_contract_verified=False,Native_schedule_inputs_used=True,
            GRU_used=False,privileged_TRAIN_teacher_only=True,
            oracle=deepcopy(self.oracle_receipt))
        return result
