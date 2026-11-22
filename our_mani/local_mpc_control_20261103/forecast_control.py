"""CPU-only current-Forecast binding for diagnostic joint-goal MPC.

Only the free object's trajectory and the fixed-deadline nominal IK seeds
change. Robot response, phase thresholds, costs, constraints and the 256-row
search budget remain owned by the existing modules. This is a contract layer,
not a public closed-loop success or visual orientation qualification.
"""
from copy import deepcopy
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Forecast, Pose
from ..local_mpc_p2_20261103.action_adapter import matrix_quaternion
from ..local_mpc_p2_20261103.execution_models import validate_step_batch
from .joint_goal_cem import JointGoalCEMPlanner, NOMINAL_BASIS_VERSION


GRU_NOMINAL_BASIS = 'GRU_Forecast_absolute_deadline_anchor_lift_v1'
_BINDING_KEY = '_current_forecast_binding'
# Fixed before shadow smoke: float32 state encoding, not task tolerance.
ANCHOR_QUANTIZATION_TOLERANCE = dict(position_m=1e-6, orientation_rad=1e-6,
    linear_velocity_m_per_s=1e-6, angular_velocity_rad_per_s=1e-6)


class BoundForecastExecutionModel:
    """A snapshot of one predicted Forecast, never a sequence of observations.

    Each initialize_batch must start at this binding's exact observed anchor.
    An initialized virtual h is valid only for this binding. Anchor-held rows
    always remain with the internal model, including hypothetical release;
    a pre-contact GRU forecast cannot describe a newly released object.
    """
    def __init__(self, execution_model, forecast, *, anchor_time, object_id='target',
                 orientation_observable=False, held_source='diagnostic_actual_anchor',
                 public_bridge_verified=False):
        if not all(callable(getattr(execution_model, name, None)) for name in ('initialize_batch', 'step_batch')):
            raise TypeError('explicit batch execution model required')
        if not isinstance(forecast, Forecast):
            raise TypeError('predicted Forecast required; observations and label dictionaries are forbidden')
        if isinstance(anchor_time, bool) or not np.isfinite(anchor_time) or anchor_time < 0:
            raise ValueError('finite current anchor_time required')
        if type(orientation_observable) is not bool or type(public_bridge_verified) is not bool:
            raise ValueError('explicit orientation/provenance booleans required')
        if held_source not in ('diagnostic_actual_anchor', 'public_estimate'):
            raise ValueError('explicit anchor held provenance required')
        expected = float(anchor_time) + np.arange(17)*.05
        if (forecast.times.shape != (17,) or not np.allclose(forecast.times, expected, atol=1e-9, rtol=0)
                or object_id not in forecast.object_ids):
            raise ValueError('current index0=t0 plus16 predicted 50ms endpoints required')
        # Do not copy caller metadata (which is not an observation input).
        self.forecast = Forecast(tuple(forecast.object_ids), forecast.times, forecast.positions,
            forecast.rotations, forecast.linear_velocities, forecast.angular_velocities,
            metadata={'source': 'bound_prediction_only'})
        self.execution_model, self.anchor_time, self.object_id = execution_model, float(anchor_time), object_id
        self.orientation_observable, self.held_source = orientation_observable, held_source
        self.public_bridge_verified = public_bridge_verified
        self.binding_id = uuid4().hex
        self._samples = {}
        self.last_anchor_errors = None
        for at in self.anchor_time + np.arange(1, 9)*.1:
            self.sample(at)  # Validate every H8 endpoint before model/search work.

    def __getattr__(self, name):
        return getattr(self.execution_model, name)

    @classmethod
    def from_public_bridge(cls, execution_model, result):
        """Accept a current FREE bridge result, never its evaluation labels."""
        if not isinstance(result, dict) or not all(result.get(k) is True for k in
                ('available', 'free_forecast_available')) or result.get('mode') != 'FREE':
            raise ValueError('current available FREE public bridge result required')
        for key in ('Native_inputs_used', 'future_observations_used', 'observations_fabricated'):
            if result.get(key) is not False:
                raise ValueError('public bridge provenance missing or disallowed: '+key)
        if result.get('rotation_observable') is not False:
            raise ValueError('current bridge has no observed visual orientation')
        return cls(execution_model, result['forecast'], anchor_time=result['timestamp'],
            orientation_observable=False, held_source='public_estimate', public_bridge_verified=True)

    def sample(self, absolute_time):
        if isinstance(absolute_time, (bool, np.bool_)):
            raise ValueError('finite absolute timestamp required')
        at = float(absolute_time)
        if not np.isfinite(at) or at < self.anchor_time-1e-9 or at > self.forecast.times[-1]+1e-9:
            raise ValueError('requested absolute time outside current Forecast; no extrapolation')
        key = round(at-self.anchor_time, 12)
        if key not in self._samples:
            self._samples[key] = self.forecast.at(self.object_id, at)
        pose, velocity, angular = self._samples[key]
        return pose, velocity.copy(), angular.copy()

    def validate_anchor(self, actual):
        if (not isinstance(actual, dict) or isinstance(actual.get('time'), (bool, np.bool_))
                or not np.isfinite(actual.get('time', np.nan))
                or abs(float(actual['time'])-self.anchor_time) > 1e-9):
            raise ValueError('fresh measured state and Forecast must have the same anchor timestamp')
        if not isinstance(actual.get('held'), (bool, np.bool_)):
            raise ValueError('explicit current anchor held boolean required')
        if bool(actual['held']):
            if self.public_bridge_verified:
                raise ValueError('FREE public Forecast cannot initialize an attached public anchor')
            return
        pose, velocity, angular = self.sample(self.anchor_time)
        current = Pose.from_xyzw(np.r_[actual['object_position'], actual['object_quaternion']])
        errors = dict(position_m=float(np.linalg.norm(current.position-pose.position)),
            orientation_rad=float(Rotation.from_matrix(current.rotation.T@pose.rotation).magnitude()),
            linear_velocity_m_per_s=float(np.linalg.norm(np.asarray(actual['object_linear_velocity'])-velocity)),
            angular_velocity_rad_per_s=float(np.linalg.norm(np.asarray(actual['object_angular_velocity'])-angular)))
        self.last_anchor_errors = errors
        if any(not np.isfinite(value) or value > ANCHOR_QUANTIZATION_TOLERANCE[key]
               for key, value in errors.items()):
            raise ValueError('Forecast index0 must match the current object anchor within declared float32 quantization tolerance')

    def initialize_batch(self, actual, batch_size):
        self.validate_anchor(actual)
        x, h = self.execution_model.initialize_batch(deepcopy(actual), batch_size)
        h = deepcopy(h)
        h[_BINDING_KEY] = self.binding_id
        h['_forecast_anchor_held'] = np.asarray(x['held'], dtype=bool).copy()
        if not np.allclose(x['time'], self.anchor_time, atol=1e-9, rtol=0):
            raise ValueError('internal initialize_batch changed the anchor clock')
        return x, h

    def step_batch(self, x, u, h, dt=.1):
        if dt != .1 or not isinstance(h, dict) or h.get(_BINDING_KEY) != self.binding_id:
            raise ValueError('two-tick macrostep with this current Forecast binding required; no stale virtual h')
        times = np.asarray(x['time'], dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError('finite batched current timestamps required')
        if np.any(times < self.anchor_time-1e-9):
            raise ValueError('virtual state precedes current Forecast anchor')
        # Check coverage before calling a model; invalid horizons cannot make
        # a partial plan or a clamped last point look valid.
        for at in np.unique(times+dt):
            self.sample(at)
        nxt, next_h, info = self.execution_model.step_batch(deepcopy(x), deepcopy(u), deepcopy(h), dt)
        validate_step_batch(x, u, nxt, next_h, info, dt)
        anchor_held = np.asarray(h.get('_forecast_anchor_held'))
        if anchor_held.shape != times.shape or anchor_held.dtype != np.bool_:
            raise ValueError('explicit immutable anchor held mask required')
        free = (~anchor_held) & (~np.asarray(nxt['held'], dtype=bool))
        nxt = {key: value.copy() for key, value in nxt.items()}
        for at in np.unique(nxt['time'][free]):
            mask = free & np.isclose(nxt['time'], at, atol=1e-12, rtol=0)
            pose, velocity, angular = self.sample(at)
            nxt['object_position'][mask] = pose.position
            nxt['object_quaternion'][mask] = pose.xyzw()[3:]
            nxt['object_linear_velocity'][mask] = velocity
            nxt['object_angular_velocity'][mask] = angular
        next_h = deepcopy(next_h)
        next_h[_BINDING_KEY] = self.binding_id
        next_h['_forecast_anchor_held'] = anchor_held.copy()
        info = deepcopy(info)
        info.update(current_forecast=self.receipt(), free_object_endpoint_overridden=True,
            forecast_override_rows=int(free.sum()), anchor_held_rows_internal_model=int(anchor_held.sum()),
            anchor_held_release_not_replaced_with_precontact_GRU=True)
        return nxt, next_h, info

    def receipt(self):
        return dict(binding_id=self.binding_id, anchor_time=self.anchor_time, object_id=self.object_id,
            prediction_end_time=float(self.forecast.times[-1]), current_plus_future_points=17,
            real_sensor_frames_created=0, evaluation_labels_accepted=False,
            public_bridge_contract_verified=self.public_bridge_verified,
            anchor_quantization_tolerance=deepcopy(ANCHOR_QUANTIZATION_TOLERANCE),
            last_anchor_errors=deepcopy(self.last_anchor_errors),
            orientation_observable=self.orientation_observable,
            rotation_source='GRU_model_prediction_with_initial_axis_prior_not_visual_orientation',
            held_source=self.held_source, diagnostic_only=True, public_closed_loop_qualified=False)


class ForecastJointGoalCEMPlanner(JointGoalCEMPlanner):
    """Same joint CEM; GRU absolute-deadline nominals and explicit warm basis.

    Construct a fresh instance/binding for every real plan. Warm control
    proposals may be reused, but never old forecasts, model states or virtual h.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.execution_model, BoundForecastExecutionModel):
            raise TypeError('current BoundForecastExecutionModel required')
        self._plan_started = False

    def _prepare_nominals(self, initial, candidates, old):
        self.execution_model.validate_anchor(initial)
        started = time.perf_counter()
        positions, quaternions, times = [], [], []
        deadlines=np.asarray([row['deadline'] for row in candidates],dtype=float)
        if not np.isfinite(deadlines).all():
            raise ValueError('finite fixed candidate deadlines required')
        deadline_valid=(deadlines>=self.execution_model.anchor_time-1e-9)&(deadlines<=self.execution_model.forecast.times[-1]+1e-9)
        deadline_reasons=np.where(deadlines<self.execution_model.anchor_time-1e-9,
            'expired_before_current_Forecast_anchor',np.where(deadline_valid,'in_domain','beyond_current_Forecast_horizon'))
        for kind in range(3):
            for index,candidate in enumerate(candidates):
                at = float(candidate['deadline'])
                if kind < 2:
                    # Invalid candidates retain their original deadline and K8
                    # slot. Finite anchor geometry is only a numeric placeholder
                    # for DLS/batch shape; every phase of that mode is masked.
                    pose = (self.execution_model.sample(at)[0] if deadline_valid[index] else
                        Pose.from_xyzw(np.r_[initial['object_position'], initial['object_quaternion']]))
                    offset = candidate['object_tcp_position'].copy()
                    if kind == 0:
                        offset += candidate['approach_direction']*candidate['pregrasp_offset']
                    p = pose.apply(offset)
                    q = (pose.rotation @ Pose.from_xyzw(np.r_[np.zeros(3), candidate['object_tcp_quaternion']]).rotation)
                    q = matrix_quaternion(q)
                else:
                    goal = old.get('lift_goal_position', initial.get('lift_goal_position',
                        initial['object_position']+np.array([0., 0., self.config.lift_height])))
                    current = Pose.from_xyzw(np.r_[initial['object_position'], initial['object_quaternion']])
                    p = goal + current.rotation @ candidate['object_tcp_position']
                    q = old.get('hold_tcp_quaternion', initial['tcp_quaternion'])
                    at = initial['time']
                positions.append(p)
                quaternions.append(q)
                times.append(at)
        poses = np.c_[positions, quaternions]
        seeds = np.broadcast_to(initial['q'], (24, 7)).copy()
        result = self.goal_helper.adapter.solve(seeds, np.asarray(positions), np.asarray(quaternions))
        goals, feasible = np.asarray(result['joint_goal']), np.asarray(result['joint_goal_feasible'])
        if goals.shape != (24, 7) or feasible.shape != (24,) or feasible.dtype != np.bool_ or not np.isfinite(goals).all():
            raise ValueError('24 finite nominal joint goals and explicit feasibility mask required')
        feasible=feasible.reshape(3,8)&deadline_valid[None,:]
        self._nominals = dict(joint_goal=goals.reshape(3, 8, 7), feasible=feasible.reshape(3, 8),
            pose=poses.reshape(3, 8, 7), time=np.asarray(times).reshape(3, 8),
            deadline_valid=deadline_valid.copy(),deadline_invalid_reason=deadline_reasons.copy())
        self.nominal_receipt = dict(DLS_calls=1, poses=24, pregrasp_grasp_lift_per_candidate=3,
            feasible_by_kind=self._nominals['feasible'].copy(), wall_seconds=time.perf_counter()-started,
            deadlines=[row['deadline'] for row in candidates], basis_version=GRU_NOMINAL_BASIS,
            deadline_valid=deadline_valid.copy(),deadline_invalid_reason=deadline_reasons.copy(),
            invalid_deadline_numeric_placeholder='current_anchor_geometry_all_phases_infeasible_original_deadline_preserved',
            object_prediction='current_GRU_Forecast_fixed_absolute_deadline',
            forecast=self.execution_model.receipt(), frozen_current_project_modified=False)

    def plan(self, state, candidates, *, warm_start=None):
        if self._plan_started:
            raise ValueError('bind a fresh current Forecast and fresh wrapper for every real plan')
        self.execution_model.validate_anchor(state)
        warm = deepcopy(warm_start or {})
        expired_warm=False
        if warm:
            if warm.get('nominal_basis_version') != GRU_NOMINAL_BASIS:
                raise ValueError('explicit GRU nominal-basis warm required; CV/unknown basis forbidden')
            _reject_prediction_payload(warm)
            if float(warm['time'])>float(state['time'])+1e-9:
                raise ValueError('future warm feedback')
            selected=next((row for row in candidates if row['id']==warm.get('selected_candidate_id')),None)
            if selected is not None and (float(selected['deadline'])<self.execution_model.anchor_time-1e-9
                    or float(selected['deadline'])>self.execution_model.forecast.times[-1]+1e-9):
                # Base CEM's reserved absolute warm row bypasses nominal IK
                # feasibility. Discard it explicitly when its deadline is no
                # longer covered rather than allowing that bypass to revive it.
                warm={};expired_warm=True
            # Private compatibility translation for inherited CEM only. All
            # published inputs/outputs retain the actual GRU basis identity.
            if warm:
                warm['nominal_basis_version'] = NOMINAL_BASIS_VERSION
                warm.pop('forecast_anchor_time', None)
        self._plan_started = True
        result = super().plan(state, candidates, warm_start=warm)
        if result.get('warm_start') is not None:
            result['warm_start']['nominal_basis_version'] = GRU_NOMINAL_BASIS
            result['warm_start']['forecast_anchor_time'] = self.execution_model.anchor_time
        result.setdefault('diagnostics', {}).update(current_forecast=self.execution_model.receipt(),
            warm_forecast_carried=False, phase_and_cost_from_current_predicted_endpoint=True,
            expired_selected_warm_discarded=expired_warm,
            endpoint_tcp_reference_source='inherited_one_macrostep_CV_from_current_GRU_endpoint_diagnostic_only',
            actual_action_parameterization='joint_goal_not_endpoint_tcp_reference',
            public_closed_loop_qualified=False)
        return result


def _reject_prediction_payload(value):
    if isinstance(value, Forecast):
        raise ValueError('warm controls cannot carry an old Forecast')
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('forecast', 'h', 'virtual_h', 'execution_h', 'actuator_state', 'predicted_states', 'actual_state'):
                raise ValueError('warm controls cannot carry a virtual/old state or Forecast')
            _reject_prediction_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_prediction_payload(item)
