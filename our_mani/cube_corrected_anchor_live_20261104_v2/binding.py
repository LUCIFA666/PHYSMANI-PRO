"""FREE state derivative ABI uses the forecaster's current public OLS anchor.

The tracker velocity remains an independent diagnostic. No current measured
position/orientation/cloud, predicted future, Sphere or anchor tolerance changes.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Forecast,Pose,ObjectState
from ..cube_corrected_live_20261104_v1.binding import CorrectedPublicStateAssembler
from ..local_mpc_control_20261103.forecast_control import ANCHOR_QUANTIZATION_TOLERANCE


def _bind_free_anchor_derivatives(state,result,*,expected_sha256,now):
    """Pure arithmetic after original same-frame public capability validation."""
    if state.get('grasp_motion_mode')!='FREE_GRU':return state,None
    if (result.get('mode')!='FREE' or result.get('available')is not True
            or result.get('free_forecast_available')is not True or state.get('held')is not False
            or any(result.get(k)is not False for k in ('Native_inputs_used','future_observations_used','observations_fabricated'))):
        raise ValueError('Original current public FREE state required')
    if abs(float(state['time'])-now)>1e-9 or abs(float(result['timestamp'])-now)>1e-9:
        raise ValueError('Same exact current state/result/robot time required')
    forecast=result.get('forecast');obj=result.get('object_observation')
    if type(forecast)is not Forecast or type(obj)is not ObjectState or not obj.visible or abs(obj.timestamp-now)>1e-9:
        raise TypeError('Exact current public ObjectState and original Forecast required')
    if (result.get('frozen_checkpoint_sha256')!=expected_sha256
            or forecast.metadata.get('checkpoint_sha256')!=expected_sha256
            or abs(float(forecast.times[0])-now)>1e-9
            or abs(float(forecast.metadata.get('observed_until',-1))-now)>1e-9):
        raise ValueError('Same current frozen forecast identity/index0 anchor required')
    current=Pose.from_xyzw(np.r_[state['object_position'],state['object_quaternion']])
    if (not np.array_equal(current.position,obj.pose.position)
            or np.linalg.norm(Rotation.from_matrix(current.rotation.T@obj.pose.rotation).as_rotvec())>1e-9):
        raise ValueError('Measured current object pose must remain unchanged')
    pose,velocity,angular=forecast.at('target',now)
    errors=dict(position_m=float(np.linalg.norm(current.position-pose.position)),
        orientation_rad=float(Rotation.from_matrix(current.rotation.T@pose.rotation).magnitude()))
    if any(not np.isfinite(value)or value>ANCHOR_QUANTIZATION_TOLERANCE[key]for key,value in errors.items()):
        raise ValueError('Current measured pose/forecast index0 disagrees beyond original quantization tolerance')
    before_v=np.asarray(state['object_linear_velocity']).copy()
    before_w=np.asarray(state['object_angular_velocity']).copy()
    updated=dict(state,object_linear_velocity=velocity.copy(),object_angular_velocity=angular.copy())
    trace=dict(timestamp=now,forecast_checkpoint_sha256=expected_sha256,
        current_derivative_source='same_frozen_Forecast_index0_causal_public_last5_OLS',
        raw_tracker_linear_velocity=before_v.tolist(),raw_tracker_angular_velocity=before_w.tolist(),
        bound_linear_velocity=velocity.tolist(),bound_angular_velocity=angular.tolist(),
        tracker_vs_forecast_linear_velocity_delta_m_per_s=float(np.linalg.norm(before_v-velocity)),
        tracker_vs_forecast_angular_velocity_delta_rad_per_s=float(np.linalg.norm(before_w-angular)),
        current_pose_forecast_quantization_error=errors,measured_pose_or_cloud_changed=False,
        forecast_samples_or_future_learned_outputs_changed=False,anchor_tolerance_changed=False,
        derivative_is_public_estimate_not_Native_measurement=True)
    return updated,trace


class AnchorConsistentPublicStateAssemblerV2(CorrectedPublicStateAssembler):
    def __init__(self,binding):
        super().__init__(binding)
        self.last_derivative_binding=None

    def build(self,result,robot,actuator_state,*,phase_state=None):
        # Includes exact binding.approve(result), exact observed robot/q/dq,
        # current h, public provenance, real robot FD and original pose gates.
        state=super().build(result,robot,actuator_state,phase_state=phase_state)
        self.last_derivative_binding=None
        if state is None or state.get('grasp_motion_mode')!='FREE_GRU':return state
        forecast=result['forecast']
        state,trace=_bind_free_anchor_derivatives(state,result,
            expected_sha256=self.binding.bridge.expected_forecast_sha256,now=float(robot.timestamp))
        if result is not self.binding.latest_result or result['forecast']is not forecast:
            raise ValueError('Detached result or replaced forecast during current assembly')
        self.last_derivative_binding=trace
        return state

    def receipt(self):
        return dict(protocol='same_current_public_forecast_derivative_anchor_v2',
            FREE_derivatives='forecaster_current_causal_OLS_last5_public_positions',
            raw_tracker_derivative_retained_as_diagnostic=True,
            measured_current_pose_cloud_Sphere_unchanged=True,learned_future_unchanged=True,
            original_anchor_tolerance_unchanged=True,held_pending_probe_assembly_unchanged=True,
            last_derivative_binding=deepcopy(self.last_derivative_binding))
