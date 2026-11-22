"""Explicit new forecast capability; all V70 measurement checks retained."""
from pathlib import Path
import hashlib
from ..local_mpc_control_20261103.goal_aware_pixel_centre_binding_v70 import *
from ..local_mpc_control_20261103.goal_aware_pixel_centre_binding_v70 import _file_stat,_PIXEL_CENTRE_BACKPROJECT
from .motion_bridge import CorrectedPublicMotionBridge,CubeResidualMotionDynamics

class ApprovedCorrectedTrackerBinding(ApprovedDepthPixelCentreTrackerCacheBindingV70):
    def validate_dependencies(self):
        if (type(self.bridge) is not CorrectedPublicMotionBridge or type(self.tracker) is not CurrentDepthPixelCentreTrackerV70
            or type(self.tracker.tracker) is not RedBoxTracker or type(self.tracker.marker) is not RedGoalMarker
            or type(self.cache) is not PublicTargetFrameCache or self.bridge.tracker is not self.cache
            or self.cache.public_tracker is not self.tracker):
            raise ValueError('Only the exact real V70 tracker/model and same-frame cache capability is accepted')
        t=self.tracker;inner=t.tracker
        if (t.prior is not inner.prior or type(inner.prior) is not BoxPrior or asdict(inner.prior)!=asdict(BoxPrior(minimum_red=100))
            or inner.object_id!='target' or inner.history.maxlen!=5 or inner.complete_geometry is not False or inner.points!=256
            or type(t.config) is not GoalSphereFilterConfig or asdict(t.config)!=asdict(GoalSphereFilterConfig())
            or getattr(t.observe,'__func__',None) is not CurrentDepthPixelCentreTrackerV70.observe
            or getattr(t.validate_refiner,'__func__',None) is not CurrentDepthPixelCentreTrackerV70.validate_refiner
            or getattr(t.model_sources,'__func__',None) is not CurrentDepthPixelCentreTrackerV70.model_sources
            or getattr(t.public_position_evidence,'__func__',None) is not CurrentDepthPixelCentreTrackerV70.public_position_evidence
            or getattr(t._private_far_observation,'__func__',None) is not CurrentDepthPixelCentreTrackerV70._private_far_observation
            or PixelCentreFarCalibration.__bases__!=(CameraCalibration,)
            or PixelCentreFarCalibration.backproject is not _PIXEL_CENTRE_BACKPROJECT
            or getattr(inner.observe,'__func__',None) is not RedBoxTracker.observe
            or getattr(t.marker.observe,'__func__',None) is not RedGoalMarker.observe):
            raise ValueError('Exact R100 measured-component/knownshape/current real methods required')
        t.validate_refiner()
        offset=np.asarray(json.loads(GOAL_CALIBRATION.read_text())['offset_train_calibrated_m'])
        if (not np.array_equal(t.marker.offset,offset) or not np.array_equal(self.bridge.goal_marker.offset,offset)
            or self.bridge._mock_forecaster is not False or self.bridge._goal_marker_injected is not False
            or self.bridge._tracker_injected is not True or type(self.bridge.goal_marker) is not RedGoalMarker
            or type(self.bridge.forecaster) is not CubeResidualMotionDynamics
            or self.bridge.forecaster.checkpoint_sha256!=self.bridge.expected_forecast_sha256):
            raise ValueError('Original frozen real GRU/default public marker required, no forecaster/goal injection')
        for relative,expected in self._approved_stats.items():
            if _file_stat(ROOT/relative)!=expected:
                raise ValueError('V70 dependency changed after full constructor SHA check: '+relative)
        if self._sphere_object is not None and (t.sphere is not self._sphere_object or digest(asdict(t.sphere))!=self._sphere_digest):
            raise ValueError('First real public Sphere cannot be replaced or silently refitted')
        self.bridge.forecaster._assert_live()

    def receipt(self):
        value=super().receipt()
        value.update(binding_version='corrected_GRU_original_V70_measurement_binding_v1',
            forecast_family='Cube_public_real50ms_CV_GRU_residual_20261104_v1',
            forecast_checkpoint_sha256=self.bridge.expected_forecast_sha256,
            legacy_ebec_authority_used=False,original_current_measurement_checks_retained=True,
            binding_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return value

class CorrectedPublicStateAssembler(PublicGraspStateAssembler):
    def __init__(self,binding):
        PublicStateAssembler.__init__(self)
        if type(binding)is not ApprovedCorrectedTrackerBinding:
            raise TypeError('Exact corrected forecast/current measurement capability required')
        binding.validate_dependencies();self.binding=binding
