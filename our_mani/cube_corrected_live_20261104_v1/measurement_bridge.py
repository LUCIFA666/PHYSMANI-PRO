"""Original V70 current measurements/probe/maintenance, new typed free forecast."""
from ..local_mpc_control_20261103.public_candidate_pixel_centre_measurement_v70 import *
from .motion_bridge import CorrectedPublicMotionBridge
from .binding import ApprovedCorrectedTrackerBinding

class CorrectedPixelPrimitive(PixelCentreGoalAwareActiveVisualPublicBridgeV70):
    def __init__(self,*,refiner_checkpoint,refiner_checkpoint_sha256,forecast_checkpoint,forecast_checkpoint_sha256,active_config=None):
        if type(forecast_checkpoint_sha256)is not str or len(forecast_checkpoint_sha256)!=64:
            raise ValueError('Explicit new forecast authority required')
        torch.set_num_threads(2)
        self.tracker=CurrentDepthPixelCentreTrackerV70(refiner_checkpoint,refiner_checkpoint_sha256)
        self.cache=PublicTargetFrameCache(self.tracker)
        self.bridge=CorrectedPublicMotionBridge(forecast_checkpoint,expected_forecast_sha256=forecast_checkpoint_sha256,tracker=self.cache)
        self.estimator=PublicAttachmentEstimator(AttachmentConfig());self.active=ActiveVisualAttachment(active_config)
        self.binding=ApprovedCorrectedTrackerBinding(self.bridge,self.tracker,self.cache)
        self.latest_attachment=self.latest_legacy_diagnostic=self.latest_result=None
        self._last_open_visual=None;self._last_close_requested=None;self._last_time=None;self.close_visual_baseline=None

class CorrectedPixelMeasurementBridge(PublicCandidatePixelCentreMeasurementBridgeV70):
    def __init__(self,*,refiner_checkpoint,refiner_checkpoint_sha256,forecast_checkpoint,forecast_checkpoint_sha256,
                 goal_sphere_filter=True,postconfirmed_maintenance=True,active_config=None):
        v70_enabled(goal_sphere_filter=goal_sphere_filter,postconfirmed_maintenance=postconfirmed_maintenance)
        self.goal_sphere_filter=goal_sphere_filter;self.postconfirmed_maintenance=postconfirmed_maintenance
        self._configured_flags=(goal_sphere_filter,postconfirmed_maintenance)
        cls=CorrectedPixelPrimitive
        self._configured_refiner=(Path(refiner_checkpoint).resolve(strict=True),refiner_checkpoint_sha256)
        self.delegate=cls(refiner_checkpoint=refiner_checkpoint,
            refiner_checkpoint_sha256=refiner_checkpoint_sha256,active_config=active_config,forecast_checkpoint=forecast_checkpoint,forecast_checkpoint_sha256=forecast_checkpoint_sha256)
        # These exact primitive objects remain covered by the existing binding.
        for name in ('tracker','cache','bridge','binding','estimator','active'):setattr(self,name,getattr(self.delegate,name))
        self.maintenance=PostConfirmedPositionMaintenance();self.retired_maintenance=[]
        self.latest_attachment=self.latest_raw_active=self.latest_result=None
        self.last_initial_admission=None
        self._validate_components()

    def _validate_components(self):
        if (self.goal_sphere_filter,self.postconfirmed_maintenance)!=self._configured_flags:
            raise ValueError('feature modifiers cannot change within an episode')
        expected=CorrectedPixelPrimitive
        if type(self.delegate) is not expected or type(self.active) is not ActiveVisualAttachment:
            raise TypeError('exact original probe owner and selected public producer required')
        for name in ('tracker','cache','bridge','binding','estimator','active'):
            if getattr(self,name) is not getattr(self.delegate,name):
                raise ValueError('outer public producer cannot detach from its authenticated '+name)
        if (self.binding.tracker is not self.tracker or self.binding.cache is not self.cache
                or self.binding.bridge is not self.bridge or type(self.maintenance) is not PostConfirmedPositionMaintenance):
            raise ValueError('authenticated actual producer/cache and exact maintenance owner required')
        if (self.tracker.refiner_checkpoint,self.tracker.expected_refiner_sha256)!=self._configured_refiner:
            raise ValueError('configured refiner checkpoint identity cannot change within an episode')
        self.tracker.validate_refiner()
        self.binding.validate_dependencies()

    def receipt(self):
        result=super().receipt()
        result.update(forecast_family='Cube_public_real50ms_CV_GRU_residual_20261104_v1',
            forecast_checkpoint_sha256=self.bridge.expected_forecast_sha256,
            original_V70_observe_track_probe_maintenance_methods_reused=True)
        return result
