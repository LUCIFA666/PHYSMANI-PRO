"""Same actual-frame public proof owners and observations; new constructor guards."""
from ..local_mpc_control_20261103.public_candidate_pixel_centre_reverification_v70 import *
from ..local_mpc_control_20261103.public_candidate_pixel_centre_reverification_v70 import _stat
from .measurement_bridge import CorrectedPixelMeasurementBridge
from .sources import corrected_live_source_map

class CorrectedReverificationBridge(PublicCandidatePixelCentreReverificationBridgeV70):
    def __init__(self,*,refiner_checkpoint,refiner_checkpoint_sha256,forecast_checkpoint,forecast_checkpoint_sha256,
                 goal_sphere_filter=True,postconfirmed_maintenance=True,active_config=None):
        self.delegate=CorrectedPixelMeasurementBridge(refiner_checkpoint=refiner_checkpoint,
            refiner_checkpoint_sha256=refiner_checkpoint_sha256,goal_sphere_filter=goal_sphere_filter,
            postconfirmed_maintenance=postconfirmed_maintenance,active_config=active_config,forecast_checkpoint=forecast_checkpoint,forecast_checkpoint_sha256=forecast_checkpoint_sha256)
        for name in ('tracker','cache','bridge','binding','estimator','active'):
            setattr(self,name,getattr(self.delegate,name))
        self.goal_sphere_filter=goal_sphere_filter;self.postconfirmed_maintenance=postconfirmed_maintenance
        sources=corrected_live_source_map();self._source_stats={p:_stat(p)for p in sources}
        self.witness=None;self.epoch_owner=None;self.used_attempt_ids=set();self.retired_reverification=[]
        self.latest_result=self.latest_attachment=self.latest_raw_active=None
        self._frame=None;self._inner_robot=None;self._last_close=None;self.last_admission=None
        self._validate_components()

    def _validate_components(self):
        if type(self.delegate) is not CorrectedPixelMeasurementBridge:
            raise TypeError('Exact unchanged V64 primitive producer required')
        self.delegate._validate_components()
        if (self.goal_sphere_filter,self.postconfirmed_maintenance)!=(True,True):
            raise ValueError('Frozen V64 measurement and original maintenance are required')
        for name in ('tracker','cache','bridge','binding','estimator','active'):
            if getattr(self,name)is not getattr(self.delegate,name):raise ValueError('Detached original primitive: '+name)
        if getattr(self.active.observe,'__func__',None)is not ActiveVisualAttachment.observe:
            raise ValueError('Original initial probe/attempt owner method required')
        if self.witness is not None and type(self.witness)is not PublicReverificationEpochPrototype:
            raise TypeError('Exact isolated natural witness required')
        if self.witness is not None:
            for name in ('observe','begin_witness','commit_verified_epoch'):
                if getattr(getattr(self.witness,name),'__func__',None)is not getattr(PublicReverificationEpochPrototype,name):
                    raise ValueError('Original natural witness method required: '+name)
        if self.epoch_owner is not None and type(self.epoch_owner)is not PublicReverifiedMaintenanceOwner:
            raise TypeError('Exact explicitly committed current owner required')
        if (self.epoch_owner is not None and getattr(self.epoch_owner.observe,'__func__',None)
                is not PublicReverifiedMaintenanceOwner.observe):
            raise ValueError('Original committed current maintenance method required')
        for name,value in self._source_stats.items():
            if _stat(name)!=value:raise ValueError('Natural re-verification source changed: '+name)

    def receipt(self):
        value=super().receipt()
        value.update(forecast_family='Cube_public_real50ms_CV_GRU_residual_20261104_v1',
            forecast_checkpoint_sha256=self.bridge.expected_forecast_sha256,
            corrected_source_sha256=corrected_live_source_map(),
            original_attempt_anchors_and_failed_owner_retained=True)
        return value
