"""New real forecaster constructor; original observer/reset/attachment math."""
from pathlib import Path
import json
import numpy as np
from ..benchmark_perception import RedBoxTracker
from ..goal_perception import RedGoalMarker
from ..local_mpc_control_20261103.public_motion_bridge import PublicMotionBridge,GOAL_CALIBRATION
from ..cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics

class CorrectedPublicMotionBridge(PublicMotionBridge):
    def __init__(self,forecast_checkpoint,*,expected_forecast_sha256,tracker,goal_calibration=GOAL_CALIBRATION,minimum_confidence=.5):
        if type(expected_forecast_sha256)is not str or len(expected_forecast_sha256)!=64:
            raise ValueError('Explicit newly fitted forecast SHA required before any model path access')
        if not np.isfinite(minimum_confidence) or not 0<minimum_confidence<=1:
            raise ValueError('Predeclared current measurement confidence required')
        self.minimum_confidence=float(minimum_confidence)
        self._tracker_injected=True
        self._goal_marker_injected=False
        self.tracker=tracker
        self.goal_marker=RedGoalMarker(json.loads(Path(goal_calibration).read_text())['offset_train_calibrated_m'])
        self._mock_forecaster=False
        self.forecaster=CubeResidualMotionDynamics(forecast_checkpoint,expected_sha256=expected_forecast_sha256)
        self.expected_forecast_sha256=expected_forecast_sha256
        self.reset()
