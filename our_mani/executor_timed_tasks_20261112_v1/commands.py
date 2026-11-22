from dataclasses import dataclass
import math
from ..executor_contract_repair_20261112_v1.pose_executor import WorldPoseGoal

@dataclass(frozen=True)
class TimedWorldTarget:
    pose: WorldPoseGoal
    lookahead_seconds: float
    timing_source: str

    def __post_init__(self):
        if type(self.pose)is not WorldPoseGoal:raise TypeError('Typed world pose required')
        if not math.isfinite(self.lookahead_seconds)or not 0<=self.lookahead_seconds<=2.:
            raise ValueError('Explicit finite 0..2 s target sample lookahead required')
        if not self.timing_source:raise ValueError('Timing provenance required')
