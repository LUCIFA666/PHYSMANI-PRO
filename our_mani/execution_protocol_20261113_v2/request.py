"""Versioned action envelope; the original decoded pose8 payload is preserved."""
from dataclasses import dataclass
from math import isfinite,sqrt

@dataclass(frozen=True)
class ExecutionRequest:
    pose8: tuple
    observation_time: float
    target_semantics: str='destination'
    target_time_offset_seconds: object=None
    stage_id: object=None
    completion: str='continuous'
    metadata_origin: str='explicit_caller'
    release_policy: str='driver_default'

    def __post_init__(self):
        values=tuple(float(x)for x in self.pose8)
        if len(values)!=8 or not all(map(isfinite,values)):raise ValueError('Finite decoded pose8 required')
        if abs(sqrt(sum(x*x for x in values[3:7]))-1.)>1e-5:raise ValueError('Unit xyzw quaternion required')
        if values[7]not in (0.,1.):raise ValueError('Decoded opening must be binary, as in the existing pose driver')
        if isinstance(self.observation_time,bool)or not isfinite(self.observation_time)or self.observation_time<0:
            raise ValueError('Actual nonnegative observation time required')
        if self.target_semantics not in ('destination','timed_sample'):raise ValueError('Explicit target semantics required')
        offset=self.target_time_offset_seconds
        if self.target_semantics=='destination':
            if offset is not None:raise ValueError('A destination has no implied target sample time')
        elif offset is None or isinstance(offset,bool)or not isfinite(offset)or not 0<=offset<=2.:
            raise ValueError('A timed sample requires explicit finite 0..2 s lookahead')
        if self.stage_id is not None and(not isinstance(self.stage_id,str)or not self.stage_id):
            raise ValueError('Stage identifier must be absent or a nonempty string')
        if self.completion not in ('continuous','static_pose','gripper'):
            raise ValueError('Supported completion: continuous, static_pose, gripper')
        if self.completion!='continuous'and self.stage_id is None:raise ValueError('Completion ACK requires a stage identifier')
        if self.target_semantics=='destination'and self.completion!='continuous':
            raise ValueError('Legacy destination routing does not implement new stage completion ACKs')
        if self.completion=='static_pose'and not(self.target_semantics=='timed_sample'and offset==0.):
            raise ValueError('Static completion cannot be claimed for a moving forecast')
        if not isinstance(self.metadata_origin,str)or not self.metadata_origin:raise ValueError('Metadata provenance required')
        if self.release_policy not in ('driver_default','terminal_pose','caller_intent','position_window','aligned_drop'):
            raise ValueError('Explicit supported release policy required')
        if self.target_semantics=='destination'and self.release_policy!='driver_default':
            raise ValueError('Unchanged legacy routing uses its existing release behavior')
        object.__setattr__(self,'pose8',values)
        object.__setattr__(self,'observation_time',float(self.observation_time))

    @classmethod
    def legacy(cls,pose8,observation_time,*,explicit_lookahead_seconds=None):
        return cls(pose8,observation_time,
            target_semantics='destination'if explicit_lookahead_seconds is None else'timed_sample',
            target_time_offset_seconds=explicit_lookahead_seconds,metadata_origin='legacy_pose8_explicit_profile')

    def world_goal(self):
        from ..executor_contract_repair_20261112_v1.pose_executor import WorldPoseGoal
        return WorldPoseGoal(self.observation_time,self.pose8[:3],self.pose8[3:7],self.pose8[7])

    def timed_target(self):
        if self.target_semantics!='timed_sample':raise ValueError('Destination is not a timed trajectory sample')
        from ..executor_timed_tasks_20261112_v1.commands import TimedWorldTarget
        return TimedWorldTarget(self.world_goal(),self.target_time_offset_seconds,self.metadata_origin)

    def to_wire(self):
        return dict(schema_version=2,pose8=list(self.pose8),observation_time=self.observation_time,
            target_semantics=self.target_semantics,target_time_offset_seconds=self.target_time_offset_seconds,
            stage_id=self.stage_id,completion=self.completion,metadata_origin=self.metadata_origin,
            release_policy=self.release_policy)

    @classmethod
    def decode(cls,value,*,observation_time=None,legacy_lookahead_seconds=None):
        if isinstance(value,cls):return value
        if isinstance(value,dict):
            fields=dict(value)
            if fields.pop('schema_version',None)!=2:raise ValueError('Explicit schema_version=2 required')
            return cls(**fields)
        if observation_time is None:raise ValueError('Legacy pose8 needs an actual observation timestamp')
        return cls.legacy(value,observation_time,explicit_lookahead_seconds=legacy_lookahead_seconds)
