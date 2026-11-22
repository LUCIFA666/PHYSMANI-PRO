"""Keep the author arm executor; qualify only our physical gripper requests."""
from .physmani_geometry_executor_v517 import PhysManiGeometryExecutor as OriginalGeometry
from ..sensor_alignment import physmani_executor_v6 as direct


class PhysManiSkillExecutor(OriginalGeometry):
    def __init__(self,*args,skill_policy,**kwargs):
        super().__init__(*args,**kwargs);self.skill_policy=skill_policy

    def execute(self,env,action,*,observer=None):
        # The original adapter already uses this reentrant lock to own its
        # direct-command scope. Restore the helper even if a native tick fails.
        with direct._reference_binding_lock:
            original=direct._apply_direct_target
            def apply(owned,joints,*,opening,phase):
                if owned is not env:raise direct.ProtocolInvariantError('gripper scope escaped owned environment')
                qualified,proof,proposed=self.skill_policy.qualify_opening(opening,env._requested_opening)
                result=original(owned,joints,opening=qualified,phase=phase)
                self.skill_policy.confirm_opening(qualified,proof,proposed,confirmed_after=float(env.time))
                return result
            direct._apply_direct_target=apply
            try:return super().execute(env,action,observer=observer)
            finally:direct._apply_direct_target=original
