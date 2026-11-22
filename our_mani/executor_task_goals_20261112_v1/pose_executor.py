from ..executor_contract_repair_20261112_v5.pose_executor import PoseGoalExecutor as PhysicalExecutor,WorldPoseGoal
from ..executor_contract_repair_20261112_v6.reach import outer_reach

class PoseGoalExecutor(PhysicalExecutor):
    def _decode(self,action):
        bound=outer_reach(self.clearance.snapshot,action.position)
        if not bound['admissible']:
            self.last_clearance_diagnostic=dict(accepted=False,alternative_IK=False,checks=[dict(
                source='outer_robot_reach_precheck',reason='outside_outer_robot_reach',**bound)])
            self.backend.last_ik_receipt=dict(success=False,reason='outside_outer_robot_reach',Native_IK_called=False,physics_steps=0)
            return None
        return super()._decode(action)
