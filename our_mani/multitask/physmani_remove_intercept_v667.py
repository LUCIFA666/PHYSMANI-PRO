"""Compensate the shared ten-tick RML acquisition request delay for remove.

The author expert's zero-delay acquisition goal is queried per expert update;
our pose action holds one request for ten counted ticks. Aim at the same TRAIN
canonical grasp at the final request tick, using only the current public CV or
Context forecast. Actual close still requires current2cm/0.1rad geometry.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .interaction import COMMAND_GOAL,_reference_at
from .physmani_geometry_goals_v593 import pose_goals
from .physmani_remove_source_roles_v653 import AuthorLiveRoleRemoveIntents


def binding(root,task):
    if task not in ('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed'):
        raise ValueError('two explicit rotating-cup acquisition tasks')
    return dict(schema='r10_public_remove_canonical_intercept_at_RML_request_tick_v667',task=task,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_remove_intercept_v667.py'),
        original_expert_acquisition_delay_seconds=0.,RML_request_ticks=10,counted_dt=.05,
        acquisition_prediction_delay_seconds=.5,current_close_position_gate_m=.02,current_close_angle_gate_rad=.1,
        original_TRAIN_canonical_geometry_unchanged=True,original_RML_and_gripper_request_timing_unchanged=True,
        future_observation_or_native_outcome_used=False,actual_interception_or_task_success_guaranteed=False)


class InterceptRemoveIntents(AuthorLiveRoleRemoveIntents):
    def __init__(self,*a,intercept_rule,**kw):
        super().__init__(*a,**kw)
        if (intercept_rule.get('task')!=self.task
            or intercept_rule.get('schema')!='r10_public_remove_canonical_intercept_at_RML_request_tick_v667'
            or intercept_rule.get('acquisition_prediction_delay_seconds')!=.5):
            raise ValueError('explicit common-RML request delay compensation required')
        self.intercept_rule=deepcopy(intercept_rule);self.intercept_events=[]

    def candidates(self,phase):
        if phase!='acquire':return super().candidates(phase)
        f=self.frontend;s=f.state;result=[]
        for row in pose_goals(f,phase,origin=self.origin,lookahead=.5,committed=self.committed):
            t=row['template']
            if t.reference_time_basis!=COMMAND_GOAL:raise ValueError('original canonical grasp command geometry required')
            local=Pose.from_xyzw(t.completion_geometry_xyzw)
            ref,_,_=_reference_at(f.forecast,row['reference_id'],t,row['predicted_reference_time'],self.origin)
            target=Pose(ref.apply(local.position),ref.rotation@local.rotation)
            result.append(dict(row,pose=target,opening=0.,
                score=-float(np.linalg.norm(target.position-s.robot.pose.position))))
            self.intercept_events.append(dict(time=float(s.robot.timestamp),reference_id=row['reference_id'],
                predicted_reference_time=float(row['predicted_reference_time']),TCP=s.robot.pose.xyzw().tolist(),
                future_canonical_grasp_intent=target.xyzw().tolist(),actual_current_close_gate_unchanged=True,
                source_phase_epoch=self.phase_epoch,native_outcome_or_future_observation_used=False))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def receipt(self):return dict(super().receipt(),public_RML_request_intercept=dict(
        binding=deepcopy(self.intercept_rule),events=deepcopy(self.intercept_events)))
