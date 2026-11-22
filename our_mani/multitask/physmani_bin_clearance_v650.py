"""Public held-mesh clearance before RML carries rubbish over the bin rim.

Reuse the existing conservative TRAIN meshes and clearance math. Keep source
XY/orientation, the author5cm release distance, original RML and native outcome.
A vertical public-geometry transit precedes lateral transport; all motion still
passes the same fresh per-RML-target collision admission and global10s clock.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .physmani_bin_filter_progress_v631 import NormalBinProgressIntents,HighBinProgressIntents
from .physmani_geometry_goals_v593 import pose_goals,current_completion_pose
from .physmani_skill_policy_v548 import errors
from .placement_clearance_v50 import PlacementClearance,vertical_shift


def binding(root,config,task):
    if task not in ('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed'):
        raise ValueError('explicit two source bin tasks only')
    from .placement_clearance_v50 import binding as geometry_binding
    geometry=geometry_binding(root,config,task)
    if config.get('placement_clearance_v50',{}).get(task)!=geometry:raise ValueError('freeze existing public clearance geometry')
    return dict(schema='r10_public_bin_clearance_lift_then_lateral_original_RML_v650',task=task,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_bin_clearance_v650.py'),
        clearance_math_source_sha256=file_sha(Path(root)/'our_mani/multitask/placement_clearance_v50.py'),
        original_public_clearance_geometry=geometry,clearance_height_margin_m=.02,
        current_clearance_minimum_margin_m=.005,release_position_guard_m=.05,
        source_XY_and_orientation_retained=True,source_height_replaced_by_explicit_public_clearance=True,
        original_physical_acquire_and_RML_parameters_unchanged=True,native_success_unchanged=True,
        instantaneous_motion_or_stopped_release_guarantee_claimed=False)


class PublicClearance:
    def __init__(self,*a,root,config,clearance_rule,**kw):
        super().__init__(*a,**kw)
        if clearance_rule!=binding(root,config,self.task):raise ValueError('freeze explicit public bin clearance intent policy')
        self.clearance_rule=deepcopy(clearance_rule)
        self.clearance=PlacementClearance(self.frontend.actor,config,root,self.task)
        self.clearance_events=[];self.last_operational_goal=None;self.release_requested=False

    def _clearance_goal(self,goal,query=None):
        f=self.frontend;a=f.actor;state=f.state
        original=current_completion_pose(f,goal,origin=self.origin)
        if (not state.robot.contact or 'rubbish' not in a.bound_attachments
            or 'rubbish' not in a.attachments or self.close_witness is None):
            if self.release_requested and self.last_operational_goal is not None:
                # This is a previously measured operational intent, not a new
                # claim of attached-body geometry after public release.
                return self.last_operational_goal
            return None
        associated={k for k,v in state.perception_metadata.get('bodies',{}).items() if v.get('motion_mode')=='attached'}
        modes,_=a.collision_port._geometry_modes()
        if associated!={'rubbish'} or modes.get('rubbish')!='primary' or modes.get('bin')!='primary':return None
        now=float(state.robot.timestamp);query=now if query is None else float(query)
        bin_pose=f.forecast.at(goal['reference_id'],query)[0]
        # The trained bin intent has a translation-only public reference.
        # Forecast XY and source orientation stay unchanged at future queries.
        if query!=now:
            present=f.forecast.at(goal['reference_id'],now)[0]
            original=Pose(original.position+bin_pose.position-present.position,original.rotation)
        shift,proof=vertical_shift(bin_pose,self.clearance.bin_vertices,original,
            a.attachments['rubbish'].tool_to_body,self.clearance.held_vertices)
        minimum=float(original.position[2]+shift)
        safe=Pose(np.r_[original.position[:2],minimum+.02],original.rotation)
        return dict(pose=safe,minimum_clearance_TCP_z=minimum,geometry=proof,
            selected_at=now,reference_id=goal['reference_id'],source_current_canonical=original)

    def _completion_target(self,goal):
        if goal['phase']!='deposit':return super()._completion_target(goal)
        try:
            result=self._clearance_goal(goal)
            return None if result is None else result['pose']
        except (ValueError,KeyError,IndexError):return None

    def candidates(self,phase):
        if phase!='deposit':return super().candidates(phase)
        f=self.frontend;now=float(f.state.robot.timestamp);result=[]
        if self.release_requested and not f.state.robot.contact:return []
        for row in pose_goals(f,phase,origin=self.origin,lookahead=0.,committed=self.committed):
            current=self._clearance_goal(row)
            if current is None:continue
            near=errors(f.state.robot,current['pose'])[0]<=.05
            high=self.task.endswith('_high_speed')
            first=high and not self.first_deposit_intent_issued
            delay=.5 if near else 1. if high and not first else 0.
            ready_height=f.state.robot.pose.position[2]>=current['minimum_clearance_TCP_z']+.005
            for future_row in pose_goals(f,phase,origin=self.origin,lookahead=delay,committed=row):
                future=self._clearance_goal(future_row,future_row['predicted_reference_time'])
                if future is None:continue
                if ready_height:
                    target=future['pose'];substage='carry_above_complete_public_bin_mesh'
                else:
                    target=Pose(np.r_[f.state.robot.pose.position[:2],future['pose'].position[2]],future['pose'].rotation)
                    substage='raise_held_mesh_before_crossing_bin_rim'
                self.last_operational_goal=current
                result.append(dict(future_row,pose=target,opening=float(near and ready_height),
                    clearance_substage=substage,current_clearance=current,
                    score=-float(np.linalg.norm(target.position-f.state.robot.pose.position))))
                self.clearance_events.append(dict(time=now,phase=phase,substage=substage,
                    TCP=f.state.robot.pose.xyzw().tolist(),source_goal=current['source_current_canonical'].xyzw().tolist(),
                    public_clearance_goal=current['pose'].xyzw().tolist(),actual_intent=target.xyzw().tolist(),
                    minimum_clearance_TCP_z=current['minimum_clearance_TCP_z'],ready_height=bool(ready_height),
                    geometry=deepcopy(current['geometry']),phase_epoch=self.phase_epoch,global_clock_reset=False,
                    source_XY_and_orientation_unchanged=True,native_pose_or_outcome_used=False))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def qualify_opening(self,requested,current_requested):
        goal=self.selected
        if goal is None or goal['phase']!='deposit':return super().qualify_opening(requested,current_requested)
        state=self.frontend.state
        if self.release_requested or requested<.5:
            command=1. if self.release_requested else float(current_requested>=.5)
            return command,dict(time=float(state.robot.timestamp),phase='deposit',desired_opening=float(requested>=.5),
                current_requested_opening=float(current_requested),qualified_opening=command,
                reason='preserve_actual_bin_release_request' if self.release_requested else 'await_original_RML_open_request'),None
        current=self._clearance_goal(goal);distance=None
        if current is not None:distance=errors(state.robot,current['pose'])[0]
        accepted=(current is not None and distance<=.05
            and state.robot.pose.position[2]>=current['minimum_clearance_TCP_z']+.005)
        return float(accepted),dict(time=float(state.robot.timestamp),phase='deposit',desired_opening=1.,
            current_requested_opening=float(current_requested),qualified_opening=float(accepted),
            reason='current_public_complete_held_mesh_above_bin_release' if accepted else 'defer_release_until_public_bin_clearance',
            current_position_error_m=distance,operational_clearance_goal_replaces_source_height=True,
            original_RML_gripper_request_tick_retained=True,native_truth_read=False),None

    def confirm_opening(self,opening,proof,proposed,*,confirmed_after):
        super().confirm_opening(opening,proof,proposed,confirmed_after=confirmed_after)
        if self.selected is not None and self.selected['phase']=='deposit' and opening>=.5:
            self.release_requested=True

    def receipt(self):return dict(super().receipt(),public_bin_clearance=dict(
        binding=deepcopy(self.clearance_rule),events=deepcopy(self.clearance_events),release_requested=self.release_requested,
        actual_body_clearance_or_native_success_guaranteed=False))


class NormalBinClearanceIntents(PublicClearance,NormalBinProgressIntents):pass
class HighBinClearanceIntents(PublicClearance,HighBinProgressIntents):pass
