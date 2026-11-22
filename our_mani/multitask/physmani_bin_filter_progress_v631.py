"""Bounded public pose intentions for original RML's 1cm waypoint resolution.

Only an actual own 'released_modified_path_too_short' receipt with one retained
waypoint can activate this near-grasp adaptation. The canonical completion and
author1cm / physical2cm-.1rad close gates remain completely unmodified.
"""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .physmani_stage_lookahead_v594 import AuthorLookaheadIntents
from .physmani_high_bin_delay_v601 import HighBinDelayIntents
from .physmani_skill_policy_v548 import errors


def binding(root):
    source=Path(root)/'our_mani/ablations/physmani_executor_v4.py'
    f=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='released_waypoint_indices')
    values={a.arg:ast.literal_eval(v) for a,v in zip(f.args.kwonlyargs,f.args.kw_defaults)}
    if values!={'discard':10,'min_dist':.01}:raise ValueError('original author Cartesian filter changed')
    return dict(schema='r10_public_bin_near_grasp_original_RML_resolution_adaptation_v631',
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_bin_filter_progress_v631.py'),
        original_filter_source_sha256=file_sha(source),cartesian_spacing_m=.01,
        required_own_error='released_modified_path_too_short',required_retained_waypoints=1,
        maximum_pose_intent_bias_m=.01,maximum_current_TCP_to_intent_m=.02,
        activation_current_canonical_distance_interval_m=[.01,.02],
        phase='acquire',own_close_or_public_attachment_forbids_adaptation=True,
        original_canonical_close_geometry_and_native_RML_unchanged=True)


class FilterResolutionIntent:
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.filter_feedback=None;self.filter_events=[]

    def _goal_key(self,goal):
        return (self.phase_index,goal['phase'],goal['template'].anchor_id,goal['reference_id'],goal['manipulated_ids'])

    def observe_owned_RML_path_result(self,receipt):
        if set(receipt)-{'error','waypoint_modification'}:raise ValueError('only sanitized own planner facts enter policy')
        goal=self.selected;modification=receipt.get('waypoint_modification') or {}
        if set(modification)-{'selected_waypoints','cartesian_spacing_m'}:raise ValueError('only own waypoint count/spacing enter policy')
        self.filter_feedback=None
        if (goal is not None and receipt.get('error')=='released_modified_path_too_short'
            and modification.get('selected_waypoints')==1 and modification.get('cartesian_spacing_m')==.01):
            self.filter_feedback=self._goal_key(goal)
            # Deliberately retain no native terminal/success/fatal fields.
            self.filter_events.append(dict(event='own_single_waypoint_stagnation_received',
                phase=goal['phase'],time=float(self.frontend.state.robot.timestamp),
                error='released_modified_path_too_short',retained_waypoints=1,native_outcome_supplied=False))

    def candidates(self,phase):
        rows=super().candidates(phase);f=self.frontend
        if (phase!='acquire' or self.close_witness is not None or f.state.robot.contact
            or not f.state.robot.gripper_open):return rows
        result=[]
        for row in rows:
            target=self._completion_target(row)
            if abs(row['predicted_reference_time']-float(f.state.robot.timestamp))>1e-8:
                result.append(row);continue
            if target is None or self.filter_feedback!=self._goal_key(row):result.append(row);continue
            distance=errors(f.state.robot,target)[0]
            if not .01<distance<=.02:result.append(row);continue
            # A2cm intent span leaves >1cm after the original first10 points
            # are discarded. Its extra displacement is bounded by one filter
            # spacing and is never substituted for the true close geometry.
            delta=target.position-f.state.robot.pose.position
            pad=.02-distance;bias=pad*delta/distance
            pose=Pose(target.position+bias,target.rotation)
            adapted=dict(row,pose=pose,score=-float(np.linalg.norm(pose.position-f.state.robot.pose.position)),
                original_unbiased_pose=row['pose'],public_filter_resolution_bias_m=pad)
            result.append(adapted)
            self.filter_events.append(dict(event='bounded_public_canonical_intent_extension',phase=phase,
                time=float(f.state.robot.timestamp),reference_id=row['reference_id'],
                current_canonical_distance_m=distance,bias_m=float(pad),bias_vector=bias.tolist(),
                original_intended_xyzw=row['pose'].xyzw().tolist(),adapted_intended_xyzw=pose.xyzw().tolist(),
                canonical_geometry_unchanged=True,author_close_gate_m=.01,physical_close_gate_m=.02,
                geometry_collision_admission_still_required=True,native_pose_or_outcome_used=False))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def receipt(self):return dict(super().receipt(),public_RML_filter_resolution=dict(
        schema='r10_public_bin_near_grasp_original_RML_resolution_adaptation_v631',
        events=deepcopy(self.filter_events),native_RML_or_grasp_gates_changed=False,
        native_success_and_global_task_clock_unchanged=True))


class NormalBinProgressIntents(FilterResolutionIntent,AuthorLookaheadIntents):pass
class HighBinProgressIntents(FilterResolutionIntent,HighBinDelayIntents):pass
