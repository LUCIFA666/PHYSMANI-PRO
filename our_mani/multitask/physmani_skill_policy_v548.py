"""Trained public skill phases supplying pose intents to the author RML core.

This policy never calls the old joint/Hermite controller. Acquisition needs an
actually dispatched close edge and a unique causally observed held identity.
Gripper qualification reads the latest already-counted public frame.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .physmani_geometry_policy_v518 import ButtonGeometryIntents
from .physmani_geometry_goals_v529 import pose_goals,current_completion_pose
from .interaction import COMMAND_GOAL,_world_target,GripperTimingConfig
from .specs import TASK_SPECS

# Original author distance guards at the last approach waypoint. These are
# policy stage transitions, never a relaxed physical grasp/success condition.
APPROACH_DISTANCE = dict(beat_the_rotating_buzz=.1,insert_onto_rotating_peg=.05,
    moving_basketball_in_hoop=.3,place_cups_on_rotating_frame=.1,
    put_rubbish_in_moving_bin=.1,remove_cups_from_rotating_frame=.05)
RELEASE_PHASES=('release','deposit','hang','place')


def errors(robot,target):
    return (float(np.linalg.norm(robot.pose.position-target.position)),
            float(Rotation.from_matrix(np.array(robot.pose.rotation@target.rotation.T,copy=True)).magnitude()))


class SkillGeometryIntents(ButtonGeometryIntents):
    def __init__(self,frontend):
        self.frontend=frontend;c=frontend.actor.controller
        self.task=c.library.task_spec.task_name
        if self.task not in TASK_SPECS or TASK_SPECS[self.task].family not in APPROACH_DISTANCE:
            raise ValueError('explicit trained nonbutton task with original approach guard required')
        self.phase_order=tuple(c.library.phase_order)
        if self.phase_order[:2]!=('approach','acquire'):
            raise ValueError('trained approach/acquire phase order required')
        # The old controller is an inactive component provider. Its optional
        # extra timing intersections are not this RML policy's physical gate.
        self.physical_grasp_gate=GripperTimingConfig(.02,.1)
        self.phase_index=0;self.selected=None;self.committed=None
        self.origin=None;self.phase_epoch=None;self.close_witness=None
        self.events=[];self.gripper_events=[];self.finished=False

    def candidates(self,phase):
        return pose_goals(self.frontend,phase,origin=self.origin,committed=self.committed)

    def _completion_target(self,goal):
        try:return current_completion_pose(self.frontend,goal,origin=self.origin)
        except (KeyError,ValueError,IndexError):
            # A publicly measured TCP-to-body transform may survive occlusion
            # after contact. It cannot be invented from a native grasp handle.
            if goal['phase']!='acquire' or goal['reference_id'] not in goal['manipulated_ids']:
                return None
            c=self.frontend.actor.controller;t=goal['template']
            pose=c._causal_attachment_pose(self.frontend.state,goal['reference_id'])
            if pose is None:return None
            if t.reference_kind=='translation_only':pose=Pose(pose.position)
            if t.reference_time_basis==COMMAND_GOAL:
                canonical=Pose.from_xyzw(t.completion_geometry_xyzw)
                return Pose(pose.apply(canonical.position),pose.rotation@canonical.rotation)
            return _world_target(pose,np.zeros(3),np.zeros(3),t,1.)[0]

    def _public_capture(self,goal):
        f=self.frontend;s=f.state;ids=goal['manipulated_ids'];w=self.close_witness
        if (len(ids)!=1 or not s.robot.contact or w is None or w['phase_index']!=self.phase_index
            or tuple(w['manipulated_ids'])!=ids or w['confirmed_after']>s.robot.timestamp+1e-8):
            return False
        metadata=s.perception_metadata;invalid=set(metadata.get('invalid_mode_ids',()))
        bodies=metadata.get('bodies',{});identity=ids[0]
        associated={k for k,v in bodies.items() if v.get('motion_mode')=='attached' and k not in invalid}
        return bool(identity not in invalid and associated=={identity}
            and bodies.get(identity,{}).get('mode_provenance')=='caller_controller_touch_action_proprioception'
            and tuple(s.role_bindings.get(goal['template'].manipulated_role,()))==ids
            and identity in f.actor.controller.measured_attachments)

    def _done(self,goal):
        f=self.frontend;s=f.state;t=goal['template'];target=self._completion_target(goal)
        if target is None:return False
        distance,angle=errors(s.robot,target)
        if goal['phase']=='approach':
            return distance<=APPROACH_DISTANCE[TASK_SPECS[self.task].family]
        if s.robot.timestamp-self.phase_epoch<t.duration_seconds-1e-8:
            return False
        if goal['phase']=='acquire':
            return self._public_capture(goal) and distance<=.02 and angle<=.1
        if goal['phase'] in RELEASE_PHASES and t.at(1.)[3]>=.5:
            if not s.robot.gripper_open or s.robot.contact:return False
        return distance<=t.position_tolerance_m and angle<=t.orientation_tolerance_rad

    def _advance(self):
        f=self.frontend;c=f.actor.controller;goal=self.selected
        if goal is None or not self._done(goal):return
        self.events.append(dict(event='current_public_skill_completed',phase=goal['phase'],
            time=float(f.state.robot.timestamp),native_success_claimed=False,
            own_physical_close_confirmed=goal['phase']=='acquire' and self.close_witness is not None))
        self.phase_index+=1;c.phase_index=self.phase_index
        self.selected=self.committed=None;self.origin=self.phase_epoch=None
        self.finished=self.phase_index==len(self.phase_order)

    def select(self):
        f=self.frontend;c=f.actor.controller
        if f.state is None:return None
        self._advance()
        if self.finished:return None
        phase=self.phase_order[self.phase_index]
        if not f.ready(phase):return None
        if self.phase_epoch is None:
            self.phase_epoch=float(f.state.robot.timestamp);self.origin=f.state.robot.pose
            c.origin=self.origin
        candidates=self.candidates(phase)
        if not candidates:return None
        chosen=dict(candidates[0],phase=phase)
        if self.committed is None:
            self.committed=chosen
            t=chosen['template']
            if t.reference_kind not in ('phase_origin','world'):
                c.committed_roles[t.reference_role]=(chosen['reference_id'],)
            c.committed_roles[t.manipulated_role]=chosen['manipulated_ids']
        self.selected=chosen
        action=np.r_[chosen['pose'].xyzw(),chosen['opening']]
        self.events.append(dict(event='trained_public_geometry_RML_pose_intent',phase=phase,
            time=float(f.state.robot.timestamp),phase_epoch=self.phase_epoch,
            reference_id=chosen['reference_id'],manipulated_ids=list(chosen['manipulated_ids']),
            action=action.tolist(),predicted_reference_time=chosen['predicted_reference_time'],
            source_model_canonical_sha256=c.library.sha256,predictor=f.actor.predictor,
            native_outcome_or_future_observation_used=False))
        # This marks the issued arm interaction, not a future gripper edge.
        f.actor.command_segments.issue('joint_trajectory')
        return action

    def qualify_opening(self,requested,current_requested):
        """Called immediately before an original counted motor/gripper tick."""
        f=self.frontend;s=f.state;goal=self.selected;desired=float(requested>=.5)
        target=None;distance=angle=None;reason='trained_opening_unmodified';proposed=None
        command=desired
        if goal is None:return float(current_requested),dict(reason='no_selected_public_goal'),None
        if goal['phase']=='acquire' and desired<.5:
            if self.close_witness is not None and self.close_witness['phase_index']==self.phase_index:
                command=0.;reason='preserve_own_single_close_edge'
            else:
                target=self._completion_target(goal)
                if target is not None:distance,angle=errors(s.robot,target)
                if target is not None and distance<=.02 and angle<=.1 and s.robot.gripper_open:
                    command=0.;reason='current_public_original_physical_grasp_SE3'
                    if current_requested>=.5:
                        proposed=dict(phase_index=self.phase_index,issued_at=float(s.robot.timestamp),
                            manipulated_ids=list(goal['manipulated_ids']),distance_m=distance,angle_rad=angle)
                else:command=1.;reason='defer_close_until_current_public_physical_grasp_SE3'
        elif goal['phase'] in RELEASE_PHASES and desired>=.5 and current_requested<.5:
            target=self._completion_target(goal)
            if target is not None:distance,angle=errors(s.robot,target)
            t=goal['template']
            if target is None or distance>t.position_tolerance_m or angle>t.orientation_tolerance_rad:
                command=0.;reason='defer_release_until_current_public_canonical_geometry'
        proof=dict(time=float(s.robot.timestamp),phase=goal['phase'],desired_opening=desired,
            current_requested_opening=float(current_requested),qualified_opening=command,
            reason=reason,current_position_error_m=distance,current_orientation_error_rad=angle,
            native_truth_read=False)
        return command,proof,proposed

    def confirm_opening(self,opening,proof,proposed,*,confirmed_after):
        self.frontend.actor.requested_opening=float(opening)
        if proposed is not None:
            self.close_witness=dict(proposed,confirmed_after=float(confirmed_after))
        self.gripper_events.append(dict(proof,confirmed_after=float(confirmed_after),
            actually_dispatched=True,own_close_edge=proposed is not None))

    def receipt(self):
        return dict(schema='r10_trained_skill_public_geometry_intents_on_author_RML_v548',
            task=self.task,events=deepcopy(self.events),gripper_events=deepcopy(self.gripper_events),
            current_phase='verify' if self.finished else self.phase_order[self.phase_index],
            own_close_witness=deepcopy(self.close_witness),native_success_owned_by_parent=True,
            physical_grasp_gate=dict(close_position_tolerance_m=.02,close_orientation_tolerance_rad=.1),
            inactive_old_controller_gripper_gate_used=False,
            old_Hermite_reference_managers_executed=False,formal_VAL_TEST=False)
