"""Author target candidates with the unchanged complete original motion gates.

No completed skill or phase timing label is claimed by these action targets.
Every commitment is capped by the original absolute10s task clock.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .interaction import (CandidateGenerator,InteractionCandidate,COMMAND_ISSUE,COMMAND_GOAL,_reference_at,_world_target)

@dataclass(frozen=True)
class ClearanceCandidate(InteractionCandidate):
    placement_clearance:dict|None=None


class ButtonIntentCandidateGenerator(CandidateGenerator):
    placement_clearance=None
    def generate(self, state, forecast, phase_index, remaining_seconds, origin):
        phase = self.library.phase_order[phase_index]
        now, rejected, result = float(state.robot.timestamp), Counter(), []
        suffix=0. # Author state machine has only the original absolute10s clock; no completed-skill timing claim.
        for template in self.library.for_phase(phase):
            ids = ((template.reference_kind + ':' + phase,) if template.reference_kind in ('phase_origin', 'world')
                   else tuple(state.role_bindings.get(template.reference_role, ())))
            if not ids:
                rejected['missing_observed_semantic_reference'] += 1
            for body_id in ids:
                try:
                    current = self.observed_reference(state, body_id, template, origin)
                except (KeyError, ValueError) as error:
                    rejected[str(error)] += 1
                    continue
                if template.reference_kind not in ('phase_origin', 'world'):
                    mode = state.perception_metadata.get('bodies', {}).get(body_id, {}).get('motion_mode')
                    if mode != template.reference_mode:
                        rejected['TRAIN_template_reference_mode_differs_from_controller_observation'] += 1
                        continue
                    if mode == 'attached':
                        rejected['attached_reference_requires_measured_attachment_continuation'] += 1
                        continue
                manipulated = tuple(state.role_bindings.get(template.manipulated_role, ()))
                if template.manipulated_role == template.reference_role:
                    manipulated = (body_id,)
                if not manipulated:
                    rejected['missing_observed_manipulated_body'] += 1
                    continue
                if len(manipulated) != 1:
                    rejected['ambiguous_observed_manipulated_identity'] += 1
                    continue
                for delay in self.config.arrival_offsets_seconds:
                    budget = min(float(remaining_seconds),10.-now)
                    if delay>budget+1e-8:
                        rejected['insufficient_full_continuation_time_budget'] += 1
                        continue
                    arrival = now+delay
                    try:
                        reference, v, w = _reference_at(forecast, body_id, template,
                                                       arrival+template.lag_seconds, origin)
                        if template.reference_time_basis in (COMMAND_ISSUE,COMMAND_GOAL):
                            # Candidate reference is predicted at arrival once;
                            # the selected command then owns this fixed pose.
                            v,w=np.zeros(3),np.zeros(3)
                        progress = min(1., delay/template.duration_seconds) if phase=='follow' else 1.
                        pose, velocity, angular, _ = _world_target(reference, v, w, template, progress)
                        clearance=None
                        if self.placement_clearance is not None:
                            reference,clearance=self.placement_clearance.adjust(state,forecast,template,reference,pose,arrival)
                            pose,velocity,angular,_=_world_target(reference,v,w,template,progress)
                    except (IndexError, KeyError, ValueError):
                        rejected['forecast_horizon_or_body_unavailable'] += 1
                        continue
                    if (np.linalg.norm(velocity) > self.config.maximum_linear_speed+1e-8
                            or np.linalg.norm(angular) > self.config.maximum_angular_speed+1e-8):
                        rejected['target_twist_outside_control_limits'] += 1
                        continue
                    self.bind_candidate_scene(state, forecast, phase, body_id, manipulated, arrival)
                    check = self.reachability.check(state.robot, pose, velocity, angular, delay)
                    if not check.feasible:
                        rejected['IK_joint_' + str(check.reason)] += 1
                        continue
                    if check.joints is None or check.joint_velocities is None:
                        raise ValueError('reachability proof lacks joint position/velocity evidence')
                    collision = self.collision_check(pose=pose, time=arrival, phase=phase,
                        observed_bodies=state.bodies, forecast=forecast, manipulated_ids=manipulated)
                    if type(collision.get('feasible')) is not bool:
                        raise ValueError('dynamic collision service must return explicit feasibility')
                    if not collision['feasible']:
                        rejected['dynamic_collision_' + str(collision.get('reason'))] += 1
                        continue
                    effort = float(np.linalg.norm(check.joints-state.robot.joints))
                    utility = -self.config.geometry_time_penalty*delay-self.config.geometry_joint_effort_penalty*effort
                    selected=ClearanceCandidate(
                        str(len(result))+':'+template.anchor_id+':'+body_id, phase, phase_index,
                        body_id, manipulated, template, now, arrival, pose, velocity, angular,
                        reference, v, w, current, check, float(utility), template.duration_seconds,
                        budget, suffix, template.reference_kind != 'observed_se3',
                        collision.get('evidence', {}),placement_clearance=clearance)
                    result.append(selected)
        self.last_diagnostics = dict(feasible_candidates=len(result), rejections=dict(rejected),
            geometry_rule='negative arrival delay and joint effort after hard time/twist/IK/joint/dynamic-collision feasibility',
            full_continuation_budget_seconds=suffix)
        return result
