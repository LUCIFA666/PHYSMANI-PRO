"""Correct placement TCP targets for the current measured rigid grasp frame."""
from copy import deepcopy
from dataclasses import dataclass,fields
from pathlib import Path
from types import MethodType
import json
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .interaction import PhaseTemplate,COMMAND_GOAL
from .forecasting import ObservedAttachment
from .rotation_equivalence import CurrentRotationEquivalence
from .evaluation import file_sha
TASK='insert_onto_rotating_peg'
PHASES=('align','insert')


def compose(a,b):return Pose(a.position+a.rotation@b.position,a.rotation@b.rotation)
def inverse(a):return Pose(-a.rotation.T@a.position,a.rotation.T)


def correction(prior,measured):
    expected=Pose.from_xyzw(prior['tool_to_ring_xyzw'])
    center=np.asarray(prior['D4_center_body'],float)
    alternatives=[compose(measured,Pose(center-s@center,s)) for s in np.asarray(prior['D4_group_rotations_body'],float)]
    actual=min(alternatives,key=lambda p:Rotation.from_matrix(p.rotation@expected.rotation.T).magnitude())
    return compose(expected,inverse(actual)),actual


@dataclass(frozen=True)
class CorrectedTemplate(PhaseTemplate):
    source_template: object=None
    tool_correction: object=None

    def at(self,progress):
        local,velocity,angular,opening=self.source_template.at(progress)
        offset=local.rotation@self.tool_correction.position
        return compose(local,self.tool_correction),velocity+np.cross(angular,offset),angular,opening


def transform(template,delta):
    if template.phase not in PHASES or template.reference_time_basis!=COMMAND_GOAL:
        raise ValueError('explicit original moving-peg command template required')
    values={f.name:getattr(template,f.name) for f in fields(PhaseTemplate)}
    values['positions']=np.asarray([compose(Pose(p,r),delta).position for p,r in zip(template.positions,template.rotations)])
    values['rotations']=np.asarray([r@delta.rotation for r in template.rotations])
    values['completion_geometry_xyzw']=compose(Pose.from_xyzw(template.completion_geometry_xyzw),delta).xyzw()
    return CorrectedTemplate(**values,source_template=template,tool_correction=delta)


def binding(root,config,task):
    if task!=TASK:raise ValueError('normal-ring compensation pilot first')
    path=Path(root)/'artifacts/framework/multitask_r10/ring_grasp_prior_v419'/task/'model.json'
    prior=json.loads(path.read_text())
    if (prior['task']!=task or prior['original_declared_parent_count']!=100 or prior['captured_parent_count']!=100
        or not prior['all_captured_parents_used_equally'] or prior['native_success_used_for_selection']
        or prior['VAL_used'] or prior['TEST_used']):raise ValueError('all100 original TRAIN capture parents required')
    sources=('our_mani/multitask/grasp_frame_compensation.py','our_mani/multitask/interaction.py',
        'our_mani/multitask/forecasting.py','our_mani/multitask/rotation_equivalence.py')
    return dict(schema='r10_TRAIN_grasp_prior_current_public_attachment_placement_compensation_v1',task=task,
        prior=dict(path=str(path.relative_to(root)),sha256=file_sha(path)),
        interaction=config['learned_artifacts'][task]['interaction'],phases=list(PHASES),
        sources_sha256={p:file_sha(Path(root)/p) for p in sources},
        correction_is_execution_policy_not_refitted_or_relabelled_TRAIN_curve=True,
        corrected_TCP='original_TRAIN_TCP_target compose TRAIN_tool_to_ring compose inverse(current_public_tool_to_ring_D4_representative)',
        completion_geometry_transformed_consistently=True,original_completion_tolerances_unchanged=True,
        actual_and_reference_full_motion_collision_gates_required=True,
        phase_epoch_deadline_grasp_and_native_success_unchanged=True,formal_VAL_TEST=False)


def install(actor,config,root,task):
    expected=binding(root,config,task)
    if config.get('grasp_frame_compensation')!=expected:raise ValueError('freeze measured grasp compensation policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first control step')
    path=Path(root)/expected['prior']['path'];prior=json.loads(path.read_text())
    if file_sha(path)!=expected['prior']['sha256']:raise ValueError('TRAIN grasp prior changed')
    originals=c.library.templates;events=[];active={};previous_step=c.step
    def step(instance,state,forecast,remaining_seconds):
        now=float(state.robot.timestamp)
        held=(state.robot.contact and state.perception_metadata.get('bodies',{}).get('ring',{}).get('motion_mode')=='attached')
        if not held and active and c.goal is None:
            c.library.templates=originals;active.clear()
            events.append(dict(time=now,event='compensation_retired_after_public_contact_loss',phase_clock_reset=False))
        phase=c.library.phase_order[c.phase_index] if c.phase_index<len(c.library.phase_order) else None
        if held and not active and phase in PHASES and c.goal is None:
            try:
                attached={k for k,v in state.perception_metadata.get('bodies',{}).items() if v.get('motion_mode')=='attached'}
                if attached!={'ring'}:raise ValueError('unique current public held ring required')
                info=state.perception_metadata['bodies']['ring'];body=state.bodies['ring']
                if info.get('mode_provenance')!='caller_controller_touch_action_proprioception':raise ValueError('public contact provenance required')
                item=actor.attachments.get('ring');evidence=info.get('rotation_equivalence_evidence')
                if not isinstance(item,ObservedAttachment) or not -1e-8<=now-item.observed_at<=c.generator.config.maximum_observation_age:
                    raise ValueError('fresh measured rigid grasp frame required')
                if not isinstance(evidence,CurrentRotationEquivalence):raise ValueError('current typed finite CAD evidence required')
                cap=actor.collision_port.ring_affordance
                if not cap.permits_geometry(evidence,body_id='ring',observed_pose=body.pose,observed_timestamp=body.timestamp,
                    now=now,maximum_age=c.generator.config.maximum_observation_age):raise ValueError('original ring geometry permission required')
                delta,actual=correction(prior,item.tool_to_body)
                p=float(np.linalg.norm(delta.position));r=float(Rotation.from_matrix(delta.rotation).magnitude())
                residuals=prior['parent_residuals'];cfg=c.gripper_timing_config
                if p>cfg.close_position_tolerance_m+max(e['position_error_m'] for e in residuals)+1e-8:
                    raise ValueError('translation correction exceeds original grasp plus full TRAIN residual')
                if r>cfg.close_orientation_tolerance_rad+max(e['orientation_error_rad'] for e in residuals)+1e-8:
                    raise ValueError('rotation correction exceeds original grasp plus full TRAIN residual')
                c.library.templates=tuple(transform(t,delta) if t.phase in PHASES else t for t in originals)
                active.update(admitted_at=now,delta=delta,measured=item)
                events.append(dict(time=now,event='measured_grasp_frame_placement_compensation_admitted',
                    original_measured_tool_to_ring_xyzw=item.tool_to_body.xyzw().tolist(),
                    D4_representative_xyzw=actual.xyzw().tolist(),tool_target_correction_xyzw=delta.xyzw().tolist(),
                    original_completion_tolerances_unchanged=True,phase_clock_reset=False,task_success_claimed=False))
            except (ValueError,KeyError) as error:
                events.append(dict(time=now,event='compensation_not_admitted_original_policy_retained',reason=str(error)))
        return previous_step(state,forecast,remaining_seconds)
    c.step=MethodType(step,c)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),grasp_frame_compensation=dict(binding=expected,
        events=deepcopy(events),source_TRAIN_templates_not_rewritten=True,extra_camera_physics_or_native_goal_reads=0))
    actor.metrics_report=MethodType(metrics,actor)
    return events
