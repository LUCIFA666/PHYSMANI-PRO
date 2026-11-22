"""Use existing all100-TRAIN grasp priors with fresh public ring measurements.

The original compensation math and limits are retained. It is called directly
by the RML goal policy, independently of the bypassed old controller.step().
Author position guards govern lift/align/insert; native success stays external.
"""
import ast
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .evaluation import file_sha
from .specs import TASK_SPECS
from .rotation_equivalence import CurrentRotationEquivalence
from .physmani_skill_policy_v548 import SkillGeometryIntents,errors
from .grasp_frame_compensation_native_compat import correction,transform


def binding(root,native_repo,task):
    if task not in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'):
        raise ValueError('explicit two ring tasks only')
    path=Path(root)/'artifacts/framework/multitask_r10/ring_grasp_prior_v419'/task/'model.json'
    prior=json.loads(path.read_text())
    if (prior['task']!=task or prior['original_declared_parent_count']!=100 or prior['captured_parent_count']!=100
        or prior['all_captured_parents_used_equally'] is not True or prior['native_success_used_for_selection']
        or prior['VAL_used'] or prior['TEST_used']):raise ValueError('all100 original TRAIN grasp parents, without outcome selection')
    source=Path(native_repo)/TASK_SPECS[task].native_source_relative_path
    if file_sha(source)!=TASK_SPECS[task].native_source_sha256:raise ValueError('original author ring task changed')
    tree=ast.parse(source.read_text());expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    thresholds={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(expert) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('th_wp2','th_wp3','th_wp4')}
    if thresholds!={'th_wp2':.05,'th_wp3':.03,'th_wp4':.03}:raise ValueError('original ring stage thresholds changed')
    return dict(schema='r10_public_RML_ring_existing_TRAIN_grasp_compensation_v580',task=task,
        prior=dict(path=str(path.relative_to(root)),sha256=file_sha(path)),
        phase_positions=dict(lift=.05,align=.03,insert=.03),native_source_sha256=file_sha(source),
        original_all_parent_TRAIN_model_not_refitted=True,full_orientation_promoted=False,
        physical_close_position_m=.02,physical_close_orientation_rad=.1,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_ring_compensation_v580.py'),
        correction_math_source_sha256=file_sha(Path(root)/'our_mani/multitask/grasp_frame_compensation_native_compat.py'))


class CorrectedRingIntents(SkillGeometryIntents):
    def __init__(self,frontend,rule,prior):
        super().__init__(frontend)
        if (rule.get('task')!=self.task or prior.get('task')!=self.task
            or rule.get('phase_positions')!=dict(lift=.05,align=.03,insert=.03)):
            raise ValueError('unchanged author ring guards required')
        self.comp_rule=deepcopy(rule);self.prior=deepcopy(prior)
        self.original_templates=frontend.actor.controller.library.templates
        self.compensation=None;self.compensation_events=[]

    def _measure_compensation(self):
        f=self.frontend;a=f.actor;c=a.controller;s=f.state;now=float(s.robot.timestamp)
        if self.compensation is not None:return
        try:
            info=s.perception_metadata['bodies']['ring'];body=s.bodies['ring']
            attached={k for k,v in s.perception_metadata['bodies'].items() if v.get('motion_mode')=='attached'}
            if (not s.robot.contact or self.close_witness is None
                or self.close_witness.get('manipulated_ids')!=['ring']
                or self.close_witness.get('confirmed_after',float('inf'))>now+1e-8
                or attached!={'ring'} or info.get('mode_provenance')!='caller_controller_touch_action_proprioception'
                or info.get('registration_valid') is not True or info.get('directly_visible') is not True
                or not body.visible or not -1e-8<=now-body.timestamp<=c.generator.config.maximum_observation_age):
                raise ValueError('fresh directly observed unique public held ring required')
            evidence=info.get('rotation_equivalence_evidence');cap=a.collision_port.ring_affordance
            if (not isinstance(evidence,CurrentRotationEquivalence) or cap is None or not cap.permits_geometry(evidence,
                body_id='ring',observed_pose=body.pose,observed_timestamp=body.timestamp,now=now,
                maximum_age=c.generator.config.maximum_observation_age)):
                raise ValueError('original current typed D4 geometry permission required')
            symmetry=a.tracker.templates['ring'].finite_rotation_symmetry.certificate
            if (not np.allclose(self.prior['D4_group_rotations_body'],symmetry['group_rotations_body'],rtol=0,atol=1e-12)
                or not np.allclose(self.prior['D4_center_body'],symmetry['center_body'],rtol=0,atol=1e-12)):
                raise ValueError('TRAIN grasp prior and actual ring CAD use different D4 frames')
            measured=Pose(s.robot.pose.inverse_apply(body.pose.position),s.robot.pose.rotation.T@body.pose.rotation)
            delta,actual=correction(self.prior,measured)
            distance=float(np.linalg.norm(delta.position));angle=float(Rotation.from_matrix(delta.rotation.copy()).magnitude())
            residuals=self.prior['parent_residuals']
            if distance>.02+max(x['position_error_m'] for x in residuals)+1e-8 or angle>.1+max(x['orientation_error_rad'] for x in residuals)+1e-8:
                raise ValueError('correction exceeds original physical grasp plus all-TRAIN residual limits')
            c.library.templates=tuple(transform(t,delta) if t.phase in ('align','insert') else t for t in self.original_templates)
            self.compensation=delta
            self.compensation_events.append(dict(time=now,event='current_public_grasp_compensation_admitted',
                measured_tool_to_ring_xyzw=measured.xyzw().tolist(),D4_representative_xyzw=actual.xyzw().tolist(),
                correction_xyzw=delta.xyzw().tolist(),source='current_RGBD_and_public_robot_pose_at_own_contact',
                full_orientation_promoted=False,TRAIN_templates_refitted=False,native_outcome_read=False))
        except (ValueError,KeyError) as error:
            self.compensation_events.append(dict(time=now,event='compensation_not_admitted_original_targets_retained',reason=str(error)))

    def candidates(self,phase):
        if phase in ('align','insert'):self._measure_compensation()
        return super().candidates(phase)

    def _done(self,goal):
        if goal['phase'] not in self.comp_rule['phase_positions']:return super()._done(goal)
        target=self._completion_target(goal)
        return target is not None and errors(self.frontend.state.robot,target)[0]<=self.comp_rule['phase_positions'][goal['phase']]

    def receipt(self):return dict(super().receipt(),public_ring_grasp_compensation=dict(
        binding=self.comp_rule,events=deepcopy(self.compensation_events),
        old_controller_step_executed=False,TRAIN_model_refitted=False,full_orientation_promoted=False))
