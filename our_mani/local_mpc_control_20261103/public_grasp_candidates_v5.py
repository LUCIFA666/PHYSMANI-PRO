"""P4 development: frozen future-candidate Transformer with public v41 MPC.

The default changes only the candidate generator. Explicit optional protocols
scope the public completion, cached probe destination and fixed-deadline cost;
the execution response, CEM budget and exact selected-action dispatch are reused.
Unavailable causal features hold actual q; gripper intent follows the inherited
v41 initial-open warmup, then preserves the existing request.
No geometry fallback, task classifier or calibrated success score is added.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from ..contracts import Pose
from .public_motion_bridge import PublicRobotSensors
from .public_grasp_v41 import (
    ObservedPublicGraspV41Backend,ActiveVisualGraspV41Coordinator,
    ROOT,append,write_new,quantiles,asdict,build_response_planner)
from ..local_mpc_p4_20261103.runtime import (
    pack_public_row,PublicRowCache,FrozenPilotInference,LearnedCandidates,ProposalUnavailable)


class ObservedPublicCandidateV5Backend(ObservedPublicGraspV41Backend):
    def __init__(self,*args,checkpoint,checkpoint_sha256,require_public_goal_completion=False,**kwargs):
        if type(require_public_goal_completion) is not bool:
            raise ValueError('explicit boolean public completion protocol required')
        self.require_public_goal_completion=require_public_goal_completion
        self.public_goal_completion=None
        torch.set_num_threads(2)
        self.candidate_cache=PublicRowCache()
        self.pilot_inference=FrozenPilotInference(checkpoint,expected_sha256=checkpoint_sha256)
        super().__init__(*args,**kwargs)
    def sample(self,state=None):
        # The outer sensor adapter sees Native read_state only to sanitize the
        # same robot channels and known command intent as the v41 observer.
        # No privileged object/held/success field enters pack_public_row/cache.
        current=self.backend.read_state() if state is None else state
        result=super().sample(current)
        index=self.samples-1
        row=pack_public_row(result,self.latest_robot,float(current['requested_opening'])<.5,index,f'rgbd/{index:04d}.npz')
        self.candidate_cache.observe(row)
        if self.require_public_goal_completion:
            if self.public_goal_completion is None and isinstance(result.get('goal_pose'),Pose):
                from .public_grasp_goal_completion_v52 import PublicGraspGoalCompletion
                self.public_goal_completion=PublicGraspGoalCompletion(result['goal_pose'])
            if self.public_goal_completion is not None:
                self.public_goal_completion.observe(result,self.latest_robot,self.public_hold_metric.receipt(),
                    close_requested=float(current['requested_opening'])<.5)
        return result
    def receipt(self):
        result=super().receipt()
        result.update(candidate_transformer=self.pilot_inference.receipt(),candidate_cache=self.candidate_cache.receipt(),
            actual_public50ms_capture_is_producer=True,live_RGBD_reference_file_check=False,
            live_file_check_scope='original real observer captures same timestamp; bounded async writer may not have drained yet',
            require_public_goal_completion=self.require_public_goal_completion,
            public_goal_completion=None if self.public_goal_completion is None else self.public_goal_completion.receipt(),
            geometry_fallback_used=False)
        return result


class CandidateGraspV5Coordinator(ActiveVisualGraspV41Coordinator):
    def __init__(self,*args,inference,cache,pi_symmetry_representative=False,cache_fixed_probe_nominal_ik=False,probe_terminal_at_fixed_deadline=False,probe_onpolicy_joint_response=False,**kwargs):
        super().__init__(*args,**kwargs)
        if type(pi_symmetry_representative) is not bool:
            raise ValueError('explicit boolean candidate representation protocol required')
        self.pi_symmetry_representative=pi_symmetry_representative
        if type(cache_fixed_probe_nominal_ik) is not bool:
            raise ValueError('explicit boolean fixed probe nominal protocol required')
        self.cache_fixed_probe_nominal_ik=cache_fixed_probe_nominal_ik
        if type(probe_terminal_at_fixed_deadline) is not bool:
            raise ValueError('explicit boolean fixed probe terminal-time protocol required')
        if probe_terminal_at_fixed_deadline and not cache_fixed_probe_nominal_ik:
            raise ValueError('fixed probe deadline protocol requires the explicit cached destination protocol')
        self.probe_terminal_at_fixed_deadline=probe_terminal_at_fixed_deadline
        if type(probe_onpolicy_joint_response) is not bool:
            raise ValueError('explicit boolean on-policy probe response protocol required')
        if probe_onpolicy_joint_response and not (cache_fixed_probe_nominal_ik and probe_terminal_at_fixed_deadline):
            raise ValueError('on-policy probe response requires fixed nominal and absolute deadline protocols')
        self.probe_onpolicy_joint_response=probe_onpolicy_joint_response
        self.probe_execution_model=None
        self.probe_execution_base=None
        self.probe_response_receipt=None
        self.probe_nominal_cache=None
        self.probe_cache_sources=None
        if cache_fixed_probe_nominal_ik:
            from .probe_nominal_cache_v53 import ProbeNominalQCache
            self.probe_nominal_cache=ProbeNominalQCache()
            self.probe_cache_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__),Path(__file__).with_name('probe_nominal_cache_v53.py'),
                Path(__file__).with_name('public_grasp_probe.py'),Path(__file__).with_name('joint_goals.py'),
                Path(__file__).with_name('joint_goal_cem.py'),Path(__file__).with_name('coupled_response.py'),
                ROOT/'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json',
                ROOT/'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt')}
            if probe_terminal_at_fixed_deadline:
                path=Path(__file__).with_name('probe_deadline_v54.py')
                self.probe_cache_sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            if probe_onpolicy_joint_response:
                from .probe_onpolicy_response_v55 import probe_response_sources
                self.probe_cache_sources.update(probe_response_sources())
        if pi_symmetry_representative:
            from ..local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates
            self.geometry=SymmetricLearnedCandidates(inference,cache)
        else:
            self.geometry=LearnedCandidates(inference,cache)
    def _verification_components(self,backend,state,planning,execution,spec):
        if not self.cache_fixed_probe_nominal_ik:
            return super()._verification_components(backend,state,planning,execution,spec)
        from .probe_nominal_cache_v53 import qualify_cached_probe_candidates,CachedNominalVerificationProbePlanner
        if self.probe_onpolicy_joint_response:
            if self.probe_execution_model is None:
                from .probe_onpolicy_response_v55 import build_onpolicy_probe_execution
                self.probe_execution_model,self.probe_response_receipt=build_onpolicy_probe_execution(execution)
                self.probe_execution_base=execution
            if self.probe_execution_base is not execution:
                raise ValueError('probe response cannot be transplanted onto a different base execution model')
            execution=self.probe_execution_model
        # Handoff expiry does not invalidate an already admitted robot-only
        # probe. Undefined representatives remain permanently unavailable.
        nominal_mask=np.asarray(['parallel_gripper_symmetry_choice' not in c
            or c['parallel_gripper_symmetry_choice'] is not None for c in planning],dtype=bool)
        candidates,qualification=qualify_cached_probe_candidates(backend,state,planning,self.helper,spec,
            nominal_cache=self.probe_nominal_cache,source_sha256=self.probe_cache_sources,nominal_mask=nominal_mask)
        planner_class=CachedNominalVerificationProbePlanner
        if self.probe_terminal_at_fixed_deadline:
            from .probe_deadline_v54 import FixedDeadlineVerificationProbePlanner
            planner_class=FixedDeadlineVerificationProbePlanner
        planner=planner_class(self.core.config,execution_model=execution,goal_helper=self.helper,
            constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config,probe_spec=spec,
            nominal_cache=self.probe_nominal_cache,source_sha256=self.probe_cache_sources,nominal_mask=nominal_mask,
            qualification=qualification)
        return candidates,qualification.receipt(),planner
    def decide(self,backend,result,robot,h):
        try:
            decision=super().decide(backend,result,robot,h)
        except ProposalUnavailable as error:
            self._mode('CANDIDATE_PUBLIC_INPUT_UNAVAILABLE',robot.timestamp)
            self.warm=None
            decision=dict(mode=self.mode,state=None,action=None,GRU_used=False,
                reason=str(error),geometry_fallback_used=False,Native_labels_used_by_policy=False,
                current_held_asserted=False)
        decision.update(candidate_transformer=self.geometry.inference.receipt(),
            candidate_pool=deepcopy(self.geometry.last_snapshot),
            candidate_feature_provenance=deepcopy(self.geometry.last_feature_provenance),
            geometry_fallback_used=False,calibrated_candidate_success_probability=False,
            candidate_pi_symmetry_representative=self.pi_symmetry_representative,
            cache_fixed_probe_nominal_ik=self.cache_fixed_probe_nominal_ik,
            probe_terminal_at_fixed_deadline=self.probe_terminal_at_fixed_deadline,
            probe_onpolicy_joint_response=self.probe_onpolicy_joint_response,
            probe_response_receipt=deepcopy(self.probe_response_receipt),
            probe_nominal_cache_stats=(None if self.probe_nominal_cache is None else
                dict(solve_count=self.probe_nominal_cache.solve_count,reuse_count=self.probe_nominal_cache.reuse_count,
                    cached_destination_only=True,actual_or_virtual_state_cached=False)))
        return decision


def _public_completion_ready(backend):
    if getattr(backend,'require_public_goal_completion',False):
        metric=backend.public_goal_completion
        return bool(metric is not None and metric.passed)
    return backend.public_hold_metric.passed


def _public_completion_reason(backend):
    return ('visible_public_grasp_and_goal_complete' if getattr(backend,'require_public_goal_completion',False)
            else 'visible_public_lift_hold_observed')


def run_public_grasp_candidates_v5(backend,executor,output,config,*,recorder=None):
    if type(config.get('finger_response_fastpath',False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    if type(backend) is not ObservedPublicCandidateV5Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_candidate_transformer_v5') is not True:raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:raise ValueError('100ms/2actual50ms budget required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    if config.get('finger_response_fastpath',False):
        from .finger_fastpath_integration_20261103 import enable_finger_fastpath
        provenance=enable_finger_fastpath(core.execution_model,provenance)
    coordinator=CandidateGraspV5Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery,inference=backend.pilot_inference,cache=backend.candidate_cache,pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',False),cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik',False),probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline',False),probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response',False))
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    provenance.update(candidate_transformer=backend.pilot_inference.receipt(),geometry_fallback_used=False,task_classifier_trained=False,calibrated_candidate_success_probability=False,candidate_pi_symmetry_representative=coordinator.pi_symmetry_representative,cache_fixed_probe_nominal_ik=coordinator.cache_fixed_probe_nominal_ik,probe_terminal_at_fixed_deadline=coordinator.probe_terminal_at_fixed_deadline,probe_onpolicy_joint_response=coordinator.probe_onpolicy_joint_response)
    write_new(output/'public_grasp_candidates_v5_provenance.json',provenance)
    maximum=int(config.get('public_grasp_max_decisions',np.ceil(float(config.get('simulation_seconds',6.))/.1)))
    if maximum<=0:raise ValueError('positive decision budget required')
    cycles=0;times=[];ever=False;stop='budget_exhausted'
    while cycles<maximum and backend.remaining_ticks()>=2:
        if any(p.exists() for p in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):stop='administrative_stop';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
        started=time.perf_counter();robot=backend.latest_robot;h=deepcopy(executor.state())
        if not isinstance(robot,PublicRobotSensors) or abs(robot.timestamp-backend.clock())>1e-8:raise ValueError('fresh public robot frame required')
        if h['gripper_target'] not in (0.,1.) and not ever:
            decision=dict(mode='UNAVAILABLE',state=None,reason='initial_actual_open_warmup');action=None;opening=1.
        else:
            decision=coordinator.decide(backend,backend.latest,robot,h);action=decision.get('action');opening=float(h['gripper_target'])
        rejected=None;qualification=None
        if action is None:
            coordinator.warm=None;goal=robot.q.copy()
            if not ever and not backend.recovery.preserves_closed_attempt(backend.bridge.active.attempt_id):opening=1.
            executed=dict(joint_goal=goal.copy(),opening=opening,source='actual_q_hold_keep_requested_gripper')
        else:
            opening=float(action['opening'])
            if opening not in (0.,1.):raise ValueError('binary modeled opening required')
            if decision['mode']=='REOPENING_UNKNOWN_CONTACT' and opening!=1.:rejected='reopen_model_proposed_close'
            if decision['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE' and opening!=0.:rejected='probe_model_proposed_open'
            goal,reason,qualification=helper.qualify_action(backend,decision['state'],action,require_predicted_joint_goal=True)
            rejected=rejected or reason
            if rejected:
                goal=robot.q.copy();opening=float(h['gripper_target']);coordinator.warm=None
                executed=dict(joint_goal=goal.copy(),opening=opening,source='rejected_model_action_hold_actual_q_keep_request')
            else:
                if not np.array_equal(goal,np.asarray(action['joint_goal'])):raise ValueError('selected modeled joint goal changed before dispatch')
                executed=dict(joint_goal=goal.copy(),opening=opening,source='exact_selected_modeled_q_and_opening')
                coordinator.note_executed(action,robot)
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P4_candidate_transformer_v5_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_grasp_candidates_v5_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
    return dict(stage='P4_public_candidate_transformer_grasp_v5_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        Transformer_used=backend.pilot_inference.calls>0,candidate_transformer=backend.pilot_inference.receipt(),candidate_events=coordinator.geometry.events,candidate_public_cache=backend.candidate_cache.receipt(),geometry_fallback_used=False,
        require_public_goal_completion=backend.require_public_goal_completion,public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(),
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),full_cycle_wall_seconds_samples=times,real_time_guaranteed=False)
