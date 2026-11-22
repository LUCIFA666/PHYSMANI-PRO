"""V68 original V67 public producer, one fresh hold response-seeded CEM row.

No perception, proof, maintenance, global goal, cost or dispatch rule changes.
Only the actual publicly confirmed hold planner is independently replaced.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,time
import numpy as np
from . import public_candidate_scope12_v67 as base
from . import public_candidate_local_hold_v66 as hold
from .response_seed_hold_planner_v68 import ResponseSeedHoldPlannerV68,response_seed_planner_source_map
from .public_candidate_local_hold_v66 import (BoundForecastExecutionModel,ForecastJointGoalCEMPlanner,
    JointGoalCEMPlanner,qualify_public_grasp_candidates)
from .public_grasp_v41 import ROOT,append,write_new,quantiles,asdict,build_response_planner
from .public_motion_bridge import PublicRobotSensors
from .public_grasp_candidates_v5 import _public_completion_ready,_public_completion_reason


def response_seed_producer_source_map():
    sources=base.position_producer_source_map()
    sources.update(response_seed_planner_source_map())
    path=Path(__file__).resolve();sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


class ObservedPublicCandidateResponseSeedV68Backend(base.ObservedPublicCandidateScope12V67Backend):
    """Exact original V67 measurement/bridge and actual50ms sampling flow."""


class _V68V4Dispatch(hold._V66V4Dispatch):
    def decide(self,backend,result,robot,h):
        if self.assembler.robot_twist is None or not self.assembler.robot_twist[2]:
            self._mode('UNAVAILABLE',robot.timestamp);return dict(mode=self.mode,state=None,reason='no_real_robot_twist')
        state=self.assembler.build(result,robot,h,phase_state=self.phase)
        spec=None
        if state is not None and state['grasp_motion_mode']=='ATTACHED_PUBLIC':
            mode='ATTACHED_PUBLIC';self.pending_started=None;self.reopening_active=False
        elif state is not None and state['grasp_motion_mode']=='FREE_GRU':
            mode='FREE_GRU';self.pending_started=None;self.reopening_active=False
        else:
            state=self.assembler.build_pending(result,robot,h,phase_state=self.phase,reopening=self.reopening_active or h['gripper_target']==1.)
            if state is None:
                self._mode('UNAVAILABLE',robot.timestamp)
                return dict(mode=self.mode,state=None,reason='current_public_object_unknown_hold_actual_q_keep_request',
                    active_visual_attachment=deepcopy(self.active.latest),GRU_used=False,Native_labels_used_by_policy=False)
            if self.pending_started is None:self.pending_started=self.active.attempt_started if self.active.attempt_started is not None else robot.timestamp
            settle_deadline=self.pending_started+self.config.pending_timeout_seconds
            if (h['gripper_target']==0. and self.active.spec is None and not self.active.failed
                    and robot.timestamp<settle_deadline-1e-8):
                self.active.begin_probe(self.assembler.binding.latest_visual_target,robot,h)
            spec=deepcopy(self.active.spec)
            deadline=spec['deadline'] if spec is not None else settle_deadline
            far=np.linalg.norm(state['object_position']-state['tcp_position'])>self.config.pending_far_distance_m
            expired=robot.timestamp>=deadline-1e-8
            if h['gripper_target']==1. or self.reopening_active or far or expired:
                mode='REOPENING_UNKNOWN_CONTACT';self.reopening_active=True;spec=None
            elif self.active.failed:
                self._mode('ACTIVE_VISUAL_PROBE_FAILED_HOLD',robot.timestamp)
                return dict(mode=self.mode,state=None,action=None,
                    reason='failed_active_attempt_hold_actual_q_keep_request_until_unchanged_deadline',
                    active_visual_attachment=deepcopy(self.active.latest),probe_deadline=deadline,
                    GRU_used=False,Native_labels_used_by_policy=False)
            elif spec is not None:mode='ACTIVE_VISUAL_VERIFICATION_PROBE'
            else:mode='PENDING_CLOSE_UNKNOWN_CONTACT'
        previous=self.mode;self._mode(mode,robot.timestamp)
        if mode=='FREE_GRU':
            if previous!=mode:state['phase_state']=None;self.phase=None
            generation=self.geometry.generation;self.candidates=self.geometry.update(state)
            if self.geometry.generation!=generation:self.warm=None
            execution=BoundForecastExecutionModel.from_public_bridge(self.core.execution_model,result);planner_class=ForecastJointGoalCEMPlanner
        else:
            if self.candidates is None:self.candidates=self.geometry.update(state)
            execution=self.core.execution_model;planner_class=JointGoalCEMPlanner
            if mode in ('PENDING_CLOSE_UNKNOWN_CONTACT','ACTIVE_VISUAL_VERIFICATION_PROBE'):
                state['phase_state']=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,
                    close_entered_at=self.pending_started,reopening=False)
            elif mode=='REOPENING_UNKNOWN_CONTACT':
                state['phase_state']=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,reopening=True)
            else:
                if self.hold_tcp_quaternion is None:self.hold_tcp_quaternion=robot.tcp_quaternion.copy()
                state['phase_state']=dict(phase='hold',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,
                    lift_goal_position=state['lift_goal_position'].copy(),hold_tcp_quaternion=self.hold_tcp_quaternion.copy())
        planning=deepcopy(self.candidates)
        if mode=='REOPENING_UNKNOWN_CONTACT':
            for c in planning:c['pregrasp_offset']=max(c['pregrasp_offset'],self.config.reopen_pregrasp_offset_m)
        if mode=='ACTIVE_VISUAL_VERIFICATION_PROBE':
            candidates,qualification,planner=self._verification_components(backend,state,planning,execution,spec)
        elif mode=='ATTACHED_PUBLIC':
            candidates=deepcopy(planning)
            planner=ResponseSeedHoldPlannerV68(self.core.config,execution_model=execution,goal_helper=self.helper,
                constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config,backend=backend)
            qualification=None
        else:
            candidates,qualification=qualify_public_grasp_candidates(backend,state,planning,self.helper,mode=mode,
                bound=execution if mode=='FREE_GRU' else None)
            planner=planner_class(self.core.config,execution_model=execution,goal_helper=self.helper,
                constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config)
        plan=planner.plan(state,candidates,warm_start=self.warm)
        if mode=='ATTACHED_PUBLIC':qualification=deepcopy(planner.fresh_nominal_qualification)
        self.phase=deepcopy(plan['phase_state']);self.warm=plan.get('warm_start');action=deepcopy(plan.get('action'))
        if action is not None:action['candidate_id']=plan['selected_candidate_id']
        return dict(mode=mode,state=state,plan=plan,action=action,qualification=qualification,
            active_visual_attachment=deepcopy(self.active.latest),probe=deepcopy(spec),
            GRU_used=mode=='FREE_GRU',free_motion_asserted=mode=='FREE_GRU',
            legacy_attachment_controls_mode=False,contact_response_qualified=False,force_grasp_proof=False)

class CandidateResponseSeedV68Coordinator(base.CandidateScope12V67Coordinator,_V68V4Dispatch):
    """Existing V67/V65 wrappers reach only the hold-phase replacement."""


def _validate_v68_runner(backend,config):
    if config.get('candidate_response_seed_v68')is not True:
        raise ValueError('Explicit independently frozen V68 response-seed protocol required')
    response_seed_producer_source_map()
    base._validate_v67_runner(backend,config)


def run_public_candidate_response_seed_v68(backend,executor,output,config,*,recorder=None):
    if type(config.get('finger_response_fastpath',False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    _validate_v68_runner(backend,config)
    if type(backend) is not ObservedPublicCandidateResponseSeedV68Backend or executor.backend is not backend:
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
    coordinator=CandidateResponseSeedV68Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery,inference=backend.pilot_inference,cache=backend.candidate_cache,pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',False),cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik',False),probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline',False),probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response',False))
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    provenance.update(candidate_transformer=backend.pilot_inference.receipt(),geometry_fallback_used=False,task_classifier_trained=False,calibrated_candidate_success_probability=False,candidate_pi_symmetry_representative=coordinator.pi_symmetry_representative,cache_fixed_probe_nominal_ik=coordinator.cache_fixed_probe_nominal_ik,probe_terminal_at_fixed_deadline=coordinator.probe_terminal_at_fixed_deadline,probe_onpolicy_joint_response=coordinator.probe_onpolicy_joint_response)
    provenance.update(optional_public_producer=backend.bridge.receipt())
    provenance.update(response_seed=dict(version='V68_fresh_static12_domain_response_inverse',fresh_row='iteration0_candidate_first_eligible_local_index1',history_warm_used=False,CEM_scoring_rows=256,inverse_generation_compute_additional=True,original_selected_endpoint_and_7_path_preserved=True))
    write_new(output/'public_candidate_response_seed_v68_provenance.json',provenance)
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P4_candidate_transformer_v68_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_candidate_position_v68_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
    return dict(stage='P4_public_candidate_response_seed_v68_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        Transformer_used=backend.pilot_inference.calls>0,candidate_transformer=backend.pilot_inference.receipt(),candidate_events=coordinator.geometry.events,candidate_public_cache=backend.candidate_cache.receipt(),geometry_fallback_used=False,
        require_public_goal_completion=backend.require_public_goal_completion,public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(),
        optional_public_producer=backend.bridge.receipt(),effective_attachment_final=deepcopy(backend.bridge.latest_attachment),
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),full_cycle_wall_seconds_samples=times,real_time_guaranteed=False)
