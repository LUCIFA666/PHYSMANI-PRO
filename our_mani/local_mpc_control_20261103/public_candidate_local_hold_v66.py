"""Independent hold-only local nominal and predicted micro-attitude guard.

Perception, original attachment proof, one natural re-verification epoch,
global object goal, costs and exact dispatch are unchanged from V65. Every
hold replan is explicitly cold. Freshly solved local nominal targets receive
new static collision qualification; old far-target certificates are not used.
The guard covers discrete model predictions, not actual or continuous safety.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import time
import numpy as np

from . import public_candidate_reverification_v65 as base
from .hold_orientation_guard_prototype_20261103 import HoldMicroOrientationGuard
from .hold_local_nominal_prototype_20261103 import LocalHoldNominalPlanner
from .joint_goal_cem import JointGoalCEMPlanner
from .public_grasp_v4 import ActiveVisualGraspCoordinator
from .public_grasp import qualify_public_grasp_candidates
from .forecast_control import BoundForecastExecutionModel,ForecastJointGoalCEMPlanner
from .public_grasp_v41 import ROOT,append,write_new,quantiles,asdict,build_response_planner
from .public_motion_bridge import PublicRobotSensors
from .public_grasp_candidates_v5 import _public_completion_ready,_public_completion_reason
from ..local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel

HOLD_BASIS_VERSION='V66_current_TCP_local12mm_hold_micro004_cold_each_actual_replan_v1'
EXTRA_SOURCES={
    'our_mani/local_mpc_control_20261103/hold_orientation_guard_prototype_20261103.py':'6ba74b9eebd57f17fe03c3e77dbbf8dbf582bd3693ef421ed4d43598225742a8',
    'our_mani/local_mpc_control_20261103/hold_local_nominal_prototype_20261103.py':'2c13f54f4b6dd432b5e289b67d724dcb1311293031394da92bb0fdedb9542fea',
    'artifacts/framework/local_mpc_20261102/V66_hold_micro_guard_local12mm_development_protocol_v1/protocol.json':'f48f6cd1cf42592eed8b527182fb2152f621a19a30f3407a78ac718f370a5020',
    'artifacts/framework/local_mpc_20261102/V66_hold_micro_guard_local12mm_development_protocol_v1/cold_hold_initialization_addendum.json':'1ad569f5db68c83de80be71aa59c398b08ed233f8040c2e32c1f2402a63ba7ed',
}


def position_producer_source_map():
    sources=base.position_producer_source_map()
    for name,expected in EXTRA_SOURCES.items():
        path=ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Exact V66 development source required: '+name)
        sources[str(path.resolve())]=expected
    path=Path(__file__).resolve();sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def _assert_current_clock(backend,now):
    current=backend.clock()
    for value in (now,current):
        if isinstance(value,(bool,np.bool_))or not np.isscalar(value)or not np.isfinite(value):
            raise ValueError('Finite non-boolean actual feedback clock required')
    if abs(float(current)-float(now))>1e-8:raise ValueError('Fresh same-frame actual hold clock required')


class ObservedPublicCandidateLocalHoldV66Backend(base.ObservedPublicCandidateReverificationV65Backend):
    """Identical real50ms V65 producer; no changed image/evidence processing."""


class LocalHoldPlannerV66(LocalHoldNominalPlanner):
    def __init__(self,*args,backend,**kwargs):
        super().__init__(*args,**kwargs)
        if type(self.execution_model)is not JointResidualFKExecutionModel:
            raise TypeError('Exact unwrapped hold JointResidualFKExecutionModel required; not forecast or probe wrappers')
        if not callable(getattr(backend,'probe_joint_collision',None))or not callable(getattr(backend,'clock',None)):
            raise TypeError('Original same-frame static collision service and real clock required')
        self.backend=backend;self._base_execution=self.execution_model
        self._sources=position_producer_source_map();self.fresh_nominal_qualification=None
        self._actual_hold_state=None

    def _prepare_nominals(self,initial,candidates,old):
        # Pure prototype DLS, unchanged24 poses and actual-q seed. Its new kind2
        # target is the ONLY pose qualified below; no previous flag is reused.
        super()._prepare_nominals(initial,candidates,old)
        now=float(initial['time'])
        _assert_current_clock(self.backend,now)
        if (np.asarray(self._nominals['joint_goal']).shape!=(3,8,7)
                or np.asarray(self._nominals['feasible']).shape!=(3,8)
                or np.asarray(self._nominals['feasible']).dtype!=np.bool_):
            raise ValueError('Actual local DLS K8 nominal ledger required')
        rows=[];started=time.perf_counter()
        for i,candidate in enumerate(candidates):
            q=np.asarray(self._nominals['joint_goal'][2,i],float)
            if q.shape!=(7,)or not np.isfinite(q).all():raise ValueError('Finite newly solved nominal joint target required')
            defined=('parallel_gripper_symmetry_choice'not in candidate
                or candidate['parallel_gripper_symmetry_choice']is not None)
            algebraic=bool(self._nominals['feasible'][2,i])
            bounds=bool(np.all(q>=self.goal_helper.adapter.lower-1e-8)
                and np.all(q<=self.goal_helper.adapter.upper+1e-8))
            available=defined and algebraic and bounds;collision=None;t=time.perf_counter()
            if available:
                value=self.backend.probe_joint_collision(q.copy())
                if not isinstance(value,(bool,np.bool_)):raise TypeError('Explicit current static collision boolean required')
                collision=bool(value)
            _assert_current_clock(self.backend,now)
            qualified=bool(available and not collision)
            candidate['native_pose_qualified']=qualified
            candidate['qualification_joint_goal']=q.copy()
            candidate['joint_goal_qualification_method']='V66_current_local_nominal_DLS_plus_original_static_collision'
            self._nominals['feasible'][2,i]=qualified
            rows.append(dict(id=candidate['id'],deadline=candidate['deadline'],deadline_shifted=False,
                feedback_time=now,actual_seed_q=initial['q'].copy(),
                actuator_state=deepcopy(initial['actuator_state']),
                target_position=self._nominals['pose'][2,i,:3].copy(),
                target_quaternion=self._nominals['pose'][2,i,3:].copy(),
                newly_solved_nominal_joint_target=q.copy(),candidate_representation_defined=defined,
                algebraic_IK_available=algebraic,bounds_valid=bounds,static_collision=collision,qualified=qualified,
                collision_wall_seconds=time.perf_counter()-t,old_far_qualification_used=False,
                free_deadline_mask_applied_to_hold=False,Native_IK_calls=0,physical_steps=0,
                whole_path_certified=False,contact_object_response_qualified=False))
        self.fresh_nominal_qualification=dict(mode='ATTACHED_PUBLIC',cases=rows,batch_DLS_calls=1,poses=24,
            local_nominal_static_collision_probes=sum(r['static_collision']is not None for r in rows),
            wall_seconds=time.perf_counter()-started,feedback_time=now,
            old_far_qualification_used=False,chosen_action_requires_original_new_endpoint_and_7_path_checks=True,
            original_global_goal_unchanged=True,formal_qualification=False)
        self.nominal_receipt.update(basis_version=HOLD_BASIS_VERSION,production_integrated=True,
            all_incoming_hold_warm_discarded=True,new_local_q_qualification=self.fresh_nominal_qualification)

    def plan(self,state,candidates,*,warm_start=None):
        if (state.get('grasp_motion_mode')!='ATTACHED_PUBLIC'or state.get('held')is not True
                or (state.get('phase_state')or{}).get('phase')!='hold'):
            raise ValueError('V66 planner can only bind actual publicly confirmed hold feedback')
        if self._sources!=position_producer_source_map():raise ValueError('Frozen V66 source closure changed')
        _assert_current_clock(self.backend,state['time'])
        guard=HoldMicroOrientationGuard(self._base_execution,state,maximum_rotation_rad=.04)
        self._actual_hold_state=deepcopy(state);self.execution_model=guard
        try:
            # Declared cold initialization for EVERY hold replan; no old or
            # same-version absolute warm sequence gets a hidden reserved row.
            result=super().plan(state,candidates,warm_start=None)
        finally:self.execution_model=self._base_execution
        _assert_current_clock(self.backend,state['time'])
        if result.get('warm_start')is not None:
            result['warm_start']['nominal_basis_version']=HOLD_BASIS_VERSION
            result['warm_start']['hold_cold_initialization_required']=True
        result['diagnostics']['V66_hold_control']=dict(nominal_basis_version=HOLD_BASIS_VERSION,
            initialization='cold_each_actual_hold_replan',incoming_warm_discarded=warm_start is not None,
            incoming_warm_basis=None if not warm_start else warm_start.get('nominal_basis_version'),
            feedback_sha256=guard.feedback_sha256,maximum_model_micro_rotation_rad=.04,
            maximum_nominal_TCP_translation_m=.012,model_guard_calls=deepcopy(guard.guard_calls),
            guard_integrated_in_v66=True,original_global_goal_and_full_cost_unchanged=True,
            actual_or_continuous_orientation_guaranteed=False,Native_steps=0)
        return result


class _V66V4Dispatch(ActiveVisualGraspCoordinator):
    """Static V4 decide copy below, changing only the actual hold planner route."""
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
            planner=LocalHoldPlannerV66(self.core.config,execution_model=execution,goal_helper=self.helper,
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


class CandidateLocalHoldV66Coordinator(base.CandidateReverificationV65Coordinator,_V66V4Dispatch):
    """V65/V5/V41 wrappers reach the hold-only dispatch through normal MRO."""


def _validate_v66_runner(backend,config):
    if config.get('hold_micro_orientation_local_nominal_v66')is not True:
        raise ValueError('Explicit independently frozen V66 hold-only protocol required')
    position_producer_source_map()
    base._validate_v65_runner(backend,config)


def run_public_candidate_local_hold_v66(backend,executor,output,config,*,recorder=None):
    if type(config.get('finger_response_fastpath',False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    _validate_v66_runner(backend,config)
    if type(backend) is not ObservedPublicCandidateLocalHoldV66Backend or executor.backend is not backend:
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
    coordinator=CandidateLocalHoldV66Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery,inference=backend.pilot_inference,cache=backend.candidate_cache,pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',False),cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik',False),probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline',False),probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response',False))
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
    write_new(output/'public_candidate_local_hold_v66_provenance.json',provenance)
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P4_candidate_transformer_v66_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_candidate_position_v66_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
    return dict(stage='P4_public_candidate_local_hold_v66_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        Transformer_used=backend.pilot_inference.calls>0,candidate_transformer=backend.pilot_inference.receipt(),candidate_events=coordinator.geometry.events,candidate_public_cache=backend.candidate_cache.receipt(),geometry_fallback_used=False,
        require_public_goal_completion=backend.require_public_goal_completion,public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(),
        optional_public_producer=backend.bridge.receipt(),effective_attachment_final=deepcopy(backend.bridge.latest_attachment),
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),full_cycle_wall_seconds_samples=times,real_time_guaranteed=False)
