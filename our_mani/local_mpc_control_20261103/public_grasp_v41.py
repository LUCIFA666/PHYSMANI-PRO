"""v4.1 recovery: historical confirmation never substitutes for current held.

An attempt which was publicly confirmed cannot be reopened by an expired
verification deadline or an untrusted far/jumped detection. Current unknown
state holds actual q and the existing gripper request. A fresh independently
confirmed public release alone clears this recovery memory in this version.
The v4 observer/attachment thresholds and all frozen planner sources are reused.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from .public_grasp_v4 import (
    ActiveVisualPublicBridge,ActiveVisualGraspConfig,ObservedPublicGraspV4Backend,
    ActiveVisualGraspCoordinator,ROOT,append,write_new,quantiles,asdict,build_response_planner)

ActiveVisualGraspV41Config=ActiveVisualGraspConfig


class ConfirmedAttemptRecovery:
    """Actual50ms public-event ledger; this does not assert current attachment."""
    def __init__(self):
        self.confirmed={};self.released={};self.last_timestamp=None
    def observe(self,result,robot):
        if not isinstance(robot,PublicRobotSensors):raise TypeError('actual current public robot sensors required')
        now=robot.timestamp
        if self.last_timestamp is not None and now<self.last_timestamp-1e-8:raise ValueError('public recovery ledger cannot read past frames')
        if (not isinstance(result,dict) or not np.isfinite(float(result.get('timestamp',np.inf)))
                or abs(float(result.get('timestamp',np.inf))-now)>1e-8):
            raise ValueError('fresh same-time actual public result required')
        self.last_timestamp=now
        a=result.get('attachment_estimate',{});e=a.get('evidence') if isinstance(a,dict) else None
        attempt=a.get('attempt_id') if isinstance(a,dict) else None
        if (not isinstance(a,dict) or a.get('protocol')!='active_vertical_visual_comotion_v4'
                or type(attempt) is not int or attempt<=0 or abs(float(a.get('timestamp',np.inf))-now)>1e-8
                or not np.isfinite(float(a.get('timestamp',np.inf)))
                or not isinstance(e,PublicMotionEvidence) or abs(e.timestamp-now)>1e-8):return
        if (e.event=='attachment_confirmed' and result.get('mode')=='ATTACHED_PUBLIC'
                and result.get('available') is True and result.get('controlled_mode_available') is True and a.get('mode')=='ATTACHED_PUBLIC'):
            self.confirmed.setdefault(attempt,now)
        if (e.event=='release_confirmed' and result.get('mode')=='FREE' and a.get('mode')=='FREE'
                and a.get('attempt_active') is False and np.all(robot.finger_open_amounts>=.60)):
            self.released.setdefault(attempt,now)
    def preserves_closed_attempt(self,attempt_id):return attempt_id in self.confirmed and attempt_id not in self.released
    def receipt(self):
        return dict(confirmed_attempt_timestamps=deepcopy(self.confirmed),released_attempt_timestamps=deepcopy(self.released),
            last_actual_public_timestamp=self.last_timestamp,historical_confirmation_is_current_held=False,
            unknown_policy='actual_q_hold_keep_existing_gripper_request_no_lift',
            expired_probe_deadline_can_release_confirmed_attempt=False,untrusted_visible_far_can_release_confirmed_attempt=False,
            clearing_condition='fresh_typed_same_attempt_public_open_and_separation_release_evidence',
            Native_inputs_used=False)


class ObservedPublicGraspV41Backend(ObservedPublicGraspV4Backend):
    def __init__(self,*args,**kwargs):
        self.recovery=ConfirmedAttemptRecovery();super().__init__(*args,**kwargs)
    def sample(self,state=None):
        result=super().sample(state);self.recovery.observe(result,self.latest_robot);return result
    def receipt(self):
        result=super().receipt();result.update(post_confirmation_recovery_revision='v41',
            post_confirmation_unknown_does_not_reenter_probe_timeout=True,recovery=self.recovery.receipt())
        return result


class ActiveVisualGraspV41Coordinator(ActiveVisualGraspCoordinator):
    def __init__(self,*args,recovery=None,**kwargs):
        super().__init__(*args,**kwargs);self.recovery=recovery or ConfirmedAttemptRecovery()
    def decide(self,backend,result,robot,h):
        # The production observer records every real50ms event, including a
        # confirmation/release between decisions. Duplicate same-time reads are
        # harmless; no estimated/cached object pose becomes current evidence.
        self.recovery.observe(result,robot)
        if self.recovery.preserves_closed_attempt(self.active.attempt_id):
            # Reuse the unchanged strict public/robot/h contracts. A current
            # valid attachment may continue ordinary lift; history alone may not.
            state=self.assembler.build(result,robot,h,phase_state=self.phase)
            if state is None or state.get('grasp_motion_mode')!='ATTACHED_PUBLIC':
                self.assembler._inputs(result,robot,h,self.phase)
                mode='POST_CONFIRMED_UNKNOWN_HOLD' if h['gripper_target']==0. else 'POST_CONFIRMED_EXPLICIT_OPEN_REQUEST_HOLD'
                self._mode(mode,robot.timestamp);self.warm=None
                return dict(mode=mode,state=None,action=None,GRU_used=False,free_motion_asserted=False,
                    reason='previously_confirmed_attempt_current_unknown_hold_without_old_probe_timeout_or_far_reopen',
                    current_held_asserted=False,continue_lift=False,automatic_reopen=False,
                    existing_gripper_request=float(h['gripper_target']),recovery=self.recovery.receipt(),
                    active_visual_attachment=deepcopy(self.active.latest),Native_labels_used_by_policy=False)
        decision=super().decide(backend,result,robot,h)
        decision['recovery']=self.recovery.receipt();return decision


def run_public_grasp_v41(backend,executor,output,config,*,recorder=None):
    if type(backend) is not ObservedPublicGraspV41Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_active_visual_v41') is not True:raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:raise ValueError('100ms/2actual50ms budget required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    coordinator=ActiveVisualGraspV41Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery)
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    write_new(output/'public_grasp_v41_provenance.json',provenance)
    maximum=int(config.get('public_grasp_max_decisions',np.ceil(float(config.get('simulation_seconds',6.))/.1)))
    if maximum<=0:raise ValueError('positive decision budget required')
    cycles=0;times=[];ever=False;stop='budget_exhausted'
    while cycles<maximum and backend.remaining_ticks()>=2:
        if any(p.exists() for p in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):stop='administrative_stop';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P3_active_visual_v41_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_grasp_v41_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
    return dict(stage='P3_active_visual_grasp_v41_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),real_time_guaranteed=False)
