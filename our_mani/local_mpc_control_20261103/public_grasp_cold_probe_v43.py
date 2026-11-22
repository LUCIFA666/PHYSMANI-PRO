"""v43 diagnostic: cold CEM only while verifying the bounded public probe.

The exact v41 observer, recovery, model/cost/budget and dispatch are reused.
This does not force the nominal action: all256 proposals remain active. Cold
replays chose nominal on19 recorded states; no future outcome is guaranteed.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from .public_motion_bridge import PublicRobotSensors
from .public_grasp_v41 import (
    ActiveVisualGraspV41Config,ActiveVisualPublicBridge,ObservedPublicGraspV41Backend,
    ActiveVisualGraspV41Coordinator,ROOT,append,write_new,quantiles,asdict,build_response_planner)

ActiveVisualGraspV43Config=ActiveVisualGraspV41Config
RUNNER_BASE_SOURCE_SHA256='039336bbba4fca69d2bff6e3e29fbb53df2db2e02e9881ff2f1ddb67693c23aa'
RUNNER_APPROVED_RENAMES={
 'run_public_grasp_v41':'run_public_grasp_cold_probe_v43',
 'public_grasp_active_visual_v41':'public_grasp_cold_probe_v43',
 'ActiveVisualGraspV41Coordinator':'ColdProbeGraspV43Coordinator',
 'public_grasp_v41_provenance.json':'public_grasp_cold_probe_v43_provenance.json',
 'public_grasp_v41_decisions.jsonl':'public_grasp_cold_probe_v43_decisions.jsonl',
 'P3_active_visual_v41_':'P3_cold_probe_v43_',
 'P3_active_visual_grasp_v41_development':'P3_cold_probe_grasp_v43_development'}


class ColdProbeGraspV43Coordinator(ActiveVisualGraspV41Coordinator):
    """Only optimizer memory changes; no action or actual feedback is edited."""
    def decide(self,backend,result,robot,h):
        incoming_probe=self.mode=='ACTIVE_VISUAL_VERIFICATION_PROBE'
        before_discarded=bool(incoming_probe and self.warm is not None)
        if incoming_probe:self.warm=None
        # First entry from pending already clears incompatible memory in v4's
        # explicit mode transition before it calls the unchanged probe planner.
        decision=super().decide(backend,result,robot,h)
        if decision['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE':
            after_discarded=self.warm is not None;self.warm=None
            decision['probe_warm_policy']=dict(version='v43_probe_only_cold_CEM',reuse_enabled=False,
                incoming_probe_memory_discarded=before_discarded,output_probe_memory_discarded=after_discarded,
                nonprobe_memory_policy_unchanged=True,nominal_action_forced=False,
                action_postprocessed=False,actual_state_or_actuator_h_modified=False)
            if decision.get('plan') is not None:
                decision['plan'].setdefault('diagnostics',{}).update(probe_warm_reuse_enabled=False,
                    probe_warm_output_discarded=after_discarded,nonprobe_warm_policy_unchanged=True)
        return decision


def run_public_grasp_cold_probe_v43(backend,executor,output,config,*,recorder=None):
    if type(backend) is not ObservedPublicGraspV41Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_cold_probe_v43') is not True:raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:raise ValueError('100ms/2actual50ms budget required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    coordinator=ColdProbeGraspV43Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery)
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    write_new(output/'public_grasp_cold_probe_v43_provenance.json',provenance)
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P3_cold_probe_v43_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_grasp_cold_probe_v43_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
    return dict(stage='P3_cold_probe_grasp_v43_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),real_time_guaranteed=False)
