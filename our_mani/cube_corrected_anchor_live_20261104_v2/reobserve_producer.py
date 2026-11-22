"""Optional add-only reobserve profile using the same corrected current anchor."""
from ..cube_bounded_reobserve_live_20261104_v1.producer import *
from ..cube_bounded_reobserve_live_20261104_v1.producer import _validate_reobserve_runner,_public_completion_ready,_public_completion_reason
from .binding import AnchorConsistentPublicStateAssemblerV2
from .sources import anchor_consistent_corrected_source_map

class ObservedAnchorConsistentReobserveCubeV72Backend(ObservedBoundedReobserveCubeV72Backend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.assembler=AnchorConsistentPublicStateAssemblerV2(self.bridge.binding)

    def receipt(self):
        value=super().receipt()
        value.update(current_derivative_binding=self.assembler.receipt(),
            anchor_consistent_live_source_sha256=anchor_consistent_corrected_source_map())
        return value


def _validate_anchor_reobserve_runner(backend,config):
    if (type(backend.assembler)is not AnchorConsistentPublicStateAssemblerV2
            or backend.assembler.binding is not backend.bridge.binding
            or getattr(backend.assembler.build,'__func__',None)is not AnchorConsistentPublicStateAssemblerV2.build):
        raise TypeError('Exact same real capability and new anchor-consistent assembler required')
    _validate_reobserve_runner(backend,config)
    anchor_consistent_corrected_source_map()


def run_anchor_consistent_reobserve_cube_v72(backend, executor, output, config, *, recorder=None):
    if type(config.get('finger_response_fastpath', False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    _validate_anchor_reobserve_runner(backend, config)
    if type(backend) is not ObservedAnchorConsistentReobserveCubeV72Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_candidate_transformer_v5') is not True:
        raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt', 0.1)) != 0.1 or float(config.get('physics_dt', 0.05)) != 0.05:
        raise ValueError('100ms/2actual50ms budget required')
    if backend.samples == 0:
        backend.sample()
    output = Path(output)
    settings = backend.public_hold_metric.config
    core, helper, provenance = build_response_planner(ROOT, executor.limits, controller='coupled', action_space='joint', joint_goal_config={'lower': tuple(config.get('workspace_lower', [-0.1, -0.65, 0.755]))}, terminal_config={'enabled': bool(config.get('terminal_goal_cost', True))}, reference_frequency=executor.config.reference_frequency)
    if config.get('finger_response_fastpath', False):
        from ..local_mpc_control_20261103.finger_fastpath_integration_20261103 import enable_finger_fastpath
        provenance = enable_finger_fastpath(core.execution_model, provenance)
    coordinator = MatchedCorrectedCubeV72Coordinator(core, helper, backend.assembler, backend.bridge.active, config=settings, recovery=backend.recovery, inference=backend.pilot_inference, cache=backend.candidate_cache, pi_symmetry_representative=config.get('candidate_pi_symmetry_representative', False), cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik', False), probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline', False), probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response', False))
    provenance = deepcopy(provenance)
    provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4', post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen', criteria=asdict(settings), active_attachment_criteria=asdict(backend.bridge.active.config), tracker_cache_binding=backend.bridge.binding.receipt(), legacy_attachment_controls_bridge=False, contact_response_qualified=False, force_or_frictional_grasp_proof=False, Native_labels_used_by_policy=False, previous_finger_response_checkpoint_requalified_for_completion_latch=False, public_hold_requires_actual_visible_4cm_and_halfsecond=True, formal_qualification=False, development_assumption='table_supported_target_without_independent_external_vertical_drive')
    provenance.update(candidate_transformer=backend.pilot_inference.receipt(), geometry_fallback_used=False, task_classifier_trained=False, calibrated_candidate_success_probability=False, candidate_pi_symmetry_representative=coordinator.pi_symmetry_representative, cache_fixed_probe_nominal_ik=coordinator.cache_fixed_probe_nominal_ik, probe_terminal_at_fixed_deadline=coordinator.probe_terminal_at_fixed_deadline, probe_onpolicy_joint_response=coordinator.probe_onpolicy_joint_response)
    provenance.update(optional_public_producer=backend.bridge.receipt())
    provenance.update(far_pixel_centre_v70=dict(offset=0.5, calibration_private=True, NN_and_initialSphere_inputs_unchanged=True, development_hypothesis_only=True, known_prior_counterexamples=73))
    provenance.update(response_seed=dict(version='V68_fresh_static12_domain_response_inverse', fresh_row='iteration0_candidate_first_eligible_local_index1', history_warm_used=False, CEM_scoring_rows=256, inverse_generation_compute_additional=True, original_selected_endpoint_and_7_path_preserved=True))
    provenance.update(lazy_free_v72=dict(version=LAZY_VERSION, applies_only_to='FREE_GRU', retained_topK=8, original_CEM_scoring_rows=256, original_RNG_elite_cost_model_unchanged=True, first_qualified_complete_bundle_before_phase_warm=True, original_endpoint_and_7_path_used_for_each_retained_action=True, final_runner_original_qualification_repeated=True, lazy_query_maximum=64, final_query_maximum=8, extra_query_and_wall_cost_separate=True, hold_probe_response_seed_unchanged=True, full_H8_or_contact_response_certified=False, Native_attachment_used=False))
    provenance.update(stable_dwell_v72=dict(applies_only_to='FREE_GRU_alignment_dwell', numerical_absolute_tolerance_seconds=1e-09, physical_dwell_threshold_seconds=0.1, scalar_and_vector_comparisons_updated=True, timestamp_modified=False, position_speed_orientation_deadline_and_budget_unchanged=True, guaranteed_grasp_success=False))
    provenance.update(bounded_reobserve=backend.reobserve_lifecycle.receipt(), reobserve_is_separate_from_selected_MPC_action=True, rejected_reobserve_uses_one_original_hold_cycle=True)
    write_new(output / 'public_candidate_stable_dwell_v72_provenance.json', provenance)
    maximum = int(config.get('public_grasp_max_decisions', np.ceil(float(config.get('simulation_seconds', 6.0)) / 0.1)))
    if maximum <= 0:
        raise ValueError('positive decision budget required')
    cycles = 0
    times = []
    ever = False
    stop = 'budget_exhausted'
    while cycles < maximum and backend.remaining_ticks() >= 2:
        if any((p.exists() for p in (output / 'STOP', output.parent / 'STOP', ROOT / 'runtime/framework/local_mpc_20261102/STOP'))):
            stop = 'administrative_stop'
            break
        if _public_completion_ready(backend):
            stop = _public_completion_reason(backend)
            break
        started = time.perf_counter()
        robot = backend.latest_robot
        h = deepcopy(executor.state())
        if not isinstance(robot, PublicRobotSensors) or abs(robot.timestamp - backend.clock()) > 1e-08:
            raise ValueError('fresh public robot frame required')
        stale_reset = clear_free_state_on_stale(coordinator, backend.reobserve_lifecycle)
        if h['gripper_target'] not in (0.0, 1.0) and (not ever):
            decision = dict(mode='UNAVAILABLE', state=None, reason='initial_actual_open_warmup')
            action = None
            opening = 1.0
        else:
            decision = coordinator.decide(backend, backend.latest, robot, h)
            decision = guard_new_close(decision, coordinator, backend.reobserve_lifecycle, h)
            action = decision.get('action')
            opening = float(h['gripper_target'])
        rejected = None
        qualification = None
        if action is None:
            coordinator.warm = None
            goal = robot.q.copy()
            if not ever and (not backend.recovery.preserves_closed_attempt(backend.bridge.active.attempt_id)):
                opening = 1.0
            executed = dict(joint_goal=goal.copy(), opening=opening, source='actual_q_hold_keep_requested_gripper')
        else:
            opening = float(action['opening'])
            if opening not in (0.0, 1.0):
                raise ValueError('binary modeled opening required')
            if decision['mode'] == 'REOPENING_UNKNOWN_CONTACT' and opening != 1.0:
                rejected = 'reopen_model_proposed_close'
            if decision['mode'] == 'ACTIVE_VISUAL_VERIFICATION_PROBE' and opening != 0.0:
                rejected = 'probe_model_proposed_open'
            goal, reason, qualification = helper.qualify_action(backend, decision['state'], action, require_predicted_joint_goal=True)
            rejected = rejected or reason
            if rejected:
                goal = robot.q.copy()
                opening = float(h['gripper_target'])
                coordinator.warm = None
                executed = dict(joint_goal=goal.copy(), opening=opening, source='rejected_model_action_hold_actual_q_keep_request')
            else:
                if not np.array_equal(goal, np.asarray(action['joint_goal'])):
                    raise ValueError('selected modeled joint goal changed before dispatch')
                executed = dict(joint_goal=goal.copy(), opening=opening, source='exact_selected_modeled_q_and_opening')
                coordinator.note_executed(action, robot)
        lazy = ((decision.get('plan') or {}).get('diagnostics') or {}).get('V71_lazy_free_qualification', {})
        decision['V71_static_collision_query_counts'] = dict(lazy_first_action_queries=int(lazy.get('Native_collision_queries', 0)), final_runner_action_queries=len((qualification or {}).get('collision_probes', [])), lazy_plus_final_queries=int(lazy.get('Native_collision_queries', 0)) + len((qualification or {}).get('collision_probes', [])), excludes_original_candidate_nominal_queries=True, original_final_check_skipped=False)
        receipt, executed, reobserve = execute_selected_or_reobserve(backend, executor, decision, goal, opening, executed)
        decision['bounded_reobserve_static_collision_query_count'] = len((((reobserve or {}).get('outcome') or {}).get('qualification') or {}).get('probes', []))
        ever = True
        after = backend.latest_robot
        append(output / 'public_candidate_position_v72_decisions.jsonl', dict(index=cycles, time=robot.timestamp, public_robot=robot, actuator_state_before=h, decision=decision, action_rejected_reason=rejected, action_qualification=qualification, executed_action=executed, executor=receipt, current_derivative_binding=deepcopy(backend.assembler.last_derivative_binding), bounded_reobserve=reobserve, stale_free_reset=stale_reset, public_robot_after=after, actuator_state_after=deepcopy(executor.state()), active_visual_after=deepcopy(backend.bridge.active.latest), recovery_after=backend.recovery.receipt(), public_lift_hold=backend.public_hold_metric.receipt(), Native_labels_used_by_policy=False))
        cycles += 1
        times.append(time.perf_counter() - started)
        if after is None or after.timestamp <= robot.timestamp + 1e-08:
            stop = 'no_actual_public_tick'
            break
        if _public_completion_ready(backend):
            stop = _public_completion_reason(backend)
            break
    return dict(bounded_reobserve=backend.reobserve_lifecycle.receipt(), reobserve_enabled=True, stage='P4_matched_corrected_bounded_reobserve_development', complete=True, cycles=cycles, canary_stop_reason=stop, public_lift_hold_observed=backend.public_hold_metric.passed, public_lift_hold=backend.public_hold_metric.receipt(), Native_task_success_claimed=False, Native_success_exit_used=False, Native_labels_used_by_policy=False, contact_response_qualified=False, formal_qualification=False, precision_qualified=False, force_or_frictional_grasp_proof=False, mode_events=coordinator.mode_events, controller_provenance=provenance, Transformer_used=backend.pilot_inference.calls > 0, candidate_transformer=backend.pilot_inference.receipt(), candidate_events=coordinator.geometry.events, candidate_public_cache=backend.candidate_cache.receipt(), geometry_fallback_used=False, require_public_goal_completion=backend.require_public_goal_completion, public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(), optional_public_producer=backend.bridge.receipt(), effective_attachment_final=deepcopy(backend.bridge.latest_attachment), active_visual_final=deepcopy(backend.bridge.active.latest), recovery_final=backend.recovery.receipt(), full_cycle_wall_seconds=quantiles(times), full_cycle_wall_seconds_samples=times, real_time_guaranteed=False)
