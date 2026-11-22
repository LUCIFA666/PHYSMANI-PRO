"""New typed candidate provider; V72 phase/CEM/qualification/dispatch unchanged.

The new observer inherits the exact old sample/tick/pack/maintenance/proof chain.
The runner is a static copy of V72 with only its observer/coordinator type names
changed. No WorldTCP intent or Native held information is injected.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,time
import numpy as np
import torch
from ..local_mpc_control_20261103 import public_candidate_stable_dwell_v72 as old
from ..local_mpc_control_20261103 import public_candidate_pixel_centre_reverification_v70 as observer
from ..local_mpc_control_20261103 import public_grasp_candidates_v5 as candidate_module
from ..local_mpc_control_20261103.public_grasp_candidates_v5 import CandidateGraspV5Coordinator
from ..local_mpc_control_20261103.public_candidate_stable_dwell_v72 import (
    ObservedPublicCandidateStableDwellV72Backend,CandidateStableDwellV72Coordinator,
    _validate_v72_runner,_public_completion_ready,_public_completion_reason,
    PublicRobotSensors,build_response_planner,ROOT,append,write_new,quantiles,asdict,LAZY_VERSION)
from ..local_mpc_p4_20261103.runtime import LearnedCandidates
from .candidate_runtime import (candidate_loader,FrozenCubeCandidateInferenceV1,
    CubeLearnedCandidatesV1,CubeSymmetricLearnedCandidatesV1)
base=observer.base
PublicCandidatePixelCentreReverificationBridgeV70=observer.PublicCandidatePixelCentreReverificationBridgeV70


class ObservedCubeV72Backend(ObservedPublicCandidateStableDwellV72Backend):
    def __init__(self, backend, output, *, checkpoint, checkpoint_sha256, refiner_checkpoint, refiner_checkpoint_sha256, goal_sphere_filter=True, postconfirmed_maintenance=True, active_config=None, public_bridge=None, recorder=None, save_frames=True, grasp_config=None, require_public_goal_completion=False, candidate_mode='legacy_pilot'):
        if type(require_public_goal_completion) is not bool:
            raise ValueError('Explicit public goal flag required')
        torch.set_num_threads(2)
        bridge = public_bridge if public_bridge is not None else PublicCandidatePixelCentreReverificationBridgeV70(refiner_checkpoint=refiner_checkpoint, refiner_checkpoint_sha256=refiner_checkpoint_sha256, goal_sphere_filter=goal_sphere_filter, postconfirmed_maintenance=postconfirmed_maintenance, active_config=active_config)
        if type(bridge) is not PublicCandidatePixelCentreReverificationBridgeV70:
            raise TypeError('Exact independent natural-epoch bridge required')
        if (bridge.goal_sphere_filter, bridge.postconfirmed_maintenance) != (goal_sphere_filter, postconfirmed_maintenance):
            raise ValueError('Matching frozen measurement modifiers required')
        if bridge.tracker.refiner_checkpoint != Path(refiner_checkpoint).resolve(strict=True) or bridge.tracker.expected_refiner_sha256 != refiner_checkpoint_sha256:
            raise ValueError('Exact configured checkpoint identity required')
        bridge._validate_components()
        self.assembler = base.PublicCandidatePixelCentreStateAssemblerV70(bridge.binding)
        self.latest_robot = None
        self.recovery = base.ConfirmedAttemptRecovery()
        self.public_hold_metric = base.PublicLiftHoldMetric(grasp_config or base.ActiveVisualGraspConfig())
        self.require_public_goal_completion = require_public_goal_completion
        self.public_goal_completion = None
        self.candidate_cache = base.PublicRowCache()
        self.pilot_inference = candidate_loader(checkpoint, checkpoint_sha256, candidate_mode)
        base.PublicShadowBackend.__init__(self, backend, output, bridge=bridge, recorder=recorder, save_frames=save_frames, controller_uses_public_forecast=True, observer_role='grasp_prototype', attachment_estimator_used=True)


class CubeV72Coordinator(CandidateStableDwellV72Coordinator):
    def __init__(self, *args, inference, cache, pi_symmetry_representative=False, cache_fixed_probe_nominal_ik=False, probe_terminal_at_fixed_deadline=False, probe_onpolicy_joint_response=False, **kwargs):
        super(CandidateGraspV5Coordinator, self).__init__(*args, **kwargs)
        if type(pi_symmetry_representative) is not bool:
            raise ValueError('explicit boolean candidate representation protocol required')
        self.pi_symmetry_representative = pi_symmetry_representative
        if type(cache_fixed_probe_nominal_ik) is not bool:
            raise ValueError('explicit boolean fixed probe nominal protocol required')
        self.cache_fixed_probe_nominal_ik = cache_fixed_probe_nominal_ik
        if type(probe_terminal_at_fixed_deadline) is not bool:
            raise ValueError('explicit boolean fixed probe terminal-time protocol required')
        if probe_terminal_at_fixed_deadline and (not cache_fixed_probe_nominal_ik):
            raise ValueError('fixed probe deadline protocol requires the explicit cached destination protocol')
        self.probe_terminal_at_fixed_deadline = probe_terminal_at_fixed_deadline
        if type(probe_onpolicy_joint_response) is not bool:
            raise ValueError('explicit boolean on-policy probe response protocol required')
        if probe_onpolicy_joint_response and (not (cache_fixed_probe_nominal_ik and probe_terminal_at_fixed_deadline)):
            raise ValueError('on-policy probe response requires fixed nominal and absolute deadline protocols')
        self.probe_onpolicy_joint_response = probe_onpolicy_joint_response
        self.probe_execution_model = None
        self.probe_execution_base = None
        self.probe_response_receipt = None
        self.probe_nominal_cache = None
        self.probe_cache_sources = None
        if cache_fixed_probe_nominal_ik:
            from ..local_mpc_control_20261103.probe_nominal_cache_v53 import ProbeNominalQCache
            self.probe_nominal_cache = ProbeNominalQCache()
            self.probe_cache_sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(candidate_module.__file__), Path(candidate_module.__file__).with_name('probe_nominal_cache_v53.py'), Path(candidate_module.__file__).with_name('public_grasp_probe.py'), Path(candidate_module.__file__).with_name('joint_goals.py'), Path(candidate_module.__file__).with_name('joint_goal_cem.py'), Path(candidate_module.__file__).with_name('coupled_response.py'), ROOT / 'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json', ROOT / 'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt')}
            if probe_terminal_at_fixed_deadline:
                path = Path(candidate_module.__file__).with_name('probe_deadline_v54.py')
                self.probe_cache_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if probe_onpolicy_joint_response:
                from ..local_mpc_control_20261103.probe_onpolicy_response_v55 import probe_response_sources
                self.probe_cache_sources.update(probe_response_sources())
        if pi_symmetry_representative:
            from ..local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates
            self.geometry = (CubeSymmetricLearnedCandidatesV1(inference, cache) if type(inference) is FrozenCubeCandidateInferenceV1 else SymmetricLearnedCandidates(inference, cache))
        else:
            self.geometry = (CubeLearnedCandidatesV1(inference, cache) if type(inference) is FrozenCubeCandidateInferenceV1 else LearnedCandidates(inference, cache))
        self._seen_reverification_epoch = None


def run_cube_v72_live(backend, executor, output, config, *, recorder=None):
    if type(config.get('finger_response_fastpath', False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    _validate_v72_runner(backend, config)
    if type(backend) is not ObservedCubeV72Backend or executor.backend is not backend:
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
    coordinator = CubeV72Coordinator(core, helper, backend.assembler, backend.bridge.active, config=settings, recovery=backend.recovery, inference=backend.pilot_inference, cache=backend.candidate_cache, pi_symmetry_representative=config.get('candidate_pi_symmetry_representative', False), cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik', False), probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline', False), probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response', False))
    provenance = deepcopy(provenance)
    provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4', post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen', criteria=asdict(settings), active_attachment_criteria=asdict(backend.bridge.active.config), tracker_cache_binding=backend.bridge.binding.receipt(), legacy_attachment_controls_bridge=False, contact_response_qualified=False, force_or_frictional_grasp_proof=False, Native_labels_used_by_policy=False, previous_finger_response_checkpoint_requalified_for_completion_latch=False, public_hold_requires_actual_visible_4cm_and_halfsecond=True, formal_qualification=False, development_assumption='table_supported_target_without_independent_external_vertical_drive')
    provenance.update(candidate_transformer=backend.pilot_inference.receipt(), geometry_fallback_used=False, task_classifier_trained=False, calibrated_candidate_success_probability=False, candidate_pi_symmetry_representative=coordinator.pi_symmetry_representative, cache_fixed_probe_nominal_ik=coordinator.cache_fixed_probe_nominal_ik, probe_terminal_at_fixed_deadline=coordinator.probe_terminal_at_fixed_deadline, probe_onpolicy_joint_response=coordinator.probe_onpolicy_joint_response)
    provenance.update(optional_public_producer=backend.bridge.receipt())
    provenance.update(far_pixel_centre_v70=dict(offset=0.5, calibration_private=True, NN_and_initialSphere_inputs_unchanged=True, development_hypothesis_only=True, known_prior_counterexamples=73))
    provenance.update(response_seed=dict(version='V68_fresh_static12_domain_response_inverse', fresh_row='iteration0_candidate_first_eligible_local_index1', history_warm_used=False, CEM_scoring_rows=256, inverse_generation_compute_additional=True, original_selected_endpoint_and_7_path_preserved=True))
    provenance.update(lazy_free_v72=dict(version=LAZY_VERSION, applies_only_to='FREE_GRU', retained_topK=8, original_CEM_scoring_rows=256, original_RNG_elite_cost_model_unchanged=True, first_qualified_complete_bundle_before_phase_warm=True, original_endpoint_and_7_path_used_for_each_retained_action=True, final_runner_original_qualification_repeated=True, lazy_query_maximum=64, final_query_maximum=8, extra_query_and_wall_cost_separate=True, hold_probe_response_seed_unchanged=True, full_H8_or_contact_response_certified=False, Native_attachment_used=False))
    provenance.update(stable_dwell_v72=dict(applies_only_to='FREE_GRU_alignment_dwell', numerical_absolute_tolerance_seconds=1e-09, physical_dwell_threshold_seconds=0.1, scalar_and_vector_comparisons_updated=True, timestamp_modified=False, position_speed_orientation_deadline_and_budget_unchanged=True, guaranteed_grasp_success=False))
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
        if h['gripper_target'] not in (0.0, 1.0) and (not ever):
            decision = dict(mode='UNAVAILABLE', state=None, reason='initial_actual_open_warmup')
            action = None
            opening = 1.0
        else:
            decision = coordinator.decide(backend, backend.latest, robot, h)
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
        receipt = executor.execute_joint_target(goal, opening=opening, phase='P4_candidate_transformer_v72_' + decision['mode'])
        ever = True
        after = backend.latest_robot
        append(output / 'public_candidate_position_v72_decisions.jsonl', dict(index=cycles, time=robot.timestamp, public_robot=robot, actuator_state_before=h, decision=decision, action_rejected_reason=rejected, action_qualification=qualification, executed_action=executed, executor=receipt, public_robot_after=after, actuator_state_after=deepcopy(executor.state()), active_visual_after=deepcopy(backend.bridge.active.latest), recovery_after=backend.recovery.receipt(), public_lift_hold=backend.public_hold_metric.receipt(), Native_labels_used_by_policy=False))
        cycles += 1
        times.append(time.perf_counter() - started)
        if after is None or after.timestamp <= robot.timestamp + 1e-08:
            stop = 'no_actual_public_tick'
            break
        if _public_completion_ready(backend):
            stop = _public_completion_reason(backend)
            break
    return dict(stage='P4_public_candidate_stable_dwell_v72_development', complete=True, cycles=cycles, canary_stop_reason=stop, public_lift_hold_observed=backend.public_hold_metric.passed, public_lift_hold=backend.public_hold_metric.receipt(), Native_task_success_claimed=False, Native_success_exit_used=False, Native_labels_used_by_policy=False, contact_response_qualified=False, formal_qualification=False, precision_qualified=False, force_or_frictional_grasp_proof=False, mode_events=coordinator.mode_events, controller_provenance=provenance, Transformer_used=backend.pilot_inference.calls > 0, candidate_transformer=backend.pilot_inference.receipt(), candidate_events=coordinator.geometry.events, candidate_public_cache=backend.candidate_cache.receipt(), geometry_fallback_used=False, require_public_goal_completion=backend.require_public_goal_completion, public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(), optional_public_producer=backend.bridge.receipt(), effective_attachment_final=deepcopy(backend.bridge.latest_attachment), active_visual_final=deepcopy(backend.bridge.active.latest), recovery_final=backend.recovery.receipt(), full_cycle_wall_seconds=quantiles(times), full_cycle_wall_seconds_samples=times, real_time_guaranteed=False)
