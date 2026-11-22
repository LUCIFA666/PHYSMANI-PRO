"""Search larger cup transit proposals; retain original hard native gates."""
from pathlib import Path
from copy import deepcopy
from types import MethodType
from .cup_physical_native_v168 import NormalCupPhysicalBackend, HighCupPhysicalBackend
from .cup_physical_native_v168 import model_preflight as original_preflight
from .bounded_transit_v180 import BoundedApproach, SCHEMA
from .cup_physical_acquire_v167 import TASKS
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def binding(root, protocol, task):
    if task not in TASKS: raise ValueError('only two cup placement development tasks')
    config = protocol['policy_overlay']['configurations']['persistent_reference']['approach_transit']
    limit = protocol['mpc_config']['maximum_linear_speed']
    maximum_step = min(.4, .5*limit*config['segment_seconds'])
    if not maximum_step > config['maximum_TCP_step_m']:
        raise ValueError('new search must explicitly extend the previous proposal range')
    return dict(schema='r10_cup_larger_transit_proposal_policy_v181', task=task,
        sources_sha256={p:file_sha(Path(root)/p) for p in (
            'our_mani/multitask/cup_transit_native_v181.py', 'our_mani/multitask/bounded_transit_v180.py')},
        previous_heuristic_step_m=config['maximum_TCP_step_m'], maximum_proposal_step_m=maximum_step,
        proposal_scale_rule='half original MPC linear-speed scale times original segment duration; proposal heuristic only',
        previous_8cm_fraction_proposals_still_searched=True,
        additional_sampled_Cartesian_reference_gates=dict(linear=limit, angular=protocol['mpc_config']['maximum_angular_speed']),
        candidate_proposal_range_increased=True, original_joint_collision_brake_clock_grasp_success_gates_unchanged=True,
        actual_motion_guaranteed=False, model_fitting_caps_unchanged=True, formal_VAL_TEST=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('cup_transit_v181', {}).get(task) != expected:
        raise ValueError('freeze actual larger transit policy')
    manager = actor.continuous_reference_manager
    if manager.confirmations or actor.controller.feedback_step_calls:
        raise ValueError('install before first control step')
    raw = deepcopy(manager.approach_transit.normalized)
    raw.update(version=SCHEMA, maximum_TCP_step_m=expected['maximum_proposal_step_m'])
    manager.approach_transit = BoundedApproach(manager, raw)
    previous = actor.metrics_report
    def metrics(instance): return dict(previous(), cup_transit_v181=deepcopy(expected))
    actor.metrics_report = MethodType(metrics, actor)


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('cup_transit_v181', {}).get(prepared['args'].task) != expected:
        raise ValueError('actual larger transit binding differs')
    checked_subset(root, prepared['files'], {str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return dict(result, cup_transit_v181=expected, candidate_proposal_range_increased=True)


class NormalCupTransitBackend(NormalCupPhysicalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity


class HighCupTransitBackend(HighCupPhysicalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
