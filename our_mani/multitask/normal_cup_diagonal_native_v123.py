"""Bounded 3D cup approach, preserving lag reservation and physical gates."""
from pathlib import Path
from copy import deepcopy
from types import MethodType
from .normal_cup_lag_native_v118 import NormalCupNativeBackend as OriginalBackend
from .normal_cup_lag_native_v118 import model_preflight as original_preflight
from .bounded_approach_v121 import BoundedApproach,SCHEMA
from .evaluation import file_sha


def binding(root,protocol):
    root=Path(root).resolve()
    return dict(schema='r10_normal_cup_bounded_3D_approach_overlay_v123',
        sources_sha256={p:file_sha(root/p) for p in (
            'our_mani/multitask/normal_cup_diagonal_native_v123.py',
            'our_mani/multitask/bounded_approach_v121.py')},
        original_prefix_configuration=deepcopy(protocol['policy_overlay']['configurations']['persistent_reference']['approach_transit']),
        horizontal_before_vertical_heuristic_removed=True,
        fitted_lag_reserved_inside_original_forecast_horizon=True,
        original_motion_collision_clock_grasp_gates_unchanged=True)


def install(actor,protocol,root):
    expected=binding(root,protocol)
    if protocol.get('normal_cup_diagonal_v123')!=expected:raise ValueError('freeze bounded3D implementation')
    manager=actor.continuous_reference_manager
    if manager.confirmations or actor.controller.feedback_step_calls:raise ValueError('install before first control step')
    raw=deepcopy(manager.approach_transit.normalized);raw['version']=SCHEMA
    manager.approach_transit=BoundedApproach(manager,raw)
    before=actor.metrics_report
    def metrics(instance):return dict(before(),bounded_3D_lag_reservation_v123=deepcopy(expected))
    actor.metrics_report=MethodType(metrics,actor)


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    result=original_preflight(root,prepared);expected=binding(root,prepared['config'])
    if prepared['config'].get('normal_cup_diagonal_v123')!=expected:raise ValueError('actual bounded3D differs')
    for p,sha in expected['sources_sha256'].items():
        if locked_digest(Path(root)/p,prepared['files'])!=sha:raise ValueError('bounded3D source omitted from input lock')
    return dict(result,bounded3D_overlay=expected)


class NormalCupNativeBackend(OriginalBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2])
        return actor,identity
