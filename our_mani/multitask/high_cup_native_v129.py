"""TRAIN-only native consumer of the complete all300 highcup operation fit."""
from pathlib import Path
from types import MethodType
from copy import deepcopy
import json
from .task_native_backend_v69 import NativeActuatorDiagnosticBackend
from .task_actor_backend_v69 import persistent_reference_actor_model_preflight
from .bounded_approach_v121 import BoundedApproach,SCHEMA as PREFIX_SCHEMA
from .evaluation import file_sha

TASK='place_cups_on_rotating_frame_high_speed'


def binding(root,protocol):
    root=Path(root).resolve();pointer=protocol['high_cup_all300_report']
    path=root/pointer['path']
    if file_sha(path)!=pointer['sha256']:raise ValueError('highcup fitting report changed')
    report=json.loads(path.read_text())
    if (report['model']!=protocol['learned_artifacts'][TASK]['interaction']
            or report['actual_original_parents']!=100 or report['teacher_acquisitions']!=300
            or report['failed_fit_keys'] or report['required_phase_failures']):
        raise ValueError('complete all300 original-gate highcup library required')
    return dict(schema='r10_highcup_all300_original_gate_TRAIN_consumer_v129',report=pointer,model=report['model'],
        original_parent_count=100,teacher_acquisitions=300,all6_original_phase_gates_passed=True,
        sources_sha256={p:file_sha(root/p) for p in (
            'our_mani/multitask/high_cup_native_v129.py','our_mani/multitask/bounded_approach_v121.py',
            'our_mani/multitask/high_cup_issue_goal_v122.py')},
        bounded_3D_prefix=True,fitted_lag_reserved_inside_original_horizon=True,
        original_motion_collision_clock_grasp_gates_not_loosened=True,TRAIN_only=True,formal_VAL_TEST=False)


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    args=prepared['args'];protocol=prepared['config'];root=Path(root).resolve()
    if args.split!='train' or args.task!=TASK or args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('highcup all300 development accepts only fixed TRAIN geometry cases')
    proof=binding(root,protocol)
    if protocol.get('high_cup_native_v129')!=proof:raise ValueError('freeze actual highcup native consumer')
    for name,checksum in proof['sources_sha256'].items():
        if locked_digest(root/name,prepared['files'])!=checksum:raise ValueError('highcup consumer source omitted')
    report=json.loads((root/proof['report']['path']).read_text())
    locked_digest(root/proof['report']['path'],prepared['files'])
    for name,checksum in report['verified_files_sha256'].items():
        if locked_digest(Path(name),prepared['files'])!=checksum:raise ValueError('actual all300 fit source changed')
    return dict(persistent_reference_actor_model_preflight(root,prepared),highcup_all300_consumer=proof)


class HighCupNativeBackend(NativeActuatorDiagnosticBackend):
    def actor(self,prepared,env,descriptions):
        root=Path(__file__).resolve().parents[2];proof=binding(root,prepared['config'])
        if prepared['config'].get('high_cup_native_v129')!=proof:raise ValueError('highcup consumer binding differs')
        actor,identity=super().actor(prepared,env,descriptions);manager=actor.continuous_reference_manager
        if manager.confirmations or actor.controller.feedback_step_calls:raise ValueError('install before first step')
        raw=deepcopy(manager.approach_transit.normalized);raw['version']=PREFIX_SCHEMA
        manager.approach_transit=BoundedApproach(manager,raw)
        previous=actor.metrics_report
        def metrics(instance):return dict(previous(),highcup_all300_native_v129=deepcopy(proof))
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity
