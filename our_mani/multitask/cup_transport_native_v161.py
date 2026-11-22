"""Bounded cup transport with one dispatch-confirmed, nonrenewable phase clock."""
from pathlib import Path
from types import MethodType
from copy import deepcopy
from contextlib import nullcontext
from .cup_support_native_v149 import NormalCupSupportBackend,HighCupSupportBackend
from .cup_support_native_v149 import model_preflight as original_preflight
from .normal_cup_native_v111 import library_scope
from .normal_cup_primary_library_v109 import PrimarySequenceLibrary
from .interaction import InteractionLibrary,CandidateConfig
from .transport_phase_clock_v158 import TransportPhaseClock
from .bounded_transport_v159 import BoundedApproach,SCHEMA
from .transport_endpoint_candidates_v160 import TransportEndpointCandidateGenerator
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha

TASKS=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')
SOURCES=('our_mani/multitask/cup_transport_native_v161.py','our_mani/multitask/transport_phase_clock_v158.py',
    'our_mani/multitask/bounded_transport_v159.py','our_mani/multitask/transport_endpoint_candidates_v160.py')


def binding(root,protocol,task):
    root=Path(root).resolve()
    if task not in TASKS:raise ValueError('only original two-cup placement models')
    cls=PrimarySequenceLibrary if task==TASKS[0] else InteractionLibrary
    library=cls.load(root/protocol['learned_artifacts'][task]['interaction']['path'])
    clock=TransportPhaseClock(library,CandidateConfig(**protocol['candidate_config']))
    return dict(schema='r10_cup_transport_fixed_confirmed_phase_endpoint_policy_v161',task=task,
        sources_sha256={name:file_sha(root/name) for name in SOURCES},
        transport_original_maximum_budget_seconds={str(i):v for i,v in clock.budgets.items()},
        epoch_start='first actual counted partial transport dispatch, never an unissued proposal',
        later_endpoint_budget='min(original candidate budget, original transport deadline minus current counted time)',
        prefix_forecast_time='current0.5s segment end plus fitted lag, within original causal horizon',
        stage_deadlines_reset_or_extended=False,global10s_and_original_suffix_reserved=True,
        original_motion_collision_grasp_success_gates_unchanged=True,formal_VAL_TEST=False)


def install_confirmation(manager,clock):
    original=manager.confirm_actual_issued
    def confirm(instance,**kwargs):
        active=deepcopy(instance.approach_transit.active)
        cert=kwargs['command'].get('feedforward_execution')
        proposed=bool(instance.plan_kind=='approach_transit' and active and active['phase']=='transport'
            and cert and cert.get('candidate_id')==instance.plan_candidate_id)
        result=original(**kwargs)  # Original full backend acknowledgement must pass first.
        execution=kwargs['execution']
        if proposed and not instance.qualification_failed:
            if (execution.get('actual_window_velocity_within_limits') is True
                    and execution.get('actual_window_acceleration_within_limits') is True):
                clock.confirm(active['phase_index'],float(execution['actual_issued_at']),float(kwargs['time']),active['deadline'])
        return result
    manager.confirm_actual_issued=MethodType(confirm,manager)


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('cup_transport_v161',{}).get(task)!=expected:raise ValueError('freeze transport phase-clock policy')
    c=actor.controller;m=actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first step')
    old=c.generator
    with library_scope(root,protocol,task) if task==TASKS[0] else nullcontext():
        generator=TransportEndpointCandidateGenerator(old.library,old.reachability,old.collision_check,old.config,
            ring_affordance=old.ring_affordance,allow_development_ring_affordance=old.allow_development_ring_affordance)
    generator.placement_clearance=old.placement_clearance
    clock=TransportPhaseClock(c.library,c.generator.config);generator.transport_clock=clock
    c.generator=generator;m.transport_phase_clock=clock
    raw=deepcopy(m.approach_transit.normalized);raw['version']=SCHEMA;m.approach_transit=BoundedApproach(m,raw)
    install_confirmation(m,clock)
    before=actor.metrics_report
    def metrics(instance):return dict(before(),cup_transport_v161=dict(binding=expected,events=deepcopy(clock.events),
        epochs=deepcopy(clock.epochs),actual_endpoint_selection_time_unchanged=True,policy_commands_changed=True,
        original_deadlines_not_renewed=True,phase_success_not_claimed_by_clock=True))
    actor.metrics_report=MethodType(metrics,actor)


def model_preflight(root,prepared):
    result=original_preflight(root,prepared);task=prepared['args'].task;expected=binding(root,prepared['config'],task)
    if prepared['config'].get('cup_transport_v161',{}).get(task)!=expected:raise ValueError('actual transport phase-clock binding differs')
    checked_subset(root,prepared['files'],{str(Path(root)/name):sha for name,sha in expected['sources_sha256'].items()})
    return dict(result,cup_transport_v161=expected,policy_commands_changed=True,physical_gates_changed=False)


class NormalCupTransportBackend(NormalCupSupportBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity


class HighCupTransportBackend(HighCupSupportBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
