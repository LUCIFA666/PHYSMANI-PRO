"""Independent TRAIN native-task outcomes, never waypoint completion or runtime input."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import acquisition_lineage as a,command_goal_reference as goal

SCHEMA='r10_TRAIN_verified_native_task_success_labels_v1'
TASKS=('push_moving_button','push_moving_button_high_speed')


def require(value,message):
    if not value:raise ValueError('native success labels: '+message)


@dataclass(frozen=True)
class TRAINVerifiedNativeTaskSuccessCompletion:
    task_name:str
    source_id:str
    parent_source_group:str
    command_id:int
    terminal_frame:int
    terminal_time:float
    raw_command_completed:bool
    raw_waypoint_arrival_observed:bool
    raw_bindings:dict
    original_command_ledger_sha256:str
    original_completion_events_sha256:str
    original_masks_sha256:dict
    native_outcome:dict
    original_actual_tcp_xyzw:tuple
    original_actual_reference_xyzw:tuple
    native_task_source_sha256:str
    provenance:dict

    def __post_init__(self):
        require(self.task_name in TASKS and self.parent_source_group.startswith('train:'+self.task_name+':')
            and type(self.command_id) is int and self.command_id>=0 and type(self.terminal_frame) is int and self.terminal_frame>0
            and np.isfinite(self.terminal_time) and self.terminal_time>0
            and type(self.raw_command_completed) is bool and type(self.raw_waypoint_arrival_observed) is bool,
            'explicit original TRAIN terminal identity required')
        require(self.native_outcome.get('success') is True and self.provenance.get('schema')==SCHEMA
            and self.provenance.get('canonical_record_and_arrays_verified') is True
            and self.provenance.get('native_success_is_waypoint_completion') is False
            and self.provenance.get('geometry_tolerance_override_allowed') is False
            and self.provenance.get('label_is_runtime_observation') is False,'native success cannot become a geometric or runtime claim')
        require(set(self.raw_bindings)=={'canonical_receipt','cache','commands'} and len(self.native_task_source_sha256)==64,
            'actual raw/source binding required')


@dataclass(frozen=True)
class TRAINNativeTaskSuccessEvidence:
    task_name:str
    source_id:str
    parent_source_group:str
    status:str
    raw_task_success:bool
    raw_terminal_failure:bool
    budget_censored:bool
    success_completion:TRAINVerifiedNativeTaskSuccessCompletion|None
    raw_bindings:dict
    raw_completion_events_sha256:str
    original_masks_sha256:dict
    derivation_proof:dict

    def __post_init__(self):
        require(self.task_name in TASKS and self.status in ('verified_native_task_success','original_unsuccessful_budget_censored','original_native_terminal_failure')
            and self.derivation_proof.get('actual_raw_source_verified') is True,'explicit positive or retained negative/censored evidence required')
        require((self.success_completion is not None)==self.raw_task_success,'native success positive label/event mismatch')
        require(not (self.raw_task_success and self.raw_terminal_failure),'success and native terminal failure conflict')


def extract(case,source,rule):
    require(case['entry']['task'] in TASKS and case['entry']['split']=='train','only explicit button TRAIN sources allowed')
    # A caller must not replace the native outcome in an otherwise valid raw view.
    require(a.read(Path(case['path']))==case['record'],'in-memory canonical native outcome/record differs from actual bytes')
    original=goal.verify_original_source(case,source,rule);r=case['record'];z=case['arrays'];align=r['alignment'];outcome=align['native_outcome']
    success=outcome['success'];failure=outcome['terminal_failure']
    require(type(success) is bool and type(failure) is bool and align['native_success_latched'] is success
        and align['native_terminal_failure_latched'] is failure,'native outcome/latches differ')
    raw=case['command_receipt'];events=[e for e in raw['phase_completion_events'] if e['evidence']=='TRAIN_native_success_during_executed_final_required_phase']
    completion=None
    if success:
        require(len(events)==1 and align['native_terminal_latched'] is True and not failure,'one exact actual native terminal event required')
        event=events[0];cid=event['origin_command_id'];command=next(c for c in raw['commands'] if c['command_id']==cid)
        end=len(z['times'])-1
        require(event['phase']==command['raw_phase']=='press' and event['to_phase']=='native_success'
            and align['native_terminal_tick']==end
            and abs(end*.05-event['attempt_end_time'])<=1e-8 and abs(z['times'][end]-event['attempt_end_time'])<=1e-8
            and event['origin_command_completed']==('command_completed_at' in command)
            and np.array_equal(np.asarray(event['current_measured_tcp_xyzw']),z['measured_tcp_xyzw'][end]),
            'task success must bind final actual press event/tick/TCP independently of command completion')
        completion=TRAINVerifiedNativeTaskSuccessCompletion(source.task_name,source.source_id,source.grouping_source,cid,end,
            float(z['times'][end]),'command_completed_at' in command,'measured_arrival_at' in command,original.raw_bindings,
            original.original_command_ledger_sha256,original.original_completion_events_sha256,original.original_masks_sha256,
            dict(outcome),tuple(event['current_measured_tcp_xyzw']),tuple(event['current_reference_native_pose_xyzw']),align['native_task_source_sha256'],
            dict(schema=SCHEMA,canonical_record_and_arrays_verified=True,native_success_is_waypoint_completion=False,
                geometry_tolerance_override_allowed=False,label_is_runtime_observation=False,
                original_raw_command_completion_and_censoring_preserved=True))
    else:require(not events,'negative native outcome contradicts original success event')
    return TRAINNativeTaskSuccessEvidence(source.task_name,source.source_id,source.grouping_source,
        'verified_native_task_success' if success else 'original_native_terminal_failure' if failure else 'original_unsuccessful_budget_censored',
        success,failure,not success and not failure,completion,original.raw_bindings,original.original_completion_events_sha256,
        original.original_masks_sha256,dict(schema=SCHEMA,actual_raw_source_verified=True,
            actual_source_snapshot=original.derivation_proof['actual_source_snapshot'],
            verified_files_sha256=original.derivation_proof['verified_files_sha256'],helper=dict(path=str(Path(__file__).resolve()),sha256=a.file_sha(__file__)),
            geometry_BC_or_runtime_gate_modified=False))
