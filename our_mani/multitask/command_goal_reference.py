"""Opt-in TRAIN command-goal intent, separate from actual terminal metrics.

The frozen collector does not import this helper. Production skill fitting
uses its explicit TRAIN rule; runtime receives an already
computed causal CV/Context candidate pose and never a TRAIN label object.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..contracts import Pose
from . import acquisition_lineage as a
from . import interaction
from .training_view import validate_arrays

SCHEMA='r10_TRAIN_command_goal_reference_intent_v1'
RULE_SCHEMA='r10_TRAIN_source_backed_command_goal_reference_rule_draft_v1'
FIT_RULE_SCHEMA='r10_TRAIN_source_backed_command_goal_reference_rule_v4'
FIT_RULE_SCHEMA_V5='r10_TRAIN_source_backed_command_goal_reference_rule_v5'
FIT_RULE_SCHEMA_V6='r10_TRAIN_source_backed_command_goal_reference_rule_v6'
OWNED_COUNTED_REGISTRY_PATH='artifacts/framework/multitask_r10/TRAIN_high_cup_requery_all100_candidate_v2_setup/execution_registration_approved.json'
OWNED_COUNTED_REGISTRY_SHA='9e00a1020f7624afe182774bbf8df8016e5e7b176cc4895b7f94eb928d6b126b'
OWNED_COUNTED_PLAN_FILE_SHA='84f5e2bce3b109ffdf934c23eb1d1df00bd0711a1c040de81c18b743d31dd2b6'
OWNED_COUNTED_ARCHIVE_SHA='61d5c516a8d42278a2a832011f33bdf6764a146de7ae2a8b4a071ba48530a19c'
INTENT_BASIS='command_goal_reference'
ISSUE='original_issue_pose'
FORECAST='original_expert_rotating_forecast'
BIN_FORECAST='original_expert_bin_unreflected_forecast'
BIN_SELECTOR='original_expert_bin_open_forecast_closed_issue'
BASKETBALL_FORECAST='original_expert_hoop_unreflected_forecast'
TASK_PHASES={
    'beat_the_rotating_buzz':{'approach':ISSUE,'acquire':FORECAST,'follow':FORECAST,'release':FORECAST},
    'beat_the_rotating_buzz_high_speed':{'approach':ISSUE,'acquire':FORECAST,'follow':FORECAST,'release':FORECAST},
    'insert_onto_rotating_peg':{'align':FORECAST,'insert':FORECAST},
    'insert_onto_rotating_peg_high_speed':{'align':FORECAST,'insert':FORECAST},
    'place_cups_on_rotating_frame_high_speed':{'transport':ISSUE},
    'put_rubbish_in_moving_bin':{'deposit':BIN_SELECTOR},
    'moving_basketball_in_hoop':{'deposit':BASKETBALL_FORECAST},
    'moving_basketball_in_hoop_high_speed':{'deposit':BASKETBALL_FORECAST},
}
HIGH_BIN_FORECAST='original_expert_high_bin_unreflected_forecast'
TASK_PHASES_V5=dict(TASK_PHASES,
    push_moving_button={'approach':ISSUE},push_moving_button_high_speed={'approach':ISSUE},
    put_rubbish_in_moving_bin_high_speed={'deposit':'original_expert_high_bin_current_then_closed1_open_half_forecast'})
MASKS=('valid','native_pose_valid','mode_valid','target_valid','target_twist_valid',
       'target_rotation_observable','rotation_observable','target_pose_valid_before_twist_mask','dynamics_segment_valid')


def require(value,message):
    if not value:raise ValueError('command-goal reference: '+message)


def frozen_pose(value):
    pose=Pose.from_xyzw(np.asarray(value,float))
    result=pose.xyzw().copy();result.setflags(write=False)
    return result


def fingerprint(arrays):
    result={}
    for name in MASKS:
        require(name in arrays,'original mask missing: '+name)
        value=np.ascontiguousarray(arrays[name])
        result[name]=hashlib.sha256(str((value.dtype.str,value.shape)).encode()+value.tobytes()).hexdigest()
    return result


@dataclass(frozen=True)
class TRAINCommandGoalReference:
    task_name: str
    source_id: str
    parent_source_group: str
    command_id: int
    issued_at: float
    binding: interaction.TRAINReferenceBinding
    reference_mode: str
    intent_kind: str
    intent_reference_xyzw: np.ndarray
    original_issue_reference_xyzw: np.ndarray
    original_completion_metric: dict | None
    provenance: dict

    def __post_init__(self):
        require(self.task_name in TASK_PHASES_V5 and self.parent_source_group.startswith('train:'+self.task_name+':')
            and type(self.command_id) is int and self.command_id>=0 and np.isfinite(self.issued_at)
            and self.issued_at>=0 and isinstance(self.binding,interaction.TRAINReferenceBinding)
            and self.binding.reference_kind in ('observed_se3','translation_only')
            and self.reference_mode in ('external','attached','released')
            and self.intent_kind in (ISSUE,FORECAST,BIN_FORECAST,BASKETBALL_FORECAST,HIGH_BIN_FORECAST),'invalid typed TRAIN command intent')
        for name in ('intent_reference_xyzw','original_issue_reference_xyzw'):
            object.__setattr__(self,name,frozen_pose(getattr(self,name)))
        require(self.provenance.get('schema')==SCHEMA and self.provenance.get('TRAIN_only') is True
            and self.provenance.get('original_events_masks_and_command_ids_preserved') is True
            and self.provenance.get('intent_label_is_actual_future_dynamics_observation') is False
            and self.provenance.get('runtime_native_driver_or_teacher_parameter_access_allowed') is False,
            'missing original label/provenance separation')
        if self.intent_kind==ISSUE:
            require(np.allclose(self.intent_reference_xyzw,self.original_issue_reference_xyzw,rtol=0,atol=1e-10),
                'issue intent must preserve actual original issued pose')
        else:
            require((self.provenance.get('source_declared_forecast_seconds') in (.5,1.) if self.intent_kind==HIGH_BIN_FORECAST
                    else self.provenance.get('source_declared_forecast_seconds')==.5)
                and self.provenance.get('privileged_TRAIN_native_driver_label_used') is True,
                'only the explicitly archived source0.5 or declared high-bin source forecast is supported')
        if self.intent_kind==BIN_FORECAST:
            require(self.task_name=='put_rubbish_in_moving_bin' and self.binding.reference_kind=='translation_only'
                and self.provenance.get('future_boundary_reflection_applied') is False
                and self.provenance.get('current_readonly_derivation_sources_not_historical_execution_dependencies') is True,
                'bin forecast must retain explicit new source provenance and original unreflected intention')
        if self.intent_kind==HIGH_BIN_FORECAST:
            require(self.task_name=='put_rubbish_in_moving_bin_high_speed' and self.binding.reference_kind=='translation_only'
                and self.binding.reference_id=='bin' and self.provenance.get('future_boundary_reflection_applied') is False
                and self.provenance.get('current_readonly_derivation_sources_not_historical_execution_dependencies') is True
                and self.provenance.get('source_branch_selected_from_original_prior_stage_not_outcome') is True,
                'high-bin intent requires explicit current/1/.5 source branch proof')
        if self.intent_kind==BASKETBALL_FORECAST:
            require(self.task_name in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
                and self.binding.reference_kind=='translation_only' and self.binding.reference_id=='hoop'
                and self.provenance.get('future_boundary_reflection_applied') is False
                and self.provenance.get('closed_and_open_both_use_original_forecast') is True
                and self.provenance.get('current_readonly_derivation_sources_not_historical_execution_dependencies') is True,
                'hoop forecast requires its exact closed/open source intention and explicit new source provenance')


@dataclass(frozen=True)
class TRAINCommandGoalEvidence:
    task_name: str
    source_id: str
    parent_source_group: str
    rule_sha256: str
    references: tuple[TRAINCommandGoalReference,...]
    pending_commands: tuple[dict,...]
    raw_bindings: dict
    original_masks_sha256: dict
    original_command_ledger_sha256: str
    original_completion_events_sha256: str
    derivation_proof: dict

    def __post_init__(self):
        ids=[row.command_id for row in self.references]+[row['command_id'] for row in self.pending_commands]
        require(self.parent_source_group.startswith('train:'+self.task_name+':') and len(ids)==len(set(ids))
            and all(row.task_name==self.task_name and row.source_id==self.source_id
                and row.parent_source_group==self.parent_source_group for row in self.references)
            and len(self.rule_sha256)==64 and self.derivation_proof.get('actual_raw_source_verified') is True,
            'mixed/duplicate or unverified command-goal evidence')


@dataclass(frozen=True)
class CausalRuntimeCommandGoalAnchor:
    """Only an existing candidate's causal predictor output crosses runtime."""
    pose: Pose
    predictor: str
    observed_until: float
    selected_at: float
    forecast_time: float

    def __post_init__(self):
        require(isinstance(self.pose,Pose) and self.predictor in ('cv','context')
            and np.isfinite([self.observed_until,self.selected_at,self.forecast_time]).all()
            and 0<=self.observed_until<=self.selected_at+1e-8<=self.forecast_time+1e-8,
            'runtime anchor must be an already selected causal predictor pose')


def freeze_runtime_reference(*,candidate_forecast_pose,predictor,observed_until,selected_at,forecast_time):
    require(isinstance(candidate_forecast_pose,Pose),'TRAIN intent objects cannot enter runtime')
    return CausalRuntimeCommandGoalAnchor(Pose(candidate_forecast_pose.position.copy(),candidate_forecast_pose.rotation.copy()),
        predictor,float(observed_until),float(selected_at),float(forecast_time))


def fixed_runtime_reference(anchor):
    require(isinstance(anchor,CausalRuntimeCommandGoalAnchor),'causal runtime anchor required')
    return anchor.pose,np.zeros(3),np.zeros(3)


@dataclass(frozen=True)
class VerifiedDraftRule:
    root: str
    pointer: dict
    tasks: dict
    contexts: dict
    source_evidence: dict
    verified_files_sha256: dict
    production_fitting_enabled: bool = False
    bin_permission: object | None = None
    basketball_permission: object | None = None
    auxiliary_contexts: dict | None = None
    translation_permission: object | None = None
    counted_teacher_contexts: dict | None = None

    @property
    def path(self):return self.pointer['path']

    @property
    def sha256(self):return self.pointer['sha256']


def load_rule(root,pointer,*,purpose):
    require(purpose in ('CPU_diagnostic','production_fitting'),'draft command-goal rule is not production-authorized')
    root=Path(root).resolve();files={};path=a.checked_pointer(root,pointer,files);rule=a.read(path)
    counted_version=rule.get('schema')==FIT_RULE_SCHEMA_V6
    extended=rule.get('schema') in (FIT_RULE_SCHEMA_V5,FIT_RULE_SCHEMA_V6)
    expected_fields={'schema','TRAIN_only','VAL_used','TEST_used','production_enabled','tasks',
        'helper','original_interaction_source','original_adapter_source','actual_registrations','actual700_evidence','source_evidence',
        'bin_intention_derivation_permission','basketball_intention_derivation_permission',
        'auxiliary_execution_registrations'}
    if extended:expected_fields.add('translation_intention_derivation_permission')
    if counted_version:expected_fields.add('counted_teacher_contexts')
    require(set(rule)==expected_fields,'unknown draft rule field')
    production=purpose=='production_fitting'
    tasks=TASK_PHASES_V5 if extended else TASK_PHASES
    require(rule['schema']==(FIT_RULE_SCHEMA_V6 if counted_version and production else FIT_RULE_SCHEMA_V5 if extended and production else FIT_RULE_SCHEMA if production else RULE_SCHEMA) and rule['TRAIN_only'] is True and rule['VAL_used'] is False
        and rule['TEST_used'] is False and rule['production_enabled'] is production and rule['tasks']==tasks,
        'unsupported draft scope or production enablement')
    require(a.checked_pointer(root,rule['helper'],files)==Path(__file__).resolve()
        and a.checked_pointer(root,rule['original_interaction_source'],files)==Path(interaction.__file__).resolve(),
        'source-backed helper/kernel binding changed')
    require(a.checked_pointer(root,rule['original_adapter_source'],files)==root/'our_mani/multitask/artifact_fitting.py',
        'original raw adapter binding changed')
    proof=a.read(a.checked_pointer(root,rule['actual700_evidence'],files))
    require(proof['original_teacher_cases']==700 and proof['all_original_sources_commands_events_masks_and_censoring_retained'] is True,
        'actual original700 TRAIN evidence required')
    contexts={}
    for registration in a.read(a.checked_pointer(root,rule['actual_registrations'],files)):
        current=a.load_archived_plan(root,registration,files);key=current['plan']['plan_sha256']
        require(key not in contexts,'duplicate actual acquisition plan')
        if 'predecessor_plan_sha256' in registration:
            a.verify_recovery_transition(root,contexts[registration['predecessor_plan_sha256']],current,files)
        contexts[key]=current
    source_evidence=a.read(a.checked_pointer(root,rule['source_evidence'],files))
    require(source_evidence['schema']=='r10_archived_expert_command_goal_source_evidence_v1'
        and source_evidence['tasks']==TASK_PHASES and source_evidence['forecast_seconds']==.5
        and source_evidence['source_parameter_searched'] is False,'archived explicit source intent evidence required')
    expected_names={task+'.py' for task in TASK_PHASES}|{'place_cups_on_rotating_frame.py','object.py'}
    require({Path(name).name for name in source_evidence['archived_sources']}==expected_names,
        'exact native expert/forecast helper/rotation source owners required')
    for original,item in source_evidence['archived_sources'].items():
        path=a.checked_pointer(root,item,files)
        require(any(str(Path(original).resolve()) in context['sources']
            and context['sources'][str(Path(original).resolve())]['sha256']==item['sha256']
            and Path(context['sources'][str(Path(original).resolve())]['path'])==path
            for context in contexts.values()),'native source is outside the actual archived acquisition chain')
    require(contexts and all(context['plan']['cases']==next(iter(contexts.values()))['plan']['cases'] for context in contexts.values()),
        'all declared actual source plans must preserve the original complete TRAIN roster')
    counted_teacher_contexts=_load_counted_teacher_contexts(root,rule['counted_teacher_contexts'],contexts,files) if counted_version else {}
    auxiliary_contexts={}
    for auxiliary_pointer in rule['auxiliary_execution_registrations']:
        from . import teacher_supplement
        a.checked_pointer(root,auxiliary_pointer,files)
        auxiliary=teacher_supplement.verify_registration(root,auxiliary_pointer,live=False,require_ready=True)
        proof=teacher_supplement.audit_collection(root,auxiliary_pointer,require_complete=True)
        files.update(proof['verified_files_sha256']);current=auxiliary['current'];key=current['plan']['plan_sha256']
        require(key not in contexts and key not in auxiliary_contexts and key not in counted_teacher_contexts,'duplicate auxiliary intention plan')
        auxiliary_contexts[key]=dict(pointer=auxiliary_pointer,context=auxiliary)
    from .bin_command_goal_reference import load_permission
    bin_permission=load_permission(root,rule['bin_intention_derivation_permission'],purpose=purpose)
    files.update(bin_permission.verified_files_sha256)
    from .basketball_command_goal_reference import load_permission as load_basketball
    basketball_permission=load_basketball(root,rule['basketball_intention_derivation_permission'],purpose=purpose)
    files.update(basketball_permission.verified_files_sha256)
    translation_permission=None
    if extended:
        from .translation_command_goal_reference import load_permission as load_translation
        translation_permission=load_translation(root,rule['translation_intention_derivation_permission'],purpose=purpose)
        files.update(translation_permission.verified_files_sha256)
        for task in ('push_moving_button','push_moving_button_high_speed','put_rubbish_in_moving_bin_high_speed'):
            name=task+'.py';sha=translation_permission.kernel_sources[task]['sha256']
            require(any(Path(original).name==name and item['sha256']==sha for context in contexts.values()
                for original,item in context['sources'].items()),'new translation task source is not in actual archived acquisition chain')
    return VerifiedDraftRule(str(root),pointer,tasks,contexts,source_evidence,files,production,bin_permission,
        basketball_permission,auxiliary_contexts,translation_permission,counted_teacher_contexts)



def _load_counted_teacher_contexts(root,rows,primary_contexts,files):
    """Only the exact owned v2 all100 policy; never an infra/aux alias."""
    from . import requery_acquisition as counted
    root=Path(root).resolve()
    require(isinstance(rows,list) and len(rows)==1 and isinstance(rows[0],dict)
        and set(rows[0])=={'registration','manifest','auditor'},'v6 requires exactly one explicit counted-teacher context')
    row=rows[0];registry=row['registration'];manifest=row['manifest'];auditor=row['auditor']
    require(set(registry)=={'path','sha256'} and registry['sha256']==OWNED_COUNTED_REGISTRY_SHA
        and a.local(root,registry['path'])==root/OWNED_COUNTED_REGISTRY_PATH,
        'only the explicitly approved owned high-cup v2 registration is allowed')
    require(a.checked_pointer(root,auditor,files)==Path(counted.__file__).resolve(),
        'counted teacher auditor source owner changed')
    value=a.read(a.checked_pointer(root,manifest,files))
    require(value.get('version')==counted.MANIFEST_VERSION and value.get('complete') is True
        and value.get('expected')==value.get('completed')==100 and not value.get('missing_case_ids')
        and value.get('execution_registration')==registry,'whole100 counted-teacher manifest required before production loading')
    context=counted.verify_registration(root,registry,live=False,require_ready=True)
    current=context['current'];key=current['plan']['plan_sha256'];primary=next(iter(primary_contexts.values()))
    require(key not in primary_contexts and context['registry']['execution']['plan']['sha256']==OWNED_COUNTED_PLAN_FILE_SHA
        and context['registry']['execution']['source_snapshot']['sha256']==OWNED_COUNTED_ARCHIVE_SHA,
        'counted policy must preserve its exact separate execution/source identity')
    require(context['primary']['cases']==primary['plan']['cases']
        and context['primary']['plan_sha256']==primary['plan']['plan_sha256'],
        'counted source is not derived from the actual original2800 primary roster')
    owner=str(Path(counted.__file__).resolve())
    require(current['sources'].get(owner,{}).get('sha256')==auditor['sha256'],
        'counted auditor is outside its actual frozen execution source closure')
    require(a.local(root,manifest['path'])==current['collection']/'manifest.json',
        'counted manifest must belong to its exact actual collection')
    proof=counted.audit_manifest(root,manifest['path'],require_complete=True)
    expected={c['case_id'] for c in current['plan']['cases']};parents={c['parent_source_group'] for c in current['plan']['cases']}
    require(proof.get('audit_passed') is True and proof.get('complete') is True
        and proof.get('base_plan_sha256')==primary['plan']['plan_sha256']
        and proof.get('primary_plan')==primary['registration']['plan']
        and proof.get('primary_Context_view_unmodified') is True and proof.get('primary_K1_roster_unmodified') is True,
        'counted whole100 audit did not preserve original Context/primary identities')
    projection=proof['acquisition_projection']
    require(projection==value and len(expected)==len(parents)==100
        and projection.get('expected')==projection.get('completed')==len(projection.get('entries',[]))==100
        and set(proof.get('per_case_actual_plan_sha256',{}))==expected
        and set(proof['per_case_actual_plan_sha256'].values())=={key}
        and set(proof.get('per_case_feedback_audit',{}))==expected
        and all(p.get('actual_feedback_arrays_verified') is True for p in proof['per_case_feedback_audit'].values())
        and {e['parent_source_group'] for e in projection['entries']}==parents,
        'counted whole100 actual arrays/parents/case/source coverage differs')
    require(all(c['task']=='place_cups_on_rotating_frame_high_speed' and c['replica']==c['behavior_replica']==1
        and c['teacher_used'] is True and c['context_primary_view'] is False for c in current['plan']['cases']),
        'counted policy cannot introduce another task/replica/Context view')
    for addition in (context['files'],proof['verified_files_sha256']):
        require(all(name not in files or files[name]==sha for name,sha in addition.items()),'counted source/input hash conflict')
        files.update(addition)
    return {key:dict(pointer=registry,manifest=manifest,auditor=auditor,context=context,whole100_proof=proof)}

def verify_original_source(case,source,rule):
    """Reverify raw execution without deriving or attaching any new basis."""
    require(isinstance(rule,VerifiedDraftRule),'verified explicit draft TRAIN rule required')
    require(source.split=='train' and case['entry']['teacher_used'] is True
        and source.source_id==case['entry']['source_id'] and source.grouping_source==case['entry']['parent_source_group'],
        'original TRAIN teacher source/parent identity required')
    root=Path(rule.root);record=case['record']
    require(a.read(a.local(root,case['path']))==record and case['entry']==record.get('entry'),
        'in-memory canonical record or entry differs from actual raw canonical bytes')
    key=record['plan_sha256'];auxiliary=(rule.auxiliary_contexts or {}).get(key)
    counted=(rule.counted_teacher_contexts or {}).get(key)
    require(not (auxiliary is not None and counted is not None),'counted policy cannot impersonate auxiliary execution')
    current=(counted['context']['current'] if counted is not None else
        rule.contexts.get(key) if auxiliary is None else auxiliary['context']['current'])
    require(current is not None,'unknown actual execution plan; auxiliary/late plans require a new explicit rule')
    expected=next((row for row in current['plan']['cases'] if row['case_id']==record['case_id']),None)
    require(expected is not None,'case is not in original fixed TRAIN roster')
    files={};parent=Path(a.command_argument(record['parent_process_receipt']['command'],'--output')).resolve().parent/'parent_receipt.json'
    bindings=dict(canonical_receipt=a.pointer(root,Path(case['path'])),cache=a.pointer(root,a.local(root,case['entry']['cache'])),
        commands=a.pointer(root,a.local(root,case['entry']['teacher_command_receipt'])))
    if counted is not None:
        from . import requery_acquisition
        a.verify_completed(root,dict(canonical_receipt=bindings['canonical_receipt'],parent_receipt=a.pointer(root,parent)),current,expected,files)
        requery_acquisition.validate_result(root,Path(case['path']),counted['context'],expected,files=files)
    elif auxiliary is not None:
        from . import teacher_supplement
        teacher_supplement.validate_result(root,Path(case['path']),auxiliary['context'],expected,files=files)
    else:
        completion=a.verify_completed(root,dict(canonical_receipt=bindings['canonical_receipt'],parent_receipt=a.pointer(root,parent)),
            current,expected,files)
    if counted is None and auxiliary is None and 'recovery_input' in current['plan']['protocol']:
        previous=rule.contexts[current['registration']['predecessor_plan_sha256']]
        payload=a.read(a.checked_pointer(root,current['plan']['protocol']['recovery_input'],files))
        preceding=[dict(actual_plan_sha256=previous['plan']['plan_sha256'],parent_receipt=p)
            for p in payload['failed_parent_receipts'] if a.read(a.local(root,p['path']))['case_id']==record['case_id']]
        a.verify_recovery_execution(root,completion,current,previous,preceding,files)
    arrays=case['arrays'];validate_arrays(arrays,case['entry'],record)
    with np.load(a.local(root,bindings['cache']['path']),allow_pickle=False) as actual:
        require(set(arrays)==set(actual.files) and all(np.array_equal(arrays[k],actual[k]) for k in arrays),
            'in-memory raw arrays differ from actual acquisition bytes')
    raw=a.read(a.checked_pointer(root,bindings['commands'],files))
    require(case['command_receipt']==raw and list(source.command_ledger)==raw['commands'],
        'original command ledger changed before intent derivation')
    from .artifact_fitting import interaction_source
    actual_source,_=interaction_source(case)
    require(source.stages==actual_source.stages and source.references==actual_source.references
        and len(source.completion_events)==len(actual_source.completion_events),
        'original source phases/references/events were replaced')
    for field in ('times','expert_tcp_xyzw','expert_opening','measured_tcp_xyzw','curve_tcp_xyzw','command_ids'):
        require(np.array_equal(getattr(source,field),getattr(actual_source,field)),
            'source differs from actual raw adapter: '+field)
    require(set(source.body_tracks)==set(actual_source.body_tracks),'original physical body track inventory changed')
    for body,track in source.body_tracks.items():
        actual_track=actual_source.body_tracks[body]
        require(track.motion_modes==actual_track.motion_modes
            and track.maximum_gap_seconds==actual_track.maximum_gap_seconds
            and all(np.array_equal(getattr(track,name),getattr(actual_track,name))
                for name in ('times','poses_xyzw','pose_valid','orientation_observable')),
            'original physical body labels/masks changed')
    for event,actual_event in zip(source.completion_events,actual_source.completion_events):
        require(event.origin_command_id==actual_event.origin_command_id
            and event.binding==actual_event.binding and event.reference_mode==actual_event.reference_mode
            and event.attempt_start_time==actual_event.attempt_start_time and event.attempt_end_time==actual_event.attempt_end_time
            and np.array_equal(event.terminal_reference_xyzw,actual_event.terminal_reference_xyzw)
            and np.array_equal(event.terminal_tcp_xyzw,actual_event.terminal_tcp_xyzw),
            'actual original completion metric changed')
    derivation=dict(schema=SCHEMA,actual_raw_source_verified=True,
        actual_source_snapshot=current['registration']['source_snapshot'],verified_files_sha256=files,
        original_source_object_replaced=False,production_fitting_enabled=rule.production_fitting_enabled)
    if counted is not None:
        derivation['counted_teacher_acquisition']=dict(registration=counted['pointer'],manifest=counted['manifest'],auditor=counted['auditor'],
            actual_plan_sha256=key,whole100_complete=True,primary_K1_roster_unmodified=True,primary_Context_view_unmodified=True,
            is_infrastructure_recovery=False,is_basketball_teacher_supplement=False)
    return TRAINCommandGoalEvidence(source.task_name,source.source_id,source.grouping_source,rule.pointer['sha256'],
        (),(),bindings,fingerprint(arrays),a.canonical_sha(raw['commands']),
        a.canonical_sha(raw['phase_completion_events']),derivation)


def derive_source(case,source,rule):
    """Original metric verification precedes every explicit TRAIN intention."""
    original=verify_original_source(case,source,rule)
    arrays=case['arrays'];raw=case['command_receipt'];record=case['record']
    masks=original.original_masks_sha256;bindings=original.raw_bindings
    files=dict(original.derivation_proof['verified_files_sha256']);phases=rule.tasks.get(source.task_name,{})
    if phases and source.task_name in ('push_moving_button','push_moving_button_high_speed','put_rubbish_in_moving_bin_high_speed'):
        from . import translation_command_goal_reference as translation
        require(rule.translation_permission is not None,'extended translation scope needs explicit v5 permission')
        return translation.derive_source(case,source,rule,rule.translation_permission,original_verification=original)
    if source.task_name in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed'):
        from . import basketball_command_goal_reference as basketball
        require(rule.basketball_permission is not None,'hoop intention requires its explicit complete400 permission')
        return basketball.derive_source(case,source,rule,rule.basketball_permission,original_verification=original)
    if source.task_name=='put_rubbish_in_moving_bin':
        from . import bin_command_goal_reference as bin_goal
        require(rule.bin_permission is not None,'bin intention requires its explicit new source permission')
        evidence=bin_goal.derive_source(case,source,rule,rule.bin_permission,original_verification=original)
        references=tuple(TRAINCommandGoalReference(r.task_name,r.source_id,r.parent_source_group,r.command_id,r.issued_at,
            r.binding,r.reference_mode,r.intent_kind,r.intent_reference_xyzw,r.original_issue_reference_xyzw,
            r.original_completion_metric,dict(r.provenance,schema=SCHEMA,bin_derivation_schema=bin_goal.SCHEMA,
                rule=rule.pointer,actual_plan_sha256=record['plan_sha256'])) for r in evidence.references)
        return TRAINCommandGoalEvidence(source.task_name,source.source_id,source.grouping_source,rule.sha256,
            references,evidence.pending_commands,bindings,masks,original.original_command_ledger_sha256,
            original.original_completion_events_sha256,dict(original.derivation_proof,
                verified_files_sha256=evidence.derivation_proof['verified_files_sha256'],
                bin_source_intention=bin_goal.evidence_proof(evidence)))
    event_map={row['origin_command_id']:row for row in raw['phase_completion_events']}
    references=[];pending=[];ids=tuple(map(str,arrays['body_ids']))
    for command in source.command_ledger:
        phase=command['raw_phase']
        if phase not in phases:continue
        cid=command['command_id'];indices=np.flatnonzero(np.isclose(source.times,command['generated_at'],atol=1e-8,rtol=0))
        require(len(indices)==1,'original issued command needs unique actual frame')
        i=int(indices[0]);body=command['reference_body'];j=ids.index(body)
        binding=interaction.TRAINReferenceBinding(command['reference_role'],body,command['reference_kind'],
            command['anchor_id'],command['manipulated_role'])
        require(binding.reference_kind in ('observed_se3','translation_only'),'explicit physical intent reference required')
        valid=bool(arrays['native_pose_valid'][i,j] and arrays['mode_valid'][i,j])
        if binding.reference_kind=='observed_se3':valid &= bool(arrays['target_rotation_observable'][i,j])
        if not valid:
            pending.append(dict(command_id=cid,reason='original issue pose/mode/orientation mask invalid',raw_source_preserved=True));continue
        issue=Pose.from_xyzw(command['issued_reference_native_pose_xyzw'])
        actual=Pose(arrays['target_positions'][i,j],arrays['target_rotations'][i,j])
        require(np.linalg.norm(issue.position-actual.position)<1e-8
            and Rotation.from_matrix(issue.rotation@actual.rotation.T).magnitude()<1e-7,'issued pose differs from actual raw frame')
        if binding.reference_kind=='translation_only':issue=Pose(issue.position,np.eye(3))
        intent=issue;kind=phases[phase];proof=dict(schema=SCHEMA,TRAIN_only=True,
            actual_plan_sha256=record['plan_sha256'],rule=rule.pointer,raw_bindings=bindings,
            original_events_masks_and_command_ids_preserved=True,
            intent_label_is_actual_future_dynamics_observation=False,
            runtime_native_driver_or_teacher_parameter_access_allowed=False,
            source_declared_forecast_seconds=0.,privileged_TRAIN_native_driver_label_used=False)
        if kind==FORECAST:
            frame=ids.index('frame');require(arrays['native_pose_valid'][i,frame],'original frame axis pose is invalid')
            schedule=record['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule']
            yaw=float(schedule['yaw_speed']);require(np.isfinite(yaw),'source native TRAIN yaw label missing')
            rotation=Rotation.from_rotvec(arrays['target_rotations'][i,frame,:,2]*np.deg2rad(yaw)*.5).as_matrix()
            center=arrays['target_positions'][i,frame]
            intent=Pose(center+rotation@(issue.position-center),rotation@issue.rotation)
            proof.update(source_declared_forecast_seconds=.5,privileged_TRAIN_native_driver_label_used=True,
                original_metric_only_yaw_speed_deg_s=yaw,original_frame_axis_xyzw=Pose(center,arrays['target_rotations'][i,frame]).xyzw().tolist(),
                archived_source_evidence=rule.source_evidence['archived_sources'])
        references.append(TRAINCommandGoalReference(source.task_name,source.source_id,source.grouping_source,cid,
            float(command['generated_at']),binding,command['issued_reference_motion_mode'],kind,intent.xyzw(),issue.xyzw(),
            event_map.get(cid),proof))
    require(masks==fingerprint(arrays),'raw masks changed during derivation')
    require({row.command_id for row in references}|{row['command_id'] for row in pending}==
        {row['command_id'] for row in source.command_ledger if row['raw_phase'] in phases},
        'every configured original command must remain derived or explicitly pending')
    return TRAINCommandGoalEvidence(source.task_name,source.source_id,source.grouping_source,rule.pointer['sha256'],
        tuple(references),tuple(pending),bindings,masks,a.canonical_sha(raw['commands']),
        a.canonical_sha(raw['phase_completion_events']),dict(original.derivation_proof,schema=SCHEMA,actual_raw_source_verified=True,
            actual_source_snapshot=original.derivation_proof['actual_source_snapshot'],verified_files_sha256=files,
            original_source_object_replaced=False,production_fitting_enabled=rule.production_fitting_enabled))


def evidence_proof(evidence):
    """Deterministic producer/consumer proof; raw metric is a separate field."""
    require(isinstance(evidence,TRAINCommandGoalEvidence),'typed command-goal evidence required')
    references=[dict(command_id=row.command_id,issued_at=row.issued_at,intent_kind=row.intent_kind,
        binding=interaction._json(row.binding.__dict__),reference_mode=row.reference_mode,
        intent_reference_xyzw=row.intent_reference_xyzw.tolist(),original_issue_reference_xyzw=row.original_issue_reference_xyzw.tolist(),
        original_completion_metric=row.original_completion_metric,provenance=row.provenance) for row in evidence.references]
    value=dict(schema=SCHEMA,rule_sha256=evidence.rule_sha256,raw_bindings=evidence.raw_bindings,
        original_masks_sha256=evidence.original_masks_sha256,
        original_command_ledger_sha256=evidence.original_command_ledger_sha256,
        original_completion_events_sha256=evidence.original_completion_events_sha256,
        references=references,pending_commands=list(evidence.pending_commands),original_actual_metric_preserved=True)
    counted_proof=evidence.derivation_proof.get('counted_teacher_acquisition')
    if counted_proof is not None:value['counted_teacher_acquisition']=counted_proof
    translation_proof=evidence.derivation_proof.get('translation_source_intention')
    if translation_proof is not None:value['translation_source_intention']=translation_proof
    bin_proof=evidence.derivation_proof.get('bin_source_intention')
    if bin_proof is not None:value['bin_source_intention']=bin_proof
    basketball_proof=evidence.derivation_proof.get('basketball_source_intention')
    if basketball_proof is not None:value['basketball_source_intention']=basketball_proof
    return dict(value,derivation_sha256=a.canonical_sha(value))


def fit_diagnostic(sources,evidence,config):
    """Actual full measured curves + exact parent CV; no production artifact.

    The unchanged production curve/rotation solver is reused. A declared fixed
    command-goal anchor has lag0 only. Other partitions retain all original lag
    candidates, validity checks, sample coverage and grouped CV selection.
    """
    require(len(sources)==len(evidence)==100 and len({s.grouping_source for s in sources})==100,
        'all100 original task teacher parents, including censored cases, required')
    require(config.minimum_completed_parent_sources==8 and config.maximum_position_tolerance_m==.03
        and config.maximum_orientation_tolerance_rad==.15 and config.development_only is False,'unchanged original8/.03/.15 gates required')
    groups={};durations={};events_by_attempt={};curation=[];invalid=[]
    for source,proof in zip(sources,evidence):
        require(proof.source_id==source.source_id and proof.parent_source_group==source.grouping_source,'evidence/source identity changed')
        goals={row.command_id:row for row in proof.references};pending={row['command_id'] for row in proof.pending_commands}
        events={row.origin_command_id:row for row in source.completion_events}
        stages={stage:phase.name for phase in interaction.get_task_spec(source.task_name).phase_skills for stage in phase.TRAIN_expert_stages}
        for command in source.command_ledger:
            cid=command['command_id'];event=events.get(cid)
            if event is None:
                excluded=next((row for row in source.excluded_completion_events if row['command_id']==cid),None)
                begin=float(command['generated_at'])
                end=float(command.get('cutoff_at',command.get('last_executed_at',source.times[-1])))
                curation.append(dict(source_id=source.source_id,command_id=cid,phase=command['raw_phase'],included=False,
                    reason='original_no_usable_completion_event',original_raw_command_preserved=True,
                    original_completion_label_exclusion=excluded,
                    duration_is_right_censored_lower_bound=excluded is None,
                    observed_duration_lower_bound_seconds=max(0.,end-begin)));continue
            if cid in pending:
                invalid.append(dict(source_id=source.source_id,command_id=cid,reason='completed command has invalid goal intent mask'));continue
            binding=event.binding;goal=goals.get(cid);basis=INTENT_BASIS if goal else binding.reference_time_basis
            indices=np.flatnonzero(source.command_ids==cid)
            require(not any(stages.get(source.stages[i])!=event.phase or source.references[i]!=binding for i in indices
                if source.references[i] is not None),'original completion phase/reference differs')
            indices=[i for i in indices if source.references[i]==binding
                and event.attempt_start_time-1e-8<=source.times[i]<event.attempt_end_time-1e-8]
            key=(event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,binding.manipulated_role,event.reference_mode,basis)
            attempt=source.source_id+':command'+str(cid);duration=event.attempt_end_time-event.attempt_start_time
            rows_by_lag={}
            for lag in config.lag_grid_seconds:
                if goal and lag!=0.:continue
                try:terminal_reference=Pose.from_xyzw(goal.intent_reference_xyzw) if goal else interaction._completion_reference(source,event,float(lag))
                except (interaction.InteractionArtifactError,KeyError):continue
                rows=[]
                for i in indices:
                    try:
                        if binding.reference_kind=='world':reference=Pose(np.zeros(3))
                        elif binding.reference_kind=='phase_origin':reference=Pose.from_xyzw(event.origin_measured_tcp_xyzw)
                        else:
                            reference=source.body_tracks[binding.reference_id].at(source.times[i]+lag,
                                orientation_required=binding.reference_kind=='observed_se3',same_mode_as=event.reference_mode,not_before=source.times[i])
                            if binding.reference_kind=='translation_only':reference=Pose(reference.position,np.eye(3))
                            if goal:reference=terminal_reference
                        curve=Pose.from_xyzw(source.curve_tcp_xyzw[i])
                    except (interaction.InteractionArtifactError,KeyError):continue
                    rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt,
                        progress=float((source.times[i]-event.attempt_start_time)/duration),
                        position=reference.rotation.T@(curve.position-reference.position),rotation=reference.rotation.T@curve.rotation,
                        opening=source.expert_opening[i],terminal_event=False))
                if not rows:continue
                terminal=Pose.from_xyzw(event.terminal_tcp_xyzw)
                rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt,progress=1.,
                    position=terminal_reference.rotation.T@(terminal.position-terminal_reference.position),
                    rotation=terminal_reference.rotation.T@terminal.rotation,opening=source.expert_opening[indices[-1]],terminal_event=True))
                rows_by_lag[float(lag)]=rows
            curation.append(dict(source_id=source.source_id,command_id=cid,phase=event.phase,included=0. in rows_by_lag,
                reason='explicit_original_completion',raw_completion_metric_preserved=True,reference_basis=basis))
            if 0. not in rows_by_lag:continue
            durations.setdefault(key,[]).append((source.grouping_source,float(duration)));events_by_attempt[attempt]=(source,event)
            for lag,rows in rows_by_lag.items():groups.setdefault((key,lag),[]).extend(rows)
    report=dict(schema='r10_actual_full_curve_command_goal_CPU_fit_draft_v1',TRAIN_only=True,
        all100_original_sources_retained=True,raw_events_and_censoring_preserved=True,curation=curation,
        invalid_evidence=invalid,phase_fits={},production_InteractionLibrary_exported=False,
        real_full_curves_and_parent_group_CV=True,unchanged_production_curve_solver=True)
    grid=np.linspace(0,1,config.progress_knots)
    for key,complete in durations.items():
        base=groups[(key,0.)];parents=sorted({row['source'] for row in base});label='|'.join(key)
        diagnostic=dict(completed_parent_count=len(parents),completed_attempt_count=len({row['attempt_id'] for row in base}),
            completed_parent_sources=parents,reference_basis=key[-1],curve_rows_including_metric_terminal=len(base))
        report['phase_fits'][label]=diagnostic
        if len(parents)<max(config.minimum_sources,config.minimum_completed_parent_sources):
            diagnostic.update(passed=False,reason='insufficient completed TRAIN parent support');continue
        folds={name:i%2 for i,name in enumerate(parents)};candidates=[]
        for lag in config.lag_grid_seconds:
            rows=groups.get((key,float(lag)),[])
            if ({row['attempt_id'] for row in rows}!={row['attempt_id'] for row in base}
                or len(rows)<config.minimum_lag_sample_fraction*len(base)):
                candidates.append(dict(lag=float(lag),valid=False,reason='original attempt/sample coverage or fixed intent lag differs'));continue
            errors=[];terminal_errors=[]
            for fold in (0,1):
                fit=[row for row in rows if folds[row['source']]!=fold];held=[row for row in rows if folds[row['source']]==fold]
                p,r,o=interaction._fit_completed_curve(fit,grid)
                for sample in held:
                    position=np.asarray([np.interp(sample['progress'],grid,p[:,axis]) for axis in range(3)])
                    rotation=Slerp(grid,Rotation.from_matrix(r))([sample['progress']]).as_matrix()[0]
                    pe=float(np.linalg.norm(position-sample['position']));re=float(Rotation.from_matrix(rotation@sample['rotation'].T).magnitude())
                    errors.append((sample['source'],pe**2+re**2))
                    if sample['terminal_event']:terminal_errors.append(dict(source=sample['source'],attempt_id=sample['attempt_id'],position=pe,orientation=re))
            candidates.append(dict(lag=float(lag),valid=True,samples=len(rows),
                grouped_TRAIN_CV_error=float(np.mean([np.mean([value for parent,value in errors if parent==p]) for p in parents])),
                terminal_crossfit_errors=terminal_errors))
        diagnostic.update(source_to_fold=folds,candidates=candidates)
        valid=[row for row in candidates if row['valid']]
        if not valid:diagnostic.update(passed=False,reason='no valid full-curve lag');continue
        chosen=min(valid,key=lambda row:(row['grouped_TRAIN_CV_error'],row['lag']));terminal=chosen['terminal_crossfit_errors']
        residual={parent:{field:float(np.median([row[field] for row in terminal if row['source']==parent]))
            for field in ('position','orientation')} for parent in parents}
        pt=max(config.minimum_position_tolerance_m,float(np.quantile([row['position'] for row in residual.values()],config.tolerance_quantile)))
        rt=max(config.minimum_orientation_tolerance_rad,float(np.quantile([row['orientation'] for row in residual.values()],config.tolerance_quantile)))
        passed=pt<=config.maximum_position_tolerance_m and rt<=config.maximum_orientation_tolerance_rad
        values=[float(np.median([duration for group,duration in complete if group==parent])) for parent in parents]
        p,r,o=interaction._fit_completed_curve(groups[(key,chosen['lag'])],grid)
        diagnostic.update(passed=passed,reason=None if passed else 'terminal residual exceeds unchanged TRAIN cap; no clipping',
            selected_lag=chosen['lag'],grouped_TRAIN_CV_error=chosen['grouped_TRAIN_CV_error'],
            terminal_parent_residuals=residual,position_tolerance_m=pt,orientation_tolerance_rad=rt,
            duration_seconds=float(np.median(values)),duration_budget_seconds=float(np.quantile(values,config.duration_quantile)),
            fitted_curve=dict(progress=grid.tolist(),positions=p.tolist(),rotation_xyzw=Rotation.from_matrix(r).as_quat().tolist(),opening=o.tolist()))
    spec=interaction.get_task_spec(sources[0].task_name);available={key.split('|')[0] for key,row in report['phase_fits'].items() if row.get('passed')}
    report['required_phase_failures']=[phase.name for phase in spec.phase_skills if not phase.optional_after_native_completion and phase.name not in available]
    report['failed_fit_keys']=[key for key,row in report['phase_fits'].items() if not row.get('passed')]
    report['all_required_phases_and_observed_partitions_passed']=not invalid and not report['failed_fit_keys'] and not report['required_phase_failures']
    return report
