"""Explicit TRAIN transport intent for old r0/r1 and periodic r2 acquisitions.

The v6 rule knows only r0/r1. This separate rule audits each actual execution
with its own typed auditor and preserves all raw completion metrics. Only the
already source-declared transport issue-pose convention is carried forward.
"""
from dataclasses import dataclass,fields,is_dataclass,replace
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from . import acquisition_lineage as a,acquisition_manifest,periodic_acquisition
from . import artifact_fitting as adapter,interaction,command_goal_reference as goal
from .input_stat_cache_v79 import verify_inputs

TASK='place_cups_on_rotating_frame_high_speed'
SCHEMA='r10_highcup_all300_explicit_transport_issue_rule_v122'


def require(value,message):
    if not value:raise ValueError('highcup issue: '+message)


@dataclass(frozen=True)
class VerifiedRule:
    root: Path
    pointer: dict
    cases: dict
    files: dict
    source_evidence: dict
    actual_policy_kinds: dict


def load_rule(root,pointer):
    root=Path(root).resolve();files={}
    value=a.read(a.checked_pointer(root,pointer,files))
    require(set(value)=={'schema','TRAIN_only','VAL_used','TEST_used','tasks','helper',
        'original_intent_rule','combined_source_manifest','original_fit_gates'},'explicit rule fields differ')
    require(value['schema']==SCHEMA and value['TRAIN_only'] is True
        and value['VAL_used'] is False and value['TEST_used'] is False
        and value['tasks']=={TASK:{'transport':goal.ISSUE}}
        and value['original_fit_gates']==dict(minimum_completed_parent_sources=8,
            maximum_position_tolerance_m=.03,maximum_orientation_tolerance_rad=.15),
        'only original highcup transport convention and original8/.03/.15 scope allowed')
    require(a.checked_pointer(root,value['helper'],files)==Path(__file__).resolve(),'helper source differs')
    old_rule=a.read(a.checked_pointer(root,value['original_intent_rule'],files))
    require(old_rule['schema']==goal.FIT_RULE_SCHEMA_V6
        and old_rule['tasks'][TASK]=={'transport':goal.ISSUE},'original v6 transport convention differs')
    require(a.checked_pointer(root,old_rule['helper'],files)==Path(goal.__file__).resolve(),
        'original typed intent helper differs')
    evidence=a.read(a.checked_pointer(root,old_rule['source_evidence'],files))
    require(evidence['tasks'][TASK]=={'transport':goal.ISSUE}
        and evidence['source_parameter_searched'] is False,'archived native source convention differs')
    source_rows={n:p for n,p in evidence['archived_sources'].items() if Path(n).name==TASK+'.py'}
    require(len(source_rows)==1,'exact highcup archived expert required')
    for p in source_rows.values():a.checked_pointer(root,p,files)
    manifest=a.read(a.checked_pointer(root,value['combined_source_manifest'],files))
    require(manifest['original_real_parents']==100 and manifest['teacher_acquisitions']==300
        and manifest['all_original_failed_censored_sources_retained'] is True
        and manifest['periodic_policy_not_exact_author_reproduction'] is True
        and manifest['context_primary_view_modified'] is False
        and manifest['VAL_used'] is False and manifest['TEST_used'] is False,'combined scope differs')
    verify_inputs(root,manifest['verified_files_sha256']);files.update(manifest['verified_files_sha256'])
    old_path=a.checked_pointer(root,manifest['typed_original_bundle'],files)
    old,old_proof=acquisition_manifest.load_acquisition_manifest(root,old_path)
    require(old_proof is not None,'actual typed old acquisition proof required')
    verify_inputs(root,old_proof['verified_files_sha256']);files.update(old_proof['verified_files_sha256'])
    periodic_path=a.checked_pointer(root,manifest['periodic100_manifest'],files)
    periodic=a.read(periodic_path)
    new_proof=periodic_acquisition.audit_manifest(root,periodic_path,require_complete=True)
    stored=a.read(a.checked_pointer(root,manifest['strict_periodic100_proof'],files))
    require(a.canonical_sha(new_proof)==a.canonical_sha(stored),
        'current strict periodic proof differs from accepted actual raw100 proof')
    verify_inputs(root,new_proof['verified_files_sha256']);files.update(new_proof['verified_files_sha256'])
    selected={};kinds={};entries=[]
    for acquisition,kind in ((old,'original_typed_r0_r1'),(periodic,'periodic10_r2')):
        for row in acquisition['case_receipts']:
            path=a.checked_pointer(root,row,files) if set(row)=={'path','sha256'} else a.checked_pointer(root,
                dict(path=row['path'],sha256=row['sha256']),files)
            record=a.read(path);entry=record['entry']
            if entry['task']!=TASK or not entry.get('interaction_eligible',True):continue
            require(str(path) not in selected,'duplicate original acquisition')
            require(record['split']=='train' and entry['teacher_used'] is True
                and entry['context_primary_view'] is False,'teacher-only TRAIN operation scope differs')
            if kind=='periodic10_r2':
                require(record['plan_sha256']==periodic['plan_sha256'] and entry['behavior_replica']==2
                    and record['periodic_replanning_execution']['execution_registration']==periodic['execution_registration'],
                    'periodic execution cannot impersonate old r0/r1')
            else:require(record['plan_sha256']==acquisition_manifest.actual_case_plan_sha(old,old_proof,record['case_id']),
                'old acquisition actual plan differs')
            selected[str(path)]=dict(path=row['path'],sha256=row['sha256'],entry=entry)
            kinds[str(path)]=kind;entries.append(entry)
    require(len(selected)==300 and Counter(e['behavior_replica'] for e in entries)=={0:100,1:100,2:100},
        'all300 original acquisitions required')
    require({(e['source_id'],e['cache_sha256']) for e in entries}==
        {(e['source_id'],e['cache_sha256']) for e in manifest['entries']},'combined source inventory differs')
    roster_path=root/'artifacts/framework/protocol/train_manifest.json'
    files[str(roster_path)]=a.file_sha(roster_path)
    expected={'train:'+TASK+':'+r['raw_episode_id'] for r in a.read(roster_path)['episodes'] if r['task']==TASK}
    require(len(expected)==100 and {e['parent_source_group'] for e in entries}==expected,
        'exact original100 TRAIN parents required')
    return VerifiedRule(root,pointer,selected,files,source_rows,kinds)


def equal_raw(left,right):
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right,equal_nan=left.dtype.kind in 'fc')
    if is_dataclass(left):
        return type(left) is type(right) and all(equal_raw(getattr(left,f.name),getattr(right,f.name)) for f in fields(left))
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(equal_raw(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):
        return type(left) is type(right) and len(left)==len(right) and all(equal_raw(x,y) for x,y in zip(left,right))
    return left==right


def derive_source(case,source,rule):
    require(isinstance(rule,VerifiedRule),'actual typed all300 rule required')
    root=rule.root;path=Path(case['path']).resolve();row=rule.cases.get(str(path))
    require(row is not None,'unknown actual acquisition; no aliasing to original r1')
    files={};actual_record=a.read(a.checked_pointer(root,dict(path=row['path'],sha256=row['sha256']),files))
    require(case['record']==actual_record and case['entry']==row['entry']==actual_record['entry'],
        'canonical raw record or entry changed')
    entry=case['entry'];require(source.task_name==TASK and source.split=='train'
        and source.source_id==entry['source_id'] and source.grouping_source==entry['parent_source_group'],
        'source task/parent identity differs')
    bindings=dict(canonical_receipt=dict(path=row['path'],sha256=row['sha256']),
        cache=dict(path=entry['cache'],sha256=entry['cache_sha256']),commands=dict(path=entry['teacher_command_receipt'],
            sha256=actual_record['files_sha256'][entry['teacher_command_receipt']]))
    cache=a.checked_pointer(root,bindings['cache'],files)
    with np.load(cache,allow_pickle=False) as actual:
        require(set(case['arrays'])==set(actual.files) and all(equal_raw(case['arrays'][k],actual[k]) for k in actual.files),
            'raw arrays or original masks changed')
    raw=a.read(a.checked_pointer(root,bindings['commands'],files))
    require(case['command_receipt']==raw,'original commands/events changed')
    actual_source,_=adapter.interaction_source(case)
    require(equal_raw(source,actual_source),'original adapter source/curation/completion metrics changed')
    masks=goal.fingerprint(case['arrays']);refs=[];pending=[];ids=tuple(map(str,case['arrays']['body_ids']))
    events={e['origin_command_id']:e for e in raw['phase_completion_events']}
    for command in source.command_ledger:
        if command['raw_phase']!='transport':continue
        indices=np.flatnonzero(np.isclose(source.times,command['generated_at'],atol=1e-8,rtol=0))
        require(len(indices)==1,'issued command needs unique actual frame')
        i=int(indices[0]);j=ids.index(command['reference_body']);arrays=case['arrays']
        valid=bool(arrays['native_pose_valid'][i,j] and arrays['mode_valid'][i,j]
            and arrays['target_rotation_observable'][i,j])
        if not valid:
            pending.append(dict(command_id=command['command_id'],reason='original issue pose/mode/orientation mask invalid',raw_source_preserved=True));continue
        issue=Pose.from_xyzw(command['issued_reference_native_pose_xyzw'])
        actual=Pose(arrays['target_positions'][i,j],arrays['target_rotations'][i,j])
        require(np.linalg.norm(issue.position-actual.position)<1e-8
            and Rotation.from_matrix(issue.rotation@actual.rotation.T).magnitude()<1e-7,'issued pose differs from raw native frame')
        binding=interaction.TRAINReferenceBinding(command['reference_role'],command['reference_body'],
            command['reference_kind'],command['anchor_id'],command['manipulated_role'])
        require(binding.reference_kind=='observed_se3','highcup transport needs full original physical reference')
        provenance=dict(schema=goal.SCHEMA,TRAIN_only=True,actual_plan_sha256=actual_record['plan_sha256'],rule=rule.pointer,
            raw_bindings=bindings,original_events_masks_and_command_ids_preserved=True,
            intent_label_is_actual_future_dynamics_observation=False,runtime_native_driver_or_teacher_parameter_access_allowed=False,
            source_declared_forecast_seconds=0.,privileged_TRAIN_native_driver_label_used=False,
            explicit_all300_rule_schema=SCHEMA,actual_teacher_policy_kind=rule.actual_policy_kinds[str(path)],
            archived_source_evidence=rule.source_evidence)
        refs.append(goal.TRAINCommandGoalReference(TASK,source.source_id,source.grouping_source,command['command_id'],
            float(command['generated_at']),binding,command['issued_reference_motion_mode'],goal.ISSUE,
            issue.xyzw(),issue.xyzw(),events.get(command['command_id']),provenance))
    require({r.command_id for r in refs}|{r['command_id'] for r in pending}==
        {c['command_id'] for c in source.command_ledger if c['raw_phase']=='transport'},'original command coverage differs')
    proof=goal.TRAINCommandGoalEvidence(TASK,source.source_id,source.grouping_source,rule.pointer['sha256'],tuple(refs),
        tuple(pending),bindings,masks,a.canonical_sha(raw['commands']),a.canonical_sha(raw['phase_completion_events']),
        dict(schema=SCHEMA,actual_raw_source_verified=True,verified_files_sha256=files,
            actual_teacher_policy_kind=rule.actual_policy_kinds[str(path)],production_fitting_enabled=True,
            original_source_object_replaced=False,old_v6_rule_not_extended_or_impersonated=True))
    return replace(source,command_goal_reference_evidence=proof)
