"""Explicit initial-sequence cup policy producer with original numeric kernels.

All commands, completion events, failed partitions and source parents remain
in the report. The sole admission change is an unconditional exclusion of
released-ring recovery from this new policy domain. The distinct schema cannot
be consumed by the old full-scope library loader without an explicit adapter.
"""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from ..contracts import Pose
from . import interaction as original_interaction
from .interaction import (InteractionArtifactError,InteractionFittingPending,PhaseTemplate,
    _completion_reference,_fit_completed_curve as _original_fit_completed_curve,_semantic_acquire_curve_rows,
    TRAINSemanticAcquireSegment,COMMAND_ISSUE,COMMAND_GOAL,REFERENCE_TIME_SCHEMA,_json)
from .specs import get_task_spec

from . import minimax_terminal_v220 as minimax
MINIMAX_AUDIT=[]


def _fit_completed_curve(samples,grid,*,minimize_terminal_radius=False):
    p,r,o=_original_fit_completed_curve(samples,grid)
    if minimize_terminal_radius:
        p[-1],audit=minimax.fit_terminal(samples)
        MINIMAX_AUDIT.append(audit)
    return p,r,o


TASK='insert_onto_rotating_peg_high_speed'
LIBRARY_SCHEMA='r10_TRAIN_highpeg_primary_minimax_insert_phase_curves_v221'
SCHEMA='r10_highpeg_explicit_primary_minimax_insert_admission_v221'


def declared_primary_scope(report,templates):
    admitted=[];outside=[]
    for key,row in report['phase_fits'].items():
        phase,anchor,role,kind,manipulated,mode,basis=key.split('|')
        if not anchor.startswith(get_task_spec(TASK).family+':'+phase+':'):
            raise InteractionArtifactError('wrong task in explicit initial-sequence policy')
        if mode=='released':
            if phase not in ('approach','acquire') or role!='ring':
                raise InteractionArtifactError('undeclared excluded recovery partition')
            outside.append(dict(key=key,original_completed_parent_count=row['completed_parent_count'],
                original_fit_passed=bool(row['passed']),reason='released-ring recovery outside declared policy domain'))
        elif mode in ('external','not_applicable'):
            admitted.append(key)
        else:raise InteractionArtifactError('undeclared initial-sequence reference mode')
    admitted.sort();outside.sort(key=lambda row:row['key'])
    return dict(schema=SCHEMA,task=TASK,admitted_partition_keys=admitted,unadmitted_partitions=outside,
        policy_domain='initial_sequence_no_released_ring_recovery',
        original_all_observed_partitions_gate_passed=not report['failed_fit_keys'],
        original_required_phase_failures=report['required_phase_failures'],
        original_raw_commands_events_masks_and_failed_partitions_preserved=True,
        original_source_parents_not_filtered=True,old_full_scope_library_release_claimed=False,
        admitted_partition_support_and_error_gates_passed=all(report['phase_fits'][k]['passed'] for k in admitted),
        native_or_formal_release_granted=False)


class PrimarySequenceLibrary(original_interaction.InteractionLibrary):
    def __init__(self,payload,*,allow_synthetic=False):
        self.metadata=payload['metadata'];m=self.metadata;scope=m.get('initial_sequence_policy_scope',{})
        if (m.get('schema')!=LIBRARY_SCHEMA or m.get('training_complete') is not True
                or m.get('task_name')!=TASK or m.get('fit_split')!='train'
                or m.get('selection_split')!='train' or m.get('TEST_used_for_fitting') is not False
                or m.get('synthetic') is not False or allow_synthetic
                or m.get('fitting_module_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or m.get('numeric_helpers_sha256')!=hashlib.sha256(Path(original_interaction.__file__).read_bytes()).hexdigest()
                or m.get('minimax_helper_sha256')!=hashlib.sha256(Path(minimax.__file__).read_bytes()).hexdigest()
                or m.get('task_spec_module_sha256')!=hashlib.sha256((Path(__file__).parent/'specs.py').read_bytes()).hexdigest()
                or scope.get('schema')!=SCHEMA or scope.get('old_full_scope_library_release_claimed') is not False):
            raise InteractionArtifactError('explicit current initial-sequence producer and domain proof required')
        fit=m['fit_configuration']
        if (fit.get('minimum_completed_parent_sources')!=8 or fit.get('maximum_position_tolerance_m')!=.03
                or fit.get('maximum_orientation_tolerance_rad')!=.15 or fit.get('development_only') is not False):
            raise InteractionArtifactError('original support and terminal-cap gates required')
        self.task_spec=get_task_spec(TASK);self.phase_order=tuple(m['fitted_phase_order'])
        if self.phase_order!=tuple(p.name for p in self.task_spec.phase_skills):
            raise InteractionArtifactError('all mandatory ring phases required')
        self.templates=tuple(PhaseTemplate(**item) for item in payload['templates'])
        keys={'|'.join((t.phase,t.anchor_id,t.reference_role,t.reference_kind,t.manipulated_role,t.reference_mode,t.reference_time_basis)) for t in self.templates}
        report=m['completed_skill_fitting']
        if scope!=declared_primary_scope(report,payload['templates']):
            raise InteractionArtifactError('scope proof differs from original complete partition report')
        if keys!=set(scope['admitted_partition_keys']) or set(t.phase for t in self.templates)!=set(self.phase_order):
            raise InteractionArtifactError('actual templates differ from declared policy domain')
        for t in self.templates:
            key='|'.join((t.phase,t.anchor_id,t.reference_role,t.reference_kind,t.manipulated_role,t.reference_mode,t.reference_time_basis))
            row=report['phase_fits'][key]
            if (t.reference_mode=='released' or not row['passed'] or row['completed_parent_count']<8
                    or t.position_tolerance_m>.03 or t.orientation_tolerance_rad>.15):
                raise InteractionArtifactError('unqualified template in initial-sequence domain')
            for actual,name in ((t.lag_seconds,'selected_lag'),(t.position_tolerance_m,'position_tolerance_m'),
                    (t.orientation_tolerance_rad,'orientation_tolerance_rad'),(t.duration_seconds,'duration_seconds'),
                    (t.duration_budget_seconds,'duration_budget_seconds')):
                if abs(actual-row[name])>1e-12:
                    raise InteractionArtifactError('template differs from fitted original numeric gate')
        if len(m['TRAIN_parent_source_groups'])!=100 or len(set(m['TRAIN_parent_source_groups']))!=100:
            raise InteractionArtifactError('retain all100 original parent sources')
        if hashlib.sha256(Path(m['TRAIN_source_manifest_path']).read_bytes()).hexdigest()!=m['TRAIN_source_manifest_sha256']:
            raise InteractionArtifactError('original100 TRAIN source manifest changed')
        self.payload=_json(payload);self.sha256=original_interaction.hashlib.sha256(
            json.dumps(self.payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


InteractionLibrary=PrimarySequenceLibrary


def fit_primary_interactions(sources,config,*,TRAIN_source_manifest_path,synthetic):
    spec=get_task_spec(sources[0].task_name)
    if (config.minimum_completed_parent_sources is None or config.maximum_position_tolerance_m is None
            or config.maximum_orientation_tolerance_rad is None):
        raise InteractionArtifactError('production completed-attempt fitting needs explicit parent support and terminal caps')
    if not config.development_only and config.minimum_completed_parent_sources < 8:
        raise InteractionArtifactError('formal completion fitting requires at least eight completed TRAIN parents per phase')
    manifest_hash=None
    if not synthetic:
        if TRAIN_source_manifest_path is None:
            raise InteractionArtifactError('production fit requires an actual TRAIN acquisition manifest')
        path=Path(TRAIN_source_manifest_path);manifest=json.loads(path.read_text());entries=manifest.get('entries',[])
        if not entries or any(row.get('split')!='train' for row in entries):
            raise InteractionArtifactError('interaction fit manifest must contain TRAIN acquisitions only')
        declared={row['source_id']:row for row in entries if row.get('task')==spec.task_name
            and row.get('interaction_eligible',True)}
        if set(declared)!={source.source_id for source in sources}:
            raise InteractionArtifactError('every declared teacher acquisition must be audited, including incomplete attempts')
        for source in sources:
            row=declared[source.source_id]
            group=':'.join(('train',row['task'],row.get('parent_raw_episode_id') or row['raw_episode_id']))
            if group!=source.grouping_source:
                raise InteractionArtifactError('TRAIN source parent groups differ from acquisition manifest')
        manifest_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    stage_phase={stage:phase.name for phase in spec.phase_skills for stage in phase.TRAIN_expert_stages}
    semantic_flags={source.semantic_acquire_evidence is not None for source in sources}
    if semantic_flags=={True,False}:
        raise InteractionArtifactError('cannot mix legacy and semantic acquire supervision within one task fit')
    semantic_enabled=semantic_flags=={True}
    if semantic_enabled and len({s.semantic_acquire_evidence.derivation_rule_sha256 for s in sources})!=1:
        raise InteractionArtifactError('mixed semantic acquire derivation rules within one task fit')
    groups={};durations={};event_records={};curation=[];failures=[]
    goal_flags={source.command_goal_reference_evidence is not None for source in sources}
    if goal_flags=={True,False}:
        raise InteractionArtifactError('cannot mix legacy and command-goal evidence within one task fit')
    if goal_flags=={True} and len({source.command_goal_reference_evidence.rule_sha256 for source in sources})!=1:
        raise InteractionArtifactError('cannot mix command-goal rules within one task fit')
    goal_records={};canonical_groups={}
    for source in sources:
        goal_evidence=source.command_goal_reference_evidence
        command_goals={} if goal_evidence is None else {row.command_id:row for row in goal_evidence.references}
        pending_goals=set() if goal_evidence is None else {row['command_id'] for row in goal_evidence.pending_commands}
        if source.command_ids is None or source.curve_tcp_xyzw is None or source.curve_label_kind!='measured_executed_TCP':
            raise InteractionArtifactError('production BC requires explicit executed TCP curve labels and command IDs')
        events={event.origin_command_id:event for event in source.completion_events}
        command_ids=set(map(int,source.command_ids[source.command_ids>=0]))
        command_ids.update(row['command_id'] for row in source.command_ledger)
        for command_id in sorted(command_ids):
            indices=np.flatnonzero(source.command_ids==command_id)
            event=events.get(command_id)
            ledger=next((row for row in source.command_ledger if row['command_id']==command_id),{})
            raw_phase=event.phase if event is not None else ledger.get('raw_phase',
                stage_phase.get(source.stages[indices[0]]) if len(indices) else None)
            if semantic_enabled and raw_phase=='acquire':
                excluded_original=next((row for row in source.excluded_completion_events
                    if row['command_id']==command_id),None)
                begin=float(ledger.get('generated_at',source.times[indices[0]] if len(indices) else source.times[0]))
                end=float(ledger.get('cutoff_at',ledger.get('last_executed_at',source.times[-1])))
                curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                    command_id=command_id,phase='acquire',included_in_BC=False,
                    reason='represented_by_verified_semantic_acquire_evidence',
                    command_waypoint_completed='command_completed_at' in ledger,
                    original_phase_completion_event_present=event is not None or excluded_original is not None,
                    original_completion_label_exclusion=excluded_original,
                    duration_is_right_censored_lower_bound=event is None and excluded_original is None,
                    observed_duration_lower_bound_seconds=max(0.,end-begin),
                    original_completion_events_unchanged=True,raw_rows_preserved=True,
                    semantic_evidence_status=source.semantic_acquire_evidence.status,
                    legacy_acquire_curve_rows_double_counted=False))
                continue
            if event is None:
                excluded_event=next((row for row in source.excluded_completion_events if row['command_id']==command_id),None)
                begin=float(ledger.get('generated_at',source.times[indices[0]] if len(indices) else source.times[0]))
                end=float(ledger.get('cutoff_at',ledger.get('last_executed_at',source.times[-1])))
                curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                    command_id=command_id,phase=ledger.get('raw_phase'),included_in_BC=False,
                    reason='no_explicit_successful_skill_completion_event' if excluded_event is None else excluded_event['reason'],
                    command_waypoint_completed='command_completed_at' in ledger,
                    skill_completion_observed=excluded_event is not None,
                    duration_is_right_censored_lower_bound=excluded_event is None,
                    observed_duration_lower_bound_seconds=max(0.,end-begin),raw_rows_preserved=True))
                continue
            if not len(indices):
                failures.append(dict(source_id=source.source_id,command_id=command_id,reason='completion has no executed rows'))
                continue
            if command_id in pending_goals:
                failures.append(dict(source_id=source.source_id,command_id=command_id,
                    reason='completed original command has invalid command-goal intent supervision'))
                continue
            binding=event.binding
            command_goal=command_goals.get(command_id)
            if (event.attempt_end_time>source.times[-1]+1e-8 or
                    any(stage_phase.get(source.stages[i])!=event.phase or source.references[i]!=binding for i in indices
                        if source.references[i] is not None)):
                failures.append(dict(source_id=source.source_id,command_id=command_id,reason='completion interval/phase/reference differs'))
                continue
            indices=np.asarray([i for i in indices if source.references[i]==binding and
                event.attempt_start_time-1e-8<=source.times[i]<event.attempt_end_time-1e-8],int)
            if not len(indices):
                failures.append(dict(source_id=source.source_id,command_id=command_id,reason='completion has no valid reference rows'))
                continue
            key=(event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,
                binding.manipulated_role,event.reference_mode,COMMAND_GOAL if command_goal else binding.reference_time_basis)
            attempt_id=source.source_id+':command'+str(command_id)
            duration=float(event.attempt_end_time-event.attempt_start_time)
            event_rows={};excluded=Counter()
            for lag in config.lag_grid_seconds:
                if command_goal and lag!=0.:
                    excluded['fixed_command_goal_has_no_forward_lag']+=1;continue
                try: terminal_reference=(Pose.from_xyzw(command_goal.intent_reference_xyzw)
                    if command_goal else _completion_reference(source,event,float(lag)))
                except (InteractionArtifactError,KeyError):
                    excluded['lag_terminal_reference_unavailable']+=1;continue
                rows=[]
                for i in indices:
                    try:
                        if binding.reference_kind=='world':reference=Pose(np.zeros(3))
                        elif binding.reference_kind=='phase_origin':reference=Pose.from_xyzw(event.origin_measured_tcp_xyzw)
                        else:
                            track=source.body_tracks[binding.reference_id]
                            reference=track.at(source.times[i]+lag,
                                orientation_required=binding.reference_kind=='observed_se3',
                                same_mode_as=event.reference_mode,not_before=source.times[i])
                            if binding.reference_kind=='translation_only':reference=Pose(reference.position,np.eye(3))
                            if command_goal or binding.reference_time_basis==COMMAND_ISSUE:
                                # Preserve original sample/mode validity; change
                                # only the declared coordinate reference time.
                                reference=terminal_reference
                        curve=Pose.from_xyzw(source.curve_tcp_xyzw[i])
                    except (InteractionArtifactError,KeyError):
                        excluded['invalid_mode_gap_or_reference_row']+=1;continue
                    rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt_id,
                        progress=float((source.times[i]-event.attempt_start_time)/duration),
                        position=reference.rotation.T@(curve.position-reference.position),
                        rotation=reference.rotation.T@curve.rotation,opening=source.expert_opening[i],terminal_event=False))
                if not rows:
                    continue
                terminal=Pose.from_xyzw(event.terminal_tcp_xyzw)
                # Opening is the last genuinely issued command, not a new
                # synthetic gripper label at the native terminal observation.
                rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt_id,
                    progress=1.,position=terminal_reference.rotation.T@(terminal.position-terminal_reference.position),
                    rotation=terminal_reference.rotation.T@terminal.rotation,
                    opening=source.expert_opening[indices[-1]],terminal_event=True))
                event_rows[float(lag)]=rows
            included=0. in event_rows
            curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                command_id=command_id,phase=event.phase,included_in_BC=included,
                reason='explicit_successful_skill_attempt' if included else 'completed_event_without_usable_reference_curve',
                completion_evidence=event.evidence,command_waypoint_completed=event.command_completed,
                completed_duration_seconds=duration,duration_is_right_censored_lower_bound=False,
                issue_reference_mode=event.reference_mode,
                reference_time_basis=COMMAND_GOAL if command_goal else binding.reference_time_basis,
                reference_time_rule_sha256=goal_evidence.rule_sha256 if command_goal else binding.reference_time_rule_sha256,
                reference_time_derivation=event.reference_time_provenance,
                terminal_reference_is_interaction_metric_label=True,
                terminal_reference_is_external_dynamics_observation=False,
                excluded_fit_rows=dict(excluded),raw_rows_preserved=True))
            if not included:
                continue
            durations.setdefault(key,[]).append((source.grouping_source,duration))
            event_records[attempt_id]=(source,event)
            if command_goal:
                goal_records[attempt_id]=command_goal
                reference=Pose.from_xyzw(command_goal.intent_reference_xyzw)
                old=Pose.from_xyzw(event.committed_world_waypoint_xyzw)
                canonical_groups.setdefault(key,[]).append(dict(source=source.grouping_source,progress=1.,terminal_event=True,
                    position=reference.rotation.T@(old.position-reference.position),rotation=reference.rotation.T@old.rotation,
                    opening=source.expert_opening[indices[-1]]))
            for lag,rows in event_rows.items():groups.setdefault((key,lag),[]).extend(rows)
        if semantic_enabled:
            evidence=source.semantic_acquire_evidence
            for segment in evidence.segments:
                binding=segment.binding
                key=('acquire',binding.anchor_id,binding.reference_role,binding.reference_kind,
                    binding.manipulated_role,segment.reference_mode,binding.reference_time_basis)
                attempt_id=source.source_id+':semantic:'+segment.segment_id
                event_rows,excluded=_semantic_acquire_curve_rows(source,segment,config,attempt_id)
                included=0. in event_rows
                duration=segment.attempt_end_time-segment.attempt_start_time
                curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                    phase='acquire',segment_id=segment.segment_id,attempt_id=attempt_id,
                    raw_command_ids=list(segment.raw_command_ids),actual_frame_indices=list(segment.frame_indices),
                    terminal_frame=segment.terminal_frame,reference_partition=list(key),included_in_BC=included,
                    reason='verified_semantic_acquire_mode_segment' if included else 'semantic_segment_without_usable_same_mode_curve',
                    completion_evidence=segment.evidence,completed_duration_seconds=duration,
                    duration_is_right_censored_lower_bound=False,issue_reference_mode=segment.reference_mode,
                    reference_time_basis=binding.reference_time_basis,reference_time_rule_sha256=binding.reference_time_rule_sha256,
                    terminal_reference_is_interaction_metric_label=True,
                    terminal_reference_is_external_dynamics_observation=False,
                    excluded_fit_rows=dict(excluded),raw_rows_preserved=True,
                    raw_bindings=evidence.raw_bindings,semantic_derivation_sha256=evidence.derivation_sha256,
                    legacy_acquire_curve_rows_double_counted=False))
                if not included:continue
                durations.setdefault(key,[]).append((source.grouping_source,duration))
                event_records[attempt_id]=(source,segment)
                for lag,rows in event_rows.items():groups.setdefault((key,lag),[]).extend(rows)
    report=dict(supervision='explicit_completed_skill_attempts_only',all_acquisitions_preserved=True,
        incomplete_attempt_durations_are_right_censored=True,curation=curation,invalid_events=list(failures),
        curve_labels='measured_executed_TCP',held_raw_waypoints_are_independent_teacher_decisions=False,
        phase_success_is_not_task_success=True,task_outcomes_do_not_filter_earlier_completed_skills=True,
        TRAIN_acquisition_sources=sorted(source.source_id for source in sources),
        completion_tolerance_source='parent-group crossfit current-reference terminal endpoint residual',
        old_committed_world_waypoint_residual_is_separate=True,phase_fits={})
    report['duration_semantics']='eligible completed command-level skill attempt interval; not total semantic phase path duration'
    if semantic_enabled:
        report['duration_semantics']='original command intervals outside acquire; verified acquired mode-segment intervals for acquire; censored raw command durations remain audit-only'
        report['semantic_acquire_derivation']=dict(default_legacy_behavior_changed=False,
            original_events_command_ids_masks_and_censoring_preserved=True,
            rule_sha256_values=sorted({s.semantic_acquire_evidence.derivation_rule_sha256 for s in sources}),
            helper_sha256_values=sorted({s.semantic_acquire_evidence.derivation_module_sha256 for s in sources}),
            per_source={s.source_id:dict(derivation_sha256=s.semantic_acquire_evidence.derivation_sha256,
                raw_bindings=s.semantic_acquire_evidence.raw_bindings,status=s.semantic_acquire_evidence.status,
                segment_ids=[x.segment_id for x in s.semantic_acquire_evidence.segments]) for s in sources})
    if goal_flags=={True}:
        from .command_goal_reference import evidence_proof
        report['command_goal_reference_derivation']=dict(
            schema='r10_TRAIN_command_goal_reference_intent_v1',
            rule_sha256_values=sorted({s.command_goal_reference_evidence.rule_sha256 for s in sources}),
            helper_sha256=hashlib.sha256((Path(__file__).parent/'command_goal_reference.py').read_bytes()).hexdigest(),
            original_raw_completion_metric_preserved=True,
            per_source={s.source_id:evidence_proof(s.command_goal_reference_evidence) for s in sources})
        report['completion_tolerance_source']='parent crossfit declared intent tracking; original actual-current metric remains separate'
        report['public_completion_verifier_calibrated']=False
    if failures:
        raise InteractionFittingPending('invalid explicit completion evidence',report)
    grid=np.linspace(0.,1.,config.progress_knots);templates=[]
    for key,completed in durations.items():
        label='|'.join(key);base=groups[(key,0.)];parents={r['source'] for r in base}
        rules={(event_records[r['attempt_id']][0].command_goal_reference_evidence.rule_sha256
            if r['attempt_id'] in goal_records else event_records[r['attempt_id']][1].binding.reference_time_rule_sha256) for r in base}
        if len(rules)!=1:
            raise InteractionArtifactError('mixed reference-time derivation rules within one fit partition')
        rule_sha=next(iter(rules))
        diagnostic=dict(completed_parent_sources=sorted(parents),completed_parent_count=len(parents),
            minimum_completed_parent_sources=config.minimum_completed_parent_sources,
            completed_attempt_count=len({r['attempt_id'] for r in base}),
            completed_rollouts=sorted({r['rollout_id'] for r in base}),reference_mode=key[-2],
            source_grouping='parent raw TRAIN source; all augmentations share a fold',
            reference_time_basis=key[-1],reference_time_rule_sha256=rule_sha)
        report['phase_fits'][label]=diagnostic
        if len(parents)<max(config.minimum_sources,config.minimum_completed_parent_sources):
            diagnostic.update(passed=False,reason='insufficient completed TRAIN parent support');continue
        folds={name:i%2 for i,name in enumerate(sorted(parents))};diagnostic['source_to_fold']=folds
        candidates=[]
        for lag in config.lag_grid_seconds:
            if key[-1] in (COMMAND_ISSUE,COMMAND_GOAL) and lag!=0.:
                candidates.append(dict(lag=float(lag),valid=False,reason='command_issue has a fixed reference; forward motion lag is inapplicable',samples=0))
                continue
            rows=groups.get((key,float(lag)),[])
            if ({r['attempt_id'] for r in rows}!={r['attempt_id'] for r in base} or
                    len(rows)<config.minimum_lag_sample_fraction*len(base)):
                candidates.append(dict(lag=lag,valid=False,reason='completed attempt/sample coverage differs',samples=len(rows)));continue
            errors=[];endpoint_errors=[]
            for fold in (0,1):
                fit=[r for r in rows if folds[r['source']]!=fold];held=[r for r in rows if folds[r['source']]==fold]
                p,r,o=_fit_completed_curve(fit,grid,minimize_terminal_radius=key[0]=='insert')
                for sample in held:
                    position=np.asarray([np.interp(sample['progress'],grid,p[:,a]) for a in range(3)])
                    rotation=Slerp(grid,Rotation.from_matrix(r))([sample['progress']]).as_matrix()[0]
                    pe=float(np.linalg.norm(position-sample['position']))
                    re=float(Rotation.from_matrix(rotation@sample['rotation'].T).magnitude())
                    errors.append((sample['source'],pe**2+re**2))
                    if sample['terminal_event']:
                        source,event=event_records[sample['attempt_id']]
                        terminal=Pose.from_xyzw(event.terminal_tcp_xyzw)
                        old=Pose.from_xyzw(source.expert_tcp_xyzw[event.frame_indices[-1]] if
                            isinstance(event,TRAINSemanticAcquireSegment) else event.committed_world_waypoint_xyzw)
                        current_reference=Pose.from_xyzw(event.terminal_reference_xyzw)
                        current_p=current_reference.rotation.T@(terminal.position-current_reference.position)
                        current_r=current_reference.rotation.T@terminal.rotation
                        endpoint_errors.append(dict(source=sample['source'],attempt_id=sample['attempt_id'],fold=fold,
                            reference_time_basis=key[-1],
                            fitted_reference_endpoint_position_error_m=pe,fitted_reference_endpoint_orientation_error_rad=re,
                            current_reference_endpoint_position_error_m=float(np.linalg.norm(position-current_p)),
                            current_reference_endpoint_orientation_error_rad=float(Rotation.from_matrix(rotation@current_r.T).magnitude()),
                            command_issue_reference_endpoint_position_error_m=pe if event.binding.reference_time_basis==COMMAND_ISSUE else None,
                            command_goal_intent_endpoint_position_error_m=pe if sample['attempt_id'] in goal_records else None,
                            command_goal_intent_endpoint_orientation_error_rad=re if sample['attempt_id'] in goal_records else None,
                            original_current_terminal_metric_pose_preserved=True,
                            committed_world_waypoint_position_error_m=float(np.linalg.norm(terminal.position-old.position)),
                            committed_world_waypoint_orientation_error_rad=float(Rotation.from_matrix(terminal.rotation@old.rotation.T).magnitude())))
            error=float(np.mean([np.mean([e for s,e in errors if s==parent]) for parent in parents]))
            candidates.append(dict(lag=float(lag),valid=True,grouped_TRAIN_CV_error=error,samples=len(rows),
                terminal_crossfit_errors=endpoint_errors))
        valid=[c for c in candidates if c['valid']];diagnostic['candidates']=candidates
        if not valid:diagnostic.update(passed=False,reason='no valid completed-attempt lag');continue
        chosen=min(valid,key=lambda row:(row['grouped_TRAIN_CV_error'],row['lag']))
        terminal_errors=chosen['terminal_crossfit_errors']
        by_parent={parent:dict(position=float(np.median([e['fitted_reference_endpoint_position_error_m'] for e in terminal_errors
            if e['source']==parent])),orientation=float(np.median([e['fitted_reference_endpoint_orientation_error_rad'] for e in terminal_errors
            if e['source']==parent]))) for parent in sorted(parents)}
        pt=max(config.minimum_position_tolerance_m,float(np.quantile([e['position'] for e in by_parent.values()],config.tolerance_quantile)))
        rt=max(config.minimum_orientation_tolerance_rad,float(np.quantile([e['orientation'] for e in by_parent.values()],config.tolerance_quantile)))
        diagnostic.update(selected_lag=chosen['lag'],terminal_parent_residuals=by_parent,
            position_tolerance_m=pt,orientation_tolerance_rad=rt,
            maximum_position_tolerance_m=config.maximum_position_tolerance_m,
            maximum_orientation_tolerance_rad=config.maximum_orientation_tolerance_rad,
            fitted_samples=len(groups[(key,chosen['lag'])]),fitted_sources=sorted(parents),
            fitted_rollouts=sorted({r['rollout_id'] for r in groups[(key,chosen['lag'])]}))
        if pt>config.maximum_position_tolerance_m or rt>config.maximum_orientation_tolerance_rad:
            diagnostic.update(passed=False,reason='terminal residual exceeds TRAIN preregistered cap; no clipping');continue
        p,r,o=_fit_completed_curve(groups[(key,chosen['lag'])],grid,minimize_terminal_radius=key[0]=='insert')
        values=[float(np.median([duration for s,duration in completed if s==parent])) for parent in sorted(parents)]
        duration=float(np.median(values));budget=max(duration,float(np.quantile(values,config.duration_quantile)))
        diagnostic.update(passed=True,completed_duration_by_parent=dict(zip(sorted(parents),values)),
            duration_seconds=duration,duration_budget_seconds=budget,
            incomplete_durations_used_for_quantile=False)
        phase,anchor,role,kind,manipulated,mode,basis=key
        template=dict(phase=phase,anchor_id=anchor,reference_role=role,reference_kind=kind,
            manipulated_role=manipulated,lag_seconds=chosen['lag'],progress=grid,positions=p,rotations=r,opening=o,
            duration_seconds=duration,duration_budget_seconds=budget,position_tolerance_m=pt,
            orientation_tolerance_rad=rt,reference_mode=mode,reference_time_basis=basis,
            reference_time_rule_sha256=rule_sha)
        if basis==COMMAND_GOAL:
            cp,cr,_=_fit_completed_curve(canonical_groups[key],grid)
            template['completion_geometry_xyzw']=Pose(cp[-1],cr[-1]).xyzw()
            diagnostic['completion_geometry_source']='parent-balanced original committed TRAIN waypoint relative to declared command-goal intent'
            diagnostic['public_completion_verifier_calibrated']=False
        templates.append(template)
    available={t['phase'] for t in templates};order=[];omitted=[]
    for phase in spec.phase_skills:
        if phase.name in available:
            if omitted:failures.append(dict(phase=phase.name,reason='missing optional phase is not trailing'))
            order.append(phase.name)
        elif phase.optional_after_native_completion:
            omitted.append(dict(phase=phase.name,reason='no supported completed TRAIN attempt for trailing native-optional phase',
                no_label_was_fabricated=True))
        else:failures.append(dict(phase=phase.name,reason='missing supported completed TRAIN skill attempts'))
    failed_keys=[key for key,d in report['phase_fits'].items() if not d.get('passed')]
    report['required_phase_failures']=failures;report['failed_fit_keys']=failed_keys
    scope=declared_primary_scope(report,templates)
    admitted_failed=[key for key in failed_keys if key in scope['admitted_partition_keys']]
    if failures or admitted_failed:
        raise InteractionFittingPending('initial-sequence mandatory support/terminal caps still pending',report)
    templates=[t for t in templates if t['reference_mode']!='released']

    used={r['rollout_id'] for (key,lag),rows in groups.items() if lag==0. for r in rows}
    metadata=dict(schema=LIBRARY_SCHEMA,task_name=spec.task_name,training_complete=True,
        minimax_helper_sha256=hashlib.sha256(Path(minimax.__file__).read_bytes()).hexdigest(),
        terminal_translation_estimator=dict(insert='source-balanced minimum enclosing ball',all_other_phases='original parent-balanced mean',physical_canonical_geometry='original mean unchanged'),
        fit_split='train',selection_split='train',TEST_used_for_fitting=False,synthetic=synthetic,
        TRAIN_source_ids=sorted(source.source_id for source in sources),
        TRAIN_parent_source_groups=sorted({source.grouping_source for source in sources}),
        TRAIN_source_provenance={s.source_id:dict(source_kind=s.source_kind,parent_source_group=s.grouping_source,
            augmentation_id=s.augmentation_id,raw_episode_id=s.raw_episode_id,parent_raw_episode_id=s.parent_raw_episode_id) for s in sources},
        TRAIN_source_manifest_path=str(TRAIN_source_manifest_path) if TRAIN_source_manifest_path is not None else None,
        TRAIN_source_manifest_sha256=manifest_hash,
        fitting_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        task_spec_module_sha256=hashlib.sha256((Path(__file__).parent/'specs.py').read_bytes()).hexdigest(),
        fit_configuration=asdict(config),source_grouped_lag_selection=report['phase_fits'],
        completed_skill_fitting=report,initial_sequence_policy_scope=scope,
        numeric_helpers_sha256=hashlib.sha256(Path(original_interaction.__file__).read_bytes()).hexdigest(),
        original_full_scope_fitter_returned_library=False,BC_used_source_ids=sorted(used),
        BC_excluded_source_ids=sorted({s.source_id for s in sources}-used),
        all_acquisition_sources_used_for_BC=used=={s.source_id for s in sources},
        completion_tolerances_are_terminal_only=True,
        reference_time_schema=REFERENCE_TIME_SCHEMA,
        reference_time_helper_sha256=hashlib.sha256((Path(__file__).parent/'reference_time.py').read_bytes()).hexdigest(),
        reference_time_rule_sha256_values=sorted({event.binding.reference_time_rule_sha256 for source in sources
            for event in source.completion_events if event.binding.reference_time_basis==COMMAND_ISSUE}),
        formal_comparison_ready=False,
        development_support_override=config.development_only,
        target_label_column_is_not_a_body_track=True,expert_actions_are_training_labels_not_runtime_inputs=True,
        fitted_phase_order=order,omitted_optional_phases=omitted,
        TRAIN_stage_coverage={s.source_id:dict(Counter(s.stages)) for s in sources})
    if semantic_enabled:
        metadata['semantic_acquire_derivation']=report['semantic_acquire_derivation']
        metadata['semantic_acquire_runtime_schema_requires_explicit_release_review']=True
    if goal_flags=={True}:
        metadata['command_goal_reference_derivation']=report['command_goal_reference_derivation']
        metadata['public_completion_verifier_calibrated']=False
        metadata['command_goal_tolerance_scope']='intent_tracking_TRAIN_crossfit; current_public_canonical_geometry_verifier_not_calibrated'
    return InteractionLibrary(dict(metadata=metadata,templates=templates),allow_synthetic=synthetic)
