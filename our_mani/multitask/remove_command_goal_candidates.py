"""Independent TRAIN remove-cup reference candidates, never runtime/library input.

Raw references/events are retained. Acquiring uses the source's nearest-spoke
coordinate. Rotating phases are an unqualified numerical hypothesis unless
actual waypoint-parent evidence is separately supplied; static source alone
cannot fill the deliberately omitted dummy hierarchy in old acquisitions.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from ..contracts import Pose
from . import acquisition_lineage as a,command_goal_reference as goal,interaction

SCHEMA='r10_independent_TRAIN_remove_reference_candidate_v1'
TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')
INTENT_BASIS='independent_TRAIN_candidate_fixed_intent'


def require(value,message):
    if not value:raise ValueError('remove TRAIN candidate: '+message)


@dataclass(frozen=True)
class TRAINRemoveReferenceCandidate:
    task_name:str
    source_id:str
    parent_source_group:str
    command_id:int
    issued_at:float
    binding:interaction.TRAINReferenceBinding
    original_binding:interaction.TRAINReferenceBinding
    reference_mode:str
    intent_reference_xyzw:np.ndarray
    phase:str
    source_semantics_qualified:bool
    provenance:dict

    def __post_init__(self):
        require(self.task_name in TASKS and self.phase in ('approach','acquire','detach')
            and self.parent_source_group.startswith('train:'+self.task_name+':') and type(self.command_id) is int
            and self.command_id>=0 and self.reference_mode in ('external','attached','released')
            and self.binding.reference_kind=='observed_se3','explicit remove TRAIN candidate identity/binding required')
        object.__setattr__(self,'intent_reference_xyzw',goal.frozen_pose(self.intent_reference_xyzw))
        require(self.provenance.get('schema')==SCHEMA and self.provenance.get('runtime_enabled') is False
            and self.provenance.get('original_binding_retained_separately') is True
            and self.provenance.get('raw_events_commands_masks_rewritten') is False,'candidate cannot impersonate raw or runtime reference')
        if self.phase=='acquire':
            require(self.binding.reference_role=='selected_spoke' and self.binding.reference_id.startswith('spoke')
                and self.original_binding.reference_role=='selected_cup' and self.source_semantics_qualified is True,
                'acquire must expose explicit source nearest-spoke relation')
        else:require(self.source_semantics_qualified is False,'unrecorded dummy hierarchy cannot be silently qualified')


@dataclass(frozen=True)
class TRAINRemoveCandidateEvidence:
    task_name:str
    source_id:str
    parent_source_group:str
    references:tuple
    pending_commands:tuple
    body_tracks:dict
    original_verified_evidence:object
    derivation_proof:dict

    def __post_init__(self):
        original=self.original_verified_evidence
        require(self.task_name in TASKS and isinstance(original,goal.TRAINCommandGoalEvidence)
            and (original.task_name,original.source_id,original.parent_source_group)==(self.task_name,self.source_id,self.parent_source_group)
            and self.derivation_proof.get('actual_raw_source_verified') is True,'strict original verified source must precede candidate')
        ids=[r.command_id for r in self.references]+[r['command_id'] for r in self.pending_commands]
        require(len(ids)==len(set(ids)) and all((r.task_name,r.source_id,r.parent_source_group)==(self.task_name,self.source_id,self.parent_source_group) for r in self.references),
            'duplicate or mixed candidate command identities')


def derive_candidate(case,source,rule,permission,*,include_unqualified_rotating=False):
    require(source.task_name in TASKS and case['entry']['split']=='train','remove TRAIN-only candidate required')
    require(a.read(Path(case['path']))==case['record'],'original canonical record changed in memory')
    original=goal.verify_original_source(case,source,rule);root=Path(rule.root);files={}
    permitted=a.read(a.checked_pointer(root,permission,files))
    require(set(permitted)=={'schema','TRAIN_only','runtime_enabled','production_enabled','helper','task_sources','rotating_helper',
        'original_readonly_evidence','waypoint_parent_hierarchy_recorded','allow_unqualified_rotating_numeric_diagnostic'}
        and permitted['schema']==SCHEMA and permitted['TRAIN_only'] is True and permitted['runtime_enabled'] is False
        and permitted['production_enabled'] is False and permitted['waypoint_parent_hierarchy_recorded'] is False,
        'independent candidate permission must retain runtime/hierarchy limits')
    require(a.checked_pointer(root,permitted['helper'],files)==Path(__file__).resolve(),'candidate helper binding changed')
    require(set(permitted['task_sources'])==set(TASKS),'exact two remove source tasks required')
    path=a.checked_pointer(root,permitted['task_sources'][source.task_name],files)
    rt=a.checked_pointer(root,permitted['rotating_helper'],files);a.checked_pointer(root,permitted['original_readonly_evidence'],files)
    tree=ast.parse(path.read_text());expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    text=ast.unparse(expert)
    require('task.w2.set_position(task.w2_rel_pos, relative_to=target_spoke, reset_dynamics=False)' in text
        and 'task.w2.set_orientation(task.w2_rel_ori, relative_to=target_spoke, reset_dynamics=False)' in text
        and 'target_spoke = min(task.spokes' in text and 'target_cup = min(task.cups' in text,
        'original wp2 nearest-spoke source relation changed')
    branches=[]
    for node in expert.body:
        if not isinstance(node,ast.If):continue
        while isinstance(node,ast.If):
            stages=[n.value.value for row in node.body for n in ast.walk(row) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='stage' for t in n.targets) and isinstance(n.value,ast.Constant)]
            delays=[n.value.value for row in node.body for n in ast.walk(row) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='t_delay' for t in n.targets) and isinstance(n.value,ast.Constant)]
            if stages:branches.append((stages,delays,ast.unparse(node.test)))
            node=node.orelse[0] if len(node.orelse)==1 and isinstance(node.orelse[0],ast.If) else None
    require(any(s==['wp2'] and d==[0.] for s,d,t in branches)
        and any(s==['wp1'] and d==[1.] for s,d,t in branches)
        and any(s==['wp1'] and d==[.5] for s,d,t in branches)
        and all(d==[.5] for s,d,t in branches if s==['wp3']), 'original stage-dependent0/1/.5 forecast declarations changed')
    operator=next(n for n in ast.parse(rt.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_target_pose')
    require('frame_base.rotate([0, 0, rot])' in ast.unparse(operator) and 'target_pose = waypoint.get_pose()' in ast.unparse(operator),
        'original rotating frame operator changed')
    if include_unqualified_rotating:require(permitted['allow_unqualified_rotating_numeric_diagnostic'] is True,'unqualified rotating hypothesis not allowed')
    z=case['arrays'];ids=list(z['body_ids']);initial=case['record']['native_reset_receipt']['native_initial_snapshot']
    require(initial['native_task_source_sha256']==permitted['task_sources'][source.task_name]['sha256'],'actual task archive/source mismatch')
    require('waypoint1' in initial['skipped_expert_waypoint_dummies'] and 'waypoint3' in initial['skipped_expert_waypoint_dummies'],
        'hierarchy omission evidence differs')
    tracks={}
    for j,body in enumerate(ids):
        tracks[body]=interaction.TRAINBodyTrack(body,source.times,np.c_[z['target_positions'][:,j],Rotation.from_matrix(z['target_rotations'][:,j]).as_quat()],
            tuple(('external','attached','released')[int(k)] for k in z['modes'][:,j]),z['target_rotation_observable'][:,j],
            pose_valid=z['native_pose_valid'][:,j]&z['mode_valid'][:,j])
    ledger={c['command_id']:c for c in source.command_ledger};references=[];pending=[]
    configured={'acquire'}|({'approach','detach'} if include_unqualified_rotating else set())
    for c in source.command_ledger:
        phase=c['raw_phase']
        if phase not in configured:continue
        cid=c['command_id'];found=np.flatnonzero(np.isclose(source.times,c['generated_at'],atol=1e-8,rtol=0));require(len(found)==1,'unique original issue frame needed');i=int(found[0])
        old=interaction.TRAINReferenceBinding(c['reference_role'],c['reference_body'],c['reference_kind'],c['anchor_id'],c['manipulated_role'])
        if phase=='acquire':
            cups=[body for body in ids if body.startswith('cup')];cup=min(cups,key=lambda body:np.linalg.norm(z['target_positions'][i,ids.index(body)]-z['measured_tcp_xyzw'][i,:3]))
            require(cup==c['manipulated_body'],'raw selected cup differs from source nearest-cup rule')
            spokes=[body for body in ids if body.startswith('spoke')]
            body=min(spokes,key=lambda body:np.linalg.norm(z['target_positions'][i,ids.index(body)]-z['target_positions'][i,ids.index(cup)]))
            binding=interaction.TRAINReferenceBinding('selected_spoke',body,'observed_se3',old.anchor_id+':independent_spoke_candidate','selected_cup')
        else:body=old.reference_id;binding=old
        j=ids.index(body)
        if not bool(z['native_pose_valid'][i,j] and z['mode_valid'][i,j] and z['target_rotation_observable'][i,j]):
            pending.append(dict(command_id=cid,phase=phase,reason='candidate issue GT pose/mode/orientation invalid',original_raw_retained=True));continue
        issue=Pose(z['target_positions'][i,j],z['target_rotations'][i,j]);intent=issue;delay=0.
        if phase in ('approach','detach'):
            previous=ledger.get(cid-1);delay=(1. if previous and previous['raw_stage']=='wp0' else .5) if phase=='approach' else .5
            fi=ids.index('frame');require(z['native_pose_valid'][i,fi],'recorded frame axis invalid')
            yaw=float(np.asarray(initial['active_motion_schedule']['yaw_speed']).reshape(-1)[0])
            rot=Rotation.from_rotvec(z['target_rotations'][i,fi,:,2]*np.deg2rad(yaw)*delay).as_matrix();center=z['target_positions'][i,fi]
            intent=Pose(center+rot@(issue.position-center),rot@issue.rotation)
        references.append(TRAINRemoveReferenceCandidate(source.task_name,source.source_id,source.grouping_source,cid,float(c['generated_at']),binding,old,
            ('external','attached','released')[int(z['modes'][i,j])],intent.xyzw(),phase,phase=='acquire',
            dict(schema=SCHEMA,runtime_enabled=False,original_binding_retained_separately=True,raw_events_commands_masks_rewritten=False,
                TRAIN_private_labels_not_runtime_observations=True,source_declared_forecast_seconds=delay,
                waypoint_parent_hierarchy_proven=False if phase!='acquire' else None,source_literal_relative_spoke_proven=phase=='acquire',
                original_permission=permission,original_raw_bindings=original.raw_bindings)))
    require(goal.fingerprint(z)==original.original_masks_sha256,'candidate modified raw masks')
    files.update(original.derivation_proof['verified_files_sha256'])
    return TRAINRemoveCandidateEvidence(source.task_name,source.source_id,source.grouping_source,tuple(references),tuple(pending),tracks,original,
        dict(schema=SCHEMA,actual_raw_source_verified=True,verified_files_sha256=files,runtime_or_production_enabled=False,
            includes_unqualified_rotating_numeric_hypothesis=include_unqualified_rotating,original_raw_bindings=original.raw_bindings))


def fit_candidate_diagnostic(sources,evidence,config):
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
            raw_binding=event.binding;goal=goals.get(cid);binding=goal.binding if goal else raw_binding;mode=goal.reference_mode if goal else event.reference_mode;basis=INTENT_BASIS if goal else binding.reference_time_basis
            indices=np.flatnonzero(source.command_ids==cid)
            require(not any(stages.get(source.stages[i])!=event.phase or source.references[i]!=raw_binding for i in indices
                if source.references[i] is not None),'original completion phase/reference differs')
            indices=[i for i in indices if source.references[i]==raw_binding
                and event.attempt_start_time-1e-8<=source.times[i]<event.attempt_end_time-1e-8]
            key=(event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,binding.manipulated_role,mode,basis)
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
                            reference=(proof.body_tracks if goal else source.body_tracks)[binding.reference_id].at(source.times[i]+lag,
                                orientation_required=binding.reference_kind=='observed_se3',same_mode_as=mode,not_before=source.times[i])
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
    report=dict(schema='r10_independent_remove_candidate_actual100_fullcurve_v1',TRAIN_only=True,
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
    report['runtime_or_production_enabled']=False
    report['rotating_hierarchy_qualified']=False
    report['original_raw_BINDINGS_changed']=False
    report['independent_candidate_binding_changes_declared']=True
    report['parent_folds_and_numeric_kernel_unchanged']=True
    return report
