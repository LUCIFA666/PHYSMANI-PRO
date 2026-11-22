"""Offline TRAIN teacher adapters and grouped interaction/colour/layout fits.

Native supervision in this module is an offline label. It never changes the
collector's causal observations, visibility or dynamics masks. Outputs remain
development artifacts pending the separately owned full comparison release.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose

from .evaluation import digest, file_sha
from .acquisition_manifest import (load_acquisition_manifest,actual_case_plan_sha,COMPOSITE_VERSION,
    BUNDLE_VERSION,COUNTED_BUNDLE_VERSION)
from .interaction import (InteractionArtifactError, InteractionFitConfig, TRAINBodyTrack,
    TRAINInteractionSource, TRAINReferenceBinding, TRAINPhaseCompletionEvent, fit_interaction_library)
from .public_roles import (ColourFitConfig, TRAINColourSample, fit_colour_artifact, _lab,
    PublicRoleResolutionError)
from .specs import TASK_SPECS, get_task_spec


def require(condition,message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def pointer(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=file_sha(path))


def inside(root,path):
    root=Path(root).resolve();path=(root/path).resolve()
    require(path.is_relative_to(root),'TRAIN cache must stay inside the desktop project')
    return path


def load_train_cases(root,manifest_path):
    """Verify canonical collector receipts, raw child receipts and every output."""
    root=Path(root).resolve();manifest_path=inside(root,manifest_path)
    manifest,composite_proof=load_acquisition_manifest(root,manifest_path)
    entries=manifest.get('entries',[])
    require(manifest.get('version') in ('r10_causal_body_tracks_v1',COMPOSITE_VERSION,BUNDLE_VERSION,COUNTED_BUNDLE_VERSION) and entries
        and manifest.get('complete') is True and manifest.get('TEST_used') is False
        and not manifest.get('missing_case_ids')
        and manifest.get('expected')==manifest.get('completed')==len(entries)==len(manifest.get('case_receipts',[])),
        'need a complete declared TRAIN acquisition manifest, not a TEST/reset manifest')
    require(all(row.get('split')=='train' and row.get('task') in TASK_SPECS
        and row.get('source_kind') in ('TRAIN_native_generated_augmentation','TRAIN_native_generated_hold') for row in entries),
        'interaction artifacts audit TRAIN teacher/hold acquisitions only')
    declared={(row['task'],row['raw_episode_id']):row for row in entries}
    require(len(declared)==len(entries),'duplicate TRAIN native rollout identity')
    cases=[];seen=set()
    for item in manifest.get('case_receipts',[]):
        path=inside(root,item['path']);require(file_sha(path)==item['sha256'],'TRAIN receipt changed')
        record=read(path);entry=record.get('entry',{});key=(entry.get('task'),entry.get('raw_episode_id'))
        require(key in declared and key not in seen and entry==declared[key],
            'receipt and declared TRAIN source differ')
        require(record.get('status')=='completed' and record.get('split')=='train'
            and record.get('case_id')==item.get('case_id')
            and record.get('plan_sha256')==actual_case_plan_sha(manifest,composite_proof,item.get('case_id'))
            and isinstance(record.get('plan_sha256'),str) and len(record['plan_sha256'])==64
            and record.get('case_isolation')=='fresh_simulator_process_per_episode'
            and record.get('teacher_actions_used') is (entry.get('source_kind')=='TRAIN_native_generated_augmentation')
            and record.get('policy_history_contains_native_GT') is False,
            'incomplete or non-TRAIN teacher acquisition cannot enter fitting')
        parent=entry.get('parent_raw_episode_id')
        require(isinstance(parent,str) and parent and entry.get('parent_source_group')==
            f"train:{entry['task']}:{parent}",'TRAIN augmentation parent identity missing')
        files=record.get('files_sha256',{});require(files,'TRAIN receipt outputs are not hashed')
        for name,checksum in files.items():
            require(file_sha(inside(root,name))==checksum,'TRAIN receipt output changed: '+name)
        require(files.get(entry['cache'])==entry['cache_sha256'],'cache is not bound by its collector receipt')
        process=record.get('parent_process_receipt')
        require(isinstance(process,dict),'fresh native TRAIN raw child/launcher receipt is required')
        child_path=inside(root,process['child_case_path'])
        require(file_sha(child_path)==process['child_case_sha256'],'raw TRAIN child receipt changed')
        child=read(child_path)
        require({k:v for k,v in record.items() if k!='parent_process_receipt'}==child
            and type(record.get('process_id')) is int and record['process_id']>0
            and child.get('process_id')==record['process_id']
            and type(process.get('process_id')) is int and process['process_id']>0,
            'canonical TRAIN receipt is not the exact raw child/process')
        if composite_proof is None:
            require(file_sha(inside(root,process['launcher']))==process['launcher_sha256'],
                'TRAIN child launcher source changed')
        with np.load(inside(root,entry['cache']),allow_pickle=False) as cache:
            arrays={name:cache[name].copy() for name in cache.files}
        clock=record.get('alignment',{}).get('clock',{});ticks=clock.get('ticks');n=len(arrays['times'])
        require(type(ticks) is int and 0<=ticks<=entry.get('maximum_collection_ticks',200)
            and n==ticks+1 and entry.get('frames')==n and clock.get('dt')==.05
            and clock.get('budget_seconds')==10.
            and abs(clock.get('simulation_seconds',-1)-ticks*.05)<=1e-6
            and all(isinstance(record['alignment'].get(field),(float,int)) and
                abs(record['alignment'][field]-ticks*.05)<=2e-5 for field in
                ('actual_physics_seconds','actual_task_seconds')),
            'TRAIN acquisition native clock differs from actual executed cache')
        case=dict(path=path,record=record,entry=entry,arrays=arrays)
        if entry.get('phase_completion_events_receipt') is not None:
            command_path=entry['phase_completion_events_receipt']
            require(command_path==entry.get('teacher_command_receipt') and command_path in files,
                'phase completion and raw command ledger must be bound by the TRAIN receipt')
            case['command_receipt']=read(inside(root,command_path))
            case['command_receipt_sha256']=files[command_path]
            case['command_receipt_verified']=True
        cases.append(case);seen.add(key)
    require(seen==set(declared),'every declared TRAIN receipt must participate; no silent omission')
    if composite_proof is not None:
        for filename,checksum in composite_proof['verified_files_sha256'].items():
            require(file_sha(filename)==checksum,'historical acquisition lineage input changed')
    return manifest,cases


def interaction_source(case, *, reference_time_rule=None, semantic_acquire_rule=None,command_goal_rule=None):
    """Use only each action's actual reference and preserve invalid mode/gaps.

    native_pose_valid & mode_valid is an offline *interaction-label* mask.
    target_valid also masks controlled/contact dynamics and remains unchanged.
    An unreferenced press plate cannot invalidate a button-relative action.
    """
    entry,a=case['entry'],case['arrays'];task=entry['task'];spec=get_task_spec(task)
    t=np.asarray(a['times'],float);ids=tuple(map(str,a['body_ids']));n,b=len(t),len(ids)
    require(entry.get('source_kind')=='TRAIN_native_generated_augmentation'
        and entry.get('interaction_eligible',True) is True and entry.get('teacher_used',True) is True,
        'only actual TRAIN teacher acquisitions are eligible for interaction BC')
    require(1<=n<=201 and t.shape==(n,) and np.isfinite(t).all() and abs(t[0])<=1e-6
        and np.allclose(np.diff(t),.05,atol=1e-6,rtol=0.),
        'teacher clock has no usable causal sequence')
    for name in ('native_pose_valid','mode_valid','target_valid','target_rotation_observable','modes'):
        require(a[name].shape==(n,b),'incorrect offline reference mask shape: '+name)
    for name in ('native_pose_valid','mode_valid','target_valid','target_rotation_observable'):
        require(a[name].dtype==np.bool_,'offline label masks must remain boolean')
    require(np.isin(a['modes'],(0,1,2)).all(),'unknown public motion-mode code')
    selected=np.asarray(a['teacher_action_valid'],bool)
    require(selected.shape==(n,),'incorrect actual TRAIN teacher execution mask')
    require(entry.get('interaction_opening_array')=='teacher_applied_opening'
        and 'teacher_applied_opening' in a and 'teacher_desired_opening' in a,
        'interaction fitting requires v4 actual applied opening labels; legacy raw opening is insufficient')
    require(a['teacher_applied_opening'].shape==(n,) and a['teacher_desired_opening'].shape==(n,)
        and np.array_equal(a['teacher_opening'][selected],a['teacher_applied_opening'][selected]),
        'teacher_opening compatibility alias must equal actual applied opening')
    require(entry.get('interaction_tcp_array')=='measured_tcp_xyzw'
        and case.get('command_receipt_verified') is True,
        'completed-skill fitting requires v5 executed TCP and SHA-verified command/phase ledger')
    command_receipt=case.get('command_receipt',{})
    require(command_receipt.get('version')=='r10_TRAIN_committed_waypoint_commands_v1'
        and isinstance(command_receipt.get('commands'),list)
        and isinstance(command_receipt.get('phase_completion_events'),list),
        'explicit command and phase completion events are required')
    command_ids=np.asarray(a['teacher_command_id'])
    require(command_ids.shape==(n,) and command_ids.dtype.kind in 'iu'
        and np.all(command_ids[selected]>=0),'command identities must align with actual executed actions')
    stage_phase={stage:phase.name for phase in spec.phase_skills for stage in phase.TRAIN_expert_stages}
    references=[None]*n;required=set();audit=Counter();per_phase=defaultdict(Counter)
    for i in np.flatnonzero(selected):
        stage=str(a['teacher_stage'][i]);require(stage in stage_phase,'unmapped TRAIN teacher stage')
        phase=stage_phase[stage]
        require(str(a['teacher_phase'][i])==phase,'TRAIN teacher stage and phase disagree')
        kind=str(a['teacher_reference_kind'][i]);body=str(a['teacher_reference_body'][i])
        binding=TRAINReferenceBinding(str(a['teacher_reference_role'][i]),body or None,kind,
            str(a['teacher_anchor_id'][i]),str(a['teacher_manipulated_role'][i]))
        per_phase[phase]['actual_teacher_commands']+=1
        if kind not in ('world','phase_origin'):
            require(body in ids,'teacher reference is not a persistent physical body')
            j=ids.index(body);required.add(body)
            usable=bool(a['native_pose_valid'][i,j] and a['mode_valid'][i,j])
            if kind=='observed_se3': usable &= bool(a['target_rotation_observable'][i,j])
            if not usable:
                audit['excluded_reference_commands']+=1
                per_phase[phase]['excluded_invalid_reference_commands']+=1;continue
            if not a['target_valid'][i,j]:
                audit['controlled_reference_teacher_labels_used']+=1
                per_phase[phase]['controlled_reference_teacher_labels_used']+=1
        references[i]=binding;audit['usable_reference_commands']+=1
    tracks={}
    for body in required:
        j=ids.index(body);valid=a['native_pose_valid'][:,j] & a['mode_valid'][:,j]
        p,r=np.asarray(a['target_positions'][:,j]),np.asarray(a['target_rotations'][:,j])
        require(p.shape==(n,3) and r.shape==(n,3,3) and np.isfinite(p).all() and np.isfinite(r).all(),
            'offline body labels must contain finite stored SE3')
        xyzw=np.c_[p,Rotation.from_matrix(r).as_quat()]
        tracks[body]=TRAINBodyTrack(body,t,xyzw,tuple(('external','attached','released')[int(k)]
            for k in a['modes'][:,j]),a['target_rotation_observable'][:,j],pose_valid=valid)
    events=[];event_exclusions=[]
    ledger={row['command_id']:row for row in command_receipt['commands']}
    require(len(ledger)==len(command_receipt['commands']),'duplicate raw TRAIN command IDs')
    def frame_at(timestamp):
        found=np.flatnonzero(np.isclose(t,float(timestamp),atol=1e-8,rtol=0.))
        require(len(found)==1,'completion evidence must match one actual acquired timestamp')
        return int(found[0])
    def same_pose(first,second):
        first,second=Pose.from_xyzw(np.asarray(first,float)),Pose.from_xyzw(np.asarray(second,float))
        return (np.linalg.norm(first.position-second.position)<=1e-8 and
            Rotation.from_matrix(first.rotation@second.rotation.T).magnitude()<=1e-8)
    for row in command_receipt['phase_completion_events']:
        require(row.get('version')=='r10_TRAIN_phase_completion_event_v1'
            and row.get('label_namespace')=='TRAIN_metric_only'
            and row.get('from_phase')==row.get('phase'),'incorrect phase completion label namespace/schema')
        cid=row['origin_command_id'];require(cid in ledger,'completion event lacks raw command provenance')
        command=ledger[cid]
        require(type(cid) is int and type(row['origin_executed_tick_count']) is int,
            'raw completion command IDs and tick counts must be integers')
        require(row['origin_executed_tick_count']==command['executed_tick_count']
            and row['origin_command_completed']==('command_completed_at' in command)
            and row['attempt_start_time']==command['generated_at']
            and row['phase']==command['raw_phase']
            and all(row[event_field]==command[command_field] for event_field,command_field in (
                ('origin_reference_body','reference_body'),('origin_reference_role','reference_role'),
                ('origin_reference_kind','reference_kind'),('origin_anchor_id','anchor_id'),
                ('origin_manipulated_role','manipulated_role'))),
            'completion event differs from its actual raw command execution')
        begin,end=frame_at(row['attempt_start_time']),frame_at(row['attempt_end_time'])
        require(same_pose(row['current_measured_tcp_xyzw'],a['measured_tcp_xyzw'][end])
            and same_pose(row['origin_measured_tcp_xyzw'],a['measured_tcp_xyzw'][begin])
            and same_pose(row['committed_world_waypoint_xyzw'],a['teacher_tcp_xyzw'][begin]),
            'phase completion TCP/old waypoint snapshot differs from acquired rows')
        if row.get('current_reference_native_pose_valid') is not True:
            event_exclusions.append(dict(command_id=cid,phase=row['phase'],reason='completed skill terminal reference metric pose invalid'))
            continue
        binding=TRAINReferenceBinding(row['origin_reference_role'],row['origin_reference_body'] or None,
            row['origin_reference_kind'],row['origin_anchor_id'],row['origin_manipulated_role'])
        mode=row['origin_reference_motion_mode']
        if binding.reference_kind in ('phase_origin','world'):mode='not_applicable'
        else:
            require(binding.reference_id in ids,'completion reference is not an acquired physical body')
            j=ids.index(binding.reference_id)
            if not a['native_pose_valid'][end,j] or not a['mode_valid'][begin,j]:
                event_exclusions.append(dict(command_id=cid,phase=row['phase'],reason='completed skill origin mode or terminal reference label invalid'))
                continue
            if binding.reference_kind=='observed_se3' and not a['target_rotation_observable'][end,j]:
                event_exclusions.append(dict(command_id=cid,phase=row['phase'],reason='completed skill terminal full orientation label unavailable'))
                continue
            reference=np.r_[a['target_positions'][end,j],Rotation.from_matrix(a['target_rotations'][end,j]).as_quat()]
            if binding.reference_kind=='translation_only':reference[3:]=[0.,0.,0.,1.]
            require(same_pose(row['current_reference_native_pose_xyzw'],reference),
                'phase completion current reference differs from actual terminal body labels')
            require(mode==('external','attached','released')[int(a['modes'][begin,j])],
                'phase completion origin mode differs from actual command issue mode')
        if mode not in ('external','attached','released','not_applicable'):
            event_exclusions.append(dict(command_id=cid,phase=row['phase'],reason='completed skill origin reference mode invalid'))
            continue
        events.append(TRAINPhaseCompletionEvent(phase=row['phase'],to_phase=row['to_phase'],evidence=row['evidence'],
            origin_command_id=cid,eligible_command_ids=tuple(row['eligible_command_ids']),
            attempt_start_time=float(row['attempt_start_time']),attempt_end_time=float(row['attempt_end_time']),
            binding=binding,reference_mode=mode,origin_measured_tcp_xyzw=row['origin_measured_tcp_xyzw'],
            terminal_tcp_xyzw=row['current_measured_tcp_xyzw'],
            terminal_reference_xyzw=row['current_reference_native_pose_xyzw'],
            committed_world_waypoint_xyzw=row['committed_world_waypoint_xyzw'],
            executed_tick_count=row['origin_executed_tick_count'],command_completed=row['origin_command_completed'],
            label_namespace=row['label_namespace']))
    result=TRAINInteractionSource(task_name=task,source_id=entry['source_id'],split='train',
        source_kind='TRAIN_native_generated_augmentation',times=t,
        expert_tcp_xyzw=a['teacher_tcp_xyzw'],expert_opening=a['teacher_applied_opening'],
        measured_tcp_xyzw=a['measured_tcp_xyzw'],stages=tuple(map(str,a['teacher_stage'])),
        references=tuple(references),body_tracks=tracks,parent_source_group=entry['parent_source_group'],
        augmentation_id=entry['augmentation_id'],raw_episode_id=entry['raw_episode_id'],
        parent_raw_episode_id=entry['parent_raw_episode_id'],
        command_ids=np.where(selected,command_ids,-1),completion_events=tuple(events),
        curve_tcp_xyzw=a['measured_tcp_xyzw'],curve_label_kind='measured_executed_TCP',
        command_ledger=tuple(command_receipt['commands']),excluded_completion_events=tuple(event_exclusions))
    original_source=result
    reference_time_audit=dict(reference_time_basis='current',rule_applied=False)
    if reference_time_rule is not None:
        from .reference_time import derive_source
        # Only after every original raw row/event/clock/mode was validated.
        result,reference_time_audit=derive_source(result,case,reference_time_rule)
    semantic_acquire_audit=dict(rule_applied=False)
    if semantic_acquire_rule is not None:
        from .semantic_acquire import derive_source
        result,semantic_acquire_audit=derive_source(result,case,semantic_acquire_rule)
    command_goal_audit=dict(rule_applied=False)
    if command_goal_rule is not None and task in command_goal_rule.tasks:
        from dataclasses import replace
        from .command_goal_reference import derive_source,evidence_proof
        evidence=derive_source(case,original_source,command_goal_rule)
        result=replace(result,command_goal_reference_evidence=evidence)
        command_goal_audit=dict(rule_applied=True,proof=evidence_proof(evidence),
            verified_files_sha256=evidence.derivation_proof['verified_files_sha256'])
    return result,dict(audit,per_phase={key:dict(value) for key,value in per_phase.items()},
        referenced_physical_bodies=sorted(required),dynamics_masks_modified=False,
        interaction_opening_array='teacher_applied_opening',raw_desired_opening_used_as_applied=False,
        interaction_tcp_array='measured_tcp_xyzw',raw_held_waypoint_used_as_independent_curve_label=False,
        command_receipt_sha256=case.get('command_receipt_sha256'),
        total_raw_commands=len(command_receipt['commands']),explicit_phase_completion_events=len(events),
        excluded_phase_completion_events=event_exclusions,
        teacher_SE3_is_runtime_policy_observation=False,
        interaction_reference_mask='native_pose_valid AND mode_valid; full orientation labels when required',
        controlled_interaction_labels_are_exogenous_dynamics=False,
        reference_time_derivation=reference_time_audit,
        semantic_acquire_derivation=semantic_acquire_audit,command_goal_reference_derivation=command_goal_audit)


def original_prior_binding(root,demonstrations_path,prior_path):
    """Verify all 1400 already fitted offline action-phase priors separately."""
    root=Path(root).resolve();dpath=inside(root,demonstrations_path);ppath=inside(root,prior_path)
    demonstrations,prior=read(dpath),read(ppath)
    train=[row for row in demonstrations.get('episodes',[]) if row.get('split')=='train']
    require(demonstrations.get('complete') is True and len(train)==1400
        and Counter(row['task'] for row in train)==dict.fromkeys(TASK_SPECS,100),
        'all 1400 original TRAIN source action-phase priors are required')
    require(prior.get('version')=='r10_original_TRAIN_action_phase_priors_v1'
        and prior.get('split')=='train' and prior.get('sources_read')==1400
        and prior.get('demonstration_manifest_sha256')==file_sha(dpath)
        and prior.get('VAL_expert_arrays_read') is False and prior.get('TEST_read') is False
        and prior.get('future_debug_target_XYZ_used') is False
        and prior.get('artifact_sha256')==digest({k:v for k,v in prior.items() if k!='artifact_sha256'}),
        'original TRAIN-only action-phase prior binding is missing or changed')
    original={row['source_group']:row for row in train}
    declared={row['source_group']:row for row in prior['sources']}
    require(len(original)==len(declared)==1400 and set(original)==set(declared),
        'original TRAIN prior source coverage differs')
    for group,row in original.items():
        item=declared[group]
        require(item['path']==row['npz_path'] and item['sha256']==row['npz_sha256']
            and file_sha(inside(root,row['npz_path']))==row['npz_sha256'],
            'original TRAIN extraction changed: '+group)
    return dict(demonstrations=pointer(dpath),original_action_phase_prior=pointer(ppath),
        original_TRAIN_source_count=1400,original_sources_by_task=dict(Counter(row['task'] for row in train)),
        original_parent_groups=sorted(original),used_as='separately fitted original action-phase statistical prior',
        body_relative_SE3_supervision_claimed=False,
        original_phase_statistics_modify_body_relative_curve_fit=False,
        original_archive_phase_timing_is_measured_native_time=False,
        generated_native_parent_coverage_must_be_reported_separately=True)


def colour_samples(root,cases,config):
    samples=[];excluded=[]
    for case in cases:
        names=[name for name in case['record']['files_sha256'] if Path(name).name=='TRAIN_color_annotations.json']
        require(len(names)==1,'each TRAIN receipt must bind its colour annotation sidecar')
        data=read(inside(root,names[0]));entry=case['entry'];group=entry['parent_source_group']
        require(data.get('version')=='r10_TRAIN_role_color_supervision_v1'
            and data.get('split')=='train' and data.get('task')==entry['task']
            and data.get('source_group')==group,'colour sidecar identity differs')
        for role,evidence in data['role_color_labels'].items():
            label=evidence.get('teacher_color_annotation')
            if label is None: continue
            require(label.get('source_split')=='train' and label.get('source_group')==group
                and label.get('label_is_runtime_policy_observation') is False
                and evidence.get('provenance')=='current_rgbd' and evidence.get('source_group')==group,
                'colour teacher label must stay separate from current TRAIN masked pixels')
            if not evidence.get('visible') or len(evidence['rgb_pixels_255'])<config.minimum_visible_pixels:
                excluded.append(dict(source_group=group,role=role,reason='insufficient_current_TRAIN_colour_pixels'))
                continue
            samples.append(TRAINColourSample(entry['task'],group,str(role),tuple(evidence['component_names']),
                str(label['color_name']),np.asarray(evidence['rgb_pixels_255'],float)))
    return samples,excluded


def colour_parent_crossfit(samples,config,*,minimum_accuracy):
    """Two source folds; a raw parent and every replica stay entirely together."""
    require(0<=minimum_accuracy<=1,'explicit TRAIN colour CV accuracy gate required')
    groups=sorted({sample.source_group for sample in samples});classes=sorted({sample.colour_name for sample in samples})
    support={name:sorted({sample.source_group for sample in samples if sample.colour_name==name}) for name in classes}
    # Rare public colour classes are balanced by whole parent, never by pixel
    # or replica. A valid split is still required; rarity is not fabricated.
    labels={group:{sample.colour_name for sample in samples if sample.source_group==group} for group in groups}
    order=sorted(groups,key=lambda group:(min(len(support[name]) for name in labels[group]),
        -len(labels[group]),group))
    counts=np.zeros((2,len(classes)),int);sizes=np.zeros(2,int);assignment={}
    target=np.asarray([len(support[name])/2 for name in classes])
    for group in order:
        additions=np.asarray([name in labels[group] for name in classes],int)
        def score(fold):
            after=counts.copy();after[fold]+=additions
            return (float(np.square(after-target[None]).sum()),int(sizes[fold]),fold)
        fold=min((0,1),key=score);assignment[group]=fold;counts[fold]+=additions;sizes[fold]+=1
    folds=[];held_results=[]
    for fold in (0,1):
        fit=[s for s in samples if assignment[s.source_group]!=fold]
        held=[s for s in samples if assignment[s.source_group]==fold]
        counts={name:len({s.source_group for s in fit if s.colour_name==name}) for name in classes}
        if len(classes)<2 or not held or any(counts[name]<config.minimum_sources_per_colour for name in classes):
            folds.append(dict(fold=fold,valid=False,reason='insufficient disjoint TRAIN parent class support',
                fit_parent_support_by_colour=counts));continue
        try:
            artifact=fit_colour_artifact(fit,config,synthetic=True)
        except PublicRoleResolutionError as error:
            folds.append(dict(fold=fold,valid=False,reason='TRAIN colour fit rejected: '+str(error),
                fit_parent_support_by_colour=counts));continue
        for sample in held:
            descriptor=np.median(_lab(sample.rgb_pixels_255),axis=0)
            distances=sorted((float(np.linalg.norm(descriptor-np.asarray(row['prototype_lab_normalised']))),name)
                for name,row in artifact['colours'].items())
            distance,predicted=distances[0];margin=distances[1][0]-distance
            supported=(distance<=artifact['colours'][predicted]['acceptance_radius'] and margin>=config.minimum_margin)
            held_results.append(dict(source_group=sample.source_group,task=sample.task_name,role=sample.role_id,
                true_public_colour=sample.colour_name,predicted_colour=predicted,supported=bool(supported),
                correct_and_supported=bool(supported and predicted==sample.colour_name)))
        folds.append(dict(fold=fold,valid=True,fit_parent_sources=sorted({s.source_group for s in fit}),
            held_parent_sources=sorted({s.source_group for s in held}),fit_parent_support_by_colour=counts))
    valid=bool(len(folds)==2 and all(row['valid'] for row in folds) and held_results)
    accuracy=(float(np.mean([np.mean([row['correct_and_supported'] for row in held_results if row['source_group']==group])
        for group in groups])) if valid else None)
    return dict(valid=valid,parent_to_fold=assignment,folds=folds,
        fold_assignment='deterministic rare-class-balanced whole TRAIN parent groups',
        public_colour_class_parent_support=support,
        held_predictions=held_results,macro_parent_correct_and_supported_fraction=accuracy,
        required_fraction=minimum_accuracy,passed=bool(valid and accuracy>=minimum_accuracy),
        split='train',VAL_used=False,TEST_used=False,unsupported_colours_are_not_frozen=True)


def fit_layout(cases,task,config):
    """Validate the same permanent relation on *every* declared TRAIN parent."""
    minimum_sources=int(config['minimum_parent_sources']);minimum_samples=int(config['minimum_samples_per_rollout'])
    pt,rt=float(config['position_tolerance_m']),float(config['orientation_tolerance_rad'])
    require(type(config['minimum_parent_sources']) is int and type(config['minimum_samples_per_rollout']) is int
        and minimum_sources>=2 and minimum_samples>=2 and np.isfinite([pt,rt]).all() and pt>0 and rt>0,
        'explicit finite valid TRAIN layout thresholds required')
    sources={case['entry']['parent_source_group'] for case in cases};spec=get_task_spec(task)
    ids={role.name for role in spec.roles};pairs=[('frame',child) for child in sorted(ids) if child.startswith(('peg','spoke'))]
    if 'button' in ids and 'wrap' in ids: pairs.append(('button','wrap'))
    accepted,rejected=[],[]
    for parent,child in pairs:
        rows=[];errors=[]
        for case in cases:
            a=case['arrays'];names=list(map(str,a['body_ids']))
            if parent not in names or child not in names:
                errors.append(dict(source=case['entry']['source_id'],reason='declared rigid body absent'));continue
            p,c=names.index(parent),names.index(child)
            valid=a['native_pose_valid'][:,p]&a['native_pose_valid'][:,c]&a['mode_valid'][:,p]&a['mode_valid'][:,c]
            if valid.sum()<minimum_samples:
                errors.append(dict(source=case['entry']['source_id'],reason='insufficient valid TRAIN native label samples'));continue
            pr,cr=a['target_rotations'][valid,p],a['target_rotations'][valid,c]
            rp=np.einsum('tji,tj->ti',pr,a['target_positions'][valid,c]-a['target_positions'][valid,p])
            rr=np.einsum('tji,tjk->tik',pr,cr)
            rows.append(dict(group=case['entry']['parent_source_group'],position=rp,rotation=rr))
        if len(sources)<minimum_sources: errors.append(dict(reason='insufficient distinct TRAIN parents'))
        if errors:
            rejected.append(dict(parent=parent,child=child,reasons=errors));continue
        groups=sorted(sources)
        centers=[np.median(np.concatenate([row['position'] for row in rows if row['group']==group]),axis=0) for group in groups]
        rotations=[Rotation.from_matrix(np.concatenate([row['rotation'] for row in rows if row['group']==group])).mean().as_matrix()
                   for group in groups]
        center=np.median(centers,axis=0);rotation=Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()
        pe=max(float(np.linalg.norm(row['position']-center,axis=1).max()) for row in rows)
        re=max(float(Rotation.from_matrix(row['rotation']@rotation.T).magnitude().max()) for row in rows)
        result=dict(parent=parent,child=child,source_groups=groups,samples=sum(len(row['position']) for row in rows),
            position_error_max_m=pe,orientation_error_max_rad=re)
        if pe>pt or re>rt:
            rejected.append(dict(result,reason='relative pose varies within/across actual TRAIN parents'));continue
        accepted.append(dict(result,child_to_parent_position=center.tolist(),child_to_parent_rotation=rotation.tolist()))
    payload=dict(version='r10_TRAIN_observed_rigid_layout_v1',task=task,split='train',
        source_group=sorted(sources)[0],source_groups=sorted(sources),source_group_field_is_legacy_not_exclusive=True,
        source='all declared TRAIN native SE3 label arrays; parent-balanced fit',thresholds=dict(config),
        accepted=accepted,rejected=rejected,shared_with_baseline=True,changes_current_episode_history=False,
        input_cache_sha256={case['entry']['source_id']:case['entry']['cache_sha256'] for case in cases},
        development_artifact_pending_formal_release=True)
    payload['evidence_sha256']=digest(payload)
    return payload


def fit_artifacts(root,collector_manifest,demonstrations,original_prior,configuration,out):
    """No default fit thresholds; partial parent coverage is explicitly pending."""
    from .evaluation import atomic_json
    root=Path(root).resolve();out=inside(root,out)
    require(not out.exists() or not any(out.iterdir()),'do not overwrite previous fitting evidence')
    out.mkdir(parents=True,exist_ok=True)
    configuration=inside(root,configuration);config=read(configuration)
    require(config.get('fit_split')=='train' and config.get('VAL_used') is False and config.get('TEST_used') is False,
        'fitting configuration must be explicit TRAIN-only')
    original=original_prior_binding(root,demonstrations,original_prior)
    manifest,cases=load_train_cases(root,collector_manifest)
    reference_time_rule=None
    if 'reference_time_derivation_rule' in config:
        from .reference_time import load_rule
        reference_time_rule=load_rule(root,config['reference_time_derivation_rule'])
    semantic_acquire_rule=None;semantic_sources={};semantic_errors={};semantic_proofs={}
    if 'semantic_acquire_derivation_rule' in config:
        from .semantic_acquire import load_rule,SCHEMA
        semantic_acquire_rule=load_rule(root,config['semantic_acquire_derivation_rule'])
    command_goal_rule=None;goal_sources={};goal_errors={};goal_proofs={}
    if 'command_goal_reference_rule' in config:
        from .command_goal_reference import load_rule,SCHEMA as GOAL_SCHEMA
        command_goal_rule=load_rule(root,config['command_goal_reference_rule'],purpose='production_fitting')
        for case in cases:
            entry=case['entry']
            if (entry['task'] not in command_goal_rule.tasks or entry['source_kind']!='TRAIN_native_generated_augmentation'
                    or entry.get('interaction_eligible',True) is not True):continue
            sid=entry['source_id']
            try:
                source,audit=interaction_source(case,reference_time_rule=reference_time_rule,
                    semantic_acquire_rule=semantic_acquire_rule,command_goal_rule=command_goal_rule)
                goal_sources[sid]=(source,audit)
                goal_proofs[sid]=audit['command_goal_reference_derivation']['proof']
            except (ValueError,InteractionArtifactError) as error:
                goal_errors[sid]=str(error)
                goal_proofs[sid]=dict(status='original_or_command_goal_adapter_validation_failed',error=str(error),raw_source_preserved=True)
    if semantic_acquire_rule is not None:
        # Validate original source/events first and freeze every derived source
        # proof before fitting.  Never rewrite the fit manifest after its SHA
        # has been consumed by a library.
        for case in cases:
            entry=case['entry']
            if (entry['task'] not in semantic_acquire_rule.tasks
                    or entry['source_kind']!='TRAIN_native_generated_augmentation'
                    or entry.get('interaction_eligible',True) is not True):continue
            sid=entry['source_id']
            try:
                if sid in goal_errors:raise InteractionArtifactError(goal_errors[sid])
                if sid in goal_sources:source,audit=goal_sources[sid]
                else:source,audit=interaction_source(case,reference_time_rule=reference_time_rule,
                    semantic_acquire_rule=semantic_acquire_rule)
                semantic_sources[sid]=(source,audit)
                evidence=audit['semantic_acquire_derivation']
                semantic_proofs[sid]={key:evidence[key] for key in
                    ('derivation_sha256','raw_bindings','status','segment_ids')}
            except (ValueError,InteractionArtifactError) as error:
                semantic_errors[sid]=str(error)
                semantic_proofs[sid]=dict(status='original_or_semantic_adapter_validation_failed',error=str(error),
                    raw_source_preserved=True)
    declared_parent=set(original['original_parent_groups'])
    require(all(case['entry']['parent_source_group'] in declared_parent for case in cases),
        'native TRAIN parents must be real original TRAIN source identities')
    fit_manifest=dict(version='r10_TRAIN_interaction_label_fit_manifest_v1',entries=[case['entry'] for case in cases],
        original_action_phase_prior_binding=original,collector_manifest=pointer(inside(root,collector_manifest)),
        fitting_configuration=pointer(configuration),interaction_adapter_source=pointer(Path(__file__)),
        fit_split='train',VAL_used=False,TEST_used=False)
    if reference_time_rule is not None:
        fit_manifest['reference_time_derivation']=dict(rule=dict(path=reference_time_rule.path,sha256=reference_time_rule.sha256),
            evidence_report=reference_time_rule.evidence_report,
            verified_files_sha256=reference_time_rule.verified_files_sha256,
            original_raw_terminal_reference_and_masks_preserved=True)
    if semantic_acquire_rule is not None:
        fit_manifest['semantic_acquire_derivation']=dict(
            rule=dict(path=semantic_acquire_rule.path,sha256=semantic_acquire_rule.sha256),
            helper=semantic_acquire_rule.helper,derivation_schema=SCHEMA,
            raw_masks_and_events_preserved=True,per_source=semantic_proofs,
            verified_files_sha256=semantic_acquire_rule.verified_files_sha256)
    if command_goal_rule is not None:
        fit_manifest['command_goal_reference_derivation']=dict(rule=command_goal_rule.pointer,
            helper=pointer(Path(__file__).parent/'command_goal_reference.py'),derivation_schema=GOAL_SCHEMA,
            original_raw_metric_events_and_masks_preserved=True,per_source=goal_proofs,
            verified_files_sha256=command_goal_rule.verified_files_sha256)
    if manifest.get('version')==COMPOSITE_VERSION:
        from . import acquisition_manifest,acquisition_lineage
        fit_manifest['historical_acquisition_lineage']=dict(
            actual_plan_sha256_values=manifest['actual_plan_sha256_values'],
            base_plan_sha256=manifest['base_plan_sha256'],single_plan_SHA_claimed=False,
            raw_acquisition_receipts_rewritten=False,
            audit_sources=[pointer(acquisition_manifest.__file__),pointer(acquisition_lineage.__file__)])
    elif manifest.get('version')==BUNDLE_VERSION:
        fit_manifest['fixed_teacher_supplement_bundle']=dict(primary_manifest=manifest['primary_manifest'],
            auxiliary_teacher_manifest=manifest['auxiliary_teacher_manifest'],primary_complete2800=True,
            auxiliary_complete200=True,TRAIN_parent_sources=1400,teacher_rollouts=1600,
            original_Context_K1_hold_rollouts=1400,primary_Context_view_unchanged=True,
            actual_plan_sha256_values=manifest['actual_plan_sha256_values'],single_plan_SHA_claimed=False,
            original_raw_receipts_and_masks_rewritten=False)
    elif manifest.get('version')==COUNTED_BUNDLE_VERSION:
        fit_manifest['counted_high_cup_teacher_bundle']=dict(
            primary3000_bundle=manifest['primary3000_bundle'],
            counted_high_cup_manifest=manifest['counted_high_cup_manifest'],
            primary_complete3000=True,counted_high_cup_complete100=True,
            TRAIN_parent_sources=1400,teacher_rollouts=1700,original_Context_K1_hold_rollouts=1400,
            primary_Context_view_unchanged=True,parent_replicas_kept_in_same_fold=True,
            actual_plan_sha256_values=manifest['actual_plan_sha256_values'],single_plan_SHA_claimed=False,
            original_raw_receipts_and_masks_rewritten=False,
            new_teacher_policy_is_not_original_author_reproduction=True)
    manifest_path=out/'TRAIN_fit_manifest.json';atomic_json(manifest_path,fit_manifest)
    ic=InteractionFitConfig(**config['interaction']);cc=ColourFitConfig(**config['colour'])
    per_task={};artifacts={}
    for task in TASK_SPECS:
        task_cases=[case for case in cases if case['entry']['task']==task]
        selected=[case for case in task_cases if case['entry']['source_kind']=='TRAIN_native_generated_augmentation'
            and case['entry'].get('interaction_eligible',True) is True]
        parents=sorted({case['entry']['parent_source_group'] for case in task_cases})
        teacher_parents=sorted({case['entry']['parent_source_group'] for case in selected})
        row=dict(native_TRAIN_rollouts=len(task_cases),native_TRAIN_parent_sources=len(parents),parent_sources=parents,
            teacher_TRAIN_rollouts=len(selected),teacher_TRAIN_parent_sources=len(teacher_parents),
            hold_acquisitions_are_not_interaction_BC=True,
            expected_original_TRAIN_parent_sources=100,all_original_TRAIN_parents_acquired=len(parents)==100)
        per_task[task]=row
        if not selected:
            row['interaction_status']='pending_missing_native_TRAIN';continue
        sources=[];diagnostics=[]
        try:
            for case in selected:
                sid=case['entry']['source_id']
                if sid in semantic_errors:
                    raise InteractionArtifactError(semantic_errors[sid])
                if sid in goal_errors:
                    raise InteractionArtifactError(goal_errors[sid])
                if sid in goal_sources:source,audit=goal_sources[sid]
                elif sid in semantic_sources:source,audit=semantic_sources[sid]
                else:source,audit=interaction_source(case,reference_time_rule=reference_time_rule,
                    semantic_acquire_rule=semantic_acquire_rule,command_goal_rule=command_goal_rule)
                sources.append(source)
                diagnostics.append(dict(source_id=source.source_id,**audit))
            library=fit_interaction_library(sources,ic,TRAIN_source_manifest_path=manifest_path)
            library.payload['metadata'].update(original_action_phase_prior_binding=original,
                development_artifact_pending_formal_release=True,
                native_TRAIN_parent_source_count=len(parents),all_100_native_TRAIN_parents_acquired=len(parents)==100,
                native_TRAIN_acquisition_coverage_is_not_BC_coverage=True)
            # Reconstruct to bind updated metadata into the semantic artifact SHA.
            from .interaction import InteractionLibrary
            library=InteractionLibrary(library.payload)
            path=out/task/'interaction.json';library.save(path)
            row.update(interaction_status='TRAIN_fitted_pending_formal_release',interaction=pointer(path))
            row['completed_skill_fitting']=library.metadata['completed_skill_fitting']
            row['BC_used_source_ids']=library.metadata['BC_used_source_ids']
            row['BC_excluded_source_ids']=library.metadata['BC_excluded_source_ids']
            artifacts.setdefault(task,{})['interaction']=pointer(path)
        except (ValueError,InteractionArtifactError) as error:
            row.update(interaction_status='pending_insufficient_or_invalid_TRAIN_supervision',interaction_error=str(error))
            if hasattr(error,'diagnostics'):row['completed_skill_fitting']=error.diagnostics
        row['interaction_adapter_diagnostics']=diagnostics
        layout=fit_layout(task_cases,task,config['rigid_layout']);path=out/task/'rigid_layout.json'
        atomic_json(path,layout);row['rigid_layout']=pointer(path);artifacts.setdefault(task,{})['rigid_layout']=pointer(path)
    samples,excluded=colour_samples(root,cases,cc)
    colour_cv=colour_parent_crossfit(samples,cc,minimum_accuracy=config['colour_parent_cv_minimum_accuracy'])
    if colour_cv['passed']:
        colour=fit_colour_artifact(samples,cc,TRAIN_manifest_path=manifest_path)
        colour['metadata'].update(development_artifact_pending_formal_release=True,
            TRAIN_parent_crossfit=colour_cv,original_action_phase_prior_binding=original)
        # public_roles uses its deliberately different JSON canonical form.
        from .public_roles import _sha
        colour['artifact_sha256']=_sha({k:v for k,v in colour.items() if k!='artifact_sha256'})
        path=out/'colour.json';atomic_json(path,colour)
        for task in TASK_SPECS:
            family=get_task_spec(task).family
            roles=(('peg0','peg1','peg2') if family=='insert_onto_rotating_peg' else
                   ('button',) if family=='push_moving_button' else ())
            if roles:
                if all(task+'|'+role in colour['role_components'] for role in roles):
                    artifacts.setdefault(task,{})['colour']=pointer(path)
                    per_task[task]['colour_status']='TRAIN_parent_CV_supported_pending_formal_release'
                else:
                    per_task[task]['colour_status']='pending_missing_task_role_component_support'
    report=dict(version='r10_TRAIN_interaction_colour_layout_fit_report_v1',fit_split='train',
        artifact_fitting_execution_complete=True,
        formal_comparison_ready=False,formal_freeze_authorized=False,status='development_TRAIN_fits_pending_full_release',
        original_action_phase_prior_binding=original,collector_manifest=pointer(inside(root,collector_manifest)),
        fitting_configuration=pointer(configuration),fitting_source=pointer(Path(__file__)),
        native_TRAIN_parent_sources=len({case['entry']['parent_source_group'] for case in cases}),
        native_TRAIN_rollouts=len(cases),expected_native_TRAIN_parent_sources=1400,
        original_action_phase_priors_are_body_SE3_supervision=False,per_task=per_task,
        colour_parent_crossfit=colour_cv,excluded_colour_samples=excluded,artifacts=artifacts,
        VAL_expert_or_outcome_used=False,TEST_used=False,dynamics_or_visibility_masks_modified=False,
        success_rate_benefit_established=False)
    atomic_json(out/'report.json',report)
    return report
