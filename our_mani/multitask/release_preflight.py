"""Read-only full-source model coverage before a formal R10 release.

Development loaders intentionally accept small honest pilots. This independent
gate compares real parent identities against the canonical TRAIN/VAL inventory,
rechecks acquisition/fit evidence, and never launches native simulation, model
fitting, CUDA, or TEST data. Passing establishes coverage/provenance, not skill
quality or a success-rate benefit.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from .evaluation import digest,file_sha,validate_completed_case,comparison_protocol_sha256
from .specs import TASK_SPECS

SCHEMA='r10_formal_model_coverage_preflight_v1'


def require(value,message):
    if not value:raise ValueError(message)


def normalized_lock(root,files):
    root=Path(root).resolve();result={}
    for name,checksum in files.items():
        path=(root/name).resolve();key=str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        require(key not in result or result[key]==checksum,'conflicting digests in formal input lock')
        result[key]=checksum
    return result


class _Inputs:
    def __init__(self,root):
        self.root=Path(root).resolve();self.files={};self._path_stats={}

    @staticmethod
    def _stat_signature(path):
        stat=path.stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def file(self,path,expected=None,*,force=False):
        path=(self.root/path).resolve()
        require(path.is_relative_to(self.root) and path.is_file(),'coverage input missing or outside project: '+str(path))
        name=str(path.relative_to(self.root));before=self._stat_signature(path)
        previous=self.files.get(name)
        require(expected is None or previous is None or previous==expected,'coverage input SHA differs: '+str(path))
        if not force and previous is not None and self._path_stats.get(name)==before:
            return path
        actual=file_sha(path)
        after=self._stat_signature(path)
        require(before==after,'coverage input changed while hashing: '+name)
        require(expected is None or actual==expected,'coverage input SHA differs: '+str(path))
        require(name not in self.files or self.files[name]==actual,'coverage input changed while auditing: '+name)
        self.files[name]=actual;self._path_stats[name]=after
        return path

    def pointer(self,pointer):
        require(isinstance(pointer,dict) and {'path','sha256'}<=set(pointer)
            and isinstance(pointer['sha256'],str) and len(pointer['sha256'])==64
            and all(c in '0123456789abcdef' for c in pointer['sha256']),'explicit coverage input path/SHA required')
        return self.file(pointer['path'],pointer['sha256'])

    def read(self,path,expected=None):
        return json.loads(self.file(path,expected).read_text())

    def read_pointer(self,pointer):
        return json.loads(self.pointer(pointer).read_text())

    def include(self,files):
        for name,checksum in files.items():self.file(name,checksum)

    def source_snapshot(self,pointer):
        """Bind explicitly declared fitting code to its immutable archive."""
        snapshot=self.read_pointer(pointer);sources=snapshot.get('sources')
        require(isinstance(sources,dict) and sources,'explicit fitting source snapshot is empty')
        for name,row in sources.items():
            require(isinstance(row,dict) and set(row)=={'live_source','archived_source'},
                'fitting source snapshot needs exact live/archive source pairs')
            live=self.pointer(row['live_source']);self.pointer(row['archived_source'])
            require(live==(self.root/name).resolve() and row['live_source']['sha256']==row['archived_source']['sha256'],
                'fitting source snapshot live/archive owner differs')
        return snapshot

    def rehash_all(self):
        """Directly rehash every input once at this consumer's final boundary."""
        for name,checksum in tuple(self.files.items()):self.file(name,checksum,force=True)


def _parents(entries,split,task=None):
    result=set();rollouts=set()
    for row in entries:
        if task is not None and row.get('task')!=task:continue
        require(row.get('split')==split and row.get('task') in TASK_SPECS,
            'model evidence contains foreign split/task')
        raw=row.get('raw_episode_id');parent=row.get('parent_raw_episode_id') or raw
        require(isinstance(raw,str) and raw and isinstance(parent,str) and parent,'explicit real parent identities required')
        key=(row['task'],raw)
        require(key not in rollouts,'duplicate rollout identity in model evidence');rollouts.add(key)
        group=f"{split}:{row['task']}:{parent}"
        require(row.get('parent_source_group',group)==group,'model parent source group differs')
        result.add((row['task'],parent))
    return result


def _canonical(inputs,path,split):
    data=inputs.read(path)
    require(data.get('split')==split,'canonical inventory split differs')
    entries=[row for row in data.get('episodes',[]) if row.get('task') in TASK_SPECS]
    require(entries,'canonical original source inventory is empty')
    keys={(row['task'],row['raw_episode_id']) for row in entries}
    require(len(keys)==len(entries),'duplicate canonical original parent')
    counts=Counter(task for task,raw in keys)
    require(set(counts)==set(TASK_SPECS),'canonical inventory lacks a remaining task')
    if split=='train':require(len(keys)==1400 and set(counts.values())=={100},'formal TRAIN requires 100 real parents in each of 14 tasks')
    else:require(len(keys)==275 and all(18<=n<=20 for n in counts.values()),'formal VAL requires the real 275 parents, including archive shortfalls')
    return keys,dict(path=str(inputs.file(path).relative_to(inputs.root)),sha256=file_sha(inputs.file(path)),
        parents=len(keys),per_task=dict(sorted(counts.items())))


def _exact(actual,expected,label):
    require(actual==expected,label+f' parent coverage differs: missing={len(expected-actual)}, unexpected={len(actual-expected)}')


def _context_view(inputs,pointer,expected):
    from .training_view import load_acquisitions,require_production_behavior_budget
    from .motion_labels import SCHEMA as MOTION_SCHEMA
    path=inputs.pointer(pointer);view=json.loads(path.read_text())
    require(view.get('version')=='r10_causal_body_tracks_v1' and view.get('complete') is True
        and view.get('training_view_only') is True and view.get('TEST_used') is False
        and view.get('development_only') is False and view.get('parent_source_coverage_passed') is True
        and view.get('full_1p1s_each_task_passed') is True
        and view.get('primary_hold_motion_context_view') is True,
        'formal Context needs a complete nondevelopment primary-hold training view')
    require(view.get('dynamics_target_label_schema')==MOTION_SCHEMA,
        'formal Context must use derived GT-pose twist labels, not the old raw velocity target')
    lineage=view.get('acquisition_lineage',[])
    require(lineage and {row.get('split') for row in lineage}=={'train','validation'},'TRAIN/VAL acquisition lineage required')
    require_production_behavior_budget(lineage)
    for row in lineage:inputs.file(row['path'],row['sha256'])
    all_entries=[];all_ledger=[];raw_files={};coverage={}
    for split in ('train','validation'):
        rows=[row for row in lineage if row['split']==split]
        require(all(row.get('fixed_augmentation_budget',{}).get('version')=='r10_fixed_behavior_budget_v1' for row in rows),
            'formal context coverage cannot use legacy teacher replacement')
        selected,ledger,verified=load_acquisitions(inputs.root,[row['path'] for row in rows],split,raw_files)
        require(verified==rows,'context acquisition lineage/replica budget differs')
        _exact(_parents(selected,split),expected[split],split+' Context usable')
        acquired={(row['task'],row['parent_raw_episode_id']) for row in ledger}
        _exact(acquired,expected[split],split+' Context acquired')
        full={row['task'] for row in ledger if row['included_in_training_view']
            and row['window_statistics']['full_1p1s_examples']>0}
        require(full==set(TASK_SPECS),'each context task/split needs real full 1.1s supervision')
        coverage[split]=dict(parent_sources=len(expected[split]),per_task=dict(Counter(task for task,raw in _parents(selected,split))),
            usable_windows=sum(row['window_statistics']['usable_causal_examples'] for row in ledger if row['included_in_training_view']),
            acquired_rollouts=len(ledger),primary_hold_rollouts=len(selected),full_horizon_tasks=sorted(full))
        all_entries.extend(selected);all_ledger.extend(ledger)
    key=lambda row:(row['split'],row['task'],row['raw_episode_id'])
    require(sorted(view['entries'],key=key)==sorted(all_entries,key=key),'training view differs from actual fixed primary acquisitions')
    stored=inputs.read_pointer(view['acquisition_ledger'])
    require(stored.get('entries')==all_ledger and stored.get('declared_attempts')==len(all_ledger)
        and stored.get('native_success_or_failure_filtering') is False,'training view ledger lost/filtered acquisitions')
    inputs.include(raw_files)
    return path,view,coverage


def _context(inputs,protocol,view_pointer,expected):
    from .context_se3 import MultiBodyContextDynamics
    from .motion_labels import SCHEMA as MOTION_SCHEMA
    require(protocol.get('dynamics_target_label_schema')==MOTION_SCHEMA,'formal protocol must bind corrected dynamics labels')
    path=inputs.pointer(protocol['context_checkpoint']);model=MultiBodyContextDynamics(path);m=model.metadata
    require(m.get('dynamics_target_label_schema')==MOTION_SCHEMA,
        'Context checkpoint metadata must bind corrected derived-twist supervision')
    require(m.get('selection_split')=='validation' and m.get('VAL_source_episodes')==275
        and m.get('maximum_training_horizon_seconds',0)>=1.1,'Context VAL coverage/horizon is a pilot or incomplete')
    view_path,view,coverage=_context_view(inputs,view_pointer,expected)
    require(m.get('track_manifest_sha256')==file_sha(view_path),'checkpoint is not fitted on the verified full view')
    source=m['training_source_manifest'];source_path=inputs.file(path.parent/source['path_relative_to_checkpoint'],source['sha256'])
    train=json.loads(source_path.read_text())
    _exact(_parents(train.get('entries',[]),'train'),expected['train'],'Context fitted TRAIN')
    require(source.get('source_episode_count')==train.get('source_episode_count')==1400
        and source.get('cache_episode_count')==len(train['entries'])
        and train.get('body_track_manifest_sha256')==file_sha(view_path)
        and train.get('normalization_fit_split')=='train','Context fitted-source proof differs')
    key=lambda row:(row['task'],row['raw_episode_id'])
    require(sorted(train['entries'],key=key)==sorted([row for row in view['entries'] if row['split']=='train'],key=key),
        'Context normalization/fit sidecar differs from actual full TRAIN view')
    report=inputs.read(path.parent/'report.json')
    require(report.get('checkpoint_sha256')==file_sha(path) and report.get('metadata')==m,'Context training report is not bound to this checkpoint')
    optimizer=report.get('optimizer_coverage',{})
    expected_groups={f'train:{task}:{raw}' for task,raw in expected['train']}
    require(set(optimizer)==expected_groups and all(type(n) is int and n>0 for n in optimizer.values()),
        'every real TRAIN parent must actually reach an optimizer update')
    require(m.get('optimizer_examples_per_parent_min')==min(optimizer.values())
        and m.get('optimizer_examples_per_parent_max')==max(optimizer.values()),'Context optimizer counter proof differs')
    require(m.get('TRAIN_windows')==coverage['train']['usable_windows']
        and m.get('VAL_windows')==coverage['validation']['usable_windows'],'Context actual usable-window counts differ')
    first=report.get('validation',{}).get('metrics',{}).get('context',{}).get('0.05',{})
    require(first.get('source_episodes')==275 and {task:row.get('source_episodes') for task,row in first.get('by_task',{}).items()}
        ==coverage['validation']['per_task'],'Context evaluated VAL parent/task coverage differs')
    for field,name in (('architecture_sha256','our_mani/multitask/context_se3.py'),
            ('fitting_sha256','our_mani/multitask/fitting.py'),('track_dataset_sha256','our_mani/multitask/track_dataset.py'),
            ('motion_labels_sha256','our_mani/multitask/motion_labels.py'),
            ('training_script_sha256','scripts/train_multitask_context_r10.py')):
        require(isinstance(m.get(field),str) and len(m[field])==64,'Context training source hash is missing: '+field)
        inputs.file(name,m[field])
    return dict(passed=True,checkpoint=str(path.relative_to(inputs.root)),TRAIN_parent_sources=1400,VAL_parent_sources=275,
        per_split=coverage,optimizer_parent_sources=len(optimizer),development_pilot_rejected=True)


def _completion_support(sources,template):
    from .interaction import _completion_reference,InteractionArtifactError,TRAINSemanticAcquireEvidence
    from .command_goal_reference import TRAINCommandGoalEvidence,INTENT_BASIS
    key=(template.phase,template.anchor_id,template.reference_role,template.reference_kind,
        template.manipulated_role,template.reference_mode,template.reference_time_basis)
    parents=set();attempts=set()
    for source in sources:
        semantic=source.semantic_acquire_evidence
        if template.phase=='acquire' and semantic is not None:
            require(isinstance(semantic,TRAINSemanticAcquireEvidence)
                and (semantic.task_name,semantic.source_id,semantic.parent_source_group)==
                    (source.task_name,source.source_id,source.grouping_source),
                'semantic completion evidence belongs to another task/source/parent')
            # An enabled empty view retains censored/invalid acquisitions. It
            # must never regain the legacy phase-transition positives.
            for segment in semantic.segments:
                b=segment.binding;event_key=(segment.phase,b.anchor_id,b.reference_role,b.reference_kind,
                    b.manipulated_role,segment.reference_mode,b.reference_time_basis)
                if event_key!=key:continue
                indices=segment.frame_indices;end=segment.terminal_frame
                require(indices and 0<=indices[0]<=indices[-1]<end<len(source.times)
                    and indices==tuple(range(indices[0],indices[-1]+1))
                    and abs(source.times[indices[0]]-segment.attempt_start_time)<1e-8
                    and abs(source.times[end]-segment.attempt_end_time)<1e-8
                    and tuple(dict.fromkeys(map(int,source.command_ids[list(indices)])))==segment.raw_command_ids
                    and all(source.references[i]==b for i in indices),
                    'semantic support differs from actual acquired frames/commands/reference')
                try:_completion_reference(source,segment,template.lag_seconds)
                except (InteractionArtifactError,KeyError):continue
                for i in indices:
                    try:
                        source.body_tracks[b.reference_id].at(source.times[i]+template.lag_seconds,
                            orientation_required=False,same_mode_as=segment.reference_mode,not_before=source.times[i])
                        parents.add(source.grouping_source)
                        attempts.add(source.source_id+':semantic:'+segment.segment_id)
                        break
                    except (InteractionArtifactError,KeyError):continue
            continue
        for event in source.completion_events:
            b=event.binding;goal=None;evidence=source.command_goal_reference_evidence
            if evidence is not None:
                require(isinstance(evidence,TRAINCommandGoalEvidence)
                    and (evidence.task_name,evidence.source_id,evidence.parent_source_group)==
                        (source.task_name,source.source_id,source.grouping_source),
                    'command-goal completion evidence belongs to another source/parent')
                if any(row['command_id']==event.origin_command_id for row in evidence.pending_commands):continue
                goal=next((row for row in evidence.references if row.command_id==event.origin_command_id),None)
                if goal is not None:
                    require(goal.binding==b and goal.reference_mode==event.reference_mode,
                        'command-goal support binding/mode differs')
            event_key=(event.phase,b.anchor_id,b.reference_role,b.reference_kind,
                b.manipulated_role,event.reference_mode,INTENT_BASIS if goal else b.reference_time_basis)
            if event_key!=key:continue
            if goal is not None:
                require(template.reference_time_rule_sha256==evidence.rule_sha256 and template.lag_seconds==0.,
                    'command-goal support rule/lag differs')
            try:
                if goal is None:_completion_reference(source,event,template.lag_seconds)
            except (InteractionArtifactError,KeyError):continue
            for i in np.flatnonzero(source.command_ids==event.origin_command_id):
                if (source.references[i]!=b or not event.attempt_start_time-1e-8<=source.times[i]<event.attempt_end_time-1e-8):continue
                try:
                    if b.reference_kind not in ('world','phase_origin'):
                        source.body_tracks[b.reference_id].at(source.times[i]+template.lag_seconds,
                            orientation_required=b.reference_kind=='observed_se3',same_mode_as=event.reference_mode,not_before=source.times[i])
                    parents.add(source.grouping_source);attempts.add(source.source_id+':command'+str(event.origin_command_id));break
                except (InteractionArtifactError,KeyError):continue
    return key,parents,attempts


def _semantic_source_proof(source):
    evidence=source.semantic_acquire_evidence
    require(evidence is not None,'semantic TRAIN source omitted its explicit evidence view')
    return dict(derivation_sha256=evidence.derivation_sha256,raw_bindings=evidence.raw_bindings,
        status=evidence.status,segment_ids=[segment.segment_id for segment in evidence.segments])


def _semantic_sources(inputs,configuration,manifest,cases,reference_time_rule,cache):
    """Rebuild every declared semantic source from actual registered raw data."""
    pointer=configuration.get('semantic_acquire_derivation_rule')
    if pointer is None:
        require(manifest.get('semantic_acquire_derivation') is None,
            'library has an undeclared semantic acquire derivation')
        return None,{}
    from .semantic_acquire import load_rule,SCHEMA
    from .artifact_fitting import interaction_source
    key=('semantic_acquire',digest(pointer),digest(manifest))
    if key in cache:return cache[key]
    rule=load_rule(inputs.root,pointer)
    inputs.pointer(pointer);inputs.pointer(rule.helper);inputs.include(rule.verified_files_sha256)
    sources={}
    for case in cases:
        entry=case['entry']
        if (entry['task'] not in rule.tasks or entry['source_kind']!='TRAIN_native_generated_augmentation'
                or entry.get('interaction_eligible',True) is not True):continue
        source,audit=interaction_source(case,reference_time_rule=reference_time_rule,semantic_acquire_rule=rule)
        require(source.source_id not in sources,'duplicate semantic TRAIN acquisition')
        evidence=source.semantic_acquire_evidence
        require(evidence.derivation_module_sha256==rule.helper['sha256']
            and evidence.derivation_rule_sha256==rule.sha256
            and evidence.derivation_rule_pointer==dict(path=rule.path,sha256=rule.sha256),
            'semantic source derivation differs from exact reviewed helper/rule')
        inputs.include(audit['semantic_acquire_derivation']['verified_files_sha256'])
        sources[source.source_id]=source
    expected=dict(rule=dict(path=rule.path,sha256=rule.sha256),helper=rule.helper,
        derivation_schema=SCHEMA,raw_masks_and_events_preserved=True,
        per_source={sid:_semantic_source_proof(source) for sid,source in sources.items()},
        verified_files_sha256=rule.verified_files_sha256)
    require(manifest.get('semantic_acquire_derivation')==expected,
        'library semantic derivation differs from recomputed complete TRAIN raw evidence')
    cache[key]=(rule,sources)
    return rule,sources


def _semantic_library_metadata(metadata,sources,rule):
    enabled=[source for source in sources if source.semantic_acquire_evidence is not None]
    if not enabled:
        require(metadata.get('semantic_acquire_derivation') is None,
            'library declares semantic acquire without verified task sources')
        return
    require(rule is not None and len(enabled)==len(sources),
        'library mixes semantic and legacy acquire supervision')
    expected=dict(default_legacy_behavior_changed=False,
        original_events_command_ids_masks_and_censoring_preserved=True,
        rule_sha256_values=[rule.sha256],helper_sha256_values=[rule.helper['sha256']],
        per_source={source.source_id:_semantic_source_proof(source) for source in sources})
    require(metadata.get('semantic_acquire_derivation')==expected,
        'library semantic source metadata differs from recomputed raw TRAIN evidence')


def _command_goal_sources(inputs,configuration,manifest,cases,reference_time_rule,cache,semantic_rule=None):
    pointer=configuration.get('command_goal_reference_rule')
    if pointer is None:
        require(manifest.get('command_goal_reference_derivation') is None,
            'library has an undeclared command-goal derivation')
        return None,{}
    from .command_goal_reference import load_rule,SCHEMA,evidence_proof
    from .artifact_fitting import interaction_source
    key=('command_goal',digest(pointer),digest(manifest))
    if key in cache:return cache[key]
    rule=load_rule(inputs.root,pointer,purpose='production_fitting')
    inputs.pointer(pointer);inputs.include(rule.verified_files_sha256)
    helper=dict(path=str(inputs.root/'our_mani/multitask/command_goal_reference.py'),
        sha256=file_sha(inputs.root/'our_mani/multitask/command_goal_reference.py'))
    inputs.pointer(helper);sources={}
    for case in cases:
        entry=case['entry']
        if (entry['task'] not in rule.tasks or entry['source_kind']!='TRAIN_native_generated_augmentation'
                or entry.get('interaction_eligible',True) is not True):continue
        source,audit=interaction_source(case,reference_time_rule=reference_time_rule,
            semantic_acquire_rule=semantic_rule,command_goal_rule=rule)
        require(source.source_id not in sources,'duplicate command-goal TRAIN acquisition')
        inputs.include(audit['command_goal_reference_derivation']['verified_files_sha256'])
        sources[source.source_id]=source
    expected=dict(rule=rule.pointer,helper=helper,derivation_schema=SCHEMA,
        original_raw_metric_events_and_masks_preserved=True,
        per_source={sid:evidence_proof(source.command_goal_reference_evidence) for sid,source in sources.items()},
        verified_files_sha256=rule.verified_files_sha256)
    require(manifest.get('command_goal_reference_derivation')==expected,
        'library command-goal derivation differs from recomputed complete TRAIN raw evidence')
    cache[key]=(rule,sources)
    return rule,sources


def _command_goal_library_metadata(metadata,sources,rule):
    from .command_goal_reference import SCHEMA,evidence_proof
    enabled=[source for source in sources if source.command_goal_reference_evidence is not None]
    if not enabled:
        require(metadata.get('command_goal_reference_derivation') is None,
            'library declares command-goal without verified task sources')
        return
    require(rule is not None and len(enabled)==len(sources),'library mixes command-goal and legacy evidence')
    expected=dict(schema=SCHEMA,rule_sha256_values=[rule.sha256],
        helper_sha256=file_sha(Path(__file__).parent/'command_goal_reference.py'),
        original_raw_completion_metric_preserved=True,
        per_source={source.source_id:evidence_proof(source.command_goal_reference_evidence) for source in sources})
    require(metadata.get('command_goal_reference_derivation')==expected
        and metadata.get('public_completion_verifier_calibrated') is False,
        'library command-goal metadata differs from recomputed raw evidence/scope')


def _command_goal_geometry(sources,template):
    from .reference_time import COMMAND_GOAL
    if template.reference_time_basis!=COMMAND_GOAL:return
    from .interaction import _fit_completed_curve
    from our_mani.contracts import Pose
    from scipy.spatial.transform import Rotation
    key,parents,attempts=_completion_support(sources,template);rows=[]
    for source in sources:
        evidence=source.command_goal_reference_evidence
        for event in source.completion_events:
            if source.source_id+':command'+str(event.origin_command_id) not in attempts:continue
            goal=next(row for row in evidence.references if row.command_id==event.origin_command_id)
            reference=Pose.from_xyzw(goal.intent_reference_xyzw);world=Pose.from_xyzw(event.committed_world_waypoint_xyzw)
            rows.append(dict(source=source.grouping_source,progress=1.,terminal_event=True,
                position=reference.rotation.T@(world.position-reference.position),
                rotation=reference.rotation.T@world.rotation,opening=0.))
    require(rows,'command-goal canonical geometry has no actual admitted original commands')
    p,r,_=_fit_completed_curve(rows,template.progress);actual=Pose.from_xyzw(template.completion_geometry_xyzw)
    require(np.linalg.norm(actual.position-p[-1])<1e-10
        and Rotation.from_matrix(actual.rotation@r[-1].T).magnitude()<1e-10,
        'command-goal public canonical geometry differs from original committed TRAIN waypoints')


def _library(inputs,protocol,task,expected,cache):
    from .artifact_fitting import load_train_cases,interaction_source,original_prior_binding
    from .interaction import InteractionLibrary
    path=inputs.pointer(protocol['learned_artifacts'][task]['interaction']);library=InteractionLibrary.load(path);m=library.metadata
    require(library.task_spec.task_name==task and m.get('synthetic') is False,'formal library task/synthetic provenance differs')
    fit=m.get('fit_configuration',{})
    require(m.get('development_support_override') is False and fit.get('development_only') is False
        and fit.get('minimum_completed_parent_sources',0)>=8
        and fit.get('maximum_position_tolerance_m')==.03 and fit.get('maximum_orientation_tolerance_rad')==.15,
        'formal skills require eight completed parents and preregistered terminal caps; development BC is forbidden')
    manifest=inputs.read(m['TRAIN_source_manifest_path'],m['TRAIN_source_manifest_sha256'])
    require(manifest.get('fit_split')=='train' and manifest.get('VAL_used') is False and manifest.get('TEST_used') is False,
        'library fitting provenance is not TRAIN-only')
    prior=m.get('original_action_phase_prior_binding',{})
    require(prior==manifest.get('original_action_phase_prior_binding'),'original prior proof differs from frozen library fit')
    for field in ('demonstrations','original_action_phase_prior'):inputs.pointer(prior[field])
    prior_key=digest(prior)
    if prior_key not in cache:
        cache[prior_key]=original_prior_binding(inputs.root,prior['demonstrations']['path'],prior['original_action_phase_prior']['path'])
        demonstration=json.loads(inputs.pointer(prior['demonstrations']).read_text())
        for row in demonstration['episodes']:
            if row['split']=='train':inputs.file(row['npz_path'],row['npz_sha256'])
    require(cache[prior_key]==prior and set(prior['original_parent_groups'])=={f'train:{t}:{raw}' for t,raw in expected['train']},
        'all 1400 original TRAIN priors must be independently bound')
    collector_path=inputs.pointer(manifest['collector_manifest']);collector_key=str(collector_path)
    if collector_key not in cache:
        collector,cases=load_train_cases(inputs.root,collector_path)
        cache[collector_key]=(collector,cases)
        for item in collector['case_receipts']:
            record=inputs.read(item['path'],item['sha256']);inputs.include(record['files_sha256'])
            process=record['parent_process_receipt'];inputs.file(process['child_case_path'],process['child_case_sha256'])
            inputs.file(process['launcher'],process['launcher_sha256'])
    collector,cases=cache[collector_key]
    require(manifest['entries']==collector['entries'],'library fit silently omitted/modified acquisition entries')
    task_cases=[case for case in cases if case['entry']['task']==task]
    _exact(_parents([case['entry'] for case in task_cases],'train'),expected['train_task'][task],task+' native acquisition')
    selected=[case for case in task_cases if case['entry'].get('interaction_eligible') is True]
    _exact(_parents([case['entry'] for case in selected],'train'),expected['train_task'][task],task+' teacher acquisition')
    require(set(m.get('TRAIN_source_ids',[]))=={case['entry']['source_id'] for case in selected}
        and set(m.get('TRAIN_parent_source_groups',[]))=={f'train:{task}:{raw}' for t,raw in expected['train_task'][task]},
        'library declared acquisition sources differ from all 100 real teacher parents')
    inputs.pointer(manifest['interaction_adapter_source']);configuration=inputs.read_pointer(manifest['fitting_configuration'])
    for field in ('fitting_core_source_snapshot','fitting_producer_source_snapshot'):
        if field in configuration:inputs.source_snapshot(configuration[field])
    require(configuration.get('interaction')==fit,'library fit configuration proof differs')
    reference_time_rule=None
    rule_pointer=configuration.get('reference_time_derivation_rule')
    if rule_pointer is not None:
        from .reference_time import load_rule
        reference_time_rule=load_rule(inputs.root,rule_pointer)
        inputs.pointer(rule_pointer)
        require(manifest.get('reference_time_derivation')==dict(
            rule=dict(path=reference_time_rule.path,sha256=reference_time_rule.sha256),
            evidence_report=reference_time_rule.evidence_report,
            verified_files_sha256=reference_time_rule.verified_files_sha256,
            original_raw_terminal_reference_and_masks_preserved=True),
            'library reference-time derivation differs from its exact TRAIN rule/evidence')
        inputs.include(reference_time_rule.verified_files_sha256)
    else:
        require(manifest.get('reference_time_derivation') is None,
            'library has an undeclared reference-time derivation')
    if any(template.reference_time_basis=='command_issue' for template in library.templates):
        require(reference_time_rule is not None and task in reference_time_rule.tasks,
            'command_issue library requires its reviewed task-specific TRAIN derivation rule')
        inputs.file('our_mani/multitask/reference_time.py',m.get('reference_time_helper_sha256'))
        require(m.get('reference_time_rule_sha256_values')==[reference_time_rule.sha256],
            'library reference-time rules differ from the explicit fit rule')
    semantic_rule,semantic_sources=_semantic_sources(inputs,configuration,manifest,cases,reference_time_rule,cache)
    goal_rule,goal_sources=_command_goal_sources(inputs,configuration,manifest,cases,reference_time_rule,cache,semantic_rule)
    sources=[goal_sources[case['entry']['source_id']] if task in getattr(goal_rule,'tasks',()) else
        semantic_sources[case['entry']['source_id']] if task in getattr(semantic_rule,'tasks',())
        else interaction_source(case,reference_time_rule=reference_time_rule)[0] for case in selected]
    _semantic_library_metadata(m,sources,semantic_rule)
    _command_goal_library_metadata(m,sources,goal_rule)
    phase_counts={}
    for template in library.templates:
        key,parents,attempts=_completion_support(sources,template);label='|'.join(key)
        _command_goal_geometry(sources,template)
        require(len(parents)>=8,task+' '+label+' lacks eight actual completed TRAIN parents')
        proof=m.get('source_grouped_lag_selection',{}).get(label,{})
        require(proof.get('passed') is True and set(proof.get('completed_parent_sources',[]))==parents
            and proof.get('completed_attempt_count')==len(attempts)
            and set(proof.get('source_to_fold',{}))==parents,'library completed-skill parent proof differs from actual events')
        residual=proof.get('terminal_parent_residuals',{})
        require(set(residual)==parents and all(np.isfinite([r['position'],r['orientation']]).all() for r in residual.values()),
            'terminal parent-crossfit evidence is incomplete/nonfinite')
        pt=max(fit['minimum_position_tolerance_m'],float(np.quantile([r['position'] for r in residual.values()],fit['tolerance_quantile'])))
        rt=max(fit['minimum_orientation_tolerance_rad'],float(np.quantile([r['orientation'] for r in residual.values()],fit['tolerance_quantile'])))
        require(abs(pt-template.position_tolerance_m)<1e-8 and abs(rt-template.orientation_tolerance_rad)<1e-8
            and pt<=.03 and rt<=.15,'library completion tolerance bypasses its held-parent evidence/cap')
        phase_counts[label]=len(parents)
    return library,dict(passed=True,native_acquisition_parent_sources=100,original_TRAIN_priors=1400,
        completed_parent_sources_by_skill=phase_counts,BC_acquisition_coverage_is_not_completed_skill_coverage=True)


def _critic(inputs,protocol,task,predictor,library,expected):
    from .candidate_success import CandidateSuccessModel
    from .candidate_training import load_executed_manifest,parent_group,_normalise
    from .factory import continuation_identity
    from .interaction import CandidateConfig,MPCConfig
    path=inputs.pointer(protocol['learned_artifacts'][task][predictor+'_value'])
    continuation=continuation_identity(inputs.root,protocol,task,predictor,library,
        CandidateConfig(**protocol['candidate_config']),MPCConfig(**protocol['mpc_config']))['sha256']
    require(protocol.get('continuation_identities',{}).get(task,{}).get(predictor,continuation)==continuation,
        'formal critic geometry continuation identity changed')
    model=CandidateSuccessModel(path,continuation_sha256=continuation,library_sha256=library.sha256);m=model.metadata
    require(m.get('task')==task and m.get('predictor')==predictor,'critic belongs to a different task/predictor')
    report=inputs.read(path.parent/'report.json')
    require(report.get('artifact_sha256')==file_sha(path) and report.get('metadata')==m
        and report.get('formal_TEST_used') is False,'critic fit report does not bind this model')
    inputs.include(m['source_sha256']);data={};audits={}
    for split,field in (('train','TRAIN_source_manifest'),('validation','VAL_calibration_manifest')):
        manifest_path=inputs.file(m[field+'_path'],m[field+'_sha256'])
        rows,subset,audit=load_executed_manifest(manifest_path,split=split,task_name=task,predictor=predictor,
            continuation_sha256=continuation,library_sha256=library.sha256,project_root=inputs.root)
        _exact(_parents(subset['entries'],split),expected[split+'_task'][task],task+' '+predictor+' '+split+' critic acquisition')
        require(not audit['censored_rollouts'] and audit['completed_native_rollouts']==audit['selected_rollouts'],
            'formal critic coverage cannot hide censored native cases')
        expected_groups={f'{split}:{task}:{raw}' for t,raw in expected[split+'_task'][task]}
        require(set(audit['completed_parent_sources'])==expected_groups,'critic completed parent coverage differs')
        for entry in subset['entries']:
            case=inputs.read(entry['path'],entry['sha256']);identity=case.get('identity',{})
            validate_completed_case(case,identity)
            require(identity.get('demo_only') is False and identity.get('method')==predictor+'_geometry',
                'critic labels require actual non-demo geometry continuation')
        data[split]=[row for row in rows if row['first_actionable']];audits[split]=audit
        group_field='TRAIN_parent_source_groups' if split=='train' else 'VAL_parent_source_groups'
        require(set(m.get(group_field,[]))==expected_groups,'critic full executed-parent metadata differs')
        eligible_field=('TRAIN' if split=='train' else 'VAL')+'_eligible_first_decision_parent_source_groups'
        require(set(m.get(eligible_field,[]))=={row['group'] for row in data[split]},
            'critic eligible labels cannot be confused with full acquisition coverage')
        original=m.get('execution_coverage_proof',{}).get('TRAIN' if split=='train' else 'VAL',{})
        original_path=inputs.file(original['source_manifest_path'],original['source_manifest_sha256'])
        original_rows,original_subset,original_audit=load_executed_manifest(original_path,split=split,
            task_name=task,predictor=predictor,continuation_sha256=continuation,
            library_sha256=library.sha256,project_root=inputs.root)
        require(original_subset==subset and original_audit==original,
            'critic raw executed-case coverage proof differs from original acquisition')
    first=library.phase_order[0];key='|'.join((task,predictor,first))
    require(set(model.heads)=={key},'formal value must have exactly the episode-first trained head')
    head,mean,scale,weights,bias,tree=model.heads[key];train=data['train'];validation=data['validation']
    require(train and all(row['phase']==first for row in train+validation),'critic population differs from episode-first decision')
    groups=np.asarray([row['group'] for row in train]);x=np.stack([row['x'] for row in train])
    require(len(set(groups))>=2,'a trained critic needs at least two actual eligible TRAIN parents')
    actual_mean,actual_scale=_normalise(x,groups)
    require(np.allclose(mean,actual_mean,atol=1e-8,rtol=1e-8) and np.allclose(scale,actual_scale,atol=1e-8,rtol=1e-8)
        and head['support_parent_sources']==groups.tolist()
        and np.allclose(head['standardized_TRAIN_support'],(x-mean)/scale,atol=1e-8,rtol=1e-8),
        'critic normalization/support did not use every eligible TRAIN first decision')
    proof=report.get('per_phase',{}).get(first,{})
    require(proof.get('TRAIN_samples')==len(train) and proof.get('VAL_samples')==len(validation)
        and proof.get('TRAIN_parent_sources')==len(set(groups))
        and proof.get('VAL_parent_sources')==len({row['group'] for row in validation}),
        'critic labelled decision/parent proof differs from actual cases')
    require(set(proof.get('TRAIN_selection',{}).get('source_to_fold',{}))==set(groups)
        and len(proof.get('bootstrap',[]))==5,'critic TRAIN folds/bootstrap coverage incomplete')
    for member in proof['bootstrap']:
        multipliers=member.get('parent_source_weights',{})
        require(set(multipliers)==set(groups) and all(np.isfinite(v) and v>0 for v in multipliers.values()),
            'each bootstrap member must positively cover every eligible TRAIN parent')
    enabled=head['ranking_enabled']
    require(proof.get('ranking_enabled') is enabled and report.get('episode_value_ranking_enabled') is enabled,
        'critic ranking guard/report differs')
    if enabled:
        cal=proof.get('calibration_crossfit',{});cm=cal.get('metrics',{});raw=cal.get('raw_metrics',{})
        cfg=m['fit_configuration']
        require(cal.get('valid') is True and proof.get('calibration_gate_passed') is True
            and all(np.isfinite(cm.get(name,float('nan'))) for name in ('ECE','Brier','NLL'))
            and cm['ECE']<=cfg['maximum_crossfit_ECE'] and cm['Brier']<=raw['Brier']+cfg['maximum_Brier_regression'],
            'enabled learned ranking lacks valid finite VAL calibration evidence')
    return dict(passed=True,TRAIN_acquired_parent_sources=100,
        VAL_acquired_parent_sources=len(expected['validation_task'][task]),
        eligible_TRAIN_label_parents=len(set(groups)),eligible_VAL_label_parents=len({r['group'] for r in validation}),
        zero_candidate_TRAIN_rollouts=len(audits['train']['zero_candidate_rollouts']),
        zero_candidate_VAL_rollouts=len(audits['validation']['zero_candidate_rollouts']),
        ranking_enabled=enabled,geometry_fallback_is_legal=not enabled,
        zero_candidate_sources_are_not_fabricated_classifier_samples=True)


def audit_formal_model_coverage(root,protocol,*,context_training_view,
        train_source_manifest='artifacts/framework/protocol/train_manifest.json',
        validation_source_manifest='artifacts/framework/protocol/validation_manifest.json',frozen_files=None):
    """Return coverage proof/errors. Public formal counts have no pilot override."""
    inputs=_Inputs(root);errors=[];sections=dict(libraries={},critics={});expected={};cache={}
    def check(label,operation):
        try:return operation()
        except Exception as error:errors.append(dict(component=label,error=repr(error)));return None
    for split,path in (('train',train_source_manifest),('validation',validation_source_manifest)):
        value=check('canonical_'+split,lambda split=split,path=path:_canonical(inputs,path,split))
        if value is not None:
            expected[split],sections['canonical_'+split]=value
            expected[split+'_task']={task:{key for key in expected[split] if key[0]==task} for task in TASK_SPECS}
    if len(expected)==4:
        require(not expected['train']&expected['validation'],'canonical TRAIN/VAL raw parents overlap')
        sections['context']=check('context',lambda:_context(inputs,protocol,context_training_view,expected))
        for task in TASK_SPECS:
            value=check('library:'+task,lambda task=task:_library(inputs,protocol,task,expected,cache))
            if value is None:continue
            library,sections['libraries'][task]=value
            for predictor in ('cv','context'):
                value=check('critic:'+task+':'+predictor,
                    lambda task=task,predictor=predictor,library=library:_critic(inputs,protocol,task,predictor,library,expected))
                sections['critics'][task+'|'+predictor]=value
    for name in ('release_preflight.py','training_view.py','artifact_fitting.py','interaction.py','candidate_training.py',
            'candidate_success.py','context_se3.py','motion_labels.py','specs.py','factory.py','evaluation.py'):
        check('coverage_source:'+name,lambda name=name:inputs.file('our_mani/multitask/'+name))
    check('coverage_cli_source',lambda:inputs.file('scripts/preflight_multitask_model_coverage_r10.py'))
    check('stable_inputs',inputs.rehash_all)
    if frozen_files is not None:
        normalized=normalized_lock(inputs.root,frozen_files)
        check('coverage_lock',lambda:require(all(normalized.get(name)==checksum for name,checksum in inputs.files.items()),
            'formal input lock does not include every model/coverage evidence digest'))
    return dict(schema=SCHEMA,complete=True,formal_model_coverage_passed=not errors,
        comparison_protocol_sha256=comparison_protocol_sha256(protocol),
        coverage_source_sha256=file_sha(Path(__file__)),errors=errors,sections=sections,
        input_files_sha256=inputs.files,TRAIN_expected_parents=1400,VAL_expected_parents=275,
        actual_TEST_read=False,model_fitting_performed=False,native_or_GPU_launched=False,
        performance_or_SR_benefit_established=False,
        limitations=['Coverage/provenance does not prove policy effectiveness or full state parity.',
            'Symmetry/affordance observability and complete five-method VAL acceptance remain separate gates.'])


def require_formal_model_coverage(*args,**kwargs):
    report=audit_formal_model_coverage(*args,**kwargs)
    require(report['formal_model_coverage_passed'],'formal model coverage failed: '+repr(report['errors']))
    return report


def require_frozen_coverage_certificate(root,pointer,protocol,frozen_files):
    """Cheap child gate after parent audit; caller already verifies its full lock."""
    inputs=_Inputs(root);path=inputs.pointer(pointer);report=json.loads(path.read_text())
    require(report.get('schema')==SCHEMA and report.get('complete') is True
        and report.get('formal_model_coverage_passed') is True and not report.get('errors')
        and report.get('comparison_protocol_sha256')==comparison_protocol_sha256(protocol)
        and report.get('coverage_source_sha256')==file_sha(Path(__file__))
        and report.get('actual_TEST_read') is False and report.get('model_fitting_performed') is False,
        'formal full-source coverage certificate is missing/stale/failed')
    sections=report.get('sections',{});context=sections.get('context',{})
    counts=sections.get('canonical_validation',{}).get('per_task',{})
    require(sections.get('canonical_train',{}).get('per_task')==dict.fromkeys(TASK_SPECS,100)
        and set(counts)==set(TASK_SPECS) and sum(counts.values())==275 and all(18<=n<=20 for n in counts.values())
        and context.get('passed') is True and context.get('TRAIN_parent_sources')==1400
        and context.get('VAL_parent_sources')==275 and context.get('optimizer_parent_sources')==1400,
        'coverage certificate lacks the full canonical/optimizer Context proof')
    libraries=sections.get('libraries',{});critics=sections.get('critics',{})
    require(set(libraries)==set(TASK_SPECS) and set(critics)=={task+'|'+p for task in TASK_SPECS for p in ('cv','context')},
        'coverage certificate must cover all fourteen libraries and twenty-eight critics')
    for task,spec in TASK_SPECS.items():
        lib=libraries[task];support=lib.get('completed_parent_sources_by_skill',{})
        required={phase.name for phase in spec.phase_skills if not phase.optional_after_native_completion}
        require(lib.get('passed') is True and lib.get('native_acquisition_parent_sources')==100
            and lib.get('original_TRAIN_priors')==1400 and support
            and required<={key.split('|')[0] for key in support}
            and all(type(n) is int and n>=8 for n in support.values()),
            'coverage certificate contains a pilot/unsupported interaction library: '+task)
        for predictor in ('cv','context'):
            critic=critics[task+'|'+predictor]
            require(isinstance(critic,dict) and critic.get('passed') is True
                and critic.get('TRAIN_acquired_parent_sources')==100 and critic.get('VAL_acquired_parent_sources')==counts[task]
                and type(critic.get('ranking_enabled')) is bool
                and critic.get('geometry_fallback_is_legal') is (not critic['ranking_enabled']),
                'coverage certificate critic coverage/fallback proof differs: '+task+'|'+predictor)
    normalized=normalized_lock(inputs.root,frozen_files)
    require(normalized.get(str(path.relative_to(inputs.root)))==pointer['sha256']
        and report.get('input_files_sha256')
        and all(normalized.get(name)==checksum for name,checksum in report['input_files_sha256'].items()),
        'TEST lock must bind the full model coverage proof and all audited inputs')
    inputs.rehash_all()
    return dict(passed=True,path=str(path.relative_to(inputs.root)),sha256=pointer['sha256'],
        formal_model_coverage_passed=True,TRAIN_parent_sources=1400,VAL_parent_sources=275,
        performance_or_SR_benefit_established=False)
