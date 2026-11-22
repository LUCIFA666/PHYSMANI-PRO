"""Independent model coverage for the measured-feedback policy overlay.

Legacy coverage/release consumers stay unchanged. Interaction/Context/colour
provenance and critic numerical gates are reused without changing thresholds.
Only actually recollected geometry labels bearing the new extended continuation
and installed feedback-controller witness can train this policy's Value heads.
"""
import ast
import inspect
import json
from pathlib import Path
import numpy as np
from .evaluation import digest,file_sha,validate_completed_case
from . import release_preflight as models
from . import task_release as legacy
from .task_release import (_task,_geometry,_colour,task_comparison_sha256,
    DEFAULT_TRAIN,DEFAULT_VAL)
from .specs import TASK_SPECS
from .validation_release import Proof,require

COVERAGE='r10_feedback_task_trained_model_coverage_v1'


def _feedback_identity(inputs,protocol,task,predictor,library):
    from .feedback_actor_backend import feedback_continuation_identity,verify_policy_overlay,IDENTITY_SCHEMA
    from .factory import continuation_identity
    from .interaction import CandidateConfig,MPCConfig
    overlay=verify_policy_overlay(inputs.root,protocol)
    candidate=CandidateConfig(**protocol['candidate_config']);mpc=MPCConfig(**protocol['mpc_config'])
    original=continuation_identity(inputs.root,protocol,task,predictor,library,candidate,mpc)
    identity=feedback_continuation_identity(inputs.root,protocol,task,predictor,library,candidate,mpc)
    require(set(identity)=={'version','task','predictor','original_continuation','policy_overlay',
        'continuation_selection_policy','downstream_selection_policy',
        'original_physics_and_observation_sources_unchanged','sha256'}
        and identity['version']==IDENTITY_SCHEMA and identity['task']==task and identity['predictor']==predictor
        and identity['original_continuation']==original and identity['policy_overlay']==overlay
        and identity['continuation_selection_policy']=='single_value_selection_then_frozen_geometry'
        and identity['downstream_selection_policy']=='geometry'
        and identity['original_physics_and_observation_sources_unchanged'] is True
        and identity['sha256']==digest({k:v for k,v in identity.items() if k!='sha256'})
        and identity['sha256']!=original['sha256']
        and protocol.get('continuation_identities',{}).get(task,{}).get(predictor)==identity['sha256'],
        'feedback continuation must bind the actual original source plus explicit new controller/config')
    inputs.include(original['observation_control_sources'])
    for pointer in overlay['sources'].values():inputs.pointer(pointer)
    # The original kernel fingerprint also binds the legacy interaction source.
    inputs.file('our_mani/multitask/interaction.py')
    return identity


def _verify_feedback_label_case(inputs,case,entry,identity):
    from .feedback_interaction import feedback_binding,SCHEMA,RECOVERABLE
    require(case.get('continuation_identity')==identity
        and case.get('continuation_sha256')==identity['sha256']
        and case.get('task')==entry['task']==identity['task']
        and case.get('predictor')==entry['predictor']==identity['predictor'],
        'critic raw label is not the complete actual extended feedback continuation')
    receipt=case.get('actor_metrics',{}).get('observed_feedback_controller')
    require(isinstance(receipt,dict),'critic raw lacks the installed measured-feedback controller witness')
    config=identity['policy_overlay']['configuration'];binding=feedback_binding(config)
    source=identity['policy_overlay']['sources']['controller']
    base=dict(path='our_mani/multitask/interaction.py',sha256=file_sha(inputs.root/'our_mani/multitask/interaction.py'))
    require(receipt.get('schema')==SCHEMA
        and receipt.get('actual_controller_class')=='our_mani.multitask.feedback_interaction.ObservedFeedbackController'
        and receipt.get('normalized_config')==binding['normalized_config']
        and receipt.get('configuration_sha256')==binding['configuration_sha256']
        and receipt.get('source')==binding['source']==source and receipt.get('base_source')==base
        and receipt.get('continuation_sha256')==identity['sha256']
        and receipt.get('installed_before_first_step') is True
        and type(receipt.get('step_calls')) is int and receipt['step_calls']>=0
        and all(receipt.get(key) is True for key in ('candidate_selection_unchanged','public_skill_completion_unchanged','grasp_timing_unchanged'))
        and all(receipt.get(key) is False for key in ('nominal_completion_claimed','total_budget_extended','additional_sensor_or_physics_queries')),
        'critic actual feedback controller/config/source/gates do not match the frozen policy')
    inputs.pointer(source);inputs.pointer(base)
    def number(value):return type(value) in (int,float) and np.isfinite(value)
    timing=receipt.get('phase_timing_events');events=receipt.get('fallback_events')
    require(isinstance(timing,list) and isinstance(events,list),'feedback timing/fallback evidence must retain the raw lists')
    for row in timing:
        names=('selected_at','observed_time','original_phase_budget_seconds','requested_slack_seconds','granted_slack_seconds',
            'total_task_remaining_seconds','suffix_reserve_seconds','original_phase_deadline','actual_phase_deadline')
        require(isinstance(row,dict) and all(number(row.get(name)) for name in names)
            and row['selected_at']<=row['observed_time']+1e-8
            and row['original_phase_budget_seconds']>=0 and 0<=row['total_task_remaining_seconds']<=10.+1e-8
            and row['suffix_reserve_seconds']>=0 and row['requested_slack_seconds']==config['phase_tracking_slack_seconds']
            and row.get('total_budget_extended') is False and row.get('nominal_trajectory_completed') is False
            and row.get('source')=='declared_feedback_config_capped_by_current_task_remaining_and_original_suffix',
            'feedback phase timing is not the explicitly declared unchanged total-budget schedule')
        deadline=row['selected_at']+row['original_phase_budget_seconds']
        available=max(0.,row['total_task_remaining_seconds']-max(0.,deadline-row['observed_time'])-row['suffix_reserve_seconds'])
        granted=min(config['phase_tracking_slack_seconds'],available)
        require(abs(row['granted_slack_seconds']-granted)<=1e-8
            and abs(row['original_phase_deadline']-deadline)<=1e-8
            and abs(row['actual_phase_deadline']-deadline-granted)<=1e-8,
            'feedback phase slack exceeds its actual task remainder or changed the original suffix reserve')
    require(receipt.get('phase_budget_unchanged') is (not any(row['granted_slack_seconds']>0 for row in timing)),
        'feedback phase-budget claim differs from the actual declared timing events')
    for row in events:
        require(isinstance(row,dict) and row.get('schema')==SCHEMA
            and number(row.get('observed_time')) and number(row.get('duration')) and abs(row['duration']-.05)<=1e-10
            and row.get('original_rejection') in RECOVERABLE
            and type(row.get('bounded_gate_feasible')) is bool and type(row.get('command_issued')) is bool
            and isinstance(row.get('bounded_gate_evidence'),dict)
            and all(row.get(key) is False for key in ('nominal_trajectory_completed','total_budget_extended','additional_sensor_or_physics_queries'))
            and type(row.get('phase_deadline_extended')) is bool,
            'feedback fallback bypassed the original gate or fabricated nominal completion')
        for name in ('measured_joints','measured_joint_velocities','nominal_joints','bounded_joints','bounded_joint_velocities'):
            array=np.asarray(row.get(name),float)
            require(array.shape==(7,) and np.isfinite(array).all(),'feedback fallback lacks finite actual measured/nominal/bounded joints')
        if row.get('nominal_joint_velocities') is not None:
            array=np.asarray(row['nominal_joint_velocities'],float)
            require(array.shape==(7,) and np.isfinite(array).all(),'feedback nominal joint velocities are invalid')
        require(not row['command_issued'] or (row['bounded_gate_feasible'] is True
            and row.get('bounded_gate_reason')=='feasible_sampled_trajectory'),
            'feedback issued a command without the complete original trajectory gate')
        if row['phase_deadline_extended']:
            require(any(item.get('candidate_id')==row.get('candidate_id') and item['granted_slack_seconds']>0
                and item['selected_at']<=row['observed_time']+1e-8 for item in timing),
                'feedback fallback claims undeclared phase slack')


def _feedback_critic(inputs,protocol,task,predictor,library,expected):
    from .candidate_success import CandidateSuccessModel
    from .candidate_training import load_executed_manifest,parent_group,_normalise
    feedback_identity=_feedback_identity(inputs,protocol,task,predictor,library)
    continuation=feedback_identity['sha256']
    path=inputs.pointer(protocol['learned_artifacts'][task][predictor+'_value'])
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
            _verify_feedback_label_case(inputs,case,entry,feedback_identity)
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


def _critic_kernel():
    """The original critic's numerical/provenance suffix remains unchanged."""
    def suffix(function):
        tree=ast.parse(inspect.getsource(function));body=tree.body[0].body
        start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='model' for t in node.targets))
        class RemoveFeedbackWitness(ast.NodeTransformer):
            def visit_Expr(self,node):
                if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                        and node.value.func.id=='_verify_feedback_label_case'):return None
                return self.generic_visit(node)
        tree=ast.Module(body=body[start:],type_ignores=[])
        return ast.dump(RemoveFeedbackWitness().visit(tree),include_attributes=False)
    before=suffix(models._critic);after=suffix(_feedback_critic)
    require(before==after,'feedback critic modified a legacy numerical/provenance gate')
    return dict(unchanged=True,normalized_AST_sha256=digest(before),
        original_consumer_source_sha256=file_sha(models.__file__))


def audit_feedback_task_model_coverage(root, task, protocol, *, context_training_view,
        train_source_manifest=DEFAULT_TRAIN, validation_source_manifest=DEFAULT_VAL, frozen_files=None):
    """Recompute existing hard model gates for one task; never waive its critic."""
    _task(task);inputs=models._Inputs(root);errors=[];sections={};expected={};cache={}
    def check(name, operation):
        try:return operation()
        except Exception as error:errors.append(dict(component=name,error=repr(error)));return None
    for split,path in (('train',train_source_manifest),('validation',validation_source_manifest)):
        result=check('canonical_'+split,lambda split=split,path=path:models._canonical(inputs,path,split))
        if result is not None:
            expected[split],sections['canonical_'+split]=result
            expected[split+'_task']={name:{key for key in expected[split] if key[0]==name} for name in TASK_SPECS}
    if len(expected)==4:
        check('split_separation',lambda:require(not expected['train']&expected['validation'],'TRAIN/VAL parents overlap'))
        # This shared checkpoint really used all1400/275. That proof is retained;
        # only interaction/critic/colour gates are scoped to the selected task.
        sections['context']=check('context',lambda:models._context(inputs,protocol,context_training_view,expected))
        sections['geometry']=check('geometry:'+task,lambda:_geometry(inputs,protocol,task,expected))
        result=check('library:'+task,lambda:models._library(inputs,protocol,task,expected,cache))
        if result is not None:
            library,sections['library']=result
            required={p.name for p in TASK_SPECS[task].phase_skills if not p.optional_after_native_completion}
            check('mandatory_skills',lambda:require(required<={t.phase for t in library.templates},
                'task library omits a mandatory skill'))
            sections['critics']={predictor:check('critic:'+predictor,
                lambda predictor=predictor:_feedback_critic(inputs,protocol,task,predictor,library,expected))
                for predictor in ('cv','context')}
        if result is not None:
            sections['feedback_continuations']={predictor:check('feedback_identity:'+predictor,
                lambda predictor=predictor:_feedback_identity(inputs,protocol,task,predictor,library))
                for predictor in ('cv','context')}
        sections['colour']=check('colour:'+task,lambda:_colour(inputs,protocol,task,expected))
    check('task_gate_source',lambda:inputs.file('our_mani/multitask/feedback_model_coverage.py'))
    check('legacy_model_provenance_source',lambda:inputs.file('our_mani/multitask/task_release.py'))
    check('critic_kernel',_critic_kernel)
    check('original_model_gate_source',lambda:inputs.file('our_mani/multitask/release_preflight.py'))
    check('stable_inputs',inputs.rehash_all)
    if frozen_files is not None:
        normalized=models.normalized_lock(inputs.root,frozen_files)
        check('coverage_lock',lambda:require(all(normalized.get(name)==sha for name,sha in inputs.files.items()),
            'task frozen lock omits actual audited model evidence'))
    return dict(schema=COVERAGE,complete=True,formal_model_coverage_passed=not errors,task=task,
        task_comparison_sha256=task_comparison_sha256(protocol),sections=sections,errors=errors,
        full_experiment_tasks=list(TASK_SPECS),TRAIN_expected_parents=100,
        VAL_expected_parents=len(expected.get('validation_task',{}).get(task,())),
        coverage_source_sha256=file_sha(Path(__file__)),
        original_model_gate_source_sha256=file_sha(Path(models.__file__)),input_files_sha256=inputs.files,
        actual_TEST_read=False,model_fitting_performed=False,native_or_GPU_launched=False,
        feedback_overlay_enabled=True,legacy_certificate_reused=False,
        legacy_model_provenance_source_sha256=file_sha(Path(legacy.__file__)),
        missing_other_task_models_waived=False,other_task_release_required_separately=True,
        performance_or_SR_benefit_established=False)


def require_feedback_task_coverage_certificate(root, task, pointer, protocol, frozen_files):
    _task(task);proof=Proof(root);report=proof.read_pointer(pointer)
    require(report.get('schema')==COVERAGE and report.get('task')==task and report.get('complete') is True
        and report.get('formal_model_coverage_passed') is True and not report.get('errors')
        and report.get('task_comparison_sha256')==task_comparison_sha256(protocol)
        and report.get('full_experiment_tasks')==list(TASK_SPECS)
        and report.get('coverage_source_sha256')==file_sha(Path(__file__))
        and report.get('original_model_gate_source_sha256')==file_sha(Path(models.__file__))
        and report.get('feedback_overlay_enabled') is True and report.get('legacy_certificate_reused') is False
        and report.get('legacy_model_provenance_source_sha256')==file_sha(Path(legacy.__file__))
        and report.get('actual_TEST_read') is False and report.get('model_fitting_performed') is False,
        'task model coverage certificate is missing, failed, foreign or stale')
    s=report['sections'];counts=s.get('canonical_validation',{}).get('per_task',{})
    require(s.get('canonical_train',{}).get('per_task')==dict.fromkeys(TASK_SPECS,100)
        and set(counts)==set(TASK_SPECS) and sum(counts.values())==275 and all(18<=n<=20 for n in counts.values())
        and report.get('TRAIN_expected_parents')==100 and report.get('VAL_expected_parents')==counts[task],
        'task coverage lost original full inventory or task denominator')
    context=s.get('context',{})
    require(context.get('passed') is True and context.get('TRAIN_parent_sources')==1400
        and context.get('VAL_parent_sources')==275 and context.get('optimizer_parent_sources')==1400,
        'task Context checkpoint is incomplete')
    require(s.get('geometry',{}).get('passed') is True and s['geometry'].get('task')==task
        and s['geometry'].get('shared_with_baseline') is True
        and s['geometry'].get('asset')==protocol['learned_artifacts'][task]['geometry'],
        'task shared geometry/observation evidence is missing or changed')
    library=s.get('library',{});support=library.get('completed_parent_sources_by_skill',{})
    required={phase.name for phase in TASK_SPECS[task].phase_skills if not phase.optional_after_native_completion}
    require(library.get('passed') is True and library.get('native_acquisition_parent_sources')==100
        and library.get('original_TRAIN_priors')==1400 and support
        and required<={key.split('|')[0] for key in support}
        and all(type(n) is int and n>=8 for n in support.values()), 'task skill coverage is incomplete/unsupported')
    require(set(s.get('critics',{}))=={'cv','context'}, 'both task critics are required')
    for predictor,critic in s['critics'].items():
        require(isinstance(critic,dict) and critic.get('passed') is True
            and critic.get('TRAIN_acquired_parent_sources')==100
            and critic.get('VAL_acquired_parent_sources')==counts[task]
            and type(critic.get('ranking_enabled')) is bool
            and critic.get('geometry_fallback_is_legal') is (not critic['ranking_enabled']),
            'task critic TRAIN/CAL/fallback coverage is incomplete: '+predictor)
    require(s.get('colour',{}).get('passed') is True
        and s['colour'].get('required') is (TASK_SPECS[task].family in ('insert_onto_rotating_peg','push_moving_button')),
        'task-specific runtime colour gate is missing')
    from .interaction import InteractionLibrary
    from .candidate_success import CandidateSuccessModel
    inputs=models._Inputs(proof.root)
    library=InteractionLibrary.load(inputs.pointer(protocol['learned_artifacts'][task]['interaction']))
    identities=s.get('feedback_continuations',{})
    require(set(identities)=={'cv','context'},'both verified extended feedback continuations required')
    for predictor in ('cv','context'):
        actual=_feedback_identity(inputs,protocol,task,predictor,library)
        require(identities[predictor]==actual,'feedback model certificate continuation/source changed')
        CandidateSuccessModel(inputs.pointer(protocol['learned_artifacts'][task][predictor+'_value']),
            continuation_sha256=actual['sha256'],library_sha256=library.sha256)
    inputs.rehash_all()
    normalized=models.normalized_lock(proof.root,frozen_files)
    require(normalized.get(proof.name(pointer['path']))==pointer['sha256']
        and report.get('input_files_sha256')
        and all(report['input_files_sha256'].get(name)==sha for name,sha in inputs.files.items())
        and all(normalized.get(proof.name(name))==sha for name,sha in report['input_files_sha256'].items()),
        'frozen lock does not bind this task certificate and every actual model input')
    return dict(passed=True,task=task,certificate=pointer,TRAIN_parent_sources=100,
        VAL_parent_sources=counts[task],formal_model_coverage_passed=True)
