"""Strict per-task five-method release, alongside the unchanged global gate.

All fourteen canonical tasks remain registered. One task may proceed after its
own actual model evidence and complete five-way VAL pass; no outcome threshold
is introduced. Inference, native execution, fitting and TEST outcomes are not
performed or read by this module.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .evaluation import METHODS, atomic_json, case_identity, digest, file_sha, validate_completed_case
from . import release_preflight as models
from .reset_parity import compare_five_way, profile_problems
from .specs import TASK_SPECS
from .trace_evidence import audit_selection_usage
from .validation_release import Proof, proof_directory, require

COVERAGE = 'r10_task_formal_model_coverage_v1'
RELEASE = 'r10_task_complete_five_method_validation_release_v1'
EXECUTION = 'r10_task_formal_case_execution_contract_v1'
COMPOSITION = 'r10_task_four_ours_plus_full_verified_baseline_VAL_composition_v1'
DEFAULT_TRAIN = 'artifacts/framework/protocol/train_manifest.json'
DEFAULT_VAL = 'artifacts/framework/protocol/validation_manifest.json'
DEFAULT_TEST = 'artifacts/framework/multitask_r10/official_test_manifest.json'
DEFAULT_EXPERIMENT = 'config/framework/multitask_r10_experiment_plan.json'
IDENTITY_ADDITIONS = {'task_model_coverage_certificate', 'task_validation_release',
                      'formal_comparison_ready'}


def task_comparison_sha256(protocol):
    """Only explicit release pointers/flag are omitted to avoid SHA cycles."""
    return digest({key:value for key,value in protocol.items() if key not in IDENTITY_ADDITIONS})


def _task(task):
    require(task in TASK_SPECS, 'unknown task; complete original fourteen-task scope is retained')


def _canonical_task(proof, path, split, task):
    _task(task)
    value = proof.read(path)
    require(value.get('split') == split, 'canonical split differs')
    rows = [row for row in value.get('episodes',[]) if row.get('task') in TASK_SPECS]
    keys = [(row['task'],row['raw_episode_id']) for row in rows]
    require(rows and len(keys)==len(set(keys)), 'duplicate/empty canonical original source inventory')
    counts = Counter(name for name,raw in keys)
    require(set(counts)==set(TASK_SPECS), 'canonical inventory must retain all fourteen tasks')
    if split == 'train':
        require(len(rows)==1400 and set(counts.values())=={100}, 'original TRAIN requires 100 parents/task')
    elif split == 'validation':
        require(len(rows)==275 and all(18<=n<=20 for n in counts.values()), 'actual VAL must retain all 275 parents')
    elif split == 'test':
        require(len(rows)==560 and set(counts.values())=={40}
            and value.get('mapping_status')=='verified' and value.get('episode_count')==560
            and value.get('training_calibration_or_parameter_selection_used') is False,
            'official TEST requires full verified 560-row archive and forty parents/task')
        require(all(row.get('mapping_status')=='verified' for row in rows), 'TEST row mapping is unverified')
    else:
        raise ValueError('unknown split')
    selected = {(row['task'],row['raw_episode_id']):row for row in rows if row['task']==task}
    return selected, dict(path=proof.name(path),sha256=file_sha(proof.root/path),
        full_original_sources=len(rows),task_sources=len(selected),per_task=dict(counts))


def _colour(inputs, protocol, task, expected):
    """Only the actual runtime colour consumers require colour evidence."""
    if TASK_SPECS[task].family not in ('insert_onto_rotating_peg','push_moving_button'):
        return dict(passed=True,required=False,reason='runtime factory does not construct a colour resolver for this task')
    from .public_roles import PublicRoleResolver
    value = inputs.read_pointer(protocol['learned_artifacts'][task]['colour'])
    PublicRoleResolver(TASK_SPECS[task],value,shared_semantic_permission=True)
    metadata = value['metadata']
    manifest = inputs.read(metadata['TRAIN_manifest_path'],metadata['TRAIN_manifest_sha256'])
    require(manifest.get('fit_split')=='train' and manifest.get('VAL_used') is False
        and manifest.get('TEST_used') is False, 'colour source must be actual TRAIN-only fitting evidence')
    groups = set(metadata['TRAIN_parent_source_groups'])
    wanted = {f'train:{task}:{raw}' for name,raw in expected['train_task'][task]}
    task_groups = {group for group in groups if group.startswith('train:'+task+':')}
    require(task_groups and task_groups<=wanted, 'colour contains foreign task TRAIN parents')
    roles=('peg0','peg1','peg2') if TASK_SPECS[task].family=='insert_onto_rotating_peg' else ('button',)
    require(all(task+'|'+role in value.get('role_components',{}) for role in roles),
        'colour artifact lacks a required public target role')
    crossfit=metadata.get('TRAIN_parent_crossfit',{})
    require(crossfit.get('valid') is True and crossfit.get('passed') is True
        and crossfit.get('split')=='train' and crossfit.get('VAL_used') is False
        and crossfit.get('TEST_used') is False,'required task colour has not passed its original TRAIN parent crossfit')
    for colour in value['colours'].values():
        require(set(colour['TRAIN_parent_source_groups'])<=groups, 'colour prototype source groups were not fitted')
    inputs.file('our_mani/multitask/public_roles.py',metadata['source_sha256'])
    return dict(passed=True,required=True,TRAIN_label_parent_sources=len(task_groups),
        native_variation_or_target_used_at_runtime=False)


def _geometry(inputs,protocol,task,expected):
    from .training_geometry import _check_asset
    pointer=protocol['learned_artifacts'][task]['geometry'];path=inputs.pointer(pointer)
    asset=inputs.read_pointer(pointer);_check_asset(asset)
    require(asset.get('task')==task and asset.get('world_pose_or_motion_schedule_in_asset') is False
        and asset.get('native_task_source_sha256')==TASK_SPECS[task].native_source_sha256,
        'runtime geometry is foreign or contains private motion/world poses')
    wanted={f'train:{task}:{raw}' for name,raw in expected['train_task'][task]}
    require(asset.get('source_group') in wanted,'geometry prior source is not an original task TRAIN parent')
    geometry_map=inputs.read_pointer(protocol['frozen_TRAIN_geometry_map'])
    require(set(geometry_map)==set(TASK_SPECS)
        and (inputs.root/geometry_map[task]).resolve()==path,'task geometry differs from shared fixed TRAIN CAD map')
    observations=protocol['body_orientation_observability'][task]
    require(set(observations)==set(asset['bodies']) and set(observations.values())<={'full','axis','none'},
        'task public orientation permissions differ from the TRAIN body roster')
    require(protocol.get('shared_semantic_instance_binding') is True
        and protocol.get('semantic_binding_available_to_baseline') is True
        and protocol.get('shared_TRAIN_CAD_geometry') is True,'task geometry/semantics are not shared with baseline')
    inputs.file('our_mani/multitask/training_geometry.py')
    return dict(passed=True,asset=pointer,task=task,shared_with_baseline=True,
        public_orientation_permissions=observations,private_world_pose_or_schedule_used=False)


def audit_task_model_coverage(root, task, protocol, *, context_training_view,
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
                lambda predictor=predictor:models._critic(inputs,protocol,task,predictor,library,expected))
                for predictor in ('cv','context')}
        sections['colour']=check('colour:'+task,lambda:_colour(inputs,protocol,task,expected))
    check('task_gate_source',lambda:inputs.file('our_mani/multitask/task_release.py'))
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
        missing_other_task_models_waived=False,other_task_release_required_separately=True,
        performance_or_SR_benefit_established=False)


def require_task_coverage_certificate(root, task, pointer, protocol, frozen_files):
    _task(task);proof=Proof(root);report=proof.read_pointer(pointer)
    require(report.get('schema')==COVERAGE and report.get('task')==task and report.get('complete') is True
        and report.get('formal_model_coverage_passed') is True and not report.get('errors')
        and report.get('task_comparison_sha256')==task_comparison_sha256(protocol)
        and report.get('full_experiment_tasks')==list(TASK_SPECS)
        and report.get('coverage_source_sha256')==file_sha(Path(__file__))
        and report.get('original_model_gate_source_sha256')==file_sha(Path(models.__file__))
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
    normalized=models.normalized_lock(proof.root,frozen_files)
    require(normalized.get(proof.name(pointer['path']))==pointer['sha256']
        and report.get('input_files_sha256')
        and all(normalized.get(proof.name(name))==sha for name,sha in report['input_files_sha256'].items()),
        'frozen lock does not bind this task certificate and every actual model input')
    return dict(passed=True,task=task,certificate=pointer,TRAIN_parent_sources=100,
        VAL_parent_sources=counts[task],formal_model_coverage_passed=True)


def _plan(proof,path,canonical,task,split='validation',*,methods=METHODS):
    methods=tuple(methods)
    require(methods in (METHODS,METHODS[:-1]),'task release accepts exact five or explicit composite four-ours methods')
    plan=proof.read(path);n=len(canonical)
    require(plan.get('version')=='r10_fresh_policy_queue_v1'
        and plan.get('sha256')==digest({k:v for k,v in plan.items() if k!='sha256'})
        and plan.get('split')==split and plan.get('methods')==list(methods)
        and plan.get('replicas')==1 and plan.get('source_episodes')==n
        and plan.get('expected_cases')==len(methods)*n and plan.get('official_TEST_used_for_fitting') is False
        and plan.get('case_isolation')=='fresh_simulator_process_per_episode',
        'immutable task queue requires all five methods and exact source count')
    for field in ('config','manifest','frozen_inputs','parent_source','child_source'):proof.pointer(plan[field])
    _,manifest_info=_canonical_task(proof,proof.pointer(plan['manifest']),split,task)
    require(manifest_info['task_sources']==n, 'task queue manifest denominator differs')
    plan_rows,_=_canonical_task(proof,proof.root/plan['manifest']['path'],split,task)
    require(plan_rows==canonical, 'task queue changed original source rows')
    seen=set();job_ids=set()
    for job in plan.get('jobs',[]):
        row=job['row'];key=(row['task'],row['raw_episode_id']);method=job['identity']['method']
        require(key in canonical and row==canonical[key] and method in methods
            and job.get('replica')==0 and job.get('exploration_seed') is None
            and job.get('source_raw_episode_id')==key[1] and job.get('record_demo') is False,
            'task job is foreign, explored, replicated or a demo')
        identity=case_identity(split=split,task=task,raw_episode_id=key[1]+('__policy_aug_r0' if split!='test' else ''),
            method=method,source_row=row,config_sha256=plan['config']['sha256'],inputs_sha256=plan['frozen_inputs']['sha256'],
            parent_raw_episode_id=key[1] if split!='test' else None,replica=0)
        require(job['identity']==identity and job['job_id']==identity['sha256'][:20]
            and job['job_id'] not in job_ids and (key,method) not in seen,'task job identity is duplicated or changed')
        seen.add((key,method));job_ids.add(job['job_id'])
    require(len(plan.get('jobs',[]))==len(methods)*n and seen=={(key,m) for key in canonical for m in methods},
        'complete paired task five-method coverage missing')
    return plan


def _execution_contract(proof,plan,files,task):
    value=plan.get('task_execution_contract')
    if value is None:return None
    require(set(value)=={'schema','task','original_evaluator','wrapper','launchers'}
        and value['schema']==EXECUTION and value['task']==task
        and value['wrapper']==plan['child_source'] and set(value['launchers'])=={'ours','physmani'},
        'task wrapper execution contract differs')
    for pointer in [value['original_evaluator'],value['wrapper'],*value['launchers'].values()]:
        proof.locked(pointer['path'],files,pointer['sha256'])
    return value


def _cases(proof,plan,task,canonical,files):
    methods=tuple(plan['methods']);require(methods in (METHODS,METHODS[:-1]),'unsupported task method set')
    output=(proof.root/plan['output']).resolve();expected=len(canonical)*len(methods)
    require(output.is_relative_to(proof.root/'artifacts/framework/multitask_r10'), 'task output escaped project')
    status=proof.read(output/'status.json')
    require(status.get('complete') is True and status.get('stage')=='complete'
        and status.get('completed')==status.get('expected')==expected
        and status.get('plan_sha256')==plan['sha256'] and not status.get('errors')
        and status.get('stop_requested') is False, 'task queue is incomplete or has infrastructure errors')
    require({p.name for p in (output/'results').glob('*.json')}=={job['job_id']+'.json' for job in plan['jobs']},
        'missing or foreign task canonical receipt')
    contract=_execution_contract(proof,plan,files,task);groups=defaultdict(dict);receipts=[];children=set()
    config=plan['_config'];baseline_sha=None
    if 'physmani' in methods:
        baseline_path=(proof.root/config['baseline_checkpoint']).resolve()
        baseline_sha=file_sha(proof.locked(baseline_path,files))
    for job in plan['jobs']:
        path=output/'results'/(job['job_id']+'.json');record=proof.read(path);validate_completed_case(record,job['identity'])
        parent=record.get('parent_case_process_receipt',{});child_path=(proof.root/parent['child_path']).resolve()
        require(child_path.name=='case.json' and child_path.is_relative_to(output/'cases'/job['job_id'])
            and child_path not in children, 'raw child is not its unique planned task case')
        children.add(child_path);child=proof.read(child_path,parent['child_sha256'])
        require(child=={k:v for k,v in record.items() if k!='parent_case_process_receipt'}
            and parent.get('child_process_id')==record.get('process_id')
            and type(parent.get('launcher_process_id')) is int and parent['launcher_process_id']>0
            and parent.get('plan_sha256')==plan['sha256']
            and parent.get('parent_source_sha256')==plan['parent_source']['sha256']
            and parent.get('returncode',0)==0, 'task raw parent/source/PID/returncode differs')
        method=job['identity']['method'];launcher=('scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh')
        if contract is not None:launcher=contract['launchers']['physmani' if method=='physmani' else 'ours']['path']
        command=['bash',launcher,plan['child_source']['path'],'--config',plan['config']['path'],
            '--manifest',plan['manifest']['path'],'--frozen-inputs',plan['frozen_inputs']['path'],
            '--split','validation','--task',task,'--raw-episode-id',job['source_raw_episode_id'],
            '--method',method,'--replica','0','--out',str(child_path.parent.relative_to(proof.root))]
        require(parent.get('command')==command,'task raw launcher command differs')
        proof.file(parent['log_path'],parent['log_sha256'])
        original=contract['original_evaluator'] if contract is not None else plan['child_source']
        require(record.get('cleanup_completed') is True and record.get('evaluation_process_pid')==record['process_id']
            and record.get('entrypoint_sha256')==original['sha256'] and record.get('config_sha256')==plan['config']['sha256']
            and record.get('manifest_sha256')==plan['manifest']['sha256']
            and record.get('source_row_sha256')==job['identity']['source_row_sha256']
            and record.get('frozen_input_lock',{}).get('sha256')==plan['frozen_inputs']['sha256']
            and (proof.root/record.get('frozen_input_lock',{}).get('path','')).resolve()
                ==(proof.root/plan['frozen_inputs']['path']).resolve()
            and record.get('demo_only') is False
            and record.get('video') is None and record.get('recorded_video_in_formal_metrics') is False,
            'task raw cleanup/config/lock/demo identity differs')
        if contract is not None:
            witness=record.get('task_execution_witness',{})
            require(set(witness)=={'schema','task','original_evaluator','actual_wrapper','actual_process_id','split',
                'coverage_certificate','task_validation_release','base_evaluator_receipt','hook_receipt'}
                and witness['schema']==EXECUTION and witness['task']==task
                and witness['original_evaluator']==original and witness['actual_wrapper']==contract['wrapper']
                and witness['actual_process_id']==record['process_id'] and witness['split']=='validation'
                and witness['coverage_certificate']==config.get('task_model_coverage_certificate')
                and witness['task_validation_release'] is None,'task wrapper witness is missing or inconsistent')
            base_path=proof.pointer(witness['base_evaluator_receipt'])
            require(base_path.parent==child_path.parent and base_path!=child_path,'task base receipt is not a separate same-case raw file')
            base=proof.read_pointer(witness['base_evaluator_receipt'])
            require(base=={k:v for k,v in child.items() if k!='task_execution_witness'},'task wrapper rewrote the actual base evaluator record')
            hook_path=proof.pointer(witness['hook_receipt'])
            require(hook_path.parent==child_path.parent,'task hooks are not bound to the same raw case directory')
            hooks=proof.read_pointer(witness['hook_receipt'])
            require(hooks.get('version')=='r10_trained_task_fresh_case_wrapper_v1'
                and hooks.get('identity')==job['identity'] and hooks.get('actual_wrapper')==contract['wrapper']
                and hooks.get('actual_process_id')==record['process_id'] and hooks.get('split')=='validation'
                and hooks.get('shared_development_hooks_applied') is True and hooks.get('complete') is True
                and hooks.get('error') is None and hooks.get('original_TEST_reset_unmodified') is False
                and hooks.get('source_resampled') is False and hooks.get('automatic_retry') is False
                and hooks.get('policy_failure_label_assigned') is False
                and hooks.get('permissions')=={key:config.get(key) for key in ('development_reset_profile','RML_factory_hook')},
                'task shared physical execution hook receipt is incomplete or inconsistent')
            reset=hooks.get('reset_recovery_receipts',[])
            require(len(reset)==1 and isinstance(reset[0],dict) and reset[0].get('complete') is True
                and reset[0].get('permission')==config.get('development_reset_profile')
                and reset[0].get('task')==task and reset[0].get('split')=='validation'
                and reset[0].get('replica')==0 and reset[0].get('source_resampled') is False
                and reset[0].get('source',{}).get('raw_episode_id')==job['source_raw_episode_id'],
                'task shared reset receipt does not belong to its actual source')
            require(all(isinstance(row,dict) and row.get('close_complete') is True and not row.get('close_errors')
                for row in hooks.get('RML_path_recovery_receipts',[])),'task RML cleanup is incomplete')
        else:require(record.get('task_execution_witness') is None,'undeclared wrapper witness')
        require(all(isinstance(record.get(name),(int,float)) and np.isfinite(record[name]) and record[name]>=0
            for name in ('compute_seconds','wall_seconds','simulation_seconds'))
            and abs(record['simulation_seconds']-record['counted_clock']['seconds'])<=2e-5,'task clock is invalid')
        if method=='physmani':
            require(record.get('checkpoint_sha256')==baseline_sha and record.get('baseline_executor')==config['baseline_executor'],
                'task baseline checkpoint/executor differs')
        else:
            predictor=method.split('_')[0];expected_continuation=config.get('continuation_identities',{}).get(task,{}).get(predictor)
            require(expected_continuation is not None and record.get('predictor')==predictor
                and record.get('continuation_sha256')==record.get('continuation_identity',{}).get('sha256')==expected_continuation
                and record.get('interaction_library_sha256')==config['learned_artifacts'][task]['interaction']['sha256']
                and record.get('unexecuted_candidates_labelled') is False
                and record.get('first_selection_policy')==('value_once' if method.endswith('_value') else 'geometry')
                and record.get('downstream_selection_policy')=='geometry','task continuation/model/value-once semantics differ')
        key=(task,job['source_raw_episode_id']);require(method not in groups[key],'duplicate actual task method')
        groups[key][method]=record
        receipts.append(dict(task=task,source_raw_episode_id=key[1],method=method,path=proof.name(path),sha256=file_sha(path),
            raw_child_path=proof.name(child_path),raw_child_sha256=parent['child_sha256'],identity_sha256=job['identity']['sha256']))
    require(set(groups)==set(canonical) and all(set(records)==set(methods) for records in groups.values()),
        'task five-way actual source coverage missing')
    return groups,receipts,status


def _finalize_task_validation_release(root,task,queue_plan,coverage_certificate,out,*,
        canonical_validation_manifest=DEFAULT_VAL,experiment_plan=DEFAULT_EXPERIMENT,composition=None):
    _task(task);proof=Proof(root);out=(proof.root/out).resolve();stage='canonical_inventory'
    require(out.is_relative_to(proof.root/'artifacts/framework/multitask_r10') and not out.exists(),'fresh task release path required')
    try:
        canonical,inventory=_canonical_task(proof,canonical_validation_manifest,'validation',task)
        component_spec=None
        if composition is not None:
            stage='explicit_task_composition';component_spec=proof.read_pointer(composition)
            require(set(component_spec)=={'schema','task','ours_queue_plan','baseline_early_queue_plan',
                'baseline_execution_contract','canonical_validation_manifest','raw_identities_rewritten',
                'expected_source_cases','expected_method_cases','sha256'}
                and component_spec['schema']==COMPOSITION and component_spec['task']==task
                and component_spec['sha256']==digest({k:v for k,v in component_spec.items() if k!='sha256'})
                and component_spec['raw_identities_rewritten'] is False
                and component_spec['expected_source_cases']==len(canonical)
                and component_spec['expected_method_cases']==5*len(canonical),
                'explicit task baseline composition coverage/identity differs')
            proof.pointer(component_spec['canonical_validation_manifest'])
            rows,_=_canonical_task(proof,proof.pointer(component_spec['canonical_validation_manifest']),'validation',task)
            require(rows==canonical,'task baseline composition changed canonical source rows')
            queue_plan=proof.pointer(component_spec['ours_queue_plan'])
        stage='task_queue_plan';plan=_plan(proof,queue_plan,canonical,task,
            methods=METHODS[:-1] if component_spec is not None else METHODS)
        config=proof.read_pointer(plan['config']);plan['_config']=config
        lock=proof.read_pointer(plan['frozen_inputs']);files=models.normalized_lock(proof.root,lock['files']);proof.include(files)
        for field in ('config','manifest','parent_source','child_source'):proof.locked(plan[field]['path'],files,plan[field]['sha256'])
        stage='task_model_coverage';coverage=proof.read_pointer(coverage_certificate);proof.include(coverage['input_files_sha256'])
        coverage_receipt=require_task_coverage_certificate(proof.root,task,coverage_certificate,config,files)
        require(config.get('task_model_coverage_certificate')==coverage_certificate,
            'task frozen VAL config must bind its exact model certificate before execution')
        stage='actual_five_way_VAL';groups,receipts,status=_cases(proof,plan,task,canonical,files)
        component=None
        if component_spec is not None:
            # Fresh full validator, never a cached component boolean. It keeps
            # each baseline's actual earlier config/lock and strict projection.
            from .baseline_component import validate_baseline_component
            from .composite_validation_release import _baseline_groups
            stage='fresh_full_baseline_component'
            component=validate_baseline_component(proof.root,component_spec['baseline_early_queue_plan'],
                component_spec['baseline_execution_contract'],plan['config'],plan['frozen_inputs'],
                component_spec['canonical_validation_manifest'])
            all_rows=proof.read_pointer(component_spec['canonical_validation_manifest'])
            all_canonical={(row['task'],row['raw_episode_id']):row for row in all_rows['episodes'] if row['task'] in TASK_SPECS}
            baseline,baseline_receipts=_baseline_groups(proof,component,all_canonical)
            for key in canonical:
                require(key in baseline and set(baseline[key])=={'physmani'} and 'physmani' not in groups[key],
                    'task baseline is missing or duplicated')
                groups[key].update(baseline[key])
            receipts.extend(row for row in baseline_receipts if row['task']==task)
            early_lock=proof.read_pointer(component['actual_early_lock']);proof.include(early_lock['files'])
            proof.read_pointer(component['actual_early_config']);proof.read_pointer(component['actual_early_queue_plan'])
            require(len(receipts)==5*len(canonical) and all(set(rows)==set(METHODS) for rows in groups.values()),
                'task component composition lacks complete actual five-way coverage')
        stage='TRAIN_tolerances';profile=proof.read_pointer(config['initial_condition_tolerance_profile'])
        proof.locked(config['initial_condition_tolerance_profile']['path'],files,config['initial_condition_tolerance_profile']['sha256'])
        require(not profile_problems(profile,evidence_root=proof.root),'task TRAIN reset profile is not valid/frozen')
        proof.pointer(profile['TRAIN_evidence']);comparisons=[]
        stage='recorded_condition_parity'
        for key,records in groups.items():
            comparison=compare_five_way(records,profile,evidence_root=proof.root,require_frozen=True)
            require(comparison.get('complete') is True and comparison.get('accepted') is True
                and len(comparison.get('checks',[]))==10,'task recorded initial-condition parity failed: '+repr(key))
            comparisons.append(dict(task=task,parent_raw_episode_id=key[1],**comparison))
        stage='selection_trace';usage=audit_selection_usage(groups)
        require(usage.get('complete') is True and usage.get('diagnostics_valid') is True,'task selection trace is incomplete')
        stage='original_stopping_rule';experiment=proof.read(experiment_plan);proof.locked(experiment_plan,files)
        stop=experiment.get('stopping_rule',{})
        require(experiment.get('remaining_tasks')==14 and experiment.get('methods')==list(METHODS)
            and experiment.get('original_sources',{}).get('validation')==275
            and experiment.get('original_sources',{}).get('official_test')==560
            and stop.get('preregistered_before_remaining14_VAL_and_TEST') is True
            and stop.get('deployment_or_success_module_benefit_assumed') is False
            and stop.get('performance_screen_failure'),'original complete experiment/stopping rule missing')
        directory=proof_directory(out);pointers={}
        artifacts={'canonical_cases':dict(entries=receipts,complete=True,expected=5*len(canonical)),
                'recorded_initial_conditions':dict(entries=comparisons,complete=True,accepted=True,sources=len(canonical),pairs=10*len(canonical)),
                'selection_usage':usage}
        if component is not None:artifacts['baseline_component']=component
        for name,value in artifacts.items():
            path=directory/(name+'.json');atomic_json(path,value);proof.file(path);pointers[name]=dict(path=proof.name(path),sha256=file_sha(path))
        proof.file(Path(__file__),file_sha(Path(__file__)))
        proof.file(Path(models.__file__),file_sha(Path(models.__file__)))
        proof.include(dict(proof.files))
        performance={method:dict(successes=sum(rows[method]['official_task_success'] for rows in groups.values()),episodes=len(groups)) for method in METHODS}
        for row in performance.values():row['success_rate']=row['successes']/row['episodes']
        release=dict(schema=RELEASE,task=task,complete=True,protocol_acceptance_passed=True,formal_test_permitted=True,
            split='validation',TEST_used=False,actual_R10_TEST_read=False,full_experiment_tasks=list(TASK_SPECS),
            expected_source_cases=len(canonical),expected_method_cases=5*len(canonical),completed_method_cases=5*len(canonical),
            task_comparison_sha256=task_comparison_sha256(config),validated_config=plan['config'],validated_input_lock=plan['frozen_inputs'],
            queue_plan=dict(path=proof.name(queue_plan),sha256=file_sha(proof.root/queue_plan)),
            model_coverage_certificate=coverage_certificate,model_coverage_receipt=coverage_receipt,
            canonical_validation_inventory=inventory,initial_condition_tolerance_profile=config['initial_condition_tolerance_profile'],
            TRAIN_repeat_evidence=profile['TRAIN_evidence'],preregistered_experiment=dict(path=proof.name(experiment_plan),sha256=file_sha(proof.root/experiment_plan)),
            **pointers,performance=performance,performance_does_not_override_protocol=True,
            post_VAL_threshold_invented=False,success_rate_benefit_validated=False,
            coverage_source_sha256=file_sha(Path(__file__)),input_files_sha256=proof.files,
            native_or_GPU_launched=False,model_fitting_performed=False,other_tasks_released=False)
        release.update(single_full_config_identity_claimed=component is None,raw_case_identities_rewritten=False)
        if component is not None:
            release.update(explicit_VAL_composition=composition,
                actual_baseline_queue_plan=component['actual_early_queue_plan'],
                actual_baseline_config=component['actual_early_config'],actual_baseline_lock=component['actual_early_lock'],
                baseline_dependency_projection_revalidated=True)
        atomic_json(out,release);return release
    except Exception as error:
        pending=dict(schema=RELEASE,task=task,complete=False,protocol_acceptance_passed=False,formal_test_permitted=False,
            stage=stage,errors=[repr(error)],TEST_used=False,actual_R10_TEST_read=False,no_release_written=True,
            native_or_GPU_launched=False,model_fitting_performed=False)
        directory=out.parent/'pending_task_release';directory.mkdir(parents=True,exist_ok=True);number=0
        while (directory/(out.stem+f'__{number:04d}.json')).exists():number+=1
        path=directory/(out.stem+f'__{number:04d}.json');atomic_json(path,pending)
        pending['pending_report']=dict(path=proof.name(path),sha256=file_sha(path));return pending


def finalize_task_validation_release(root,task,queue_plan,coverage_certificate,out,*,
        canonical_validation_manifest=DEFAULT_VAL,experiment_plan=DEFAULT_EXPERIMENT):
    return _finalize_task_validation_release(root,task,queue_plan,coverage_certificate,out,
        canonical_validation_manifest=canonical_validation_manifest,experiment_plan=experiment_plan)


def finalize_task_composite_validation_release(root,task,composition,coverage_certificate,out,*,
        canonical_validation_manifest=DEFAULT_VAL,experiment_plan=DEFAULT_EXPERIMENT):
    """Four ours task cases plus a freshly verified full275 baseline component.

    Partial early queues or unknown config differences remain rejected by the
    original baseline validator. No subset component or source rewrite exists.
    """
    return _finalize_task_validation_release(root,task,None,coverage_certificate,out,
        canonical_validation_manifest=canonical_validation_manifest,experiment_plan=experiment_plan,composition=composition)


def require_task_test_release(root,task,release_pointer,protocol,frozen_files,*,official_test_manifest=DEFAULT_TEST):
    _task(task);proof=Proof(root);report=proof.read_pointer(release_pointer);normalized=models.normalized_lock(proof.root,frozen_files)
    require(report.get('schema')==RELEASE and report.get('task')==task and report.get('complete') is True
        and report.get('protocol_acceptance_passed') is True and report.get('formal_test_permitted') is True
        and report.get('split')=='validation' and report.get('TEST_used') is False and report.get('actual_R10_TEST_read') is False
        and report.get('full_experiment_tasks')==list(TASK_SPECS)
        and report.get('coverage_source_sha256')==file_sha(Path(__file__))
        and report.get('task_comparison_sha256')==task_comparison_sha256(protocol),
        'task TEST release is failed, stale, foreign or changed policy/config')
    canonical,inventory=_canonical_task(proof,DEFAULT_VAL,'validation',task);n=len(canonical)
    require(report.get('expected_source_cases')==n
        and report.get('expected_method_cases')==report.get('completed_method_cases')==5*n,
        'task TEST requires every original task source times all five VAL methods')
    validated=proof.read_pointer(report['validated_config']);old_lock=proof.read_pointer(report['validated_input_lock'])
    require(task_comparison_sha256(validated)==task_comparison_sha256(protocol),'task changed candidates/shared physics/models since VAL')
    old_files=models.normalized_lock(proof.root,old_lock['files'])
    require(all(normalized.get(name)==sha for name,sha in old_files.items()),'TEST omitted or changed a validated task input')
    require(normalized.get(proof.name(release_pointer['path']))==release_pointer['sha256']
        and report.get('input_files_sha256')
        and all(normalized.get(proof.name(name))==sha for name,sha in report['input_files_sha256'].items()),
        'TEST lock does not bind task release and every original proof input')
    coverage=require_task_coverage_certificate(proof.root,task,report['model_coverage_certificate'],protocol,frozen_files)
    test,test_inventory=_canonical_task(proof,official_test_manifest,'test',task)
    proof.locked(official_test_manifest,normalized)
    require(len(test)==40,'task official TEST denominator is not forty')
    for key in ('canonical_cases','recorded_initial_conditions','selection_usage'):
        proof.pointer(report[key]);proof.locked(report[key]['path'],normalized,report[key]['sha256'])
    return dict(passed=True,task=task,release=release_pointer,model_coverage=coverage,
        VAL_source_cases=n,VAL_method_cases=5*n,TEST_source_cases=40,TEST_method_cases=200,
        official_test_inventory=test_inventory,other_tasks_released=False,TEST_outcomes_read=False)
