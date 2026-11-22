"""Version2 full-five task VAL release with a strict task-local baseline.

The old global gate and task_release v1 remain frozen. This version explicitly
binds the runtime wrapper schema to its real source and distinguishes a skill
library's semantic identity from its file checksum. All five methods of this
task must finish; no method-only TEST admission exists.
"""
import ast
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .evaluation import METHODS,atomic_json,digest,file_sha,validate_completed_case
from . import release_preflight as models
from . import task_release as previous
from .task_release import (_task,_canonical_task,_plan,task_comparison_sha256,
    require_task_coverage_certificate,DEFAULT_VAL,DEFAULT_TEST,DEFAULT_EXPERIMENT,EXECUTION)
from .baseline_task_component import validate_baseline_task_component,COMPONENT as BASELINE_COMPONENT
from . import gpu_budget_audit
from .interaction import InteractionLibrary
from .reset_parity import compare_five_way,profile_problems
from .specs import TASK_SPECS
from .trace_evidence import audit_selection_usage
from .validation_release import Proof,proof_directory,require

RELEASE='r10_task_complete_five_method_validation_release_v2_subset_baseline'
COMPOSITION='r10_task_four_ours_plus_same_config_subset_baseline_VAL_composition_v2'
REVIEWED_WRAPPERS={
    'r10_trained_task_fresh_case_wrapper_v1':'scripts/run_trained_task_case_r10.py',
    'r10_trained_task_fresh_case_wrapper_v2_first_actionable_value':'scripts/run_trained_task_case_v2_r10.py',
    'r10_trained_task_formal_case_wrapper_v3':'scripts/run_task_formal_case_r10.py',
}


def _execution_contract(proof,plan,files,task):
    value=plan.get('task_execution_contract')
    require(isinstance(value,dict) and set(value)=={'schema','task','original_evaluator','wrapper','wrapper_schema_version','launchers'}
        and value['schema']==EXECUTION and value['task']==task and value['wrapper']==plan['child_source']
        and set(value['launchers'])=={'ours','physmani'},'version2 task wrapper execution contract differs')
    for pointer in [value['original_evaluator'],value['wrapper'],*value['launchers'].values()]:
        proof.locked(pointer['path'],files,pointer['sha256'])
    wrapper=proof.pointer(value['wrapper']);versions=[]
    for node in ast.parse(wrapper.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='VERSION' for target in node.targets):
            versions.append(ast.literal_eval(node.value))
    expected=value['wrapper_schema_version']
    require(versions==[expected] and expected in REVIEWED_WRAPPERS
        and wrapper==(proof.root/REVIEWED_WRAPPERS[expected]).resolve(),
        'task wrapper schema tag is not its exact reviewed actual source VERSION')
    return value


def _baseline_groups(proof,component,canonical,task):
    require(component.get('schema')==BASELINE_COMPONENT and component.get('complete') is True
        and component.get('task')==task and component.get('baseline_task_component_reusable') is True
        and component.get('sources')==len(canonical) and component.get('full_plan_sources')==275
        and component.get('raw_identity_rewritten') is False and component.get('raw_receipts_rewritten') is False
        and component.get('single_full_config_identity_claimed') is True
        and component.get('single_lock_identity_claimed') is False,
        'complete same-config actual task baseline component required')
    groups={};receipts=[]
    for item in component.get('completed_cases',[]):
        key=(item['task'],item['raw_episode_id'])
        require(key in canonical and key not in groups,'baseline task source is foreign or duplicated')
        record=proof.read_pointer(item['canonical_receipt']);validate_completed_case(record,item['actual_identity'])
        child=proof.read_pointer(item['raw_child_receipt'])
        require(child=={k:v for k,v in record.items() if k!='parent_case_process_receipt'}
            and record['method']=='physmani' and record['split']=='validation'
            and item['actual_identity']['source_row_sha256']==digest(canonical[key]),'task baseline raw identity was rewritten')
        parent=record['parent_case_process_receipt'];proof.file(parent['log_path'],parent['log_sha256'])
        groups[key]={'physmani':record}
        receipts.append(dict(task=task,parent_raw_episode_id=key[1],method='physmani',**item['canonical_receipt'],
            raw_child_path=item['raw_child_receipt']['path'],raw_child_sha256=item['raw_child_receipt']['sha256'],
            identity_sha256=item['actual_identity']['sha256'],actual_config=component['actual_config'],
            actual_lock=component['actual_lock'],actual_queue_plan=component['actual_queue_plan']))
    require(set(groups)==set(canonical) and len(receipts)==len(canonical),'task baseline complete original coverage missing')
    proof.include(component['verified_files_sha256']);proof.include(component['execution_dependency_files_sha256'])
    return groups,receipts


def _resource_budget(proof,config,component,spec):
    if config.get('GPU_execution_resource_budget') is None:
        require(spec.get('resource_budget_receipt') is None,'undeclared GPU resource receipt')
        return None
    budget=config['GPU_execution_resource_budget']
    require(isinstance(budget,dict) and set(budget)=={'profile','environment_contract'}
        and spec.get('resource_budget_receipt') is not None,
        'declared actual GPU resource budget requires its separately verified source-bound receipt')
    pids=[]
    for item in component['completed_cases']:
        record=proof.read_pointer(item['raw_child_receipt']);pid=record.get('process_id')
        require(type(pid) is int and pid>0,'baseline task actual native PID missing')
        pids.append(pid)
    require(len(pids)==component['sources'] and len(set(pids))==len(pids),
        'baseline task budget must cover every unique actual native process')
    result=gpu_budget_audit.validate_budget_execution(proof.root,spec['resource_budget_receipt'],
        budget['profile'],required_case_process_ids=pids)
    require(result.get('valid') is True
        and proof.pointer(result['environment_contract'])==proof.pointer(budget['environment_contract'])
        and result['environment_contract']['sha256']==budget['environment_contract']['sha256'],
        'actual resource environment contract differs from common configuration')
    # Freeze the exact historical witness chain, including external execution
    # binaries already explicitly hash-bound by the resource validator. This
    # has no live process probe and never launches CUDA or the simulator.
    visited=set()
    def include(value):
        if isinstance(value,list):
            for item in value:include(item)
        elif isinstance(value,dict):
            if {'path','sha256'}<=set(value):
                path=proof.pointer(value)
                if path not in visited:
                    visited.add(path)
                    if path.suffix=='.json':include(json.loads(path.read_text()))
            else:
                for item in value.values():include(item)
    include(spec['resource_budget_receipt']);include(budget)
    proof.file(gpu_budget_audit.__file__)
    proof.file('our_mani/multitask/gpu_budget.py')
    return result


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
    library_pointer=config['learned_artifacts'][task]['interaction']
    library=InteractionLibrary.load(proof.locked(library_pointer['path'],files,library_pointer['sha256']))
    require(library.task_spec.task_name==task,'actual library belongs to a different task')
    if 'physmani' in methods:
        baseline_path=(proof.root/config['baseline_checkpoint']).resolve()
        baseline_sha=file_sha(proof.locked(baseline_path,files))
    for job in plan['jobs']:
        path=output/'results'/(job['job_id']+'.json');record=proof.read(path);validate_completed_case(record,job['identity'])
        parent=record.get('parent_case_process_receipt',{});child_path=(proof.root/parent['child_path']).resolve()
        require(child_path==output/'cases'/job['job_id']/'attempt0000'/'case.json'
            and {p.name for p in child_path.parent.parent.glob('attempt[0-9]*') if p.is_dir()}=={'attempt0000'}
            and child_path not in children, 'raw child is not its unique planned task case')
        children.add(child_path);child=proof.read(child_path,parent['child_sha256'])
        require(child=={k:v for k,v in record.items() if k!='parent_case_process_receipt'}
            and parent.get('child_process_id')==record.get('process_id')
            and type(parent.get('launcher_process_id')) is int and parent['launcher_process_id']>0
            and parent.get('plan_sha256')==plan['sha256']
            and parent.get('parent_source_sha256')==plan['parent_source']['sha256']
            and parent.get('returncode')==0, 'task raw parent/source/PID/returncode differs')
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
            require(hooks.get('version')==contract['wrapper_schema_version']
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
                and record.get('interaction_library_sha256')==library.sha256
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
            require(set(component_spec)=={'schema','task','ours_queue_plan','baseline_queue_plan',
                'baseline_execution_contract','canonical_validation_manifest','raw_identities_rewritten',
                'expected_source_cases','expected_method_cases','resource_budget_receipt','sha256'}
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
            stage='fresh_task_baseline_component'
            component=validate_baseline_task_component(proof.root,task,component_spec['baseline_queue_plan'],
                component_spec['baseline_execution_contract'],plan['config'],plan['frozen_inputs'],
                component_spec['canonical_validation_manifest'])
            baseline,baseline_receipts=_baseline_groups(proof,component,canonical,task)
            for key in canonical:
                require('physmani' not in groups[key],'duplicate baseline method')
                groups[key].update(baseline[key])
            receipts.extend(baseline_receipts)
            proof.read_pointer(component['actual_config']);proof.read_pointer(component['actual_lock'])
            proof.read_pointer(component['actual_queue_plan'])
            stage='actual_resource_budget'
            budget=_resource_budget(proof,config,component,component_spec)
            require(len(receipts)==5*len(canonical) and all(set(rows)==set(METHODS) for rows in groups.values()),
                'task baseline composition lacks complete actual five-way coverage')
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
        proof.file(Path(previous.__file__),file_sha(Path(previous.__file__)))
        proof.file('our_mani/multitask/baseline_task_component.py')
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
        release.update(single_full_config_identity_claimed=True,raw_case_identities_rewritten=False,
            model_coverage_consumer_source=dict(path=str(Path(previous.__file__).resolve()),sha256=file_sha(previous.__file__)))
        if component is not None:
            release.update(explicit_VAL_composition=composition,
                actual_baseline_queue_plan=component['actual_queue_plan'],
                actual_baseline_config=component['actual_config'],actual_baseline_lock=component['actual_lock'],
                baseline_dependency_projection_revalidated=True,resource_budget_receipt=budget,
                actual_baseline_lock_preserved=True,single_lock_identity_claimed=False)
        atomic_json(out,release);return release
    except Exception as error:
        pending=dict(schema=RELEASE,task=task,complete=False,protocol_acceptance_passed=False,formal_test_permitted=False,
            stage=stage,errors=[repr(error)],TEST_used=False,actual_R10_TEST_read=False,no_release_written=True,
            native_or_GPU_launched=False,model_fitting_performed=False)
        directory=out.parent/'pending_task_release';directory.mkdir(parents=True,exist_ok=True);number=0
        while (directory/(out.stem+f'__{number:04d}.json')).exists():number+=1
        path=directory/(out.stem+f'__{number:04d}.json');atomic_json(path,pending)
        pending['pending_report']=dict(path=proof.name(path),sha256=file_sha(path));return pending


def finalize_task_composite_validation_release(root,task,composition,coverage_certificate,out,*,
        canonical_validation_manifest=DEFAULT_VAL,experiment_plan=DEFAULT_EXPERIMENT):
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
    require(report.get('canonical_validation_inventory',{}).get('sha256')==inventory['sha256']
        and report.get('expected_source_cases')==n
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
