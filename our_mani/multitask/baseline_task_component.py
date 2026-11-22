"""Same-config PhysMani task VAL component from a partially finished queue.

The original full275 consumer and execution sources are unchanged. The full
canonical plan remains immutable; only one task's exact committed source set
is consumed. Its case kernel is a static copy, with AST parity checked against
the frozen original consumer on every audit. No code is dynamically executed,
no old module globals are patched, and no formal release is issued here.
"""
import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import math
from pathlib import Path

from . import baseline_component as original
from .baseline_component import (require,read,pointer,absolute,normalized,
    canonical_VAL_roster,compare_config_dependencies,verify_execution_contract,
    calibration_dependency,validate_virtual_owner_bindings,VERSION,runtime_state,jit_artifacts)
from .evaluation import digest,file_sha,case_identity,validate_completed_case
from .specs import TASK_SPECS

COMPONENT='r10_verified_same_config_PhysMani_task_VAL_component_v1'


def _case_kernel_proof():
    """The complete original per-job raw/physics/runtime kernel stays exact."""
    def loop(path,function):
        tree=ast.parse(Path(path).read_text())
        owner=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==function)
        return next(node for node in owner.body if isinstance(node,ast.For)
            and isinstance(node.target,ast.Name) and node.target.id=='job'
            and isinstance(node.iter,ast.Name) and node.iter.id=='jobs')
    before=ast.dump(loop(original.__file__,'validate_baseline_component'),include_attributes=False)
    after=ast.dump(loop(__file__,'_validate_selected_jobs'),include_attributes=False)
    require(before==after,'task baseline case kernel differs from the original frozen full consumer')
    return dict(original_consumer=dict(path=str(Path(original.__file__).resolve()),sha256=file_sha(original.__file__)),
        task_consumer=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)),
        shared_case_kernel_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        original_case_checks_weakened=False,dynamic_code_execution=False,original_module_patched=False)


def _same_pointer(root,left,right,label):
    require(isinstance(left,dict) and isinstance(right,dict)
        and absolute(root,left.get('path',''))==absolute(root,right.get('path',''))
        and left.get('sha256')==right.get('sha256'),label+' actual path/SHA differs; historical projection reuse is not allowed')


def _all_jobs(plan,roster):
    by_source={(row['task'],row['raw_episode_id']):row for row in roster}
    jobs=plan.get('jobs',[]);seen=set();ids=set()
    require(len(jobs)==275,'full baseline plan cannot silently delete unready tasks')
    for job in jobs:
        row=job['row'];key=(row['task'],row['raw_episode_id'])
        require(key in by_source and row==by_source[key] and key not in seen
            and job.get('source_raw_episode_id')==key[1] and job.get('replica')==0
            and job.get('exploration_seed') is None and job.get('record_demo') is False,
            'full baseline task plan source/replica/demo coverage differs')
        expected=case_identity(split='validation',task=key[0],raw_episode_id=key[1]+'__policy_aug_r0',
            method='physmani',source_row=row,config_sha256=plan['config']['sha256'],
            inputs_sha256=plan['frozen_inputs']['sha256'],parent_raw_episode_id=key[1],replica=0)
        require(job['identity']==expected and job['job_id']==expected['sha256'][:20] and job['job_id'] not in ids,
            'full baseline plan job identity is duplicated or changed')
        seen.add(key);ids.add(job['job_id'])
    require(seen==set(by_source),'baseline full plan lost a canonical original source')
    return jobs,by_source


def _validate_selected_jobs(root,plan,early_config,early_lock,projection,closure,execution_contract,jobs,by_source):
    seen=set();completed=[];archive=closure['dependencies'];output=absolute(root,plan['output'])
    for job in jobs:
        row = job['row']; key = (row['task'], row['raw_episode_id'])
        require(key in by_source and row == by_source[key] and key not in seen
            and job.get('source_raw_episode_id') == row['raw_episode_id']
            and job.get('replica') == 0 and job.get('exploration_seed') is None and job.get('record_demo') is False,
            'baseline VAL job roster/replica/demo differs')
        expected = case_identity(split='validation', task=key[0], raw_episode_id=key[1]+'__policy_aug_r0',
            method='physmani', source_row=row, config_sha256=plan['config']['sha256'],
            inputs_sha256=plan['frozen_inputs']['sha256'], parent_raw_episode_id=key[1], replica=0)
        require(job['identity'] == expected and job['job_id'] == expected['sha256'][:20], 'actual early job identity changed')
        case_path = output/'results'/(job['job_id']+'.json'); record = read(case_path)
        validate_completed_case(record, expected); parent = record.get('parent_case_process_receipt', {})
        child = absolute(root, parent['child_path'])
        require(child.name == 'case.json' and child.is_relative_to(output/'cases'/job['job_id']) and file_sha(child) == parent['child_sha256']
            and read(child) == {k: v for k, v in record.items() if k != 'parent_case_process_receipt'}
            and parent.get('plan_sha256') == plan['sha256']
            and parent.get('parent_source_sha256') == plan['parent_source']['sha256']
            and parent.get('child_process_id') == record['process_id']
            and type(parent.get('launcher_process_id')) is int and parent['launcher_process_id'] > 0,
            'raw child/parent/actual plan chain changed')
        command = parent.get('command', [])
        require(command.count('--config') == command.count('--frozen-inputs') == command.count('--manifest') == 1
            and command[command.index('--config')+1] == plan['config']['path']
            and command[command.index('--frozen-inputs')+1] == plan['frozen_inputs']['path']
            and command[command.index('--manifest')+1] == plan['manifest']['path'], 'actual historical command/config/lock changed')
        expected_command = ['bash', 'scripts/run_aligned_gpu.sh', plan['child_source']['path'],
            '--config', plan['config']['path'], '--manifest', plan['manifest']['path'],
            '--frozen-inputs', plan['frozen_inputs']['path'], '--split', 'validation',
            '--task', key[0], '--raw-episode-id', key[1], '--method', 'physmani', '--replica', '0',
            '--out', str(child.parent.relative_to(root))]
        if 'baseline_execution_contract' in plan:
            expected_command.insert(3, '--collect-case')
            require(plan['baseline_execution_contract'] == execution_contract,
                    'actual wrapper plan contract differs from the component contract')
            expected_command += ['--execution-contract', execution_contract['path']]
        require(command == expected_command, 'actual baseline command differs from its exact planned argv')
        for name in ('parent_source', 'child_source'):
            require(archive.get(str(absolute(root, plan[name]['path']))) == plan[name]['sha256'],
                    'actual evaluator/queue source is outside frozen execution archive')
        require(file_sha(absolute(root, parent['log_path'])) == parent['log_sha256'], 'original native log changed')
        require(record.get('video') is None and record.get('demo_only') is False
            and record.get('cleanup_completed') is True and record.get('formal_test_used') is False
            and record.get('baseline_executor') == early_config['baseline_executor']
            and record.get('checkpoint_sha256') == archive[str(absolute(root, early_config['baseline_checkpoint']))],
            'baseline VAL execution/checkpoint/demo/cleanup evidence differs')
        require(all(isinstance(record.get(name), (int, float)) and math.isfinite(record[name]) and record[name] >= 0
            for name in ('compute_seconds', 'wall_seconds', 'simulation_seconds'))
            and abs(record['simulation_seconds']-record['counted_clock']['seconds']) <= 2e-5
            and record.get('evaluation_process_pid') == record['process_id']
            and record.get('source_row_sha256') == expected['source_row_sha256'],
            'actual baseline runtime/source/pid measurement differs')
        witness = record['recorded_initial_conditions'].get('baseline_execution_dependency_witness', {})
        require(witness.get('version') == VERSION and witness.get('execution_contract') == execution_contract
            and witness.get('projected_config_sha256') == projection['projection_sha256']
            and witness.get('full_config_identity_preserved') is True and witness.get('complete') is True
            and witness.get('projection_helper_sha256') == archive[str(root/'our_mani/multitask/baseline_component.py')]
            and witness.get('calibration_binding_sha256') == digest(calibration_dependency(row))
            and witness.get('split') == 'validation' and witness.get('method') == 'physmani',
            'missing actual projected backend execution witness')
        reset = record['recorded_initial_conditions']['reset_receipt'].get('reset_recovery_receipt')
        require(isinstance(reset, dict) and reset == witness.get('shared_reset_receipt')
            and reset.get('complete') is True and reset.get('permission') == early_config['development_reset_profile']
            and reset.get('source_resampled') is False and reset.get('official_TEST_allowed') is False,
            'missing actual shared same-source development reset witness')
        require(record.get('config_sha256') == plan['config']['sha256']
            and record.get('manifest_sha256') == plan['manifest']['sha256']
            and record.get('frozen_input_lock') == dict(path=str(absolute(root, plan['frozen_inputs']['path'])),
                sha256=plan['frozen_inputs']['sha256'])
            and record.get('entrypoint_sha256') == (archive.get(str(root/'scripts/evaluate_multitask_case_r10.py'))
                if 'baseline_execution_contract' in plan else plan['child_source']['sha256']),
            'actual early full config/lock/source identity was rewritten')
        if 'baseline_execution_contract' in plan:
            wrapper = record.get('baseline_projected_wrapper_execution', {})
            require(wrapper.get('version') == 'r10_actual_projected_PhysMani_VAL_wrapper_v1'
                and wrapper.get('execution_contract') == execution_contract
                and wrapper.get('actual_wrapper') == plan['child_source']
                and wrapper.get('evaluator_source') == dict(path='scripts/evaluate_multitask_case_r10.py',
                    sha256=archive.get(str(root/'scripts/evaluate_multitask_case_r10.py')))
                and wrapper.get('full_config_identity_preserved') is True
                and wrapper.get('original_record_fields_rewritten') is False
                and wrapper.get('projected_config_sha256') == projection['projection_sha256'],
                'actual projected wrapper and original evaluator identities are missing or rewritten')
            base = pointer(root, wrapper['base_run_case_receipt'])
            inputs_path = pointer(root, wrapper['base_run_case_inputs'])
            require(base == child.parent/'evaluate_run_case_receipt.json' and inputs_path == child.parent/'inputs.json'
                and read(base) == {k:v for k,v in record.items()
                    if k not in ('parent_case_process_receipt','baseline_projected_wrapper_execution')},
                'wrapper changes original evaluator receipt or its actual input provenance')
            inputs = read(inputs_path)
            require(inputs.get('identity') == expected and inputs.get('process_id') == record['process_id']
                and inputs.get('frozen_input_lock') == record['frozen_input_lock']
                and inputs.get('entrypoint_sha256') == record['entrypoint_sha256']
                and inputs.get('manifest_sha256') == plan['manifest']['sha256']
                and inputs.get('original_manifest_row_sha256') == digest(job['row'])
                and normalized(root, inputs.get('files', {})) == normalized(root, early_lock['files'])
                and inputs.get('argv') == [plan['child_source']['path'], *command[3:]],
                'wrapper original evaluator inputs do not bind actual full config/lock/argv/process')
            final_witness = wrapper.get('dependency_witness_after_cleanup', {})
            dynamic = {'imported_dependency_files_sha256','loaded_native_files_sha256','actual_python_executable',
                       'virtual_module_owner_bindings','generated_python_module_bindings','generated_native_file_bindings'}
            require({k:v for k,v in final_witness.items() if k not in dynamic}
                == {k:v for k,v in witness.items() if k not in dynamic}
                and final_witness.get('actual_python_executable') == witness.get('actual_python_executable'),
                'wrapper post-cleanup dependency observer changed reset or execution evidence')
            generated={}
            profile_pointer=closure['contract'].get('jit_artifact_profile')
            if profile_pointer is not None:
                require(witness.get('jit_artifact_profile')==profile_pointer
                    and final_witness.get('jit_artifact_profile')==profile_pointer,'actual JIT profile differs from contract')
                profile=jit_artifacts.load_profile(root,profile_pointer,archive)
                execution=wrapper.get('jit_artifact_execution')
                require((root/execution['path']).resolve()==child.parent/'jit_artifacts/execution.json',
                    'JIT execution receipt escaped actual child namespace')
                generated=jit_artifacts.validate_observed_bindings(root,execution,profile,archive,final_witness,
                    identity=expected,process_id=record['process_id'])
                jit_artifacts.validate_observed_bindings(root,execution,profile,archive,witness,
                    identity=expected,process_id=record['process_id'])
                for field in ('generated_python_module_bindings','generated_native_file_bindings'):
                    require(all(final_witness.get(field,{}).get(key)==value for key,value in witness.get(field,{}).items()),
                        'generated loader binding changed during cleanup')
            else:
                require(wrapper.get('jit_artifact_execution') is None
                    and not witness.get('generated_python_module_bindings') and not witness.get('generated_native_file_bindings')
                    and not final_witness.get('generated_python_module_bindings') and not final_witness.get('generated_native_file_bindings'),
                    'case claims undeclared generated code/native authority')
            for field in ('imported_dependency_files_sha256','loaded_native_files_sha256'):
                final_files = normalized(root, final_witness.get(field, {}))
                require(all(final_files.get(name) == checksum for name,checksum in witness.get(field, {}).items())
                    and all(archive.get(name) == checksum or generated.get(name)==checksum for name,checksum in final_files.items()),
                    'wrapper actual post-cleanup import/native closure differs or is unbound')
            validate_virtual_owner_bindings(final_witness.get('virtual_module_owner_bindings', {}), archive,
                final_witness.get('imported_dependency_files_sha256', {}))
            require(all(final_witness.get('virtual_module_owner_bindings', {}).get(key) == value
                for key, value in witness.get('virtual_module_owner_bindings', {}).items()),
                'wrapper post-cleanup virtual namespace owners changed')
            if closure['contract'].get('mutable_state_permission') is not None:
                runtime_state.validate_case_witness(wrapper.get('mutable_runtime_state_execution'),
                    closure['mutable_state_permission'],closure['contract']['mutable_state_permission'],
                    witness.get('mutable_runtime_state_initial'))
                runtime=read(pointer(root,closure['contract']['runtime_inventory']))
                runtime_state.validate_launcher_environment(witness.get('actual_launcher_environment'),archive,root,
                    early_config,runtime['interpreter'],jit_profile=profile if profile_pointer is not None else None)
            else:
                require(wrapper.get('mutable_runtime_state_execution') is None,
                        'case claims an undeclared mutable runtime exception')
        imports = normalized(root, witness.get('imported_dependency_files_sha256', {}))
        validate_virtual_owner_bindings(witness.get('virtual_module_owner_bindings', {}), archive, imports)
        generated=generated if 'baseline_execution_contract' in plan else {}
        require(all(archive.get(name) == checksum or generated.get(name)==checksum for name, checksum in imports.items()),
            'actual imported module was unbound')
        natives = normalized(root, witness.get('loaded_native_files_sha256', {}))
        require(witness.get('actual_python_executable') in natives
            and all(archive.get(name) == checksum or generated.get(name)==checksum for name, checksum in natives.items()),
            'actual interpreter/native mapped library was unbound')
        if closure['mutable_state_permission'] is not None:
            binary=str(Path(closure['mutable_state_permission']['resolved_path']).parent.parent/'libcoppeliaSim.so')
            require(natives.get(binary)==closure['mutable_state_permission']['source_bindings_sha256'][binary],
                    'actual loaded CoppeliaSim binary does not match the exact bookkeeping runtime')
        completed.append(dict(task=key[0], raw_episode_id=key[1], actual_identity=expected,
            canonical_receipt=dict(path=str(case_path.relative_to(root)), sha256=file_sha(case_path)),
            raw_child_receipt=dict(path=str(child.relative_to(root)), sha256=parent['child_sha256'])))
        seen.add(key)
    require(seen==set(by_source),'baseline task VAL source coverage incomplete')
    return completed


def validate_baseline_task_component(root,task,plan_pointer,execution_contract,
        common_config_pointer,common_lock_pointer,canonical_roster_pointer):
    """Audit an exact task without waiting for unrelated queue jobs.

    The caller must supply the same actual new common config, not an early12
    plan projected onto a later policy. The union common lock may additionally
    bind ours sources; the separate actual baseline lock is fully preserved.
    Full model/five-way parity and
    formal TEST authorization are the separate task release consumer's job.
    """
    root=Path(root).resolve();require(task in TASK_SPECS,'unknown remaining task')
    kernel=_case_kernel_proof()
    plan_path=pointer(root,plan_pointer);plan=read(plan_path)
    require(plan.get('version')=='r10_fresh_policy_queue_v1'
        and plan.get('sha256')==digest({k:v for k,v in plan.items() if k!='sha256'})
        and plan.get('split')=='validation' and plan.get('methods')==['physmani']
        and plan.get('replicas')==1 and plan.get('source_episodes')==plan.get('expected_cases')==275
        and plan.get('official_TEST_used_for_fitting') is False
        and plan.get('case_isolation')=='fresh_simulator_process_per_episode'
        and plan.get('baseline_execution_contract')==execution_contract,
        'same-config task component requires the original full275 baseline VAL plan')
    _same_pointer(root,plan['config'],common_config_pointer,'common config')
    early_config=read(pointer(root,plan['config']));early_lock=read(pointer(root,plan['frozen_inputs']))
    common_config=read(pointer(root,common_config_pointer));common_lock=read(pointer(root,common_lock_pointer))
    require(early_config==common_config,'same-config component cannot mix changed fields')
    roster=canonical_VAL_roster(read(pointer(root,canonical_roster_pointer)))
    require(canonical_VAL_roster(read(pointer(root,plan['manifest'])))==roster,'baseline canonical source rows changed')
    all_jobs,all_sources=_all_jobs(plan,roster)
    jobs=[job for job in all_jobs if job['row']['task']==task]
    by_source={key:row for key,row in all_sources.items() if key[0]==task};expected=len(by_source)
    require(expected in (18,19,20) and len(jobs)==expected,'actual task VAL denominator differs')
    projection=compare_config_dependencies(early_config,common_config)
    early_files=normalized(root,early_lock['files'])
    for field in ('config','manifest','parent_source','child_source'):
        require(early_files.get(str(absolute(root,plan[field]['path'])))==plan[field]['sha256'],
            'actual task config/roster/executor/queue was not frozen: '+field)
    closure=verify_execution_contract(root,execution_contract,early_config,early_lock,roster)
    entrypoints=closure['contract']['execution_entrypoints']
    require(all(plan[field]==entrypoints[field] for field in ('parent_source','child_source'))
        and absolute(root,entrypoints['launcher_source']['path'])==root/'scripts/run_aligned_gpu.sh',
        'task actual evaluator/queue/launcher differs from frozen reviewed sources')
    require(datetime.fromisoformat(closure['contract']['frozen_utc'])<=datetime.fromisoformat(plan['created_utc']),
        'execution contract was not frozen before queue execution')
    final_files=normalized(root,common_lock['files'])
    require(all(final_files.get(name)==checksum for name,checksum in early_files.items())
        and final_files.get(str(absolute(root,plan['frozen_inputs']['path'])))==plan['frozen_inputs']['sha256'],
        'common union lock must bind every actual baseline input and its separate actual lock bytes')
    for name,checksum in closure['dependencies'].items():
        require(final_files.get(name)==checksum and file_sha(name)==checksum,'task baseline execution closure changed: '+name)
    original_source=str((root/'our_mani/multitask/baseline_component.py').resolve())
    require(closure['dependencies'].get(original_source)==kernel['original_consumer']['sha256'],
        'actual frozen baseline consumer source differs from the shared case kernel')
    output=absolute(root,plan['output'])
    require(output.is_relative_to(root/'artifacts/framework/multitask_r10'),'task baseline output escaped project')
    status_path=output/'status.json';status=read(status_path)
    require(status.get('plan_sha256')==plan['sha256'] and status.get('expected')==275
        and type(status.get('completed')) is int and 0<=status['completed']<=275
        and status.get('stop_requested') is False,'baseline queue status source/budget/stop is invalid')
    names={p.name for p in (output/'results').glob('*.json')};known={job['job_id']+'.json' for job in all_jobs}
    target={job['job_id']+'.json' for job in jobs}
    require(names<=known and target<=names and status['completed']<=len(names),
        'missing task canonical receipt or foreign queue output')
    target_ids={job['job_id'] for job in jobs};all_ids={job['job_id'] for job in all_jobs}
    for error in status.get('errors',[]):
        require(isinstance(error,dict) and error.get('job_id') in all_ids
            and error['job_id'] not in target_ids,'task infrastructure failure or unidentifiable queue error remains')
    for job in jobs:
        directory=output/'cases'/job['job_id']
        require(not (output/'infrastructure_prelaunch'/(job['job_id']+'.json')).exists()
            and not (directory/'infrastructure_parent_receipt.json').exists(),
            'task has a prior infrastructure attempt; no hidden retry/promotion')
        case_path=output/'results'/(job['job_id']+'.json');record=read(case_path)
        parent=record.get('parent_case_process_receipt',{})
        require(absolute(root,parent.get('child_path',''))==directory/'attempt0000/case.json'
            and {p.name for p in directory.glob('attempt[0-9]*') if p.is_dir()}=={'attempt0000'}
            and parent.get('returncode',0)==0,'task raw attempt path/retry/returncode differs')
        from .simulation import development_seed
        seed,seed_identity=development_seed(task,job['row'],'validation',seed=early_config['seed'],replica=0)
        source=dict(task=task,raw_episode_id=job['source_raw_episode_id'],path=job['row']['path'])
        if 'variation' in job['row']:source['variation']=job['row']['variation']
        reset=record.get('recorded_initial_conditions',{}).get('reset_receipt',{}).get('reset_recovery_receipt',{})
        require(reset.get('task')==task and reset.get('split')=='validation' and reset.get('replica')==0
            and reset.get('seed')==seed and reset.get('seed_identity')==seed_identity and reset.get('source')==source,
            'task physical reset witness belongs to another actual source/seed/variation')
        variation=job['row'].get('variation',TASK_SPECS[task].supported_variations[seed % TASK_SPECS[task].variation_count])
        require(reset.get('variation')==variation and type(reset.get('recovery_applied')) is bool
            and type(reset.get('original_reset_succeeded')) is bool,'task reset variation/branch is missing')
        if reset['recovery_applied']:
            require(task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
                and reset['original_reset_succeeded'] is False
                and reset.get('expert_waypoint_validation_skipped_after_original40') is True
                and reset.get('independent_physical_guards_passed') is True
                and reset.get('original_native10_and_owned1_physics_steps_measured') is True,
                'task recovered reset lacks original physical-guard evidence')
        else:require(reset['original_reset_succeeded'] is True,'unrecovered task reset was not originally successful')
    completed=_validate_selected_jobs(root,plan,early_config,early_lock,projection,closure,execution_contract,jobs,by_source)
    # Recheck selected raw/parent/source bytes after all individual audits.
    verified={str(plan_path):file_sha(plan_path),str(pointer(root,common_config_pointer)):common_config_pointer['sha256'],
        str(pointer(root,common_lock_pointer)):common_lock_pointer['sha256'],
        str(pointer(root,canonical_roster_pointer)):canonical_roster_pointer['sha256'],
        str(pointer(root,execution_contract)):execution_contract['sha256']}
    for source in (kernel['original_consumer'],kernel['task_consumer']):verified[source['path']]=source['sha256']
    for item in completed:
        canonical=pointer(root,item['canonical_receipt']);raw=pointer(root,item['raw_child_receipt'])
        verified[str(canonical)]=item['canonical_receipt']['sha256'];verified[str(raw)]=item['raw_child_receipt']['sha256']
        record=read(canonical);parent=record['parent_case_process_receipt']
        log=absolute(root,parent['log_path']);require(file_sha(log)==parent['log_sha256'],'actual task log changed after audit')
        verified[str(log)]=parent['log_sha256']
        wrapper=record['baseline_projected_wrapper_execution']
        for field in ('base_run_case_receipt','base_run_case_inputs','jit_artifact_execution'):
            value=wrapper.get(field)
            if value is not None:
                path=pointer(root,value);verified[str(path)]=value['sha256']
        final=wrapper.get('dependency_witness_after_cleanup',{})
        for field in ('generated_python_module_bindings','generated_native_file_bindings'):
            for binding in final.get(field,{}).values():
                path=absolute(root,binding['path']);require(file_sha(path)==binding['sha256'],'generated task dependency changed after audit')
                verified[str(path)]=binding['sha256']
                event=binding.get('loader_event',binding.get('verification_event'))
                if event is not None:
                    path=pointer(root,event);verified[str(path)]=event['sha256']
    for name,sha in verified.items():require(file_sha(name)==sha,'task baseline input changed during audit: '+name)
    return dict(schema=COMPONENT,complete=True,task=task,baseline_task_component_reusable=True,
        actual_queue_plan=deepcopy(plan_pointer),actual_config=deepcopy(plan['config']),actual_lock=deepcopy(plan['frozen_inputs']),
        common_config=deepcopy(common_config_pointer),common_lock=deepcopy(common_lock_pointer),
        execution_contract=deepcopy(execution_contract),canonical_validation_manifest=deepcopy(canonical_roster_pointer),
        sources=expected,completed_cases=completed,full_plan_sources=275,full_plan_scope_preserved=True,
        actual_other_task_completed_snapshot=len(names)-expected,global_queue_complete_claimed=False,
        raw_identity_rewritten=False,raw_receipts_rewritten=False,single_full_config_identity_claimed=True,
        single_lock_identity_claimed=False,actual_baseline_lock_preserved=True,
        historical_projection_reuse_allowed=False,common_config_sha256=plan['config']['sha256'],
        projection_sha256=projection['projection_sha256'],execution_dependency_fingerprint=digest(closure['fingerprint']),
        shared_case_kernel=kernel,verified_files_sha256=verified,
        execution_dependency_files_sha256=dict(closure['dependencies']),
        formal_validation_release_issued=False,formal_TEST_permitted=False,actual_TEST_read=False,
        requires_complete_task_five_method_VAL_cases=5*expected,
        requires_task_model_coverage_and_initial_condition_parity=True,native_or_GPU_launched=False)
