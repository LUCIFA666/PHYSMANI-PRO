"""Read-only acceptance of a complete frozen five-method R10 VAL queue.

Incomplete evidence produces a pending report, never a release. Performance is
described without inventing a post-VAL numeric gate. No native work, training or
official R10 TEST outcomes are read here.
"""
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .evaluation import (METHODS,atomic_json,case_identity,comparison_protocol_sha256,
    digest,file_sha,validate_completed_case)
from .release_preflight import normalized_lock,require_frozen_coverage_certificate
from .reset_parity import compare_five_way,profile_problems
from .specs import TASK_SPECS
from .trace_evidence import audit_selection_usage

SCHEMA='r10_complete_five_method_validation_release_v1'


def require(value,message):
    if not value:raise ValueError(message)


def proof_directory(out):
    """Preserve a prior interrupted publication's partial proof directory."""
    out=Path(out);directory=out.parent/(out.stem+'_proofs');number=0
    while directory.exists():
        directory=out.parent/(out.stem+f'_proofs__{number:04d}');number+=1
    directory.mkdir(parents=True,exist_ok=False)
    return directory


class Proof:
    """Project evidence plus explicitly hash-bound external author inputs."""
    def __init__(self,root):self.root=Path(root).resolve();self.files={}
    def name(self,path):
        path=(self.root/path).resolve()
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
    def file(self,path,expected=None):
        path=(self.root/path).resolve()
        require(path.is_file() and (path.is_relative_to(self.root) or expected is not None),
            'missing or unbound external proof input: '+str(path))
        checksum=file_sha(path);require(expected is None or checksum==expected,'proof input SHA differs: '+str(path))
        name=self.name(path);require(name not in self.files or self.files[name]==checksum,'proof input changed during acceptance')
        self.files[name]=checksum;return path
    def pointer(self,value):
        require(isinstance(value,dict) and {'path','sha256'}<=set(value)
            and isinstance(value['sha256'],str) and len(value['sha256'])==64
            and all(c in '0123456789abcdef' for c in value['sha256']),'explicit proof path/SHA required')
        return self.file(value['path'],value['sha256'])
    def read(self,path,expected=None):return json.loads(self.file(path,expected).read_text())
    def read_pointer(self,value):return json.loads(self.pointer(value).read_text())
    def include(self,files):
        for name,checksum in files.items():self.file(name,checksum)
    def locked(self,path,files,checksum=None):
        actual=files.get(self.name(path));require(actual is not None and (checksum is None or actual==checksum),
            'model/config/protocol was not bound by original VAL lock: '+self.name(path))
        return self.file(path,actual)


def _manifest(proof,path):
    manifest=proof.read(path);require(manifest.get('split')=='validation','only canonical VAL inventory is allowed')
    rows=[row for row in manifest.get('episodes',[]) if row.get('task') in TASK_SPECS]
    result={(row['task'],row['raw_episode_id']):row for row in rows}
    counts=Counter(task for task,raw in result)
    require(len(rows)==len(result)==275 and set(counts)==set(TASK_SPECS) and all(18<=n<=20 for n in counts.values()),
        'formal VAL requires exact real 275 original parent sources')
    return result


def _plan(proof,path,canonical,*,methods=METHODS):
    # The public single-queue release keeps the complete five-method default.
    # Only the separate component composer may request the exact four ours;
    # it must independently verify the missing 275 baseline executions.
    methods=tuple(methods)
    require(methods in (METHODS,METHODS[:-1]),'unsupported formal VAL method coverage')
    expected_cases=275*len(methods)
    plan=proof.read(path)
    require(plan.get('version')=='r10_fresh_policy_queue_v1'
        and plan.get('sha256')==digest({k:v for k,v in plan.items() if k!='sha256'})
        and plan.get('split')=='validation' and set(plan.get('methods',[]))==set(methods)
        and len(plan['methods'])==len(methods) and plan.get('replicas')==1
        and plan.get('source_episodes')==275 and plan.get('expected_cases')==expected_cases
        and plan.get('official_TEST_used_for_fitting') is False
        and plan.get('case_isolation')=='fresh_simulator_process_per_episode',
        'need the immutable complete '+str(len(methods))+'-method '+str(expected_cases)+'-case VAL plan')
    for field in ('config','manifest','frozen_inputs','parent_source','child_source'):proof.pointer(plan[field])
    source=proof.read_pointer(plan['manifest'])
    rows=[row for row in source.get('episodes',[]) if row.get('task') in TASK_SPECS]
    require({(row['task'],row['raw_episode_id']):row for row in rows}==canonical,
        'VAL queue source rows differ from canonical inventory')
    jobs=plan.get('jobs',[]);seen=set();job_ids=set()
    for job in jobs:
        row=job['row'];key=(row['task'],row['raw_episode_id']);identity=job['identity'];method=identity['method']
        require(key in canonical and row==canonical[key] and method in methods
            and job.get('replica')==0 and job.get('exploration_seed') is None
            and job.get('record_demo') is False and job.get('source_raw_episode_id')==key[1],
            'VAL job has different source/method/replica or TRAIN exploration/demo selection')
        expected=case_identity(split='validation',task=key[0],raw_episode_id=key[1]+'__policy_aug_r0',method=method,
            source_row=row,config_sha256=plan['config']['sha256'],inputs_sha256=plan['frozen_inputs']['sha256'],
            parent_raw_episode_id=key[1],replica=0)
        require(identity==expected and job['job_id']==expected['sha256'][:20]
            and job['job_id'] not in job_ids and (key,method) not in seen,'VAL job identity/coverage is duplicated or changed')
        seen.add((key,method));job_ids.add(job['job_id'])
    require(len(jobs)==expected_cases and seen=={(key,method) for key in canonical for method in methods},
        'complete paired five-method VAL job coverage missing' if len(methods)==5 else
        'complete paired four-ours VAL job coverage missing')
    return plan


def _cases(proof,plan,*,split='validation',expected_cases=1375):
    output=(proof.root/plan['output']).resolve()
    require(output.is_relative_to(proof.root/'artifacts/framework/multitask_r10'),'VAL output must stay in the R10 project')
    status=proof.read(output/'status.json')
    require(status.get('complete') is True and status.get('stage')=='complete'
        and status.get('plan_sha256')==plan['sha256'] and status.get('expected')==status.get('completed')==expected_cases
        and not status.get('errors') and status.get('stop_requested') is False,
        'full clean 1375-case VAL queue is not completed')
    names={path.name for path in (output/'results').glob('*.json')}
    require(names=={job['job_id']+'.json' for job in plan['jobs']},'missing or foreign canonical VAL result')
    groups=defaultdict(dict);receipts=[];children=set();baseline_sha=None
    if any(job['identity']['method']=='physmani' for job in plan['jobs']):
        baseline_path=proof.root/plan['_config']['baseline_checkpoint']
        proof.file(baseline_path,proof.files.get(proof.name(baseline_path)))
        baseline_sha=proof.files[proof.name(baseline_path)]
    for job in plan['jobs']:
        record_path=output/'results'/(job['job_id']+'.json');record=proof.read(record_path)
        validate_completed_case(record,job['identity']);receipt=record.get('parent_case_process_receipt',{})
        child_path=(proof.root/receipt['child_path']).resolve()
        require(child_path.is_relative_to(output/'cases'/job['job_id']) and child_path.name=='case.json'
            and child_path not in children,'raw child is not this unique planned VAL job')
        children.add(child_path);child=proof.read(child_path,receipt['child_sha256'])
        require({k:v for k,v in record.items() if k!='parent_case_process_receipt'}==child
            and receipt.get('child_process_id')==child.get('process_id')
            and type(receipt.get('launcher_process_id')) is int and receipt['launcher_process_id']>0
            and receipt.get('plan_sha256')==plan['sha256']
            and receipt.get('parent_source_sha256')==plan['parent_source']['sha256'],
            'canonical VAL parent is not its exact clean raw child under this source/plan')
        wrapper='scripts/run_aligned_gpu.sh' if record['method']=='physmani' else 'scripts/run_local_sim.sh'
        command=['bash',wrapper,plan['child_source']['path'],'--config',plan['config']['path'],
            '--manifest',plan['manifest']['path'],'--frozen-inputs',plan['frozen_inputs']['path'],
            '--split',split,'--task',record['task'],'--raw-episode-id',job['source_raw_episode_id'],
            '--method',record['method'],'--replica','0','--out',str(child_path.parent.relative_to(proof.root))]
        if job.get('record_demo'):command+=['--record-demo']
        require(receipt.get('command')==command,'VAL raw launcher command differs from frozen planned case')
        proof.file(receipt['log_path'],receipt['log_sha256'])
        require(record.get('cleanup_completed') is True and record.get('evaluation_process_pid')==record['process_id']
            and record.get('entrypoint_sha256')==plan['child_source']['sha256']
            and record.get('config_sha256')==plan['config']['sha256']
            and record.get('manifest_sha256')==plan['manifest']['sha256']
            and record.get('source_row_sha256')==job['identity']['source_row_sha256']
            and record.get('frozen_input_lock',{}).get('sha256')==plan['frozen_inputs']['sha256']
            and (proof.root/record['frozen_input_lock']['path']).resolve()==(proof.root/plan['frozen_inputs']['path']).resolve()
            and record.get('demo_only') is False,
            'VAL cleanup/source/config/lock identity proof differs')
        if split=='validation':require(record.get('video') is None and job.get('record_demo') is False,'VAL cannot include demo selection')
        else:
            require(record.get('source_motion_replayed') is True
                and record.get('recorded_video_in_formal_metrics') is bool(job['record_demo'])
                and ((record.get('video') is not None) is bool(job['record_demo'])),
                'official TEST recording/source replay differs from fixed first-case job')
        require(all(isinstance(record.get(name),(float,int)) and np.isfinite(record[name]) and record[name]>=0
            for name in ('compute_seconds','wall_seconds','simulation_seconds'))
            and abs(record['simulation_seconds']-record['counted_clock']['seconds'])<=2e-5,
            'VAL native counted or compute/wall clock is invalid')
        if record['method']=='physmani':
            require(record.get('checkpoint_sha256')==baseline_sha
                and record.get('baseline_executor')==plan['_config']['baseline_executor'],'baseline checkpoint/executor changed during VAL')
        else:
            expected=plan['_config'].get('continuation_identities',{}).get(record['task'],{}).get(record['predictor'])
            require(expected is not None and record.get('continuation_sha256')==expected
                and record.get('continuation_identity',{}).get('sha256')==expected
                and record.get('unexecuted_candidates_labelled') is False
                and record.get('first_selection_policy')==('value_once' if record['method'].endswith('_value') else 'geometry')
                and record.get('downstream_selection_policy')=='geometry','VAL policy continuation/value-once semantics changed')
        key=(record['task'],job['source_raw_episode_id']);groups[key][record['method']]=record
        receipts.append(dict(task=key[0],parent_raw_episode_id=record['parent_raw_episode_id'],
            source_raw_episode_id=key[1],method=record['method'],
            path=proof.name(record_path),sha256=file_sha(record_path),raw_child_path=proof.name(child_path),
            raw_child_sha256=receipt['child_sha256'],identity_sha256=job['identity']['sha256']))
    return groups,receipts,status


def _performance(groups):
    per_task={};total={method:dict(successes=0,episodes=0) for method in METHODS}
    for task in TASK_SPECS:
        rows=[records for (name,raw),records in groups.items() if name==task];methods={}
        for method in METHODS:
            successes=sum(records[method]['official_task_success'] for records in rows);n=len(rows)
            methods[method]=dict(successes=successes,episodes=n,success_rate=successes/n)
            total[method]['successes']+=successes;total[method]['episodes']+=n
        per_task[task]=dict(methods=methods,value_minus_geometry={p:
            dict(success_count_difference=methods[p+'_value']['successes']-methods[p+'_geometry']['successes'],
                success_rate_difference=methods[p+'_value']['success_rate']-methods[p+'_geometry']['success_rate'])
            for p in ('cv','context')},minus_baseline={method:methods[method]['success_rate']-methods['physmani']['success_rate']
                for method in METHODS if method!='physmani'})
    for row in total.values():row['success_rate']=row['successes']/row['episodes']
    negatives=[dict(task=task,predictor=p,**row) for task,data in per_task.items()
        for p,row in data['value_minus_geometry'].items() if row['success_count_difference']<0]
    return dict(passed=None,status='not_preregistered_numeric_gate',per_task=per_task,pooled=total,
        negative_value_geometry_differences=negatives,post_VAL_threshold_invented=False,
        performance_does_not_override_hard_protocol_checks=True,success_module_benefit_established=False)


def finalize_validation_release(root,queue_plan,coverage_certificate,out,*,
        canonical_validation_manifest='artifacts/framework/protocol/validation_manifest.json',
        experiment_plan='config/framework/multitask_r10_experiment_plan.json'):
    """Write an immutable release only after all real evidence passes."""
    proof=Proof(root);out=(proof.root/out).resolve()
    require(out.is_relative_to(proof.root/'artifacts/framework/multitask_r10') and not out.exists(),
        'use a new release path under desktop multitask_r10; preserve existing releases')
    stage='inventory';plan=None
    try:
        canonical=_manifest(proof,canonical_validation_manifest);stage='queue_plan'
        plan=_plan(proof,queue_plan,canonical);config=proof.read_pointer(plan['config']);plan['_config']=config
        old_lock=proof.read_pointer(plan['frozen_inputs']);old_files=normalized_lock(proof.root,old_lock['files']);proof.include(old_files)
        for field in ('config','manifest','child_source'):
            proof.locked(plan[field]['path'],old_files,plan[field]['sha256'])
        for pointer in [config['context_checkpoint']]+[pointer for task in TASK_SPECS
                for pointer in config['learned_artifacts'][task].values() if isinstance(pointer,dict) and 'path' in pointer]:
            proof.locked(pointer['path'],old_files,pointer['sha256'])
        proof.locked(config['baseline_checkpoint'],old_files)
        stage='completed_native_cases';groups,receipts,status=_cases(proof,plan)
        require(set(groups)==set(canonical),'completed VAL sources differ from canonical real inventory')
        stage='coverage_certificate';coverage=proof.read_pointer(coverage_certificate)
        proof.include(coverage['input_files_sha256'])
        coverage_receipt=require_frozen_coverage_certificate(proof.root,coverage_certificate,config,proof.files)
        stage='TRAIN_frozen_tolerances';profile=proof.read_pointer(config['initial_condition_tolerance_profile'])
        proof.locked(config['initial_condition_tolerance_profile']['path'],old_files,config['initial_condition_tolerance_profile']['sha256'])
        require(not profile_problems(profile,evidence_root=proof.root),'TRAIN tolerance profile is not valid/frozen')
        proof.pointer(profile['TRAIN_evidence'])
        stage='recorded_initial_conditions';comparisons=[]
        for key,records in groups.items():
            comparison=compare_five_way(records,profile,evidence_root=proof.root,require_frozen=True)
            require(comparison.get('complete') is True and comparison.get('accepted') is True
                and len(comparison.get('checks',[]))==10,'recorded initial conditions fail for '+repr(key))
            comparisons.append(dict(task=key[0],parent_raw_episode_id=key[1],**comparison))
        stage='selection_trace';usage=audit_selection_usage(groups)
        stage='preregistered_stopping_rule';experiment=proof.read(experiment_plan)
        proof.locked(experiment_plan,old_files)
        stopping=experiment.get('stopping_rule',{})
        require(stopping.get('preregistered_before_remaining14_VAL_and_TEST') is True
            and stopping.get('deployment_or_success_module_benefit_assumed') is False
            and stopping.get('performance_screen_failure')
            and experiment.get('methods')==list(METHODS)
            and experiment.get('original_sources',{}).get('validation')==275,
            'fixed comparison stopping rule was not registered in the VAL inputs')
        performance=_performance(groups);proof_dir=proof_directory(out)
        pair_path=proof_dir/'recorded_initial_conditions.json';atomic_json(pair_path,dict(
            complete=True,accepted=True,split='validation',sources=275,pairs=2750,entries=comparisons,
            full_native_physics_state_identical_claimed=False,metric_only=True,policy_use_allowed=False))
        receipt_path=proof_dir/'canonical_cases.json';atomic_json(receipt_path,dict(complete=True,expected=1375,entries=receipts))
        usage_path=proof_dir/'selection_usage.json';atomic_json(usage_path,usage)
        proof.file(pair_path);proof.file(receipt_path);proof.file(usage_path)
        proof.file(proof.root/'our_mani/multitask/trace_evidence.py')
        proof.file(Path(__file__),file_sha(Path(__file__)));proof.file(proof.root/'scripts/finalize_multitask_validation_release_r10.py')
        proof.include(dict(proof.files))
        release=dict(schema=SCHEMA,complete=True,protocol_acceptance_passed=True,split='validation',TEST_used=False,
            formal_test_permitted=True,expected_source_cases=275,expected_method_cases=1375,completed_method_cases=1375,
            config_sha256=plan['config']['sha256'],validated_config=plan['config'],validated_input_lock=plan['frozen_inputs'],
            comparison_protocol_sha256=comparison_protocol_sha256(config),queue_plan=dict(path=proof.name(queue_plan),sha256=file_sha(proof.root/queue_plan)),
            model_coverage_certificate=coverage_certificate,model_coverage_receipt=coverage_receipt,
            recorded_initial_conditions=dict(path=proof.name(pair_path),sha256=file_sha(pair_path),sources=275,pairs=2750,accepted=True),
            canonical_cases=dict(path=proof.name(receipt_path),sha256=file_sha(receipt_path),cases=1375),
            selection_usage=dict(path=proof.name(usage_path),sha256=file_sha(usage_path),diagnostics_valid=True),
            initial_condition_tolerance_profile=config['initial_condition_tolerance_profile'],TRAIN_repeat_evidence=profile['TRAIN_evidence'],
            preregistered_experiment=dict(path=proof.name(experiment_plan),sha256=file_sha(proof.root/experiment_plan)),
            development_performance_screen=performance,development_performance_screen_passed=None,
            fixed_TEST_continuation_reason='Complete requested frozen comparison under preregistered stopping rule; no numeric performance gate was registered',
            success_rate_benefit_validated=False,input_files_sha256=proof.files,
            actual_R10_TEST_read=False,native_or_GPU_launched=False,model_fitting_performed=False,
            full_native_physics_state_identical_claimed=False,
            limitations=['Only recorded initial conditions are matched; opaque solver/contact caches are unrecorded.',
                'VAL is development evidence; a complete formal TEST comparison is still required.'])
        atomic_json(out,release);return release
    except Exception as error:
        pending=dict(schema=SCHEMA,complete=False,protocol_acceptance_passed=False,formal_test_permitted=False,
            split='validation',TEST_used=False,stage=stage,errors=[repr(error)],expected_source_cases=275,
            expected_method_cases=1375,no_release_written=True,input_files_sha256=proof.files,
            actual_R10_TEST_read=False,native_or_GPU_launched=False,model_fitting_performed=False,
            no_infrastructure_failure_scored_as_policy_negative=True)
        directory=out.parent/'pending_validation_release';directory.mkdir(parents=True,exist_ok=True)
        number=0
        while (directory/(out.stem+f'__{number:04d}.json')).exists():number+=1
        path=directory/(out.stem+f'__{number:04d}.json');atomic_json(path,pending)
        pending['pending_report']=dict(path=proof.name(path),sha256=file_sha(path))
        return pending
