"""Independent PhysMani VAL entrypoint with an explicit projected backend.

Preparation/checks start neither GPU nor native. Native execution requires the
separately approved, immutable full dependency contract. Original run_case and
all early full-config/lock identities remain visible in the raw receipts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from our_mani.multitask import baseline_component as component
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, validate_completed_case, verify_inputs
from scripts import evaluate_multitask_case_r10 as evaluator
from scripts.run_multitask_fresh_queue_r10 import build_jobs

VERSION = 'r10_actual_projected_PhysMani_VAL_wrapper_v1'
BASE = ROOT/'artifacts/framework/multitask_r10'
STOP = ROOT/'runtime/framework/multitask_r10_STOP'


def read(path): return json.loads(Path(path).read_text())


def inside(path):
    path = (ROOT/Path(path)).resolve()
    component.require(path.is_relative_to(BASE) and path != BASE, 'fresh baseline output must stay in R10 artifacts')
    return path


def ptr(path):
    path = Path(path).resolve()
    component.require(path.is_relative_to(ROOT), 'project execution pointer escaped workspace')
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def contract_for(prepared, path):
    path = (ROOT/Path(path)).resolve()
    checksum = evaluator.locked_digest(path, prepared['files'])
    return dict(path=str(path.relative_to(ROOT)), sha256=checksum)


def run_projected_case(prepared, execution_contract, *, backend_factory=None, run_case=None):
    """Call the original evaluator with the actual explicit backend object."""
    args = prepared['args']
    component.require(args.split == 'validation' and args.method == 'physmani' and args.replica == 0
        and args.exploration_seed is None and not args.record_demo and not args.demo_only,
        'early component is only original VAL PhysMani replica0 without exploration/videos')
    config = prepared['config']
    roster = component.canonical_VAL_roster(read(prepared['manifest_path']))
    contract_path = component.pointer(ROOT, execution_contract)
    component.require(prepared['files'].get(str(contract_path)) == execution_contract['sha256'],
                      'execution contract is not in actual early lock')
    for path in (Path(__file__), Path(evaluator.__file__), ROOT/'our_mani/multitask/baseline_component.py'):
        evaluator.locked_digest(path, prepared['files'])
    component.verify_execution_contract(ROOT, execution_contract, config, {'files':prepared['files']}, roster)
    backend_factory = component.ProjectedBaselineBackend if backend_factory is None else backend_factory
    backend = backend_factory(execution_contract=execution_contract, root=ROOT)
    # Full prepared config is intentional: original identity/preflight/raw inputs
    # remain full. The backend projects at both environment and policy boundaries.
    runner = evaluator.run_case if run_case is None else run_case
    jit_execution=None
    try:
        if hasattr(backend,'start_jit_observer'):backend.start_jit_observer(prepared)
        runner(prepared, backend=backend)
        # Observe cleanup imports while generated module identity authority is
        # still in scope; only then restore every exact stock hook.
        backend._imports()
        if hasattr(backend,'close_jit_observer'):jit_execution=backend.close_jit_observer()
    except Exception as error:
        if hasattr(backend,'close_jit_observer'):
            try:jit_execution=backend.close_jit_observer(error)
            except Exception as close_error:
                atomic_json(prepared['out']/'baseline_jit_cleanup_error.json',dict(
                    error_class=type(close_error).__name__,error=str(close_error),complete=False))
        diagnostic=backend.infrastructure_runtime_state()
        if diagnostic is not None:
            atomic_json(prepared['out']/'baseline_runtime_state_infrastructure_receipt.json',dict(
                identity=deepcopy(prepared['identity']),actual_wrapper=ptr(Path(__file__)),
                error_class=type(error).__name__,error=str(error),runtime_observations=diagnostic,
                complete=False,counted_as_policy_failure=False))
        if jit_execution is not None:
            atomic_json(prepared['out']/'baseline_jit_infrastructure_receipt.json',dict(
                identity=deepcopy(prepared['identity']),jit_artifact_execution=jit_execution,
                error_class=type(error).__name__,error=str(error),complete=False,counted_as_policy_failure=False))
        raise
    out = prepared['out']; case_path = out/'case.json'
    base_record = read(case_path)
    validate_completed_case(base_record, prepared['identity'])
    component.require(base_record.get('entrypoint_sha256') == file_sha(evaluator.__file__),
                      'original evaluator entrypoint identity must remain unchanged')
    component.require(backend.witness is not None and backend.witness.get('complete') is True,
                      'actual projected backend/reset/runtime witness is missing')
    # This final observer may see imports made during cleanup. Put its additional
    # evidence in our own namespace rather than mutating the raw evaluator file.
    final_witness = deepcopy(backend.witness)
    mutable_state_execution=backend.finalize_runtime_state()
    evaluator.verify_preflight(prepared)
    base_path = out/'evaluate_run_case_receipt.json'
    component.require(not base_path.exists(), 'refusing to replace prior base evaluator receipt')
    case_path.rename(base_path)
    record = deepcopy(base_record)
    record['baseline_projected_wrapper_execution'] = dict(version=VERSION,
        execution_contract=deepcopy(execution_contract), actual_wrapper=ptr(Path(__file__)),
        evaluator_source=ptr(Path(evaluator.__file__)), base_run_case_receipt=ptr(base_path),
        base_run_case_inputs=ptr(out/'inputs.json'),
        full_config_identity_preserved=True, projected_config_sha256=digest(component.project_baseline_config(config)),
        dependency_witness_after_cleanup=final_witness,mutable_runtime_state_execution=mutable_state_execution,
        jit_artifact_execution=jit_execution,
        original_record_fields_rewritten=False)
    atomic_json(case_path, record)
    return record


def prepare_plan(config_path, manifest_path, lock_path, contract_path, *, plan_path, output):
    paths = [(ROOT/Path(value)).resolve() for value in (config_path,manifest_path,lock_path,contract_path)]
    config_path,manifest_path,lock_path,contract_path = paths
    config,manifest,lock,contract = map(read, paths)
    verify_inputs(ROOT, lock['files'])
    for path in (*paths[:2], contract_path, Path(__file__), Path(evaluator.__file__)):
        evaluator.locked_digest(path, lock['files'])
    rows = component.canonical_VAL_roster(manifest)
    execution_contract = ptr(contract_path)
    component.require(contract.get('schema') == component.CONTRACT
        and contract.get('projected_config_sha256') == digest(component.project_baseline_config(config))
        and contract.get('canonical_VAL_roster_sha256') == digest(rows), 'draft or approved contract config/roster differs')
    jobs = build_jobs(rows, 'validation', ['physmani'], file_sha(config_path), file_sha(lock_path), replicas=1)
    plan = dict(version='r10_fresh_policy_queue_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        split='validation', config=ptr(config_path), manifest=ptr(manifest_path), frozen_inputs=ptr(lock_path),
        parent_source=ptr(Path(__file__)), child_source=ptr(Path(__file__)),
        baseline_execution_contract=execution_contract, output=str(inside(output).relative_to(ROOT)),
        source_episodes=275, expected_cases=275, methods=['physmani'], replicas=1, jobs=jobs,
        case_isolation='fresh_simulator_process_per_episode', original_development_source_motion_replayed=False,
        demo_per_method_per_task=1, unexecuted_candidates_labelled=False, infra_failures_are_not_policy_failures=True,
        official_TEST_used_for_fitting=False)
    plan['sha256'] = digest(plan)
    atomic_json(inside(plan_path), plan)
    return plan


def verify_plan(plan_path, *, approved=True):
    plan = read(plan_path)
    component.require(plan.get('version') == 'r10_fresh_policy_queue_v1'
        and plan.get('sha256') == digest({k:v for k,v in plan.items() if k!='sha256'})
        and plan.get('split') == 'validation' and plan.get('methods') == ['physmani']
        and plan.get('replicas') == 1 and plan.get('source_episodes') == plan.get('expected_cases') == 275
        and plan.get('official_TEST_used_for_fitting') is False
        and plan.get('case_isolation') == 'fresh_simulator_process_per_episode', 'full original baseline-only VAL plan required')
    for key in ('config','manifest','frozen_inputs','parent_source','child_source','baseline_execution_contract'):
        component.pointer(ROOT, plan[key])
    component.require(plan['parent_source'] == ptr(Path(__file__)) == plan['child_source'],
                      'actual projected wrapper must own both parent and child command')
    lock = read(ROOT/plan['frozen_inputs']['path']); verify_inputs(ROOT,lock['files'])
    config = read(ROOT/plan['config']['path'])
    roster = component.canonical_VAL_roster(read(ROOT/plan['manifest']['path']))
    expected = build_jobs(roster,'validation',['physmani'],plan['config']['sha256'],plan['frozen_inputs']['sha256'],replicas=1)
    component.require(plan['jobs'] == expected, 'baseline plan jobs differ from original source/identity/replica budget')
    if approved:
        component.verify_execution_contract(ROOT,plan['baseline_execution_contract'],config,lock,roster)
    else:
        component.verify_execution_contract(ROOT,plan['baseline_execution_contract'],config,lock,roster,require_approved=False)
        component.require(read(ROOT/plan['baseline_execution_contract']['path']).get('approved_before_VAL_execution') is False,
                          'draft check never grants implicit execution approval')
    return plan


def worker_command(plan, job, child_out):
    return ['bash','scripts/run_aligned_gpu.sh',plan['child_source']['path'],'--collect-case',
        '--config',plan['config']['path'],'--manifest',plan['manifest']['path'],
        '--frozen-inputs',plan['frozen_inputs']['path'],'--split','validation',
        '--task',job['row']['task'],'--raw-episode-id',job['row']['raw_episode_id'],
        '--method','physmani','--replica','0','--out',str(child_out.relative_to(ROOT)),
        '--execution-contract',plan['baseline_execution_contract']['path']]


def run_queue(plan_path, *, workers=1, case_timeout=3600, job_ids=None):
    plan = verify_plan(plan_path)
    component.require(type(workers) is int and 1<=workers<=4 and type(case_timeout) is int and case_timeout>0,
                      'one to four explicit GPU workers and positive timeout required')
    known = {job['job_id']:job for job in plan['jobs']}
    component.require(job_ids is None or (isinstance(job_ids,(list,tuple)) and len(job_ids)>0
        and len(set(job_ids))==len(job_ids) and all(item in known for item in job_ids)),
        'queue subset must be explicit unique known job IDs from the unchanged full275 plan')
    selected_ids=set(known if job_ids is None else job_ids)
    selected=[job for job in plan['jobs'] if job['job_id'] in selected_ids]
    output = inside(plan['output']); output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'queue_owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed_ids=set(); started_ids=set(); outcomes={}; errors={}
        cancelled=threading.Event(); launch_lock=threading.Lock(); status_lock=threading.Lock()
        def resume(job):
            canonical=output/'results'/(job['job_id']+'.json')
            record=read(canonical);validate_completed_case(record,job['identity'])
            parent=record['parent_case_process_receipt'];raw=ROOT/parent['child_path']
            component.require(file_sha(raw)==parent['child_sha256']
                and read(raw)=={k:v for k,v in record.items() if k!='parent_case_process_receipt'}
                and parent.get('plan_sha256')==plan['sha256']
                and parent.get('parent_source_sha256')==plan['parent_source']['sha256'],
                'baseline readonly resume raw/parent/plan identity differs')
        for job in plan['jobs']:
            case_base = output/'cases'/job['job_id']; canonical=output/'results'/(job['job_id']+'.json')
            prior_prelaunch=output/'infrastructure_prelaunch'/(job['job_id']+'.json')
            if (case_base.exists() or prior_prelaunch.exists()) and not canonical.exists():
                raise ValueError('no automatic retry of a previous baseline infrastructure attempt: '+job['job_id'])
            if canonical.exists():
                resume(job);completed_ids.add(job['job_id'])
                if job['job_id'] in selected_ids:
                    outcomes[job['job_id']]=dict(job_id=job['job_id'],status='resumed_verified')
        def status_locked(*, running):
            pending=selected_ids-started_ids-set(outcomes)
            return dict(complete=not running and len(completed_ids)==275 and not errors and not STOP.exists(),
                queue_running=running,plan_sha256=plan['sha256'],expected=275,completed=len(completed_ids),
                selected_expected=len(selected),selected_complete=len(selected_ids & completed_ids),
                selected_job_ids=[job['job_id'] for job in selected],
                errors=[errors[key] for key in sorted(errors)],
                infrastructure_cancellation_requested=cancelled.is_set(),stop_requested=STOP.exists(),
                pending_job_ids=sorted(pending),running_job_ids=sorted(started_ids-set(outcomes)),
                remaining_unattempted_job_ids=sorted(set(known)-started_ids-completed_ids-set(errors)),
                remaining_case_ids=sorted(set(known)-completed_ids),
                outcomes=[outcomes[key] for key in sorted(outcomes)],
                automatic_retries=False,formal_TEST_used=False,
                no_infrastructure_failure_scored_as_policy_failure=True)
        def publish(*, running=True):
            with status_lock:
                value=status_locked(running=running)
                atomic_json(output/'status.json',value,overwrite=True)
                return value
        def record_outcome(row, *, completed=False):
            with status_lock:
                outcomes[row['job_id']]=row
                if completed:completed_ids.add(row['job_id'])
                if row['status']=='infrastructure_error':errors[row['job_id']]=row
                atomic_json(output/'status.json',status_locked(running=True),overwrite=True)
            return row
        publish()
        def one(job):
            job_id=job['job_id'];process=None;command=None;log=None;job_base=None
            canonical = output/'results'/(job['job_id']+'.json')
            if job_id in completed_ids:return outcomes[job_id]
            try:
                if cancelled.is_set() or STOP.exists():
                    return record_outcome(dict(job_id=job_id,status='unattempted_cancelled'))
                verify_plan(plan_path)
                # A bounded scheduler plus this final serialized launch check
                # prevents pending work from starting after the first infra error.
                with launch_lock:
                    if cancelled.is_set() or STOP.exists():
                        return record_outcome(dict(job_id=job_id,status='unattempted_cancelled'))
                    job_base=output/'cases'/job_id;job_base.mkdir(parents=True,exist_ok=False)
                    child_out=job_base/'attempt0000';child_out.mkdir()
                    command=worker_command(plan,job,child_out);log=job_base/'attempt0000.log'
                    with log.open('wb') as stream:
                        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    with status_lock:started_ids.add(job_id)
                publish()
                try:code=process.wait(timeout=case_timeout)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code=None
                child=child_out/'case.json'
                component.require(code==0 and child.is_file(), 'native child failed or did not produce a complete raw case')
                record=read(child);validate_completed_case(record,job['identity'])
                component.require(type(record.get('process_id')) is int and record['process_id']>0,
                                  'native Python child PID must remain independently recorded')
                record['parent_case_process_receipt']=dict(child_path=str(child.relative_to(ROOT)),child_sha256=file_sha(child),
                    child_process_id=record['process_id'],launcher_process_id=process.pid,command=command,
                    log_path=str(log.relative_to(ROOT)),log_sha256=file_sha(log),plan_sha256=plan['sha256'],
                    parent_source_sha256=plan['parent_source']['sha256'])
                verify_plan(plan_path);atomic_json(canonical,record)
                return record_outcome(dict(job_id=job_id,status='completed'),completed=True)
            except Exception as error:
                cancelled.set()
                receipt=dict(command=command,launcher_process_id=None if process is None else process.pid,
                    returncode=None if process is None else process.returncode,plan_sha256=plan['sha256'],
                    counted_as_policy_failure=False,automatic_retries=False,
                    error_class=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                if process is not None and process.poll() is None:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    receipt['cleanup_returncode']=process.returncode
                target=(job_base/'infrastructure_parent_receipt.json') if job_base is not None else (
                    output/'infrastructure_prelaunch'/(job_id+'.json'))
                atomic_json(target,receipt)
                return record_outcome(dict(job_id=job_id,status='infrastructure_error',
                    receipt=str(target.relative_to(ROOT)),log=None if log is None else str(log.relative_to(ROOT))))
        todo=iter(job for job in selected if job['job_id'] not in completed_ids)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active=set()
            for _ in range(workers):
                job=next(todo,None)
                if job is not None:active.add(pool.submit(one,job))
            while active:
                done,active=wait(active,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                if not cancelled.is_set() and not STOP.exists():
                    for _ in done:
                        job=next(todo,None)
                        if job is not None:active.add(pool.submit(one,job))
        return publish(running=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-plan','check','check-draft','run-queue','collect-case'):
        action.add_argument('--'+name,action='store_true')
    for name in ('config','manifest','frozen-inputs','execution-contract','plan','output'):
        parser.add_argument('--'+name)
    parser.add_argument('--split',choices=('validation',),default='validation')
    parser.add_argument('--task');parser.add_argument('--raw-episode-id');parser.add_argument('--method',choices=('physmani',),default='physmani')
    parser.add_argument('--replica',type=int,default=0);parser.add_argument('--exploration-seed',type=int)
    parser.add_argument('--record-demo',action='store_true');parser.add_argument('--demo-only',action='store_true')
    parser.add_argument('--out');parser.add_argument('--workers',type=int,default=1);parser.add_argument('--case-timeout',type=int,default=3600)
    parser.add_argument('--queue-job-id',action='append',help='Explicit unique existing job ID; the plan remains full275')
    args=parser.parse_args()
    def need(*names):
        for name in names:
            if getattr(args,name) is None:parser.error('requires --'+name.replace('_','-'))
    if args.collect_case:
        need('config','manifest','frozen_inputs','execution_contract','task','raw_episode_id','out')
        prepared=evaluator.preflight(args)
        record=run_projected_case(prepared,contract_for(prepared,args.execution_contract))
        result=dict(completed=True,case_identity_sha256=record['identity']['sha256'])
    else:
        need('plan')
        if args.prepare_plan:
            need('config','manifest','frozen_inputs','execution_contract','output')
            plan=prepare_plan(args.config,args.manifest,args.frozen_inputs,args.execution_contract,plan_path=args.plan,output=args.output)
            result=dict(plan_sha256=plan['sha256'],cases=275,native_started=False,GPU_started=False)
        elif args.check or args.check_draft:
            plan=verify_plan(ROOT/args.plan,approved=not args.check_draft)
            result=dict(plan_sha256=plan['sha256'],cases=275,execution_approved=not args.check_draft,native_started=False,GPU_started=False)
        else:result=run_queue(ROOT/args.plan,workers=args.workers,case_timeout=args.case_timeout,job_ids=args.queue_job_id)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
