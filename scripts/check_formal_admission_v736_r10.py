"""Check representative genuine VAL commands through the frozen admission driver.

Only actual --check child processes run, in two CPU slots. Source selection is
the first original row for each declared task, without reading native outcomes.
No native output directory, simulator, GPU work or TEST source is opened.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,sys,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,digest
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source,locked_pointer
from scripts.run_formal_suite16_owner_v731_r10 import child,driver_bindings,require_CPU_driver

CASES=(('beat_the_rotating_buzz','cv_geometry'),
       ('insert_onto_rotating_peg_high_speed','context_geometry'),
       ('pick_moving_target_on_the_table','context_geometry'),
       ('pick_moving_target_on_the_table','physmani'),
       ('place_cups_on_rotating_frame','context_geometry'))
NATIVE_OUT=Path('artifacts/framework/multitask_r10/formal_suite16_v730/validation')


def utc():return datetime.now(timezone.utc).isoformat()


def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('project evidence pointer required')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def require_receipt(receipt,row,source_sha):
    require_CPU_driver(receipt,row)
    identity=receipt.get('identity',{});domain=receipt.get('component_domain',{})
    if (receipt.get('complete') is not True or receipt.get('native_started') is not False
        or receipt.get('GPU_started') is not False or receipt.get('actual_split')!='validation'
        or domain.get('purpose')!='evaluation' or domain.get('split')!='validation'
        or domain.get('task')!=row['task'] or domain.get('method')!=row['method']
        or domain.get('raw_episode_id')!=row['raw']
        or identity.get('split')!='validation' or identity.get('method')!=row['method']
        or identity.get('task')!=row['task'] or identity.get('source_raw_episode_id')!=row['raw']
        or identity.get('source_row_sha256')!=row['source_row_sha256']
        or identity.get('config_sha256')!=row['configuration_sha256']
        or identity.get('frozen_inputs_sha256')!=row['input_lock_sha256']
        or identity.get('source_input_lock_sha256')!=source_sha
        or identity.get('demo_only') is not False):
        raise ValueError('actual formal VAL CPU source/domain/configuration/lock receipt mismatch')
    if receipt.get('unchanged_native_runtime')!=row['native_runtime']:
        raise ValueError('actual unchanged physical runtime664 binding required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('suite','suite-lock','source-inputs','out'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if (not out.is_relative_to(ROOT/'runtime/framework') or out==ROOT/'runtime/framework' or out.exists()):
        raise ValueError('fresh isolated runtime/framework CPU evidence directory required')
    out.mkdir(parents=True);cancelled=threading.Event();completed={};failed={};jobs=[];policy=None
    try:
        files=json.loads((ROOT/args.suite_lock).read_text())['files'];verify_inputs(ROOT,files)
        suite_path=(ROOT/args.suite).resolve();locked_pointer(ROOT,files,pointer(suite_path))
        suite=json.loads(suite_path.read_text());driver,guard=driver_bindings(suite,files)
        _,sources,proof=read_formal_source(ROOT,suite,files,'validation')
        policy=proof['policy_sha256'];source_lock=(ROOT/args.source_inputs).resolve()
        source_sha=file_sha(source_lock);source_inputs=json.loads(source_lock.read_text())
        if (source_inputs.get('schema')!='r10_full16_post_admission_split_source_lock_v1'
            or source_inputs.get('split')!='validation' or source_inputs.get('source_count')!=315
            or source_inputs.get('policy_sha256')!=policy
            or source_inputs.get('source_manifest_sha256')!=suite['source_manifests']['validation']['sha256']
            or source_inputs.get('TEST_used_for_fitting') is not False):
            raise ValueError('actual admitted full16 VAL source data lock required')
        verify_inputs(ROOT,source_inputs['files'])
        for task,method in CASES:
            source=next(r for r in sources if r['task']==task)
            route=next(r for r in proof['routes']['routes'] if (r['task'],r['method'])==(task,method))
            if (route['worker']['path']!='scripts/run_task_executor_smoke_rml_inputs_v709_r10.py'
                or route['consumer']['path']!='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'):
                raise ValueError('preserve the exact previously qualified physical709/664 route')
            native_out=NATIVE_OUT/task/source['raw_episode_id']/method
            if (ROOT/native_out).exists():raise FileExistsError('CPU check must precede native case creation')
            argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',driver['path'],
                '--config',route['configuration']['path'],'--manifest',suite['source_manifests']['validation']['path'],
                '--frozen-inputs',route['frozen_inputs']['path'],'--suite',args.suite,'--suite-lock',args.suite_lock,
                '--source-inputs',args.source_inputs,'--task',task,'--raw-episode-id',source['raw_episode_id'],
                '--method',method,'--split','validation','--out',str(native_out)]
            jobs.append(dict(task=task,raw=source['raw_episode_id'],method=method,argv=argv,
                job_id=task+'_'+source['raw_episode_id']+'_'+method,source_row_sha256=digest(source),
                worker_sha256=driver['sha256'],admission_driver=driver,admission_guard=guard,
                unchanged_case_worker=route['worker'],native_runtime={k:route['consumer'][k] for k in ('path','sha256')},
                configuration_sha256=route['configuration']['sha256'],input_lock_sha256=route['frozen_inputs']['sha256']))
        atomic_json(out/'launch.json',dict(created_utc=utc(),source_sha256=file_sha(__file__),actual_split='validation',
            suite=pointer(suite_path),suite_lock=pointer(ROOT/args.suite_lock),source_input_lock=pointer(source_lock),
            admission_driver=driver,admission_guard=guard,policy_sha256=policy,
            native_started=False,GPU_started=False,max_CPU_slots=2,automatic_retry=False,TEST_read=False))
        atomic_json(out/'plan.json',dict(jobs=jobs,actual_child_check_flag='--check',source_selection='first_original_source_per_task',
            source_outcome_filtering=False,physical_policy_changed=False,native_started=False,GPU_started=False,TEST_read=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(child,row,out/'CPU'/row['job_id'],cancelled,check=True):row for row in jobs}
            for future in as_completed(pending):
                row=pending[future];name=row['job_id'];result=future.result()
                try:
                    if result.get('complete') is not True:raise ValueError('actual CPU check subprocess failed: '+repr(result))
                    receipt=json.loads((out/'CPU'/name/'stdout.log').read_text());require_receipt(receipt,row,source_sha)
                    if file_sha(ROOT/row['argv'][2])!=row['worker_sha256']:raise ValueError('actual frozen admission entry changed')
                    native_out=ROOT/row['argv'][row['argv'].index('--out')+1]
                    if native_out.exists():raise ValueError('CPU preflight created a forbidden native output')
                    atomic_json(out/'CPU'/name/'CPU_preflight.json',receipt)
                    completed[name]=dict(identity=receipt['identity'],receipt=pointer(out/'CPU'/name/'CPU_preflight.json'),
                        actual_argv=row['argv']+['--check'],native_started=False,GPU_started=False)
                except BaseException as error:
                    failed[name]=dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        complete=len(completed)==len(CASES) and not failed
    except BaseException as error:
        failed['admission_or_setup']=dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        complete=False
    report=dict(complete=complete,finished_utc=utc(),actual_split='validation',expected=len(CASES),completed=completed,
        failed=failed,policy_sha256=policy,source_sha256=file_sha(__file__),native_started=False,GPU_started=False,
        physical_policy_changed=False,TEST_read=False,source_outcome_filtering=False,automatic_retry=False)
    atomic_json(out/'report.json',report);atomic_json(out/'terminal.json',report)
    print(json.dumps(dict(complete=complete,completed=len(completed),failed=len(failed),expected=len(CASES),
        out=str(out.relative_to(ROOT)),native_started=False,GPU_started=False,TEST_read=False)),flush=True)
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
