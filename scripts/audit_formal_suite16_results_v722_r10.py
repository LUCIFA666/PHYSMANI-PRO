"""Audit unchanged native cases and their explicit guard-only driver evidence."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v718 import read_formal_source,locked_pointer
from our_mani.multitask.three_method_release_v531 import validation_release,audit_grid
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS


def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('project evidence pointer required')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def driver_bindings(suite,files):
    expected={'formal_case_driver':'scripts/run_formal_case_admission_v719_r10.py',
              'formal_admission_guard':'our_mani/multitask/formal_suite_admission_v718.py',
              'formal_result_auditor':'scripts/audit_formal_suite16_results_v722_r10.py'}
    for key,name in expected.items():
        if locked_pointer(ROOT,files,suite[key])!=(ROOT/name).resolve():
            raise ValueError('exact frozen formal admission dependency required: '+key)
    owner=ROOT/'scripts/run_formal_suite16_owner_v721_r10.py'
    locked_pointer(ROOT,files,pointer(owner))
    return suite['formal_case_driver'],suite['formal_admission_guard']


def require_driver_evidence(receipt,driver,guard,route,case_path,split):
    expected=dict(admission_driver=driver,admission_guard=guard,unchanged_case_worker=route['worker'],
        native_runtime={k:route['consumer'][k] for k in ('path','sha256')})
    if (receipt.get('schema')!='r10_guard_only_formal_driver_v719' or receipt.get('complete') is not True
        or receipt.get('physical_policy_changed') is not False or receipt.get('actual_split')!=split
        or receipt.get('case')!=pointer(case_path)
        or any(receipt.get(k)!=v for k,v in expected.items())):
        raise ValueError('external admission receipt differs from the exact unchanged native case/driver')
    return expected


def require_actual_launch(launch,row,expected,*,check,identity):
    if (launch.get('argv')!=row['argv']+(['--check'] if check else [])
        or launch.get('check_only') is not check or launch.get('physical_policy_changed') is not False
        or launch.get('native_slots')!=(0 if check else 1)
        or launch.get('GPU_slots')!=(0 if check else int(row['method']=='physmani'))
        or launch.get('expected_identity_sha256')!=(None if check else identity['sha256'])
        or any(launch.get(k)!=v for k,v in expected.items())):
        raise ValueError('actual child launch must declare the exact frozen driver argv and case identity')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','suite','suite-lock','source-inputs','split','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--validation-release');args=p.parse_args()
    runtime=ROOT/args.runtime;out=ROOT/args.out
    if out.exists():raise FileExistsError('preserve independent full-cohort results')
    files=json.loads((ROOT/args.suite_lock).read_text())['files'];verify_inputs(ROOT,files)
    suite_path=ROOT/args.suite;locked_pointer(ROOT,files,pointer(suite_path))
    suite=json.loads(suite_path.read_text());release=None
    driver,guard=driver_bindings(suite,files)
    if args.split=='test':
        if not args.validation_release or (ROOT/args.validation_release).resolve()!=(ROOT/suite['validation_release_path']).resolve():
            raise ValueError('exact complete frozen VAL release required')
        release=json.loads((ROOT/args.validation_release).read_text())
    manifest,sources,proof=read_formal_source(ROOT,suite,files,args.split,validation_release=release)
    terminal=json.loads((runtime/'terminal.json').read_text());launch=json.loads((runtime/'launch.json').read_text())
    plan=json.loads((runtime/'plan.json').read_text());cases=[];pointers=[];driver_pointers=[];inputs={}
    def evidence(path):
        path=Path(path).resolve();item=pointer(path);inputs[str(path)]=item['sha256'];return item
    for path in (suite_path,ROOT/args.suite_lock,ROOT/args.source_inputs,runtime/'launch.json',runtime/'plan.json',runtime/'terminal.json',
                 ROOT/driver['path'],ROOT/guard['path'],Path(__file__),ROOT/'scripts/run_formal_suite16_owner_v721_r10.py'):
        evidence(path)
    source_sha=file_sha(ROOT/args.source_inputs)
    if (terminal.get('complete') is not True or launch.get('source_sha256')!=file_sha(ROOT/'scripts/run_formal_suite16_owner_v721_r10.py')
        or launch.get('split')!=args.split or launch.get('suite_lock_sha256')!=file_sha(ROOT/args.suite_lock)
        or launch.get('split_source_lock_sha256')!=source_sha or launch.get('policy_sha256')!=proof['policy_sha256']
        or launch.get('admission_driver')!=driver or launch.get('admission_guard')!=guard
        or launch.get('physical_policy_changed') is not False):
        raise ValueError('complete explicitly bound driver721 single-owner native stage required')
    source_rows={(r['task'],r['raw_episode_id']):r for r in sources}
    for row in plan['jobs']:
        name=row['job_id'];CPU=json.loads((runtime/'CPU'/name/'CPU_preflight.json').read_text())
        result=json.loads((runtime/'native'/name/'terminal.json').read_text())
        if result.get('complete') is not True:raise ValueError('infrastructure/partial cases cannot release the next stage')
        route=next(r for r in proof['routes']['routes'] if (r['task'],r['method'])==(row['task'],row['method']))
        expected=dict(admission_driver=driver,admission_guard=guard,unchanged_case_worker=route['worker'],
            native_runtime={k:route['consumer'][k] for k in ('path','sha256')})
        if (row.get('worker_sha256')!=driver['sha256'] or row['argv'][:3]!=['bash',
            'scripts/run_aligned_gpu.sh' if row['method']=='physmani' else 'scripts/run_local_sim.sh',driver['path']]
            or any(row.get(k)!=v for k,v in expected.items())
            or route['worker']['path']!='scripts/run_task_executor_smoke_rml_inputs_v709_r10.py'
            or route['consumer']['path']!='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'):
            raise ValueError('plan must expose driver719 and preserve the qualified709/664 physical route')
        for option,value in (('--config',route['configuration']['path']),('--frozen-inputs',route['frozen_inputs']['path']),
            ('--manifest',suite['source_manifests'][args.split]['path']),('--suite',args.suite),('--suite-lock',args.suite_lock),
            ('--source-inputs',args.source_inputs),('--split',args.split),('--task',row['task']),('--raw-episode-id',row['raw']),('--method',row['method'])):
            if row['argv'][row['argv'].index(option)+1]!=value:raise ValueError('actual frozen formal argv changed: '+option)
        if (CPU!=json.loads((runtime/'CPU'/name/'stdout.log').read_text())
            or CPU.get('complete') is not True or CPU.get('native_started') is not False or CPU.get('GPU_started') is not False
            or CPU.get('actual_split')!=args.split or CPU.get('component_domain',{}).get('purpose')!='evaluation'
            or any(CPU.get(k)!=v for k,v in expected.items() if k!='native_runtime')
            or CPU['identity'].get('source_row_sha256')!=digest(source_rows[(row['task'],row['raw'])])
            or CPU['identity'].get('config_sha256')!=route['configuration']['sha256']
            or CPU['identity'].get('frozen_inputs_sha256')!=route['frozen_inputs']['sha256']):
            raise ValueError('actual CPU driver/source/configuration proof changed')
        for stage,check in (('CPU',True),('native',False)):
            child_launch=runtime/stage/name/'launch.json'
            require_actual_launch(json.loads(child_launch.read_text()),row,expected,check=check,identity=CPU['identity'])
            evidence(child_launch)
        evidence(runtime/'CPU'/name/'CPU_preflight.json');evidence(runtime/'native'/name/'terminal.json')
        path=ROOT/row['argv'][row['argv'].index('--out')+1]/'case.json';case=json.loads(path.read_text())
        driver_path=path.parent/'admission_driver_receipt.json';driver_receipt=json.loads(driver_path.read_text())
        require_driver_evidence(driver_receipt,driver,guard,route,path,args.split)
        driver_pointer=evidence(driver_path)
        if result.get('case')!=pointer(path) or result.get('admission_driver_receipt')!=driver_pointer:
            raise ValueError('actual native terminal must freeze both case and external driver receipt')
        driver_pointers.append(driver_pointer)
        validate_completed_case(case,CPU['identity'])
        if case['identity'].get('source_input_lock_sha256')!=source_sha:raise ValueError('actual immutable source scope changed')
        motion=case.get('all_stage_counted_joint_motion',{})
        if len(motion.get('events',[]))!=case['counted_clock']['ticks']:raise ValueError('one measured motion window per counted tick required')
        if case.get('component_domain',{}).get('purpose')!='evaluation' or case['component_domain'].get('split')!=args.split:
            raise ValueError('genuine evaluation purpose and actual formal split required')
        cases.append(case);pointers.append(dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)))
    result=(validation_release(cases,manifest,proof['routes']) if args.split=='validation' else
            audit_grid(cases,manifest,'test',proof['routes'],completed_VAL_audit=True))
    scores=[]
    for task in TASKS:
        methods={}
        for method in METHODS:
            selected=[c for c in cases if c['task']==task and c['method']==method]
            successes=sum(c['official_task_success'] for c in selected)
            methods[method]=dict(successes=successes,episodes=len(selected),success_rate=successes/len(selected),
                native_terminal_failures=sum(c['native_terminal_failure'] for c in selected),
                actual_window_acceleration_peak=max(c['all_stage_counted_joint_motion']['peak_actual_window_acceleration'] for c in selected),
                measured_motion_violation_cases=sum(not c['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed'] for c in selected))
        scores.append(dict(task=task,methods=methods))
    atomic_json(out,dict(result,created_utc=datetime.now(timezone.utc).isoformat(),cases=pointers,per_task=scores,
        admission_driver_receipts=driver_pointers,inputs_sha256=inputs,
        admission_driver=driver,admission_guard=guard,physical_policy_changed=False,
        source_sha256=file_sha(__file__),actual_stage_owner=dict(path=str((runtime/'terminal.json').relative_to(ROOT)),sha256=file_sha(runtime/'terminal.json')),
        source_input_lock=dict(path=args.source_inputs,sha256=source_sha),
        one_fixed_original_checkpoint_not_top3_selected=True,source_outcome_filtering=False,
        native_failures_retained=True,Value_NGFF_FASTER_used=False))
    print(json.dumps(dict(complete=True,split=args.split,cases=len(cases),native_successes_by_method=result['native_successes_by_method'],
        motion_is_diagnostic_only=True,out=str(out.relative_to(ROOT)),TEST_permitted=args.split=='validation')),flush=True)


if __name__=='__main__':main()
