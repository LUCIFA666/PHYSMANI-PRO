"""Independently score every formal case; only complete945 VAL releases TEST."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,validate_completed_case
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v532 import read_formal_source
from our_mani.multitask.three_method_release_v531 import validation_release,audit_grid
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','suite','suite-lock','source-inputs','split','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--validation-release');args=p.parse_args()
    runtime=ROOT/args.runtime;out=ROOT/args.out
    if out.exists():raise FileExistsError('preserve independent full-cohort results')
    files=json.loads((ROOT/args.suite_lock).read_text())['files'];verify_inputs(ROOT,files)
    suite=json.loads((ROOT/args.suite).read_text());release=None
    if args.split=='test':
        if not args.validation_release or (ROOT/args.validation_release).resolve()!=(ROOT/suite['validation_release_path']).resolve():
            raise ValueError('exact complete frozen VAL release required')
        release=json.loads((ROOT/args.validation_release).read_text())
    manifest,_,proof=read_formal_source(ROOT,suite,files,args.split,validation_release=release)
    terminal=json.loads((runtime/'terminal.json').read_text());launch=json.loads((runtime/'launch.json').read_text())
    plan=json.loads((runtime/'plan.json').read_text());cases=[];pointers=[]
    if terminal.get('complete') is not True or launch.get('source_sha256')!=file_sha(ROOT/'scripts/run_formal_suite16_owner_v689_r10.py'):
        raise ValueError('complete original single-owner native stage required')
    source_sha=file_sha(ROOT/args.source_inputs)
    for row in plan['jobs']:
        name=row['job_id'];CPU=json.loads((runtime/'CPU'/name/'CPU_preflight.json').read_text())
        result=json.loads((runtime/'native'/name/'terminal.json').read_text())
        if result.get('complete') is not True:raise ValueError('infrastructure/partial cases cannot release the next stage')
        path=ROOT/row['argv'][row['argv'].index('--out')+1]/'case.json';case=json.loads(path.read_text())
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
        source_sha256=file_sha(__file__),actual_stage_owner=dict(path=str((runtime/'terminal.json').relative_to(ROOT)),sha256=file_sha(runtime/'terminal.json')),
        source_input_lock=dict(path=args.source_inputs,sha256=source_sha),
        one_fixed_original_checkpoint_not_top3_selected=True,source_outcome_filtering=False,
        native_failures_retained=True,Value_NGFF_FASTER_used=False))
    print(json.dumps(dict(complete=True,split=args.split,cases=len(cases),native_successes_by_method=result['native_successes_by_method'],
        motion_is_diagnostic_only=True,out=str(out.relative_to(ROOT)),TEST_permitted=args.split=='validation')),flush=True)


if __name__=='__main__':main()
