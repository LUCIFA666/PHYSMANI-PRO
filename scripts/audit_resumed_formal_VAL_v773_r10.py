"""Audit one unchanged945-case VAL grid across an explicit user-authorized resume.

The frozen v754 checks remain the base per-case checks. Old accepted cases
retain their actual argv, output paths and log provenance; new attempts retain
their real v772 owner identity. This auditor never impersonates v753/v754.
"""
import argparse,json,sys
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_formal_suite16_results_v754_r10 as base
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS,option
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.formal_suite_admission_v728 import locked_pointer,read_formal_source
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release_v531 import validation_release
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS

CONTROL_SCHEMA='r10_user_authorized_formal_VAL_resume_control_v772'
LOCK_SCHEMA='r10_user_authorized_formal_VAL_resume_inputs_v772'
OWNER='scripts/run_resumed_formal_VAL_owner_v772_r10.py'
RUNTIME='runtime/framework/formal_VAL_resume_v772'
OUTPUT='artifacts/framework/multitask_r10/formal_VAL_resume_v772/validation'
ROW_PROVENANCE={'evidence_runtime','imported_completed'}


def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('project-only resume evidence required')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def project_resume_inputs(supplement,base_files):
    """Carry new project evidence, retaining external inputs via locked maps."""
    result={}
    for name,checksum in supplement.items():
        absolute=str((ROOT/name).resolve())
        if not Path(absolute).is_relative_to(ROOT):continue
        if base_files.get(absolute,base_files.get(name))==checksum:continue
        if absolute in result and result[absolute]!=checksum:raise ValueError('conflicting resume evidence aliases')
        result[absolute]=checksum
    return result


def require_resume_plan(control,old_plan,old_status,old_terminal,new_plan,*,runtime):
    """Pure provenance/partition check, independent of outcomes and file reads."""
    if (control.get('schema')!=CONTROL_SCHEMA or control.get('split')!='validation'
        or control.get('user_authorized_resume') is not True or control.get('expected_cases')!=945):
        raise ValueError('explicit user-authorized945-case VAL resume required')
    old_jobs=old_plan.get('jobs',[]);new_jobs=new_plan.get('jobs',[])
    old={r['job_id']:r for r in old_jobs};new={r['job_id']:r for r in new_jobs}
    completed=control.get('completed_job_ids');remaining=control.get('remaining_job_ids');interrupted=control.get('interrupted_job_ids')
    if (not all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in (completed,remaining,interrupted))
        or len(completed)!=len(set(completed))):
        raise ValueError('explicit unique resume partitions required')
    if (len(completed)!=95 or len(remaining)!=850 or len(set(remaining))!=850 or len(interrupted)!=2 or len(set(interrupted))!=2
        or len(old_jobs)!=len(old)!=945 or len(new_jobs)!=len(new)!=945 or set(new)!=set(old)
        or set(completed)&set(remaining) or set(completed)|set(remaining)!=set(old)
        or not set(interrupted)<=set(remaining)
        or control.get('completed_key_sha256')!=digest(sorted(completed))
        or control.get('remaining_key_sha256')!=digest(sorted(remaining))):
        raise ValueError('exact95 preserved plus850 remaining original-grid partition required')
    policy=control.get('policy_sha256')
    if (not isinstance(policy,str) or len(policy)!=64
        or any(p.get('policy_sha256')!=policy for p in (old_plan,old_terminal,new_plan))
        or old_plan.get('split')!='validation' or new_plan.get('split')!='validation'
        or old_plan.get('expected_cases')!=945 or new_plan.get('expected_cases')!=945):
        raise ValueError('resume cannot change the frozen execution policy or split')
    if (set(old_status.get('completed',{}))!=set(completed) or set(old_status.get('failed',{}))!=set(interrupted)
        or old_status.get('active')!=[] or old_status.get('stop_requested') is not True
        or old_terminal.get('complete') is not False or old_terminal.get('stop_requested') is not True
        or old_terminal.get('completed')!=95 or old_terminal.get('failed')!=2 or old_terminal.get('expected')!=945):
        raise ValueError('frozen previous user-STOP terminal must explain all95 accepted and2 interrupted cases')
    for name in interrupted:
        failed=old_status['failed'][name]
        if (failed.get('stage')!='native' or failed.get('complete') is not False
            or failed.get('interrupted') is not True or failed.get('timed_out') is not False):
            raise ValueError('only explicit user-interrupted native attempts may be replaced')
    previous=(ROOT/control['previous_runtime']).resolve();runtime=Path(runtime).resolve()
    if runtime!=(ROOT/RUNTIME).resolve() or previous==runtime:
        raise ValueError('fresh separately named resume runtime required')
    for name,row in new.items():
        imported=name in set(completed)
        if type(row.get('imported_completed')) is not bool or row['imported_completed'] is not imported:
            raise ValueError('resume provenance cannot relabel imported or new native cases')
        expected_runtime=previous if imported else runtime
        if (ROOT/row.get('evidence_runtime','')).resolve()!=expected_runtime:
            raise ValueError('each row must use its real old or new evidence runtime')
        actual={key:value for key,value in row.items() if key not in ROW_PROVENANCE}
        wanted=deepcopy(old[name])
        if not imported:
            new_out=str(Path(OUTPUT)/row['task']/row['raw']/row['method'])
            # option() rejects repeated/missing options; only this one value
            # may differ. Scientific identity has no output-path component.
            option(wanted['argv'],'--out');option(actual['argv'],'--out')
            wanted['argv'][wanted['argv'].index('--out')+1]=new_out
        if actual!=wanted:
            raise ValueError('resume row changed beyond the authorized new output namespace: '+name)
    return old,new


def require_owner_launch(launch,terminal,control,*,suite_lock_sha,source_lock_sha,driver,guard):
    protocol=control['author_outcome_protocol']
    if (launch.get('source_sha256')!=control['resume_owner']['sha256']
        or terminal.get('complete') is not True or terminal.get('expected')!=945
        or terminal.get('completed')!=945 or terminal.get('failed')!=0
        or launch.get('split')!='validation' or terminal.get('split')!='validation'
        or launch.get('suite_lock_sha256')!=suite_lock_sha or launch.get('split_source_lock_sha256')!=source_lock_sha
        or launch.get('policy_sha256')!=control['policy_sha256'] or terminal.get('policy_sha256')!=control['policy_sha256']
        or launch.get('admission_driver')!=driver or launch.get('admission_guard')!=guard
        or launch.get('execution_protocol_changed') is not True
        or launch.get('author_outcome_protocol')!=protocol or terminal.get('author_outcome_protocol')!=protocol
        or launch.get('native_runtime')!=protocol['runtime_consumer'] or launch.get('original_native_runtime')!=protocol['original_runtime']):
        raise ValueError('complete actual v772 owner launch/terminal required; never impersonate v753')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','suite','suite-lock','source-inputs','split','resume-control','resume-lock','out'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if args.split!='validation':raise ValueError('this resume auditor admits VAL only, never TEST')
    runtime=(ROOT/args.runtime).resolve();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('fresh project resume audit output required')
    files=json.loads((ROOT/args.suite_lock).read_text())['files'];verify_inputs(ROOT,files)
    suite_path=locked_pointer(ROOT,files,pointer(ROOT/args.suite));suite=json.loads(suite_path.read_text())
    driver,guard=base.driver_bindings(suite,files);protocol=suite['author_outcome_protocol']
    control_path=(ROOT/args.resume_control).resolve();resume_lock_path=(ROOT/args.resume_lock).resolve()
    lock=json.loads(resume_lock_path.read_text())
    if lock.get('schema')!=LOCK_SCHEMA or lock.get('approved_for_execution') is not True:
        raise ValueError('independently frozen resume-control supplement required')
    supplement=lock['files'];verify_inputs(ROOT,supplement)
    locked_pointer(ROOT,supplement,pointer(control_path));control=json.loads(control_path.read_text())
    if control.get('author_outcome_protocol')!=protocol:
        raise ValueError('resume must retain the original frozen author-outcome protocol')
    inputs={}
    def evidence(path,*,frozen=False):
        item=pointer(path);absolute=str((ROOT/item['path']).resolve())
        if frozen:locked_pointer(ROOT,supplement,item)
        if absolute in inputs and inputs[absolute]!=item['sha256']:raise ValueError('conflicting actual evidence SHA')
        inputs[absolute]=item['sha256'];return item
    # The immutable supplement file pins inherited external model/native
    # dependencies; the release's own evidence map remains project-only.
    inputs.update(project_resume_inputs(supplement,files))
    for field,path in (('suite',suite_path),('suite_lock',ROOT/args.suite_lock),('source_inputs',ROOT/args.source_inputs)):
        if control.get(field)!=pointer(path):raise ValueError('resume changed original '+field)
    for field,path in (('resume_owner',ROOT/OWNER),('resume_auditor',Path(__file__))):
        if control.get(field)!=pointer(path):raise ValueError('explicit actual resume implementation required: '+field)
        evidence(path,frozen=True)
    for field in ('user_stop_record','user_resume_record'):
        locked_pointer(ROOT,supplement,control[field]);evidence(ROOT/control[field]['path'],frozen=True)
    previous=(ROOT/control['previous_runtime']).resolve()
    if not previous.is_relative_to(ROOT/'runtime/framework'):raise ValueError('original project runtime required')
    original={}
    for field,filename in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json')):
        path=previous/filename
        if control.get(field)!=pointer(path):raise ValueError('exact immutable original stage pointer required: '+field)
        evidence(path,frozen=True);original[field]=json.loads(path.read_text())
    if original['previous_launch'].get('source_sha256')!=file_sha(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py'):
        raise ValueError('imported cases must originate from the real frozen753 owner')
    old_launch=original['previous_launch']
    if (old_launch.get('split')!='validation' or old_launch.get('policy_sha256')!=control['policy_sha256']
        or old_launch.get('suite_lock_sha256')!=control['suite_lock']['sha256']
        or old_launch.get('split_source_lock_sha256')!=control['source_inputs']['sha256']
        or old_launch.get('admission_driver')!=driver or old_launch.get('admission_guard')!=guard
        or old_launch.get('author_outcome_protocol')!=protocol):
        raise ValueError('original accepted stage did not bind this exact unchanged suite and policy')
    manifest,sources,proof=read_formal_source(ROOT,suite,files,'validation')
    if proof['policy_sha256']!=control['policy_sha256']:raise ValueError('resume policy changed')
    source_meta=json.loads((ROOT/args.source_inputs).read_text());source_sha=file_sha(ROOT/args.source_inputs)
    if (source_meta.get('schema')!='r10_full16_post_admission_split_source_lock_v1' or source_meta.get('split')!='validation'
        or source_meta.get('policy_sha256')!=proof['policy_sha256'] or source_meta.get('source_count')!=315
        or source_meta.get('source_manifest_sha256')!=suite['source_manifests']['validation']['sha256']
        or source_meta.get('TEST_used_for_fitting') is not False):raise ValueError('unchanged original VAL data scope required')
    verify_inputs(ROOT,source_meta['files'])
    launch=json.loads((runtime/'launch.json').read_text());terminal=json.loads((runtime/'terminal.json').read_text());plan=json.loads((runtime/'plan.json').read_text())
    old,new=require_resume_plan(control,original['previous_plan'],original['previous_status'],original['previous_terminal'],plan,runtime=runtime)
    require_owner_launch(launch,terminal,control,suite_lock_sha=file_sha(ROOT/args.suite_lock),source_lock_sha=source_sha,driver=driver,guard=guard)
    if launch.get('resume_control')!=pointer(control_path) or launch.get('resume_control_lock')!=pointer(resume_lock_path):
        raise ValueError('actual resumed owner must expose the exact frozen resume control and lock')
    for path in (suite_path,ROOT/args.suite_lock,ROOT/args.source_inputs,control_path,resume_lock_path,
                 runtime/'launch.json',runtime/'plan.json',runtime/'terminal.json',ROOT/driver['path'],ROOT/guard['path'],ROOT/suite['formal_result_auditor']['path']):evidence(path)
    # The two censored attempts remain evidence, not SR negatives or selected
    # completed cases. Their authorized replacements still occupy the same keys.
    for name in control['interrupted_job_ids']:
        old_terminal_path=previous/'native'/name/'terminal.json'
        old_result=json.loads(old_terminal_path.read_text());evidence(old_terminal_path,frozen=True)
        if old_result.get('complete') is not False or old_result.get('interrupted') is not True or old_result.get('timed_out') is not False:
            raise ValueError('replaced attempt is not an explicit non-timeout user interruption')
        old_out=ROOT/option(old[name]['argv'],'--out')
        if (old_out/'case.json').exists() or (old_out/'admission_driver_receipt.json').exists():
            raise ValueError('a completed historical case cannot be replaced as interrupted')
    sources_by_key={(r['task'],r['raw_episode_id']):r for r in sources};cases=[];case_pointers=[];driver_pointers=[]
    profile=fingerprint(proof['routes']['common_author_RML_profile'])
    for row in plan['jobs']:
        name=row['job_id'];imported=row['imported_completed'];stage=(ROOT/row['evidence_runtime']).resolve()
        CPU_leaf=stage/'CPU'/name;native_leaf=stage/'native'/name
        CPU=json.loads((CPU_leaf/'CPU_preflight.json').read_text());native_result=json.loads((native_leaf/'terminal.json').read_text())
        if native_result.get('complete') is not True:raise ValueError('every selected grid key needs clean completed native execution')
        route=next(r for r in proof['routes']['routes'] if (r['task'],r['method'])==(row['task'],row['method']))
        expected=dict(admission_driver=driver,admission_guard=guard,case_worker=route['worker'],
            native_runtime={k:route['consumer'][k] for k in ('path','sha256')},original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol)
        if (row['worker_sha256']!=driver['sha256'] or any(row.get(k)!=v for k,v in expected.items())
            or row['argv'][:3]!=['bash','scripts/run_aligned_gpu.sh' if row['method']=='physmani' else 'scripts/run_local_sim.sh',driver['path']]):
            raise ValueError('actual original751/743/744 executable route required')
        for flag,value in (('--config',route['configuration']['path']),('--frozen-inputs',route['frozen_inputs']['path']),
            ('--manifest',suite['source_manifests']['validation']['path']),('--suite',args.suite),('--suite-lock',args.suite_lock),
            ('--source-inputs',args.source_inputs),('--split','validation'),('--task',row['task']),('--raw-episode-id',row['raw']),('--method',row['method'])):
            if option(row['argv'],flag)!=value:raise ValueError('actual unchanged VAL argv required: '+flag)
        if (CPU!=json.loads((CPU_leaf/'stdout.log').read_text()) or CPU.get('complete') is not True
            or CPU.get('native_started') is not False or CPU.get('GPU_started') is not False
            or CPU.get('execution_protocol_changed') is not True or CPU.get('actual_split')!='validation'
            or CPU.get('component_domain',{}).get('purpose')!='evaluation' or any(CPU.get(k)!=v for k,v in expected.items())
            or CPU['identity'].get('source_row_sha256')!=digest(sources_by_key[(row['task'],row['raw'])])
            or CPU['identity'].get('config_sha256')!=route['configuration']['sha256']
            or CPU['identity'].get('frozen_inputs_sha256')!=route['frozen_inputs']['sha256']
            or CPU['identity'].get('source_input_lock_sha256')!=source_sha):raise ValueError('actual source/config/domain CPU proof changed')
        stage_owner=original['previous_launch'] if imported else launch
        for leaf,check in ((CPU_leaf,True),(native_leaf,False)):
            actual=json.loads((leaf/'launch.json').read_text())
            base.require_actual_launch(actual,row,expected,check=check,identity=CPU['identity'])
            if actual.get('owner_pid')!=stage_owner.get('pid'):raise ValueError('actual child belongs to another stage owner')
            for filename in ('launch.json','stdout.log','stderr.log','terminal.json'):
                evidence(leaf/filename,frozen=imported)
        evidence(CPU_leaf/'CPU_preflight.json',frozen=imported)
        path=ROOT/option(row['argv'],'--out')/'case.json';case=json.loads(path.read_text())
        driver_path=path.parent/'admission_driver_receipt.json';external=json.loads(driver_path.read_text())
        base.require_driver_evidence(external,driver,guard,route,path,'validation');base.require_native_protocol(case,protocol)
        case_pointer=evidence(path,frozen=imported);driver_pointer=evidence(driver_path,frozen=imported)
        if native_result.get('case')!=case_pointer or native_result.get('admission_driver_receipt')!=driver_pointer:
            raise ValueError('actual native terminal must bind case and external driver bytes')
        if imported:
            accepted=original['previous_status']['completed'][name]
            if accepted.get('case')!=case_pointer or accepted.get('admission_driver_receipt')!=driver_pointer:
                raise ValueError('imported case differs from the exact prior accepted evidence')
        validate_completed_case(case,CPU['identity']);require_attestation(case.get('common_author_RML_parameters',{}),profile)
        if (case.get('cleanup_completed') is not True or case['identity'].get('source_input_lock_sha256')!=source_sha
            or case.get('component_domain',{}).get('purpose')!='evaluation' or case['component_domain'].get('split')!='validation'
            or len(case.get('all_stage_counted_joint_motion',{}).get('events',[]))!=case['counted_clock']['ticks']):
            raise ValueError('genuine full native VAL cleanup/domain/motion evidence required')
        cases.append(case);case_pointers.append(case_pointer);driver_pointers.append(driver_pointer)
    result=validation_release(cases,manifest,proof['routes'])
    per_task=[]
    for task in TASKS:
        per_method={}
        for method in METHODS:
            selected=[c for c in cases if (c['task'],c['method'])==(task,method)];successes=sum(c['official_task_success'] for c in selected)
            per_method[method]=dict(successes=successes,episodes=len(selected),success_rate=successes/len(selected),
                native_terminal_failures=sum(c['native_terminal_failure'] for c in selected),
                actual_window_acceleration_peak=max(c['all_stage_counted_joint_motion']['peak_actual_window_acceleration'] for c in selected),
                measured_motion_violation_cases=sum(not c['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed'] for c in selected))
        per_task.append(dict(task=task,methods=per_method))
    atomic_json(out,dict(result,created_utc=datetime.now(timezone.utc).isoformat(),cases=case_pointers,per_task=per_task,
        admission_driver_receipts=driver_pointers,inputs_sha256=inputs,admission_driver=driver,admission_guard=guard,
        execution_protocol_changed=True,native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol,
        source_sha256=file_sha(__file__),base_result_auditor=suite['formal_result_auditor'],
        resume_control=pointer(control_path),resume_control_lock=pointer(resume_lock_path),
        previous_stage_owner=control['previous_terminal'],actual_stage_owner=pointer(runtime/'terminal.json'),
        imported_completed_cases=95,newly_completed_cases=850,user_interrupted_attempts_preserved=2,
        source_input_lock=pointer(ROOT/args.source_inputs),one_fixed_original_checkpoint_not_top3_selected=True,
        source_outcome_filtering=False,native_failures_retained=True,Value_NGFF_FASTER_used=False))
    print(json.dumps(dict(complete=True,split='validation',cases=945,imported=95,new_native=850,TEST_permitted=True,out=str(out.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
