"""Audit all945 VAL cases across exact historical owners and frozen OS recovery.

The frozen v754 checks remain the base per-case checks. Old accepted cases
retain their actual argv, output paths and log provenance; new attempts retain
their real v823 owner and namespace identity. The actual auditor is824,never754.
"""
import argparse,json,sys
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_formal_suite16_results_v754_r10 as base
from scripts import audit_resumed_formal_VAL_v773_r10 as legacy
from scripts import run_frozen_system_environment_v820_r10 as environment
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS,option
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.formal_suite_admission_v728 import locked_pointer,read_formal_source
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release_v531 import validation_release
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS

CONTROL_SCHEMA='r10_frozen_environment_formal_VAL_resume_control_v823'
LOCK_SCHEMA='r10_frozen_environment_formal_VAL_resume_inputs_v823'
OWNER='scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py'
RUNTIME='runtime/framework/formal_VAL_env_resume_v823'
OUTPUT='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/validation'
ROW_PROVENANCE={'evidence_runtime','imported_completed','previous_row_sha256','previous_imported_completed'}
PREVIOUS_RUNTIME='runtime/framework/formal_VAL_resume_v772'
ENVIRONMENT_FIELDS=('environment_wrapper','environment_manifest','environment_namespace_proof')


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
    """Pure complete-grid and nested-row preservation check; no source reads."""
    if (control.get('schema')!=CONTROL_SCHEMA or control.get('split')!='validation'
        or control.get('user_authorized_resume') is not True or control.get('expected_cases')!=945
        or control.get('standing_authorization_reused') is not True
        or control.get('new_user_event_invented') is not False
        or control.get('physical_policy_changed') is not False):
        raise ValueError('existing standing authorization and unchanged945 VAL environment recovery required')
    old_jobs=old_plan.get('jobs',[]);new_jobs=new_plan.get('jobs',[])
    old={r['job_id']:r for r in old_jobs};new={r['job_id']:r for r in new_jobs}
    completed=control.get('completed_job_ids');remaining=control.get('remaining_job_ids');failed=control.get('failed_job_ids')
    groups=(completed,remaining,failed)
    if not all(isinstance(g,list) and all(isinstance(n,str) for n in g) and len(g)==len(set(g)) for g in groups):
        raise ValueError('unique explicit original-grid partitions required')
    if (len(old_jobs)!=len(old)!=945 or len(new_jobs)!=len(new)!=945 or set(new)!=set(old)
        or not completed or not remaining or not failed or set(completed)&set(remaining)
        or set(completed)|set(remaining)!=set(old) or not set(failed)<=set(remaining)
        or any(control.get(field)!=digest(sorted(group)) for field,group in
               (('completed_key_sha256',completed),('remaining_key_sha256',remaining),('failed_key_sha256',failed)))):
        raise ValueError('all accepted cases must be preserved and every remaining key completed exactly once')
    policy=control.get('policy_sha256')
    if (not isinstance(policy,str) or len(policy)!=64
        or any(p.get('policy_sha256')!=policy for p in (old_plan,old_terminal,new_plan))
        or any(p.get('split')!='validation' or p.get('expected_cases')!=945 for p in (old_plan,new_plan))):
        raise ValueError('environment recovery cannot alter split,policy or full945 grid')
    if (set(old_status.get('completed',{}))!=set(completed) or set(old_status.get('failed',{}))!=set(failed)
        or old_status.get('active')!=[] or old_status.get('checking')!=[]
        or old_status.get('stop_requested') is not False or old_status.get('drain_requested') is not True
        or old_terminal.get('complete') is not False or old_terminal.get('stop_requested') is not False
        or old_terminal.get('drain_requested') is not True or old_terminal.get('drained') is not True
        or old_terminal.get('completed')!=len(completed)
        or old_terminal.get('failed')!=len(failed) or old_terminal.get('expected')!=945):
        raise ValueError('immutable drained old owner must explain every accepted and infrastructure-failed key')
    for name in failed:
        result=old_status['failed'][name]
        if (result.get('stage')!='native' or result.get('complete') is not False
            or type(result.get('returncode')) is not int or result['returncode']==0
            or result.get('interrupted') is not False or result.get('timed_out') is not False
            or result.get('automatic_retry') is not False):
            raise ValueError('environment replacement is infrastructure error,never a selected native outcome')
    previous=(ROOT/control['previous_runtime']).resolve();runtime=Path(runtime).resolve()
    if previous!=(ROOT/PREVIOUS_RUNTIME).resolve() or runtime!=(ROOT/RUNTIME).resolve() or previous==runtime:
        raise ValueError('separate exact environment recovery namespace required')
    for name,row in new.items():
        prior=old[name];imported=name in set(completed)
        if (type(row.get('imported_completed')) is not bool or row['imported_completed'] is not imported
            or type(prior.get('imported_completed')) is not bool
            or type(row.get('previous_imported_completed')) is not bool
            or row['previous_imported_completed'] is not prior['imported_completed']
            or row.get('previous_row_sha256')!=digest(prior)):
            raise ValueError('current import and complete previous-row identity must remain explicit: '+name)
        actual={key:value for key,value in row.items() if key not in {'previous_row_sha256','previous_imported_completed'}}
        wanted=deepcopy(prior)
        if imported:
            actual['imported_completed']=row['previous_imported_completed']
        else:
            wanted['argv'][wanted['argv'].index('--out')+1]=str(Path(OUTPUT)/row['task']/row['raw']/row['method'])
            wanted['evidence_runtime']=RUNTIME;wanted['imported_completed']=False
        option(actual['argv'],'--out');option(wanted['argv'],'--out')
        if actual!=wanted:
            raise ValueError('environment resume changed old row beyond new output/provenance: '+name)
    return old,new


def require_environment_receipt(receipt,control,*,owner_pid=None,expected=None):
    """Validate captured820 bindings; runtime capture remains the sole OS probe."""
    if not isinstance(receipt,dict):raise ValueError('actual frozen environment receipt required')
    bindings=dict(wrapper=control.get('environment_wrapper'),manifest=control.get('environment_manifest'),
                  namespace_qualification=control.get('environment_namespace_proof'))
    current=receipt.get('current_mount_namespace');reference=receipt.get('reference_mount_namespace')
    sha=receipt.get('current_OS_target_SHA_map_sha256')
    restored=receipt.get('restored_four_target_SHA256')
    if (receipt.get('schema')!='r10_actual_frozen_OS_namespace_receipt_v820'
        or receipt.get('complete') is not True or type(receipt.get('checked_OS_files')) is not int
        or receipt.get('checked_OS_files')!=6418
        or receipt.get('all_current_OS_targets_full_byte_SHA_verified') is not True
        or receipt.get('host_PID_network_namespace_preserved') is not True
        or receipt.get('proc_or_dev_replaced') is not False
        or receipt.get('task_parameters_or_strategy_changed') is not False
        or receipt.get('project_writable') is not True
        or any(receipt.get(key)!=value or not isinstance(value,dict) for key,value in bindings.items())
        or not isinstance(current,str) or not current or current!=receipt.get('mount_namespace')
        or not isinstance(reference,str) or not reference or current==reference
        or receipt.get('qualified_host_mount_namespace')!=reference
        or type(receipt.get('observer_pid')) is not int or receipt['observer_pid']<=0
        or receipt.get('pid')!=receipt['observer_pid']
        or type(receipt.get('observer_start_ticks')) is not int or receipt['observer_start_ticks']<=0
        or type(receipt.get('outer_wrapper_pid')) is not int or receipt['outer_wrapper_pid']<=0
        or type(receipt.get('outer_wrapper_start_ticks')) is not int or receipt['outer_wrapper_start_ticks']<=0
        or not isinstance(sha,str) or len(sha)!=64 or set(sha)-set('0123456789abcdef')
        or not isinstance(restored,dict) or set(restored)!=set(environment.RESTORED_TARGETS)
        or any(not isinstance(value,str) or len(value)!=64 or set(value)-set('0123456789abcdef') for value in restored.values())):
        raise ValueError('qualified actual6418-file private namespace receipt required')
    if owner_pid is not None and (type(owner_pid) is not int or receipt['observer_pid']!=owner_pid):
        raise ValueError('environment receipt must belong to the real stage owner PID')
    if expected is not None:
        for field in ('schema','manifest','wrapper','source_locks','checked_OS_files',
                      'current_OS_target_SHA_map_sha256','restored_four_target_SHA256',
                      'namespace_qualification','qualified_host_mount_namespace','reference_mount_namespace'):
            if receipt.get(field)!=expected.get(field):raise ValueError('frozen environment binding changed: '+field)
    return receipt


def capture_environment(control):
    """Called only by real future entry points inside the qualified wrapper."""
    wrapper=ROOT/'scripts/run_frozen_system_environment_v820_r10.py'
    for field in ENVIRONMENT_FIELDS:
        value=control.get(field)
        if not isinstance(value,dict) or set(value)!={'path','sha256'} or pointer(ROOT/value['path'])!=value:
            raise ValueError('frozen environment pointer changed: '+field)
    if control['environment_wrapper']!=pointer(wrapper):raise ValueError('actual820 outer wrapper required')
    manifest=control['environment_manifest'];qualification=control['environment_namespace_proof']
    receipt=environment.capture_namespace_evidence(ROOT/manifest['path'],manifest['sha256'],
        ROOT/qualification['path'],qualification['sha256'])
    return require_environment_receipt(receipt,control)


def require_owner_launch(launch,terminal,control,*,suite_lock_sha,source_lock_sha,driver,guard,expected_environment=None):
    legacy.require_owner_launch(launch,terminal,control,suite_lock_sha=suite_lock_sha,
        source_lock_sha=source_lock_sha,driver=driver,guard=guard)
    first=require_environment_receipt(launch.get('frozen_environment'),control,
        owner_pid=launch.get('pid'),expected=expected_environment)
    last=require_environment_receipt(terminal.get('frozen_environment'),control,
        owner_pid=launch.get('pid'),expected=first)
    for field in ('current_mount_namespace','observer_pid','observer_start_ticks','outer_wrapper_pid','outer_wrapper_start_ticks'):
        if first[field]!=last[field]:raise ValueError('stage environment changed during owner execution: '+field)


def require_inherited_environment(actual,stage_owner,control,expected_environment):
    first=require_environment_receipt(stage_owner.get('frozen_environment'),control,
        owner_pid=stage_owner.get('pid'),expected=expected_environment)
    if actual.get('inherited_frozen_environment')!=first or actual.get('current_mount_namespace')!=first['current_mount_namespace']:
        raise ValueError('actual child must inherit its recorded real823 owner namespace')


def require_clean_terminal(result):
    if (result.get('complete') is not True or type(result.get('returncode')) is not int or result.get('returncode')!=0
        or result.get('timed_out') is not False or result.get('interrupted') is not False
        or result.get('automatic_retry') is not False):
        raise ValueError('each selected CPU/native terminal must be a clean one-time completion')



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
    current_environment=capture_environment(control)
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
    for field in ('user_resume_record','standing_continuation_record',*ENVIRONMENT_FIELDS,
                  'previous_resume_control','previous_resume_control_lock'):
        locked_pointer(ROOT,supplement,control[field]);evidence(ROOT/control[field]['path'],frozen=True)
    standing=json.loads((ROOT/control['standing_continuation_record']['path']).read_text())
    if standing.get('continue_until_user_stop') is not True or standing.get('user_stop_received') is not False:
        raise ValueError('real standing continuation must remain active without an invented user event')
    old_control=json.loads((ROOT/control['previous_resume_control']['path']).read_text())
    old_lock=json.loads((ROOT/control['previous_resume_control_lock']['path']).read_text())
    if (old_control.get('schema')!=legacy.CONTROL_SCHEMA or old_lock.get('schema')!=legacy.LOCK_SCHEMA
        or old_lock.get('approved_for_execution') is not True):
        raise ValueError('exact previous user-authorized772 control graph required')
    for name,checksum in old_lock['files'].items():
        if supplement.get(name)!=checksum:raise ValueError('environment lock must retain all exact old772 inputs')
    for field in ('suite','suite_lock','source_inputs','user_resume_record'):
        if control.get(field)!=old_control.get(field):raise ValueError('environment recovery changed original '+field)
    if control.get('author_outcome_protocol')!=old_control.get('author_outcome_protocol'):
        raise ValueError('nested author protocol changed')

    previous=(ROOT/control['previous_runtime']).resolve()
    if not previous.is_relative_to(ROOT/'runtime/framework'):raise ValueError('original project runtime required')
    original={}
    for field,filename in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json')):
        path=previous/filename
        if control.get(field)!=pointer(path):raise ValueError('exact immutable original stage pointer required: '+field)
        evidence(path,frozen=True);original[field]=json.loads(path.read_text())
    if original['previous_launch'].get('source_sha256')!=file_sha(ROOT/'scripts/run_resumed_formal_VAL_owner_v772_r10.py'):
        raise ValueError('previous accepted tier must originate from the real frozen772 owner')
    old_launch=original['previous_launch']
    if (old_launch.get('split')!='validation' or old_launch.get('policy_sha256')!=control['policy_sha256']
        or old_launch.get('suite_lock_sha256')!=control['suite_lock']['sha256']
        or old_launch.get('split_source_lock_sha256')!=control['source_inputs']['sha256']
        or old_launch.get('admission_driver')!=driver or old_launch.get('admission_guard')!=guard
        or old_launch.get('author_outcome_protocol')!=protocol):
        raise ValueError('original accepted stage did not bind this exact unchanged suite and policy')
    if (old_launch.get('resume_control')!=control['previous_resume_control']
        or old_launch.get('resume_control_lock')!=control['previous_resume_control_lock']):
        raise ValueError('old772 launch must bind its genuine old resume graph')
    ancestor=(ROOT/old_control['previous_runtime']).resolve();ancestor_state={}
    for field,filename in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json')):
        path=ancestor/filename
        if old_control[field]!=pointer(path):raise ValueError('nested original753 evidence pointer changed')
        evidence(path,frozen=True);ancestor_state[field]=json.loads(path.read_text())
    legacy.require_resume_plan(old_control,ancestor_state['previous_plan'],ancestor_state['previous_status'],
        ancestor_state['previous_terminal'],original['previous_plan'],runtime=previous)
    ancestor_launch=ancestor_state['previous_launch']
    if (ancestor_launch.get('source_sha256')!=file_sha(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py')
        or ancestor_launch.get('policy_sha256')!=control['policy_sha256']
        or ancestor_launch.get('admission_driver')!=driver or ancestor_launch.get('admission_guard')!=guard
        or ancestor_launch.get('author_outcome_protocol')!=protocol
        or ancestor_launch.get('suite_lock_sha256')!=control['suite_lock']['sha256']
        or ancestor_launch.get('split_source_lock_sha256')!=control['source_inputs']['sha256']
        or ancestor_launch.get('split')!='validation'):
        raise ValueError('original753 physical/source policy did not match the nested lineage')
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
    require_owner_launch(launch,terminal,control,suite_lock_sha=file_sha(ROOT/args.suite_lock),source_lock_sha=source_sha,driver=driver,guard=guard,expected_environment=current_environment)
    if launch.get('resume_control')!=pointer(control_path) or launch.get('resume_control_lock')!=pointer(resume_lock_path):
        raise ValueError('actual resumed owner must expose the exact frozen resume control and lock')
    for path in (suite_path,ROOT/args.suite_lock,ROOT/args.source_inputs,control_path,resume_lock_path,
                 runtime/'launch.json',runtime/'plan.json',runtime/'terminal.json',ROOT/driver['path'],ROOT/guard['path'],ROOT/suite['formal_result_auditor']['path']):evidence(path)
    # Environment admission/post-run errors are preserved,never SR negatives.
    for name in control['failed_job_ids']:
        old_terminal_path=previous/'native'/name/'terminal.json'
        old_result=json.loads(old_terminal_path.read_text());evidence(old_terminal_path,frozen=True)
        if (old_result.get('complete') is not False or old_result.get('interrupted') is not False
            or old_result.get('timed_out') is not False or old_result.get('returncode')==0):
            raise ValueError('replaced environment attempt must remain incomplete infrastructure evidence')
        stderr_path=previous/'native'/name/'stderr.log';evidence(stderr_path,frozen=True)
        last_line=[line.strip() for line in stderr_path.read_text().splitlines() if line.strip()][-1]
        prefix='our_mani.multitask.protocol.MultitaskProtocolError: frozen input SHA changed: '
        if not last_line.startswith(prefix) or last_line[len(prefix):] not in environment.RESTORED_TARGETS:
            raise ValueError('only the exact frozen OS environment mismatch may be replaced')
        old_out=ROOT/option(old[name]['argv'],'--out')
        if (old_out/'case.json').exists() or (old_out/'admission_driver_receipt.json').exists():
            raise ValueError('a completed historical case cannot be replaced as infrastructure')
    stage_owners={str(previous):old_launch,str(ancestor):ancestor_launch,str(runtime):launch}
    sources_by_key={(r['task'],r['raw_episode_id']):r for r in sources};cases=[];case_pointers=[];driver_pointers=[]
    profile=fingerprint(proof['routes']['common_author_RML_profile'])
    for row in plan['jobs']:
        name=row['job_id'];imported=row['imported_completed'];stage=(ROOT/row['evidence_runtime']).resolve()
        CPU_leaf=stage/'CPU'/name;native_leaf=stage/'native'/name
        CPU=json.loads((CPU_leaf/'CPU_preflight.json').read_text());native_result=json.loads((native_leaf/'terminal.json').read_text())
        require_clean_terminal(native_result)
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
        if str(stage) not in stage_owners:raise ValueError('unrecognized nested evidence owner')
        stage_owner=stage_owners[str(stage)]
        for leaf,check in ((CPU_leaf,True),(native_leaf,False)):
            actual=json.loads((leaf/'launch.json').read_text())
            require_clean_terminal(json.loads((leaf/'terminal.json').read_text()))
            base.require_actual_launch(actual,row,expected,check=check,identity=CPU['identity'])
            if actual.get('owner_pid')!=stage_owner.get('pid'):raise ValueError('actual child belongs to another stage owner')
            if not imported:require_inherited_environment(actual,stage_owner,control,current_environment)
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
        if imported and stage==ancestor:
            accepted=ancestor_state['previous_status']['completed'].get(name,{})
            if accepted.get('case')!=case_pointer or accepted.get('admission_driver_receipt')!=driver_pointer:
                raise ValueError('original753 case differs from exact original accepted evidence')
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
        imported_completed_cases=len(control['completed_job_ids']),newly_completed_cases=len(control['remaining_job_ids']),
        environment_failed_attempts_preserved=len(control['failed_job_ids']),
        user_interrupted_attempts_preserved=len(old_control['interrupted_job_ids']),
        provenance_runtime_counts={name:sum(str((ROOT/row['evidence_runtime']).resolve())==name for row in plan['jobs']) for name in stage_owners},
        environment_wrapper=control['environment_wrapper'],environment_manifest=control['environment_manifest'],
        environment_namespace_proof=control['environment_namespace_proof'],actual_auditor_environment=current_environment,
        previous_resume_control=control['previous_resume_control'],previous_resume_control_lock=control['previous_resume_control_lock'],
        standing_authorization_reused=True,new_user_event_invented=False,physical_policy_changed=False,
        source_input_lock=pointer(ROOT/args.source_inputs),one_fixed_original_checkpoint_not_top3_selected=True,
        source_outcome_filtering=False,native_failures_retained=True,Value_NGFF_FASTER_used=False))
    print(json.dumps(dict(complete=True,split='validation',cases=945,imported=len(control['completed_job_ids']),new_native=len(control['remaining_job_ids']),TEST_permitted=True,out=str(out.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
