"""Freeze591 unchanged VAL outcomes and354 fresh attempts after OS restoration.

Run only inside820's qualified private namespace. No simulator, training,
TEST source selection, STOP mutation, or retry is performed by this preparer.
"""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,digest,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import fingerprint
from scripts import prepare_formal_resume_v777_r10 as base
from scripts import run_resumed_formal_VAL_owner_v772_r10 as previous_owner
from scripts import audit_resumed_formal_VAL_v773_r10 as previous_auditor
from scripts import run_environment_resumed_formal_VAL_owner_v823_r10 as owner

SETUP='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup'
PREVIOUS=owner.PREVIOUS
SUITE=base.SUITE;SUITE_LOCK=base.SUITE_LOCK;SOURCE_INPUTS=base.SOURCE_INPUTS
OLD_SETUP='artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup'
OWNER='scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py'
AUDITOR='scripts/audit_environment_resumed_formal_VAL_v824_r10.py'
TEST_PREPARER='scripts/prepare_formal_TEST_after_environment_resume_v826_r10.py'
CONTROLLER='scripts/continue_environment_resumed_VAL_TEST_v827_r10.py'
PREPARER='scripts/prepare_environment_formal_resume_v825_r10.py'
CONTROL_SCHEMA=owner.CONTROL_SCHEMA;LOCK_SCHEMA=owner.LOCK_SCHEMA
RESUME_RECORD='runtime/framework/user_resume_v772.json'
STANDING='runtime/framework/latest_executor_continuation_v817.json'
MANIFEST='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json'
NAMESPACE_PROOF='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json'
WRAPPER='scripts/run_frozen_system_environment_v820_r10.py'
RESTORE_RECEIPT='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/frozen_environment_receipt_v820.json'
LIBRARY_NAMESPACE_PROOF='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_library_namespace_inventory_v820.json'
REQUIRED_SOURCES=(OWNER,AUDITOR,PREPARER,TEST_PREPARER,CONTROLLER,WRAPPER,
    'tests/test_frozen_system_environment_v820.py',
    'tests/test_environment_resumed_formal_VAL_owner_v823.py',
    'tests/test_environment_resumed_formal_VAL_audit_v824.py',
    'tests/test_prepare_environment_formal_resume_v825.py',
    'tests/test_formal_TEST_environment_resume_gate_v826.py',
    'tests/test_environment_resumed_VAL_TEST_controller_v827.py')
FAILED=frozenset(('place_cups_on_rotating_frame_high_speed_episode8_physmani',
    'push_moving_button_episode168_context_geometry','push_moving_button_episode206_cv_geometry'))
require=base.require;pointer=base.pointer;freeze_file=base.freeze_file;read_evidence=base.read_evidence


def stopped(root):
    return any((Path(root)/path).exists() for path in (base.GLOBAL_STOP,
        'runtime/framework/ordered_resume_VAL_TEST_v775/STOP',PREVIOUS+'/STOP',
        'runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP',owner.RUNTIME+'/STOP'))


def partition(plan,status,terminal):
    require(plan.get('split')==status.get('split')==terminal.get('split')=='validation','VAL-only recovery required')
    require(plan.get('expected_cases')==status.get('expected')==terminal.get('expected')==945,'same945 grid required')
    rows=plan.get('jobs');require(isinstance(rows,list) and len(rows)==945,'complete previous945 plan required')
    jobs={r['job_id']:r for r in rows};require(len(jobs)==945,'duplicate previous job')
    done=status.get('completed');failed=status.get('failed')
    require(isinstance(done,dict) and len(done)==591,'exact591 completed records required, irrespective of success')
    require(isinstance(failed,dict) and set(failed)==FAILED,'exact three environment failures required')
    require(set(done)<=set(jobs) and not set(done)&set(failed),'foreign/overlapping previous results')
    require(status.get('active')==[] and status.get('checking')==[] and status.get('native_slots')==0
        and status.get('GPU_slots')==0,'previous CPU/native work must have drained')
    require(status.get('drain_requested') is True and status.get('stop_requested') is False
        and terminal.get('complete') is False and terminal.get('drained') is True
        and terminal.get('drain_requested') is True and terminal.get('stop_requested') is False
        and terminal.get('completed')==591 and terminal.get('failed')==3,
        'actual environment DRAIN591/3, not a fabricated userSTOP, is required')
    require(plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False,
        'unfiltered original one-time plan required')
    for row in failed.values():
        require(row.get('stage')=='native' and row.get('complete') is False and row.get('returncode')==1
            and row.get('interrupted') is False and row.get('timed_out') is False
            and row.get('automatic_retry') is False,'preserve exact non-user environment failures')
    tiers={}
    for key in done:
        row=jobs[key];runtime=row.get('evidence_runtime')
        require(runtime in ('runtime/framework/formal_VAL_v753',PREVIOUS),'unknown imported evidence tier')
        require(type(row.get('imported_completed')) is bool
            and row['imported_completed'] is (runtime=='runtime/framework/formal_VAL_v753'),'old772 import marker/tier changed')
        tiers[runtime]=tiers.get(runtime,0)+1
    require(tiers=={'runtime/framework/formal_VAL_v753':95,PREVIOUS:496},'exact95 old753 plus496 old772 required')
    completed=sorted(done);remaining=sorted(set(jobs)-set(done));failed_keys=sorted(failed)
    require(len(remaining)==354 and len(set(remaining)-set(failed))==351,'exact354=3 failed+351 unexecuted required')
    policy=plan.get('policy_sha256')
    require(isinstance(policy,str) and len(policy)==64 and terminal.get('policy_sha256')==policy,'same frozen policy required')
    return jobs,completed,failed_keys,remaining,policy


def require_previous_exited(root,runtime,launch):
    result={'checked_prior_stage_processes':[],'recorded_child_launches_checked':0}
    for name in (PREVIOUS,'runtime/framework/ordered_resume_VAL_TEST_v775','runtime/framework/formal_VAL_report_v776',
                 'runtime/framework/observability_watch_v778'):
        path=Path(root)/name/'launch.json';record=json.loads(path.read_text())
        base.process_exited(record['pid'],record['start_ticks']);result['checked_prior_stage_processes'].append(pointer(root,path))
    for stage in ('CPU','native'):
        for path in sorted((runtime/stage).glob('*/launch.json')):
            record=json.loads(path.read_text());base.process_exited(record['launcher_pid'],record['launcher_start_ticks'])
            result['recorded_child_launches_checked']+=1
    return dict(result,previous_owner_controller_reporter_and_watch_exited=True)


def check_previous_plan(root,files,previous):
    old_control,old_files=previous_owner.load_resume_control(OLD_SETUP+'/resume_control.json',OLD_SETUP+'/resume_inputs.json',SUITE,SUITE_LOCK,SOURCE_INPUTS)
    old=Path(root)/old_control['previous_runtime']
    original={name:json.loads((old/(name+'.json')).read_text()) for name in ('plan','status','terminal')}
    previous_auditor.require_resume_plan(old_control,original['plan'],original['status'],original['terminal'],previous['plan'],runtime=Path(root)/PREVIOUS)
    # Keep the entire old lock so its external graph remains byte-for-byte pinned.
    for key,value in old_files.items():
        require(files.get(key,value)==value,'old752/772 frozen maps conflict: '+key);files[key]=value
    for name in ('resume_control.json','resume_inputs.json'):freeze_file(root,files,OLD_SETUP+'/'+name)


def check_source_plan(root,plan,suite,sources,proof):
    normalized=deepcopy(plan)
    for row in normalized['jobs']:
        row['argv'][row['argv'].index('--out')+1]='artifacts/framework/multitask_r10/formal_suite16_v752/validation/'+'/'.join((row['task'],row['raw'],row['method']))
    return base.check_plan(normalized,suite,sources,proof)


def freeze_tier_owner(root,files,row):
    runtime=Path(root)/row['evidence_runtime'];launch,item=read_evidence(root,files,runtime/'launch.json')
    expected='scripts/run_formal_suite16_owner_v753_r10.py' if row['evidence_runtime']=='runtime/framework/formal_VAL_v753' else 'scripts/run_resumed_formal_VAL_owner_v772_r10.py'
    require(launch.get('source_sha256')==file_sha(Path(root)/expected),'actual old tier owner source changed')
    for stage in ('CPU','native'):
        leaf=runtime/stage/row['job_id'];record,_=read_evidence(root,files,leaf/'launch.json')
        require(record.get('owner_pid')==launch['pid'],'old child real ownerPID/tier changed')
    return runtime


def freeze_environment_failure(root,files,runtime,row,result,identity):
    """Freeze the real failed attempt; never derive a native SR outcome."""
    base.freeze_interrupted(root,files,runtime,row,result,identity)
    stderr=runtime/'native'/row['job_id']/'stderr.log'
    lines=[line.strip() for line in stderr.read_text().splitlines() if line.strip()]
    prefix='our_mani.multitask.protocol.MultitaskProtocolError: frozen input SHA changed: '
    from scripts.run_frozen_system_environment_v820_r10 import RESTORED_TARGETS
    require(bool(lines) and lines[-1].startswith(prefix) and lines[-1][len(prefix):] in RESTORED_TARGETS,
        'only actual frozen OS SHA failures may receive a new attempt')


def freeze_environment_materials(root,files,manifest_pointer):
    """Freeze all project snapshot bytes; inventory pins links/directories too."""
    manifest_path=Path(root)/manifest_pointer['path']
    freeze_file(root,files,manifest_path,expected=manifest_pointer['sha256'])
    manifest,_=read_evidence(root,files,manifest_path)
    require(manifest.get('schema')=='r10_exact_frozen_OS_hybrid_snapshot_v820' and manifest.get('complete') is True,
        'complete qualified hybrid environment manifest required')
    original_pointer=manifest['base_individual_file_manifest']
    original_path=Path(root)/original_pointer['path']
    freeze_file(root,files,original_path,expected=original_pointer['sha256'])
    original,_=read_evidence(root,files,original_path)
    rows=original.get('files',[])
    require(len(rows)==6418,'complete original6418-file snapshot required')
    for row in rows:
        require(not stopped(root),'STOP while freezing environment materials')
        freeze_file(root,files,original_path.parent/row['snapshot_file'],expected=row['sha256'])
    library=manifest['library_binding'];inventory_pointer=library['inventory']
    inventory_path=Path(root)/inventory_pointer['path']
    freeze_file(root,files,inventory_path,expected=inventory_pointer['sha256'])
    inventory,_=read_evidence(root,files,inventory_path)
    directory=Path(library['snapshot_directory']).resolve()
    require(directory.is_relative_to(Path(root).resolve()) and directory==Path(inventory['snapshot_directory']).resolve()
        and inventory.get('complete') is True,'full project library snapshot and topology inventory required')
    regular=0
    for relative,row in inventory['snapshot']['entries'].items():
        require(not stopped(root),'STOP while freezing complete library copy')
        if row['kind']=='file':
            freeze_file(root,files,directory/relative,expected=row['sha256']);regular+=1
    require(regular==4828==library['locked_regular_files'],'all4828 frozen regular library copies required')
    material=manifest['restore_materials'];freeze_file(root,files,material['path'],expected=material['sha256'])
    return dict(original_snapshot_manifest=original_pointer,complete_library_inventory=inventory_pointer,
                individual_snapshot_files=len(rows),complete_library_regular_files=regular,
                links_and_directories_pinned_by_inventory=True)


def prepare(root=ROOT):
    root=Path(root).resolve();require(not stopped(root),'STOP has priority over environment preparation')
    setup=root/SETUP;require(not setup.exists(),'preserve one-time environment-resume setup')
    for name in REQUIRED_SOURCES:require((root/name).is_file(),'new source/test not ready: '+name)
    env={key:pointer(root,name) for key,name in (('environment_manifest',MANIFEST),('environment_wrapper',WRAPPER),('environment_namespace_proof',NAMESPACE_PROOF))}
    actual_environment=owner.require_frozen_environment(env)
    files=dict(json.loads((root/OLD_SETUP/'resume_inputs.json').read_text())['files'])
    suite,sources,proof=base.admit_validation(root,files)
    pointers={key:freeze_file(root,files,name) for key,name in (('suite',SUITE),('suite_lock',SUITE_LOCK),('source_inputs',SOURCE_INPUTS))}
    runtime=root/PREVIOUS;previous={}
    for name in ('launch','plan','status','terminal'):
        previous[name],pointers['previous_'+name]=read_evidence(root,files,runtime/(name+'.json'))
    jobs,completed,failed,remaining,policy=partition(previous['plan'],previous['status'],previous['terminal'])
    require(policy==proof['policy_sha256'],'admitted physical policy differs from previous run')
    check_previous_plan(root,files,previous)
    require(previous['launch'].get('source_sha256')==file_sha(root/'scripts/run_resumed_formal_VAL_owner_v772_r10.py')
        and previous['launch'].get('author_outcome_protocol')==suite['author_outcome_protocol']
        and previous['terminal'].get('author_outcome_protocol')==suite['author_outcome_protocol'],
        'exact previous772 physical/source protocol required')
    exit_proof=require_previous_exited(root,runtime,previous['launch'])
    for item in exit_proof['checked_prior_stage_processes']:freeze_file(root,files,item['path'],expected=item['sha256'])
    for name in ('launch','status','failure'):
        path=root/'runtime/framework/ordered_resume_VAL_TEST_v775'/(name+'.json')
        if path.is_file():pointers['previous_controller_'+name]=freeze_file(root,files,path)
    routes,source_rows=check_source_plan(root,previous['plan'],suite,sources,proof)
    profile=fingerprint(proof['routes']['common_author_RML_profile']);successes=0
    for key in completed+failed:
        require(not stopped(root),'STOP during evidence freeze')
        row=jobs[key];tier=freeze_tier_owner(root,files,row)
        identity=base.check_CPU(root,files,tier,row,source_rows[row['task'],row['raw']])
        if key in previous['status']['completed']:
            successes+=int(base.check_completed(root,files,tier,row,previous['status']['completed'][key],identity,profile,policy))
        else:
            freeze_environment_failure(root,files,tier,row,previous['status']['failed'][key],identity)
    require(previous['terminal'].get('native_successes')==successes,'no completed nativeFalse outcome may be filtered')
    # Preserve any ready-but-never-native CPU logs too; they are not reused.
    for path in sorted((runtime/'CPU').rglob('*')):
        if path.is_file():freeze_file(root,files,path)
    permission,pointers['user_resume_record']=read_evidence(root,files,RESUME_RECORD)
    standing,pointers['standing_continuation_record']=read_evidence(root,files,STANDING)
    pointers['previous_resume_control']=freeze_file(root,files,OLD_SETUP+'/resume_control.json')
    pointers['previous_resume_control_lock']=freeze_file(root,files,OLD_SETUP+'/resume_inputs.json')
    require(permission.get('user_authorized_resume') is True and permission.get('revokes_previous_stop') is True
        and standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False,
        'real existing user continuation, not a new fabricated user event, is required')
    for name in REQUIRED_SOURCES:freeze_file(root,files,name)
    for key,item in env.items():pointers[key]=freeze_file(root,files,item['path'],expected=item['sha256'])
    frozen_materials=freeze_environment_materials(root,files,env['environment_manifest'])
    pointers['environment_restore_receipt']=freeze_file(root,files,RESTORE_RECEIPT)
    pointers['environment_complete_library_proof']=freeze_file(root,files,LIBRARY_NAMESPACE_PROOF)
    require(str((root/base.GLOBAL_STOP).resolve()) not in files and base.GLOBAL_STOP not in files,'mutable STOP cannot be frozen')
    verify_inputs(root,files);require(not stopped(root),'STOP before setup write')
    control=dict(schema=CONTROL_SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),previous_runtime=PREVIOUS,
        completed_job_ids=completed,failed_job_ids=failed,remaining_job_ids=remaining,expected_cases=945,
        completed_key_sha256=digest(completed),failed_key_sha256=digest(failed),remaining_key_sha256=digest(remaining),
        split='validation',policy_sha256=policy,author_outcome_protocol=suite['author_outcome_protocol'],
        resume_owner=pointer(root,OWNER),resume_auditor=pointer(root,AUDITOR),user_authorized_resume=True,
        standing_authorization_reused=True,new_user_event_invented=False,
        previous_process_exit_proof=exit_proof,prepared_frozen_environment=actual_environment,
        frozen_environment_materials=frozen_materials,
        completed_results_selected_by_success=False,previous_environment_failures_are_task_outcomes=False,
        physical_policy_changed=False,models_retrained=False,old_evidence_modified=False,
        native_started=False,GPU_started=False,TEST_read=False,**pointers)
    setup.mkdir(parents=True,exist_ok=False);atomic_json(setup/'resume_control.json',control)
    control_pointer=freeze_file(root,files,setup/'resume_control.json')
    atomic_json(setup/'resume_inputs.json',dict(schema=LOCK_SCHEMA,approved_for_execution=True,files=files,
        control=control_pointer,split='validation',policy_sha256=policy,native_started=False,GPU_started=False,TEST_read=False))
    return dict(complete=True,completed=591,remaining=354,previous_environment_failed=3,unexecuted=351,
        policy_sha256=policy,resume_control=control_pointer,resume_control_lock=pointer(root,setup/'resume_inputs.json'),
        native_started=False,GPU_started=False,TEST_read=False,STOP_modified=False)


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False),flush=True)
