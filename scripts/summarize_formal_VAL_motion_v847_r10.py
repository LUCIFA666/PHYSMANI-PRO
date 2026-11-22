"""Prepare full-VAL saved-motion statistics only after the real824 release.

This standalone reader never launches a job, reads TEST/HDF5/model contents,
probes a live namespace, writes TXT, or grants a release. It consumes the exact
315-source/945-case VAL bijection and the real753/772/823 evidence selected by824.
The audited838 reducer is SHA-pinned and reused without executing its main.
Native False and motion False are valid outcomes, including a zero-success grid.
Zero counted ticks retain a case with unobserved motion peaks left unknown.

Future invocation requires --release-sha256 from the completed824 artifact.
No real-data invocation is part of preparing or unit-testing this script.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import types

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/VAL_release.json'
OUTPUT = 'runtime/framework/formal_VAL_env_resume_v823/full_sixteen_task_motion_summary_v847.json'
CONTROL = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_control.json'
LOCK = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_inputs.json'
SOURCE_MANIFEST = 'artifacts/framework/protocol/validation_manifest.json'
SOURCE_LOCK = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/validation_source_inputs.json'
SUITE = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/suite.json'
SUITE_LOCK = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/frozen_inputs.json'
REDUCER = 'scripts/summarize_completed_VAL_motion_v838_r10.py'
SUBSET = 'runtime/framework/formal_VAL_env_resume_v823/completed_ten_task_motion_summary_v838.json'
AUDITOR = 'scripts/audit_environment_resumed_formal_VAL_v824_r10.py'
AUDIT_STAGE = 'runtime/framework/ordered_env_resume_VAL_TEST_v827/stages/independent_resumed_VAL_release'
CONTROLLER = 'runtime/framework/ordered_env_resume_VAL_TEST_v827'
CONTROLLER_SOURCE = 'scripts/continue_environment_resumed_VAL_TEST_v827_r10.py'
PINS = {CONTROL:'435d684b415b26461005556f339f82c731a90ede873e8b64db2f6f6533d3fa36',
    LOCK:'45cc32204b28a3c3990d4d9f8a86488dd73d211e7b990a80e2152b9706227258',
    SOURCE_MANIFEST:'70498a21eabd28e55235672371a224047ec25f10857b90bdddfde7d24aa9f6bf',
    SOURCE_LOCK:'49389a10fd876194c1e0ff3e1b81d19442ebd57a42df24da2849cf2d32cbb0ab',
    REDUCER:'88d54d03ca2fa86b56a0a99ece158db26b29cd341edd64e951f0f5a1a6e93971',
    SUBSET:'e6d6d43108f403044a7f07689f297034e919851395eb64a267cf25500d019ebd'}
POLICY = 'eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
PROFILE = 'd9a36e5ba8a73123c5deb65229d7fa151089145dca4b55c68c1e9ad1dec4e562'
PARAMETERS = dict(max_velocity=3.,max_acceleration=12.,max_jerk=1000.,counted_dt=.05,decision_ticks=10,budget_seconds=10.)
METHODS = ('cv_geometry','context_geometry','physmani')
COUNTS = {task:20 for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed',
    'insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed','moving_basketball_in_hoop',
    'moving_basketball_in_hoop_high_speed','pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed',
    'place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed','push_moving_button','push_moving_button_high_speed',
    'put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed','remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')}
COUNTS.update(insert_onto_rotating_peg=18,insert_onto_rotating_peg_high_speed=19,put_rubbish_in_moving_bin_high_speed=18)
TASKS = tuple(sorted(COUNTS))
STAGES = {'runtime/framework/formal_VAL_v753':('artifacts/framework/multitask_r10/formal_suite16_v752','scripts/run_formal_suite16_owner_v753_r10.py'),
    'runtime/framework/formal_VAL_resume_v772':('artifacts/framework/multitask_r10/formal_VAL_resume_v772','scripts/run_resumed_formal_VAL_owner_v772_r10.py'),
    'runtime/framework/formal_VAL_env_resume_v823':('artifacts/framework/multitask_r10/formal_VAL_env_resume_v823','scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py')}
CURRENT = 'runtime/framework/formal_VAL_env_resume_v823'
EXPECTED_PROVENANCE = dict(zip(STAGES,(95,496,354)))
BINDINGS = ('admission_driver','admission_guard','case_worker','native_runtime','original_native_runtime','author_outcome_protocol')
SOURCES = {AUDITOR,REDUCER,CONTROLLER_SOURCE,'scripts/audit_formal_suite16_results_v754_r10.py',
    'scripts/run_formal_case_author_outcome_v751_r10.py','scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
    'scripts/run_task_executor_smoke_author_suite16_v744_r10.py','scripts/run_task_executor_smoke_rml_suite16_v664_r10.py',
    'our_mani/multitask/formal_suite_admission_v728.py','our_mani/multitask/native_outcome_sampling_v741.py',
    *(value[1] for value in STAGES.values())}


def require(condition,message):
    if not condition: raise ValueError(message)


def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef')


def option(argv,key):
    require(isinstance(argv,list) and argv.count(key)==1 and argv.index(key)+1<len(argv),'one explicit argv option required: '+key)
    return argv[argv.index(key)+1]


def allowed(name):
    if name in {*PINS,RELEASE,SUITE,SUITE_LOCK,CONTROLLER+'/launch.json',*SOURCES}: return True
    if name.startswith(AUDIT_STAGE+'/') and name[len(AUDIT_STAGE)+1:] in ('launch.json','terminal.json','stdout.log','stderr.log'): return True
    for runtime,(artifact,_) in STAGES.items():
        suffix=name[len(runtime)+1:] if name.startswith(runtime+'/') else ''
        if suffix in ('launch.json','terminal.json','plan.json'): return True
        parts=suffix.split('/')
        if len(parts)==3 and parts[0] in ('CPU','native') and parts[2] in ('launch.json','terminal.json','CPU_preflight.json'): return True
        if name.startswith(artifact+'/validation/') and Path(name).name in ('case.json','admission_driver_receipt.json'): return True
    return False


class Reader:
    """Only consumed VAL evidence is opened; large transitive maps are not walked."""
    def __init__(self,root):
        self.root=Path(root);self.inputs={};self.evidence={};self.frozen={}
    def lookup(self,mapping,name):
        values={mapping[k] for k in (name,str(self.root/name)) if k in mapping}
        require(len(values)==1 and valid_sha(next(iter(values))),'missing/conflicting evidence SHA: '+name)
        return next(iter(values))
    def raw(self,name,*,expected=None,evidence=False,frozen=False):
        require(isinstance(name,str) and not Path(name).is_absolute() and str(PurePosixPath(name))==name
            and '..' not in PurePosixPath(name).parts and allowed(name),'non-VAL or undeclared input refused: '+str(name))
        path=(self.root/name).resolve(strict=True)
        require(path.is_relative_to(self.root) and str(path.relative_to(self.root))==name,
                'canonical project evidence required; symlink aliases are refused before open')
        raw=path.read_bytes();actual=sha_bytes(raw)
        wanted=[v for v in (expected,PINS.get(name),self.lookup(self.evidence,name) if evidence else None,
                            self.lookup(self.frozen,name) if frozen else None) if v is not None]
        require(all(valid_sha(v) and v==actual for v in wanted),'input SHA changed: '+name)
        require(name not in self.inputs or self.inputs[name]==actual,'input changed during summary: '+name)
        self.inputs[name]=actual;return raw
    def json(self,name,**kwargs): return json.loads(self.raw(name,**kwargs))
    def pointer(self,name): return dict(path=name,sha256=self.inputs[name])
    def pointed(self,pointer,**kwargs):
        require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},'explicit path/SHA pointer required')
        return self.json(pointer['path'],expected=pointer['sha256'],**kwargs)


def require_release_header(release):
    require(release.get('schema')=='r10_full16_common_author_RML_validation_release_v531'
        and release.get('complete') is True and release.get('split')=='validation'
        and release.get('source_cases')==315 and release.get('method_cases')==945
        and release.get('completed_method_cases')==945 and release.get('methods')==list(METHODS)
        and release.get('tasks')==list(TASKS) and release.get('policy_sha256')==POLICY
        and release.get('common_RML_profile_sha256')==PROFILE,'real complete16/315/945 VAL release required')
    for key,value in dict(formal_test_permitted=True,TEST_used=False,TEST_used_for_fitting=False,
        motion_is_diagnostic_only=True,motion_metrics_required_for_all_methods=True,native_task_success_unchanged=True,
        native_success_is_not_a_release_threshold=True,source_outcome_filtering=False,native_failures_retained=True,
        Value_NGFF_FASTER_used=False,physical_policy_changed=False).items():
        require(release.get(key) is value,'unchanged diagnostic/native-outcome release semantics required: '+key)
    require(isinstance(release.get('cases'),list) and len(release['cases'])==945
        and isinstance(release.get('admission_driver_receipts'),list) and len(release['admission_driver_receipts'])==945,
        '945 explicit case and driver pointers required')
    require(isinstance(release.get('inputs_sha256'),dict) and release['inputs_sha256'],'actual824 evidence map required')


def require_clean_terminal(value,*,controller_stage=False):
    require(value.get('complete') is True and type(value.get('returncode')) is int and value['returncode']==0
        and value.get('automatic_retry') is False,'clean one-time completion required')
    if not controller_stage:
        require(value.get('timed_out') is False and value.get('interrupted') is False,'uninterrupted CPU/native completion required')


def require_environment(value,control,*,observer=None):
    require(value.get('complete') is True and value.get('schema')=='r10_actual_frozen_OS_namespace_receipt_v820'
        and value.get('checked_OS_files')==6418 and value.get('all_current_OS_targets_full_byte_SHA_verified') is True
        and value.get('host_PID_network_namespace_preserved') is True and value.get('proc_or_dev_replaced') is False
        and value.get('task_parameters_or_strategy_changed') is False
        and value.get('current_mount_namespace')==value.get('mount_namespace')
        and isinstance(value.get('current_mount_namespace'),str)
        and value['current_mount_namespace']!=value.get('reference_mount_namespace'),'recorded qualified environment required')
    for a,b in (('manifest','environment_manifest'),('wrapper','environment_wrapper'),('namespace_qualification','environment_namespace_proof')):
        require(value.get(a)==control.get(b),'recorded environment binding changed: '+a)
    if observer is not None: require(value.get('observer_pid')==observer and value.get('pid')==observer,'actual environment observer PID mismatch')


def source_grid(manifest,source_lock):
    require(manifest.get('split')=='validation' and isinstance(manifest.get('episodes'),list),'original VAL metadata required')
    require(source_lock.get('split')=='validation' and source_lock.get('source_count')==315
        and source_lock.get('policy_sha256')==POLICY and source_lock.get('source_manifest_sha256')==PINS[SOURCE_MANIFEST]
        and source_lock.get('TEST_used_for_fitting') is False,'original315-source VAL lock metadata required')
    sources={}
    for row in manifest['episodes']:
        key=(row.get('task'),row.get('raw_episode_id'))
        require(key[0] in COUNTS and isinstance(key[1],str) and key[1] and key not in sources
            and row.get('split','validation')=='validation' and valid_sha(source_lock['files'].get(row.get('path'))),
            'unique canonical VAL source row and declared data SHA required')
        sources[key]=digest(row)
    require(len(sources)==315 and Counter(k[0] for k in sources)==COUNTS,'exact original sixteen-task315-source roster required')
    return sources


def require_plan(plan,old_plan,control,sources):
    require(plan.get('split')=='validation' and plan.get('expected_cases')==945 and plan.get('policy_sha256')==POLICY
        and plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False,'unchanged complete VAL plan required')
    rows=plan.get('jobs',[]);old={r['job_id']:r for r in old_plan.get('jobs',[])}
    expected={(task,raw,method) for task,raw in sources for method in METHODS};seen=set();ids=set();provenance=Counter()
    completed=control.get('completed_job_ids',[]);remaining=control.get('remaining_job_ids',[])
    require(len(completed)==591 and len(set(completed))==591 and len(remaining)==354 and len(set(remaining))==354
        and not set(completed)&set(remaining) and len(old)==945,'exact591 preserved/354 new partition required')
    for row in rows:
        key=(row['task'],row['raw'],row['method']);name=row['job_id'];stage=row['evidence_runtime']
        require(key in expected and key not in seen and name not in ids and name in old
            and name=='_'.join(key) and row['source_row_sha256']==sources[key[:2]] and stage in STAGES,
            'complete unique original source/method/provenance grid required')
        seen.add(key);ids.add(name);provenance[stage]+=1
        imported=name in completed
        require(type(row.get('imported_completed')) is bool and row['imported_completed'] is imported
            and row.get('previous_row_sha256')==digest(old[name])
            and row.get('previous_imported_completed') is old[name].get('imported_completed'),'exact previous-row provenance required')
        actual={k:v for k,v in row.items() if k not in ('previous_row_sha256','previous_imported_completed')}
        wanted=deepcopy(old[name])
        if imported: actual['imported_completed']=row['previous_imported_completed']
        else:
            wanted['argv'][wanted['argv'].index('--out')+1]=STAGES[CURRENT][0]+'/validation/'+ '/'.join(key)
            wanted.update(evidence_runtime=CURRENT,imported_completed=False)
        require(actual==wanted,'accepted original row changed beyond approved new output/provenance')
        require(option(row['argv'],'--out')==STAGES[stage][0]+'/validation/'+'/'.join(key),'case output does not match actual owner tier')
    require(len(rows)==945 and seen==expected and ids==set(completed)|set(remaining)
        and dict(provenance)==EXPECTED_PROVENANCE,'complete945 source/method bijection and true95/496/354 tiers required')
    return rows,dict(provenance)


def require_case(case,row,source_sha,protocol):
    identity=case.get('identity',{});task,raw,method=row['task'],row['raw'],row['method']
    require(case.get('complete') is True and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None
        and type(case.get('official_task_success')) is bool and type(case.get('native_terminal_failure')) is bool,'complete native outcome/cleanup required')
    require(case.get('split')==identity.get('split')=='validation' and case.get('task')==identity.get('task')==task
        and case.get('method')==identity.get('method')==method and case.get('source_raw_episode_id')==identity.get('source_raw_episode_id')==raw
        and identity.get('parent_raw_episode_id')==raw and identity.get('raw_episode_id')==raw+'__policy_aug_r0'
        and identity.get('replica')==0 and identity.get('source_row_sha256')==source_sha
        and identity.get('config_sha256')==row['configuration_sha256'] and identity.get('frozen_inputs_sha256')==row['input_lock_sha256']
        and identity.get('source_input_lock_sha256')==PINS[SOURCE_LOCK],'actual original source/method/CPU identity required')
    for key,value in dict(demo_only=False,development_demo_only=False,formal_validation_metric=True,formal_TEST_metric=False,
                          TEST_used_for_fitting=False,Value_NGFF_FASTER_used=False).items():
        require(case.get(key) is value,'formal VAL-only case required: '+key)
    frozen=case.get('full16_frozen_execution',{});domain=case.get('component_domain',{})
    require(frozen.get('policy_sha256')==POLICY and frozen.get('source_row_sha256')==source_sha
        and frozen.get('actual_split')=='validation' and frozen.get('VAL_TEST_relabelled_as_TRAIN') is False
        and domain.get('purpose')=='evaluation' and domain.get('split')=='validation'
        and domain.get('task')==task and domain.get('method')==method and domain.get('raw_episode_id')==raw
        and case.get('author_outcome_protocol')==protocol,'unchanged evaluation protocol/domain required')
    rml=case.get('common_author_RML_parameters',{});motion=case.get('all_stage_counted_joint_motion',{})
    require(rml.get('complete') is True and rml.get('native_parameters_mutated') is False
        and rml.get('actual_motion_bound_claimed') is False and rml.get('profile_sha256')==PROFILE
        and rml.get('actual_parameters')==PARAMETERS and type(motion.get('all_stage_actual_window_checks_passed')) is bool
        and motion.get('actual_window_acceleration_criterion_rad_s2')==6.
        and motion.get('planning_metadata_used_as_actual_acceleration_cap') is False and motion.get('policy_or_physics_modified') is False,
        'common RML and diagnostic-only measured-motion metadata required')
    clock=case.get('counted_clock',{});alignment=case.get('alignment_receipt',{});native_clock=alignment.get('clock',{});ticks=clock.get('ticks')
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt')==.05 and type(clock.get('seconds')) in (int,float)
        and math.isfinite(clock['seconds']) and abs(clock['seconds']-ticks*.05)<=1e-9
        and native_clock.get('ticks')==ticks and native_clock.get('dt')==.05 and native_clock.get('budget_seconds')==10.
        and abs(native_clock.get('simulation_seconds',float('inf'))-clock['seconds'])<=1e-9
        and sum(native_clock.get('phases',{}).values())==ticks
        and isinstance(motion.get('events'),list) and len(motion['events'])==ticks,'saved counted-clock/window coverage required')
    for key in ('actual_physics_seconds','actual_task_seconds'):
        value=alignment.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and abs(value-clock['seconds'])<=1e-5,'native counted-clock mismatch: '+key)


def require_case_pointers(rows,cases,drivers):
    require(len(rows)==len(cases)==len(drivers)==945,'exact945 ordered case/driver bijection required')
    seen_cases=set();seen_drivers=set()
    for row,case,driver in zip(rows,cases,drivers):
        out=option(row['argv'],'--out')
        require(isinstance(case,dict) and set(case)=={'path','sha256'} and valid_sha(case['sha256'])
            and isinstance(driver,dict) and set(driver)=={'path','sha256'} and valid_sha(driver['sha256'])
            and case['path']==out+'/case.json' and driver['path']==out+'/admission_driver_receipt.json'
            and case['path'] not in seen_cases and driver['path'] not in seen_drivers,
            'exact unique case/driver paths must match the released source/method plan')
        seen_cases.add(case['path']);seen_drivers.add(driver['path'])


def load_reducer(reader):
    raw=reader.raw(REDUCER,expected=PINS[REDUCER])
    module=types.ModuleType('validated_VAL_motion_v838');module.__file__=str(reader.root/REDUCER)
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    return module


def collect(reader,release_sha):
    release=reader.json(RELEASE,expected=release_sha);require_release_header(release)
    reader.evidence=release['inputs_sha256']
    control=reader.json(CONTROL,evidence=True);lock=reader.json(LOCK,evidence=True)
    require(control.get('schema')=='r10_frozen_environment_formal_VAL_resume_control_v823'
        and control.get('expected_cases')==945 and control.get('policy_sha256')==POLICY
        and control.get('physical_policy_changed') is False and control.get('completed_results_selected_by_success') is False
        and lock.get('schema')=='r10_frozen_environment_formal_VAL_resume_inputs_v823'
        and lock.get('approved_for_execution') is True,'actual frozen823 recovery graph required')
    reader.frozen=lock['files'];protocol=control['author_outcome_protocol']
    require(release.get('resume_control')==reader.pointer(CONTROL) and release.get('resume_control_lock')==reader.pointer(LOCK)
        and release.get('author_outcome_protocol')==protocol and release.get('source_sha256')==control['resume_auditor']['sha256'],
        'release must be produced by actual frozen824/control')
    reader.raw(AUDITOR,evidence=True,frozen=True)
    require(reader.inputs[AUDITOR]==release['source_sha256'],'actual824 source SHA required')
    suite=reader.json(SUITE,evidence=True,frozen=True)
    require(release.get('base_result_auditor')==suite.get('formal_result_auditor')
        and release.get('admission_driver')==suite.get('formal_case_driver')
        and release.get('admission_guard')==suite.get('formal_admission_guard')
        and suite.get('author_outcome_protocol')==protocol,'original751/754/protocol suite bindings required')
    for pointer in (release['base_result_auditor'],release['admission_driver'],release['admission_guard'],
                    protocol['implementation'],protocol['runtime_consumer'],protocol['original_runtime']):
        reader.raw(pointer['path'],expected=pointer['sha256'],frozen=True)
    reader.raw(CONTROLLER_SOURCE,evidence=True,frozen=True)
    controller=reader.json(CONTROLLER+'/launch.json')
    require(controller.get('source_sha256')==reader.inputs[CONTROLLER_SOURCE]
        and controller.get('resume_control')==reader.pointer(CONTROL) and controller.get('resume_control_lock')==reader.pointer(LOCK),
        'actual frozen827 controller launch required')
    require_environment(controller.get('frozen_environment',{}),control,observer=controller.get('pid'))
    stage_launch=reader.json(AUDIT_STAGE+'/launch.json');stage_terminal=reader.json(AUDIT_STAGE+'/terminal.json')
    stage_stdout=reader.json(AUDIT_STAGE+'/stdout.log');require_clean_terminal(stage_terminal,controller_stage=True)
    audit_argv=[str(reader.root/'.conda-env/bin/python'),'-u',AUDITOR,'--suite',SUITE,'--suite-lock',SUITE_LOCK,
        '--source-inputs',SOURCE_LOCK,'--split','validation','--runtime',CURRENT,'--resume-control',CONTROL,'--resume-lock',LOCK,'--out',RELEASE]
    require(stage_launch.get('argv')==audit_argv and stage_launch.get('source_sha256')==release['source_sha256']
        and stage_launch.get('automatic_retry') is False and stage_stdout.get('complete') is True
        and stage_stdout.get('split')=='validation' and stage_stdout.get('cases')==945
        and stage_stdout.get('imported')==591 and stage_stdout.get('new_native')==354
        and stage_stdout.get('TEST_permitted') is True and stage_stdout.get('out')==RELEASE,'actual clean827 independent824 audit stage required')
    require(stage_launch.get('inherited_frozen_environment')==controller.get('frozen_environment'),
            'independent824 stage must inherit its actual827 controller environment')
    env=release.get('actual_auditor_environment',{});require_environment(env,control,observer=stage_launch.get('pid'))
    require(env.get('observer_start_ticks')==stage_launch.get('start_ticks')
        and env['current_mount_namespace']==stage_launch.get('current_mount_namespace')
        and env['current_mount_namespace']==stage_launch.get('inherited_frozen_environment',{}).get('current_mount_namespace'),
        'actual824 PID/start/namespace must match its real827 launch')
    manifest=reader.json(SOURCE_MANIFEST);source_lock=reader.json(SOURCE_LOCK,evidence=True)
    require(release.get('source_input_lock')==reader.pointer(SOURCE_LOCK),'release source lock pointer mismatch')
    sources=source_grid(manifest,source_lock)
    plan=reader.json(CURRENT+'/plan.json',evidence=True)
    old_plan=reader.pointed(control['previous_plan'],evidence=True,frozen=True)
    rows,provenance=require_plan(plan,old_plan,control,sources)
    require(release.get('provenance_runtime_counts')=={str(reader.root/k):v for k,v in provenance.items()}
        and release.get('imported_completed_cases')==591 and release.get('newly_completed_cases')==354,'real release tier counts required')
    owners={}
    for stage,(_,source) in STAGES.items():
        # The original753 source is carried by the immutable base/supplement
        # lock, not necessarily repeated in824's project-only evidence map.
        reader.raw(source,frozen=True)
        launch=reader.json(stage+'/launch.json',evidence=True,frozen=stage!=CURRENT)
        require(launch.get('source_sha256')==reader.inputs[source] and launch.get('split')=='validation'
            and launch.get('policy_sha256')==POLICY and launch.get('author_outcome_protocol')==protocol,'real frozen stage owner required')
        owners[stage]=launch
    owner_terminal=reader.json(CURRENT+'/terminal.json',evidence=True)
    require(owner_terminal.get('complete') is True and owner_terminal.get('completed')==945
        and owner_terminal.get('expected')==945 and owner_terminal.get('failed')==0
        and owner_terminal.get('split')=='validation' and owner_terminal.get('policy_sha256')==POLICY
        and owner_terminal.get('stop_requested') is False and owner_terminal.get('automatic_retry') is False
        and release.get('actual_stage_owner')==reader.pointer(CURRENT+'/terminal.json'),'completed actual823 owner required')
    reader.pointed(control['previous_terminal'],evidence=True,frozen=True)
    require(release.get('previous_stage_owner')==control['previous_terminal'],'release must retain the real previous772 terminal')
    require_environment(owners[CURRENT].get('frozen_environment',{}),control,observer=owners[CURRENT]['pid'])
    require_environment(owner_terminal.get('frozen_environment',{}),control,observer=owners[CURRENT]['pid'])
    for key in ('current_mount_namespace','observer_start_ticks','outer_wrapper_pid','outer_wrapper_start_ticks'):
        require(owner_terminal['frozen_environment'].get(key)==owners[CURRENT]['frozen_environment'].get(key),'823 owner environment changed during execution: '+key)
    reducer=load_reducer(reader);selected=[]
    case_pointers=release['cases'];driver_pointers=release['admission_driver_receipts'];seen=set()
    require_case_pointers(rows,case_pointers,driver_pointers)
    for row,case_pointer,driver_pointer in zip(rows,case_pointers,driver_pointers):
        stage=row['evidence_runtime'];old=stage!=CURRENT;out=option(row['argv'],'--out');name=row['job_id']
        require(case_pointer['path']==out+'/case.json' and driver_pointer['path']==out+'/admission_driver_receipt.json'
            and case_pointer['path'] not in seen,'exact one-to-one release/plan case and driver evidence required')
        seen.add(case_pointer['path'])
        require(row['argv'][:3]==['bash','scripts/run_aligned_gpu.sh' if row['method']=='physmani' else 'scripts/run_local_sim.sh',release['admission_driver']['path']],
                'actual unchanged751 route required')
        for flag,value in (('--split','validation'),('--task',row['task']),('--raw-episode-id',row['raw']),('--method',row['method']),
            ('--manifest',SOURCE_MANIFEST),('--suite',SUITE),('--suite-lock',SUITE_LOCK),('--source-inputs',SOURCE_LOCK)):
            require(option(row['argv'],flag)==value,'actual VAL argv binding mismatch: '+flag)
        cpu_leaf=stage+'/CPU/'+name;native_leaf=stage+'/native/'+name
        cpu=reader.json(cpu_leaf+'/CPU_preflight.json',evidence=True,frozen=old)
        native_terminal=reader.json(native_leaf+'/terminal.json',evidence=True,frozen=old);require_clean_terminal(native_terminal)
        require(cpu.get('complete') is True and cpu.get('native_started') is False and cpu.get('GPU_started') is False
            and cpu.get('actual_split')=='validation' and cpu.get('execution_protocol_changed') is True,'actual CPU-only VAL admission required')
        for leaf,check in ((cpu_leaf,True),(native_leaf,False)):
            launch=reader.json(leaf+'/launch.json',evidence=True,frozen=old)
            terminal=reader.json(leaf+'/terminal.json',evidence=True,frozen=old);require_clean_terminal(terminal)
            require(launch.get('argv')==row['argv']+(['--check'] if check else []) and launch.get('check_only') is check
                and launch.get('owner_pid')==owners[stage]['pid'] and launch.get('native_slots')==(0 if check else 1)
                and launch.get('GPU_slots')==(0 if check else int(row['method']=='physmani'))
                and launch.get('expected_identity_sha256')==(None if check else cpu['identity']['sha256']),
                'actual child argv/owner/resources/identity mismatch')
            if not old:
                require(launch.get('inherited_frozen_environment')==owners[stage]['frozen_environment']
                    and launch.get('current_mount_namespace')==owners[stage]['frozen_environment']['current_mount_namespace'],
                    'new823 child must inherit its real owner environment')
            for key in BINDINGS: require(launch.get(key)==row.get(key) and cpu.get(key)==row.get(key),'actual executable protocol binding changed: '+key)
        case=reader.pointed(case_pointer,evidence=True,frozen=old);driver=reader.pointed(driver_pointer,evidence=True,frozen=old)
        require(native_terminal.get('case')==case_pointer and native_terminal.get('admission_driver_receipt')==driver_pointer
            and cpu.get('identity')==case.get('identity') and cpu.get('component_domain')==case.get('component_domain'),
            'actual native terminal/CPU/case bindings required')
        require(driver.get('schema')=='r10_explicit_author_outcome_formal_driver_v751' and driver.get('complete') is True
            and driver.get('actual_split')=='validation' and driver.get('execution_protocol_changed') is True and driver.get('case')==case_pointer,
            'actual751 external driver receipt required')
        for key in BINDINGS: require(driver.get(key)==row.get(key),'external driver binding changed: '+key)
        require_case(case,row,sources[(row['task'],row['raw'])],protocol)
        metrics,diagnostics=reducer.reduce_saved_case(case)
        metrics={key:(value if reducer.finite(value) else None) for key,value in metrics.items()}
        selected.append(dict(task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],source_row_sha256=row['source_row_sha256'],
            case=case_pointer,admission_driver_receipt=driver_pointer,evidence_runtime=stage,actual_owner_pid=owners[stage]['pid'],
            CPU_receipt=reader.pointer(cpu_leaf+'/CPU_preflight.json'),native_launch=reader.pointer(native_leaf+'/launch.json'),
            native_terminal=reader.pointer(native_leaf+'/terminal.json'),official_task_success=case['official_task_success'],
            native_terminal_failure=case['native_terminal_failure'],metrics=metrics,recorded_window_diagnostics=diagnostics))
    require(len(selected)==945 and len(seen)==945,'exact945 valid VAL cases required')
    for method in METHODS:
        require(sum(r['official_task_success'] for r in selected if r['method']==method)==release['native_successes_by_method'][method],
                'all native outcomes must remain in the released cohort')
    subset=reader.json(SUBSET)
    by_path={r['case']['path']:r for r in selected}
    for previous in subset['case_evidence']:
        current=by_path[previous['case']['path']]
        require(current['case']==previous['case'] and current['metrics']==previous['metrics']
            and current['recorded_window_diagnostics']==previous['recorded_window_diagnostics'],'prior838 case/metric regression changed')
    return selected,release,subset,reducer,provenance


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args(argv);require(valid_sha(args.release_sha256),'explicit completed824 release SHA required')
    require(not (ROOT/OUTPUT).exists(),'preserve existing847 output; never overwrite')
    reader=Reader(ROOT);rows,release,subset,reducer,provenance=collect(reader,args.release_sha256)
    reader.inputs[str(Path(__file__).relative_to(ROOT))]=sha_bytes(Path(__file__).read_bytes())
    result=dict(schema='r10_full945_VAL_saved_motion_summary_v847',complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        read_only_analysis=True,split='validation',task_count=16,source_count=315,case_count=945,methods=list(METHODS),
        task_source_counts=COUNTS,policy_sha256=POLICY,validation_release=reader.pointer(RELEASE),
        actual_validation_auditor=dict(path=AUDITOR,sha256=release['source_sha256']),provenance_runtime_counts=provenance,
        metric_definitions=subset['metric_definitions'],statistical_definition=dict(subset['statistical_definition'],
            overall='pool315 equally weighted cases per method; never average task statistics'),
        diagnostic_definition=subset['diagnostic_definition'],hard_budget_disclosure=subset['hard_budget_disclosure'],
        common_RML_planner_configuration=dict(parameters=PARAMETERS,profile_sha256=PROFILE,cases=945,
            parameter_kind='saved path-parameter planning configuration, not a per-joint measured bound'),
        overall_by_method={m:reducer.summarize([r for r in rows if r['method']==m]) for m in METHODS},
        by_task_method=[dict(task=t,method=m,**reducer.summarize([r for r in rows if (r['task'],r['method'])==(t,m)])) for t in TASKS for m in METHODS],
        case_evidence=rows,prior838_per_case_metrics_and_flags_reproduced=591,inputs_sha256=reader.inputs,
        source_sha256=reader.inputs[str(Path(__file__).relative_to(ROOT))],reducer_source=dict(path=REDUCER,sha256=PINS[REDUCER]),
        full945_VAL_release_observed=True,new_release_granted_by_this_summary=False,native_false_retained=True,
        success_rate_or_acceleration_filter_used=False,new_joint_differencing_used=False,alignment_motion_mirror_pooled=False,
        transitive_TRAIN_model_OS_inputs_walked=False,raw_HDF5_contents_opened=False,TEST_data_read=False,
        native_GPU_or_jobs_started=False,frozen_source_or_lock_modified=False,TXT_or_handoff_written=False)
    with (ROOT/OUTPUT).open('x') as f: f.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,output=OUTPUT,sha256=sha_bytes((ROOT/OUTPUT).read_bytes()),case_count=945)))


if __name__=='__main__': main()
