"""Inert metadata-only resource lease after complete mixed-source formal TEST.

Import and context construction do no I/O. This module never starts work,
installs hooks, reads case/raw/model contents or walks any declared-input map.
A future explicit enabled scope needs external final906 and915-lock SHA pins,
complete907/902/906 evidence, shared resource locks and a fresh host inventory.
The independent906 audit remains the scientific release authority. This guard
only validates its recorded completion/provenance and reserves resources.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from weakref import WeakKeyDictionary

VERSION='r10_inert_post_complete_composite_TEST_resource_guard_v924'
CONTROLLER='runtime/framework/ordered_concurrent_TEST_v907'
TEST_RUNTIME='runtime/framework/formal_TEST_concurrent_resume_v902'
OLD_CONTROLLER='runtime/framework/ordered_env_resume_VAL_TEST_v827'
OLD_RUNTIME='runtime/framework/formal_TEST_after_environment_resume_v753'
ART='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902'
OLD_ART='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823'
RESULT=ART+'/TEST_results.json'
VAL_RELEASE=OLD_ART+'/VAL_release.json'
SETUP='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
CONTROL=SETUP+'/resume_control.json'
CONTROL_LOCK=SETUP+'/resume_inputs.json'
BATCH_CONTROL=SETUP+'/formal_batch_control.json'
BATCH_LOCK=SETUP+'/formal_batch_inputs.json'
ORIGINAL_SETUP='artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup'
SUITE=ORIGINAL_SETUP+'/suite.json'
SUITE_LOCK=ORIGINAL_SETUP+'/frozen_inputs.json'
SOURCE_LOCK=ORIGINAL_SETUP+'/test_source_inputs.json'
MANIFEST=ORIGINAL_SETUP+'/official_test_manifest16.json'
OWNER_STAGE=CONTROLLER+'/stages/formal_TEST_1920'
AUDIT_STAGE=CONTROLLER+'/stages/independent_TEST_results'
VAL_OWNER_STAGE=OLD_CONTROLLER+'/stages/environment_resume_VAL_945'
CONTROLLER_SOURCE='scripts/continue_concurrent_TEST_v907_r10.py'
OWNER_SOURCE='scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'
AUDITOR_SOURCE='scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py'
OLD_CONTROLLER_SOURCE='scripts/continue_environment_resumed_VAL_TEST_v827_r10.py'
OLD_OWNER_SOURCE='scripts/run_formal_suite16_owner_v753_r10.py'
BASE_AUDITOR_SOURCE='scripts/audit_formal_suite16_results_v754_r10.py'
VAL_OWNER_SOURCE='scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py'
QUALIFIER_SOURCE='scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py'
QUALIFICATION='runtime/framework/concurrent_TRAIN_resource_qualification_v904.json'
QUALIFICATION_LAUNCH='runtime/framework/concurrent_TRAIN_resource_audit_v904/launch.json'
BATCH_SOURCE='scripts/run_batched_formal_driver_v901_r10.py'
LOOKUP_SOURCE='our_mani/multitask/frozen_lookup_batch_v900.py'
PREPARER_SOURCE='scripts/prepare_concurrent_TEST_resume_v915_r10.py'
SOURCE_SHA={
    OLD_CONTROLLER_SOURCE:'6f0c13d410aba891df7ae51acfeacbe9714cfb9dabf87c5850763a3326f86e30',
    OLD_OWNER_SOURCE:'ece7f0d57f4e533b22a62d35a4b5220deedb88dbfc44767b060fc889b30cb183',
    BASE_AUDITOR_SOURCE:'3b35c47796d0294ace8dbcdef7f5ca49fa4a227d8c3f2488e0864af08d599301',
    VAL_OWNER_SOURCE:'d0cb699117fbd74f949d27b4a863249d4ab27a384f7a53756dc4da709a919ab4',
    CONTROLLER_SOURCE:'31d0b31e869756779f7b8b61c7eaeea0d55289174769248a59775fadea9015af',
    OWNER_SOURCE:'ef55aaf86e9334dcae0f02113e76fcbdcc360709a769aaa34f2b1f33a77c6c43',
    AUDITOR_SOURCE:'d73cab726f8885b35d590051f29bfde710e6c6fa6aa6c5743704159a5bc93d44',
    BATCH_SOURCE:'8a33cdfadcc98d89adea49cad4f73d918f428b6eb6d53974522a8e780a4e8e5b',
    LOOKUP_SOURCE:'35776d7e6fd2389e8ad363b0de8a7dd86e59b59e9b34eafc9399e2c8664cbf89',
    PREPARER_SOURCE:'01800222dd019a6d892b340e1fc4110237e36f30479a5b3e2a9b23a41273f801',
    QUALIFIER_SOURCE:'44dfe11785aa02885a79c47853a908e4d9b7b5e3d8f5521fcb67d9efabacaa63',
}
POLICY = 'eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
PROFILE = 'd9a36e5ba8a73123c5deb65229d7fa151089145dca4b55c68c1e9ad1dec4e562'
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
TASKS = tuple(sorted((
    'beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed',
    'insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed',
    'moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed',
    'pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed',
    'place_cups_on_rotating_frame', 'place_cups_on_rotating_frame_high_speed',
    'push_moving_button', 'push_moving_button_high_speed',
    'put_rubbish_in_moving_bin', 'put_rubbish_in_moving_bin_high_speed',
    'remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed')))
ENVIRONMENT_POINTERS = dict(
    manifest=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json', sha256='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'),
    wrapper=dict(path='scripts/run_frozen_system_environment_v820_r10.py', sha256='3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3'),
    namespace_qualification=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json', sha256='1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa'))
PROTOCOL = dict(schema='r10_author_action_outcome_protocol_v741',
    implementation=dict(path='our_mani/multitask/native_outcome_sampling_v741.py', sha256='2289a93fcb9d8c27a700d66164da7e2ea8fbcc0fa4f0c775e986ab9f40a9cb12'),
    runtime_consumer=dict(path='scripts/run_task_executor_smoke_author_suite16_v744_r10.py', sha256='1ffb2922521da8b77b4e005ffa0a9b22d9d4bc02a4944253fbc76605ce6f930d'),
    original_runtime=dict(path='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py', sha256='e1ba1b1409762329d7624fc4e8e27cf2475860142d63de39e05f4fb08f58941f'),
    failure_sampling='current_native_result_at_explicit_action_end', success_sampling='original_per_tick_success_latch',
    pose_action_max_ticks=10, public_hold_max_ticks=1, budget_partial_action_consumed=True,
    native_RML_parameters_unchanged=True, block_native_fail_conditions=0, block_extra_task_success_calls=0)
DRIVER = dict(path='scripts/run_formal_case_author_outcome_v751_r10.py', sha256='06053736b8cb868c6783a4bb8a81c3f36698ef8137bf62f1e5f53e007ebe9d36')
GUARD = dict(path='our_mani/multitask/formal_suite_admission_v728.py', sha256='f325b48a9778e02aa9d6a7bb32593aa11c454903f87cddb59f1c0248b5395bea')
WORKER = dict(path='scripts/run_task_executor_smoke_author_outcome_v743_r10.py', sha256='714fb3ba7e5f282cdd9f68618ed2615edbbed0d5ebfb1921ee7d202424ca6174')

LOCKS=(CONTROLLER+'.lock','runtime/framework/ordered_resume_VAL_TEST_v775.lock',
    OLD_CONTROLLER+'.lock','runtime/framework/fixed_TRAIN104_geometry_native.lock')
STOP_PATHS=('runtime/framework/multitask_r10_STOP',CONTROLLER+'/STOP',TEST_RUNTIME+'/STOP',
    OLD_CONTROLLER+'/STOP',OLD_RUNTIME+'/STOP','runtime/framework/formal_VAL_env_resume_v823/STOP',
    'runtime/framework/ordered_resume_VAL_TEST_v775/STOP','runtime/framework/formal_VAL_resume_v772/STOP',
    'runtime/framework/concurrent_TRAIN_resource_canary_v903/STOP','runtime/framework/post_TEST_resource_guard_v924/STOP')
NATIVE_PREFIXES=('run_geometry_','run_native_','run_continuous_reference_pair_','collect_multitask_',
    'run_ready_candidate_','launch_multitask_','run_multitask_fresh_queue','run_task_formal_','run_trained_task_',
    'evaluate_multitask_case_','run_PhysMani_task_','run_task_executor_smoke_','run_formal_case_',
    'run_formal_suite16_owner_','run_resumed_formal_VAL_owner_','run_environment_resumed_formal_VAL_owner_',
    'continue_resumed_VAL_TEST_','continue_environment_resumed_VAL_TEST_','continue_ordered_','continue_author_outcome_',
    'run_frozen_environment_TRAIN_canary_','diagnose_fixed_TRAIN_','diagnose_TRAIN_','run_TRAIN_',
    'run_concurrent_resumed_formal_TEST_owner_','run_batched_formal_driver_','run_concurrent_TRAIN_',
    'continue_concurrent_TEST_','audit_concurrent_TRAIN_resource_canary_','audit_concurrent_resumed_formal_TEST_')
PROFILE_LIMITS=dict(schema='r10_frozen_concurrent_native_resource_profile_v902',max_native_slots=4,
    max_GPU_slots=2,max_CPU_preflight_slots=2,ready_limit=8)
ROW_PROVENANCE={'evidence_runtime','imported_completed','previous_row_sha256','execution_entry','physical_argv','previous_worker_sha256'}
_LEASES=WeakKeyDictionary()


def require(value,message):
    if not value:raise ValueError('post TEST924: '+message)


def valid_sha(value):return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
def same(a,b):return a==b

def option(argv,name):
    require(isinstance(argv,list) and argv.count(name)==1,'one explicit '+name+' required')
    i=argv.index(name);require(i+1<len(argv) and isinstance(argv[i+1],str),'missing '+name+' value');return argv[i+1]

def _pointer(value,name=None):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value.get('path'),str)
        and bool(value['path']) and not Path(value['path']).is_absolute() and '..' not in PurePosixPath(value['path']).parts
        and str(PurePosixPath(value['path']))==value['path'] and valid_sha(value.get('sha256'))
        and (name is None or value['path']==name),'exact project metadata pointer required')
    return value

def _pid(value):
    require(type(value.get('pid')) is int and value['pid']>0 and type(value.get('start_ticks')) is int
        and value['start_ticks']>0,'real PID/start identity required')
    return dict(pid=value['pid'],start_ticks=value['start_ticks'])

def _lookup(mapping,root,name):
    require(isinstance(mapping,dict),'recorded SHA map required')
    values=[mapping[k] for k in (name,str(Path(root)/name)) if k in mapping]
    require(len(values)==1 and valid_sha(values[0]),'one recorded SHA for selected metadata required: '+name)
    return values[0]

def _allowed(name):
    if name in {RESULT,CONTROL,CONTROL_LOCK,QUALIFICATION,QUALIFICATION_LAUNCH,*SOURCE_SHA}:return True
    if name in {OLD_RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json','DRAIN')}:return True
    if name in {OLD_CONTROLLER+'/'+n for n in ('launch.json','status.json','failure.json')}:return True
    if name in {r+'/'+n for r in (CONTROLLER,TEST_RUNTIME) for n in ('launch.json','plan.json','status.json','terminal.json')}:return True
    return name==VAL_OWNER_STAGE+'/launch.json' or any(name==stage+'/'+n for stage in (OWNER_STAGE,AUDIT_STAGE)
        for n in ('launch.json','terminal.json','stdout.log') if stage==AUDIT_STAGE or n!='stdout.log')


class _Metadata:
    def __init__(self,root,callback):self.root,self.callback,self.inputs,self.signatures=Path(root).resolve(),callback,{},{}
    def raw(self,name,expected=None):
        require(isinstance(name,str) and _allowed(name) and not Path(name).is_absolute()
            and '..' not in PurePosixPath(name).parts and str(PurePosixPath(name))==name,'non-metadata read forbidden: '+str(name))
        if self.callback is not None:raw=self.callback(name)
        else:
            path=self.root/name
            require(path.resolve(strict=True)==path,'metadata aliases refused')
            signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            before=signature(path.stat());raw=path.read_bytes()
            require(before==signature(path.stat()),'metadata changed while reading')
            require(name not in self.signatures or self.signatures[name]==before,'metadata identity changed')
            self.signatures[name]=before
        require(type(raw) is bytes,'exact metadata bytes required');checksum=sha(raw)
        require((expected is None or valid_sha(expected) and checksum==expected)
            and (name not in self.inputs or self.inputs[name]==checksum),'metadata SHA changed: '+name)
        self.inputs[name]=checksum;return raw
    def json(self,name,expected=None):
        value=json.loads(self.raw(name,expected));require(isinstance(value,dict),'JSON metadata object required');return value
    def pointer(self,name):return dict(path=name,sha256=self.inputs[name])
    def pointed(self,value,name=None):
        _pointer(value,name);return self.json(value['path'],value['sha256'])
    def close(self):
        for name,expected in tuple(self.inputs.items()):self.raw(name,expected)


def _environment(env,record):
    _pid(record)
    require(isinstance(env,dict) and env.get('schema')=='r10_actual_frozen_OS_namespace_receipt_v820'
        and env.get('complete') is True and env.get('all_current_OS_targets_full_byte_SHA_verified') is True
        and type(env.get('checked_OS_files')) is int and env['checked_OS_files']==6418
        and env.get('host_PID_network_namespace_preserved') is True and env.get('proc_or_dev_replaced') is False
        and env.get('task_parameters_or_strategy_changed') is False and env.get('observer_pid')==record['pid']
        and env.get('pid')==record['pid'] and env.get('observer_start_ticks')==record['start_ticks']
        and isinstance(env.get('current_mount_namespace'),str) and env['current_mount_namespace']==env.get('mount_namespace')
        and env['current_mount_namespace']!=env.get('reference_mount_namespace')
        and all(env.get(k)==p for k,p in ENVIRONMENT_POINTERS.items()),'recorded qualified820 environment required')
    return env

def _clean_stage(value):
    require(value.get('complete') is True and type(value.get('returncode')) is int and value['returncode']==0
        and value.get('automatic_retry') is False,'clean907 stage required')

def _common(value):
    require(value.get('split')=='test' and value.get('policy_sha256')==POLICY
        and value.get('author_outcome_protocol')==PROTOCOL and value.get('automatic_retry') is False
        and value.get('execution_protocol_changed') is True,'fixed scientific TEST owner binding required')

def _row_grid(plan):
    require(plan.get('split')=='test' and plan.get('expected_cases')==1920 and plan.get('policy_sha256')==POLICY
        and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False,'complete unchanged TEST plan required')
    rows=plan.get('jobs');require(isinstance(rows,list) and len(rows)==1920,'full1920 rows required')
    seen=set();ids=set();sources={};groups={}
    for row in rows:
        task,raw,method=row.get('task'),row.get('raw'),row.get('method');key=(task,raw,method)
        require(task in TASKS and isinstance(raw,str) and raw and '/' not in raw and '__policy_aug' not in raw
            and method in METHODS and key not in seen and row.get('job_id')=='_'.join(key)
            and row['job_id'] not in ids and valid_sha(row.get('source_row_sha256')),'original source/method bijection required')
        require(all(row.get(k)==v for k,v in dict(admission_driver=DRIVER,admission_guard=GUARD,case_worker=WORKER,
            native_runtime=PROTOCOL['runtime_consumer'],original_native_runtime=PROTOCOL['original_runtime'],author_outcome_protocol=PROTOCOL).items()),'original physical row binding changed')
        source=(task,raw);require(source not in sources or sources[source]==row['source_row_sha256'],'source SHA differs across methods')
        sources[source]=row['source_row_sha256'];seen.add(key);ids.add(row['job_id']);groups.setdefault((task,method),set()).add(raw)
    require(len(sources)==640 and len(groups)==48 and all(len(groups[t,m])==40 for t in TASKS for m in METHODS)
        and all(groups[t,m]==groups[t,METHODS[0]] for t in TASKS for m in METHODS),'640 originals times3 with40 same sources/task required')
    return rows


def _entry(control,files,root):
    value=control.get('execution_entry');require(isinstance(value,dict) and set(value)=={'kind','pointer','batch_control','batch_lock'}
        and value['kind']=='batched901','actual qualified901 continuation entry required')
    for key,name in (('pointer',BATCH_SOURCE),('batch_control',BATCH_CONTROL),('batch_lock',BATCH_LOCK)):
        _pointer(value[key],name);require(_lookup(files,root,name)==value[key]['sha256'],'frozen batch entry binding changed')
    names=(BATCH_SOURCE,LOOKUP_SOURCE,DRIVER['path'],GUARD['path'],'our_mani/multitask/rml_case_preflight_v742.py',WORKER['path'],PROTOCOL['runtime_consumer']['path'])
    sources={name:dict(path=name,sha256=_lookup(files,root,name)) for name in names}
    require(sources[BATCH_SOURCE]==value['pointer'],'selected901 source mismatch')
    implementation=dict(schema='r10_resource_execution_implementation_v902',kind='batched901',entrypoint=value['pointer'],
        lookup_batch=sources[LOOKUP_SOURCE],original_formal_driver=DRIVER,original_case_worker=WORKER,
        original_native_runtime=PROTOCOL['runtime_consumer'],sources=sources)
    require(control.get('execution_implementation')==implementation,'same qualified seven-source implementation required')
    return value,implementation


def _resumed(old,done,entry):
    result=[]
    for previous in old:
        require(not ROW_PROVENANCE.intersection(previous),'only exact original753 tier may be adopted')
        row=deepcopy(previous);adopt=row['job_id'] in done
        row.update(evidence_runtime=OLD_RUNTIME if adopt else TEST_RUNTIME,imported_completed=adopt,
            previous_row_sha256=digest(previous),previous_worker_sha256=previous['worker_sha256'])
        if adopt:row.update(physical_argv=deepcopy(previous['argv']),execution_entry=dict(kind='original751',pointer=DRIVER))
        else:
            inner=deepcopy(previous['argv']);inner[inner.index('--out')+1]=ART+'/test/'+row['task']+'/'+row['raw']+'/'+row['method']
            row.update(physical_argv=inner,argv=[*inner[:2],BATCH_SOURCE,'--batch-control',entry['batch_control']['path'],
                '--batch-lock',entry['batch_lock']['path'],'--batch-lock-sha256',entry['batch_lock']['sha256'],'--entry','formal','--',*inner[3:]],
                worker_sha256=entry['pointer']['sha256'],execution_entry=deepcopy(entry))
        result.append(row)
    return result

def _stop_guard(root):
    if any((Path(root) / name).exists() for name in STOP_PATHS):
        raise InterruptedError('global/formal/diagnostic STOP prevents a new TRAIN diagnostic')


@contextmanager
def _reserve(root):
    import fcntl
    with ExitStack() as stack:
        for name in LOCKS:
            path = Path(root) / name
            require(path.parent.resolve()==path.parent and path.parent.is_relative_to(Path(root).resolve())
                and not path.is_symlink(), 'canonical project lock path required')
            stream = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('formal/global native resource lock busy: ' + name) from error
        yield


def _is_alive(record):
    _pid(record)
    try:
        fields = (Path('/proc') / str(record['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] not in ('Z', 'X') and int(fields[19]) == record['start_ticks']


def _inventory(root):
    """Same-UID host inventory, excluding only this process and actual ancestors."""
    root, proc, uid = Path(root).resolve(), Path('/proc'), os.getuid()
    def stat(path):
        f = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return f[0], int(f[1]), int(f[19])
    own, parent = set(), os.getpid()
    while parent > 0 and parent not in own:
        own.add(parent); parent = stat(proc / str(parent))[1]
    comm = (proc / '1/comm').read_text().strip()
    active, unresolved = [], []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own: continue
        try:
            before = stat(entry)
            if before[0] in ('Z', 'X'): continue
            process_uid = entry.stat().st_uid
            try:
                tokens = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            except PermissionError:
                if process_uid == uid: unresolved.append(dict(pid=int(entry.name), reason='same-UID cmdline unreadable'))
                continue
            matches = [Path(x).name for x in tokens if '\n' not in x and '\r' not in x
                       and Path(x).name.endswith('.py') and Path(x).name.startswith(NATIVE_PREFIXES)]
            if not matches or process_uid != uid: continue
            cwd = Path(os.readlink(entry / 'cwd')).resolve()
            after = stat(entry)
            if before[2] == after[2] and after[0] not in ('Z', 'X') and cwd == root:
                active.append(dict(pid=int(entry.name), start_ticks=after[2], scripts=matches))
        except (FileNotFoundError, ProcessLookupError): continue
        except (OSError, ValueError, IndexError) as error:
            unresolved.append(dict(pid=int(entry.name), reason=repr(error)))
    return dict(host_visibility_available=comm in ('systemd', 'init'), same_uid_inventory_incomplete=bool(unresolved),
                live_project_native_processes=active, unresolved_processes=unresolved,
                excluded_self_and_ancestor_pids=sorted(own), ancestor_chain_complete=parent==0, host_namespace_links_claimed_readable=False)




def require_complete_TEST_metadata(root,expected_test_result_sha256,*,resume_lock_sha256,read_bytes=None):
    """Read only fixed completion metadata; never follow case/source-data pointers.

    Both SHA pins come from the future caller's frozen execution materials.
    Injected bytes are CPU-test evidence and cannot create a production lease.
    """
    require(valid_sha(expected_test_result_sha256) and valid_sha(resume_lock_sha256),'external final906 and915-lock SHA pins required')
    r=_Metadata(root,read_bytes)
    for source,expected in SOURCE_SHA.items():r.raw(source,expected)
    lock=r.json(CONTROL_LOCK,resume_lock_sha256)
    require(lock.get('schema')=='r10_concurrent_resumed_formal_TEST_inputs_v902' and lock.get('approved_for_execution') is True,
        'actual frozen915 continuation lock required')
    files=lock.get('files');control=r.json(CONTROL,_lookup(files,r.root,CONTROL));cp,lp=r.pointer(CONTROL),r.pointer(CONTROL_LOCK)
    require(control.get('schema')=='r10_concurrent_resumed_formal_TEST_control_v902' and control.get('split')=='test'
        and control.get('expected_cases')==1920 and control.get('source_count')==640 and control.get('policy_sha256')==POLICY
        and control.get('author_outcome_protocol')==PROTOCOL and control.get('resource_profile')==PROFILE_LIMITS
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False,
        'same-policy full1920 continuation required')
    for key,name in (('resume_controller',CONTROLLER_SOURCE),('resume_owner',OWNER_SOURCE),('resume_auditor',AUDITOR_SOURCE),
        ('resource_qualifier',QUALIFIER_SOURCE),('resource_qualification',QUALIFICATION),('suite',SUITE),('suite_lock',SUITE_LOCK),
        ('source_inputs',SOURCE_LOCK),('validation_release',VAL_RELEASE)):
        p=_pointer(control.get(key),name);require(_lookup(files,r.root,name)==p['sha256'],'frozen control pointer differs: '+key)
        if name in SOURCE_SHA:require(p['sha256']==SOURCE_SHA[name],'actual frozen source differs: '+key)
    for key,field in (('environment_wrapper','wrapper'),('environment_manifest','manifest'),('environment_namespace_proof','namespace_qualification')):
        require(control.get(key)==ENVIRONMENT_POINTERS[field],'qualified820 control binding changed')
    entry,implementation=_entry(control,files,r.root)
    controller=r.json(CONTROLLER+'/launch.json');state=r.json(CONTROLLER+'/status.json');final=r.json(CONTROLLER+'/terminal.json')
    known=[_pid(controller)]
    require(controller.get('source_sha256')==SOURCE_SHA[CONTROLLER_SOURCE] and controller.get('resume_control')==cp
        and controller.get('resume_control_lock')==lp and controller.get('automatic_retry') is False,'actual907 source/control required')
    env=_environment(controller.get('frozen_environment'),controller)
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False and state.get('frozen_environment')==env
        and not state.get('infrastructure_failure_detected') and state.get('owner_error') is None,'actual907 complete status required')
    require(final.get('complete') is True and final.get('automatic_retry') is False
        and _pointer(final.get('TEST_results'),RESULT)['sha256']==expected_test_result_sha256,'actual907 complete final result pointer required')
    owner_args=[]
    for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
        owner_args.extend((flag,control[key]['path']))
    owner_args+=['--runtime',TEST_RUNTIME,'--out',ART+'/test','--resume-control',CONTROL,'--resume-lock',CONTROL_LOCK,'--resume-lock-sha256',resume_lock_sha256]
    audit_args=['--resume-control',CONTROL,'--resume-lock',CONTROL_LOCK,'--resume-lock-sha256',resume_lock_sha256]
    stages={}
    for stage,source,args in ((OWNER_STAGE,OWNER_SOURCE,owner_args),(AUDIT_STAGE,AUDITOR_SOURCE,audit_args)):
        record,end=r.json(stage+'/launch.json'),r.json(stage+'/terminal.json');known.append(_pid(record));_clean_stage(end)
        require(record.get('argv')==[str(r.root/'.conda-env/bin/python'),'-B','-u',source,*args]
            and record.get('source_sha256')==SOURCE_SHA[source] and record.get('automatic_retry') is False
            and record.get('inherited_frozen_environment')==env and record.get('current_mount_namespace')==env['current_mount_namespace'],
            'real907 stage argv/source/environment required')
        stages[source]=record
    launch,plan,status,terminal=(r.json(TEST_RUNTIME+'/'+name) for name in ('launch.json','plan.json','status.json','terminal.json'))
    for value in (launch,plan,status,terminal):
        _common(value)
        require(all(value.get(k)==v for k,v in dict(resume_control=cp,resume_control_lock=lp,resource_profile=PROFILE_LIMITS,
            resource_qualification=control['resource_qualification'],execution_implementation=implementation).items()),'new902 continuation binding changed')
    _environment(launch.get('frozen_environment'),launch)
    require(_pid(launch)==_pid(stages[OWNER_SOURCE]) and launch.get('source_sha256')==SOURCE_SHA[OWNER_SOURCE]
        and launch.get('argv')==stages[OWNER_SOURCE]['argv'][3:] and launch.get('execution_entry')==entry
        and launch.get('base_stage_owner')==dict(path=OLD_OWNER_SOURCE,sha256=SOURCE_SHA[OLD_OWNER_SOURCE])
        and all(launch.get(k)==PROFILE_LIMITS[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots'))
        and launch['frozen_environment']['current_mount_namespace']==env['current_mount_namespace']
        and all(v.get('frozen_environment')==launch['frozen_environment'] for v in (plan,status,terminal)), 'actual902 owner identity/environment required')
    require(terminal.get('complete') is True and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('owner_error') is None and terminal.get('stop_requested') is False
        and terminal.get('drain_requested') is False and terminal.get('drained') is False and terminal.get('CPU_ready_unexecuted')==[]
        and type(terminal.get('CPU_not_launched')) is int and terminal['CPU_not_launched']==0
        and terminal.get('formal_release_requires_independent_audit') is True,'complete1920 clean902 terminal required')
    records=status.get('completed')
    require(isinstance(records,dict) and len(records)==1920 and status.get('expected')==1920 and status.get('failed')=={}
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('CPU_ready')==status.get('native_slots')==status.get('GPU_slots')==0
        and status.get('owner_error') is None and status.get('stop_requested') is False and status.get('drain_requested') is False,
        'complete1920 idle902 status required')
    # All real controller/stage/owner completion gates precede the first result read.
    result=r.json(RESULT,expected_test_result_sha256)
    require(result.get('schema')=='r10_composite_concurrent_resumed_formal_TEST_results_v906' and result.get('complete') is True
        and result.get('split')=='test' and result.get('source_cases')==640 and result.get('method_cases')==1920
        and result.get('completed_method_cases')==1920 and result.get('methods')==list(METHODS) and result.get('tasks')==list(TASKS)
        and result.get('policy_sha256')==POLICY and result.get('common_RML_profile_sha256')==PROFILE
        and result.get('source_sha256')==SOURCE_SHA[AUDITOR_SOURCE], 'actual independent906 full640/1920 audit required')
    expected=dict(resume_control=cp,resume_control_lock=lp,source_input_lock=control['source_inputs'],validation_release=control['validation_release'],
        resource_profile=PROFILE_LIMITS,resource_qualification=control['resource_qualification'],execution_implementation=implementation,
        admission_driver=DRIVER,admission_guard=GUARD,native_runtime=PROTOCOL['runtime_consumer'],original_native_runtime=PROTOCOL['original_runtime'],
        author_outcome_protocol=PROTOCOL,base_result_auditor=dict(path=BASE_AUDITOR_SOURCE,sha256=SOURCE_SHA[BASE_AUDITOR_SOURCE]))
    require(all(result.get(k)==v for k,v in expected.items()),'actual906 frozen provenance changed')
    flags=dict(TEST_used_for_fitting=False,motion_is_diagnostic_only=True,motion_metrics_required_for_all_methods=True,
        native_task_success_unchanged=True,native_success_is_not_a_release_threshold=True,source_outcome_filtering=False,native_failures_retained=True,
        Value_NGFF_FASTER_used=False,execution_protocol_changed=True,used_evidence_rechecked_at_close=True,old753_results_not_rerun=True,
        independent_composite_audit=True,original754_single_owner_result_claimed=False,automatic_retry=False)
    require(all(result.get(k) is v for k,v in flags.items()),'native outcome/diagnostic/composite semantics changed')
    invocation=result.get('auditor_invocation',{})
    require(_pid(invocation)==_pid(stages[AUDITOR_SOURCE]) and invocation.get('argv')==stages[AUDITOR_SOURCE]['argv'][3:]
        and invocation.get('source')==control['resume_auditor'] and invocation.get('inherited_physical_stage_identity_claimed') is False,
        'actual906 independent audit invocation required')
    audit_env=_environment(result.get('frozen_environment'),invocation)
    require(audit_env['current_mount_namespace']==env['current_mount_namespace'],'actual906 must share recorded907 namespace')
    stdout=r.json(AUDIT_STAGE+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema')==result['schema'] and stdout.get('split')=='test'
        and stdout.get('cases')==1920 and stdout.get('out')==RESULT and stdout.get('motion_is_diagnostic_only') is True,
        'actual906 complete stdout required; TEST_permitted is absent from this writer')
    old={}
    for key,name in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json'),('previous_DRAIN','DRAIN')):
        p=_pointer(control.get(key),OLD_RUNTIME+'/'+name);require(_lookup(files,r.root,p['path'])==p['sha256'],'frozen historical evidence required')
        old[name]=r.pointed(p)
    old_launch,old_plan,old_status,old_end=(old[n] for n in ('launch.json','plan.json','status.json','terminal.json'))
    known.append(_pid(old_launch));_common(old_launch)
    require(old_launch.get('source_sha256')==SOURCE_SHA[OLD_OWNER_SOURCE] and old_launch.get('max_native_slots')==2
        and old_launch.get('max_GPU_slots')==1 and old_launch.get('max_CPU_preflight_slots')==2,'real old753 identity/resource limits required')
    old_rows=_row_grid(old_plan);ids={row['job_id'] for row in old_rows}
    done,remaining=control.get('completed_job_ids'),control.get('remaining_job_ids')
    require(all(isinstance(v,list) and all(isinstance(k,str) for k in v) and len(v)==len(set(v)) for v in (done,remaining))
        and done and remaining and not set(done)&set(remaining) and set(done)|set(remaining)==ids
        and control.get('completed_key_sha256')==digest(sorted(done)) and control.get('remaining_key_sha256')==digest(sorted(remaining)),
        'complete dynamic imported/remaining partition required')
    count=len(done);old_records=old_status.get('completed')
    require(isinstance(old_records,dict) and set(old_records)==set(done) and old_status.get('split')=='test' and old_status.get('expected')==1920
        and old_status.get('failed')=={} and old_status.get('active')==old_status.get('checking')==[]
        and old_status.get('drain_requested') is True and old_status.get('stop_requested') is False and old_status.get('automatic_retry') is False
        and old_status.get('native_slots')==old_status.get('GPU_slots')==0
        and old_end.get('complete') is False and old_end.get('drained') is True and old_end.get('drain_requested') is True
        and old_end.get('stop_requested') is False and old_end.get('expected')==1920 and old_end.get('completed')==count
        and old_end.get('failed')==0 and old_end.get('automatic_retry') is False and old_end.get('policy_sha256')==POLICY,
        'preserve actual partial naturally drained old753 evidence')
    require(plan.get('jobs')==_resumed(old_rows,set(done),entry) and len(plan['jobs'])==1920 and set(records)==ids
        and launch.get('imported_completed_cases')==plan.get('imported_completed_cases')==status.get('imported_completed')==terminal.get('imported_completed_cases')==count
        and terminal.get('resumed_new_cases')==status.get('CPU_launched')==1920-count,'exact mixed continuation plan/counts required')
    for row in old_rows:
        require(not ROW_PROVENANCE.intersection(row) and row.get('worker_sha256')==DRIVER['sha256'],'old rows remain original751')
        argv=row.get('argv');out=OLD_ART+'/test/'+row['task']+'/'+row['raw']+'/'+row['method']
        require(isinstance(argv,list) and argv[:3]==['bash','scripts/run_aligned_gpu.sh' if row['method']=='physmani' else 'scripts/run_local_sim.sh',DRIVER['path']], 'actual old751 argv required')
        for flag,value in (('--split','test'),('--task',row['task']),('--raw-episode-id',row['raw']),('--method',row['method']),
            ('--suite',SUITE),('--suite-lock',SUITE_LOCK),('--source-inputs',SOURCE_LOCK),('--manifest',MANIFEST),('--validation-release',VAL_RELEASE),('--out',out)):
            require(option(argv,flag)==value,'old751 scope changed: '+flag)
    old_ctl={}
    for key,name in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
        p=_pointer(control.get(key),OLD_CONTROLLER+'/'+name);require(_lookup(files,r.root,p['path'])==p['sha256'],'frozen old827 evidence required');old_ctl[name]=r.pointed(p)
    known.append(_pid(old_ctl['launch.json']))
    require(old_ctl['launch.json'].get('source_sha256')==SOURCE_SHA[OLD_CONTROLLER_SOURCE]
        and old_ctl['status.json'].get('stage')=='requires_concrete_repair' and old_ctl['status.json'].get('automatic_retry') is False
        and old_ctl['failure.json'].get('complete') is False and old_ctl['failure.json'].get('user_stop') is False
        and old_ctl['failure.json'].get('automatic_retry') is False,'genuine historical827 failure; never require a fabricated success terminal')
    historic=r.json(VAL_OWNER_STAGE+'/launch.json');known.append(_pid(historic))
    require(historic.get('source_sha256')==SOURCE_SHA[VAL_OWNER_SOURCE] and historic.get('argv',[])[:3]==[str(r.root/'.conda-env/bin/python'),'-u',VAL_OWNER_SOURCE]
        and option(historic['argv'],'--split')=='validation','real historical823 stage identity required')
    q=r.pointed(control['resource_qualification'],QUALIFICATION)
    require(q.get('schema')=='r10_concurrent_TRAIN_resource_qualification_v902' and q.get('complete') is True and q.get('split')=='train'
        and type(q.get('case_count')) is int and q['case_count']==6 and q.get('source_sha256')==SOURCE_SHA[QUALIFIER_SOURCE] and q.get('resource_profile')==PROFILE_LIMITS
        and q.get('execution_entry')==implementation and q.get('policy_sha256')==POLICY and q.get('author_outcome_protocol')==PROTOCOL
        and type(q.get('infrastructure_failed')) is int and q['infrastructure_failed']==0
        and all(q.get(k) is True for k in ('cleanup_completed','resource_bounds_verified','overlap_verified','independent_audit_complete'))
        and type(q.get('observed_native_peak')) is int and q['observed_native_peak']==4
        and type(q.get('observed_GPU_peak')) is int and q['observed_GPU_peak']==2
        and q.get('formal_score_contribution') is False and q.get('TEST_read') is False,'real independent904 resource qualification required; not a TEST release')
    audit_launch=r.pointed(_pointer(q.get('actual_independent_audit'),QUALIFICATION_LAUNCH));known.append(_pid(audit_launch))
    require(audit_launch.get('source_sha256')==SOURCE_SHA[QUALIFIER_SOURCE],'actual904 source identity required')
    cases,drivers,tiers=result.get('cases'),result.get('admission_driver_receipts'),result.get('evidence_tiers')
    require(all(isinstance(v,list) and len(v)==1920 for v in (cases,drivers,tiers)),'1920 ordered case/751/tier pointers required')
    sidecars=[];seen_case=set();seen_driver=set()
    for row,tier,case,driver in zip(plan['jobs'],tiers,cases,drivers):
        key=row['job_id'];adopt=row['imported_completed'];stage_owner=old_launch if adopt else launch;record=records[key]
        require(isinstance(record,dict) and record.get('complete') is True and type(record.get('returncode')) is int and record['returncode']==0
            and record.get('timed_out') is False and record.get('interrupted') is False and record.get('automatic_retry') is False
            and record.get('error') is None and record.get('leaf_future_exception') is not True
            and type(record.get('native_success')) is bool and 'motion_pass' in record
            and (record['motion_pass'] is None or type(record['motion_pass']) is bool),'only clean native outcomes count; motion unknown is retained')
        if adopt:require(record==old_records[key],'every historical nativeFalse/motionFalse outcome must stay literal')
        out=option(row['physical_argv'],'--out');_pointer(case,out+'/case.json');_pointer(driver,out+'/admission_driver_receipt.json')
        require(case['path'] not in seen_case and driver['path'] not in seen_driver and record.get('case')==case
            and record.get('admission_driver_receipt')==driver,'unique actual native pointer bindings required')
        seen_case.add(case['path']);seen_driver.add(driver['path'])
        require(tier==dict(job_id=key,evidence_runtime=row['evidence_runtime'],imported_completed=adopt,
            actual_stage_owner_pid=stage_owner['pid'],actual_stage_owner_start_ticks=stage_owner['start_ticks'],previous_row_sha256=row['previous_row_sha256'],
            execution_entry=row['execution_entry'],case=case,admission_driver_receipt=driver),'exact true evidence tier required')
        if adopt:require('batch_driver_receipt' not in record,'old751 cannot acquire a901 receipt')
        else:sidecars.append(_pointer(record.get('batch_driver_receipt'),out+'/batch_driver_receipt.json'))
    require(result.get('batch_driver_receipts')==sidecars,'all new901 and no old901 sidecar pointers required')
    expected_owners=[dict(runtime=OLD_RUNTIME,launch=control['previous_launch'],terminal=control['previous_terminal'],pid=old_launch['pid'],start_ticks=old_launch['start_ticks'],
        completed=count,resource_limits={k:old_launch[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}),
        dict(runtime=TEST_RUNTIME,launch=r.pointer(TEST_RUNTIME+'/launch.json'),terminal=r.pointer(TEST_RUNTIME+'/terminal.json'),pid=launch['pid'],start_ticks=launch['start_ticks'],
        completed=1920-count,resource_profile=PROFILE_LIMITS)]
    require(result.get('actual_stage_owners')==expected_owners,'two actual owners with dynamic counts required')
    inputs=result.get('inputs_sha256');names=(CONTROL,CONTROL_LOCK,OWNER_SOURCE,AUDITOR_SOURCE,OLD_OWNER_SOURCE,BASE_AUDITOR_SOURCE,QUALIFIER_SOURCE,QUALIFICATION,
        *[TEST_RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json')],
        *[OLD_RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json','DRAIN')],
        *[OLD_CONTROLLER+'/'+n for n in ('launch.json','status.json','failure.json')])
    require(all(_lookup(inputs,r.root,n)==r.inputs[n] for n in names),'actual906 consumed metadata SHA differs')
    require(_lookup(inputs,r.root,SOURCE_LOCK)==control['source_inputs']['sha256'],'original640 source lock must remain bound, not opened')
    require(len({(p['pid'],p['start_ticks']) for p in known})==len(known), 'each controller/owner/auditor must have a distinct process identity')
    r.close()
    return dict(schema=VERSION,complete=True,expected_cases=1920,source_cases=640,policy_sha256=POLICY,TEST_results=r.pointer(RESULT),
        resume_control=cp,resume_control_lock=lp,known_processes=known,inputs_sha256=dict(r.inputs),imported_completed=count,resumed_completed=1920-count,
        actual_controller=r.pointer(CONTROLLER+'/launch.json'),actual_controller_terminal=r.pointer(CONTROLLER+'/terminal.json'),
        actual_owner=r.pointer(TEST_RUNTIME+'/launch.json'),actual_owner_terminal=r.pointer(TEST_RUNTIME+'/terminal.json'),
        actual_audit_stage=r.pointer(AUDIT_STAGE+'/launch.json'),actual_stage_owners=deepcopy(expected_owners),
        source_case_bytes_read=False,source_manifest_or_raw_bytes_read=False,transitive_input_map_walked=False,
        scores_used_for_gate=False,success_rate_threshold_used=False,motion6_used_for_gate=False,new_release_granted=False,
        old775_terminal_required=False,old827_success_terminal_required=False,
        namespace_evidence_scope='recorded907/902/906 captures; historical753 has no per-child namespace receipt')


class _Permit:
    __slots__=('__weakref__',)


def require_permit(permit,root,*,allow_test_callbacks=False):
    require(type(permit) is _Permit and permit in _LEASES,'live owned resource permit required')
    lease=_LEASES[permit]
    require(lease['pid']==os.getpid() and lease['root']==Path(root).resolve(),'same-process canonical-root lease required')
    require(not lease['test_only'] or allow_test_callbacks is True,'CPU-injected callbacks cannot authorize native execution')
    lease['stop'](lease['root']);return deepcopy(lease['proof'])


@contextmanager
def reserved_resources_after_TEST(root,expected_test_result_sha256,*,resume_lock_sha256,enabled=False,read_bytes=None,
    stop_check=None,reserve=None,inventory=None,is_alive=None):
    """Future explicit lease; hold through caller cleanup. No runner is installed."""
    require(enabled is True,'future enabled=True opt-in required; default remains inert')
    require(valid_sha(expected_test_result_sha256) and valid_sha(resume_lock_sha256),'both external SHA pins required')
    root=Path(root).resolve();test_only=any(v is not None for v in (read_bytes,stop_check,reserve,inventory,is_alive))
    stop,hold,scan,alive=stop_check or _stop_guard,reserve or _reserve,inventory or _inventory,is_alive or _is_alive
    stop(root)
    with hold(root):
        stop(root)
        proof=require_complete_TEST_metadata(root,expected_test_result_sha256,resume_lock_sha256=resume_lock_sha256,read_bytes=read_bytes)
        require(not any(alive(p) for p in proof['known_processes']),'formal/predecessor/resource-audit process still live')
        snapshot=scan(root)
        require(isinstance(snapshot,dict) and snapshot.get('host_visibility_available') is True and snapshot.get('ancestor_chain_complete') is True
            and snapshot.get('same_uid_inventory_incomplete') is False and snapshot.get('live_project_native_processes')==[],
            'complete empty host-visible project inventory required')
        stop(root);permit=_Permit();_LEASES[permit]=dict(pid=os.getpid(),root=root,proof=deepcopy(proof),stop=stop,test_only=test_only)
        try:yield permit
        finally:_LEASES.pop(permit,None)
