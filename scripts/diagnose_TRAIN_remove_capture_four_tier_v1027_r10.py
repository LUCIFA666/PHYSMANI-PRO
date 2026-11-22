"""Disabled single-case TRAIN remove-cup expression diagnostic candidate.

The reviewed1026 metadata contract delegates to1025's actual full1013/1011/1012 and
1016/1015 gate. The same-process1025 lease protects original743/744 cleanup, the
unchanged892 observer's restoration and this entry's independent sidecar.
The892 observer itself has no permit API; no old completion gate is called.
No activation materials, jobs, qualification or physical-equivalence claim.
"""
import argparse
import ast
from contextlib import ExitStack, contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SOURCE = 'scripts/diagnose_TRAIN_remove_capture_four_tier_v1027_r10.py'
TEST_SOURCE = 'tests/test_TRAIN_remove_capture_four_tier_v1027.py'
WORKER = 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
RUNTIME_SOURCE = 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
OBSERVER_SOURCE = 'our_mani/multitask/train_remove_capture_ast_v892.py'
GUARD_SOURCE = 'our_mani/multitask/post_four_tier_TEST_resource_guard_v1025.py'
SHARED_GUARD = 'our_mani/multitask/post_TEST_resource_guard_v924.py'
GATE_SOURCE = 'scripts/diagnose_TRAIN_public_collision_four_tier_v1026_r10.py'
GATE_TEST = 'tests/test_TRAIN_public_collision_four_tier_v1026.py'
GATE_SHA = 'eb89c909b966e5e57177e4d5c6a51f35e5ff20b16b60cb1b8177519ca0943ec6'
GUARD_SHA = 'ddd798e3c631cada0595845e075733caa52a2433a996e747523ccdc9710a8096'
GATE_TEST_SHA = '195b991f9422c5491ee44ff807cb7c4f45ac0c7eb6014135c190fa7b56fcc31d'
ENV_SOURCE = 'scripts/run_frozen_system_environment_v820_r10.py'
ORIGINAL_RUNTIME = 'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'
OUTCOME_SOURCE = 'our_mani/multitask/native_outcome_sampling_v741.py'
TRAIN_MANIFEST = 'artifacts/framework/protocol/train_manifest.json'
TRAIN_INPUTS = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
CONFIG_BASE = 'artifacts/framework/multitask_r10/post_training_qualification_v680_setup'
CHECKPOINT = dict(path='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
    sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
RESULT = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011/TEST_results.json'
FORMAL_LOCK = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011_setup/resume_inputs.json'
SUMMARY = 'runtime/framework/formal_TEST_composite_resume_v1011/full_sixteen_task_motion_summary_v1015.json'
POST_RUNTIME = 'runtime/framework/post_complete_composite_TEST_motion_v1016'
POST_SETUP = 'artifacts/framework/multitask_r10/post_complete_composite_TEST_motion_v1016_setup'
POST_PATHS = dict(control=POST_SETUP+'/control.json',input_lock=POST_SETUP+'/inputs.json',
    **{key:POST_RUNTIME+'/'+key+'.json' for key in ('launch','status','terminal')},
    **{'analysis_'+key:POST_RUNTIME+'/stages/saved_motion_v1015/'+key+'.json' for key in ('launch','terminal')},
    analysis_stdout=POST_RUNTIME+'/stages/saved_motion_v1015/stdout.log')
SETUP = 'artifacts/framework/multitask_r10/TRAIN_remove_capture_four_tier_v1027_setup'
CONTROL, INPUTS = SETUP+'/control.json', SETUP+'/inputs.json'
RUNTIME = 'runtime/framework/TRAIN_remove_capture_four_tier_v1027'
OUTPUT = 'artifacts/framework/multitask_r10/TRAIN_remove_capture_four_tier_v1027/development_demo_only'
CONTROL_SCHEMA = 'r10_unactivated_TRAIN_remove_capture_four_tier_control_v1027'
LOCK_SCHEMA = 'r10_TRAIN_remove_capture_four_tier_inputs_v1027'
SCHEMA = 'r10_instrumented_TRAIN_remove_capture_four_tier_entry_v1027'
ENVIRONMENT = dict(
    wrapper=dict(path=ENV_SOURCE,sha256='3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3'),
    manifest=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json',sha256='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'),
    qualification=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json',sha256='1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa'))
TASKS = ('remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed')
METHODS = ('cv_geometry', 'context_geometry')
PREDICATE_PINS = {
 'physmani_skill_policy_v548.py':'cbc413b3d1c601344f7871b315cbe10abe1ad87533d6ef7210eff55952a982d2',
 'physmani_remove_stages_v611.py':'1d71b23d37d7c847c9614b1672598c907f20d8b7fce4e7f16b8af52b6e8658ba',
 'physmani_terminal_opening_v620.py':'2f27f0a4a1e3a5972ef5c026a05bd02ec489a223f1ec8a67920ac6880811292d',
 'physmani_remove_wp0_v637.py':'c131bb72c35fcca8e5f1981cee938ab0a3146efa1ee1b3652243e4d1d0077c62',
 'physmani_remove_source_roles_v653.py':'a5275003e73d654c20536f335a9c537874bc2e1d1b2220bffced469087d8f199',
 'physmani_remove_intercept_v667.py':'40693b3792250aea65b6b129984bd1a63e31d5997327a4850e2e2e9f7fe337bc',
 'physmani_geometry_policy_v518.py':'ea23722e08bd955675a33411bfc657258f1273534652e7a4a33dd12741c865e0',
}
# Source/class/method/AST-body fingerprints, independently narrower than the
# observer's six-method minimum. Python AST locations are deliberately omitted.
EXPECTED_BODIES = (
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','_completion_target','7adbc0399bc35780dae3ae5c809bfd702d25fe96774e53cef5d4b3eb204b391c'),
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','_public_capture','646ed94e257b76e6cbda727cf810775aab83888c56327352fd0adbe06f1813d6'),
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','_done','bd12dd04bc14f96a03778e04a20de1d1a9bc7c8f1a8ccf558dc3fed0724858d6'),
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','_advance','1d03da98bfdc7333e9761dccb9c839b8325d98ef2f12d504fbb81b0daa1295a4'),
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','select','afdd56b87fd994639944cc83870d0bdc7907057d9d100da4b1004fa7737819df'),
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','qualify_opening','2172053769134258efeaae66edeff1bda54bf8833249162c6c4136e21a6b0868'),
 ('physmani_skill_policy_v548.py','SkillGeometryIntents','confirm_opening','5aa5bdac86a0d53cdb23061b7cdb87f1da7f5667391f40add73221988c293fb4'),
 ('physmani_remove_stages_v611.py','RemoveAuthorIntents','_done','c18300a34bf235ce4407c0e7c443a89adb803f0f636c0229330494be2695a36b'),
 ('physmani_remove_stages_v611.py','RemoveAuthorIntents','qualify_opening','9b5569e0c9a48a8805f8d9413a7e0a78e036ec762c016cc8060877d0d72b417b'),
 ('physmani_terminal_opening_v620.py','OriginalTerminalOpening','qualify_opening','105e86adc150242e4a4d09494a0850a003090e170d7cde9827fc121758c312d4'),
 ('physmani_remove_wp0_v637.py','AuthorPreApproachRemoveIntents','_completion_target','325e49e0ae6556d2efc4297231b6ea3807335d593dcc0ff0c60f9e2c8d7af833'),
 ('physmani_remove_wp0_v637.py','AuthorPreApproachRemoveIntents','_advance','4c0dd5d542b7492e921ba91754792739ecddf6f1046e0c2dba5f833253c1ad6c'),
 ('physmani_remove_wp0_v637.py','AuthorPreApproachRemoveIntents','candidates','0ecf8fdc0b5ac792a002eeeb7043b4a197bc111306064746794e1da99a83c891'),
 ('physmani_remove_wp0_v637.py','AuthorPreApproachRemoveIntents','select','7cbe3d6ff897efaed6dba81639fb4584ca847261e903ad8a59ec163bcf677325'),
 ('physmani_remove_source_roles_v653.py','AuthorLiveRoleRemoveIntents','select','e9166a2bb1f23f443e2f04ea1da45404f55134813494cc0798dc2ae0e5af9dee'),
 ('physmani_remove_intercept_v667.py','InterceptRemoveIntents','candidates','a0affd625f1e949831b9c93d5f798c6578cffd68e843e9d162cb0cdf95f2197d'),
 ('physmani_geometry_policy_v518.py','PublicModelFrontend','ready','b159e9954d45f02b0e8d58eaf2759bf1057065e02a8c3f8354ce42de09a6755a'),
)
PINNED_SOURCES = {
 WORKER:'714fb3ba7e5f282cdd9f68618ed2615edbbed0d5ebfb1921ee7d202424ca6174',
 RUNTIME_SOURCE:'1ffb2922521da8b77b4e005ffa0a9b22d9d4bc02a4944253fbc76605ce6f930d',
 OBSERVER_SOURCE:'5ec7af7c4d9022dd3954ab13fed3ab34620f028728aec9a496bb369c3676d670',
 GUARD_SOURCE:GUARD_SHA,
 SHARED_GUARD:'635d7e7a570982599ffb912a8cb65b5badb25907b4100677159955e4eb3afbed',
 GATE_SOURCE:GATE_SHA,GATE_TEST:GATE_TEST_SHA,
 ENV_SOURCE:ENVIRONMENT['wrapper']['sha256'],
 ORIGINAL_RUNTIME:'e1ba1b1409762329d7624fc4e8e27cf2475860142d63de39e05f4fb08f58941f',
 OUTCOME_SOURCE:'2289a93fcb9d8c27a700d66164da7e2ea8fbcc0fa4f0c775e986ab9f40a9cb12',
 **{'our_mani/multitask/'+name:value for name,value in PREDICATE_PINS.items()},
}
SOURCES = tuple(dict.fromkeys((SOURCE,TEST_SOURCE,SHARED_GUARD,GATE_SOURCE,GATE_TEST,
 'tests/test_post_four_tier_TEST_resource_guard_v1025.py',OBSERVER_SOURCE,GUARD_SOURCE,WORKER,RUNTIME_SOURCE,ENV_SOURCE,ORIGINAL_RUNTIME,OUTCOME_SOURCE,
 'tests/test_train_remove_capture_ast_v892.py','tests/test_post_TEST_resource_guard_v924.py',
 'our_mani/multitask/rml_case_preflight_v742.py','our_mani/multitask/rml_component_domain_v640.py',
 'our_mani/multitask/evaluation_roster16_v283.py',
 'our_mani/multitask/rml_component_factory_v643.py','our_mani/multitask/rml_component_extensions_v708.py',
 'our_mani/multitask/rml_policy_components_v702.py','our_mani/multitask/frozen_input_edges_v463.py',
 'scripts/evaluate_multitask_case_r10.py','our_mani/multitask/evaluation.py',
 *('our_mani/multitask/'+name for name in PREDICATE_PINS))))
_ACTIVE_LOCK = threading.Lock()


def require(value,message):
    if not value:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def valid_sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None
def utc():return datetime.now(timezone.utc).isoformat()


def canonical(root,name):
    require(type(name) is str and name and not Path(name).is_absolute()
        and '..' not in Path(name).parts,'project-relative canonical path required')
    path=Path(root)/name
    require(path.resolve()==path and path.is_relative_to(Path(root)),'symlink/alias/outside project path refused')
    return path


def raw(root,name,expected=None):
    path=canonical(root,name);before=path.stat();value=path.read_bytes();after=path.stat()
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(canonical(root,name)==path and stamp(before)==stamp(after),'input changed while read: '+name)
    require(expected is None or valid_sha(expected) and sha(value)==expected,'input SHA changed: '+name)
    return value


def pointer(value,name=None):
    require(type(value) is dict and set(value)=={'path','sha256'} and valid_sha(value['sha256']),'exact path/SHA pointer required')
    if name is not None:require(value['path']==name,'unexpected pointer path')
    return value


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--control',default=CONTROL);p.add_argument('--lock',default=INPUTS)
    for name in ('lock-sha256','test-results-sha256','resume-lock-sha256','motion-summary-sha256','postprocessing-lock-sha256'):
        p.add_argument('--'+name)
    return p


def load_control(root,args):
    """Only own small declarations and code bytes, never TRAIN or TEST payloads."""
    root=Path(root).resolve()
    require(args.execute is True and all(valid_sha(getattr(args,k)) for k in
        ('lock_sha256','test_results_sha256','resume_lock_sha256','motion_summary_sha256','postprocessing_lock_sha256')),
        'explicit execution plus five external SHA pins required')
    require(args.control==CONTROL and args.lock==INPUTS,'fixed independent TRAIN supplement required')
    require(all(valid_sha(value) for value in PINNED_SOURCES.values()),'reviewed1025/1026 source pins are not final; activation disabled')
    lock_bytes=raw(root,INPUTS,args.lock_sha256);lock=json.loads(lock_bytes)
    require(lock.get('schema')==LOCK_SCHEMA and lock.get('approved_for_execution') is True and type(lock.get('files')) is dict,'approved small TRAIN supplement required')
    files=lock['files'];control_bytes=raw(root,CONTROL,files.get(str(root/CONTROL)));c=json.loads(control_bytes)
    require(c.get('schema')==CONTROL_SCHEMA and c.get('execution_enabled') is True
        and c.get('formal_score_contribution') is False and c.get('TEST_used_for_fitting') is False,'development-only explicit control required')
    selection=c.get('selection',{});require(set(selection)=={'task','method','raw_episode_id'},'one explicit selection required')
    task,method,episode=(selection.get(k) for k in ('task','method','raw_episode_id'))
    require(task in TASKS and method in METHODS and type(episode) is str and re.fullmatch('[A-Za-z0-9_-]+',episode),
        'only two TRAIN remove tasks and CV/Context methods')
    for key,name in (('configuration',CONFIG_BASE+'/'+task+'_ours_configuration.json'),('original_frozen_inputs',TRAIN_INPUTS),('train_manifest',TRAIN_MANIFEST)):
        pointer(c.get(key),name)
    require(c.get('checkpoint')==CHECKPOINT and c.get('environment')==ENVIRONMENT,'original671 and qualified820 bindings required')
    for key,name,expected in (('completed_TEST_results',RESULT,args.test_results_sha256),('formal_resume_lock',FORMAL_LOCK,args.resume_lock_sha256),
        ('saved_motion_summary',SUMMARY,args.motion_summary_sha256)):
        require(pointer(c.get(key),name)['sha256']==expected,'external completion SHA differs: '+key)
    post=c.get('postprocessing');require(type(post) is dict and set(post)==set(POST_PATHS),'eight actual1016/1015 metadata pointers required')
    for key,name in POST_PATHS.items():pointer(post[key],name)
    require(post['input_lock']['sha256']==args.postprocessing_lock_sha256,'external1016 lock SHA differs')
    require(c.get('expected_body_pins')==[list(row) for row in EXPECTED_BODIES],'exact seventeen original body pins required')
    suffix='/'+task+'/'+episode+'/'+method
    require(c.get('runtime')==RUNTIME+suffix and c.get('output')==OUTPUT+suffix,'fresh single-case1027 namespace required')
    require(type(c.get('max_calls')) is int and 1<=c['max_calls']<=10000,'bounded observer max_calls required')
    sources=c.get('diagnostic_sources');require(type(sources) is dict and set(sources)==set(SOURCES)
        and all(valid_sha(v) for v in sources.values()),'all diagnostic source SHA pins required')
    require(all(sources[name]==expected for name,expected in PINNED_SOURCES.items()),'original observer/physical/predicate or new gate source changed')
    expected={str(root/name):value for name,value in sources.items()}
    for value in (c['configuration'],c['original_frozen_inputs'],c['train_manifest'],*c['environment'].values(),*post.values()):
        canonical(root,value['path']);key=str(root/value['path'])
        require(key not in expected or expected[key]==value['sha256'],'conflicting small-lock source binding')
        expected[key]=value['sha256']
    expected[str(root/CONTROL)]=sha(control_bytes)
    require(files==expected,'exact source/declaration/completion small map required; no transitive graph')
    for name,checksum in sources.items():raw(root,name,checksum)
    return dict(control=c,control_pointer=dict(path=CONTROL,sha256=sha(control_bytes)),lock_pointer=dict(path=INPUTS,sha256=sha(lock_bytes)))


def forwarded_argv(binding,check=False):
    c=binding['control'];s=c['selection']
    argv=[WORKER,'--config',c['configuration']['path'],'--manifest',TRAIN_MANIFEST,'--frozen-inputs',TRAIN_INPUTS,
        '--task',s['task'],'--raw-episode-id',s['raw_episode_id'],'--method',s['method'],'--split','train',
        '--development-demo','--out',c['output']]
    return argv+(['--check'] if check else [])


def _import(root,name):
    text=str(root)
    if text not in sys.path:sys.path.insert(0,text)
    return importlib.import_module(name)

def _load_guard(root):return _import(root,'our_mani.multitask.post_four_tier_TEST_resource_guard_v1025')
def _load_observer(root):return _import(root,'our_mani.multitask.train_remove_capture_ast_v892')


def _load_gate(root):
    require(valid_sha(GATE_SHA),'reviewed1026 source pin is not final; activation disabled')
    raw(root,GATE_SOURCE,PINNED_SOURCES[GATE_SOURCE])
    module=_import(root,'scripts.diagnose_TRAIN_public_collision_four_tier_v1026_r10')
    require(Path(module.__file__).resolve()==canonical(root,GATE_SOURCE),'canonical reviewed1026 contract module required')
    return module


def require_complete_motion_metadata(root,binding,args):
    # The actual full1013/1011/1012 then1016/1015 gate is delegated, not duplicated.
    return _load_gate(root).require_complete_motion_metadata(root,binding,args)


def verify_training_bindings(root,binding):
    #1026's helper checks the selected task's original first four TRAIN parents;
    #it contains no peg-only task restriction and runs only inside the1025 lease.
    return _load_gate(root).verify_training_bindings(root,binding)


def _load_worker(root):
    worker=_import(root,'scripts.run_task_executor_smoke_author_outcome_v743_r10')
    edge=_import(root,'our_mani.multitask.frozen_input_edges_v463')
    evaluator=_import(root,'scripts.evaluate_multitask_case_r10')
    return worker,edge,evaluator


def _require_checkpoint(root,binding):
    require(binding['config'].get('context_checkpoint')==CHECKPOINT,'actual configuration checkpoint changed')
    raw(root,CHECKPOINT['path'],CHECKPOINT['sha256'])


def _environment(root,binding):
    env=binding['control']['environment'];module=_import(root,'scripts.run_frozen_system_environment_v820_r10')
    receipt=module.capture_namespace_evidence(root/env['manifest']['path'],env['manifest']['sha256'],
        root/env['qualification']['path'],env['qualification']['sha256'])
    require(receipt.get('complete') is True and receipt.get('wrapper')==env['wrapper']
        and receipt.get('manifest')==env['manifest'] and receipt.get('namespace_qualification')==env['qualification']
        and receipt.get('observer_pid')==os.getpid(),'actual same-process qualified v820 environment required')
    return receipt


def require_body_plan(root,observer,policy):
    require(observer.PINS==PREDICATE_PINS,'observer seven source pins changed')
    plan=observer.default_plan(policy);actual=[]
    for target,owner,name,source,checksum in plan:
        basename=Path(source).name;require(basename in PREDICATE_PINS and checksum==PREDICATE_PINS[basename],'unrecognized planned source')
        path='our_mani/multitask/'+basename
        require(Path(source).resolve()==canonical(root,path),'method source path differs from frozen source')
        tree=ast.parse(raw(root,path,checksum));classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner.__name__]
        require(len(classes)==1,'unique original policy class required')
        methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(methods)==1,'unique original method body required')
        actual.append((basename,owner.__name__,name,sha(ast.dump(methods[0],include_attributes=False).encode())))
    require(len(actual)==17 and len(set(actual))==17 and set(actual)==set(EXPECTED_BODIES),'exact actual seventeen-method MRO/body set required')
    return [list(row) for row in sorted(actual)]


@contextmanager
def scoped_geometry(root,runtime,observer,binding,events):
    require(_ACTIVE_LOCK.acquire(blocking=False),'one isolated same-thread diagnostic scope per process')
    original=runtime._geometry;thread=threading.get_ident();instances=[]
    try:
        with ExitStack() as stack:
            def geometry(p,env,descriptions):
                require(threading.get_ident()==thread,'diagnostic geometry escaped owner thread')
                require(not instances,'single case permits exactly one geometry construction')
                result=original(p,env,descriptions)
                require(type(result) is tuple and len(result)==5,'original geometry tuple required')
                s=binding['control']['selection'];args=p['args'];domain=p['component_domain']
                require(args.split=='train' and args.task==s['task'] and args.method==s['method'] and args.raw_episode_id==s['raw_episode_id']
                    and p['config'].get('context_checkpoint')==CHECKPOINT,'actual original prepared case differs from selected TRAIN route')
                require(domain.require(args.task,args.method)=='train_qualification' and domain.split=='train' and domain.raw==s['raw_episode_id'],
                    'actual source-bound TRAIN component capability required')
                body_plan=require_body_plan(root,observer,result[3])
                value=observer.TrainRemoveCaptureValueObserver(result[3],domain=domain,enabled=True,development_only=True,
                    max_calls=binding['control']['max_calls'])
                instances.append(value);stack.enter_context(value)
                events.update(expected_identity=p['identity'],source_row_sha256=sha(json.dumps(p['row'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()),
                    component_domain=domain.receipt(),expected_body_pins=body_plan)
                # No bind_context: no extra clock/phase/predicate/observe calls.
                return result
            with patch.object(runtime,'_geometry',geometry):yield
    finally:
        try:
            original_error=sys.exc_info()[1]
            try:
                events['observer_receipts']=[value.receipt() for value in instances]
                events['geometry_hook_restored']=runtime._geometry is original
            except BaseException as recording_error:
                events['observer_receipt_recording_error']=type(recording_error).__name__
                if original_error is None:raise
        finally:_ACTIVE_LOCK.release()


def _process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=int(fields[19]),executable=sys.executable,argv=list(sys.argv))


def write_fresh_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def _validate_native_case(case,events,binding):
    from our_mani.multitask.evaluation import validate_completed_case
    validate_completed_case(case,events['expected_identity'])
    c=binding['control'];s=c['selection']
    require(type(case.get('process_id')) is int and case['process_id']==os.getpid(),'case must be produced by this actual isolated process')
    require(case.get('split')=='train' and case.get('task')==s['task'] and case.get('method')==s['method']
        and case.get('source_raw_episode_id')==s['raw_episode_id'] and case.get('parent_raw_episode_id')==s['raw_episode_id']
        and case.get('shared_context_checkpoint')==CHECKPOINT and case.get('cleanup_completed') is True
        and case.get('formal_TEST_metric') is False and case.get('formal_validation_metric') is False
        and case.get('TEST_used_for_fitting') is False,'actual clean original TRAIN result required; native False is retained')
    identity=case['identity']
    require(identity.get('raw_episode_id')==s['raw_episode_id']+'__policy_aug_r0' and identity.get('parent_raw_episode_id')==s['raw_episode_id'],
        'original TRAIN parent/aug_r0 identity must remain unchanged')
    require(case.get('component_domain')==events['component_domain'],'actual case component domain changed')


def run(root,args):
    root=Path(root).resolve();binding=load_control(root,args);c=binding['control']
    readiness=require_complete_motion_metadata(root,binding,args);gate=_load_gate(root)
    out=canonical(root,c['output']);runtime=canonical(root,c['runtime']+('/CPU_check' if args.check else '/native'))
    require(not out.exists() and not runtime.exists(),'fresh output/runtime only; never overwrite or retry')
    guard=_load_guard(root)
    with guard.reserved_resources_after_TEST(root,args.test_results_sha256,**gate.guard_arguments(args),enabled=True) as permit:
        proof=guard.require_permit(permit,root,allow_test_callbacks=False)
        require(load_control(root,args)==binding and require_complete_motion_metadata(root,binding,args)==readiness,'TRAIN supplement or complete metadata changed before admission')
        config=verify_training_bindings(root,binding)
        _require_checkpoint(root,dict(binding,config=config));environment=_environment(root,binding)
        runtime.mkdir(parents=True,exist_ok=False)
        invocation=_process_identity();inner=forwarded_argv(binding,args.check)
        launch=dict(schema=SCHEMA,created_utc=utc(),actual_entry=dict(path=SOURCE,sha256=c['diagnostic_sources'][SOURCE]),
            invocation=invocation,forwarded_worker_argv=inner,base_worker=dict(path=WORKER,sha256=c['diagnostic_sources'][WORKER]),
            base_native_runtime=dict(path=RUNTIME_SOURCE,sha256=c['diagnostic_sources'][RUNTIME_SOURCE]),
            original_native_runtime=dict(path=ORIGINAL_RUNTIME,sha256=c['diagnostic_sources'][ORIGINAL_RUNTIME]),
            original_outcome_sampling=dict(path=OUTCOME_SOURCE,sha256=c['diagnostic_sources'][OUTCOME_SOURCE]),
            diagnostic_sources=c['diagnostic_sources'],expected_body_pins=c['expected_body_pins'],
            supplement=binding['control_pointer'],supplement_lock=binding['lock_pointer'],post_TEST_resource_proof=proof,
            complete_TEST_and_motion=readiness,metadata_gate_contract=dict(path=GATE_SOURCE,sha256=GATE_SHA),
            observer_consumes_resource_permit=False,entry_checks_1025_permit=True,
            actual_frozen_environment=environment,instrumented_entry=True,original_751_or_901_entry_claimed=False,
            formal_score_contribution=False,physical_equivalence_claimed=False,unqualified=True)
        write_fresh_json(runtime/'launch.json',launch)
        events={};failure=None;completion=None
        try:
            worker,edge,evaluator=_load_worker(root)
            require(load_control(root,args)==binding,'TRAIN source/control lock changed before original entry')
            guard.require_permit(permit,root,allow_test_callbacks=False)
            with ExitStack() as stack:
                # Reproduce the original743 __main__ outer scopes, in order.
                stack.enter_context(edge.declared_edge_scope());stack.enter_context(edge.evaluator_scope(evaluator))
                if not args.check:stack.enter_context(scoped_geometry(root,worker.runtime,_load_observer(root),binding,events))
                stack.enter_context(patch.object(sys,'argv',inner))
                if args.check:
                    stream=stack.enter_context(redirect_stdout(io.StringIO()));worker.main()
                    cpu=json.loads(stream.getvalue());require(cpu.get('complete') is True and cpu.get('native_started') is False
                        and cpu.get('GPU_started') is False and cpu.get('actual_split')=='train','original CPU proof required')
                    completion=dict(mode='CPU_check',original_CPU_proof=cpu,observer_installed=False,native_started=False)
                else:
                    # A saved CPU receipt is neither consumed nor a permit.
                    # Original743 runs original742.prepare/cpu_preflight again.
                    worker.main()  # Native stdout is passed through unchanged.
            if not args.check:
                receipts=events.get('observer_receipts',[])
                require(len(receipts)==1 and receipts[0].get('restored') is True and receipts[0].get('native_validated') is False
                    and receipts[0].get('unqualified') is True and events.get('geometry_hook_restored') is True,'original observer restored and remains unqualified')
                case_bytes=raw(root,c['output']+'/case.json');case=json.loads(case_bytes)
                _validate_native_case(case,events,binding)
                completion=dict(mode='instrumented_native_TRAIN',native_started=True,
                    case=dict(path=c['output']+'/case.json',sha256=sha(case_bytes)),native_success=case['official_task_success'],
                    diagnostic_events=events,case_original_bytes_unchanged=True)
                require(raw(root,c['output']+'/case.json')==case_bytes,'original case bytes changed during sidecar preparation')
            guard.require_permit(permit,root,allow_test_callbacks=False)
            require(load_control(root,args)==binding,'TRAIN supplement changed during original execution')
        except BaseException as error:
            failure=dict(type=type(error).__name__,original_exception_rethrown=True)
            try:
                write_fresh_json(runtime/'failure.json',dict(schema=SCHEMA,complete=False,failure=failure,diagnostic_events=events,
                    automatic_retry=False,infrastructure_failure_not_native_task_failure=True,formal_score_contribution=False))
            except Exception as recording_error:
                failure['recording_error']=type(recording_error).__name__
            raise
        finally:
            # Still inside the same-process1025 lease, after original744 cleanup
            # and class/helper restoration. This never edits original case.json.
            if failure is None and completion is not None:
                receipt=dict(schema=SCHEMA,complete=True,finished_utc=utc(),selection=c['selection'],split='train',
                    invocation=invocation,forwarded_worker_argv=inner,supplement=binding['control_pointer'],supplement_lock=binding['lock_pointer'],
                    actual_entry=launch['actual_entry'],base_worker=launch['base_worker'],base_native_runtime=launch['base_native_runtime'],
                    original_native_runtime=launch['original_native_runtime'],original_outcome_sampling=launch['original_outcome_sampling'],
                    diagnostic_sources=c['diagnostic_sources'],post_TEST_resource_proof=proof,actual_frozen_environment=environment,
                    complete_TEST_and_motion=readiness,metadata_gate_contract=dict(path=GATE_SOURCE,sha256=GATE_SHA),
                    observer_consumes_resource_permit=False,entry_checks_1025_permit=True,
                    actual_checkpoint=CHECKPOINT,execution=completion,instrumented_entry=True,unqualified=True,
                    native_validated=False,physical_equivalence_claimed=False,formal_score_contribution=False,
                    saved_CPU_receipt_used_as_native_permit=False,original_prepare_runs_on_each_invocation=True,
                    participates_in_VAL_TEST=False,TEST_used_for_fitting=False,policy_fix_claimed=False,
                    original_751_or_901_entry_claimed=False,automatic_retry=False,unknown_values_not_filled=True)
                write_fresh_json(runtime/'diagnostic_sidecar.json',receipt)
        return receipt


def main(argv=None):
    args=parser().parse_args(argv)
    if not args.execute:return dict(enabled=False,complete=False,native_started=False,GPU_started=False,materials_created=False)
    result=run(ROOT,args)
    print(json.dumps(dict(complete=True,schema=SCHEMA,formal_score_contribution=False,mode=result['execution']['mode'])))
    return result


if __name__=='__main__':main()
