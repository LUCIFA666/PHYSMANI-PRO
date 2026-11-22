"""Disabled single-case TRAIN ring collision-call diagnostic candidate.

Future activation needs an independently frozen supplement, direct full TEST
metadata, the actual completed920 saved-motion chain, and a same-process924
lease through original cleanup, tap restoration and sidecar writing. Nothing
creates activation materials. No predicate/author-stage rewrite, new probe,
formal qualification, physical-equivalence claim or automatic retry is made.
"""
import argparse
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

ROOT=Path(__file__).parents[1]
SOURCE='scripts/diagnose_TRAIN_public_collision_v936_r10.py'
TEST_SOURCE='tests/test_train_public_collision_entry_v936.py'
WORKER='scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
NATIVE_RUNTIME='scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
ORIGINAL_RUNTIME='scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'
OUTCOME='our_mani/multitask/native_outcome_sampling_v741.py'
TAP='our_mani/multitask/train_public_collision_tap_v934.py'
GUARD='our_mani/multitask/post_TEST_resource_guard_v924.py'
ENV_SOURCE='scripts/run_frozen_system_environment_v820_r10.py'
POST_SOURCE='scripts/continue_post_TEST_motion_v929_r10.py'
SUMMARY_SOURCE='scripts/summarize_full_TEST_motion_v920_r10.py'
TRAIN_MANIFEST='artifacts/framework/protocol/train_manifest.json'
TRAIN_INPUTS='artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
CONFIG_BASE='artifacts/framework/multitask_r10/post_training_qualification_v680_setup'
CHECKPOINT=dict(path='artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt',
    sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
ENVIRONMENT=dict(
    wrapper=dict(path=ENV_SOURCE,sha256='3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3'),
    manifest=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json',sha256='1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'),
    qualification=dict(path='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json',sha256='1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa'))
FORMAL_SETUP='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
FORMAL_CONTROL=FORMAL_SETUP+'/resume_control.json';FORMAL_LOCK=FORMAL_SETUP+'/resume_inputs.json'
RESULT='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/TEST_results.json'
SUMMARY='runtime/framework/formal_TEST_concurrent_resume_v902/full_sixteen_task_motion_summary_v920.json'
POST_RUNTIME='runtime/framework/post_complete_TEST_motion_v929'
POST_SETUP='artifacts/framework/multitask_r10/post_complete_TEST_motion_v929_setup'
POST_PATHS=dict(control=POST_SETUP+'/control.json',input_lock=POST_SETUP+'/inputs.json',
    **{key:POST_RUNTIME+'/'+key+'.json' for key in ('launch','status','terminal')},
    **{'analysis_'+key:POST_RUNTIME+'/stages/saved_motion_v920/'+key+'.json' for key in ('launch','terminal')},
    analysis_stdout=POST_RUNTIME+'/stages/saved_motion_v920/stdout.log')
SETUP='artifacts/framework/multitask_r10/TRAIN_public_collision_v936_setup'
CONTROL=SETUP+'/control.json';INPUTS=SETUP+'/inputs.json'
RUNTIME='runtime/framework/TRAIN_public_collision_v936'
OUTPUT='artifacts/framework/multitask_r10/TRAIN_public_collision_v936/development_demo_only'
SCHEMA='r10_instrumented_TRAIN_public_collision_entry_v936'
CONTROL_SCHEMA='r10_unactivated_TRAIN_public_collision_control_v936'
LOCK_SCHEMA='r10_TRAIN_public_collision_inputs_v936'
POLICY='eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
TASKS=('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed')
METHODS=('cv_geometry','context_geometry')
POST_PINS={
 POST_SOURCE:'bb9157554157ed4186fd46221d627062515ee52ea8b7dad96e131f1761195daf',
 'tests/test_post_TEST_motion_controller_v929.py':'661db9bdca71b2eee4d08d713b65c0036912410d8cab091c004d19c8b47c45e5',
 SUMMARY_SOURCE:'684c8c3f799b3177f0e8d328808d36ddd584f697bb6c37a7467ec4fbe79d463a',
 'tests/test_full_TEST_motion_v920.py':'7af00d2044b99ae9535be49a403a5ea5222e756da8bded36b15f80486607c5e1',
 'scripts/continue_concurrent_TEST_v907_r10.py':'31d0b31e869756779f7b8b61c7eaeea0d55289174769248a59775fadea9015af',
 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py':'ef55aaf86e9334dcae0f02113e76fcbdcc360709a769aaa34f2b1f33a77c6c43',
 'scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py':'d73cab726f8885b35d590051f29bfde710e6c6fa6aa6c5743704159a5bc93d44'}
PINNED_SOURCES={**POST_PINS,
 WORKER:'714fb3ba7e5f282cdd9f68618ed2615edbbed0d5ebfb1921ee7d202424ca6174',
 NATIVE_RUNTIME:'1ffb2922521da8b77b4e005ffa0a9b22d9d4bc02a4944253fbc76605ce6f930d',
 ORIGINAL_RUNTIME:'e1ba1b1409762329d7624fc4e8e27cf2475860142d63de39e05f4fb08f58941f',
 OUTCOME:'2289a93fcb9d8c27a700d66164da7e2ea8fbcc0fa4f0c775e986ab9f40a9cb12',
 TAP:'aef0493e4cb0b4e5894200dd5ca4699732a8bd07d3a1843d33854497bcdc6bea',
 GUARD:'635d7e7a570982599ffb912a8cb65b5badb25907b4100677159955e4eb3afbed',
 ENV_SOURCE:ENVIRONMENT['wrapper']['sha256']}
SOURCES=tuple(dict.fromkeys((SOURCE,TEST_SOURCE,*PINNED_SOURCES,
 'tests/test_train_public_collision_tap_v934.py','tests/test_post_TEST_resource_guard_v924.py',
 'our_mani/multitask/rml_case_preflight_v742.py','our_mani/multitask/rml_component_domain_v640.py',
 'our_mani/multitask/rml_component_factory_v643.py','our_mani/multitask/rml_component_extensions_v708.py',
 'our_mani/multitask/rml_policy_components_v702.py','our_mani/multitask/evaluation_roster16_v283.py',
 'our_mani/multitask/frozen_input_edges_v463.py','our_mani/multitask/evaluation.py',
 'scripts/evaluate_multitask_case_r10.py','our_mani/multitask/maps.py','our_mani/multitask/pipeline.py',
 'our_mani/contracts.py','our_mani/reachability.py','our_mani/multitask/physmani_skill_executor_v534.py',
 'our_mani/multitask/physmani_geometry_executor_v517.py','our_mani/multitask/physmani_skill_policy_v548.py',
 'our_mani/multitask/physmani_ring_compensation_v580.py','our_mani/multitask/physmani_ring_release_v590.py',
 'our_mani/multitask/physmani_terminal_opening_v620.py','our_mani/multitask/physmani_ring_release_contact_v657.py')))
_ACTIVE_LOCK=threading.Lock()


def require(value,message):
    if not value:raise ValueError(message)
def sha(value):return hashlib.sha256(value).hexdigest()
def valid_sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None
def utc():return datetime.now(timezone.utc).isoformat()


def canonical(root,name):
    require(type(name) is str and name and not Path(name).is_absolute() and '..' not in Path(name).parts,'canonical relative project path required')
    path=Path(root)/name
    require(path.resolve()==path and path.is_relative_to(Path(root)),'symlink/alias/outside project path refused')
    return path


def raw(root,name,expected=None):
    path=canonical(root,name);before=path.stat();value=path.read_bytes();after=path.stat()
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(canonical(root,name)==path and stamp(before)==stamp(after),'input changed while read')
    require(expected is None or valid_sha(expected) and sha(value)==expected,'bound input SHA changed: '+name)
    return value


def pointer(value,name=None):
    require(type(value) is dict and set(value)=={'path','sha256'} and valid_sha(value['sha256'])
        and type(value['path']) is str,'exact path/SHA pointer required')
    if name is not None:require(value['path']==name,'unexpected pointer path')
    return value


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--control',default=CONTROL);p.add_argument('--lock',default=INPUTS)
    for name in ('lock-sha256','test-results-sha256','resume-lock-sha256','motion-summary-sha256'):p.add_argument('--'+name)
    return p


def load_control(root,args):
    """Only own declarations/lock and code bytes; no TRAIN or saved TEST bytes."""
    root=Path(root).resolve()
    require(args.execute is True and all(valid_sha(getattr(args,k)) for k in
        ('lock_sha256','test_results_sha256','resume_lock_sha256','motion_summary_sha256')),'explicit execution plus four external SHA pins required')
    require(args.control==CONTROL and args.lock==INPUTS,'fixed fresh independent TRAIN supplement required')
    lb=raw(root,INPUTS,args.lock_sha256);lock=json.loads(lb)
    require(lock.get('schema')==LOCK_SCHEMA and lock.get('approved_for_execution') is True and type(lock.get('files')) is dict,'approved small TRAIN lock required')
    files=lock['files'];cb=raw(root,CONTROL,files.get(str(root/CONTROL)));c=json.loads(cb)
    require(c.get('schema')==CONTROL_SCHEMA and c.get('execution_enabled') is True and c.get('formal_score_contribution') is False
        and c.get('TEST_used_for_fitting') is False,'explicit development-only supplement required')
    s=c.get('selection',{});require(set(s)=={'task','method','raw_episode_id'},'one selected case required')
    task,method,episode=(s.get(k) for k in ('task','method','raw_episode_id'))
    require(task in TASKS and method in METHODS and type(episode) is str and re.fullmatch('[A-Za-z0-9_-]+',episode),'only two ring tasks and CV/Context TRAIN source allowed')
    for key,name in (('configuration',CONFIG_BASE+'/'+task+'_ours_configuration.json'),('original_frozen_inputs',TRAIN_INPUTS),('train_manifest',TRAIN_MANIFEST)):
        pointer(c.get(key),name)
    require(c.get('checkpoint')==CHECKPOINT and c.get('environment')==ENVIRONMENT,'actual671 and qualified original820 bindings required')
    for key,name,expected in (('completed_TEST_results',RESULT,args.test_results_sha256),('formal_resume_lock',FORMAL_LOCK,args.resume_lock_sha256),
        ('saved_motion_summary',SUMMARY,args.motion_summary_sha256)):
        require(pointer(c.get(key),name)['sha256']==expected,'external completion SHA differs: '+key)
    post=c.get('postprocessing');require(type(post) is dict and set(post)==set(POST_PATHS),'all eight actual929/920 metadata pointers required')
    for key,name in POST_PATHS.items():pointer(post[key],name)
    suffix='/'+task+'/'+episode+'/'+method
    require(c.get('runtime')==RUNTIME+suffix and c.get('output')==OUTPUT+suffix,'fresh isolated936 output/runtime required')
    require(type(c.get('max_events')) is int and 1<=c['max_events']<=10000,'bounded tap event count required')
    sources=c.get('diagnostic_sources');require(type(sources) is dict and set(sources)==set(SOURCES) and all(valid_sha(v) for v in sources.values()),'all diagnostic code sources required')
    require(all(sources[name]==expected for name,expected in PINNED_SOURCES.items()),'frozen base/tap/guard/postprocessing implementation changed')
    expected={str(root/name):value for name,value in sources.items()}
    for value in (c['configuration'],c['original_frozen_inputs'],c['train_manifest'],*c['environment'].values(),*post.values()):
        canonical(root,value['path']);key=str(root/value['path'])
        require(key not in expected or expected[key]==value['sha256'],'small-lock conflicting source mapping')
        expected[key]=value['sha256']
    expected[str(root/CONTROL)]=sha(cb)
    require(files==expected,'exact source/declaration/completion small map required; no transitive graph')
    # Do not read TRAIN configuration/manifest/745 bytes, environment data,
    # result, summary or completion evidence here. The gates own that ordering.
    for name,value in sources.items():raw(root,name,value)
    return dict(control=c,control_pointer=dict(path=CONTROL,sha256=sha(cb)),lock_pointer=dict(path=INPUTS,sha256=sha(lb)))


def _import(root,name):
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return importlib.import_module(name)
def _load_guard(root):return _import(root,'our_mani.multitask.post_TEST_resource_guard_v924')
def _load_tap(root):return _import(root,'our_mani.multitask.train_public_collision_tap_v934')
def _load_postprocessor(root):return _import(root,'scripts.continue_post_TEST_motion_v929_r10')


def _pid(record):
    require(type(record.get('pid')) is int and record['pid']>0 and type(record.get('start_ticks')) is int and record['start_ticks']>0,'actual PID/start identity required')
    return record['pid'],record['start_ticks']


def require_complete_motion_metadata(root,binding,args):
    """Direct full TEST first, then completed929/920 metadata; never score gates."""
    c=binding['control']
    proof=_load_guard(root).require_complete_TEST_metadata(root,args.test_results_sha256,resume_lock_sha256=args.resume_lock_sha256)
    require(proof.get('complete') is True and proof.get('expected_cases')==1920 and proof.get('source_cases')==640
        and proof.get('TEST_results')==c['completed_TEST_results'] and proof.get('resume_control_lock')==c['formal_resume_lock'],
        'direct complete907/902/906 full1920 prerequisite required before saved-motion evidence')
    require(proof.get('policy_sha256')==POLICY,'direct full TEST policy must remain unchanged')
    pointer(proof.get('resume_control'),FORMAL_CONTROL)
    pointer(proof.get('actual_controller'),'runtime/framework/ordered_concurrent_TEST_v907/launch.json')
    meta={key:json.loads(raw(root,p['path'],p['sha256'])) for key,p in c['postprocessing'].items()}
    ctl,lock,launch,state,end,child,child_end,stdout=(meta[k] for k in
        ('control','input_lock','launch','status','terminal','analysis_launch','analysis_terminal','analysis_stdout'))
    post=c['postprocessing'];resume=proof['resume_control'];formal=c['formal_resume_lock'];result=c['completed_TEST_results'];summaryp=c['saved_motion_summary']
    sources={name:dict(path=name,sha256=value) for name,value in POST_PINS.items()}
    require(ctl.get('schema')=='r10_readonly_post_complete_TEST_motion_control_v929' and ctl.get('execution_enabled') is True
        and ctl.get('read_only_analysis') is True and ctl.get('automatic_retry') is False and ctl.get('policy_sha256')==POLICY
        and ctl.get('sources')==sources and ctl.get('resume_control')==resume and ctl.get('resume_control_lock')==formal
        and ctl.get('runtime')==POST_RUNTIME and ctl.get('test_results')==RESULT and ctl.get('motion_summary')==SUMMARY
        and ctl.get('controller_runtime')=='runtime/framework/ordered_concurrent_TEST_v907'
        and ctl.get('owner_runtime')=='runtime/framework/formal_TEST_concurrent_resume_v902'
        and ctl.get('controller_launch')==proof['actual_controller'],'actual bound929 control required')
    require(lock.get('schema')=='r10_readonly_post_complete_TEST_motion_inputs_v929' and lock.get('execution_enabled') is True
        and type(lock.get('files')) is dict and len(lock['files'])==11,'actual929 activation input lock required')
    post_module=_load_postprocessor(root)
    for p in (post['control'],resume,formal,ctl.get('controller_launch'),*sources.values()):
        pointer(p);require(post_module._lookup(lock.get('files'),root,p['path'])==p['sha256'],'actual929 frozen source/batch binding differs')
    require(_pid(launch) and launch.get('source_sha256')==POST_PINS[POST_SOURCE] and launch.get('control')==post['control']
        and launch.get('input_lock')==post['input_lock'] and launch.get('monitored_controller_launch')==ctl.get('controller_launch')
        and launch.get('standard_host_analysis') is True and launch.get('frozen_environment_inherited_claimed') is False
        and launch.get('automatic_retry') is False,'actual929 launch/source/control binding required')
    argv=launch.get('argv');require(type(argv) is list and all(type(v) is str for v in argv) and argv
        and argv[0] in (POST_SOURCE,str(Path(root)/POST_SOURCE)),'actual929 invocation required')
    parsed=argparse.ArgumentParser(add_help=False,allow_abbrev=False);parsed.add_argument('--execute',action='store_true')
    parsed.add_argument('--control',default=POST_PATHS['control']);parsed.add_argument('--lock',default=POST_PATHS['input_lock']);parsed.add_argument('--lock-sha256')
    # Parse only the recorded small argv, with no @file expansion or actions.
    try:recorded,unknown=parsed.parse_known_args(argv[1:])
    except SystemExit as error:raise ValueError('malformed recorded929 invocation') from error
    require(not unknown and argv.count('--execute')==1 and argv.count('--lock-sha256')==1
        and argv.count('--control')<=1 and argv.count('--lock')<=1 and recorded.execute is True
        and recorded.control==POST_PATHS['control'] and recorded.lock==POST_PATHS['input_lock']
        and recorded.lock_sha256==post['input_lock']['sha256'],'recorded929 invocation must bind actual activation lock')
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False and state.get('motion_summary')==summaryp,'929 must actually complete')
    require(end.get('schema')=='r10_readonly_post_complete_TEST_motion_terminal_v929' and end.get('complete') is True
        and end.get('motion_summary')==summaryp and end.get('TEST_results')==result and end.get('control')==post['control']
        and end.get('input_lock')==post['input_lock'] and end.get('resume_control')==resume and end.get('resume_control_lock')==formal
        and end.get('standard_host_analysis') is True and end.get('frozen_environment_inherited_claimed') is False
        and end.get('Native_GPU_TRAIN_started') is False and end.get('TXT_modified') is False and end.get('automatic_retry') is False,'actual929 final terminal required')
    expected=[str(Path(root)/'.conda-env/bin/python'),'-B','-u',SUMMARY_SOURCE,'--test-results-sha256',result['sha256'],
        '--resume-lock-sha256',formal['sha256'],'--resume-control',FORMAL_CONTROL,'--resume-lock',FORMAL_LOCK]
    require(_pid(child) and _pid(child)!=_pid(launch) and child.get('argv')==expected and child.get('source')==sources[SUMMARY_SOURCE]
        and child.get('source_sha256')==POST_PINS[SUMMARY_SOURCE] and (child.get('owner_pid'),child.get('owner_start_ticks'))==_pid(launch)
        and child.get('control')==post['control'] and child.get('input_lock')==post['input_lock'] and child.get('TEST_results')==result
        and child.get('standard_host_analysis') is True and child.get('frozen_environment_inherited_claimed') is False
        and child.get('automatic_retry') is False,'real920 child invocation/PID/source required')
    require(child_end.get('complete') is True and type(child_end.get('returncode')) is int and child_end['returncode']==0
        and child_end.get('interrupted') is False and child_end.get('automatic_retry') is False,'actual clean920 child terminal required')
    # First saved920 output read, after direct full TEST and completed929/child.
    summary=json.loads(raw(root,SUMMARY,summaryp['sha256']))
    post_module.validate_summary(summary,summaryp,stdout,ctl,dict(sources=sources),
        dict(result=result,result_sha256=result['sha256']),root)
    return dict(complete=True,direct_full_TEST=proof,summary=summaryp,postprocessing=post,
        actual920_pid=child['pid'],actual920_start_ticks=child['start_ticks'],score_or_motion_values_used=False,
        case_or_raw_bytes_read_by_this_gate=False,transitive_map_walked=False)


def verify_training_bindings(root,binding):
    c=binding['control']
    for p in (c['configuration'],c['original_frozen_inputs'],c['train_manifest'],*c['environment'].values()):raw(root,p['path'],p['sha256'])
    config=json.loads(raw(root,c['configuration']['path'],c['configuration']['sha256']))
    require(config.get('context_checkpoint')==CHECKPOINT,'original680 configuration must retain actual671')
    roster=_import(root,'our_mani.multitask.evaluation_roster16_v283').validate_roster(json.loads(raw(root,TRAIN_MANIFEST,c['train_manifest']['sha256'])),'train')
    s=c['selection'];fixed=[row['raw_episode_id'] for row in roster if row['task']==s['task']][:4]
    require(len(fixed)==4 and s['raw_episode_id'] in fixed,'original fixed first four TRAIN parents only')
    return config


def _require_checkpoint(root,binding):raw(root,CHECKPOINT['path'],CHECKPOINT['sha256'])


def _environment(root,binding):
    e=binding['control']['environment'];m=_import(root,'scripts.run_frozen_system_environment_v820_r10')
    receipt=m.capture_namespace_evidence(root/e['manifest']['path'],e['manifest']['sha256'],root/e['qualification']['path'],e['qualification']['sha256'])
    require(receipt.get('complete') is True and receipt.get('observer_pid')==os.getpid() and receipt.get('wrapper')==e['wrapper']
        and receipt.get('manifest')==e['manifest'] and receipt.get('namespace_qualification')==e['qualification'],'actual same-process820 environment required')
    return receipt


def _load_worker(root):
    return (_import(root,'scripts.run_task_executor_smoke_author_outcome_v743_r10'),
        _import(root,'our_mani.multitask.frozen_input_edges_v463'),_import(root,'scripts.evaluate_multitask_case_r10'))


def forwarded_argv(binding,check=False):
    c=binding['control'];s=c['selection']
    argv=[WORKER,'--config',c['configuration']['path'],'--manifest',TRAIN_MANIFEST,'--frozen-inputs',TRAIN_INPUTS,
        '--task',s['task'],'--raw-episode-id',s['raw_episode_id'],'--method',s['method'],'--split','train','--development-demo','--out',c['output']]
    return argv+(['--check'] if check else [])


@contextmanager
def scoped_geometry(root,runtime,tap_module,binding,permit,events):
    require(_ACTIVE_LOCK.acquire(blocking=False),'one isolated diagnostic scope per process')
    original=runtime._geometry;thread=threading.get_ident();instances=[]
    try:
        with ExitStack() as stack:
            def geometry(p,env,descriptions):
                require(threading.get_ident()==thread and not instances,'single case owner-thread geometry construction only')
                result=original(p,env,descriptions)
                require(type(result) is tuple and len(result)==5,'original geometry tuple required')
                s=binding['control']['selection'];a=p['args'];domain=p['component_domain']
                require(a.split=='train' and a.task==s['task'] and a.method==s['method'] and a.raw_episode_id==s['raw_episode_id']
                    and p['config'].get('context_checkpoint')==CHECKPOINT and domain.require(a.task,a.method)=='train_qualification'
                    and domain.split=='train' and domain.raw==s['raw_episode_id'],'actual original source-bound TRAIN route required')
                port=result[0].collision_port
                require(port.map is env.map_service and port.model is env.robot_model,'actual original actor public collision assembly required')
                tap=tap_module.TrainPublicCollisionTap(env.map_service,port=port,domain=domain,permit=permit,root=root,
                    enabled=True,development_only=True,max_events=binding['control']['max_events'])
                instances.append(tap);stack.enter_context(tap)
                events.update(expected_identity=p['identity'],component_domain=domain.receipt(),
                    source_row_sha256=sha(json.dumps(p['row'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()))
                return result
            with patch.object(runtime,'_geometry',geometry):yield
    finally:
        try:
            original_error=sys.exc_info()[1]
            try:
                events['tap_receipts']=[tap.receipt() for tap in instances]
                events['geometry_hook_restored']=runtime._geometry is original
            except BaseException as recording_error:
                events['tap_receipt_recording_error']=type(recording_error).__name__
                if original_error is None:raise
        finally:_ACTIVE_LOCK.release()


def _process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=int(fields[19]),executable=sys.executable,argv=list(sys.argv))


def write_fresh_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def _validate_native_case(case,events,binding):
    from our_mani.multitask.evaluation import validate_completed_case
    validate_completed_case(case,events['expected_identity']);s=binding['control']['selection'];identity=case['identity']
    require(type(case.get('process_id')) is int and case['process_id']==os.getpid() and case.get('split')=='train'
        and case.get('task')==s['task'] and case.get('method')==s['method'] and case.get('source_raw_episode_id')==s['raw_episode_id']
        and case.get('parent_raw_episode_id')==s['raw_episode_id'] and identity.get('raw_episode_id')==s['raw_episode_id']+'__policy_aug_r0'
        and identity.get('parent_raw_episode_id')==s['raw_episode_id'] and case.get('shared_context_checkpoint')==CHECKPOINT
        and case.get('component_domain')==events['component_domain'] and case.get('cleanup_completed') is True
        and case.get('formal_TEST_metric') is False and case.get('formal_validation_metric') is False
        and case.get('TEST_used_for_fitting') is False,'original clean TRAIN case identity/cleanup required; nativeFalse retained')


def run(root,args):
    root=Path(root).resolve();binding=load_control(root,args);c=binding['control']
    readiness=require_complete_motion_metadata(root,binding,args)
    out=canonical(root,c['output']);runtime=canonical(root,c['runtime']+('/CPU_check' if args.check else '/native'))
    require(not out.exists() and not runtime.exists(),'fresh outputs only; never retry/overwrite')
    guard=_load_guard(root)
    with guard.reserved_resources_after_TEST(root,args.test_results_sha256,resume_lock_sha256=args.resume_lock_sha256,enabled=True) as permit:
        resource=guard.require_permit(permit,root)
        require(load_control(root,args)==binding and require_complete_motion_metadata(root,binding,args)==readiness,'completion/source supplement changed before TRAIN')
        verify_training_bindings(root,binding);_require_checkpoint(root,binding);environment=_environment(root,binding)
        runtime.mkdir(parents=True,exist_ok=False);invocation=_process_identity();inner=forwarded_argv(binding,args.check)
        sources={name:dict(path=name,sha256=value) for name,value in c['diagnostic_sources'].items()}
        launch=dict(schema=SCHEMA,created_utc=utc(),invocation=invocation,forwarded_worker_argv=inner,actual_entry=sources[SOURCE],
            base_worker=sources[WORKER],base_native_runtime=sources[NATIVE_RUNTIME],original_native_runtime=sources[ORIGINAL_RUNTIME],
            original_outcome_sampling=sources[OUTCOME],tap=sources[TAP],diagnostic_sources=c['diagnostic_sources'],
            supplement=binding['control_pointer'],supplement_lock=binding['lock_pointer'],complete_TEST_and_motion=readiness,
            post_TEST_resource_proof=resource,actual_frozen_environment=environment,original_751_or_901_entry_claimed=False,
            instrumented_entry=True,unqualified=True,physical_equivalence_claimed=False,planning_wall_clock_invariance_claimed=False)
        write_fresh_json(runtime/'launch.json',launch);events={};completion=None;failure=None
        try:
            worker,edge,evaluator=_load_worker(root)
            require(load_control(root,args)==binding,'source/control changed before original entry');guard.require_permit(permit,root)
            with ExitStack() as stack:
                stack.enter_context(edge.declared_edge_scope());stack.enter_context(edge.evaluator_scope(evaluator))
                if not args.check:stack.enter_context(scoped_geometry(root,worker.runtime,_load_tap(root),binding,permit,events))
                stack.enter_context(patch.object(sys,'argv',inner))
                if args.check:
                    stream=stack.enter_context(redirect_stdout(io.StringIO()));worker.main();cpu=json.loads(stream.getvalue())
                    require(cpu.get('complete') is True and cpu.get('native_started') is False and cpu.get('GPU_started') is False
                        and cpu.get('actual_split')=='train','actual original743 CPU proof required')
                    completion=dict(mode='CPU_check',original_CPU_proof=cpu,tap_installed=False,native_started=False)
                else:worker.main()  # Native stdout unchanged; original prepare/preflight runs again.
            if not args.check:
                taps=events.get('tap_receipts',[])
                require(len(taps)==1 and taps[0].get('restored') is True and taps[0].get('native_validated') is False
                    and taps[0].get('unqualified') is True and events.get('geometry_hook_restored') is True,'tap must be restored and remain unqualified')
                cb=raw(root,c['output']+'/case.json');case=json.loads(cb);_validate_native_case(case,events,binding)
                completion=dict(mode='instrumented_native_TRAIN',native_started=True,case=dict(path=c['output']+'/case.json',sha256=sha(cb)),
                    native_success=case['official_task_success'],diagnostic_events=events,case_original_bytes_unchanged=True)
                require(raw(root,c['output']+'/case.json')==cb,'original case bytes changed')
            guard.require_permit(permit,root)
            require(load_control(root,args)==binding,'source/control changed during original execution')
        except BaseException as error:
            failure=dict(type=type(error).__name__,original_exception_rethrown=True)
            try:write_fresh_json(runtime/'failure.json',dict(schema=SCHEMA,complete=False,failure=failure,diagnostic_events=events,automatic_retry=False))
            except Exception:pass
            raise
        finally:
            if failure is None and completion is not None:
                receipt=dict(launch,complete=True,finished_utc=utc(),selection=c['selection'],split='train',actual_checkpoint=CHECKPOINT,
                    execution=completion,native_validated=False,formal_qualification=False,formal_score_contribution=False,
                    participates_in_VAL_TEST=False,TEST_used_for_fitting=False,policy_fix_claimed=False,automatic_retry=False,
                    predicates_rewritten=False,author_stages_observed=False,saved_CPU_receipt_used_as_native_permit=False,
                    original_prepare_runs_on_each_invocation=True,unknown_values_not_filled=True)
                write_fresh_json(runtime/'diagnostic_sidecar.json',receipt)
        return receipt


def main(argv=None):
    args=parser().parse_args(argv)
    if not args.execute:return dict(enabled=False,complete=False,native_started=False,GPU_started=False,materials_created=False)
    result=run(ROOT,args);print(json.dumps(dict(complete=True,schema=SCHEMA,formal_score_contribution=False,mode=result['execution']['mode'])))
    return result


if __name__=='__main__':main()
