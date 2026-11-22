"""Deferred semantic-eV2 component owner, default metadata-only/no children.

Root explicitly executes an externally hashed plan. The complete prior CF and
centre chain is required; only its actual child wait receipts are asserted.
The Root parent process wait is neither inferred nor fabricated. This owner
holds the correct runtime/framework lock pair around each GPU child and passes
--pipeline-owned-locks so unchanged eV2 skips its legacy wrong-directory locks. No Native VAL/TEST is enabled by this runner.
"""
import argparse
from contextlib import contextmanager
import fcntl
import os
import subprocess
from datetime import datetime,timezone
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'scripts/run_large_network_training_continuation_20261104_v1.py'
spec=importlib.util.spec_from_file_location('frozen_continuation_lifecycle',HELPER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SCHEMA='semantic_task_e207M_v2_deferred_owner_20261104_v3'
DEFAULT_PLAN=ROOT/'config/framework/semantic_task_e_v2_deferred_20261104_v3/plan.json'
LOCK=ROOT/'runtime/framework/semantic_task_e_v2_deferred_20261104_v3.lock'
CENTRE_HELPER=ROOT/'scripts/run_centre_large_v2_continuation_20261104_v1.py'
CENTRE_HELPER_SHA='b3b2c7cd11ce92b0a18122453aa7593999761408214a1176d9cb58809a539763'
CENTRE_CONFIG_SHA='c53c45f7915041391bc5a05c0a5a073c820e82df18bfc2c74d00c26a73649c62'
PREVIOUS_ROOT_SHA='6014be6bc74a00a1d8718f2db954c7592d4544cfd7973547dcddecd241412d8f'
CENTRE_STAGES=base.STAGES[6:]
STAGES=('e2_GPU_preflight','e2_fit10','e2_selected646_evaluation')
RESOURCE_NAMES=('fixed_TRAIN104_geometry_native.lock','TRAIN_postcomplete_diagnostics_v1106.lock')
_ACTIVE_RESOURCE_STATE=None
read=base.read;sha=base.sha;pointer=base.pointer;checked_pointer=base.checked_pointer
write_new=base.write_new;append=base.append


def now():return datetime.now(timezone.utc).isoformat()
def exact_zero_wait(value,*,source=True,stage=None):
 if (type(value.get('actual_process_returncode'))is not int or value['actual_process_returncode']!=0
     or value.get('actual_wait_completed')is not True
     or source and value.get('source_unchanged')is not True
     or value.get('stop_reason')is not None or value.get('error')is not None
     or value.get('unreaped_descendants',[])!=[]):raise ValueError('Actual zero/wait/source/no-unreaped child receipt required')
 if stage is not None and value.get('stage')!=stage:raise ValueError('Exact child stage identity required')
 return value


def resource_paths():return tuple(ROOT/'runtime/framework'/name for name in RESOURCE_NAMES)
def resource_policy():
 return dict(owner='this_deferred_v3_parent_for_e2',paths=[str(p)for p in resource_paths()],order=list(RESOURCE_NAMES),child_CLI_bypass_argument='--pipeline-owned-locks',held_while_waiting_for_prior_chain=False,held_from_stage_launch_through_actual_wait_and_descendant_cleanup=True)

@contextmanager
def correct_stage_resource_locks(plan,name):
 global _ACTIVE_RESOURCE_STATE
 if _ACTIVE_RESOURCE_STATE is not None:raise ValueError('Nested or duplicate stage resource ownership forbidden')
 # Called only after all prior-chain proofs validate, never around the wait.
 log=Path(plan['output_root'])/'resource_ownership'/name
 log.mkdir(parents=True,exist_ok=False)
 write_new(log/'request.json',dict(stage=name,requested_utc=now(),policy=resource_policy()))
 handles=[];acquired=[];begin=time.monotonic()
 try:
  for path in resource_paths():
   path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('a+');handles.append(stream)
   while True:
    assert_running(plan)
    if time.monotonic()-begin>plan['resource_wait_timeout_seconds']:raise TimeoutError('Correct resource locks busy')
    try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:time.sleep(min(plan['poll_seconds'],.2))
   stat=os.fstat(stream.fileno());acquired.append(dict(path=str(path.resolve()),device=stat.st_dev,inode=stat.st_ino))
  receipt=dict(stage=name,owner_pid=os.getpid(),acquired_utc=now(),actual_outer_locks_held=True,correct_Native_centre_directory=True,locks=acquired,child_must_use_pipeline_owned_locks=True)
  write_new(log/'acquired.json',receipt)
  receipt=dict(receipt,receipt_file=pointer(log/'acquired.json'))
  _ACTIVE_RESOURCE_STATE=dict(stage=name,receipt=receipt,handles=handles)
  yield receipt
 finally:
  _ACTIVE_RESOURCE_STATE=None
  for stream in reversed(handles):
   try:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
   finally:stream.close()
  terminal=Path(plan['output_root'])/'owners'/name/'actual_exit.json'
  actual=read(terminal)if terminal.exists()else None
  write_new(log/'released.json',dict(stage=name,released_utc=now(),locks=acquired,locks_released=True,child_started=actual is not None,actual_child_wait_completed=None if actual is None else actual['actual_wait_completed'],unreaped_descendants=[]if actual is None else actual['unreaped_descendants'],stage_actual_exit=None if actual is None else pointer(terminal)))


def load_plan(path,expected_sha=None):
 path=Path(path).resolve(strict=True)
 if expected_sha is not None and sha(path)!=expected_sha:raise ValueError('External frozen deferred plan SHA required')
 plan=read(path)
 if (plan.get('schema')!=SCHEMA or plan.get('component_only')is not True
     or plan.get('all_task_Native_VAL_TEST_authorized')is not False or tuple(plan.get('stages',()))!=STAGES):raise ValueError('Fixed component-only deferred stage protocol required')
 for key in ('prior_config','v2_delivery','v2_method_sources','v2_config','split_manifest','preprocess_completion','shared_registry','e1_actual_exit','method_sources','centre_config','centre_method_sources','previous_root_exit'):
  checked_pointer(plan[key])
 for key in ('python','output_root','prior_continuation_root','CF_output_root'):
  if not isinstance(plan.get(key),str)or not Path(plan[key]).is_absolute():raise ValueError('Explicit absolute deferred paths required')
 if set(plan.get('stage_timeout_seconds',{}))!=set(STAGES):raise ValueError('All three bounded stage timeouts required')
 for value in(plan['wait_timeout_seconds'],plan['poll_seconds'],plan['stop_grace_seconds'],plan['resource_wait_timeout_seconds'],*plan['stage_timeout_seconds'].values()):
  if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:raise ValueError('Finite positive deferred time bounds required')
 prior_config=checked_pointer(plan['prior_config']);old=base.load_config(prior_config,plan['prior_config']['sha256'])
 v2=read(checked_pointer(plan['v2_method_sources']))
 if len(v2)!=20 or any(sha(p)!=h for p,h in v2.items()):raise ValueError('Exact independent frozen e2 source20 required')
 if v2.get(str(ROOT/'our_mani/semantic_task_e_large_20261104_v2/model.py'))!='1487995e8b452939c1faa839063141bc9899bfbf68675302431120f8c04045e8':raise ValueError('Normalized raw-decoder207M model pin required')
 if plan['v2_config']['sha256']!='70fe807233a0ea531391a3508bbd5534a05445ee8d93046c1c90986b6d81dc2f':raise ValueError('Root-fixed v2 configuration required')
 if (plan['split_manifest']!=old['split_manifest']or plan['preprocess_completion']!=old['preprocess_completion']
     or plan['e1_actual_exit']['path']!=old['e_actual_exit']or plan['CF_output_root']!=old['CF_output_root']
     or plan['prior_continuation_root']!=old['output_root']):raise ValueError('Exact prior continuation/data/CF identity required')
 if ROOT/'artifacts/framework/local_mpc_20261102'not in Path(plan['output_root']).parents:raise ValueError('Fresh component artifact owner required')
 if Path(plan['output_root'])==Path(plan['prior_continuation_root']):raise ValueError('Prior owner output must not be reused')
 if plan.get('resource_lock_policy')!=resource_policy():raise ValueError('Exact correct-directory outer ownership and CLI bypass required')
 if plan['centre_config']['sha256']!=CENTRE_CONFIG_SHA or plan['previous_root_exit']['sha256']!=PREVIOUS_ROOT_SHA or sha(CENTRE_HELPER)!=CENTRE_HELPER_SHA:raise ValueError('Exact Root-approved failed-prefix and deterministic-centre successor required')
 centre=read(checked_pointer(plan['centre_config']))
 if plan['centre_continuation_root']!=centre['output_root']or centre['previous_root_exit']!=plan['previous_root_exit']or centre['method_sources']!=plan['centre_method_sources']:raise ValueError('Exact independent centre continuation identity required')
 base.verify_sources(dict(method_sources=plan['centre_method_sources']))
 exact_zero_wait(read(checked_pointer(plan['e1_actual_exit'])))
 plan['_config_path']=str(path);plan['_config_sha256']=sha(path)
 base.verify_sources(plan)
 return plan


def assert_running(plan):
 base.assert_running(plan)
 if (DEFAULT_PLAN.parent/'STOP').exists():raise InterruptedError('Deferred owner STOP')


def wait_for_prior(plan):
 root=Path(plan['centre_continuation_root']);begin=time.monotonic()
 while True:
  assert_running(plan)
  if(root/'failure.json').exists():raise ValueError('Independent new centre continuation failed; no e2 authorization/retry')
  if(root/'completion.json').is_file()and(root/'terminal_status.json').is_file():return
  if time.monotonic()-begin>plan['wait_timeout_seconds']:raise TimeoutError('Waiting for explicit new centre6 completion')
  time.sleep(plan['poll_seconds'])


def validate_prior_stage_rows(rows,expected,*,owner_rows=None,request_rows=None,expected_sources=None):
 if len(rows)!=len(expected)or [r.get('stage')for r in rows]!=list(expected):raise ValueError('All exact twelve ordered prior stages are required')
 for index,(name,row)in enumerate(zip(expected,rows)):
  exact_zero_wait(row,stage=name)
  if owner_rows is not None and row!=owner_rows[index]:raise ValueError('Ledger and actual owner wait file disagree')
  if request_rows is not None:
   request=request_rows[index]
   if request.get('stage')!=name or request.get('source_sha256')!=expected_sources or request.get('driver_holds_resource_locks')is not False:raise ValueError('Actual prior child request/source identity required')
 return rows


def centre_helper():
 if sha(CENTRE_HELPER)!=CENTRE_HELPER_SHA:raise ValueError('Frozen Root centre evidence helper changed')
 spec=importlib.util.spec_from_file_location('frozen_deterministic_centre_owner',CENTRE_HELPER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def validate_prior(plan):
 """Two explicit parts: preserved failed prefix, then new actual centre6 success."""
 base.verify_sources(plan);helper=centre_helper()
 cfg=helper.load_config(checked_pointer(plan['centre_config']),CENTRE_CONFIG_SHA)
 closure=helper.validate_previous(cfg) # verifies exact old failure + all40/source/archive/compiled data
 previous=read(checked_pointer(plan['previous_root_exit']))
 if(type(previous.get('actual_process_returncode'))is not int or previous['actual_process_returncode']!=1 or previous.get('actual_wait_completed')is not True or previous.get('all40_capture_completed')is not True or previous.get('all520_compiled')is not True or previous.get('fit_optimizer_steps')!=0):raise ValueError('Explicit Root23027 actual1/wait and compiled40/520 zero-fit failure required')
 oldroot=Path(plan['prior_continuation_root']);prefix=list(base.STAGES[:6])
 oldrows=[json.loads(x)for x in checked_pointer(previous['stage_ledger']).read_text().splitlines()if x]
 oldowners=[read(oldroot/'owners'/name/'actual_exit.json')for name in prefix]
 oldrequests=[read(oldroot/'owners'/name/'request.json')for name in prefix]
 oldconfig=base.load_config(checked_pointer(plan['prior_config']))
 oldsources=base.verify_sources(oldconfig)
 validate_prior_stage_rows(oldrows,prefix,owner_rows=oldowners,request_rows=oldrequests,expected_sources=oldsources)
 failed=read(checked_pointer(previous['failed_child']))
 if(type(failed.get('actual_process_returncode'))is not int or failed['actual_process_returncode']!=1 or failed.get('stage')!='centre_GPU_preflight'or failed.get('actual_wait_completed')is not True or failed.get('source_unchanged')is not True or failed.get('unreaped_descendants',[])!=[]):raise ValueError('Preserved actually reaped failed preflight required; not counted as successful')
 root=Path(plan['centre_continuation_root'])
 if(root/'failure.json').exists():raise ValueError('New centre continuation failed')
 completion=read(root/'completion.json');terminal=read(root/'terminal_status.json')
 for value in(completion,terminal):
  if(value.get('complete')is not True or value.get('completed_stages')!=list(CENTRE_STAGES)or value.get('previous_root_exit')!=plan['previous_root_exit']or value.get('current_network_end_to_end_ready')is not False or value.get('Native_VAL_TEST_started')is not False):raise ValueError('Complete independent new centre6 child chain required')
 if {k:v for k,v in completion.items()if k!='finished_utc'}!={k:v for k,v in terminal.items()if k!='finished_utc'}:raise ValueError('Centre completion/terminal disagree')
 request=read(root/'request.json')
 if request.get('config')!=plan['centre_config']or request.get('source_map')!=plan['centre_method_sources']:raise ValueError('Actual frozen new centre owner request required')
 sources=base.verify_sources(cfg);rows=[json.loads(x)for x in(root/'stage_ledger.jsonl').read_text().splitlines()if x]
 owners=[read(root/'owners'/name/'actual_exit.json')for name in CENTRE_STAGES];requests=[read(root/'owners'/name/'request.json')for name in CENTRE_STAGES]
 validate_prior_stage_rows(rows,CENTRE_STAGES,owner_rows=owners,request_rows=requests,expected_sources=sources)
 manifest=checked_pointer(completion['compiled_manifest'])
 if pointer(manifest)!=closure['manifest']or completion['compiled_manifest']!=cfg['manifest']:raise ValueError('Same40 captured compiled cohort required; no recapture/replacement')
 admission,cfplan=base.validate_CF_admission(oldconfig,oldconfig['CF_final_admission']);cohort=Path(plan['CF_output_root']);native_owners=[];native_results=[]
 for parent in cfplan['parents']:
  ident=parent['parent_id'];op=cohort/(ident+'.owner.actual_exit.json');rp=cohort/ident/'result.json';o=exact_zero_wait(read(op));r=read(rp)
  if o.get('parent_id')!=ident or r.get('native_closed')is not True or r.get('source_unchanged')is not True:raise ValueError('Exact40 closed Native owner/result identities required')
  native_owners.append(pointer(op));native_results.append(pointer(rp))
 if len(native_owners)!=40:raise ValueError('Whole40 Native cohort required')
 status=read(root/'artifacts/centre_fit/fit_status.json')
 if status.get('training_complete')is not True or status.get('epochs_completed')!=40 or status.get('actual_optimizer_steps')!=520 or status.get('selected_checkpoint')!=completion.get('selected_centre'):raise ValueError('Actual newcentre40/520 selected status required')
 checked_pointer(completion['selected_centre']);exact_zero_wait(read(checked_pointer(plan['e1_actual_exit'])))
 facts=dict(prior_completion=pointer(root/'completion.json'),prior_terminal=pointer(root/'terminal_status.json'),prior_ledger=pointer(root/'stage_ledger.jsonl'),actual_stage_owners=[pointer(oldroot/'owners'/n/'actual_exit.json')for n in prefix]+[pointer(root/'owners'/n/'actual_exit.json')for n in CENTRE_STAGES],CF_capture_plan=admission['capture_plan'],CF_native_owners=native_owners,CF_native_results=native_results,CF_manifest=pointer(manifest),prior_training_owners=[plan['e1_actual_exit'],previous['failed_child'],pointer(root/'owners/centre_fit40/actual_exit.json')],preserved_previous_Root_actual1_wait=plan['previous_root_exit'],preserved_failed_child_not_success=previous['failed_child'],old_six_success_ledger=previous['stage_ledger'],old_six_success_stages=prefix,new_six_success_stages=list(CENTRE_STAGES),new_centre_selected=completion['selected_centre'],successor_Root6691_parent_actual_wait_claimed=False,metadata_only_no_NPZ_or_GT_decode=True)
 base.verify_sources(plan);return facts


def build_authorization(plan,facts):
 v2=read(checked_pointer(plan['v2_method_sources']))
 return dict(schema='semantic_e207M_GPU_authorization_20261104_v2',execution_authorized=True,
     method='current_network',fitted_component='semantic_task_e8_only',split_manifest=plan['split_manifest'],
     config=plan['v2_config'],source_sha256=v2,preparation_template_only=False,
     native_lifecycle=dict(all_native_closed=True,all_native_children_reaped=True,owners=facts['CF_native_owners'],results=facts['CF_native_results'],proof_scope='Exact40 frozen CFcap12 v3 parents; each actual child wait plus Native-closed result/source/archive'),
     prior_training_lifecycle=dict(all_training_children_reaped=True,owners=facts['prior_training_owners']),
     prior_continuation_evidence=facts,successor_Root6691_parent_actual_wait_claimed=False,previous_Root23027_actual_wait_is_explicit_pinned_evidence=True,
     authorization_basis='Root explicitly executed externally hashed deferred plan; exact6014 oldRootactual1 plus six preserved successes and six new centre successes; no inferred successor Root parent wait',
     deferred_Root_plan=dict(path=plan['_config_path'],sha256=plan['_config_sha256']),
     outer_resource_lock_policy=resource_policy(),
     GPU_release_policy='Six oldprefix and six newcentre children actually waited0; old failed preflight1 and Root23027actual1 preserved, CF40 Native closed and newcentre fit actual0; this deferred v2 owner acquires correct runtime/framework dual locks around each GPU stage, then invokes unchanged e2 CLI with --pipeline-owned-locks. Per-stage actual ownership receipts bind held locks through child wait/reap.',
     current_network_end_to_end_ready=False,all_task_Native_VAL_TEST_authorized=False,created_utc=now())


verify_sources=base.verify_sources
descendants=base.descendants
reap_adopted=base.reap_adopted
signal_known=base.signal_known

def run_gpu_stage(config,name,args,resource_receipt):
 if (_ACTIVE_RESOURCE_STATE is None or _ACTIVE_RESOURCE_STATE['stage']!=name or _ACTIVE_RESOURCE_STATE['receipt']is not resource_receipt):raise ValueError('Live exact outer resource ownership token required before GPU child')
 for path,stream,identity in zip(resource_paths(),_ACTIVE_RESOURCE_STATE['handles'],resource_receipt['locks']):
  current=os.stat(path);actual=os.fstat(stream.fileno())
  if (current.st_dev,current.st_ino)!=(actual.st_dev,actual.st_ino)or(actual.st_dev,actual.st_ino)!=(identity['device'],identity['inode']):raise ValueError('Correct resource lock inode changed')
 """No retries. The artifact belongs to the child; only owner/log dir is precreated."""
 native=False;cpu=False
 assert_running(config);sources=verify_sources(config)
 out=Path(config['output_root']);owner=out/'owners'/name;owner.mkdir(parents=True,exist_ok=False)
 command=[config['python'],'-B']+list(args)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 env['CUDA_VISIBLE_DEVICES']=''if cpu or native else '0'
 request=dict(stage=name,argv=command,started_utc=now(),source_sha256=sources,
              driver_holds_resource_locks=True,native_host_owns_its_locks=False,child_pipeline_owned_locks=True,resource_lock_ownership=resource_receipt)
 write_new(owner/'request.json',request)
 process=None;code=None;reason=None;error=None;tracked={};reaped=[];begin=time.monotonic();stop_at=term_at=kill_at=None
 try:
  with (owner/'stdout.log').open('x')as log:
   process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   write_new(owner/'child.json',dict(pid=process.pid,process_group=process.pid))
   while process.poll()is None:
    tracked.update(descendants());elapsed=time.monotonic()-begin
    if reason is None:
     try:assert_running(config)
     except InterruptedError:reason='STOP_or_owner_signal'
     if elapsed>config['stage_timeout_seconds'][name]:reason='stage_timeout'
     if reason:
      stop_at=time.monotonic();write_new(owner/'STOP',dict(reason=reason))
      cohort=Path(config['CF_output_root'])
      if native and cohort.is_dir()and not(cohort/'STOP').exists():(cohort/'STOP').write_text(reason+'\n')
    if stop_at is not None and term_at is None and time.monotonic()-stop_at>=config['stop_grace_seconds']:
     try:os.killpg(process.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     signal_known(tracked,signal.SIGTERM);term_at=time.monotonic()
    if term_at is not None and kill_at is None and time.monotonic()-term_at>=10:
     signal_known(tracked,signal.SIGKILL)
     try:os.killpg(process.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     kill_at=time.monotonic()
    time.sleep(min(config['poll_seconds'],.5))
   tracked.update(descendants());code=process.wait()
  if reason:raise RuntimeError('Stage stopped: '+reason)
  if code!=0:raise RuntimeError('Stage actual nonzero exit; no retry: '+name)
  alive,extra=reap_adopted(tracked,process.pid);reaped.extend(extra)
  if alive:raise RuntimeError('Stage returned with live descendants; stop/cleanup required')
  verify_sources(config)
 except BaseException as exception:
  error=dict(type=type(exception).__name__,message=str(exception));raise
 finally:
  tracked.update(descendants())
  if process is not None and process.poll()is None:
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   signal_known(tracked,signal.SIGTERM)
   try:code=process.wait(timeout=10)
   except subprocess.TimeoutExpired:
    signal_known(tracked,signal.SIGKILL)
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    code=process.wait()
  if process is not None:
   alive,extra=reap_adopted(tracked,process.pid);reaped.extend(extra)
   if alive:
    signal_known(alive,signal.SIGTERM);deadline=time.monotonic()+10
    while alive and time.monotonic()<deadline:
     time.sleep(.05);alive,extra=reap_adopted(alive,process.pid);reaped.extend(extra)
    if alive:
     signal_known(alive,signal.SIGKILL);deadline=time.monotonic()+5
     while alive and time.monotonic()<deadline:
      time.sleep(.05);alive,extra=reap_adopted(alive,process.pid);reaped.extend(extra)
   else:alive={}
  else:alive={}
  unchanged=False
  try:verify_sources(config);unchanged=True
  except (ValueError,OSError):pass
  terminal=dict(stage=name,started_utc=request['started_utc'],finished_utc=now(),
      actual_process_returncode=code,actual_wait_completed=process is not None and process.returncode is not None,
      correct_outer_resource_locks_held_through_actual_wait=True,resource_lock_ownership=resource_receipt,
      source_unchanged=unchanged,stop_reason=reason,error=error,
      adopted_descendants_reaped=reaped,unreaped_descendants=list(alive),seconds=time.monotonic()-begin)
  write_new(owner/'actual_exit.json',terminal)
 return terminal

def execute(path,expected_sha):
 plan=load_plan(path,expected_sha);out=Path(plan['output_root'])
 with base.driver_lock(LOCK):
  if out.exists():raise FileExistsError('Fresh deferred owner only, no overwrite/resumption/retry')
  assert_running(plan);out.mkdir(parents=True,exist_ok=False);(out/'owners').mkdir();(out/'artifacts').mkdir()
  base.enable_subreaper();write_new(out/'request.json',dict(plan=pointer(path),source_map=plan['method_sources'],started_utc=now(),successor_Root6691_parent_wait_not_claimed=True,component_only=True,outer_resource_policy=resource_policy()))
  previous={}
  for signum in(signal.SIGINT,signal.SIGTERM):previous[signum]=signal.signal(signum,lambda number,frame:base.SIGNAL_REQUEST.update(signal=number))
  status=dict(complete=False,completed_stages=[],component_only=True,current_network_end_to_end_ready=False,Native_VAL_TEST_started=False,model_or_runtime_qualified=False)
  try:
   wait_for_prior(plan);facts=validate_prior(plan);write_new(out/'prior_chain_validation.json',facts)
   auth=out/'Root_GPU_authorization.json';write_new(auth,build_authorization(plan,facts));auth_pointer=pointer(auth)
   common=['--manifest',plan['split_manifest']['path'],'--data-root',str(Path(plan['preprocess_completion']['path']).parent),'--root-authorization',str(auth),'--root-authorization-sha256',auth_pointer['sha256']]
   trainer='scripts/train_semantic_task_e_large_20261104_v2.py';artifact=out/'artifacts'
   def run(name,args):
    assert_running(plan);base.verify_sources(plan)
    # Validate previously sealed proof pointers rather than invent a parent exit.
    for item in(facts['prior_completion'],facts['prior_terminal'],facts['prior_ledger'],*facts['actual_stage_owners'],*facts['CF_native_owners'],*facts['CF_native_results'],facts['CF_manifest'],*facts['prior_training_owners'],facts['preserved_previous_Root_actual1_wait'],facts['old_six_success_ledger']):checked_pointer(item)
    checked_pointer(auth_pointer)
    if '--pipeline-owned-locks'in args:raise ValueError('Owner inserts the sole explicit child bypass flag')
    with correct_stage_resource_locks(plan,name)as resource_receipt:
     value=run_gpu_stage(plan,name,list(args)+['--pipeline-owned-locks'],resource_receipt)
    exact_zero_wait(value,stage=name);status['completed_stages'].append(name);append(out/'stage_ledger.jsonl',value)
   preflight=artifact/'GPU_preflight'
   run(STAGES[0],[trainer,'--gpu-preflight']+common+['--microbatch','4','--output',str(preflight)])
   p=read(preflight/'preflight.json')
   if (p.get('complete')is not True or p.get('passed')is not True or p.get('source_unchanged')is not True
       or p.get('optimizer_steps')!=1 or p.get('optimizer_step_is_ephemeral_and_discarded')is not True
       or p.get('preflight_weights_reused_for_training')is not False or p.get('actual_valid_frames')!=20):raise ValueError('Actual B4 full20-frame discarded-step v2 GPU preflight required')
   fit=artifact/'fit10'
   run(STAGES[1],[trainer,'--train']+common+['--preflight-report',str(preflight/'preflight.json'),'--preflight-sha256',sha(preflight/'preflight.json'),'--output',str(fit)])
   completion=read(fit/'completion.json')
   if (completion.get('complete')is not True or completion.get('source_unchanged')is not True
       or completion.get('fitted_epochs')!=10 or completion.get('optimizer_steps')!=1930
       or completion.get('source_sha256')!=read(checked_pointer(plan['v2_method_sources']))):raise ValueError('Actual complete fixed1930-update source-bound v2 fit required')
   selected=checked_pointer(completion['selected_model'])
   if selected!=fit/'selected.pt':raise ValueError('Actual selected fit sibling required')
   evaluation=artifact/'selected646_evaluation'
   run(STAGES[2],['scripts/evaluate_semantic_task_e_large_20261104_v2.py','--evaluate']+common+['--checkpoint',str(selected),'--checkpoint-sha256',sha(selected),'--output',str(evaluation)])
   evaluated=read(evaluation/'completion.json')
   if evaluated.get('complete')is not True or evaluated.get('source_unchanged')is not True or evaluated.get('selected_model')!=completion['selected_model']:raise ValueError('Actual selected646 v2 evaluation completion required')
   metrics=read(checked_pointer(evaluated['metrics']))
   if metrics.get('windows')!=646 or len(metrics.get('per_task',{}))!=16:raise ValueError('All646 windows/16 task versions required for selected component evaluation')
   status.update(complete=True,selected_model=pointer(selected),Root_GPU_authorization=auth_pointer,prior_chain_validation=pointer(out/'prior_chain_validation.json'),selection_evaluation=pointer(evaluation/'completion.json'))
   write_new(out/'completion.json',status);return status
  except BaseException as error:
   status.update(error_type=type(error).__name__,error=str(error));write_new(out/'failure.json',status);raise
  finally:
   write_new(out/'terminal_status.json',dict(status,finished_utc=now()))
   for signum,handler in previous.items():signal.signal(signum,handler)


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN);parser.add_argument('--plan-sha256');parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
 if args.execute:
  if not args.plan_sha256:parser.error('--execute requires Root external --plan-sha256')
  value=execute(args.plan,args.plan_sha256)
 else:
  plan=load_plan(args.plan,args.plan_sha256)
  value=dict(complete=True,preparation_only=True,plan=pointer(args.plan),stage_order=list(STAGES),waiting_for=str(Path(plan['centre_continuation_root'])/'completion.json'),prior_finished=Path(plan['centre_continuation_root'],'completion.json').exists(),child_processes_started=0,model_forward_calls=0,Native_started=False,GPU_used=False,all_task_Native_VAL_TEST_authorized=False,successor_Root6691_actual_wait_not_claimed=True,previous_Root23027_actual_wait_has_pinned_receipt=True,resource_policy=resource_policy(),resource_locks_acquired=False)
 print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
