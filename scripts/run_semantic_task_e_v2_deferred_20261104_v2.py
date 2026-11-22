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
SCHEMA='semantic_task_e207M_v2_deferred_owner_20261104_v2'
DEFAULT_PLAN=ROOT/'config/framework/semantic_task_e_v2_deferred_20261104_v2/plan.json'
LOCK=ROOT/'runtime/framework/semantic_task_e_v2_deferred_20261104_v2.lock'
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
 return dict(owner='this_deferred_v2_parent',paths=[str(p)for p in resource_paths()],order=list(RESOURCE_NAMES),child_CLI_bypass_argument='--pipeline-owned-locks',held_while_waiting_for_prior_chain=False,held_from_stage_launch_through_actual_wait_and_descendant_cleanup=True)

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
 for key in ('prior_config','v2_delivery','v2_method_sources','v2_config','split_manifest','preprocess_completion','shared_registry','e1_actual_exit','method_sources'):
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
 exact_zero_wait(read(checked_pointer(plan['e1_actual_exit'])))
 plan['_config_path']=str(path);plan['_config_sha256']=sha(path)
 base.verify_sources(plan)
 return plan


def assert_running(plan):
 base.assert_running(plan)
 if (DEFAULT_PLAN.parent/'STOP').exists():raise InterruptedError('Deferred owner STOP')


def wait_for_prior(plan):
 root=Path(plan['prior_continuation_root']);begin=time.monotonic()
 while True:
  assert_running(plan)
  if (root/'failure.json').exists():raise ValueError('Prior continuation failed; no deferred authorization or retry')
  if (root/'completion.json').is_file()and(root/'terminal_status.json').is_file():return
  if time.monotonic()-begin>plan['wait_timeout_seconds']:raise TimeoutError('Prior complete12-stage owner evidence did not arrive')
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


def validate_prior(plan):
 """After whole-chain completion only; no NPZ pixels, labels, or models are read."""
 base.verify_sources(plan)
 root=Path(plan['prior_continuation_root'])
 if (root/'failure.json').exists():raise ValueError('Prior failed owner cannot authorize e2')
 complete=read(root/'completion.json');terminal=read(root/'terminal_status.json')
 for value in (complete,terminal):
  if (value.get('complete')is not True or value.get('completed_stages')!=list(base.STAGES)
      or value.get('current_network_end_to_end_ready')is not False
      or value.get('model_or_runtime_qualified')is not False or value.get('Native_VAL_TEST_started')is not False):raise ValueError('Actual complete component-only main continuation required')
 if {k:v for k,v in terminal.items()if k!='finished_utc'}!=complete:raise ValueError('Main terminal and completion records disagree')
 request=read(root/'request.json')
 if request.get('config')!=plan['prior_config']:raise ValueError('Externally frozen main owner config required')
 old=base.load_config(checked_pointer(plan['prior_config']),plan['prior_config']['sha256'])
 if request.get('source_map')!=old['method_sources']:raise ValueError('Main owner frozen source map required')
 expected_sources=base.verify_sources(old)
 rows=[json.loads(line)for line in(root/'stage_ledger.jsonl').read_text().splitlines()if line]
 owners=[read(root/'owners'/name/'actual_exit.json')for name in base.STAGES]
 requests=[read(root/'owners'/name/'request.json')for name in base.STAGES]
 validate_prior_stage_rows(rows,base.STAGES,owner_rows=owners,request_rows=requests,expected_sources=expected_sources)
 # The actual completed source-bound CF admission is preserved in completion.
 admission,cfplan=base.validate_CF_admission(old,old['CF_final_admission'])
 if complete.get('CF_admission')!=admission:raise ValueError('Completed main CF admission changed')
 base.queue_audit(old,admission,cfplan,count=40)
 cohort=Path(plan['CF_output_root']);native_owners=[];native_results=[]
 for parent in cfplan['parents']:
  ident=parent['parent_id'];owner_path=cohort/(ident+'.owner.actual_exit.json');result_path=cohort/ident/'result.json'
  owner=exact_zero_wait(read(owner_path));result=read(result_path)
  if (owner.get('parent_id')!=ident or Path(owner.get('output','')).resolve()!=cohort/ident
      or result.get('parent_id')!=ident or result.get('native_closed')is not True
      or result.get('source_unchanged')is not True or result.get('complete')is not True):raise ValueError('Every exact CF40 parent must have closed Native/source result and actual wait')
  native_owners.append(pointer(owner_path));native_results.append(pointer(result_path))
 compiled=checked_pointer(complete['compiled_manifest']);manifest=read(compiled)
 if (manifest.get('capture_plan')!=admission['capture_plan']or manifest.get('complete')is not True
     or any(manifest.get(k)is not True for k in ('all_native_closed','all_process_exits_observed','source_unchanged'))
     or [p['parent_id']for p in manifest.get('parents',[])]!=[p['parent_id']for p in cfplan['parents']]):raise ValueError('Exact whole40 compiled manifest/source identity required')
 for parent,owner,result in zip(manifest['parents'],native_owners,native_results):
  if parent['capture']['owner_actual_exit']!=owner or parent['capture']['result']!=result:raise ValueError('Compiled dataset must bind those same actual parents')
 e1=read(checked_pointer(plan['e1_actual_exit']));exact_zero_wait(e1)
 if complete.get('e_actual_exit')!=plan['e1_actual_exit']:raise ValueError('Main continuation must bind actual e1 Root wait receipt')
 centre_exit=root/'owners/centre_fit40/actual_exit.json';exact_zero_wait(read(centre_exit),stage='centre_fit40')
 centre_status=read(root/'artifacts/centre_fit/fit_status.json')
 if centre_status.get('training_complete')is not True or centre_status.get('epochs_completed')!=40 or centre_status.get('actual_optimizer_steps')!=520 or centre_status.get('selected_checkpoint')!=complete.get('selected_centre'):raise ValueError('Actual complete centre40/520 fit required')
 checked_pointer(complete['selected_e']);checked_pointer(complete['selected_centre'])
 facts=dict(prior_completion=pointer(root/'completion.json'),prior_terminal=pointer(root/'terminal_status.json'),prior_ledger=pointer(root/'stage_ledger.jsonl'),actual_stage_owners=[pointer(root/'owners'/name/'actual_exit.json')for name in base.STAGES],CF_capture_plan=admission['capture_plan'],CF_native_owners=native_owners,CF_native_results=native_results,CF_manifest=pointer(compiled),prior_training_owners=[plan['e1_actual_exit'],pointer(centre_exit)],Root_parent_process_actual_wait_claimed=False,Root_parent_process='23027 not asserted; proof consists only of its twelve actually waited child processes',metadata_only_no_NPZ_or_GT_decode=True)
 base.verify_sources(plan)
 return facts


def build_authorization(plan,facts):
 v2=read(checked_pointer(plan['v2_method_sources']))
 return dict(schema='semantic_e207M_GPU_authorization_20261104_v2',execution_authorized=True,
     method='current_network',fitted_component='semantic_task_e8_only',split_manifest=plan['split_manifest'],
     config=plan['v2_config'],source_sha256=v2,preparation_template_only=False,
     native_lifecycle=dict(all_native_closed=True,all_native_children_reaped=True,owners=facts['CF_native_owners'],results=facts['CF_native_results'],proof_scope='Exact40 frozen CFcap12 v3 parents; each actual child wait plus Native-closed result/source/archive'),
     prior_training_lifecycle=dict(all_training_children_reaped=True,owners=facts['prior_training_owners']),
     prior_continuation_evidence=facts,Root_parent_process_actual_wait_claimed=False,
     authorization_basis='Root explicitly executed externally hashed deferred plan; no inferred Root parent wait',
     deferred_Root_plan=dict(path=plan['_config_path'],sha256=plan['_config_sha256']),
     outer_resource_lock_policy=resource_policy(),
     GPU_release_policy='All12 prior children actually waited0, CF40 Native closed and centre fit actual0; this deferred v2 owner acquires correct runtime/framework dual locks around each GPU stage, then invokes unchanged e2 CLI with --pipeline-owned-locks. Per-stage actual ownership receipts bind held locks through child wait/reap.',
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
  base.enable_subreaper();write_new(out/'request.json',dict(plan=pointer(path),source_map=plan['method_sources'],started_utc=now(),Root_parent_wait_not_claimed=True,component_only=True,outer_resource_policy=resource_policy()))
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
    for item in(facts['prior_completion'],facts['prior_terminal'],facts['prior_ledger'],*facts['actual_stage_owners'],*facts['CF_native_owners'],*facts['CF_native_results'],facts['CF_manifest'],*facts['prior_training_owners']):checked_pointer(item)
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
  value=dict(complete=True,preparation_only=True,plan=pointer(args.plan),stage_order=list(STAGES),waiting_for=str(Path(plan['prior_continuation_root'])/'completion.json'),prior_finished=Path(plan['prior_continuation_root'],'completion.json').exists(),child_processes_started=0,model_forward_calls=0,Native_started=False,GPU_used=False,all_task_Native_VAL_TEST_authorized=False,Root_parent_actual_wait_not_claimed=True,resource_policy=resource_policy(),resource_locks_acquired=False)
 print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
