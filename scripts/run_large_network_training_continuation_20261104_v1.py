"""One fresh serial owner for semantic-e and CF centre pilot continuation.

Default validates source/metadata only: no child, model, accelerator or Native.
Root alone supplies --execute plus the externally frozen config SHA. This owner
never edits plans, grants visual QA admission, retries a failed stage, or starts
all-task Native VAL/TEST. Existing CLIs/shells own their own resource locks.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='large_network_training_continuation_20261104_v1'
DEFAULT_CONFIG=ROOT/'config/framework/large_network_training_continuation_20261104_v1/plan.json'
CF_ADMISSION_SCHEMA=SCHEMA+'_CF_admission'
QA_ADMISSION_SCHEMA=SCHEMA+'_QA_admission'
DRIVER_LOCK=ROOT/'runtime/framework/large_network_training_continuation_20261104_v1.lock'
STAGES=('semantic_selected_evaluation','CF_first_fit13','CF_first_fit_QA',
        'CF_remaining39','CF_all40_audit','CF_compile','centre_GPU_preflight',
        'centre_fit40','centre_selection_predict208','centre_selection_score208',
        'centre_reserved_predict104','centre_reserved_score104')
SIGNAL_REQUEST={'signal':None}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def pointer(path):
 path=Path(path).resolve(strict=True);return dict(path=str(path),sha256=sha(path))
def checked_pointer(value):
 if not isinstance(value,dict)or set(value)!={'path','sha256'}:raise ValueError('Exact path/SHA pointer required')
 path=Path(value['path']).resolve(strict=True)
 if sha(path)!=value['sha256']:raise ValueError('Bound file changed: '+str(path))
 return path
def write_new(path,value):
 with Path(path).open('x')as stream:
  json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def append(path,value):
 with Path(path).open('a')as stream:
  stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
def sources_from(value):
 result=value.get('source_sha256',value)
 if not isinstance(result,dict)or not result or any(not isinstance(k,str)or not isinstance(v,str)or len(v)!=64 for k,v in result.items()):raise ValueError('Explicit nonempty source map required')
 return result
def verify_sources(config):
 if '_config_path' in config and sha(config['_config_path'])!=config['_config_sha256']:raise ValueError('Continuation config changed during ownership')
 frozen=read(checked_pointer(config['method_sources']))['source_sha256']
 for name,digest in frozen.items():
  if sha(name)!=digest:raise ValueError('Continuation source changed: '+name)
 return frozen

def load_config(path,expected_sha=None):
 path=Path(path).resolve(strict=True)
 if expected_sha is not None and sha(path)!=expected_sha:raise ValueError('External exact continuation config SHA required')
 config=read(path)
 if config.get('schema')!=SCHEMA or config.get('allowed_scope')!='semantic_e_and_Cube_centre_pilot_only' or config.get('all_task_Native_VAL_TEST_authorized')is not False:raise ValueError('Component-only continuation protocol required')
 if tuple(config.get('stages',()))!=STAGES:raise ValueError('Fixed stage order required')
 for key in ('python','output_root','e_fit_root','e_actual_exit','CF_final_admission','QA_admission','CF_output_root'):
  if not isinstance(config.get(key),str)or not Path(config[key]).is_absolute():raise ValueError('Explicit absolute input/output paths required: '+key)
 out=Path(config['output_root'])
 if ROOT/'artifacts/framework/local_mpc_20261102'not in out.parents:raise ValueError('New independent artifact owner required')
 for key in ('e_delivery','e_authorization','split_manifest','preprocess_completion','CF_prepared_snapshot','CF_delivery','centre_delivery','encoder'):
  checked_pointer(config[key])
 if config['encoder']['sha256']!='dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa':raise ValueError('Original dbf encoder only')
 e=sources_from(read(checked_pointer(config['e_delivery'])))
 c=sources_from(read(checked_pointer(config['centre_delivery'])))
 if len(e)!=20 or len(c)!=21:raise ValueError('Frozen semantic20/centre21 maps required')
 prepared=read(checked_pointer(config['CF_prepared_snapshot']))
 delivery=read(checked_pointer(config['CF_delivery']))
 if (prepared.get('preparation_only')is not True or prepared.get('execution_authorized')is not False
     or len(prepared['source_sha256'])!=208 or len(prepared['external_source_sha256'])!=12
     or delivery.get('source_repo_count')!=208):raise ValueError('Original prepared CF208/12 identity required')
 if Path(config['CF_output_root']).resolve()!=ROOT/prepared['output_root']:raise ValueError('New all40 v3 cohort required; no mixing v2')
 verify_sources(config)
 for value in [config['wait_timeout_seconds'],config['poll_seconds'],config['stop_grace_seconds'],*config['stage_timeout_seconds'].values()]:
  if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:raise ValueError('Finite positive fixed time bounds required')
 if set(config['stage_timeout_seconds'])!=set(STAGES):raise ValueError('All stage timeouts required')
 config['_config_path']=str(path);config['_config_sha256']=sha(path)
 return config

def validate_e_exit(path):
 value=read(path)
 if type(value.get('actual_process_returncode'))is not int or value['actual_process_returncode']!=0 or value.get('actual_wait_completed')is not True:raise ValueError('Root actual exit0 AND actual completed wait required; PID disappearance is insufficient')
 return value

def validate_e_completion(config):
 directory=Path(config['e_fit_root']);path=directory/'completion.json';value=read(path)
 sources=sources_from(read(checked_pointer(config['e_delivery'])))
 if (value.get('complete')is not True or value.get('source_unchanged')is not True
     or value.get('stage')!='fit' or value.get('method')!='current_network'
     or value.get('fitted_epochs')!=5 or type(value.get('optimizer_steps'))is not int or value['optimizer_steps']<=0
     or value.get('source_sha256')!=sources or value.get('split_manifest')!=config['split_manifest']
     or value.get('current_network_end_to_end_ready')is not False
     or value.get('Native_started')is not False):raise ValueError('Complete source-bound semantic component fit required')
 selected=checked_pointer(value['selected_model'])
 if selected!=directory/'selected.pt':raise ValueError('Selected checkpoint must be the actual fit sibling')
 checked_pointer(value['history'])
 return value

def validate_CF_admission(config,path):
 admission=read(path)
 if admission.get('schema')!=CF_ADMISSION_SCHEMA or admission.get('execution_authorized')is not True:raise ValueError('Root final CF admission required')
 final_path=checked_pointer(admission['capture_plan']);final=read(final_path)
 prepared=read(checked_pointer(config['CF_prepared_snapshot']))
 allowed=('preparation_only','execution_authorized','capture_ready','required_pending')
 if {k:v for k,v in final.items()if k not in allowed}!={k:v for k,v in prepared.items()if k not in allowed}:raise ValueError('Only Root execution flags/pending declaration may differ from prepared plan')
 if final.get('preparation_only')is not False or final.get('execution_authorized')is not True or final.get('capture_ready')is not True:raise ValueError('Final CF plan must be explicitly admitted')
 if admission.get('prepared_plan')!=config['CF_prepared_snapshot']or admission.get('output_root')!=config['CF_output_root']:raise ValueError('Prepared snapshot and fresh v3 output identity required')
 if ROOT/'config/framework'not in final_path.parents:raise ValueError('Versioned Root plan required')
 return admission,final

def validate_QA_admission(config,path,qa_output,final_pointer):
 value=read(path);qa_output=Path(qa_output)
 if (value.get('schema')!=QA_ADMISSION_SCHEMA or value.get('approved')is not True
     or value.get('visually_inspected')is not True or value.get('capture_plan')!=final_pointer
     or checked_pointer(value['report'])!=qa_output/'report.json'
     or checked_pointer(value['figure'])!=qa_output/'RGBD_owner_dual_crop_QA.png'):raise ValueError('Root visual QA admission must bind exact report/PNG/final plan')
 report=read(qa_output/'report.json')
 prepared=read(checked_pointer(config['CF_prepared_snapshot']))
 if (report.get('complete')is not True or report.get('role')!='fit'
     or report.get('parent_id')!=prepared['parents'][0]['parent_id']or len(report.get('frames',[]))!=13):raise ValueError('Only exact first fixed fit13 may be visually admitted')
 return value

def stop_paths(config):
 out=Path(config['output_root'])
 return (out/'STOP',out.parent/'STOP',DEFAULT_CONFIG.parent/'STOP',ROOT/'runtime/framework/STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP',ROOT/'runtime/framework/near_goal_perception_20261103/STOP')
def assert_running(config):
 if SIGNAL_REQUEST['signal']is not None or any(p.exists()for p in stop_paths(config)):raise InterruptedError('Continuation STOP/signal; no further stages')
def wait_file(config,path):
 begin=time.monotonic();path=Path(path)
 while not path.is_file():
  assert_running(config);verify_sources(config)
  if time.monotonic()-begin>config['wait_timeout_seconds']:raise TimeoutError('Waiting for explicit Root receipt/admission: '+str(path))
  time.sleep(config['poll_seconds'])
 assert_running(config);return path

@contextmanager
def driver_lock(path=DRIVER_LOCK):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+')as stream:
  fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:fcntl.flock(stream,fcntl.LOCK_UN)

def enable_subreaper():
 # Descendants orphaned during abort remain this owner’s actual wait children.
 import ctypes
 if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise RuntimeError('Linux child-subreaper setup failed')
def process_identity(pid):
 try:
  text=(Path('/proc')/str(pid)/'stat').read_text();parts=text[text.rfind(')')+2:].split()
  return int(parts[1]),int(parts[19]) # ppid and kernel start ticks
 except (FileNotFoundError,ProcessLookupError):return None

def descendants():
 rows={}
 for name in Path('/proc').iterdir():
  if name.name.isdigit():
   value=process_identity(int(name.name))
   if value is not None:rows[int(name.name)]=value
 result={};parents={os.getpid()}
 while True:
  found={pid:v for pid,v in rows.items()if v[0]in parents and pid not in result}
  if not found:return result
  result.update(found);parents.update(found)

def signal_known(tracked,signum):
 for pid,(_,started)in tracked.items():
  current=process_identity(pid)
  if current is not None and current[1]==started:
   try:os.kill(pid,signum)
   except ProcessLookupError:pass

def reap_adopted(tracked,direct_pid):
 reaped=[];alive={}
 for pid,value in tracked.items():
  if pid==direct_pid:continue
  current=process_identity(pid)
  if current is None or current[1]!=value[1]:continue
  try:
   waited,status=os.waitpid(pid,os.WNOHANG)
   if waited:reaped.append(dict(pid=pid,wait_status=status));continue
  except ChildProcessError:pass
  alive[pid]=value
 return alive,reaped

def run_stage(config,name,args,*,native=False,cpu=False):
 """No retries. The artifact belongs to the child; only owner/log dir is precreated."""
 assert_running(config);sources=verify_sources(config)
 out=Path(config['output_root']);owner=out/'owners'/name;owner.mkdir(parents=True,exist_ok=False)
 command=[config['python'],'-B']+list(args)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 env['CUDA_VISIBLE_DEVICES']=''if cpu or native else '0'
 request=dict(stage=name,argv=command,started_utc=now(),source_sha256=sources,
              driver_holds_resource_locks=False,native_host_owns_its_locks=native)
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
      source_unchanged=unchanged,stop_reason=reason,error=error,
      adopted_descendants_reaped=reaped,unreaped_descendants=list(alive),seconds=time.monotonic()-begin)
  write_new(owner/'actual_exit.json',terminal)
 return terminal

def require_complete(path,**fields):
 value=read(path)
 if any(value.get(k)!=v or type(value.get(k))is not type(v)for k,v in fields.items()):raise ValueError('Actual output gate failed: '+str(path))
 return value

def queue_audit(config,admission,plan,*,count):
 path=ROOT/'scripts/run_current_cube_dual_crop_capture_queue_20261104_v1.py'
 spec=importlib.util.spec_from_file_location('continuation_frozen_CF_queue',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 root=Path(config['CF_output_root']);digest=admission['capture_plan']['sha256']
 rows=[module.audit_closed(root,p,plan,digest)for p in plan['parents'][:count]]
 if count==1 and any((root/p['parent_id']).exists()for p in plan['parents'][1:]):raise ValueError('Preview must not start any other parent')
 return rows

def execute(config_path,config_sha):
 config=load_config(config_path,config_sha);out=Path(config['output_root'])
 with driver_lock():
  if out.exists():raise FileExistsError('Fresh continuation only; stages are never resumed/retried')
  assert_running(config);out.mkdir(parents=True,exist_ok=False);(out/'artifacts').mkdir();(out/'owners').mkdir()
  enable_subreaper()
  write_new(out/'request.json',dict(config=pointer(config_path),source_map=config['method_sources'],started_utc=now(),component_only=True,all_task_Native_VAL_TEST_authorized=False))
  previous={}
  for signum in (signal.SIGINT,signal.SIGTERM):
   previous[signum]=signal.signal(signum,lambda number,frame:SIGNAL_REQUEST.update(signal=number))
  status=dict(complete=False,completed_stages=[],current_network_end_to_end_ready=False,model_or_runtime_qualified=False,Native_VAL_TEST_started=False)
  def run(name,args,**kwargs):
   validate_e_exit(config['e_actual_exit']);verify_sources(config)
   if pointer(config['e_actual_exit'])!=status['e_actual_exit'] or pointer(Path(config['e_fit_root'])/'completion.json')!=status['e_completion']:raise ValueError('Completed semantic fit evidence changed')
   if 'QA_admission'in status:
    current=validate_QA_admission(config,config['QA_admission'],artifact/'CF_first_fit_QA',status['CF_admission']['capture_plan'])
    if current!=status['QA_admission']:raise ValueError('Root QA admission changed')
   if 'CF_admission'in status:
    live,_=validate_CF_admission(config,config['CF_final_admission'])
    if live!=status['CF_admission']:raise ValueError('Root CF admission changed during continuation')
   value=run_stage(config,name,args,**kwargs)
   if type(value.get('actual_process_returncode'))is not int or value['actual_process_returncode']!=0 or value.get('actual_wait_completed')is not True or value.get('source_unchanged')is not True or value.get('unreaped_descendants'):raise ValueError('Actual clean wait/source terminal required')
   status['completed_stages'].append(name)
   append(out/'stage_ledger.jsonl',value);return value
  artifact=out/'artifacts'
  try:
   exit_path=wait_file(config,config['e_actual_exit']);validate_e_exit(exit_path)
   status['e_actual_exit']=pointer(exit_path);e=validate_e_completion(config);status['e_completion']=pointer(Path(config['e_fit_root'])/'completion.json')
   semantic=artifact/'semantic_selected_evaluation'
   run(STAGES[0],['scripts/evaluate_semantic_task_e_large_20261104_v1.py','--evaluate','--manifest',config['split_manifest']['path'],'--data-root',str(Path(config['preprocess_completion']['path']).parent),'--checkpoint',e['selected_model']['path'],'--checkpoint-sha256',e['selected_model']['sha256'],'--root-authorization',config['e_authorization']['path'],'--root-authorization-sha256',config['e_authorization']['sha256'],'--output',str(semantic)])
   completed=require_complete(semantic/'completion.json',complete=True,source_unchanged=True,stage='selection_evaluation')
   if completed['selected_model']!=e['selected_model']:raise ValueError('Selected-e evaluation checkpoint mismatch')
   admission_path=wait_file(config,config['CF_final_admission']);admission,plan=validate_CF_admission(config,admission_path);status['CF_admission']=admission
   if Path(config['CF_output_root']).exists():raise FileExistsError('Fresh all40 v3 capture root required')
   common=['--plan',admission['capture_plan']['path'],'--plan-sha256',admission['capture_plan']['sha256'],'--output-root',config['CF_output_root'],'--execute']
   run(STAGES[1],['scripts/run_current_cube_dual_crop_capture_queue_20261104_v1.py']+common,native=True);queue_audit(config,admission,plan,count=1)
   qa=artifact/'CF_first_fit_QA'
   run(STAGES[2],['scripts/audit_current_cube_dual_crop_first_fit_20261103_v1.py','--audit-closed-first-fit','--capture-plan',admission['capture_plan']['path'],'--capture-root',config['CF_output_root'],'--output',str(qa)],cpu=True)
   qa_path=wait_file(config,config['QA_admission']);qa_admission=validate_QA_admission(config,qa_path,qa,admission['capture_plan']);status['QA_admission']=qa_admission
   run(STAGES[3],['scripts/run_current_cube_dual_crop_capture_queue_20261104_v1.py']+common+['--start-index','1'],native=True);queue_audit(config,admission,plan,count=40)
   compiler='scripts/prepare_current_cube_dual_crop_TRAIN_manifest_20261104_v2.py';capture=['--capture-plan',admission['capture_plan']['path'],'--capture-root',config['CF_output_root']]
   run(STAGES[4],[compiler]+capture+['--audit-closed'],cpu=True)
   dataset=artifact/'CF_dataset'
   run(STAGES[5],[compiler]+capture+['--compile','--output',str(dataset)],cpu=True)
   receipt=require_complete(dataset/'receipt.json',complete=True,decoded_fit_samples=208,decoded_selection_samples=208,decoded_reserved_samples=0)
   manifest=checked_pointer(receipt['manifest']);encoder=config['encoder']
   trainer='scripts/train_current_cube_dual_crop_centre_large_20261104_v1.py';data=['--manifest',str(manifest),'--encoder-checkpoint',encoder['path'],'--expected-encoder-sha256',encoder['sha256']]
   preflight=artifact/'centre_GPU_preflight'
   run(STAGES[6],[trainer,'--gpu-preflight']+data+['--output',str(preflight),'--device','cuda:0'])
   require_complete(preflight/'gpu_preflight.json',passed=True,actual_optimizer_steps=1,model_saved=False,updated_weights_discarded=True)
   fit=artifact/'centre_fit'
   run(STAGES[7],[trainer,'--train']+data+['--preflight',str(preflight/'gpu_preflight.json'),'--expected-preflight-sha256',sha(preflight/'gpu_preflight.json'),'--output',str(fit),'--device','cuda:0'])
   trained=require_complete(fit/'fit_status.json',training_complete=True,epochs_completed=40,actual_optimizer_steps=520)
   selected=checked_pointer(trained['selected_checkpoint'])
   if selected!=fit/'selected.pt':raise ValueError('Actual selected centre checkpoint identity required')
   evaluator='scripts/evaluate_current_cube_dual_crop_centre_large_20261104_v1.py'
   for role,count,start in [('selection',208,8),('reserved_fresh_centre_VAL',104,10)]:
    prediction=artifact/('centre_'+role)
    run(STAGES[start],[evaluator,'--predict','--role',role,'--manifest',str(manifest),'--selected',str(selected),'--expected-sha256',sha(selected),'--encoder-checkpoint',encoder['path'],'--output',str(prediction)],cpu=True)
    frozen=require_complete(prediction/'prediction_freeze.json',complete=True,role=role,samples=count,torch_num_threads=2,target_label_array_decodes_before_freeze=0)
    for item in frozen['files'].values():checked_pointer(item)
    run(STAGES[start+1],[evaluator,'--score','--output',str(prediction)],cpu=True)
    report=read(prediction/'report.json')
    if report.get('role')!=role or len(report.get('rows',[]))!=count or report.get('model_qualified')is not False:raise ValueError('Fixed complete role report required; no runtime qualification')
   verify_sources(config);status.update(complete=True,selected_e=e['selected_model'],selected_centre=pointer(selected),compiled_manifest=pointer(manifest))
   write_new(out/'completion.json',status)
   return status
  except BaseException as error:
   status.update(error_type=type(error).__name__,error=str(error));write_new(out/'failure.json',status);raise
  finally:
   write_new(out/'terminal_status.json',dict(status,finished_utc=now()))
   for signum,handler in previous.items():signal.signal(signum,handler)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);parser.add_argument('--config-sha256');parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
 if args.execute:
  if not args.config_sha256:parser.error('Root external frozen --config-sha256 required for --execute')
  result=execute(args.config,args.config_sha256)
 else:
  config=load_config(args.config,args.config_sha256)
  result=dict(complete=True,preparation_only=True,config=pointer(args.config),stage_order=list(STAGES),pending_inputs={key:not Path(config[key]).is_file()for key in ('e_actual_exit','CF_final_admission','QA_admission')},child_processes_started=0,model_forward_calls=0,optimizer_steps=0,Native_started=False,all_task_Native_VAL_TEST_authorized=False)
 print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
