"""Withdraw only never-started superseded requests; preserve every request byte."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,time,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
BASE=ROOT/'artifacts/framework/multitask_r10';OWNER=ROOT/'runtime/framework/native_resource_owner_v401'
PAIRS=(('block_next_sample_v459','block_next_position_v477'),('cup_next_sample_v462','cup_next_position_v480'))

def start_ticks(pid):return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
def signature(r):
 a=r['argv'];return tuple(a[a.index(k)+1] for k in ('--task','--raw-episode-id','--method'))
def main():
 launch=json.loads((OWNER/'launch.json').read_text());pid=launch['pid']
 if start_ticks(pid)!=launch['start_ticks']:raise ValueError('sole owner identity changed')
 receipt=OWNER/'retired_requests_v481/receipt.json'
 if receipt.exists():raise FileExistsError('retirement already recorded')
 replacement={}
 for old,new in PAIRS:
  setup=BASE/(new+'_setup');cpu=json.loads((setup/'CPU_acceptance.json').read_text())
  if cpu['complete'] is not True:raise ValueError('replacement actual argv CPU checks incomplete')
  replacement[old]={signature(json.loads(p.read_text())):p for p in (OWNER/'requests').glob('*.json')
   if json.loads(p.read_text())['job_id'].startswith(new+'_')}
  expected=16 if 'block_' in new else 4
  if len(replacement[old])!=expected:raise ValueError('all fixed-source replacement requests must already be queued')
 before=json.loads((OWNER/'status.json').read_text())
 if before['stop_requested'] or len(before['active_jobs'])!=2:raise ValueError('require two occupied live slots before scheduler bookkeeping pause')
 for job in before['active_jobs']:
  l=json.loads((OWNER/'jobs'/job/'launch.json').read_text())
  if start_ticks(l['launcher_pid'])!=l['launcher_start_ticks']:raise ValueError('occupied job already exited; inspect again')
 stopped=False;moved=[]
 try:
  os.kill(pid,signal.SIGSTOP);stopped=True
  for _ in range(100):
   state=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]
   if state in ('T','t'):break
   time.sleep(.01)
  else:raise ValueError('scheduler did not enter brief bookkeeping pause')
  status=json.loads((OWNER/'status.json').read_text());active=set(status['active_jobs']);completed=set(status['completed'])
  for path in sorted((OWNER/'requests').glob('*.json')):
   request=json.loads(path.read_text());job=request['job_id'];old=next((n for n,_ in PAIRS if job.startswith(n+'_')),None)
   if old is None or job in active or job in completed or (OWNER/'jobs'/job).exists():continue
   counterpart=replacement[old].get(signature(request))
   if counterpart is None:raise ValueError('no checked replacement for this exact task/source/method')
   dest=receipt.parent/path.name;dest.parent.mkdir(exist_ok=True)
   if dest.exists():raise FileExistsError('preserve previous withdrawn request')
   sha=file_sha(path);path.rename(dest)
   moved.append(dict(job_id=job,old_request_sha256=sha,retained_request=str(dest.relative_to(ROOT)),
    checked_replacement_request=str(counterpart.relative_to(ROOT)),native_started=False,completed_result_modified=False))
  atomic_json(receipt,dict(created_utc=datetime.now(timezone.utc).isoformat(),retired=moved,retired_count=len(moved),
   running_native_jobs_untouched=list(active),every_completed_case_retained=True,
   scheduler_only_briefly_paused_for_request_move=True,original_request_bytes_preserved=True,
   source_sha256=file_sha(__file__),automatic_retry=False))
 finally:
  if stopped:os.kill(pid,signal.SIGCONT)
 print(json.dumps(dict(retired_count=len(moved),running_jobs_untouched=True)),flush=True)
if __name__=='__main__':main()
