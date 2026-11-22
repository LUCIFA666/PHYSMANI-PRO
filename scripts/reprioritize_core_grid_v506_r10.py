"""Move never-started checked core jobs ahead of expansion, without relaunch."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,time,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
OWNER=ROOT/'runtime/framework/native_resource_owner_v401'
def ticks(pid):return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
def main():
 receipt=OWNER/'priority_changes_v506/receipt.json'
 if receipt.exists():raise FileExistsError('priority change already recorded')
 launch=json.loads((OWNER/'launch.json').read_text());pid=launch['pid']
 if ticks(pid)!=launch['start_ticks']:raise ValueError('sole owner identity changed')
 before=json.loads((OWNER/'status.json').read_text())
 if before['stop_requested'] or len(before['active_jobs'])!=2:raise ValueError('require two occupied live slots')
 for job in before['active_jobs']:
  child=json.loads((OWNER/'jobs'/job/'launch.json').read_text())
  if ticks(child['launcher_pid'])!=child['launcher_start_ticks']:raise ValueError('occupied worker already finished; inspect again')
 stopped=False;moved=[]
 try:
  os.kill(pid,signal.SIGSTOP);stopped=True
  for _ in range(100):
   if Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0] in ('T','t'):break
   time.sleep(.01)
  else:raise ValueError('scheduler bookkeeping pause failed')
  status=json.loads((OWNER/'status.json').read_text())
  for path in sorted((OWNER/'requests').glob('0050_*.json')):
   r=json.loads(path.read_text());job=r['job_id']
   if not job.startswith(('core10_fixed_four_v446_','core_input_edges_v465_')):continue
   if job in status['active_jobs'] or job in status['completed'] or path.name in status['rejected'] or (OWNER/'jobs'/job).exists():continue
   target=path.with_name('0041_'+path.name.removeprefix('0050_'))
   if target.exists():raise FileExistsError('destination already submitted')
   sha=file_sha(path);path.rename(target)
   if file_sha(target)!=sha:raise ValueError('request bytes changed')
   moved.append(dict(job_id=job,old_request=str(path.relative_to(ROOT)),new_request=str(target.relative_to(ROOT)),
    identical_request_sha256=sha,native_started=False,argv_and_output_unchanged=True))
  atomic_json(receipt,dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=moved,moved_count=len(moved),
   new_priority='0041 after baseline0036 and first-source repairs0040, before remove expansion0043',
   running_native_jobs_untouched=status['active_jobs'],new_jobs_started_by_this_tool=0,automatic_retry=False,source_sha256=file_sha(__file__)))
 finally:
  if stopped:os.kill(pid,signal.SIGCONT)
 print(json.dumps(dict(moved_count=len(moved),new_native_launches=0,running_jobs_untouched=True)))
if __name__=='__main__':main()
