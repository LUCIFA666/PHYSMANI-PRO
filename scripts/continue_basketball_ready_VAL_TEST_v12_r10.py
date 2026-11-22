"""Continue existing model audit into one frozen three-method VAL/TEST run."""
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'basketball_three_method_readiness_coordinator_v12'
LATEST=ROOT/'runtime/framework/handoff_runtime_latest.json'
def read(p):return json.loads(Path(p).read_text())
def ptr(p):return dict(path=str(p.resolve().relative_to(ROOT)),sha256=file_sha(p))
def process(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,start_ticks=int(fields[19]),command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
 except OSError:return None
def main():
 if OUT.exists():raise RuntimeError('Preserve an existing readiness coordinator; no automatic restart')
 OUT.mkdir()
 with (OUT/'owner.lock').open('a') as owner:
  fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  latest=read(LATEST);audit=latest['model_coverage_CPU_process'];own=process(os.getpid())
  launch=dict(created_utc=datetime.now(timezone.utc).isoformat(),process=own,source=ptr(Path(__file__)),model_audit=audit,
   methods=['cv_geometry','context_geometry','physmani'],automatic_retry=False,read_official_TEST_before_VAL_release=False)
  atomic_json(OUT/'launch.json',launch)
  latest.update(status='basketball_three_method_readiness_coordinator_running',updated_utc=launch['created_utc'],
   readiness_coordinator=dict(**own,source=launch['source']),current_core_TRAIN_reset=dict(complete=True,actual_cases=56,
    shared_hook_verified_each_case=True,report=ptr(BASE/'current_core_TRAIN_reset_repeat_v12/pending_report.json')),
   native_processes_all_terminal=True,next_required_steps=['Monitor the existing readiness coordinator; no duplicate audit/native jobs.','Complete original model proof, freeze VAL60, release audit, then same-policy TEST120.'])
  atomic_json(LATEST,latest,overwrite=True)
  try:
   report=BASE/'basketball_supported_v10_model_readiness_v1/coverage_initial.json'
   while not report.exists():
    state=process(audit['pid'])
    if state is None or state['start_ticks']!=audit['start_ticks']:raise RuntimeError('Original model audit ended without a completed report; retain pending state')
    if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise RuntimeError('Controlled STOP prevents formal launch')
    atomic_json(OUT/'status.json',dict(stage='waiting_for_existing_complete_model_proof',updated_utc=datetime.now(timezone.utc).isoformat(),actual_model_process=state),overwrite=True)
    time.sleep(10)
   proof=read(report)
   if not proof.get('formal_model_coverage_passed') or proof.get('errors'):raise RuntimeError('Model proof did not pass: '+repr(proof.get('errors')))
   raw=read(BASE/'current_core_TRAIN_reset_repeat_v12/pending_report.json')
   if not (raw['fit_repeat_data_passed'] and raw['independent_repeat_validation_passed']):raise RuntimeError('Current shared reset repeat did not pass')
   for p in raw['record_files_sha256']:
    hooks=read(p).get('shared_development_reset_hook_receipts',[])
    if len(hooks)!=1 or not hooks[0].get('complete'):raise RuntimeError('Actual shared reset witness missing')
   atomic_json(OUT/'status.json',dict(stage='preparing_frozen_formal_VAL',updated_utc=datetime.now(timezone.utc).isoformat(),model_report=ptr(report)),overwrite=True)
   environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   for script,logname in [('scripts/prepare_basketball_three_method_v12_r10.py','prepare.log'),('scripts/run_basketball_VAL_TEST_v12_r10.py','VAL_TEST.log')]:
    command=[sys.executable,script]
    with (OUT/logname).open('xb') as log:
     child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     record=dict(command=command,process=process(child.pid),launched_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False)
     atomic_json(OUT/(logname+'.launch.json'),record)
     code=child.wait()
     atomic_json(OUT/(logname+'.terminal.json'),dict(record,returncode=code,finished_utc=datetime.now(timezone.utc).isoformat()))
     if code!=0:raise RuntimeError('Authorized continuation stopped with retained evidence: '+logname)
   atomic_json(OUT/'status.json',dict(stage='complete',updated_utc=datetime.now(timezone.utc).isoformat()),overwrite=True)
  except BaseException as error:
   atomic_json(OUT/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),automatic_retry=False,partial_evidence_preserved=True))
   atomic_json(OUT/'status.json',dict(stage='pending_failure',failure=ptr(OUT/'failure.json')),overwrite=True)
   raise
if __name__=='__main__':main()
