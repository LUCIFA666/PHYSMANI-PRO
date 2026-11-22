"""Run the frozen basketball three-method VAL, release audit, then exact TEST.

No fitting, parameter changes, outcome-based seed replacement or retries. TEST
is not opened until complete three-method VAL protocol release succeeds.
"""
from copy import deepcopy
from datetime import datetime,timezone
import json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask import causal_safety_brake_geometry_task_release_v12 as release
from scripts import run_causal_safety_brake_geometry_formal_queue_v12_r10 as queue
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair,process_visibility
BASE=ROOT/'artifacts/framework/multitask_r10/basketball_three_method_causal_v12'
TASK='moving_basketball_in_hoop'
def read(p):return json.loads(Path(p).read_text())
def ptr(p):return dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=file_sha(p))
def status(stage,**extra):
 p=ROOT/'runtime/framework/handoff_runtime_latest.json';s=read(p)
 fields=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
 s.update(status=stage,updated_utc=datetime.now(timezone.utc).isoformat(),formal_task=TASK,
  formal_methods=list(release.METHODS),formal_supervisor=dict(pid=os.getpid(),start_ticks=int(fields[19]),source=ptr(Path(__file__))),
  native_processes_all_terminal=stage in ('basketball_VAL_TEST_complete','basketball_formal_pending_failure'),
  next_launch_command=None,automatic_retry=False,**extra)
 atomic_json(p,s,overwrite=True)
 atomic_json(BASE/'supervisor_status.json',dict(stage=stage,process_id=os.getpid(),updated_utc=s['updated_utc'],**extra),overwrite=True)

def main():
 if (BASE/'supervisor_launch.json').exists():raise RuntimeError('Preserve existing formal supervisor attempt; no automatic restart')
 inventory=process_visibility()
 if not inventory['host_visibility_available'] or inventory['live_project_native_processes'] or inventory['same_uid_inventory_incomplete']:raise RuntimeError('duplicate native jobs or unavailable host process view')
 with exclusive_pair():
  from our_mani.multitask.source_verification_cache_v12 import install as install_source_cache
  install_source_cache(ROOT,read(BASE/'frozen_inputs_VAL.json')['files'])
  plan=BASE/'VAL_plan.json';accepted=read(BASE/'formal_CPU_acceptance.json');assert accepted['VAL_expected_cases']==60 and set(accepted['methods'])==set(release.METHODS)
  atomic_json(BASE/'supervisor_launch.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),source=ptr(Path(__file__)),VAL_plan=ptr(plan),methods=list(release.METHODS),workers=2,GPU_workers_maximum=1,case_timeout_seconds=900,automatic_retry=False,TEST_before_VAL_release=False))
  try:
   status('basketball_VAL_three_methods_running',formal_VAL_plan=ptr(plan),formal_VAL_or_TEST_started=True,formal_TEST_started=False)
   val=queue.run(plan,2,900)
   if not val.get('complete'):raise RuntimeError('VAL queue pending; no TEST permission')
   status('basketball_complete_VAL_release_audit',formal_TEST_started=False)
   coverage=read(BASE/'configuration_VAL.json')['task_model_coverage_certificate']
   output=BASE/'VAL_release.json'
   result=release._finalize_task_validation_release(ROOT,TASK,plan,coverage,output)
   if result.get('formal_test_permitted') is not True:raise RuntimeError('VAL protocol release failed; '+repr(result))
   c=read(BASE/'configuration_VAL.json');c.update(task_validation_release=ptr(output),formal_comparison_ready=True)
   atomic_json(BASE/'configuration_TEST.json',c)
   files=dict(read(BASE/'frozen_inputs_VAL.json')['files'])
   for name,sha in result['input_files_sha256'].items():
    p=(ROOT/name).resolve()
    if str(p) in files and files[str(p)]!=sha:raise RuntimeError('TEST source conflicts with VAL freeze')
    files[str(p)]=sha
   official=ROOT/release.DEFAULT_TEST  # First TEST roster read, after complete VAL release.
   for p in [official,BASE/'configuration_TEST.json',output,BASE/'frozen_inputs_VAL.json',plan]:files[str(p.resolve())]=file_sha(p)
   verify_inputs(ROOT,files)
   atomic_json(BASE/'frozen_inputs_TEST.json',dict(schema='r10_same_policy_three_method_basketball_TEST_lock_v12',files=files,approved_for_execution=True,VAL_release=ptr(output),automatic_retry=False,parameters_changed_after_VAL=False))
   args=SimpleNamespace(config=str((BASE/'configuration_TEST.json').relative_to(ROOT)),manifest=str(official.relative_to(ROOT)),frozen_inputs=str((BASE/'frozen_inputs_TEST.json').relative_to(ROOT)),task=TASK,split='test',output=str((BASE/'test').relative_to(ROOT)),plan=str((BASE/'TEST_plan.json').relative_to(ROOT)),launcher='scripts/run_local_sim.sh')
   test_plan=queue.prepare(args);assert test_plan['source_episodes']==40 and test_plan['expected_cases']==120
   status('basketball_TEST_three_methods_running',formal_TEST_started=True,formal_TEST_plan=ptr(BASE/'TEST_plan.json'),formal_VAL_release=ptr(output))
   test=queue.run(BASE/'TEST_plan.json',2,900)
   if not test.get('complete'):raise RuntimeError('TEST queue pending')
   status('basketball_VAL_TEST_complete',formal_TEST_started=True,formal_VAL_release=ptr(output),formal_TEST_plan=ptr(BASE/'TEST_plan.json'))
  except BaseException as error:
   failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),automatic_retry=False,partial_cases_preserved=True,policy_failure_label_fabricated=False)
   atomic_json(BASE/'supervisor_failure.json',failure)
   status('basketball_formal_pending_failure',formal_failure=ptr(BASE/'supervisor_failure.json'))
   raise
if __name__=='__main__':main()
