"""Prepare exact three-method VAL after TRAIN motion/core/model evidence passes.

No TEST roster is opened here. Verified unchanged training evidence is reused,
while actual new controller sources/identities are recomputed and frozen.
"""
from copy import deepcopy
from datetime import datetime,timezone
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.task_release import task_comparison_sha256
from our_mani.multitask.release_preflight import _Inputs
from our_mani.multitask import supported_geometry_task_release_v10 as prior
from our_mani.multitask import causal_safety_brake_geometry_task_release_v12 as current
from our_mani.multitask.reset_parity import profile_problems
from our_mani.multitask.interaction import InteractionLibrary
from scripts import evaluate_multitask_case_r10 as evaluator
from scripts import run_causal_safety_brake_geometry_formal_queue_v12_r10 as queue
from scripts import run_geometry_causal_safety_brake_formal_case_v12_r10 as child
from scripts import audit_current_core_train_resets_v12_r10 as reset
from types import SimpleNamespace
TASK='moving_basketball_in_hoop';BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'basketball_three_method_causal_v12'

def read(p):return json.loads(Path(p).read_text())
def ptr(p):return dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=file_sha(p))
def include(files,p):
 p=Path(p).resolve();sha=file_sha(p)
 if str(p) in files and files[str(p)]!=sha:raise RuntimeError('previous frozen file changed: '+str(p))
 files[str(p)]=sha

def upgrade_coverage(config):
 previous=BASE/'basketball_supported_v10_model_readiness_v1/coverage_initial.json';old=read(previous)
 old_config=BASE/'geometry_supported_reference_v10_setup/configuration.json';old_protocol=read(old_config)
 assert old['formal_model_coverage_passed'] and not old['errors']
 files=dict(old['input_files_sha256']);verify_inputs(ROOT,files);files[str(previous.resolve())]=file_sha(previous)
 prior.require_task_coverage_certificate(ROOT,TASK,ptr(previous),old_protocol,files)
 changed={'policy_overlay','continuation_identities','native_metric_diagnostics','initial_condition_tolerance_profile',
  'task_demo_policy','executor_TRAIN_qualification','task_model_coverage_certificate','task_validation_release','formal_comparison_ready','execution_scope','authorized_scope','experiment_plan_path','experiment_plan_sha256'}
 assert {k:v for k,v in old_protocol.items() if k not in changed}=={k:v for k,v in config.items() if k not in changed}
 # Context1400/VAL275 and the100-parent skill proof are reused only because all
 # their actual source/model/raw input bytes have just been verified unchanged.
 inputs=_Inputs(ROOT);inputs.include(files);inputs.file(old_config);inputs.file(Path(__file__))
 library=InteractionLibrary.load(inputs.pointer(config['learned_artifacts'][TASK]['interaction']))
 report=deepcopy(old);report['sections']['continuations']={p:current._continuation_identity(inputs,config,TASK,p,library) for p in ('cv','context')}
 inputs.file(current.__file__);inputs.file('our_mani/multitask/audit_sha_cache_v14.py');inputs.file('our_mani/multitask/source_verification_cache_v12.py');inputs.file('tests/test_audit_sha_cache_v14.py');inputs.file('tests/test_source_verification_cache_v12.py');inputs.rehash_all()
 report.update(schema=current.COVERAGE,coverage_source_sha256=file_sha(current.__file__),
  task_comparison_sha256=task_comparison_sha256(config),input_files_sha256=inputs.files,
  source_bound_unchanged_model_evidence_reuse=ptr(previous),model_evidence_rehashed=True,
  new_policy_continuations_recomputed=True,model_fitting_performed=False,actual_TEST_read=False,
  changed_configuration_fields=sorted(k for k in set(old_protocol)|set(config) if old_protocol.get(k)!=config.get(k)))
 return report

def main():
 if OUT.exists():raise RuntimeError('Preserve prepared formal namespace')
 qualification=BASE/'geometry_causal_safety_brake_v12/actual_smoke_report.json';q=read(qualification)
 assert q['actual_motion_fixed_TRAIN_pair_passed'] and all(r['actual_motion_qualification'] for r in q['methods'].values())
 # No task-success threshold is imposed or inferred from these two TRAIN cases.
 plan_path=BASE/'current_core_TRAIN_reset_repeat_v12_plan.json';reset_plan=reset.verify_plan(plan_path)
 repeat=BASE/'current_core_TRAIN_reset_repeat_v12/pending_report.json';r=read(repeat)
 assert r['complete'] and r['actual_cases']==56 and r['fit_repeat_data_passed'] and r['independent_repeat_validation_passed']
 verify_inputs(ROOT,r['record_files_sha256']);assert r['thresholds']==reset_plan['profile']['thresholds']
 for name in r['record_files_sha256']:
  raw=read(name);hooks=raw.get('shared_development_reset_hook_receipts',[])
  assert len(hooks)==1 and hooks[0].get('complete') is True and hooks[0].get('permission')==reset_plan['protocol']['development_reset_profile']
 profile=deepcopy(reset_plan['profile']);profile.update(status='frozen_after_TRAIN_repeat_validation',TRAIN_evidence=ptr(repeat),
  ROOT_current_core_review_passed=True,thresholds_changed=False)
 assert not profile_problems(profile,evidence_root=ROOT)
 c=read(BASE/'geometry_causal_safety_brake_v12_setup/configuration.json')
 OUT.mkdir();atomic_json(OUT/'TRAIN_reset_tolerance_profile.json',profile)
 atomic_json(OUT/'TRAIN_reset_ROOT_review.json',dict(complete=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
  raw_cases_verified=56,all14_tasks_fit_and_independent_repeat_passed=True,original_thresholds_retained=True,
  shared_development_hooks_used=True,profile=ptr(OUT/'TRAIN_reset_tolerance_profile.json'),evidence=ptr(repeat),
  no_full_physics_bit_identity_claim=True,TEST_read=False))
 atomic_json(OUT/'demo_policy.json',dict(methods=list(current.METHODS),videos_per_method_per_task=1,source_selection='fixed first original official TEST source; keep it in formal metrics',success_selection=False))
 c.update(experiment_plan_path='artifacts/framework/multitask_r10/geometry_only_comparison_v1/experiment_plan.json',experiment_plan_sha256=file_sha(BASE/'geometry_only_comparison_v1/experiment_plan.json'),execution_scope='fixed_normal_basketball_complete_three_method_VAL_then_official_TEST_v12',authorized_scope='User-authorized CV_geometry/Context_geometry/PhysMani comparison; fixed full VAL then release-gated full TEST; no Value, NGFF or FASTER restart',initial_condition_tolerance_profile=ptr(OUT/'TRAIN_reset_tolerance_profile.json'),task_demo_policy=ptr(OUT/'demo_policy.json'),executor_TRAIN_qualification=ptr(qualification))
 report=upgrade_coverage(c);atomic_json(OUT/'model_coverage.json',report)
 c.update(task_model_coverage_certificate=ptr(OUT/'model_coverage.json'),formal_comparison_ready=True)
 atomic_json(OUT/'configuration_VAL.json',c)
 files=read(BASE/'geometry_causal_safety_brake_v12_setup/frozen_inputs.json')['files'];files.update(report['input_files_sha256'])
 # Freeze the actual installed released baseline dependencies. Historical VAL
 # outcomes/config identities are not reused or relabelled by this queue.
 historical=BASE/'PhysMani_early_VAL_approved_v12/frozen_inputs.json';runtime=read(historical)['files'];verify_inputs(ROOT,runtime)
 for name,sha in runtime.items():
  if name in files and files[name]!=sha:raise RuntimeError('baseline/runtime freeze conflicts with actual ours inputs: '+name)
  files[name]=sha
 for p in [Path(__file__),OUT/'configuration_VAL.json',OUT/'model_coverage.json',OUT/'TRAIN_reset_tolerance_profile.json',OUT/'TRAIN_reset_ROOT_review.json',OUT/'demo_policy.json',qualification,repeat,plan_path,historical,
  Path(current.__file__),Path(queue.__file__),Path(child.__file__),ROOT/'scripts/run_basketball_VAL_TEST_v12_r10.py',ROOT/'scripts/continue_basketball_ready_VAL_TEST_v12_r10.py',ROOT/'scripts/run_multitask_fresh_queue_r10.py',ROOT/'our_mani/multitask/ready_candidate_queue.py',BASE/'geometry_only_comparison_v1/experiment_plan.json',ROOT/c['baseline_checkpoint'],BASE.parent/'protocol/validation_manifest.json']:
  include(files,p)
 for m in current.METHODS:
  for p in evaluator.required_local_sources(m,split='validation'):include(files,p)
 for name,sha in r['record_files_sha256'].items():files[str(Path(name).resolve())]=sha
 for name,sha in reset_plan['inputs_sha256'].items():
  if str(Path(name).resolve()) in files and files[str(Path(name).resolve())]!=sha:raise RuntimeError('reset source differs')
  files[str(Path(name).resolve())]=sha
 verify_inputs(ROOT,files)
 atomic_json(OUT/'frozen_inputs_VAL.json',dict(schema='r10_three_method_basketball_actual_VAL_source_lock_v12',files=files,approved_for_execution=True,prepared_only=True,native_started=False,TEST_roster_read=False,Value_or_NGFF_or_FASTER_enabled=False))
 current.require_task_coverage_certificate(ROOT,TASK,c['task_model_coverage_certificate'],c,files)
 args=SimpleNamespace(config=str((OUT/'configuration_VAL.json').relative_to(ROOT)),manifest='artifacts/framework/protocol/validation_manifest.json',frozen_inputs=str((OUT/'frozen_inputs_VAL.json').relative_to(ROOT)),task=TASK,split='validation',output=str((OUT/'validation').relative_to(ROOT)),plan=str((OUT/'VAL_plan.json').relative_to(ROOT)),launcher='scripts/run_local_sim.sh')
 plan=queue.prepare(args);checks={}
 for method in current.METHODS:
  job=next(j for j in plan['jobs'] if j['identity']['method']==method)
  argv=['--config',args.config,'--manifest',args.manifest,'--frozen-inputs',args.frozen_inputs,'--task',TASK,'--split','validation','--raw-episode-id',job['source_raw_episode_id'],'--method',method,'--replica','0','--out',str((OUT/'validation/cases'/job['job_id']/'attempt0000').relative_to(ROOT)),'--check']
  prepared=child.preflight(child.parser().parse_args(argv));checks[method]=prepared['model_preflight']
 atomic_json(OUT/'formal_CPU_acceptance.json',dict(actual_plan_argv_derived=True,methods=checks,VAL_expected_cases=plan['expected_cases'],TEST_started=False,TEST_roster_read=False))
 print(json.dumps(dict(prepared=True,task=TASK,methods=list(current.METHODS),VAL_cases=plan['expected_cases'],frozen_files=len(files))),flush=True)
if __name__=='__main__':
 from our_mani.multitask.audit_sha_cache_v14 import AuditSHACache
 from our_mani.multitask import acquisition_lineage,reference_time,reference_overrides
 from our_mani.multitask import evaluation
 originals=(evaluation.file_sha,acquisition_lineage.file_sha,reference_time.sha,reference_overrides.sha)
 audit_cache=AuditSHACache()
 for name,module in list(sys.modules.items()):
  if name=='__main__' or name.startswith(('our_mani','scripts')):
   for attribute in ('file_sha','sha'):
    function=getattr(module,attribute,None)
    if any(function is original for original in originals):setattr(module,attribute,audit_cache)
 main()
