"""CPU-only large-centre replay on every closed V72 development public frame.

Default is metadata/source preparation only. Explicit prediction requires the
whole twelve-stage centre continuation to finish. All1046 public predictions
(including failed rows) are frozen before any Native trajectory/initial truth
is opened. Reconstructed public7 is NOT claimed byte-identical to old raw7.
No old GPU/Native resource lock, Native launch, control action or training.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/framework/local_mpc_20261102'
MAIN=BASE/'centre_large_v2_continuation_run_v1'
MAIN_CONFIG=ROOT/'config/framework/centre_large_v2_continuation_20261104_v1/plan.json'
MAIN_CONFIG_SHA='c53c45f7915041391bc5a05c0a5a073c820e82df18bfc2c74d00c26a73649c62'
PROOF_HELPER=ROOT/'scripts/run_semantic_task_e_v2_deferred_20261104_v3.py'
PROOF_HELPER_SHA='c372bdc98a26b8625c76f47484ec04b65a330a460a602f257d4c82489102e6a2'
PROOF_PLAN=ROOT/'config/framework/semantic_task_e_v2_deferred_20261104_v3/plan.json'
PROOF_PLAN_SHA='84bfcb635f065f9c070f363a5d8338af8c951f5ea49c3377a9ba52c5647735a8'
PLAN=ROOT/'config/framework/P4_position_dwell_v72_development_canaries_v1/plan.json'
PLAN_SHA='489ca3197b50dd0495ba0c3ba90957141d6cda1c001fcd63172e5b50e0f835e6'
V72_FREEZE=BASE/'V72_fixed6_execution_preparation_v1/execution_source_freeze.json'
V72_FREEZE_SHA='a175465beae2f4b7a7ff1912bd502f6e2c314c052c507a6dcaa9746dbc580abb'
TINY_HELPER=ROOT/'scripts/replay_current_cube_dual_crop_closed_v72_20261104_v1.py'
TINY_HELPER_SHA='0cb67964b8e4527b1602e8e4b8141991f636f7ce1ed72425cb86494afe2f0e4c'
LARGE_DELIVERY=BASE/'current_cube_centre_large_deterministic_pool_v2_delivery/receipt.json'
ENCODER=ROOT/'artifacts/framework/near_goal_segmentation_20261103/full32_geometryF_unknownKL_fit_v5/selected.pt'
ENCODER_SHA='dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa'
LOCK=ROOT/'runtime/framework/large_centre_V72_CPU_replay_20261104_v2.lock'
CASES=('00_val_static','01_val_static','03_val_uniform','04_val_uniform','uniform_002','static_no_proxy')
COUNTS=dict(zip(CASES,(161,161,161,161,201,201)))
STAGES=('semantic_selected_evaluation','CF_first_fit13','CF_first_fit_QA','CF_remaining39','CF_all40_audit','CF_compile','centre_GPU_preflight','centre_fit40','centre_selection_predict208','centre_selection_score208','centre_reserved_predict104','centre_reserved_score104')
DEFAULT_OUTPUT=BASE/'current_cube_dual_crop_large_V72_closed6_replay_v2'
DT=.05


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pointer(p):return dict(path=str(Path(p).resolve(strict=True)),sha256=sha(p))
def checked(v):
 if not isinstance(v,dict)or set(v)!={'path','sha256'}:raise ValueError('Exact path/SHA pointer required')
 p=Path(v['path']).resolve(strict=True)
 if sha(p)!=v['sha256']:raise ValueError('Bound file changed: '+str(p))
 return p
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):
 with Path(p).open()as f:
  for s in f:
   if s.strip():yield json.loads(s)
def verify(mapping):
 for p,h in mapping.items():
  if sha(p)!=h:raise ValueError('Bound source/public data changed: '+p)
def stage_ok(v,name=None):
 if (type(v.get('actual_process_returncode'))is not int or v['actual_process_returncode']!=0
     or v.get('actual_wait_completed')is not True or v.get('source_unchanged')is not True
     or v.get('error')is not None or v.get('stop_reason')is not None or v.get('unreaped_descendants',[])!=[]):raise ValueError('Actual closed source0/wait/no-unreaped required')
 if name is not None and v.get('stage')!=name:raise ValueError('Exact stage owner required')


def method_sources():
 if sha(TINY_HELPER)!=TINY_HELPER_SHA or sha(PLAN)!=PLAN_SHA or sha(V72_FREEZE)!=V72_FREEZE_SHA or sha(MAIN_CONFIG)!=MAIN_CONFIG_SHA:raise ValueError('Frozen helper/development plan/main identities required')
 if sha(PROOF_HELPER)!=PROOF_HELPER_SHA or sha(PROOF_PLAN)!=PROOF_PLAN_SHA:raise ValueError('Frozen two-part closure evidence helper required')
 large=read(checked(read(LARGE_DELIVERY)['method_sources']))
 if len(large)!=21:raise ValueError('New large-centre21 source graph required')
 sources=dict(large)
 for p in (Path(__file__).resolve(),TINY_HELPER,PLAN,V72_FREEZE,MAIN_CONFIG,LARGE_DELIVERY,PROOF_HELPER,PROOF_PLAN):sources[str(p)]=sha(p)
 verify(sources);return sources


def main_gate():
 """Exact old failed-prefix + independent newcentre6; no invented parent waits."""
 if sha(PROOF_HELPER)!=PROOF_HELPER_SHA or sha(PROOF_PLAN)!=PROOF_PLAN_SHA:raise ValueError('Frozen two-part proof changed')
 spec=importlib.util.spec_from_file_location('frozen_two_part_meta_proof_only',PROOF_HELPER);proof=importlib.util.module_from_spec(spec);spec.loader.exec_module(proof)
 plan=proof.load_plan(PROOF_PLAN,PROOF_PLAN_SHA);facts=proof.validate_prior(plan)
 completion=read(MAIN/'completion.json')
 if completion.get('selected_centre')!=facts['new_centre_selected']or completion.get('compiled_manifest')!=facts['CF_manifest']:raise ValueError('Actual newcentre selected/manifest identity required')
 selected=checked(completion['selected_centre'])
 if selected!=MAIN/'artifacts/centre_fit/selected.pt':raise ValueError('New deterministic large-family actual fit sibling required')
 return completion,facts


def closed_gate():
 """All six lifecycle/source274 gates before truth file access, even hashing.

result.json is deliberately not parsed here: it may carry Native outcomes.
initial_snapshot/native_ticks/evaluation_labels are not opened or hashed.
"""
 if sha(PLAN)!=PLAN_SHA or sha(V72_FREEZE)!=V72_FREEZE_SHA:raise ValueError('Fixed V72 plan/source freeze changed')
 plan=read(PLAN);jobs=plan['jobs'];expected=read(V72_FREEZE)['files_sha256']
 if tuple(j['id']for j in jobs)!=CASES or len(expected)!=274:raise ValueError('Exact closed V72 fixed6/source274 required')
 cases=[];live={};metadata={};archives=0
 for job in jobs:
  case=(ROOT/job['path']).resolve(strict=True)
  if any('test'in p.casefold()for p in case.parts):raise ValueError('No TEST replay')
  owner_path=case.with_name(case.name+'.owner.actual_exit.json');worker_path=case/'worker_exit.json';run_path=case/'run_inputs.json';freeze_path=case/'execution_source_freeze.json'
  owner=read(owner_path);worker=read(worker_path);run=read(run_path);frozen=read(freeze_path)
  stage_ok(owner)
  if (Path(owner.get('output','')).resolve()!=case or type(worker.get('exit_code'))is not int
      or worker['exit_code']!=0 or worker.get('native_closed')is not True or worker.get('result_recorded')is not True or run.get('TEST_read')is not False
      or run.get('source_sha256')!=expected or frozen.get('files_sha256')!=expected
      or run.get('config',{}).get('formal_evaluation')is True):raise ValueError('All6 actual0/wait/nativeclosed/common source required')
  for relative,h in expected.items():
   r=Path(relative)
   if r.is_absolute()or'..'in r.parts:raise ValueError('Repo-relative source paths required')
   archived=case/'source_snapshot'/r
   if sha(archived)!=h:raise ValueError('Captured source archive changed')
   archives+=1
   if relative not in live:
    if sha(ROOT/r)!=h:raise ValueError('Frozen V72 current source changed')
    live[relative]=h
  for p in (owner_path,worker_path,run_path,freeze_path):metadata[str(p)]=sha(p)
  negative=any(run['config'].get(k)is True for k in ('counterfactual_no_proxy','diagnostic_no_proxy_attachment'))
  if negative!=(job['id']=='static_no_proxy'):raise ValueError('Predeclared single NoProxy case required')
  cases.append(dict(id=job['id'],case=case,negative=negative,expected_frames=COUNTS[job['id']]))
 return cases,dict(complete=True,cohort_cases=6,all_actual_exits0_wait_and_nativeclosed=True,source_files_per_case=274,archive_files_verified=archives,current_source_files_verified=len(live),metadata_sha256=metadata,source_sha256={str(ROOT/k):v for k,v in live.items()},GT_trajectories_or_initial_snapshots_or_resultJSON_opened=False)


def load_public_helper():
 if sha(TINY_HELPER)!=TINY_HELPER_SHA:raise ValueError('STOP public reconstruction helper changed')
 spec=importlib.util.spec_from_file_location('stopped_public7_reconstruction_only',TINY_HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper


def public_anchor(first_result):
 metadata=first_result['goal_metadata'];goal=first_result['goal_pose']['position'];offset=metadata['marker_to_goal_m']
 if metadata.get('simulator_pose_used')is not False or metadata.get('backend')!='rgbd_red_goal_sphere_train_calibrated_frame' or len(goal)!=3 or len(offset)!=3:raise ValueError('Original public RGBD initial sphere fit required')
 anchor=[float(a)-float(b)for a,b in zip(goal,offset)]
 if not all(math.isfinite(v)for v in anchor):raise ValueError('Finite public anchor required')
 return anchor


def reconstructed_public_input(raw,calibration,first_result,helper):
 # The helper uses only raw images/depth/K and stored PUBLIC camera pose/goal fit.
 value=helper.public_input(raw,calibration,first_result)
 provenance=dict(reconstruction='public_camera_xyz_quaternion_to_R_and_public_goal_minus_marker_offset',original_raw7_bitexact_claim=False,original_rgb_depth_and_stored_K_values_preserved=True,rawRGBD_intrinsics_modified=False,Native_object_pose_or_robot_or_task_as_model_input=False)
 return value,provenance


def tick_key(timestamp):
 t=float(timestamp)
 if not math.isfinite(t):raise ValueError('Finite actual timestamp required')
 tick=int(round(t/DT))
 if abs(t-tick*DT)>1e-8:raise ValueError('Actual50ms same-time grid required; no interpolation')
 return tick


def decode_outputs(values,count):
 import numpy as np
 keys=('centre_world','measurement_availability_logit','uncertainty_score_m','uncertainty_score_valid')
 expected=((count,3),(count,),(count,3),(count,))
 if np.asarray(values.get('uncertainty_score_valid',[])).dtype!=np.bool_:raise ValueError('Boolean large-head support flag required')
 if any(k not in values or np.asarray(values[k]).shape!=shape for k,shape in zip(keys,expected)):raise ValueError('Complete aligned large-head output shapes required')
 decoded=[]
 for i in range(count):
  parts={k:np.asarray(values[k])[i]for k in keys}
  if any(not np.isfinite(v).all()for v in parts.values()):
   decoded.append(dict(prediction_status='failed_nonfinite_output',failure_reason='nonfinite large-head output',centre_world=None,availability_probability=None,uncertainty_score_m=None,uncertainty_score_valid=False,candidate_available=False));continue
  probability=float(np.exp(-np.logaddexp(0.,-float(parts['measurement_availability_logit']))))
  valid=bool(parts['uncertainty_score_valid'])
  decoded.append(dict(prediction_status='ok',failure_reason=None,centre_world=parts['centre_world'].astype(float).tolist(),availability_probability=probability,uncertainty_score_m=parts['uncertainty_score_m'].astype(float).tolist(),uncertainty_score_valid=valid,candidate_available=probability>=.5 and valid))
 return decoded


def failed_output(reason):return dict(prediction_status='failed',failure_reason=reason,centre_world=None,availability_probability=None,uncertainty_score_m=None,uncertainty_score_valid=False,candidate_available=False)

def guard(output):
 if any(p.exists()for p in (Path(output)/'STOP',Path(output).with_suffix('.STOP'),BASE/'STOP',ROOT/'runtime/framework/STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):raise InterruptedError('Replay administrative STOP')

@contextmanager
def replay_lock():
 LOCK.parent.mkdir(parents=True,exist_ok=True)
 with LOCK.open('a+')as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:fcntl.flock(handle,fcntl.LOCK_UN)


def predict(selected,expected_sha,output):
 import numpy as np
 guard(output);sources=method_sources();complete,main_proof=main_gate();cases,gate=closed_gate()
 selected=Path(selected).resolve(strict=True)
 if complete['selected_centre']!=dict(path=str(selected),sha256=expected_sha):raise ValueError('Actual main selected centre pointer required')
 output=Path(output).resolve()
 if output.exists()or BASE not in output.parents:raise FileExistsError('Fresh independent artifact output only')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from our_mani.current_cube_rgbd_centre_large_20261104_v2.evaluation import _validate_selected
 from our_mani.current_cube_rgbd_centre_large_20261104_v2.model import LargeDualCropCubeCentreV2,CentreConfig
 from our_mani.current_cube_rgbd_centre_large_20261104_v2.config import fixed_config
 from our_mani.current_cube_rgbd_centre_20261103_v1.features import public_model_arrays
 payload,meta=_validate_selected(selected,expected_sha,complete['compiled_manifest']['path'])
 if sha(ENCODER)!=ENCODER_SHA:raise ValueError('Original dbf prefix initializer changed')
 config=fixed_config();batch_size=config['micro_batch_size']
 if batch_size!=2:raise ValueError('Fixed large evaluation microbatch2 required')
 torch.set_num_threads(2);model=LargeDualCropCubeCentreV2(CentreConfig(**config['architecture']),encoder_trainable=True)
 model.load_selected_v5_encoder(ENCODER,ENCODER_SHA);model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
 for parameter in model.parameters():parameter.requires_grad_(False)
 helper=load_public_helper();sources[str(ENCODER)]=ENCODER_SHA;sources[str(selected)]=expected_sha;sources.update(gate['source_sha256'])
 output.mkdir(parents=True,exist_ok=False);predictions=[];bound={};calls=0;forward_rows=0;returned_calls=0;batch_failures=[]
 try:
  write(output/'request.json',dict(selected=pointer(selected),main_completion=main_proof,closed6_gate=gate,method_source_sha256=sources,frames=1046,threshold=.5,CPU_threads=2,batch_size=2,GT_reads=0,original_raw7_bitexact_claim=False,qualification=False))
  with torch.no_grad():
   for case in cases:
    shadow=case['case']/'public_shadow';path=shadow/'public_outputs.jsonl';calpath=shadow/'camera_calibration.json'
    bound[str(path)]=sha(path);bound[str(calpath)]=sha(calpath);packets=list(lines(path));calibration=read(calpath)
    if len(packets)!=case['expected_frames']:raise ValueError('All fixed public frames required per case')
    first=packets[0]['result'];case_anchor=public_anchor(first);cache=[];indices=[]
    def flush():
     nonlocal calls,forward_rows,returned_calls
     if not cache:return
     try:
      batch={k:torch.as_tensor(np.stack([v[k]for v in cache]),dtype=torch.float32,device='cpu')for k in cache[0]}
      calls+=1;forward_rows+=len(cache)
      value=model(**batch);returned_calls+=1;arrays={k:value[k].detach().cpu().numpy()for k in ('centre_world','measurement_availability_logit','uncertainty_score_m','uncertainty_score_valid')}
      decoded=decode_outputs(arrays,len(cache))
     except Exception as error:
      reason=type(error).__name__+': '+str(error);batch_failures.append(dict(call=calls,frame_indices=list(indices),reason=reason));decoded=[failed_output(reason)for _ in cache]
     for index,result in zip(indices,decoded):predictions[index].update(result)
     cache.clear();indices.clear()
    for local,packet in enumerate(packets):
     guard(output);timestamp=float(packet['result']['timestamp'])
     if tick_key(timestamp)!=local or abs(timestamp-float(packet['public_robot']['timestamp']))>1e-8:raise ValueError('Exact original same-time ordered frame ledger required')
     obj=packet['result'].get('object_observation');trace=packet['result'].get('depth_refiner_trace')or{}
     row=dict(case_id=case['id'],index=packet['index'],timestamp=timestamp,tick_index=local,expected_negative=case['negative'],original_public_target_available=obj is not None,original_pipeline_available=bool(packet['result'].get('available')),original_public_centre=None if obj is None else obj['pose']['position'],original_runtime_packed_public_sha256=trace.get('packed_current_input_sha256'),public_sphere_center=case_anchor,reconstructed_public_input_sha256=None,packed_digest_equal=None,**failed_output('not_processed'))
     predictions.append(row);index=len(predictions)-1
     try:
      raw=(shadow/packet['rgbd']).resolve(strict=True)
      if shadow/'rgbd'not in raw.parents:raise ValueError('Current RGBD cache path required')
      bound[str(raw)]=sha(raw);inputs,provenance=reconstructed_public_input(raw,calibration,first,helper)
      if abs(float(inputs['timestamp'])-timestamp)>1e-8:raise ValueError('Raw current timestamp mismatch; no interpolation')
      row.update(public_sphere_center=inputs['public_sphere_center'].tolist(),reconstructed_public_input_sha256=helper.public_digest(inputs),input_reconstruction=provenance)
      if row['original_runtime_packed_public_sha256']is not None:row['packed_digest_equal']=row['reconstructed_public_input_sha256']==row['original_runtime_packed_public_sha256']
      cache.append(public_model_arrays(inputs));indices.append(index)
     except Exception as error:row.update(failed_output('public_input '+type(error).__name__+': '+str(error)))
     if len(cache)==batch_size or local==len(packets)-1:flush()
    write(output/(case['id']+'.prediction_progress.json'),dict(frames=case['expected_frames'],case_id=case['id'],GT_reads=0))
  if len(predictions)!=1046 or Counter(p['case_id']for p in predictions)!=COUNTS:raise ValueError('All fixed1046 frames must remain')
  verify(sources);verify(bound);write(output/'public_predictions.json',predictions)
  freeze=dict(complete=True,version='deterministic_large_current_centre_V72_closed6_CPU_replay_20261104_v2',centre_family='current_cube_rgbd_centre_large_20261104_v2',predictions=pointer(output/'public_predictions.json'),selected=pointer(selected),fit_manifest=complete['compiled_manifest'],main_chain_gate=main_proof,closed6_gate=gate,cases=[dict(id=c['id'],path=str(c['case']),expected_frames=c['expected_frames'],negative=c['negative'])for c in cases],method_source_sha256=sources,public_data_sha256=bound,frames=1046,case_counts=COUNTS,failed_output_rows=sum(r['prediction_status']!='ok'for r in predictions),batch_failures=batch_failures,actual_model_forward_batch_calls=calls,actual_model_forward_rows=forward_rows,returned_model_forward_batch_calls=returned_calls,CPU_threads=2,device='cpu',autocast=False,threshold=.5,GT_trajectories_or_initial_snapshots_read_before_freeze=False,original_raw7_bitexact_claim=False,input_reconstruction='Original rgb/depth/K values; camera R reconstructed from public quaternion; public Sphere from cached goal-fit metadata; floating roundtrips may differ.',packed_digest_matches=sum(r['packed_digest_equal']is True for r in predictions),packed_digest_mismatches=sum(r['packed_digest_equal']is False for r in predictions),NoProxy_is_predeclared_metadata_not_model_input=True,current_trajectory_unchanged=True,Native_or_training_started=False,GPU_used=False,runtime_qualified=False,TEST_read=False)
  write(output/'prediction_freeze.json',freeze);return dict(prediction_freeze=pointer(output/'prediction_freeze.json'),frames=1046,failed_output_rows=freeze['failed_output_rows'],model_forward_batch_calls=calls)
 except BaseException as error:
  write(output/'partial_public_predictions.json',predictions);write(output/'failure.json',dict(complete=False,error=type(error).__name__+': '+str(error),frames_processed=len(predictions),GT_read=False));raise
 finally:del model,payload


def statistics(values):
 import numpy as np
 a=np.asarray(values,float)
 return dict(count=len(a),mean_m=float(a.mean()),p95_m=float(np.quantile(a,.95)),max_m=float(a.max()),within3cm=int((a<=.03).sum()),within5cm=int((a<=.05).sum()))if len(a)else dict(count=0)

def summarize(group):
 valid=[r for r in group if r['centre_error_m']is not None];accepted=[r for r in valid if r['candidate_available']]
 old=[r for r in group if r['original_centre_error_m']is not None]
 return dict(frames=len(group),prediction_failed=sum(r['prediction_status']!='ok'for r in group),truth_missing=sum(r['truth_status']!='matched'for r in group),centre_errors_available=len(valid),new_available=sum(r['candidate_available']for r in group),error_all_returned=statistics([r['centre_error_m']for r in valid]),error_new_available=statistics([r['centre_error_m']for r in accepted]),error_original_public_available=statistics([r['original_centre_error_m']for r in old]),NoProxy_new_available=sum(r['expected_negative']and r['candidate_available']for r in group),scope_known=sum(r['actual_in_scope_0p20']is not None for r in group),actual_in_scope=sum(r['actual_in_scope_0p20']is True for r in group),in_scope_new_available=sum(r['actual_in_scope_0p20']is True and r['candidate_available']for r in group))


def score(output):
 import numpy as np
 output=Path(output);guard(output);freeze=read(output/'prediction_freeze.json')
 if freeze.get('complete')is not True or freeze.get('frames')!=1046 or freeze.get('GT_trajectories_or_initial_snapshots_read_before_freeze')is not False:raise ValueError('Complete all1046 public-only freeze required before truth')
 verify(freeze['method_source_sha256']);verify(freeze['public_data_sha256']);predictions=read(checked(freeze['predictions']))
 if len(predictions)!=1046 or Counter(r['case_id']for r in predictions)!=COUNTS:raise ValueError('Fixed complete prediction denominator required')
 _,gate=closed_gate();main_gate()
 write(output/'score_consumption.json',dict(prediction_freeze=pointer(output/'prediction_freeze.json'),one_attempt=True))
 truth={};ambiguous=set();bound={};duplicates=0
 for case in freeze['cases']:
  root=Path(case['path'])
  for name in ('initial_snapshot.json','native_ticks.jsonl'):
   path=root/name;bound[str(path)]=sha(path);samples=[read(path)['state']]if name.endswith('.json')else lines(path)
   for value in samples:
    key=(case['id'],tick_key(value['time']))
    selected=dict(time=float(value['time']),object_position=value['object_position'],held=bool(value['held']))
    if key in truth:
     duplicates+=1
     if truth[key]!=selected:ambiguous.add(key)
    else:truth[key]=selected
 scored=[]
 for prediction in predictions:
  key=(prediction['case_id'],prediction['tick_index']);actual=truth.get(key)
  match=actual is not None and key not in ambiguous and abs(actual['time']-prediction['timestamp'])<=1e-8
  if not match:
   scored.append(dict(prediction,truth_status='ambiguous'if key in ambiguous else'missing',actual_held=None,actual_in_scope_0p20=None,centre_error_m=None,original_centre_error_m=None));continue
  centre=np.asarray(actual['object_position'],float);anchor=prediction['public_sphere_center'];estimated=prediction['centre_world'];old=prediction['original_public_centre']
  if centre.shape!=(3,)or not np.isfinite(centre).all():raise ValueError('Finite actual same-time Native centre required')
  in_scope=None if anchor is None else bool(np.linalg.norm(centre-np.asarray(anchor,float))<=.20)
  scored.append(dict(prediction,truth_status='matched',actual_held=actual['held'],actual_in_scope_0p20=in_scope,centre_error_m=None if estimated is None else float(np.linalg.norm(np.asarray(estimated,float)-centre)),original_centre_error_m=None if old is None else float(np.linalg.norm(np.asarray(old,float)-centre))))
 verify(bound);groups=dict(positive_expected=[r for r in scored if not r['expected_negative']],NoProxy=[r for r in scored if r['expected_negative']],held=[r for r in scored if r['actual_held']is True],not_held=[r for r in scored if r['actual_held']is False],old_public_target_missing=[r for r in scored if not r['original_public_target_available']],old_pipeline_unavailable=[r for r in scored if not r['original_pipeline_available']],positive_in_scope_0p20=[r for r in scored if not r['expected_negative']and r['actual_in_scope_0p20']is True],held_in_scope_0p20=[r for r in scored if r['actual_held']is True and r['actual_in_scope_0p20']is True],old_public_target_missing_in_scope=[r for r in scored if not r['original_public_target_available']and r['actual_in_scope_0p20']is True],in_scope_0p20=[r for r in scored if r['actual_in_scope_0p20']is True],out_of_scope_0p20=[r for r in scored if r['actual_in_scope_0p20']is False])
 report=dict(complete=True,aggregate=summarize(scored),cases={name:summarize([r for r in scored if r['case_id']==name])for name in CASES},groups={k:summarize(v)for k,v in groups.items()},rows=scored,truth_files=bound,truth_duplicate_same_time_rows=duplicates,truth_conflicting_time_keys=len(ambiguous),prediction_freeze=pointer(output/'prediction_freeze.json'),all1046_including_failed_retained=True,Native_in_scope_is_NOT_visibility_ground_truth=True,held_is_posthoc_Native_label_not_model_input=True,uncertainty_score_is_NOT_CI=True,availability_threshold_fixed=.5,original_trajectory_and_actions_unchanged=True,new_task_rollouts=0,new_model_forward_calls=0,task_success_improvement_demonstrated=False,runtime_or_measurement_qualified=False,GPU_used=False,Native_started=False,training_started=False,TEST_read=False)
 write(output/'report.json',report);return dict(report=pointer(output/'report.json'),aggregate=report['aggregate'])


def watch_main(output):
 sources=method_sources();watch=Path(output).with_suffix('.watch.json')
 write(watch,dict(pid=os.getpid(),source_sha256=sources,waiting_for=str(MAIN/'completion.json'),output=str(output),GPU_or_Native_resource_locks_acquired=False))
 begin=time.monotonic()
 while not(MAIN/'completion.json').is_file()or not(MAIN/'terminal_status.json').is_file():
  guard(output)
  if(MAIN/'failure.json').exists():raise RuntimeError('Main continuation stopped; no replay')
  if time.monotonic()-begin>172800:raise TimeoutError('Main continuation wait limit')
  time.sleep(5)
 verify(sources);complete,_=main_gate();selected=checked(complete['selected_centre'])
 result=predict(selected,complete['selected_centre']['sha256'],output);result['score']=score(output);return result


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selected',type=Path);parser.add_argument('--expected-sha256');parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
 modes=parser.add_mutually_exclusive_group();modes.add_argument('--predict',action='store_true');modes.add_argument('--score',action='store_true');modes.add_argument('--watch-main',action='store_true');args=parser.parse_args(argv)
 if args.predict or args.score or args.watch_main:
  with replay_lock():
   if args.predict:
    if args.selected is None or args.expected_sha256 is None:parser.error('--predict requires exact large selected/SHA')
    result=predict(args.selected,args.expected_sha256,args.output)
   elif args.score:result=score(args.output)
   else:result=watch_main(args.output)
 else:
  sources=method_sources();result=dict(complete=True,preparation_only=True,expected_cases=list(CASES),expected_frames=1046,case_counts=COUNTS,large_source_count=21,source_pins=len(sources),source_sha256=sources,main_complete=(MAIN/'completion.json').exists(),model_forward_calls=0,Native_started=False,GPU_used=False,training_started=False,GT_read=False,original_raw7_bitexact_claim=False,old_GPU_Native_resource_locks_acquired=False,availability_threshold=.5,runtime_qualified=False)
 print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
