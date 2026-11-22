"""Actual PhysMani stage adapter preparation; no model, Native or rawTEST reads.

--prepare-manifest fixes roles and header indices. --check validates representative
real public windows. --preprocess publishes a lazy-original256 window index and
validates the same real adapter; original HDF pixels are never replaced by64px.
"""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask_data_preparation_20261104_v1.manifest import prepare,TASKS,BASE_TASKS,sha,pointer,write
from our_mani.multitask_data_preparation_20261104_v1.data import SemanticTaskHistoryDataset,source_map


def run(manifest,output,stage,tasks=None,image_size=256):
 started=time.perf_counter();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
 method_sources=source_map();selected=TASKS if not tasks else tuple(tasks)
 if not set(selected)<=set(TASKS):raise ValueError('Only explicit16 tasks')
 fit=SemanticTaskHistoryDataset(manifest,'fit',image_size=image_size,tasks=selected)
 val=SemanticTaskHistoryDataset(manifest,'selection',image_size=image_size,tasks=selected)
 checks=[]
 # First preassigned eligible source per requested task, without model results.
 for task in selected:
  indices=[i for i,r in enumerate(fit.episodes)if r['task']==task]
  if not indices:raise ValueError('No eligible fit source for '+task)
  i=indices[0];sample=fit.public_episode(i,0)
  checks.append(dict(task=task,source_parent_group=sample['source_parent_group'],current_frame_index=0,
   rgbd_shape=list(sample['rgbd'].shape),robot_shape=list(sample['robot_history'].shape),length=int(sample['lengths']),
   RGB_range=[float(sample['rgbd'][:int(sample['lengths']),:,:3].min()),float(sample['rgbd'][:int(sample['lengths']),:,:3].max())],
   depth_m_range=[float(sample['rgbd'][:int(sample['lengths']),:,3].min()),float(sample['rgbd'][:int(sample['lengths']),:,3].max())],
   gripper_input_fixed_zero=bool((sample['robot_history'][:,-1]==0).all()),expert_labels_read=False,
   original256_pixels_read=image_size==256,physical_dt_valid=False))
 index=dict(schema='current_network16_lazy_original_HDF_window_index_v1',split_manifest=pointer(manifest),
  image_size=image_size,supported_original_resolutions=[64,128,256],original_HDF256_retained=True,
  storage='lazy original HDF; actual pixel decode/normalization in frozen adapter getter, no interpolation',
  fit_windows=[dict(source_parent_group=r['episode']['source_parent_group'],current_frame_index=r['current_frame_index'])for r in fit.records],
  selection_windows=[dict(source_parent_group=r['episode']['source_parent_group'],current_frame_index=r['current_frame_index'])for r in val.records],
  role_counts=fit.metadata['role_counts'],actual_public_window_checks=checks,representative_model_forward_calls=0)
 write(out/'window_index.json',index)
 fit.source_verification();val.source_verification()
 if method_sources!=source_map():raise ValueError('Adapter code/source changed')
 report=dict(schema='current_network16_actual_HDF_adapter_preparation_report_v1',complete=True,stage=stage,method='current_network',
  split_manifest=pointer(manifest),window_index=pointer(out/'window_index.json'),source_sha256=method_sources,
  source_unchanged=True,eligible_fit_parents=len(fit.episodes),eligible_selection_parents=len(val.episodes),fit_windows=len(fit),selection_windows=len(val),
  representative_public_windows_decoded=len(checks),representative_tasks=list(selected),actual_public_checks=checks,
  RGBD_shape=[5,4,4,image_size,image_size],robot_shape=[5,15],real_clock_available=False,
  time_origin='recorded_frame_index_unknown_physical_dt',Native_contact_conditioned_gripper_scalar_masked_from_inputs=True,
  expert_targets_not_current_object_GT=True,expert_targets_not_handoff_tau=True,WM_predictions_are_pseudo_not_GT=True,
  raw_TEST_payload_reads=0,reserved_or_excluded_payload_reads=0,model_forward_calls=0,optimizer_steps=0,Native_started=False,training_started=False,
  all_task_network_trained=False,Native_task_evaluation_ready=False)
 write(out/'report.json',report)
 completion=dict(complete=True,method='current_network',stage=stage,source_unchanged=True,split_manifest=pointer(manifest),
  report=pointer(out/'report.json'),window_index=pointer(out/'window_index.json'),source_sha256=method_sources,
  actual_public_window_checks=len(checks),fit_windows=len(fit),selection_windows=len(val),
  raw_TEST_payload_reads=0,model_forward_calls=0,optimizer_steps=0,Native_started=False,training_started=False)
 write(out/'completion.json',completion)
 return completion


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest');p.add_argument('--output');p.add_argument('--tasks',nargs='*');p.add_argument('--image-size',type=int,choices=(64,128,256),default=256)
 mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare-manifest',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--preprocess',action='store_true');a=p.parse_args()
 if a.prepare_manifest:
  if not a.output:p.error('--prepare-manifest requires fresh output')
  value=prepare(a.output);result=dict(manifest=pointer(Path(a.output)/'manifest.json'),role_counts=value['role_counts'],overlap_counts=value['overlap_counts'],raw_TEST_payload_reads=0)
 else:
  if not a.manifest or not a.output:p.error('--check/--preprocess requires manifest and fresh output')
  result=run(a.manifest,a.output,'preprocess'if a.preprocess else'data_check',a.tasks,a.image_size)
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
