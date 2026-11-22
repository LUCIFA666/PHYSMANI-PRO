"""Actual official RGBD HDF-to-causal-stage samples, without Native/contact input."""
from pathlib import Path
import json
import numpy as np
from .manifest import ROOT,SCHEMA,TASKS,BASE_TASKS,sha,pointer

CAMERAS=('front','8','16','36')
DEPTH_SCALE=float(2**24-1)
ROLE_LABELS={'beat_the_rotating_buzz':'wand','insert_onto_rotating_peg':'ring_not_peg','moving_basketball_in_hoop':'ball_only_second_hoop_dropped_by_packager','pick_moving_target_on_the_table':'Cube','place_cups_on_rotating_frame':'active_target_cup_identity_can_switch','push_moving_button':'target_button','put_rubbish_in_moving_bin':'bin_not_rubbish','remove_cups_from_rotating_frame':'active_target_cup_identity_can_switch'}


def source_map():
 paths=[Path(__file__).resolve(),Path(__file__).with_name('manifest.py'),Path(__file__).with_name('__init__.py'),ROOT/'scripts/prepare_current_network_multitask_data_20261104_v1.py',
  Path('/home/ubuntu/桌面/phsimani/PhysMani/packages/physmani_bench/physmani_bench/episode_io.py'),
  Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/backend/scene.py'),
  Path('/home/ubuntu/桌面/phsimani/PhysMani/packages/physmani_wm/physmani_wm/offline/run.py'),
  Path('/home/ubuntu/桌面/phsimani/PhysMani/packages/physmani_policy/physmani_policy/datasets/dataset_engine.py'),
  Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/backend/const.py')]
 return {str(p.resolve()):sha(p)for p in paths}


def read_manifest(path):
 p=Path(path).resolve(strict=True);m=json.loads(p.read_text())
 if m.get('schema')!=SCHEMA or m.get('complete')is not True:raise ValueError('Frozen complete16-task source manifest required')
 if any(sha(k)!=v for k,v in m['metadata_source_sha256'].items()):raise ValueError('Original metadata/source roles changed')
 if len(m['episodes'])!=2555 or {r['task']for r in m['episodes']}!=set(TASKS):raise ValueError('All1600TRAIN/315VAL/640TEST metadata required')
 return m


def check_stat(row):
 if row['split']=='test'or not row['offline_eligible']:raise ValueError('TEST/reserved/excluded/quarantined payload decode forbidden')
 p=Path(row['source_path']).resolve(strict=True);s=p.stat();expected=row['source_stat']
 if dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)!=expected:raise ValueError('Official HDF source changed')
 return p


def decode_depth(rgb,near,far):
 if rgb.dtype!=np.uint8 or rgb.shape[-1]!=3 or not 0<near<far:raise ValueError('Real24bit RGB depth and public clip planes required')
 a=rgb.astype(np.uint32);encoded=(a[...,0]<<16)|(a[...,1]<<8)|a[...,2]
 return (encoded.astype(np.float32)/DEPTH_SCALE*(far-near)+near).astype(np.float32)


def episode_window(row,current,*,history_frames=5,image_size=64,include_labels=True):
 if image_size not in (64,128,256)or history_frames!=5:raise ValueError('Exact original256 or actual stride2/4, history5 only')
 p=check_stat(row);current=int(current);nframes=row['frame_count']
 if not 0<=current<nframes:raise ValueError('Current recorded index outside actual episode')
 indices=np.arange(max(0,current-history_frames+1),current+1,dtype=np.int64);length=len(indices);stride=256//image_size
 rgbd=np.zeros((history_frames,4,4,image_size,image_size),np.float32);robot=np.zeros((history_frames,15),np.float32)
 K=[];T=[];clips=[]
 import h5py
 with h5py.File(p,'r')as h:
  for camera_index,name in enumerate(CAMERAS):
   k=np.asarray(h[f'cam_params/{name}/intrinsics'][()],np.float32);t=np.asarray(h[f'cam_params/{name}/extrinsics'][()],np.float32)
   near=float(h[f'cam_params/{name}/near'][()]);far=float(h[f'cam_params/{name}/far'][()])
   if k.shape!=(3,3)or t.shape!=(4,4)or not np.isfinite(k).all()or not np.isfinite(t).all()or abs(np.linalg.det(k))<1e-8 or not np.allclose(t[3],[0,0,0,1],rtol=0,atol=1e-6)or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),rtol=0,atol=1e-5):raise ValueError('Public signedK/camera-to-world calibration invalid')
   K.append(k);T.append(t);clips.append((near,far))
   for j,index in enumerate(indices):
    rgb=np.asarray(h[f'observations/{name}/rgb'][int(index)]);packed=np.asarray(h[f'observations/{name}/depth'][int(index)])
    if rgb.shape!=(256,256,3)or rgb.dtype!=np.uint8 or packed.shape!=(256,256,3):raise ValueError('Original4camera256 RGBD schema required')
    depth=decode_depth(packed,near,far)
    rgbd[j,camera_index,:3]=np.moveaxis(rgb[::stride,::stride].astype(np.float32)/127.5-1.,-1,0)
    rgbd[j,camera_index,3]=depth[::stride,::stride]
  q=np.asarray(h['low_dim_obs/joint_positions'][indices],np.float32);pose=np.asarray(h['low_dim_obs/gripper_pose'][indices],np.float32)
  if q.shape!=(length,7)or pose.shape!=(length,7)or not np.isfinite(q).all()or not np.isfinite(pose).all()or not np.allclose(np.linalg.norm(pose[:,3:],axis=-1),1.,rtol=0,atol=1e-3):raise ValueError('Recorded q/TCP finite7/xyzw quality gate failed')
  robot[:length,:7]=q;robot[:length,7:14]=pose
  # HDFgripper_open is Native-contact-conditioned by RLBench scene.py. It is
  # intentionally NEVER read into this public input; last channel stayszero.
  labels={}
  if include_labels:
   expert=np.asarray(h['low_dim_obs/expert_trajectory'][current],np.float32);target=np.asarray(h['low_dim_obs/expert_trajectory_target_position'][current],np.float32)
   stage=h['low_dim_obs/expert_trajectory_stage'][current,0];stage=stage.decode('utf-8')if isinstance(stage,bytes)else str(stage)
   if expert.shape!=(1,8)or target.shape!=(3,)or not np.isfinite(expert).all()or not np.isfinite(target).all():raise ValueError('Expert pose/action label schema invalid')
   family=row['task'].removesuffix('_high_speed')
   labels=dict(operation_class=BASE_TASKS.index(family),expert_command_xyzw_open=expert,expert_stage=stage,
    expert_debug_target_position=target,expert_debug_target_role=ROLE_LABELS[family],current_object_GT_pose_available=False,
    visibility_target=-1,centre_valid=False,command_issue_frame_index=current,
    command_target_prediction_time_s=None,source_nominal_prediction_lead_s=.5 if family in ('pick_moving_target_on_the_table','insert_onto_rotating_peg')else None,
    command_lead_not_handoff_tau=True,next_observed_frame_index=current+1 if current+1<nframes else None,
    next_observed_TCP_pose=None if current+1>=nframes else np.asarray(h['low_dim_obs/gripper_pose'][current+1],np.float32),
    next_physical_dt_s=None,physical_dt_valid=False,expert_and_future_labels_never_model_input=True)
 check_stat(row)
 if not np.isfinite(rgbd).all()or not np.isfinite(robot).all():raise ValueError('Finite public-stage inputs required')
 frame_indices=np.full(history_frames,-1,np.int64);frame_indices[:length]=indices
 mask=np.zeros((history_frames,15),bool);mask[:length,:14]=True
 return dict(rgbd=rgbd,robot_history=robot,lengths=np.int64(length),robot_observed_mask=mask,frame_indices=frame_indices,
  operation_label=BASE_TASKS.index(row['task'].removesuffix('_high_speed')),labels=labels,
  task=row['task'],source_parent_group=row['source_parent_group'],role=row['role'],split=row['split'],current_frame_index=current,
  physical_clock_verified=False,time_origin='recorded_frame_indices_unknown_physical_dt',
  calibration=dict(K=np.stack(K),T=np.stack(T),near_far=np.asarray(clips),original_resolution=256,stride=stride),
  gripper_open_input_masked_for_Native_contact_contamination=True)


class SemanticTaskHistoryDataset:
 def __init__(self,manifest,role='fit',history_frames=5,windows_per_parent=2,*,image_size=64,resolution=None,data_root=None,tasks=None):
  if resolution is not None:image_size=resolution
  if role not in ('fit','selection')or history_frames!=5 or windows_per_parent!=2 or image_size not in (64,128,256):raise ValueError('Predeclared fit/selection first+last2/history5/R64|128|256 only')
  self.path=Path(manifest).resolve(strict=True);self.metadata=read_manifest(self.path);self.role=role;self.history_frames=5;self.image_size=image_size;self.data_root=data_root
  chosen=set(TASKS if tasks is None else tasks)
  if not chosen<=set(TASKS):raise ValueError('Unknown task')
  self.episodes=[r for r in self.metadata['episodes']if r['task']in chosen and r['offline_eligible']and ((role=='fit'and r['split']=='train'and r['role']=='fit')or(role=='selection'and (r['role']=='selection'or r['split']=='validation')))]
  self.records=[]
  for episode_index,episode in enumerate(self.episodes):
   if episode['split']=='test'or episode['role']in ('reserved_fresh_centre_VAL','excluded_consumed_calibration'):raise ValueError('Protected payload cannot enter model dataset')
   for index in (0,episode['frame_count']-1):self.records.append(dict(episode=episode,episode_index=episode_index,current_frame_index=index))
  self.rows=self.records
 def __len__(self):return len(self.records)
 def __getitem__(self,index):
  record=self.records[index]
  return episode_window(record['episode'],record['current_frame_index'],image_size=self.image_size,include_labels=True)
 def public_episode(self,index,current_frame_index=0):return episode_window(self.episodes[index],current_frame_index,image_size=self.image_size,include_labels=False)
 def source_verification(self):
  read_manifest(self.path)
  for row in self.episodes:check_stat(row)
  return dict(source_unchanged=True,source_parents=len(self.episodes),samples=len(self),raw_TEST_payload_reads=0)
