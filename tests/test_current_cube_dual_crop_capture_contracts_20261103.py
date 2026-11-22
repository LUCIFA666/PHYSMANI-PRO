"""Pure capture/owner lifecycle contracts; no Native, renderer, model or HDF read."""
from pathlib import Path
from copy import deepcopy
import contextlib,importlib.util,io,json,signal,tempfile,unittest
from unittest.mock import patch
from contextlib import contextmanager,ExitStack
from types import SimpleNamespace
import numpy as np
from our_mani.local_mpc_20261103 import current_cube_dual_crop_capture_v1 as c
from our_mani.local_mpc_20261103 import depth_owner_position_v10_calibration8 as frozen
from scripts import collect_current_cube_dual_crop_20261103_v1 as cli
ROOT=Path(__file__).resolve().parents[1]

def host_module():
 p=ROOT/'scripts/launch_current_cube_dual_crop_20261103_v1.py'
 spec=importlib.util.spec_from_file_location('cf40_owner_contract',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class DoneProcess:
 pid=123456
 def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs;self.returncode=0;self.waits=0
 def poll(self):return self.returncode
 def wait(self,*args,**kwargs):self.waits+=1;return self.returncode

class RunningProcess(DoneProcess):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.returncode=None
 def wait(self,*args,**kwargs):self.waits+=1;self.returncode=-15;return self.returncode

class CaptureContracts(unittest.TestCase):
 def test_full_capture_body_writes_all13_inputs_and_private_labels_then_closes(self):
  # Explicit synthetic services exercise the real capture body/serialization.
  # No real renderer, Sphere fit, kinematic setter, qualification or Native is
  # claimed by this fixture; the exact real transaction is checked separately.
  from our_mani.local_mpc_20261102 import worker
  from our_mani.local_mpc_20261103 import depth_owner_capture_sensors,perception_capture_sensors
  from scripts import run_frozen_system_environment_v820_r10 as environment
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);output=root/c.FAMILY/'pilot'/'parent';plan_path=root/'plan.json';plan_path.write_text('{}')
   limits={'lower':[-2.]*7,'upper':[2.]*7};(root/'limits.json').write_text(json.dumps({'joint_limits':limits}));(root/'kin.json').write_text('{}');(root/'q.file').write_text('synthetic')
   parent=dict(parent_id='parent',seed=1,role='fit',fold='fit',source_task=c.SOURCE_TASKS[0],near_direction_ordinal=0,near_direction_unit_world=c.old.DIRECTIONS[0].tolist(),cube_rotation_degrees=0.,cube_rotation_axis='x',TRAIN_q_source={'joint_positions':[0.]*7,'source_parent_group':'group','file':{'path':'q.file'}})
   parent['shots']=c.layout.shot_specs_for_parent(parent)
   shots=[]
   for i,spec in enumerate(parent['shots']):
    x=deepcopy(spec);absent=i in (0,2);distance=spec['requested_distance_from_public_sphere_m'];position=[100.,100.,-100.]if absent else (np.asarray(parent['near_direction_unit_world'])*distance).tolist()
    x.update(materialized_cube_pose_world=position+[0.,0.,0.,1.],absence_from_all_camera_frusta_by_construction=absent,distance_from_initial_public_sphere_m=float(np.linalg.norm(position)),joint_positions=[0.]*7,finger_positions_m=[.04,.04]);shots.append(x)
   plan=dict(preparation_only=False,frozen_environment_sha256='test',source_sha256={},external_source_sha256={},native_config={},world_outside_position_m=[100.,100.,-100.],joint_limits={'path':'limits.json'},kinematics={'path':'kin.json'},goal_calibration={'path':'goal.json'})
   backend=SimpleNamespace(staging_receipt={},raw=SimpleNamespace(ledger=SimpleNamespace(ticks=0)),clock=lambda:0.,joint_limits=lambda:limits,close=lambda:None)
   camera=SimpleNamespace(get_near_clipping_plane=lambda:.01,get_far_clipping_plane=lambda:10.)
   bundle=SimpleNamespace(rgb_cameras={k:camera for k in c.CAMERAS},close=lambda:None)
   current={};transactions=[]
   @contextmanager
   def transaction(backend,shot,sphere):
    current['shot']=shot;receipt={'restored':False};transactions.append(receipt)
    try:yield receipt
    finally:receipt['restored']=True
   def capture(bundle,roles,output,shot_id):
    shot=current['shot'];pose=shot.get('materialized_cube_pose_world',shot.get('fixed_cube_pose_world'))
    arrays=dict(rgb=np.zeros((4,2,2,3),np.uint8),depth_m=np.ones((4,2,2),np.float32),camera_intrinsics=np.tile(np.eye(3),(4,1,1)),camera_to_world=np.tile(np.eye(4),(4,1,1)),camera_names=np.array(c.CAMERAS))
    annotations=dict(semantic_role_ids=np.zeros((4,2,2),np.uint8),cube_raw_positive=np.zeros((4,2,2),bool),without_cube_rgb=np.ones((4,2,2,3),np.float32),without_cube_depth_m=np.ones((4,2,2),np.float32))
    labels=dict(cube_pose_world=np.asarray(pose),cube_local_bbox=np.array([-.025,.025]*3))
    return arrays,annotations,{'all_contracts_explicitly_stubbed_for_serialization_test':True},labels
   with ExitStack()as stack:
    for owner,name,value in [(c,'checked_capture_plan',lambda *a:(plan,{},parent)),(c,'archive_capture_sources',lambda *a:None),(c,'pointer',lambda root,p:Path(root)/p['path']),(c,'clock',lambda b:(0.,0.,0.)),(c,'bootstrap_robot_diagnostics',lambda *a:{}),(c,'normalize_baseline_fingers',lambda b,q,receipt:receipt.update(normalized=True)),(c,'snapshot_pose',lambda b:{'cube_pose':np.array([0.,0.,0.,0.,0.,0.,1.])}),(c,'zero_clock_shot',transaction),(c,'capture_checked',capture),(c,'fit_public_sphere',lambda *a:({'center':np.zeros(3),'radius_m':.025},{})),(c,'PoEKinematics',lambda a:object()),(c,'materialize_dual_crop_shots',lambda *a:dict(complete=True,failures=[],shots=shots,outside_frusta_proof={'verified':True})),(worker,'native_source_snapshot',lambda cfg:{'native':'explicit-test-stub'}),(worker,'make_backend',lambda *a,**k:backend),(depth_owner_capture_sensors,'create_depth_owner_cameras',lambda *a,**k:bundle),(perception_capture_sensors,'resolve_annotation_roles',lambda b:({}, {},{})),(environment,'namespace_context',lambda a:{'test':True})]:
     stack.enter_context(patch.object(owner,name,value))
    stack.enter_context(patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'','DISPLAY':':explicit-nonnative-test'}))
    with contextlib.redirect_stdout(io.StringIO()):result=c.capture_parent(root,plan_path,'parent',output)
   self.assertTrue(result['complete']);self.assertTrue(result['native_closed']);self.assertEqual(len(result['shots']),13);self.assertEqual(len(transactions),14);self.assertTrue(all(t['restored']for t in transactions))
   self.assertEqual(len(list((output/'inputs').glob('*.npz'))),13)
   for shot_id in c.SHOT_IDS:
    with np.load(output/'inputs'/f'{shot_id}.npz',allow_pickle=False)as inputs:self.assertEqual(set(inputs.files),{'rgb','depth_m','K','T','camera_names','timestamp','public_sphere_center'})
    with np.load(output/'labels'/f'{shot_id}.npz',allow_pickle=False)as labels:self.assertIn('without_cube_rgb',labels);self.assertIn('cube_pose_world',labels)
   self.assertFalse(result['data_admitted_for_training']);self.assertFalse(result['oracle_center_evaluation_required_before_fit']);self.assertTrue(result['current_CF_supervision_and_visibility_validation_required_before_fit'])

 def test_old_transaction_and_camera_functions_reused_exactly(self):
  self.assertIs(c.zero_clock_shot,frozen.zero_clock_shot)
  for name in ('capture_checked','normalize_baseline_fingers','snapshot_pose','archive_capture_sources','public_payload'):
   self.assertIs(getattr(c,name),getattr(frozen,name))
  self.assertEqual(c.SHOT_IDS,tuple(f'shot{i:02d}'for i in range(13)))
  self.assertEqual(c.ROLE_COUNTS,dict(fit=16,selection=16,reserved_fresh_centre_VAL=8))

 def test_capture_branch_stdout_handles_nested_real_prediction_shaped_numpy(self):
  value=dict(complete=True,shots=[dict(q=np.arange(7,dtype=np.float64),cf=np.zeros((4,2,2),dtype=np.bool_),restored=np.bool_(True))])
  with patch.object(cli.producer,'capture_parent',return_value=value)as capture,patch.object(cli.producer,'check_capture_plan',side_effect=AssertionError('capture branch expected')):
   out=io.StringIO()
   with contextlib.redirect_stdout(out):self.assertEqual(cli.main(['--capture','--plan','plan.json','--parent-id','fixed-parent','--output','fresh']),0)
  parsed=json.loads(out.getvalue());self.assertEqual(parsed['shots'][0]['q'],list(range(7)));self.assertTrue(parsed['shots'][0]['restored']);capture.assert_called_once()
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(['--capture','--plan','plan.json'])

 def test_preparation_capture_refuses_before_output_or_native_import(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);output=root/'never-created'
   with patch.object(c,'checked_capture_plan',return_value=({'preparation_only':True},{},{})):
    with self.assertRaisesRegex(ValueError,'Preparation-only'):c.capture_parent(root,'plan','parent',output)
   self.assertFalse(output.exists())

 def test_scope_negative_is_labels_only_actual_distance_not_semantic_absence(self):
  with tempfile.TemporaryDirectory()as directory:
   output=Path(directory);(output/'labels').mkdir();(output/'labels/outside_frusta_proof.json').write_text('{}')
   p=dict(role='fit',source_task=c.SOURCE_TASKS[0],TRAIN_q_source={'source_parent_group':'TRAIN-group'},near_direction_ordinal=0,near_direction_unit_world=[1.,0.,0.])
   shot=dict(shot_id='shot01',condition='cube_outside_scope_nearhand',condition_detail='fixed',position_band='outside_scope_0p20',absence_from_all_camera_frusta_by_construction=False)
   meta=c.annotation_metadata(output,output,p,shot,'parent/shot01',{'center':[0.,0.,0.]},{'cube_pose_world':[.25,0.,0.,0.,0.,0.,1.]})
   self.assertEqual(meta['intended_scope_m'],.2);self.assertEqual(meta['construction'],'outside_scope');self.assertTrue(meta['cube_scene_present']);self.assertIsNone(meta['absence_proof'])
   proof=json.loads((output/meta['scope_negative_proof']['path']).read_text());self.assertFalse(proof['semantic_absence_claimed']);self.assertFalse(proof['model_input_allowed']);self.assertEqual(proof['distance_from_initial_public_sphere_m'],.25)
   with self.assertRaises(c.CaptureInvariantError):c.annotation_metadata(output,output,p,shot,'parent/shot01',{'center':[0.,0.,0.]},{'cube_pose_world':[.2,0.,0.,0.,0.,0.,1.]})

 def owner_fixture(self,root,m):
  (root/'scripts').mkdir();(root/'config').mkdir();family=root/'runtime';family.mkdir()
  shell=root/'scripts/run_current_cube_dual_crop_20261103_v1.sh';shell.write_text('not executed')
  host=root/'scripts/launch_current_cube_dual_crop_20261103_v1.py';host.write_text('test metadata')
  (root/'scripts/collect_current_cube_dual_crop_20261103_v1.py').write_text('not executed')
  plan=root/'config/plan.json';plan.write_text(json.dumps({'preparation_only':False}))
  return family,shell,host,plan

 def test_host_execute_complete_metadata_path_and_actual_wait(self):
  m=host_module()
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);family,shell,host,plan=self.owner_fixture(root,m);process=DoneProcess()
   with patch.multiple(m,ROOT=root,FAMILY=family,SHELL=shell,__file__=str(host)),patch.object(c,'check_capture_plan',return_value={'capture_ready':True}),patch.object(m.subprocess,'Popen',return_value=process)as popen,contextlib.redirect_stdout(io.StringIO()):
    rc=m.main(['--execute','--plan',str(plan),'--parent-id','fixed','--output',str(family/'case')])
   self.assertEqual(rc,0);self.assertEqual(process.waits,1);self.assertEqual(len(popen.call_args.args[0]),8)
   owner=json.loads((family/'case.owner.json').read_text());closed=json.loads((family/'case.owner.actual_exit.json').read_text())
   self.assertEqual(owner['parent_id'],'fixed');self.assertTrue(closed['actual_wait_completed']);self.assertEqual(closed['actual_process_returncode'],0);self.assertTrue(closed['source_unchanged']);self.assertFalse(closed['benchmark_success_claimed'])
   self.assertFalse((family/'case').exists())

 def test_owner_error_after_popen_still_terminates_reaps_and_preserves_failure(self):
  m=host_module()
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);family,shell,host,plan=self.owner_fixture(root,m);process=RunningProcess();write=m.write_new
   def failing(path,value):
    if str(path).endswith('.owner.json'):raise RuntimeError('explicit metadata-write failure')
    return write(path,value)
   with patch.multiple(m,ROOT=root,FAMILY=family,SHELL=shell,__file__=str(host)),patch.object(c,'check_capture_plan',return_value={'capture_ready':True}),patch.object(m.subprocess,'Popen',return_value=process),patch.object(m,'write_new',side_effect=failing),patch.object(m.os,'killpg')as kill:
    with self.assertRaisesRegex(RuntimeError,'metadata-write'):m.main(['--execute','--plan',str(plan),'--parent-id','fixed','--output',str(family/'case')])
   # Original owner waits once during cleanup, then reads the reaped status
   # with wait() again while writing its failure receipt. Neither is a retry.
   self.assertEqual(process.waits,2);kill.assert_called_once_with(process.pid,signal.SIGTERM)
   closed=json.loads((family/'case.owner.actual_exit.json').read_text());self.assertEqual(closed['actual_process_returncode'],-15);self.assertTrue(closed['actual_wait_completed']);self.assertTrue(closed['terminated_during_owner_cleanup']);self.assertFalse(closed['source_unchanged']);self.assertFalse(closed['data_admitted_for_training'])

if __name__=='__main__':unittest.main()
