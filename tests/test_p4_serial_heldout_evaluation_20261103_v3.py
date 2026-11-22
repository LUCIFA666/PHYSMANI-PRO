"""V59 V3 source/assignment/admission contracts; no TEST materialization/Native."""
import ast
from copy import deepcopy
import importlib.util
import inspect,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
api=module('heldout_v3_independent',ROOT/'scripts/run_p4_serial_heldout_evaluation_20261103_v3.py')
old=module('heldout_v2_readonly_reference',ROOT/'scripts/run_p4_serial_heldout_evaluation_20261103_v2.py')
prep=module('heldout_v3_prepare_independent',ROOT/'scripts/prepare_p4_new_heldout_test_20261103_v3.py')

class V3Contracts(unittest.TestCase):
 def test_preassigned_new16_and_all48_prior_assignments_without_result_reads(self):
  p=api.fixed_partition(ROOT)
  refs={role:{'path':v['path'],'sha256':v['sha256']}for role,v in p['preassigned_partitions'].items()}
  parents,seeds,points,_=api.partitions(ROOT,dict(preassigned_partitions=refs))
  self.assertEqual(len(parents),48);self.assertEqual(len(seeds),48)
  self.assertEqual([j['seed']for j in p['jobs']],list(range(261103500,261103516)))
  self.assertEqual(api.Counter(j['scene']for j in p['jobs']),dict(static=2,uniform=8,speed_change=6))
  self.assertFalse(parents&{j['parent_id']for j in p['jobs']})
  self.assertFalse(seeds&{j['seed']for j in p['jobs']})
  self.assertFalse(points&{tuple(j['initial_xy'])for j in p['jobs']})
  self.assertEqual(prep.fixed_jobs(),p['jobs'])
  for name,digest in api.DERIVED_TOOLS.items():self.assertEqual(api.sha(ROOT/name),digest)
  self.assertEqual(api.sha(ROOT/api.ENTRY_GATE_PROTOCOL_PATH),api.ENTRY_GATE_PROTOCOL_SHA)

 def test_V59_flags_and_only_explicit_annotations_excluded(self):
  for k in('candidate_darkred_target_v59','candidate_goal_sphere_filter_v56','postconfirmed_position_maintenance_v56','finger_response_fastpath'):
   self.assertIs(api.PROTOCOL[k],True)
  a=dict(api.PROTOCOL,gripper_velocity=.2,P4_collection_metadata={'parent_id':'a','split':'val'})
  b=deepcopy(a);b['P4_collection_metadata']['parent_id']='b';self.assertEqual(api.behavior_configuration(a),api.behavior_configuration(b))
  for key,value in [('gripper_velocity',.3),('unreviewed_annotation','must remain behavior')]:
   b=deepcopy(a);b[key]=value;self.assertNotEqual(api.behavior_configuration(a),api.behavior_configuration(b))
  b=deepcopy(a);b['P4_collection_metadata']['hidden_controller_gain']=99
  with self.assertRaises(ValueError):api.behavior_configuration(b)

 def test_actual_wait_STOP_signal_execute_and_model_chain_preserved_from_V2(self):
  for name in('supervise_child','execute','requested_stop','propagate_stop','audit_exit','process_identity','verify_case_checkpoint_chain'):
   self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(api,name)))),ast.dump(ast.parse(inspect.getsource(getattr(old,name)))))
  a=inspect.getsource(api.evaluate_after_exit).replace('P4_newheldout_V59_modified_protocol_postexit_metrics_20261103_v3','P4_newheldout_modified_protocol_postexit_metrics_20261103_v2')
  self.assertEqual(ast.dump(ast.parse(a)),ast.dump(ast.parse(inspect.getsource(old.evaluate_after_exit))))

 def test_admission_requires_exact_root_path_allowTrue_and_three_bound_pointers(self):
  with TemporaryDirectory()as directory:
   root=Path(directory);file=root/api.ADMISSION_PATH;file.parent.mkdir(parents=True)
   val={'path':'config/framework/synthetic_devVAL.json','sha256':'a'*64};partition={'path':api.PARTITION_PATH,'sha256':api.PARTITION_SHA}
   protocol=root/api.ENTRY_GATE_PROTOCOL_PATH;protocol.write_text(json.dumps(dict(schema='P4_V59_devVAL_to_newheldout_entry_gate_protocol_v1',full_devVAL_plan=val['path'],new_TEST_partition=partition['path'])))
   record=dict(schema='P4_V59_devVAL_to_newheldout_admission_v1',allow_materialize_and_execute=True,
    devval_plan=val,heldout_partition=partition,entry_gate_protocol={'path':api.ENTRY_GATE_PROTOCOL_PATH,'sha256':api.sha(protocol)})
   def verify(r):
    file.write_text(json.dumps(r))
    with patch.object(api,'ENTRY_GATE_PROTOCOL_SHA',api.sha(protocol)):
     return api.verify_admission(root,{'path':api.ADMISSION_PATH,'sha256':api.sha(file)},devval_plan=val,heldout_partition=partition)
   self.assertTrue(verify(record)['exact_binding_verified'])
   for key,value in(('allow_materialize_and_execute',False),('allow_materialize_and_execute',1),('devval_plan',dict(val,sha256='b'*64)),('heldout_partition',dict(partition,sha256='c'*64))):
    bad=deepcopy(record);bad[key]=value
    with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):verify(bad)
   bad=deepcopy(record);bad['entry_gate_protocol']['sha256']='d'*64
   with self.assertRaises(RuntimeError):verify(bad)
   with self.assertRaises(ValueError):api.verify_admission(root,{'path':'arbitrary.json','sha256':'e'*64},devval_plan=val,heldout_partition=partition)

 def test_actual_pinned_predeclared_protocol_uses_path_strings_and_is_accepted(self):
  original=ROOT/api.ENTRY_GATE_PROTOCOL_PATH;declaration=api.load(original)
  self.assertIsInstance(declaration['full_devVAL_plan'],str);self.assertIsInstance(declaration['new_TEST_partition'],str)
  val={'path':declaration['full_devVAL_plan'],'sha256':api.sha(ROOT/declaration['full_devVAL_plan'])}
  partition={'path':api.PARTITION_PATH,'sha256':api.PARTITION_SHA}
  with TemporaryDirectory()as directory:
   root=Path(directory);protocol=root/api.ENTRY_GATE_PROTOCOL_PATH;protocol.parent.mkdir(parents=True)
   protocol.write_bytes(original.read_bytes())
   self.assertEqual(api.sha(protocol),api.ENTRY_GATE_PROTOCOL_SHA)
   record=dict(schema='P4_V59_devVAL_to_newheldout_admission_v1',allow_materialize_and_execute=True,
    devval_plan=val,heldout_partition=partition,entry_gate_protocol={'path':api.ENTRY_GATE_PROTOCOL_PATH,'sha256':api.ENTRY_GATE_PROTOCOL_SHA})
   path=root/api.ADMISSION_PATH;path.write_text(json.dumps(record))
   result=api.verify_admission(root,{'path':api.ADMISSION_PATH,'sha256':api.sha(path)},devval_plan=val,heldout_partition=partition)
   self.assertTrue(result['exact_binding_verified'])

 def test_current_generic_V59_role_normalization_exact_preassigned_identity_no_writes(self):
  valpath=ROOT/'config/framework/P4_student_closedloop_devVAL_v59_v1/plan.json';assignedpath=ROOT/api.PARTITION_PATHS['devval']
  val=api.load(valpath);assigned=api.load(assignedpath);before=api.sha(valpath)
  rows=api.normalized_v59_devval_jobs(ROOT,val,assigned)
  self.assertEqual(len(rows),8);self.assertTrue(all(j['split']=='val'for j in rows))
  self.assertEqual([j['parent_id']for j in rows],[j['parent_id']for j in assigned['jobs']])
  self.assertEqual(api.sha(valpath),before);self.assertNotIn('split',val)
  bad=deepcopy(val);bad['jobs'][-1]['id']='changed_assignment'
  with self.assertRaises(RuntimeError):api.normalized_v59_devval_jobs(ROOT,bad,assigned)

 def fake_prepare_api(self,root,*,missing):
  student=root/'config/framework/student.json';owner=root/api.FAMILY/'owner';runner=root/'runner.py'
  jobs=[dict(id=str(i),config=f'config/framework/val{i}.json',path=f'{api.FAMILY}/val{i}')for i in range(8)]
  partition={'jobs':[],'preassigned_partitions':{role:{'path':name,'sha256':'0'*64}for role,name in api.PARTITION_PATHS.items()}}
  documents={str(student):dict(schema=api.V59_DEVVAL_SCHEMA,jobs=jobs),str(owner/'inputs.json'):dict(plan=str(student),plan_sha256='0'*64,runner_sha256='0'*64),
   str(root/jobs[0]['config']):dict(api.PROTOCOL),str(root/jobs[0]['path']/'run_inputs.json'):dict(native_sources_sha256={},source_sha256={f'file{i}':'0'*64 for i in range(89)},information_setting=api.EXPECTED_INFORMATION),
   str(root/prep.MODEL):dict(files_sha256={})}
  for role,name in api.PARTITION_PATHS.items():documents[str(root/name)]={'jobs':[{}]*api.PARTITION_COUNTS[role]}
  fake=SimpleNamespace(**{name:getattr(api,name)for name in('PARTITION_PATH','PARTITION_SHA','PARTITION_PATHS','PROTOCOL','ADMISSION_PATH')},
   sha=lambda p:api.PARTITION_SHA if Path(p)==root/api.PARTITION_PATH else '0'*64,load=lambda p:deepcopy(documents[str(p)]),fixed_partition=lambda r:partition,
   partitions=lambda r,p:(set(),set(),set(),{}),check_map=lambda *a:None,behavior_configuration=lambda c:c,DERIVED_TOOLS={},check_native_sources=lambda *a:None,
   verify_devval=Mock(side_effect=FileNotFoundError('missing8')if missing=='eight'else None,return_value={'complete':True}),
   verify_admission=Mock(side_effect=FileNotFoundError('missing admission')))
  return fake,student,owner,runner

 def test_missing_eighth_exit_or_quality_admission_refuses_before_first_TEST_mkdir(self):
  for missing in('eight','admission'):
   with self.subTest(missing=missing),TemporaryDirectory()as directory:
    root=Path(directory);target=root/'config/framework/not_materialized';fake,student,owner,runner=self.fake_prepare_api(root,missing=missing)
    with patch.object(prep,'ROOT',root),patch.object(prep,'module',return_value=fake),patch.object(api.subprocess,'Popen',side_effect=AssertionError('Native forbidden')):
     with self.assertRaises(FileNotFoundError):prep.prepare(target,student_plan=student,devval_owner=owner,devval_runner=runner,
      partition_path=root/api.PARTITION_PATH,admission_path=root/api.ADMISSION_PATH,materialize=True)
    self.assertFalse(target.exists());self.assertEqual(list(root.iterdir()),[])
    if missing=='eight':fake.verify_admission.assert_not_called()

 def test_materializer_default_check_never_implicitly_materializes(self):
  self.assertIs(inspect.signature(prep.prepare).parameters['materialize'].default,False)
  args=['prepare_v3','--config-dir','/synthetic/config','--student-devval-plan','/synthetic/val','--devval-owner','/synthetic/owner','--devval-runner','/synthetic/runner']
  with patch.object(prep.sys,'argv',args),patch.object(prep,'prepare',return_value={'TEST_configs_materialized':False})as call:
   prep.main()
  self.assertIs(call.call_args.kwargs['materialize'],False)

if __name__=='__main__':unittest.main()
