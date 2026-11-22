"""V3 capture-only deltas and immutable forty-parent assignments; zero rendering."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'config/framework/current_cube_dual_crop_CF_pilot40_20261103_v1/plan.json'
NEW=ROOT/'config/framework/current_cube_dual_crop_CF_pilot40_cap12_20261104_v3/plan.json'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def functions(path):
 return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


class V3SourceContracts(unittest.TestCase):
 def test_old_graph_immutable_new_graph_appends_and_old_data_not_mixed(self):
  old,new=read(OLD),read(NEW)
  self.assertTrue(set(old['source_sha256'])<=set(new['source_sha256']))
  self.assertEqual(new['external_source_sha256'],old['external_source_sha256'])
  self.assertEqual(len(new['external_source_sha256']),12)
  for name,digest in old['source_sha256'].items():
   self.assertEqual(new['source_sha256'][name],digest)
   self.assertEqual(sha(ROOT/name),digest)
  self.assertNotEqual(old['output_root'],new['output_root'])
  self.assertTrue(new['preparation_only']);self.assertFalse(new['execution_authorized']);self.assertFalse(new['capture_ready'])
  self.assertNotIn('scripts/prepare_current_cube_dual_crop_TRAIN_manifest_20261104_v2.py',new['source_sha256'])

 def test_all40_cube_and_source_recipes_unchanged_only_hand_slots_change(self):
  old,new=read(OLD),read(NEW)
  self.assertEqual(len(new['parents']),40)
  for a,b in zip(old['parents'],new['parents']):
   self.assertEqual({k:v for k,v in a.items()if k!='shots'},{k:v for k,v in b.items()if k!='shots'})
   for i,(x,y)in enumerate(zip(a['shots'],b['shots'])):
    self.assertEqual({k:v for k,v in x.items()if k not in ('hand_layout','condition_detail')},{k:v for k,v in y.items()if k not in ('hand_layout','condition_detail')})
    if i not in (1,10,12):self.assertEqual(x,y)
    else:self.assertEqual(y['hand_layout'],'offset_hand_asset_reuse_d120_pad_full_open')
  self.assertEqual(new['hand_pad_radius_cap_m'],.12)
  profile=read(ROOT/new['layout_profile']['path'])
  self.assertEqual(profile['schema'],'current_cube_dual_crop_fixed13_handcap120_profile_20261104_v3')
  self.assertEqual(profile['far_complete_asset_reference_shot_id'],'shot08')
  self.assertEqual(profile['planned_slot_target_indices'],[2,0,1,2,2,2])
  self.assertEqual(profile['layout_source']['sha256'],'b9cd0dd915465e3e457ae92a9cf431a8cf07348782751cb68dd570ca2bf79010')

 def test_capture_body_annotation_and_compiler_math_exact_old(self):
  base=ROOT/'our_mani/local_mpc_20261103'
  old,new=(functions(base/f'current_cube_dual_crop_capture_v{i}.py')for i in (2,3))
  for name in ('capture_parent','annotation_metadata','stop_paths','stop_requested','check_capture_plan'):
   self.assertEqual(old[name],new[name],name)
  oldc=functions(ROOT/'scripts/prepare_current_cube_dual_crop_TRAIN_manifest_20261103_v1.py')
  newc=functions(ROOT/'scripts/prepare_current_cube_dual_crop_TRAIN_manifest_20261104_v2.py')
  for name in oldc:
   self.assertEqual(oldc[name],newc[name].replace('manifest_compile_receipt_v2','manifest_compile_receipt_v1'),name)


if __name__=='__main__':unittest.main()
