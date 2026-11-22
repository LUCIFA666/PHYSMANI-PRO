"""Only the new teacher gate/freeze identity changes; numerical v3 stays fixed."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'scripts/compile_cube_teacher_candidate_dataset_20261104_v3.py'
NEW=ROOT/'scripts/compile_cube_teacher_candidate_dataset_20261104_v4.py'
REPLACEMENTS={
 'Cube_privileged_canonical_P1_contact_teacher_TRAIN12_20261104_v1':'Cube_private_exact_future_contact_teacher_TRAIN12_20261104_v2',
 '_teacherpass_v1':'_teacherpass_v2','cube_contact_teacher_TRAIN_20261104_v1':'cube_exact_future_teacher_TRAIN_20261104_v2',
 'Cube_privileged_contact_teacher_execution_freeze_v1':'Cube_private_exact_future_teacher_execution_freeze_v2',
 'Only the fixed privileged teacher v1 capture family is accepted':'Only the fixed exact-future teacher v2 capture family is accepted',
 'Cube_corrected_GRU_teacher_public_prediction_freeze_20261104_v3':'Cube_corrected_GRU_exact_teacher2_public_prediction_freeze_20261104_v4'}
class Replace(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):n.value=REPLACEMENTS.get(n.value,n.value)
  return n

def fn(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
class Teacher2Driver(unittest.TestCase):
 def test_numeric_replay_compile_and_normalization_unchanged(self):
  for name in ('predict_public','compile_contacts','checked','sha','read','write_rows'):
   self.assertEqual(ast.dump(Replace().visit(fn(OLD,name)),include_attributes=False),ast.dump(fn(NEW,name),include_attributes=False),name)
 def test_gate_only_declared_new_identities(self):
  for name in ('metadata','teacher_gate','validate_public_freeze'):
   self.assertEqual(ast.dump(Replace().visit(fn(OLD,name)),include_attributes=False),ast.dump(fn(NEW,name),include_attributes=False),name)
 def test_heldout_not_new_teacher_role_and_v3_numerical_module_reused(self):
  source=NEW.read_text();self.assertIn("choices=('fit','selection')",source)
  self.assertIn('from our_mani.cube_candidate_teacher_20261104_v3.data import',source)
  self.assertIn('from our_mani.cube_candidate_teacher_20261104_v3.replay import',source)
  self.assertNotIn("+'_teacherpass_v1'",source)
if __name__=='__main__':unittest.main()
