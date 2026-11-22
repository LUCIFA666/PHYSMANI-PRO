from pathlib import Path
from unittest.mock import patch
import json,unittest
from our_mani.multitask import formal_suite_admission_v482 as module
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def test_actual_incomplete_TRAIN_suite_cannot_read_any_formal_source(self):
  def pointer(name):
   p=ROOT/name;return dict(path=name,sha256=file_sha(p))
  routes=pointer('artifacts/framework/multitask_r10/executor_suite_routes_v467/routes.json')
  readiness=pointer('artifacts/framework/multitask_r10/full16_fixed_four_readiness_v468/report.json')
  files={str(ROOT/p['path']):p['sha256'] for p in (routes,readiness)}
  suite=dict(methods=list(METHODS),tasks=list(TASKS),routes=routes,readiness=readiness)
  with self.assertRaisesRegex(ValueError,'complete unchanged full16x4x3'):
   module.read_formal_source(ROOT,suite,files,'validation')
 def test_TEST_release_is_required_before_TEST_pointer_or_bytes_access(self):
  # No source_manifests key exists: any premature pointer access would fail.
  suite={}
  with patch.object(module,'admit',return_value={'routes':{}}),patch.object(module,'locked_json') as reader:
   with self.assertRaisesRegex(ValueError,'VAL release'):
    module.read_formal_source(ROOT,suite,{},'test')
   reader.assert_not_called()
 def test_old_five_method_suite_rejected_before_any_artifact_is_read(self):
  with patch.object(module,'locked_json') as reader:
   with self.assertRaisesRegex(ValueError,'three authorized'):
    module.admit(ROOT,{'methods':['cv_geometry','cv_value','context_geometry','context_value','physmani'],'tasks':list(TASKS)}, {})
   reader.assert_not_called()
if __name__=='__main__':unittest.main()
