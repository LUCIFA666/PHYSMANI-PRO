from pathlib import Path
import hashlib,tempfile,unittest
from our_mani.multitask import evaluation
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask import source_verification_cache_v12 as helper
class TestSourceVerificationCache(unittest.TestCase):
 def test_frozen_implementation_required_and_late_content_change_still_fails_gate(self):
  root=Path(__file__).resolve().parents[1]
  with self.assertRaises(MultitaskProtocolError):helper.install(root,{})
  files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(helper.__file__).resolve(),root/'our_mani/multitask/audit_sha_cache_v14.py']}
  helper.install(root,files)
  with tempfile.TemporaryDirectory(dir=root/'runtime/framework/tmp') as tmp:
   p=Path(tmp)/'source';p.write_bytes(b'old');declared={str(p):hashlib.sha256(b'old').hexdigest()}
   evaluation.verify_inputs(root,declared);evaluation.verify_inputs(root,declared)
   p.write_bytes(b'new')
   with self.assertRaises(MultitaskProtocolError):evaluation.verify_inputs(root,declared)
  r=helper.receipt();self.assertFalse(r['policy_state_cached']);self.assertFalse(r['policy_commands_changed']);self.assertFalse(r['physics_or_camera_queries_added'])
if __name__=='__main__':unittest.main()
