from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest,hashlib,os
from our_mani.multitask.frozen_input_index_v93 import FrozenInputIndex,evaluator_scope
from our_mani.multitask import frozen_input_index_v93 as module
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.input_stat_cache_v79 import ROOT


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'runtime/framework/tmp');self.root=Path(self.tmp.name)
        self.path=self.root/'data';self.path.write_bytes(b'data');self.sha=hashlib.sha256(b'data').hexdigest();self.files={'data':self.sha}
    def tearDown(self):self.tmp.cleanup()
    def test_matches_original_single_canonical_alias_rule(self):
        index=FrozenInputIndex(self.root,self.files);self.assertEqual(index.checked(self.path),self.sha)
        link=self.root/'alias';link.symlink_to(self.path)
        with self.assertRaisesRegex(MultitaskProtocolError,'refresh'):index.checked(self.path)
        duplicate=FrozenInputIndex(self.root,dict(self.files,alias=self.sha))
        with self.assertRaisesRegex(MultitaskProtocolError,'conflicting'):duplicate.checked(self.path)
    def test_symlink_retarget_invalidates_index_and_detects_new_duplicate_alias(self):
        other=self.root/'other';other.write_bytes(b'else');link=self.root/'alias';link.symlink_to(other)
        files=dict(self.files,alias=hashlib.sha256(b'else').hexdigest());index=FrozenInputIndex(self.root,files)
        link.unlink();link.symlink_to(self.path);self.assertFalse(index.valid())
        with self.assertRaises(MultitaskProtocolError):FrozenInputIndex(self.root,files).checked(self.path)
    def test_changed_current_expected_value_is_not_hidden_by_index(self):
        index=FrozenInputIndex(self.root,self.files);self.files['data']='0'*64
        with self.assertRaises(MultitaskProtocolError):index.checked(self.path)
    def test_input_mutation_with_restored_mtime_is_rejected(self):
        index=FrozenInputIndex(self.root,self.files);index.checked(self.path);s=self.path.stat()
        self.path.write_bytes(b'xxxx');os.utime(self.path,ns=(s.st_atime_ns,s.st_mtime_ns))
        with self.assertRaises(MultitaskProtocolError):index.checked(self.path)
    def test_index_rebuilds_for_changed_declared_keys_and_scope_restores(self):
        sentinel=lambda *a:None;e=SimpleNamespace(ROOT=self.root,locked_digest=sentinel,verify_inputs=sentinel)
        with evaluator_scope(e):
            self.assertEqual(e.locked_digest(self.path,self.files),self.sha)
            self.files.clear();self.files[str(self.path)]=self.sha
            self.assertEqual(e.locked_digest(self.path,self.files),self.sha)
        self.assertIs(e.locked_digest,sentinel);self.assertIs(e.verify_inputs,sentinel)
    def test_hot_lookup_does_not_resolve_every_declared_path_again(self):
        index=FrozenInputIndex(self.root,self.files);original=Path.resolve;calls=[]
        def counted(path,*a,**kw):calls.append(str(path));return original(path,*a,**kw)
        with patch.object(Path,'resolve',counted):self.assertEqual(index.checked(self.path),self.sha)
        self.assertLessEqual(len(calls),2)
    def test_scope_restores_after_exception(self):
        sentinel=lambda *a:None;e=SimpleNamespace(ROOT=self.root,locked_digest=sentinel,verify_inputs=sentinel)
        with self.assertRaises(RuntimeError):
            with evaluator_scope(e):raise RuntimeError()
        self.assertIs(e.locked_digest,sentinel);self.assertIs(e.verify_inputs,sentinel)

    def test_directory_change_between_lookup_and_checked_rebuilds_exact_index(self):
        sentinel=lambda *a:None;e=SimpleNamespace(ROOT=self.root,locked_digest=sentinel,verify_inputs=sentinel)
        attempts=[]
        class ConcurrentOnce(FrozenInputIndex):
            def checked(index,path):
                attempts.append(path)
                if len(attempts)==1:
                    (self.root/'unrelated_output').mkdir()
                return super().checked(path)
        with patch.object(module,'FrozenInputIndex',ConcurrentOnce),evaluator_scope(e):
            self.assertEqual(e.locked_digest(self.path,self.files),self.sha)
        self.assertEqual(len(attempts),2)

    def test_continuously_changing_topology_fails_after_three_validation_checks(self):
        sentinel=lambda *a:None;e=SimpleNamespace(ROOT=self.root,locked_digest=sentinel,verify_inputs=sentinel)
        with patch.object(module,'FrozenInputIndex',side_effect=module.InputTopologyChanged('changed')) as ctor:
            with evaluator_scope(e),self.assertRaisesRegex(MultitaskProtocolError,'three index checks'):
                e.locked_digest(self.path,self.files)
        self.assertEqual(ctor.call_count,3)

    def test_byte_hash_mismatch_is_never_retried_as_topology(self):
        sentinel=lambda *a:None;e=SimpleNamespace(ROOT=self.root,locked_digest=sentinel,verify_inputs=sentinel)
        with patch.object(module,'checked_sha',side_effect=MultitaskProtocolError('changed bytes')) as check:
            with evaluator_scope(e),self.assertRaisesRegex(MultitaskProtocolError,'changed bytes'):
                e.locked_digest(self.path,self.files)
        self.assertEqual(check.call_count,1)


if __name__=='__main__':unittest.main()
