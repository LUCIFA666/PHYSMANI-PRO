import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from our_mani.multitask import input_stat_cache_v79 as cache
from our_mani.multitask.protocol import MultitaskProtocolError


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=cache.ROOT/"runtime/framework/tmp");self.root=Path(self.tmp.name);self.file=self.root/'source';self.file.write_bytes(b'abcd');self.store=self.root/'cache';self.sha=hashlib.sha256(b'abcd').hexdigest()
    def tearDown(self):self.tmp.cleanup()
    def check(self):return cache.checked_sha(self.file,self.sha,cache=self.store)
    def test_reuses_only_after_actual_full_byte_hash(self):
        self.assertEqual(self.check(),(self.sha,False))
        with patch.object(cache,'file_sha',side_effect=AssertionError('unnecessary full hash')):self.assertEqual(self.check(),(self.sha,True))
    def test_same_size_mutation_with_restored_mtime_is_detected_by_ctime(self):
        self.check();old=self.file.stat();self.file.write_bytes(b'efgh');os.utime(self.file,ns=(old.st_atime_ns,old.st_mtime_ns))
        with self.assertRaises(MultitaskProtocolError):self.check()
    def test_atomic_replacement_of_identical_bytes_still_requires_new_hash(self):
        self.check();other=self.root/'replacement';other.write_bytes(b'abcd');other.replace(self.file)
        self.assertEqual(self.check(),(self.sha,False))
    def test_mutation_during_hash_cannot_commit_cache_record(self):
        original=cache.file_sha
        def mutate(path):
            result=original(path);self.file.write_bytes(b'efgh');return result
        with patch.object(cache,'file_sha',side_effect=mutate):
            with self.assertRaisesRegex(MultitaskProtocolError,'during byte hashing'):self.check()
        self.assertFalse(list(self.store.glob('*.json')))
    def test_different_expected_SHA_cannot_reuse_old_record_and_declared_map_is_preserved(self):
        self.check()
        with self.assertRaises(MultitaskProtocolError):cache.checked_sha(self.file,'0'*64,cache=self.store)
        stats={};result=cache.verify_inputs(self.root,{'source':self.sha},cache=self.store,stats=stats)
        self.assertEqual(result,{'source':self.sha});self.assertEqual(stats['strong_stat_reuses'],1)
    def test_scoped_installer_restores_on_error(self):
        from types import SimpleNamespace
        sentinel=lambda *a:None;e=SimpleNamespace(verify_inputs=sentinel)
        with self.assertRaises(RuntimeError):
            with cache.evaluator_scope(e):self.assertIs(e.verify_inputs,cache.verify_inputs);raise RuntimeError()
        self.assertIs(e.verify_inputs,sentinel)


if __name__=='__main__':unittest.main()
