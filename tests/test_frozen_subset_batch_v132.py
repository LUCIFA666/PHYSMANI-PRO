from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from our_mani.multitask import frozen_subset_batch_v132 as batch
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.protocol import MultitaskProtocolError


class Tests(unittest.TestCase):
    def fixture(self,root):
        p=root/'a';p.write_text('original');q=root/'b';q.write_text('second')
        return p,q,{str(p):file_sha(p),str(q):file_sha(q)}

    def test_exact_sha_and_unique_canonical_subset(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);p,_,lock=self.fixture(root)
            self.assertEqual(batch.checked_subset(root,lock,{str(p):lock[str(p)]}),{str(p):lock[str(p)]})

    def test_subset_hash_mismatch_and_same_size_content_tamper_fail(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);p,_,lock=self.fixture(root)
            with self.assertRaises(MultitaskProtocolError):batch.checked_subset(root,lock,{str(p):'0'*64})
            p.write_text('tampered')
            with self.assertRaises(MultitaskProtocolError):batch.checked_subset(root,lock,{str(p):lock[str(p)]})

    def test_duplicate_canonical_alias_is_not_collapsed(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);p,_,lock=self.fixture(root);alias=root/'alias';alias.symlink_to(p)
            lock[str(alias)]=lock[str(p)]
            with self.assertRaises(MultitaskProtocolError):batch.checked_subset(root,lock,{str(p):lock[str(p)]})

    def test_changed_topology_is_rebuilt_without_retrying_a_native_job(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);p,q,lock=self.fixture(root);original=batch.checked_sha;changed=[]
            def check(path,expected):
                result=original(path,expected)
                if not changed:(root/'new_unrelated_name').write_text('new');changed.append(True)
                return result
            with patch.object(batch,'checked_sha',side_effect=check):
                self.assertEqual(batch.checked_subset(root,lock,lock),lock)

    def test_canonical_alias_added_during_batch_is_detected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);p,q,lock=self.fixture(root);alias=root/'alias';alias.symlink_to(q)
            lock[str(alias)]=lock[str(q)];original=batch.checked_sha;changed=[]
            def check(path,expected):
                result=original(path,expected)
                if not changed:alias.unlink();alias.symlink_to(p);changed.append(True)
                return result
            with patch.object(batch,'checked_sha',side_effect=check):
                with self.assertRaises(MultitaskProtocolError):batch.checked_subset(root,lock,{str(p):lock[str(p)]})


if __name__=='__main__':unittest.main()
