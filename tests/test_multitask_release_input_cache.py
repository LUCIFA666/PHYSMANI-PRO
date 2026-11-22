"""CPU file-integrity boundaries for one release consumer's input closure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import release_preflight as gate
from our_mani.multitask.evaluation import file_sha


class ReleaseInputCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'evidence.bin';self.path.write_bytes(b'original raw evidence')
        self.sha=file_sha(self.path)

    def tearDown(self):self.temp.cleanup()

    def test_same_consumer_reuses_only_verified_unchanged_file_and_finally_rehashes(self):
        inputs=gate._Inputs(self.root)
        with patch.object(gate,'file_sha',wraps=file_sha) as hashing:
            for _ in range(25):inputs.include({self.path.name:self.sha})
            self.assertEqual(hashing.call_count,1)
            inputs.rehash_all()
            self.assertEqual(hashing.call_count,2)
        self.assertEqual(inputs.files,{self.path.name:self.sha})

    def test_changed_expected_sha_is_rejected_even_if_stat_is_unchanged(self):
        inputs=gate._Inputs(self.root);inputs.file(self.path,self.sha)
        with patch.object(gate,'file_sha',wraps=file_sha) as hashing:
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                inputs.file(self.path,'f'*64)
            self.assertEqual(hashing.call_count,0)

    def test_changed_bytes_rehash_and_reject_in_same_consumer(self):
        inputs=gate._Inputs(self.root);inputs.file(self.path,self.sha)
        self.path.write_bytes(b'changed raw evidence')
        with patch.object(gate,'file_sha',wraps=file_sha) as hashing:
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                inputs.file(self.path,self.sha)
            self.assertEqual(hashing.call_count,1)

    def test_changed_bytes_without_expected_do_not_replace_original_digest(self):
        inputs=gate._Inputs(self.root);inputs.file(self.path)
        self.path.write_bytes(b'changed raw evidence')
        with self.assertRaisesRegex(ValueError,'changed while auditing'):inputs.file(self.path)
        self.assertEqual(inputs.files[self.path.name],self.sha)

    def test_new_consumer_cannot_reuse_previous_consumer_hash(self):
        with patch.object(gate,'file_sha',wraps=file_sha) as hashing:
            gate._Inputs(self.root).file(self.path,self.sha)
            gate._Inputs(self.root).file(self.path,self.sha)
            self.assertEqual(hashing.call_count,2)

    def test_end_boundary_directly_hashes_even_if_stat_claims_no_change(self):
        inputs=gate._Inputs(self.root);inputs.file(self.path,self.sha)
        signature=inputs._path_stats[self.path.name]
        self.path.write_bytes(b'changed raw evidence')
        # This deliberately simulates unreliable metadata; final content SHA
        # must still reject it rather than trusting the scoped stat shortcut.
        with patch.object(inputs,'_stat_signature',return_value=signature):
            inputs.file(self.path,self.sha)
            with self.assertRaisesRegex(ValueError,'SHA differs'):inputs.rehash_all()

    def test_change_during_direct_hash_is_rejected(self):
        inputs=gate._Inputs(self.root)
        def changed(path):
            checksum=file_sha(path)
            path.write_bytes(b'changed during hash')
            return checksum
        with patch.object(gate,'file_sha',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'changed while hashing'):
                inputs.file(self.path,self.sha)
        self.assertFalse(inputs.files)

    def test_removed_input_is_rejected_at_final_boundary(self):
        inputs=gate._Inputs(self.root);inputs.file(self.path,self.sha);self.path.unlink()
        with self.assertRaisesRegex(ValueError,'input missing'):inputs.rehash_all()

    def test_formal_audit_calls_force_final_boundary_even_after_component_failure(self):
        with patch.object(gate._Inputs,'rehash_all',autospec=True) as finalize:
            result=gate.audit_formal_model_coverage(self.root,{},context_training_view={})
        self.assertFalse(result['formal_model_coverage_passed']);finalize.assert_called_once()

    def snapshot(self,*,owner=None,archive_bytes=None):
        archive=self.root/'archived.bin';archive.write_bytes(self.path.read_bytes() if archive_bytes is None else archive_bytes)
        row=dict(live_source=dict(path=self.path.name,sha256=file_sha(self.path)),
            archived_source=dict(path=archive.name,sha256=file_sha(archive)))
        snapshot=self.root/'source_snapshot.json';snapshot.write_text(json.dumps(dict(sources={owner or self.path.name:row})))
        return dict(path=snapshot.name,sha256=file_sha(snapshot)),archive

    def test_declared_source_snapshot_binds_current_bytes_and_immutable_archive(self):
        pointer,archive=self.snapshot();inputs=gate._Inputs(self.root)
        inputs.source_snapshot(pointer);inputs.rehash_all()
        self.assertEqual(set(inputs.files),{self.path.name,archive.name,pointer['path']})

    def test_declared_source_snapshot_rejects_mutated_current_or_archive_bytes(self):
        for field in ('current','archive'):
            self.path.write_bytes(b'original raw evidence');pointer,archive=self.snapshot()
            (self.path if field=='current' else archive).write_bytes(b'tampered source')
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'SHA differs'):
                gate._Inputs(self.root).source_snapshot(pointer)

    def test_declared_source_snapshot_rejects_wrong_owner_or_different_archived_code(self):
        pointer,_=self.snapshot(owner='another_source.py')
        with self.assertRaisesRegex(ValueError,'live/archive owner differs'):
            gate._Inputs(self.root).source_snapshot(pointer)
        pointer,_=self.snapshot(archive_bytes=b'different archived implementation')
        with self.assertRaisesRegex(ValueError,'live/archive owner differs'):
            gate._Inputs(self.root).source_snapshot(pointer)


if __name__=='__main__':unittest.main()
