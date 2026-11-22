"""Small synthetic TRAIN fixtures only; no actual VAL, TEST, simulator or GPU."""
from pathlib import Path
import hashlib,importlib.util,json,os,tempfile,unittest
from unittest.mock import patch
from our_mani.multitask import formal_suite_admission_v728 as module
from our_mani.multitask.input_stat_cache_v79 import checked_sha as actual_checked_sha
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
from our_mani.multitask.protocol import MultitaskProtocolError

# Reuse the existing complete1600-parent/192-run/48-route synthetic fixture.
_spec=importlib.util.spec_from_file_location('_scientific_v718_fixture',Path(__file__).with_name('test_formal_optional_colour_v718.py'))
fixture_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(fixture_module)


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.data=self.root/'data';self.data.write_bytes(b'data')
        self.other=self.root/'other';self.other.write_bytes(b'else')
        self.files={str(self.data):file_sha(self.data),str(self.other):file_sha(self.other)}
        self.pointer=dict(path='data',sha256=file_sha(self.data))
        cache=self.root/'strong_stat_cache'
        self.sha_patch=patch.object(module,'checked_sha',side_effect=lambda path,sha:actual_checked_sha(path,sha,cache=cache))
        self.sha_patch.start();self.addCleanup(self.sha_patch.stop)

    def transaction(self,files=None):return module.AdmissionTransaction(self.root,self.files if files is None else files)

    def test_every_pointer_is_checked_without_per_pointer_full_graph_scan(self):
        original=module.FrozenInputIndex.valid;calls=[]
        def track(index):calls.append(index);return original(index)
        with patch.object(module.FrozenInputIndex,'valid',track):
            with self.transaction() as transaction:
                for _ in range(514):
                    self.assertEqual(transaction.locked_pointer(self.root,self.files,self.pointer),self.data)
        self.assertEqual(len(calls),3) # construction, start boundary, end boundary
        self.assertEqual(transaction.metrics['full_graph_boundary_checks'],2)
        self.assertEqual(transaction.metrics['pointer_checks'],514)

    def test_unrelated_output_creation_remains_allowed(self):
        with self.transaction() as transaction:
            for index in range(12):
                (self.root/('output'+str(index))).mkdir()
                self.assertEqual(transaction.locked_pointer(self.root,self.files,self.pointer),self.data)

    def test_persistent_unused_alias_retarget_is_rejected_at_end(self):
        alias=self.root/'unused_alias';alias.symlink_to(self.other)
        self.files[str(alias)]=file_sha(self.other)
        with self.assertRaises(module.InputTopologyChanged):
            with self.transaction() as transaction:
                transaction.locked_pointer(self.root,self.files,self.pointer)
                alias.unlink();alias.symlink_to(self.data)

    def test_input_map_key_and_value_changes_are_rejected(self):
        for mutate in (lambda files:files.update(unexpected='0'*64),
                       lambda files:files.update({str(self.other):'0'*64})):
            with self.subTest(mutation=mutate):
                files=dict(self.files)
                with self.assertRaises(module.InputTopologyChanged):
                    with self.transaction(files) as transaction:
                        transaction.locked_pointer(self.root,files,self.pointer);mutate(files)

    def test_used_declared_hash_change_is_rejected_immediately(self):
        with self.assertRaises(module.InputTopologyChanged):
            with self.transaction() as transaction:
                self.files[str(self.data)]='0'*64
                transaction.locked_pointer(self.root,self.files,self.pointer)

    def test_conflicting_canonical_names_cannot_select_an_input(self):
        alias=self.root/'alias';alias.symlink_to(self.data);self.files[str(alias)]=self.pointer['sha256']
        with self.assertRaisesRegex(MultitaskProtocolError,'conflicting'):
            with self.transaction() as transaction:
                transaction.locked_pointer(self.root,self.files,self.pointer)

    def test_used_symlink_retarget_during_byte_check_is_rejected(self):
        duplicate=self.root/'same_bytes';duplicate.write_bytes(b'data')
        alias=self.root/'used_alias';alias.symlink_to(self.data)
        files={str(alias):self.pointer['sha256']};pointer=dict(path=alias.name,sha256=self.pointer['sha256'])
        changed=[]
        def checking(path,sha):
            result=actual_checked_sha(path,sha,cache=self.root/'cache2')
            if not changed:alias.unlink();alias.symlink_to(duplicate);changed.append(True)
            return result
        with self.assertRaises(module.InputTopologyChanged):
            with self.transaction(files) as transaction,patch.object(module,'checked_sha',side_effect=checking):
                transaction.locked_pointer(self.root,files,pointer)

    def test_declared_parent_replacement_after_lookup_is_rejected(self):
        folder=self.root/'folder';folder.mkdir();path=folder/'entry';path.write_bytes(b'data')
        files={str(path):self.pointer['sha256']};pointer=dict(path='folder/entry',sha256=self.pointer['sha256'])
        with self.assertRaises(module.InputTopologyChanged):
            with self.transaction(files) as transaction:
                transaction.locked_pointer(self.root,files,pointer)
                folder.rename(self.root/'old_folder');folder.mkdir();(folder/'entry').write_bytes(b'data')

    def test_same_size_changed_bytes_with_restored_mtime_are_rejected_at_end(self):
        with self.assertRaisesRegex(MultitaskProtocolError,'frozen input SHA changed'):
            with self.transaction() as transaction:
                transaction.locked_pointer(self.root,self.files,self.pointer)
                stat=self.data.stat();self.data.write_bytes(b'xxxx')
                os.utime(self.data,ns=(stat.st_atime_ns,stat.st_mtime_ns))

    def test_unused_alias_changed_during_final_byte_checks_is_not_missed(self):
        alias=self.root/'unused_alias';alias.symlink_to(self.other);self.files[str(alias)]=file_sha(self.other)
        calls=[]
        def checking(path,sha):
            result=actual_checked_sha(path,sha,cache=self.root/'cache3');calls.append(path)
            if len(calls)==2:alias.unlink();alias.symlink_to(self.data)
            return result
        with patch.object(module,'checked_sha',side_effect=checking):
            with self.assertRaises(module.InputTopologyChanged):
                with self.transaction() as transaction:
                    transaction.locked_pointer(self.root,self.files,self.pointer)

    def test_JSON_parser_cannot_consume_bytes_different_from_its_frozen_SHA(self):
        path=self.root/'model.json';path.write_text('{"model":1}');pointer=dict(path=path.name,sha256=file_sha(path))
        files={str(path):pointer['sha256']};original=Path.read_bytes
        def substituted(current):
            return b'{"model":2}' if current==path else original(current)
        with self.assertRaisesRegex(MultitaskProtocolError,'parsed JSON bytes'):
            with self.transaction(files) as transaction,patch.object(Path,'read_bytes',substituted):
                transaction.locked_json(self.root,files,pointer)

    def test_scoped_scientific_reader_patch_is_restored_on_error(self):
        pointer_reader=module.scientific.locked_pointer;json_reader=module.scientific.locked_json
        suite=dict(methods=list(METHODS),tasks=list(TASKS))
        def failing(root,suite,files):
            self.assertNotEqual(module.scientific.locked_pointer,pointer_reader)
            module.scientific.locked_pointer(root,files,self.pointer)
            raise RuntimeError('scientific rejection')
        with patch.object(module.scientific,'admit',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'scientific rejection'):module.admit(self.root,suite,self.files)
        self.assertIs(module.scientific.locked_pointer,pointer_reader)
        self.assertIs(module.scientific.locked_json,json_reader)


class ScientificParityTests(fixture_module.OptionalColourTests):
    def setUp(self):
        selected=patch.object(fixture_module,'module',module);selected.start();self.addCleanup(selected.stop)

    def test_complete_scientific_admit_has_constant_full_graph_checks(self):
        fixture=self.fixture();original=module.FrozenInputIndex.valid;calls=[]
        def track(index):calls.append(index);return original(index)
        with patch.object(module.FrozenInputIndex,'valid',track):result=fixture.admit()
        self.assertTrue(result['complete']);self.assertEqual(len(calls),3)
        self.assertGreater(module.admission_metrics()['pointer_checks'],500)
        self.assertEqual(module.admission_metrics()['full_graph_boundary_checks'],2)

    def test_scientific_result_is_identical_to_original718(self):
        fixture=self.fixture(current_colour='pointer',accepted_colour='pointer')
        expected=module.scientific.admit(fixture.root,fixture.suite,fixture.files)
        self.assertEqual(fixture.admit(),expected)


if __name__=='__main__':unittest.main()
