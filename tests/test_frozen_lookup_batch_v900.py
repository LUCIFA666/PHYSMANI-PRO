"""Temporary CPU files only. Never import preflight/native or open real locks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import frozen_lookup_batch_v900 as module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'project'
        self.root.mkdir()
        self.cache = self.base / 'cache'
        self.a = self.root / 'a.json'
        self.a.write_bytes(b'{"a":1}')
        self.b = self.root / 'b'
        self.b.write_bytes(b'else')
        self.files = {'a.json': sha(self.a), 'b': sha(self.b)}
        self.pointer = dict(path='a.json', sha256=sha(self.a))

    def batch(self):
        return module.LookupBatch(self.root, cache=self.cache)


class BatchTests(Fixture, unittest.TestCase):

    def test_constructor_is_inert_and_batch_is_single_use(self):
        batch = self.batch()
        self.assertFalse(self.cache.exists())
        with self.assertRaises(RuntimeError):
            batch.checked_external(self.root, self.files, self.a)
        with batch:
            batch.checked_external(self.root, self.files, self.a)
        self.assertTrue(batch.metrics['closed'])
        self.assertTrue(batch.metrics['complete'])
        json.dumps(batch.metrics)
        with self.assertRaises(RuntimeError):
            batch.__enter__()
        with self.assertRaises(RuntimeError):
            batch.locked_pointer(self.root, self.files, self.pointer)

    def test_many_lookups_have_constant_complete_graph_checks(self):
        original = module.FrozenInputIndex.valid
        calls = []
        def tracked(index):
            calls.append(index)
            return original(index)
        with patch.object(module.FrozenInputIndex, 'valid', tracked):
            with self.batch() as batch:
                for _ in range(200):
                    self.assertEqual(batch.locked_pointer(self.root, self.files, self.pointer), self.a)
        self.assertEqual(len(calls), 4)  # two construction checks plus two boundaries
        self.assertEqual(batch.metrics['full_graph_boundary_checks'], 2)
        self.assertEqual(batch.metrics['pointer_checks'], 200)
        self.assertEqual(batch.metrics['used_path_end_checks'], 1)
        self.assertEqual(batch.metrics['actual_SHA_checks'], 201)
        self.assertGreater(batch.metrics['strong_stat_reuses'], 0)

    def test_maps_with_same_contents_keep_distinct_identity_and_snapshot(self):
        copied = dict(self.files)
        with self.batch() as batch:
            batch.checked_external(self.root, self.files, self.a)
            batch.checked_external(self.root, copied, self.a)
        self.assertEqual(batch.metrics['maps'], 2)
        self.assertEqual(batch.metrics['full_graph_boundary_checks'], 4)
        self.assertEqual(batch.metrics['declared_files'], 4)

    def test_root_cannot_be_changed(self):
        with self.assertRaisesRegex(module.MultitaskProtocolError, 'root differs'):
            with self.batch() as batch:
                batch.checked_external(self.base, self.files, self.a)

    def test_all_declared_paths_build_graph_even_when_unused(self):
        self.files['missing-unused'] = '0' * 64
        with self.assertRaises(FileNotFoundError):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)

    def test_external_dependencies_are_allowed_only_by_external_api(self):
        external = self.base / 'external.json'
        external.write_bytes(b'{"external":true}')
        files = {str(external): sha(external)}
        with self.batch() as batch:
            self.assertEqual(batch.checked_external(self.root, files, external), external)
            self.assertEqual(batch.read_json(self.root, files, external), {'external': True})
        self.assertGreater(batch.metrics['used_external'], 0)
        for api in ('locked_pointer', 'locked_json'):
            with self.subTest(api=api), self.assertRaisesRegex(module.MultitaskProtocolError, 'inside this project'):
                with self.batch() as batch:
                    getattr(batch, api)(self.root, files, dict(path=str(external), sha256=sha(external)))

    def test_project_symlink_to_external_cannot_bypass_pointer_boundary(self):
        external = self.base / 'external'
        external.write_bytes(b'outside')
        alias = self.root / 'alias'
        alias.symlink_to(external)
        files = {'alias': sha(external)}
        with self.assertRaisesRegex(module.MultitaskProtocolError, 'inside this project'):
            with self.batch() as batch:
                batch.locked_pointer(self.root, files, dict(path='alias', sha256=sha(external)))

    def test_undeclared_file_and_json_are_not_implicitly_authorized(self):
        unknown = self.root / 'unknown'
        unknown.write_bytes(b'{}')
        for api in ('checked_external', 'read_json'):
            with self.subTest(api=api), self.assertRaisesRegex(module.MultitaskProtocolError, 'missing or conflicting'):
                with self.batch() as batch:
                    getattr(batch, api)(self.root, self.files, unknown)

    def test_pointer_sha_must_be_explicit_lowercase_and_exact(self):
        for api in ('locked_pointer', 'locked_json'):
            for value in (None, '', 123, 'A' * 64, 'z' * 64, '0' * 64):
                pointer = dict(self.pointer, sha256=value)
                with self.subTest(api=api, value=value), self.assertRaises(module.MultitaskProtocolError):
                    with self.batch() as batch:
                        getattr(batch, api)(self.root, self.files, pointer)
            for pointer in ({'path': 'a.json'}, dict(self.pointer, extra=True)):
                with self.subTest(api=api, pointer=pointer), self.assertRaises(module.MultitaskProtocolError):
                    with self.batch() as batch:
                        getattr(batch, api)(self.root, self.files, pointer)

    def test_symlink_and_lexical_alias_canonical_duplicates_are_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.a)
        folder = self.root / 'folder'
        folder.mkdir()
        for name in ('alias', 'folder/../a.json', str(self.a)):
            files = dict(self.files, **{name: self.pointer['sha256']})
            with self.subTest(name=name), self.assertRaisesRegex(module.MultitaskProtocolError, 'conflicting'):
                with self.batch() as batch:
                    batch.locked_pointer(self.root, files, self.pointer)

    def test_map_add_remove_and_value_changes_fail_at_boundary(self):
        mutations = (lambda files: files.update(unused='0' * 64),
                     lambda files: files.pop('b'),
                     lambda files: files.update(b='0' * 64))
        for mutation in mutations:
            files = dict(self.files)
            with self.subTest(mutation=mutation), self.assertRaises(module.InputTopologyChanged):
                with self.batch() as batch:
                    batch.checked_external(self.root, files, self.a)
                    mutation(files)
            self.assertFalse(batch.metrics['complete'])

    def test_used_sha_value_change_fails_immediately(self):
        with self.assertRaisesRegex(module.InputTopologyChanged, 'SHA changed'):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                self.files['a.json'] = '0' * 64
                batch.checked_external(self.root, self.files, self.a)

    def test_swallowed_lookup_rejection_cannot_rebaseline_batch(self):
        with self.assertRaises(module.MultitaskProtocolError):
            with self.batch() as batch:
                try:
                    batch.locked_pointer(self.root, self.files, dict(self.pointer, sha256=None))
                except module.MultitaskProtocolError:
                    pass
                with self.assertRaisesRegex(module.MultitaskProtocolError, 'previously rejected'):
                    batch.checked_external(self.root, self.files, self.a)
        self.assertFalse(batch.metrics['complete'])

    def test_same_size_content_change_with_restored_mtime_fails_real_SHA(self):
        with self.assertRaisesRegex(module.MultitaskProtocolError, 'frozen input SHA changed'):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                previous = self.a.stat()
                self.a.write_bytes(b'{"a":2}')
                os.utime(self.a, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertFalse(batch.metrics['complete'])

    def test_same_path_same_bytes_new_inode_is_rejected(self):
        replacement = self.root / 'replacement'
        replacement.write_bytes(self.a.read_bytes())
        with self.assertRaises(module.InputTopologyChanged):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                replacement.replace(self.a)

    def test_used_file_disappearance_fails_close(self):
        with self.assertRaises(FileNotFoundError):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                self.a.unlink()

    def test_declared_parent_directory_replacement_is_rejected(self):
        folder = self.root / 'folder'
        folder.mkdir()
        child = folder / 'child'
        child.write_bytes(b'child')
        files = {'folder/child': sha(child)}
        with self.assertRaises(module.InputTopologyChanged):
            with self.batch() as batch:
                batch.checked_external(self.root, files, child)
                folder.rename(self.root / 'old-folder')
                folder.mkdir()
                (folder / 'child').write_bytes(b'child')

    def test_unused_declared_symlink_retarget_is_rejected(self):
        alias = self.root / 'unused'
        alias.symlink_to(self.b)
        self.files['unused'] = sha(self.b)
        with self.assertRaises(module.InputTopologyChanged):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                alias.unlink()
                alias.symlink_to(self.a)

    def test_unused_resolved_external_target_ancestor_is_watched(self):
        folder = self.base / 'external-folder'
        folder.mkdir()
        child = folder / 'child'
        child.write_bytes(b'child')
        alias = self.root / 'unused'
        alias.symlink_to(child)
        self.files['unused'] = sha(child)
        with self.assertRaises(module.InputTopologyChanged):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                folder.rename(self.base / 'old-external')
                folder.mkdir()
                (folder / 'child').write_bytes(b'child')

    def test_retarget_during_checked_SHA_is_rejected(self):
        other = self.root / 'same-bytes'
        other.write_bytes(self.a.read_bytes())
        alias = self.root / 'alias'
        alias.symlink_to(self.a)
        files = {'alias': sha(self.a)}
        original = module.checked_sha
        def change(path, expected, **kwargs):
            result = original(path, expected, **kwargs)
            if alias.readlink() == self.a:
                alias.unlink()
                alias.symlink_to(other)
            return result
        with self.assertRaises(module.InputTopologyChanged):
            with self.batch() as batch, patch.object(module, 'checked_sha', change):
                batch.checked_external(self.root, files, alias)

    def test_exact_JSON_payload_substitution_is_rejected(self):
        original = Path.read_bytes
        def substitute(path):
            return b'{"a":2}' if path == self.a else original(path)
        with self.assertRaisesRegex(module.MultitaskProtocolError, 'parsed JSON bytes'):
            with self.batch() as batch, patch.object(Path, 'read_bytes', substitute):
                batch.locked_json(self.root, self.files, self.pointer)

    def test_JSON_parser_cannot_change_file_after_verified_payload(self):
        original = json.loads
        def changing(payload, *args, **kwargs):
            result = original(payload, *args, **kwargs)
            if payload == b'{"a":1}':
                self.a.write_bytes(b'{"a":2}')
            return result
        with self.assertRaises(module.MultitaskProtocolError):
            with self.batch() as batch, patch.object(module.json, 'loads', changing):
                batch.locked_json(self.root, self.files, self.pointer)

    def test_valid_JSON_is_decoded_from_verified_payload(self):
        with self.batch() as batch:
            self.assertEqual(batch.locked_json(self.root, self.files, self.pointer), {'a': 1})
        self.assertEqual(batch.metrics['exact_JSON_payloads'], 1)

    def test_unrelated_outputs_do_not_invalidate_declared_edges(self):
        with self.batch() as batch:
            batch.checked_external(self.root, self.files, self.a)
            for n in range(6):
                (self.root / ('output-' + str(n))).mkdir()
                batch.checked_external(self.root, self.files, self.a)

    def test_all_maps_are_final_checked_even_after_one_fails(self):
        second = {'b': self.files['b']}
        with self.assertRaises(module.InputTopologyChanged):
            with self.batch() as batch:
                batch.checked_external(self.root, self.files, self.a)
                batch.checked_external(self.root, second, self.b)
                self.files['b'] = '0' * 64
        self.assertEqual(batch.metrics['full_graph_boundary_checks'], 4)
        self.assertEqual(batch.metrics['used_path_end_checks'], 2)


class HookTests(Fixture, unittest.TestCase):
    def owner(self):
        return SimpleNamespace(_checked=lambda *args: 'external-original',
                               locked_pointer=lambda *args: 'pointer-original',
                               locked_json=lambda *args: 'json-original',
                               verify_inputs=lambda *args: 'verify-original',
                               admit=lambda *args: 'scientific-original')

    def test_scope_restores_originals_and_does_not_replace_scientific_gates(self):
        owner = self.owner()
        originals = dict(vars(owner))
        with self.assertRaisesRegex(RuntimeError, 'scientific rejection'):
            with self.batch() as batch:
                with batch.scoped_hooks(external=((owner, '_checked'),),
                                        pointers=((owner, 'locked_pointer'),),
                                        json_readers=((owner, 'locked_json'),)):
                    self.assertEqual(owner._checked(self.root, self.files, self.a), self.a)
                    self.assertEqual(owner.locked_pointer(self.root, self.files, self.pointer), self.a)
                    self.assertEqual(owner.locked_json(self.root, self.files, self.pointer), {'a': 1})
                    self.assertIs(owner.verify_inputs, originals['verify_inputs'])
                    self.assertIs(owner.admit, originals['admit'])
                    raise RuntimeError('scientific rejection')
        self.assertEqual(vars(owner), originals)
        self.assertFalse(batch.metrics['complete'])

    def test_other_thread_uses_original_while_owner_uses_batch(self):
        owner = self.owner()
        result = []
        with self.batch() as batch:
            with batch.scoped_hooks(external=((owner, '_checked'),)):
                worker = threading.Thread(target=lambda: result.append(owner._checked(None, None, None)))
                worker.start()
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(owner._checked(self.root, self.files, self.a), self.a)
        self.assertEqual(result, ['external-original'])

    def test_direct_batch_use_from_other_thread_is_rejected_without_poisoning_owner(self):
        errors = []
        with self.batch() as batch:
            def wrong_thread():
                try:
                    batch.checked_external(self.root, self.files, self.a)
                except Exception as error:
                    errors.append(error)
            worker = threading.Thread(target=wrong_thread)
            worker.start()
            worker.join(timeout=3)
            self.assertIsInstance(errors[0], RuntimeError)
            batch.checked_external(self.root, self.files, self.a)
        self.assertTrue(batch.metrics['complete'])

    def test_nested_hook_scopes_are_rejected_and_outer_restored(self):
        owner = self.owner()
        original = owner._checked
        with self.batch() as batch:
            with batch.scoped_hooks(external=((owner, '_checked'),)):
                with self.assertRaisesRegex(RuntimeError, 'nested'):
                    with batch.scoped_hooks(external=((owner, '_checked'),)):
                        self.fail('nested scope entered')
                self.assertEqual(owner._checked(self.root, self.files, self.a), self.a)
        self.assertIs(owner._checked, original)

    def test_hook_scopes_serialize_across_threads_and_keep_each_batch_separate(self):
        owner = self.owner()
        attempted, entered, finished = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def second_scope():
            try:
                with self.batch() as second:
                    attempted.set()
                    with second.scoped_hooks(external=((owner, '_checked'),)):
                        entered.set()
                        self.assertEqual(owner._checked(self.root, self.files, self.a), self.a)
                self.assertEqual(second.metrics['maps'], 1)
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()
        with self.batch() as first:
            with first.scoped_hooks(external=((owner, '_checked'),)):
                worker = threading.Thread(target=second_scope)
                worker.start()
                self.assertTrue(attempted.wait(3))
                self.assertFalse(entered.wait(.05))
                self.assertEqual(owner._checked(self.root, self.files, self.a), self.a)
            self.assertTrue(finished.wait(3))
            worker.join(timeout=3)
        self.assertFalse(errors)
        self.assertTrue(entered.is_set())

    def test_forbidden_verify_or_scientific_hooks_fail_without_mutation(self):
        owner = self.owner()
        originals = dict(vars(owner))
        for attribute in ('verify_inputs', 'admit'):
            with self.batch() as batch:
                with self.assertRaisesRegex(ValueError, 'only explicit lookup'):
                    with batch.scoped_hooks(external=((owner, attribute),)):
                        self.fail('forbidden hook entered')
        self.assertEqual(vars(owner), originals)


if __name__ == '__main__':
    unittest.main()
