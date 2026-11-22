"""No Native/PyRep import; CFFI ownership and exceptional paths only."""
from types import SimpleNamespace
import unittest

from our_mani.local_mpc_20261102.collection_audit import (
    CollectionBufferReleaseError, CollectionQueryError, collection_objects,
)


class Pointer:
    def __init__(self, values, fail_at=None):
        self.values, self.fail_at = values, fail_at

    def __getitem__(self, index):
        if index == self.fail_at:
            raise RuntimeError("injected C buffer read failure")
        return self.values[index]


class FakeFFI:
    NULL = object()

    def __init__(self):
        self.casts = []

    def new(self, kind):
        assert kind == 'int *'
        return [0]

    def cast(self, kind, pointer):
        assert kind == 'char *'
        self.casts.append((kind,pointer))
        return pointer


class FakeLib:
    def __init__(self, ffi, values, *, count=None, null=False, fail_at=None, release_status=1):
        self.ffi, self.pointer = ffi, Pointer(values,fail_at)
        self.count = len(values) if count is None else count
        self.null, self.release_status = null, release_status
        self.queries, self.releases = [], []

    def simGetCollectionObjects(self, handle, out_count):
        self.queries.append(handle)
        out_count[0] = self.count
        return self.ffi.NULL if self.null else self.pointer

    def simReleaseBuffer(self, pointer):
        self.releases.append(pointer)
        assert pointer is self.pointer
        return self.release_status


def fixture(values=(1,4,7),**kwargs):
    ffi=FakeFFI()
    return SimpleNamespace(ffi=ffi,lib=FakeLib(ffi,values,**kwargs))


class CollectionGetterTests(unittest.TestCase):
    def test_members_copied_before_buffer_is_freed_exactly_once(self):
        sim=fixture()
        self.assertEqual(collection_objects(sim,19),[1,4,7])
        self.assertEqual(sim.lib.queries,[19])
        self.assertEqual(sim.lib.releases,[sim.lib.pointer])
        self.assertEqual(sim.ffi.casts,[('char *',sim.lib.pointer)])

    def test_nonnull_empty_collection_frees_owned_buffer(self):
        sim=fixture(())
        self.assertEqual(collection_objects(sim,19),[])
        self.assertEqual(sim.lib.releases,[sim.lib.pointer])

    def test_null_is_error_and_is_never_freed(self):
        sim=fixture(null=True,count=0)
        with self.assertRaisesRegex(CollectionQueryError,'NULL'):
            collection_objects(sim,19)
        self.assertEqual(sim.lib.releases,[])
        self.assertEqual(sim.ffi.casts,[])

    def test_invalid_counts_still_release_nonnull_buffer(self):
        for count in (-1,100001):
            with self.subTest(count=count):
                sim=fixture(count=count)
                with self.assertRaisesRegex(CollectionQueryError,'count outside'):
                    collection_objects(sim,19)
                self.assertEqual(sim.lib.releases,[sim.lib.pointer])

    def test_read_exception_still_releases_once_and_preserves_error(self):
        sim=fixture(fail_at=1)
        with self.assertRaisesRegex(RuntimeError,'injected C buffer read failure'):
            collection_objects(sim,19)
        self.assertEqual(sim.lib.releases,[sim.lib.pointer])

    def test_negative_member_fails_but_releases(self):
        sim=fixture((1,-1,3))
        with self.assertRaisesRegex(CollectionQueryError,'negative object handle'):
            collection_objects(sim,19)
        self.assertEqual(sim.lib.releases,[sim.lib.pointer])

    def test_cleanup_failure_is_explicit_and_chains_original_error(self):
        sim=fixture(fail_at=1,release_status=-1)
        with self.assertRaises(CollectionBufferReleaseError) as error:
            collection_objects(sim,19)
        self.assertIsInstance(error.exception.__cause__,RuntimeError)
        self.assertIn('injected C buffer read failure',str(error.exception.__cause__))
        self.assertEqual(sim.lib.releases,[sim.lib.pointer])

    def test_bad_python_arguments_make_no_native_call(self):
        sim=fixture()
        for handle in (-1,True,1.2):
            with self.subTest(handle=handle):
                with self.assertRaises((TypeError,ValueError)):
                    collection_objects(sim,handle)
        self.assertEqual(sim.lib.queries,[])
        self.assertEqual(sim.lib.releases,[])


if __name__=='__main__':
    unittest.main()
