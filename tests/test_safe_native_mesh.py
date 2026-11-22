"""Fake-CFFI pointer ownership tests; no native engine or GPU imports."""
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.multitask.safe_native_mesh import read_native_mesh
from our_mani.multitask.protocol import MultitaskInfrastructureError


class Pointer:
    def __init__(self, values, *, fail_read=False): self.values, self.fail_read = values, fail_read
    def __getitem__(self, index):
        if self.fail_read: raise ValueError('injected buffer-copy failure')
        return self.values[index]


class FakeFFI:
    NULL = None
    def new(self, kind): return [None] if kind.endswith('**') else [0]
    def cast(self, kind, pointer):
        assert kind == 'char *' and isinstance(pointer, Pointer)
        return pointer


class FakeBackend:
    def __init__(self, *, copy_failure=False, native_failure=False, release_failure=False):
        self.ffi = FakeFFI();self.released=[];self.native_failure=native_failure
        # A nonzero first index reproduces the bundled wrapper's invalid-free trigger.
        self.vertices=Pointer([0.,0.,0.,1.,0.,0.,0.,1.,0.],fail_read=copy_failure)
        self.indices=Pointer([2,0,1]);self.normals=Pointer([0.,0.,1.] * 3)
        self.lib=SimpleNamespace(simGetShapeMesh=self.get_mesh);self.release_failure=release_failure
    def get_mesh(self, handle, vertices, nv, indices, ni, normals):
        assert handle == 81
        vertices[0]=self.vertices;nv[0]=9;indices[0]=self.indices;ni[0]=3;normals[0]=self.normals
        return -1 if self.native_failure else 1
    def simReleaseBuffer(self, pointer):
        self.released.append(pointer)
        if self.release_failure and pointer is self.vertices: raise RuntimeError('injected release failure')


class NativeShape:
    __module__ = 'pyrep.objects.shape'
    def get_handle(self): return 81
    def get_mesh_data(self): raise AssertionError('unsafe native fallback must never execute')


class SafeMeshTests(unittest.TestCase):
    def test_correct_buffers_and_release_pointer_identity_with_nonzero_first_triangle_index(self):
        backend=FakeBackend();vertices,indices,normals=read_native_mesh(NativeShape(),backend=backend)
        np.testing.assert_array_equal(indices,[[2,0,1]])
        np.testing.assert_array_equal(normals,[[0,0,1]] * 3)
        assert vertices.dtype == np.float64 and indices.dtype == np.int64
        assert backend.released == [backend.vertices,backend.indices,backend.normals]

    def test_copy_exception_releases_all_three_buffers(self):
        backend=FakeBackend(copy_failure=True)
        with self.assertRaisesRegex(ValueError,'buffer-copy'): read_native_mesh(NativeShape(),backend=backend)
        assert backend.released == [backend.vertices,backend.indices,backend.normals]

    def test_native_failure_releases_allocated_buffers_and_does_not_fallback(self):
        backend=FakeBackend(native_failure=True)
        with self.assertRaisesRegex(MultitaskInfrastructureError,'returned failure'):
            read_native_mesh(NativeShape(),backend=backend)
        assert backend.released == [backend.vertices,backend.indices,backend.normals]

    def test_nonnegative_return_matches_original_wrapper_convention(self):
        backend=FakeBackend();original=backend.lib.simGetShapeMesh
        def populated_zero(*args): original(*args);return 0
        backend.lib.simGetShapeMesh=populated_zero
        vertices,indices,_=read_native_mesh(NativeShape(),backend=backend)
        assert vertices.shape == (3,3) and indices.shape == (1,3)
        assert len(backend.released) == 3

    def test_one_release_failure_still_attempts_other_buffers_and_preserves_copy_cause(self):
        backend=FakeBackend(copy_failure=True,release_failure=True)
        with self.assertRaisesRegex(MultitaskInfrastructureError,'cleanup failed') as error:
            read_native_mesh(NativeShape(),backend=backend)
        assert isinstance(error.exception.__cause__,ValueError)
        assert backend.released == [backend.vertices,backend.indices,backend.normals]

    def test_mock_only_fallback_keeps_existing_cpu_shape_contract(self):
        value=(np.zeros((3,3)),np.array([[0,1,2]]),np.ones((3,3)))
        mock=SimpleNamespace(get_mesh_data=lambda:value)
        assert read_native_mesh(mock) is value
        with self.assertRaisesRegex(MultitaskInfrastructureError,'requires a native'):
            read_native_mesh(mock,allow_mock_fallback=False)


if __name__=='__main__': unittest.main()
