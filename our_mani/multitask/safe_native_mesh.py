"""Read native mesh buffers without the bundled PyRep release-pointer bug.

The audited external simGetShapeMesh overwrites ``outNormals`` with a list
read from ``outIndices`` (three times that buffer's length), then casts the
first integer index to a pointer and frees it. Index zero masks the bad free;
a nonzero first index can abort the process. The external repository remains
unchanged. This reader calls the same native API, preserves all three owned
buffer pointers and copies their correct types before releasing each buffer.
Vertices/indices are not simplified, transformed or otherwise altered.
"""
import numpy as np

from .protocol import MultitaskInfrastructureError

VERSION = 'r10_safe_CFFI_mesh_buffers_v1'


def _is_native_shape(shape):
    return any(cls.__module__.startswith('pyrep.') for cls in type(shape).__mro__)


def read_native_mesh(shape, *, backend=None, allow_mock_fallback=True):
    """Return float64 vertices/normals and int64 triangles, each shaped (-1,3).

    Native PyRep shapes never fall back to the defective wrapper, including
    when the direct C API fails. ``backend`` injection supports CPU fake-CFFI
    tests. Only non-native test doubles may use their ordinary get_mesh_data.
    """
    if backend is None and not _is_native_shape(shape):
        if not allow_mock_fallback:
            raise MultitaskInfrastructureError('native mesh reader requires a native shape/backend')
        return shape.get_mesh_data()
    if backend is None:
        from pyrep.backend import sim as backend
    ffi = backend.ffi
    vertices_pointer, vertex_count = ffi.new('float **'), ffi.new('int *')
    indices_pointer, index_count = ffi.new('int **'), ffi.new('int *')
    normals_pointer = ffi.new('float **')
    primary_error = None
    try:
        result = backend.lib.simGetShapeMesh(int(shape.get_handle()), vertices_pointer, vertex_count,
                                            indices_pointer, index_count, normals_pointer)
        # Match the bundled native wrapper's failure convention (ret < 0);
        # successful data are independently validated below.
        if result < 0:
            raise MultitaskInfrastructureError('native simGetShapeMesh returned failure: ' + str(result))
        nv, ni = int(vertex_count[0]), int(index_count[0])
        if nv < 3 or ni < 3 or nv % 3 or ni % 3:
            raise MultitaskInfrastructureError('native mesh buffer dimensions are invalid')
        if vertices_pointer[0] == ffi.NULL or indices_pointer[0] == ffi.NULL:
            raise MultitaskInfrastructureError('native mesh returned an empty vertex/index pointer')
        vertices = np.array([vertices_pointer[0][i] for i in range(nv)], dtype=np.float64).reshape(-1, 3)
        triangles = np.array([indices_pointer[0][i] for i in range(ni)], dtype=np.int64).reshape(-1, 3)
        normals = (np.empty((0, 3), dtype=np.float64) if normals_pointer[0] == ffi.NULL else
                   np.array([normals_pointer[0][i] for i in range(ni * 3)], dtype=np.float64).reshape(-1, 3))
        if (not np.isfinite(vertices).all() or not np.isfinite(normals).all()
                or triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise MultitaskInfrastructureError('native mesh data are nonfinite or out of range')
        return vertices, triangles, normals
    except BaseException as error:
        primary_error = error
        raise
    finally:
        released, errors = [], []
        for holder in (vertices_pointer, indices_pointer, normals_pointer):
            pointer = holder[0]
            if pointer == ffi.NULL or any(pointer == previous for previous in released):
                continue
            released.append(pointer)
            try:
                backend.simReleaseBuffer(ffi.cast('char *', pointer))
            except Exception as error:
                errors.append(repr(error))
        if errors:
            cleanup = MultitaskInfrastructureError('native mesh buffer cleanup failed: ' + '; '.join(errors))
            if primary_error is not None:
                raise cleanup from primary_error
            raise cleanup
