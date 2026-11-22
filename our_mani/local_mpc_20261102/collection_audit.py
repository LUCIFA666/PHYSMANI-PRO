"""Read-only CoppeliaSim 4.1 collection getter with owned-buffer cleanup.

No Native import or initialization occurs here. The caller passes its already
initialized ``pyrep.backend.sim`` module. Local primary API declarations:

* PyRep/cffi_build/cffi_build.py:425 / sim.h:410:
  simInt* simGetCollectionObjects(simInt collectionHandle,simInt* objectCount);
* cffi_build.py:222 / sim.h:207:
  simInt simReleaseBuffer(const simChar* buffer);

The local CoppeliaSim help says nullptr is an error and requires the caller to
release every returned buffer with simReleaseBuffer. This helper never creates,
edits, empties or removes a collection, never changes object state, and never
steps physics. NULL is deliberately not interpreted as an empty collection.
"""
import operator


class CollectionQueryError(RuntimeError):
    pass


class CollectionBufferReleaseError(RuntimeError):
    pass


def collection_objects(sim, collection_handle, *, max_objects=100000):
    """Return native member handles as a Python list, freeing its C buffer once.

    Example in the root's zero-action canary:

        handles = collection_objects(sim, backend.arm._collision_collection)
        names = [sim.simGetObjectName(handle) for handle in handles]

    Compare these handles with the gripper's actual shape handles; collection
    name alone cannot establish whether fingers/palm geometry is included.
    The caller can wrap this function with its physics/task/ledger clock check.
    """
    if isinstance(collection_handle, bool) or isinstance(max_objects, bool):
        raise TypeError("integer native collection handle and object-count bound required")
    handle, bound = operator.index(collection_handle), operator.index(max_objects)
    if handle < 0 or bound < 0:
        raise ValueError("nonnegative collection handle and object-count bound required")
    count = sim.ffi.new('int *')
    pointer = sim.lib.simGetCollectionObjects(handle, count)
    if pointer == sim.ffi.NULL:
        raise CollectionQueryError("simGetCollectionObjects returned NULL for collection "
                                   + str(handle) + " (reported count=" + str(int(count[0])) + ")")
    primary_error = None
    try:
        size = int(count[0])
        if size < 0 or size > bound:
            raise CollectionQueryError("native collection count outside explicit bound: " + str(size))
        result = [int(pointer[index]) for index in range(size)]
        if any(member < 0 for member in result):
            raise CollectionQueryError("native collection returned a negative object handle")
        return result
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Matches simGetObjectsInTree's local PyRep pattern, but uses the raw
        # simInt return so a release failure is not silently dropped.
        try:
            released = sim.lib.simReleaseBuffer(sim.ffi.cast('char *', pointer))
            if released < 0:
                raise CollectionBufferReleaseError("simReleaseBuffer returned " + str(int(released)))
        except BaseException as error:
            if isinstance(error, CollectionBufferReleaseError):
                raise error from primary_error
            raise CollectionBufferReleaseError("failed to release collection buffer: "
                + type(error).__name__ + ": " + str(error)) from (primary_error or error)
