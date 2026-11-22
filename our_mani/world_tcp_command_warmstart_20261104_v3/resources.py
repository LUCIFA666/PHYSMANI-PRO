"""Correct shared Native/training lock inodes; no old source is changed."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
GLOBAL_LOCKS=(ROOT/'runtime/framework/fixed_TRAIN104_geometry_native.lock',
              ROOT/'runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock')


@contextmanager
def resources(*,pipeline_owned=False,inherited_lock_fds=None):
    if type(pipeline_owned)is not bool:raise ValueError('Explicit external lock ownership required')
    if pipeline_owned:
        # A boolean does not prove ownership. Actual inherited descriptors
        # must name the two correct global lock inodes. flock on the same
        # open-file description confirms exclusive ownership (or acquires it
        # if free); another Native owner makes this fail before GPU activity.
        if (not isinstance(inherited_lock_fds,tuple)or len(inherited_lock_fds)!=2
                or any(type(fd)is not int or fd<0 for fd in inherited_lock_fds)):
            raise ValueError('Actual inherited global lock descriptors required')
        copies=[]
        try:
            for path,fd in zip(GLOBAL_LOCKS,inherited_lock_fds):
                actual=os.fstat(fd);expected=path.stat()
                if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
                    raise ValueError('Inherited lock must match exact global path/inode')
                copy=os.dup(fd);copies.append(copy)
                fcntl.flock(copy,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield
        finally:
            # Never LOCK_UN an inherited shared description: that would
            # release the real outer owner's lock. Closing duplicates is safe.
            for fd in reversed(copies):os.close(fd)
        return
    streams=[]
    try:
        for path in GLOBAL_LOCKS:
            path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('a+')
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                stream.close();raise RuntimeError('Native/training resources busy; no GPU work begins')
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):fcntl.flock(stream.fileno(),fcntl.LOCK_UN);stream.close()
