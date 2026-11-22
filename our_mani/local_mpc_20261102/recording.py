"""Bounded background compression; Native camera capture stays on main thread."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


class FrameRecorder:
    def __init__(self, max_pending=2):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mpc_frame_writer")
        self.pending = []
        self.max_pending = max_pending
        self.frames = 0

    def submit(self, path, arrays):
        while self.pending and self.pending[0].done():
            self.pending.pop(0).result()
        if len(self.pending) >= self.max_pending:
            self.pending.pop(0).result()
        immutable = {k: np.array(v, copy=True) for k, v in arrays.items()}
        self.pending.append(self.pool.submit(self._write, Path(path), immutable))
        self.frames += 1

    @staticmethod
    def _write(path, arrays):
        if path.exists():
            raise FileExistsError(str(path))
        temporary = path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.rename(path)

    def close(self):
        pending, self.pending = self.pending, []
        try:
            for future in pending:
                future.result()
        finally:
            self.pool.shutdown(wait=True)

