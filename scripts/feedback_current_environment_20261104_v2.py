"""Record the current Native runtime without claiming the old OS snapshot.

No environment mutation, simulator import, or launch is performed here.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys


def pointer(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(data)
    return dict(path=str(path), sha256=h.hexdigest())


def current_environment():
    import numpy as np
    expected = Path('/home/ubuntu/physmani-conda/envs/physmani5090/bin/python').resolve()
    if Path(sys.executable).resolve() != expected: raise ValueError('The Native conda interpreter is required')
    root = Path(os.environ['COPPELIASIM_ROOT']).resolve(strict=True)
    core = ['coppeliaSim', 'libcoppeliaSim.so', 'libsimExtDynamicsBullet-2-78.so',
            'libsimExtDynamicsBullet-2-83.so', 'libsimExtDynamicsODE.so']
    files = [root / name for name in core if (root / name).is_file()]
    if not (root / 'libcoppeliaSim.so').is_file(): raise ValueError('Original CoppeliaSim core not found')
    packages = {}
    for name in ('pyrep', 'rlbench'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None: raise ValueError('Actual Native package not found: ' + name)
        packages[name] = pointer(spec.origin)
    return dict(profile='current_Native_runtime_feedback_validation_20261104_v2',
        old_v820_OS_equivalence_claimed=False, current_host_libraries_not_frozen=True,
        reason='Old snapshot launcher failed before Native: removed host libdns filename after system update',
        python=sys.version, executable=pointer(sys.executable), numpy=np.__version__,
        platform=platform.platform(), CoppeliaSim_root=str(root),
        engine_files=[pointer(p) for p in files], packages=packages,
        env={k: os.environ.get(k) for k in ('DISPLAY', 'LD_LIBRARY_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH',
            'PYTHONPATH', 'CUDA_VISIBLE_DEVICES')})
