"""Check the original X11 display before importing Qt/PyRep or running policy."""
import ctypes
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def paths(argv):
    worker = ROOT / 'scripts/run_geometry_native_actuator_diagnostic_v5_r10.py'
    if not argv or (ROOT / argv[0]).resolve() != worker:
        raise ValueError('display guard accepts only the frozen TRAIN v5 worker')
    if argv.count('--out') != 1:
        raise ValueError('one explicit development output is required')
    index = argv.index('--out')
    if index + 1 >= len(argv):
        raise ValueError('missing development output')
    out = (ROOT / argv[index + 1]).resolve()
    base = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2'
    if out not in {base / 'development_demo_only' / leaf for leaf in ('cv_normal', 'context_normal')}:
        raise ValueError('display guard output must be in the fresh v5_2 namespace')
    raw = base / 'parent_raw' / out.name
    if not raw.is_dir():
        raise ValueError('the owned parent must create its raw directory first')
    return worker, raw


def x11_available():
    name = ctypes.util.find_library('X11')
    if name is None:
        raise RuntimeError('libX11 is unavailable')
    library = ctypes.CDLL(name)
    library.XOpenDisplay.argtypes = [ctypes.c_char_p]
    library.XOpenDisplay.restype = ctypes.c_void_p
    display = library.XOpenDisplay(None)
    if not display:
        return False
    library.XCloseDisplay.argtypes = [ctypes.c_void_p]
    library.XCloseDisplay(display)
    return True


def readiness(environ, opener=x11_available):
    value = dict(schema='r10_original_X11_native_display_readiness_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), ready=False,
        display=environ.get('DISPLAY'), qt_platform=environ.get('QT_QPA_PLATFORM'),
        CUDA_VISIBLE_DEVICES=environ.get('CUDA_VISIBLE_DEVICES'),
        policy_started=False, physics_started=False, rendered_sensor_validated=False)
    if environ.get('CUDA_VISIBLE_DEVICES') != '':
        value['reason'] = 'CPU_native_requires_explicit_empty_CUDA_VISIBLE_DEVICES'
    elif environ.get('QT_QPA_PLATFORM') not in (None, '', 'xcb'):
        value['reason'] = 'original_OPENGL_RGBD_path_requires_X11_not_offscreen_substitution'
    elif not environ.get('DISPLAY'):
        value['reason'] = 'DISPLAY_unavailable'
    else:
        try:
            value['ready'] = bool(opener())
            value['reason'] = 'X11_connection_available' if value['ready'] else 'X11_connection_failed'
        except Exception as error:
            value.update(reason='X11_connection_query_failed', error_type=type(error).__name__, message=str(error))
    return value


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    paths_only = bool(argv and argv[0] == '--error-path')
    if paths_only:
        argv.pop(0)
    worker, raw = paths(argv)
    if paths_only:
        print(raw / 'xvfb.log')
        return 0
    result = readiness(os.environ)
    receipt = raw / 'display_readiness.json'
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if not result['ready']:
        print(json.dumps(result), file=sys.stderr)
        return 78
    # Replace this process only after readiness. The original worker retains
    # its own source-lock, models, physical reset, policy and counted clock.
    os.execv(sys.executable, [sys.executable, str(worker), *argv[1:]])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
