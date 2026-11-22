from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/framework/multitask_executor_cycle_20261112_v1'
PLAN = ROOT / 'config/framework/multitask_executor_cycle_20261112_v1/plan.json'
PYTHON = '/home/ubuntu/physmani-conda/envs/physmani5090/bin/python'
HISTORICAL_STOP = dict(sha256='e2bb2d8aff8697cf3f0c18a7a0707b2bd7e8e817d18b563513c365bf0e5a1850',
                       mtime_ns=1793780243677293196, inode=220594231)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from ..local_mpc_20261102.records import write_new
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_new(path, value)


def append(path, value):
    from ..local_mpc_20261102.records import append as add
    add(path, value)


def utc():
    return datetime.now(timezone.utc).isoformat()


def pointer(path):
    return dict(path=str(Path(path).resolve(strict=True)), sha256=sha(path))


def identity(path):
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return dict(sha256=sha(path), mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def stopped():
    if (identity(ROOT / 'runtime/framework/STOP') != HISTORICAL_STOP
            or (ROOT / 'runtime/framework/multitask_executor_cycle_20261112_v1/STOP').exists()
            or (OUT / 'STOP').exists()):
        raise InterruptedError('New scoped cycle STOP; historical STOP and old queues preserved')


def verify(files):
    for name, checksum in files.items():
        if sha(name) != checksum:
            raise RuntimeError('Frozen source/input changed: ' + name)


def sources():
    from ..trained_world_reference_live_20261104_v1.runtime import source_map
    from ..multitask_execution_pipeline_20261104_v1.registry import inventory
    result = source_map()
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'our_mani/execution_interface_20261111_v1').glob('*.py'))
    paths += list((ROOT / 'our_mani/physmani_feedback_interface_20261104_v1').glob('*.py'))
    paths += [ROOT / 'scripts/run_multitask_executor_cycle_native_20261112_v1.sh',
              ROOT / 'our_mani/task_time_limits_20261111_v1/feedback_backend.py',
              ROOT / 'our_mani/generic_reference_mpc_20261104_v1/native_backend.py']
    for row in inventory(ROOT)['tasks']:
        result[row['native_source']['path']] = row['native_source']['sha256']
        result.update(row['native_condition_sources'])
    for path in paths:
        result[str(path.resolve())] = sha(path)
    return result
