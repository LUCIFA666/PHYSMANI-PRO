"""Preserve v5 policy bytes; freeze a fresh guarded execution namespace."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs


def main():
    base = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_setup'
    out = ROOT / 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2_setup'
    if out.exists():
        raise RuntimeError('Preserve this prepared attempt; never overwrite')
    previous = json.loads((base / 'frozen_inputs.json').read_text())
    verify_inputs(ROOT, previous['files'])
    out.mkdir()
    shutil.copyfile(base / 'configuration.json', out / 'configuration.json')
    assert file_sha(base / 'configuration.json') == file_sha(out / 'configuration.json')
    files = dict(previous['files'])
    added = (out / 'configuration.json', Path(__file__),
        ROOT / 'scripts/run_native_actuator_diagnostic_pair_v5_2_r10.py',
        ROOT / 'scripts/native_display_guard_r10.py',
        ROOT / 'scripts/run_local_sim_display_checked_r10.sh',
        ROOT / 'tests/test_guarded_native_v5_2.py')
    for path in added:
        files[str(path.resolve())] = file_sha(path)
    verify_inputs(ROOT, files)
    atomic_json(out / 'frozen_inputs.json', dict(previous,
        schema='r10_fresh_guarded_fixed_TRAIN104_v5_2_actual_lock_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), files=files,
        scope='Unchanged v5 policy/model/deadlines/native physics; guarded X11 launch in fresh namespace; fixed TRAIN104 CV/Context only; no fitting, VAL, TEST, videos or retries.',
        native_started=False))
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    rows = []
    for index, (name, checksum) in enumerate(sorted(files.items())):
        path = Path(name)
        if path.suffix not in ('.py', '.sh'):
            continue
        target = snapshot / f'{index:04d}_{path.name}'
        shutil.copyfile(path, target)
        assert file_sha(target) == checksum
        rows.append(dict(original_path=name, sha256=checksum, archived_path=str(target.relative_to(ROOT))))
    atomic_json(out / 'source_snapshot.json', dict(files=rows, native_started=False))
    verify_inputs(ROOT, files)
    atomic_json(out / 'setup_receipt.json', dict(
        schema='r10_guarded_v5_2_setup_receipt_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        previous_setup=str(base.relative_to(ROOT)), previous_lock_sha256=file_sha(base / 'frozen_inputs.json'),
        configuration_sha256=file_sha(out / 'configuration.json'),
        frozen_inputs_sha256=file_sha(out / 'frozen_inputs.json'),
        original_frozen_inputs_preserved=True, previous_input_count=len(previous['files']),
        frozen_input_count=len(files), controller_models_and_policy_identity_unchanged=True,
        only_changes=['fresh output/setup namespace', 'X11 readiness before Qt/PyRep',
            'captured Xvfb errors', 'host-process preflight', 'shared kernel run lock'],
        methods=['cv_geometry', 'context_geometry'], split='train', original_parent='episode104',
        simulation_seconds=10., dt=.05, native_started=False, formal_metric=False))
    print(json.dumps(dict(setup=str(out), frozen_input_count=len(files),
        configuration_sha256=file_sha(out / 'configuration.json'))))


if __name__ == '__main__':
    main()
