"""Exactly fixed TRAIN sources for the original released two-block baseline."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, digest
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.block_physmani_v346 import binding, run_TRAIN_baseline
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.case_output_isolation_v238 import require_development_output
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def preflight(args):
    if args.split != 'train' or not args.development_demo or args.method != 'physmani' or args.task not in BLOCKS:
        raise ValueError('only original PhysMani fixed TRAIN block development')
    manifest = (ROOT / args.manifest).resolve()
    if manifest != ROOT / 'artifacts/framework/protocol/train_manifest.json':
        raise ValueError('original full TRAIN manifest required')
    configuration, lock = (ROOT / args.config).resolve(), (ROOT / args.frozen_inputs).resolve()
    files = json.loads(lock.read_text())['files']
    verify_inputs(ROOT, files)
    for path in (manifest, configuration, Path(__file__)):
        if files.get(str(path)) != file_sha(path):
            raise ValueError('actual worker/config/manifest absent from frozen lock')
    c = json.loads(configuration.read_text())
    original_path = ROOT / 'config/framework/aligned_protocol_r09_baseline_v6.json'
    if files.get(str(original_path)) != file_sha(original_path):
        raise ValueError('original R09 V6 baseline declaration absent from frozen lock')
    original = json.loads(original_path.read_text())
    for key in ('baseline_checkpoint', 'baseline_repo', 'baseline_decision_ticks', 'baseline_executor',
                'baseline_reference_filter', 'current_instance_masks', 'sensor_protocol', 'gripper_velocity',
                'dt', 'simulation_seconds', 'global_joint_acceleration', 'seed'):
        if c[key] != original[key]:
            raise ValueError('original baseline algorithm/sensor/physics field changed: ' + key)
    expected = binding(ROOT, c)
    if c.get('block_physmani_v346') != expected:
        raise ValueError('actual released baseline binding changed')
    checked_subset(ROOT, files, {str(ROOT / name): sha for name, sha in expected['sources_sha256'].items()})
    checkpoint = Path(c['baseline_checkpoint']).resolve()
    if files.get(str(checkpoint)) != expected['checkpoint']['sha256']:
        raise ValueError('original author checkpoint absent from frozen lock')
    rows = [r for r in json.loads(manifest.read_text())['episodes'] if r['task'] == args.task][:4]
    selected = [r for r in rows if r['raw_episode_id'] == args.raw_episode_id]
    if len(selected) != 1:
        raise ValueError('fixed first four original TRAIN rows only')
    row = selected[0]
    if files.get(row['path']) != file_sha(row['path']):
        raise ValueError('original HDF5 source absent from frozen lock')
    out = require_development_output(ROOT, args.out, task=args.task, raw=args.raw_episode_id, method='physmani')
    return dict(args=args, config=c, row=row, out=out, files=files,
        binding=expected, identity=dict(task=args.task, method='physmani', split='train',
            parent=row['raw_episode_id'], source_row_sha256=digest(row), config_sha256=file_sha(configuration),
            frozen_inputs_sha256=file_sha(lock), wrapper_sha256=file_sha(__file__), formal_VAL_TEST=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'frozen-inputs', 'task', 'raw-episode-id', 'method', 'split', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--development-demo', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    prepared = preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False, GPU_started=False,
            identity=prepared['identity'], actual_original_baseline=prepared['binding'])), flush=True)
        return
    out = prepared['out']
    if out.exists():
        raise FileExistsError('fresh output required; baseline case is never retried')
    out.mkdir(parents=True)
    atomic_json(out / 'inputs.json', dict(identity=prepared['identity'], actual_original_baseline=prepared['binding']))
    try:
        record = run_TRAIN_baseline(ROOT, prepared['row'], prepared['config'])
        verify_inputs(ROOT, prepared['files'])
        record.update(identity=prepared['identity'], evaluation_process_pid=os.getpid(),
            complete=True, infrastructure_error=None, cleanup_completed=True, automatic_retry=False)
        atomic_json(out / 'case.json', record)
    except BaseException as error:
        atomic_json(out / 'infrastructure_error.json', dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), counted_as_policy_failure=False, automatic_retry=False))
        raise
    print(json.dumps(dict(complete=True, native_success=record['official_task_success'],
        actual_motion_passed=record['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed'],
        ticks=record['counted_clock']['ticks'], case=str(out / 'case.json'))), flush=True)


if __name__ == '__main__':
    main()
