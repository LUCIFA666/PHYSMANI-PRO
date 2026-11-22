"""Fresh-process, fixed TRAIN block smoke with the actual shared16 checkpoint."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json, file_sha, digest
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.joint16_checkpoint_v172 import verified_checkpoint
from our_mani.multitask.joint16_block_context_v97 import Joint16BlockContext
from our_mani.multitask.joint16_tracks_v31 import BLOCKS


def preflight(args):
    if args.split != 'train' or not args.development_demo or args.method not in ('cv_geometry', 'context_geometry'):
        raise ValueError('fixed TRAIN development only; exactly authorized geometry methods')
    configuration = (ROOT/args.config).resolve()
    lock = (ROOT/args.frozen_inputs).resolve()
    manifest = (ROOT/args.manifest).resolve()
    out = (ROOT/args.out).resolve()
    if (manifest != ROOT/'artifacts/framework/protocol/train_manifest.json' or args.task not in BLOCKS
            or not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10')
            or 'development_demo_only' not in out.parts):
        raise ValueError('exact original TRAIN manifest, block task and explicit development output required')
    files = json.loads(lock.read_text())['files']
    verify_inputs(ROOT, files)
    for p in (configuration, manifest, Path(__file__)):
        if files.get(str(p)) != file_sha(p): raise ValueError('worker/config/roster absent from frozen lock')
    c = json.loads(configuration.read_text())
    from our_mani.multitask.block_clip_brake_v315 import binding as held_binding
    from our_mani.multitask.frozen_subset_batch_v132 import checked_subset
    expected=held_binding(ROOT)
    if c.get('block_clip_brake_v315') != expected:raise ValueError('held support/contact policy binding differs')
    extra={str(ROOT/name):sha for name,sha in expected['sources_sha256'].items()}
    path=ROOT/'our_mani/multitask/joint16_block_fresh_tail_episode_v449.py';extra[str(path)]=file_sha(path)
    from our_mani.multitask.block_fresh_tail_v448 import binding as fresh_binding
    fresh=fresh_binding(ROOT)
    if c.get('block_fresh_tail_v448')!=fresh:raise ValueError('freeze fresh counted tail and dense stopping policy')
    extra.update({str(ROOT/name):sha for name,sha in fresh['sources_sha256'].items()})
    checked_subset(ROOT,files,extra)
    if (c['dt'] != .05 or c['simulation_seconds'] != 10. or c['global_joint_acceleration'] != 6.
            or c['warmup_ticks'] != 5 or c['forecast_seconds'] != 1.1):
        raise ValueError('original block motion/history/clock protocol required')
    rows = [r for r in json.loads(manifest.read_text())['episodes'] if r['task'] == args.task][:4]
    selected = [r for r in rows if r['raw_episode_id'] == args.raw_episode_id]
    if len(selected) != 1: raise ValueError('only fixed first four original TRAIN sources')
    row = selected[0]
    if files.get(row['path']) != file_sha(row['path']): raise ValueError('original HDF5 source differs')
    checkpoint = verified_checkpoint(ROOT, dict(config=c, files=files))
    if c['joint16_context_checkpoint'] != c['context_checkpoint']:
        raise ValueError('block must use the same actual shared16 checkpoint')
    adapter = Joint16BlockContext(ROOT/c['context_checkpoint']['path'], task=args.task, root=ROOT)
    return dict(args=args, config=c, files=files, row=row, out=out,
        model_preflight=checkpoint, actual_block_adapter_binding=adapter.binding,
        identity=dict(task=args.task, method=args.method, split='train', parent=row['raw_episode_id'],
            source_row_sha256=digest(row), config_sha256=file_sha(configuration),
            frozen_inputs_sha256=file_sha(lock), wrapper_sha256=file_sha(__file__),
            original_HDF5_source_sha256=files[row['path']], formal_VAL_TEST=False))


def run(prepared):
    import numpy as np
    import torch
    from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation
    from our_mani.multitask.joint16_block_fresh_tail_episode_v449 import run_TRAIN_development_episode
    args, c, row, out = (prepared[k] for k in ('args', 'config', 'row', 'out'))
    if out.exists(): raise FileExistsError('fresh output required; no native case retry')
    out.mkdir(parents=True)
    atomic_json(out/'inputs.json', dict(identity=prepared['identity'], model_preflight=prepared['model_preflight'],
        actual_block_adapter_binding=prepared['actual_block_adapter_binding']))
    torch.set_num_threads(4)
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed'])
    env = None
    record = None
    try:
        verify_inputs(ROOT, prepared['files'])
        env = AlignedLocalSimulation(row['task'], row['path'], protocol=c, seed=c['seed'], servo_lead=c['local_servo_lead'])
        record, frames = run_TRAIN_development_episode(env, row, c, method=args.method, root=ROOT)
        if frames: raise ValueError('no additional development video requested')
        record.update(identity=deepcopy(prepared['identity']), evaluation_process_pid=os.getpid(),
            split='train', method=args.method, complete=True, infrastructure_error=None,
            counted_clock=dict(ticks=int(env.ledger.ticks), dt=.05, seconds=float(env.time)),
            actual_block_adapter_binding=prepared['actual_block_adapter_binding'],
            automatic_retry=False, TEST_used_for_fitting=False)
    except BaseException as error:
        atomic_json(out/'infrastructure_error.json', dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), counted_as_policy_failure=False, automatic_retry=False))
        raise
    finally:
        if env is not None: env.close()
    verify_inputs(ROOT, prepared['files'])
    record['cleanup_completed'] = True
    atomic_json(out/'case.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'frozen-inputs', 'task', 'raw-episode-id', 'method', 'out', 'split'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--development-demo', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    prepared = preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True, check_only=True, native_started=False, GPU_started=False,
            identity=prepared['identity'], model_preflight=prepared['model_preflight'],
            actual_block_adapter_binding=prepared['actual_block_adapter_binding'])))
        return
    record = run(prepared)
    print(json.dumps(dict(complete=True, native_success=bool(record['official_task_success']),
        ticks=record['counted_clock']['ticks'], case=str(prepared['out']/'case.json'))), flush=True)


if __name__ == '__main__': main()
