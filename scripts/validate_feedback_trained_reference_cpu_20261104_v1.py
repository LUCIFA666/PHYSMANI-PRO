"""Real public Native feedback -> existing trained neural reference, CPU only.

All16 Native interface workers must be actually closed before this diagnostic.
No task truth, success label, or expert action is read as a model input.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def pointer(path): return dict(path=str(Path(path).resolve(strict=True)), sha256=sha(path))


def json_value(value):
    if hasattr(value, 'tolist'): return value.tolist()
    raise TypeError('Unsupported evidence value: ' + type(value).__name__)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False,
            default=json_value)
        stream.write('\n')


def native_gate(root):
    completion = read(root / 'completion.json')
    if completion.get('complete') is not True or completion.get('all16_native_closed') is not True or completion.get('all16_actual_worker_wait_completed') is not True:
        raise ValueError('Actual entire Native cohort completion and shutdown required')
    if len(completion.get('cases', [])) != 16: raise ValueError('All16 fixed cases required')
    cases = []
    for row in completion['cases']:
        path = Path(row['actual_exit']['path'])
        if sha(path) != row['actual_exit']['sha256']: raise ValueError('Owner receipt changed')
        owner = read(path)
        if owner.get('actual_process_returncode') != 0 or owner.get('actual_wait_completed') is not True or owner.get('source_unchanged') is not True or owner.get('unreaped_descendants'):
            raise ValueError('Actual clean waited Native owner required')
        case = Path(row['result']['path']).parent
        cases.append(dict(case_id=row['case_id'], task=row['task'], path=case))
    request = read(root / 'request.json')
    source_pointer = request['source_map']
    if sha(source_pointer['path']) != source_pointer['sha256']: raise ValueError('Native source map changed')
    sources = read(source_pointer['path'])['source_sha256']
    if any(sha(name) != digest for name, digest in sources.items()): raise ValueError('Native sources changed')
    return completion, cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    completion, cases = native_gate(args.native_root)
    if not args.execute:
        print(json.dumps(dict(complete=True, cases=16, NN_forward_calls=0, Native_started=False)))
        return 0
    if args.output is None: p.error('Fresh output required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from our_mani.trained_world_reference_live_20261104_v1.runtime import TrainedWorldReferenceLiveV1
    from our_mani.generic_reference_mpc_20261104_v1.public_observation import pack_public_observation
    import numpy as np
    import torch
    torch.set_num_threads(2)
    write_new(out / 'request.json', dict(native_completion=pointer(args.native_root / 'completion.json'),
        script=pointer(__file__), fixed_frame_indices=[0, 1, 2], all16_cases=True,
        device='cpu', task_policy_actuated=False, Native_started=False))
    started = time.monotonic()
    bridge = TrainedWorldReferenceLiveV1(device='cpu')
    rows = []
    for case in cases:
        bridge.clear_history()
        for index in (0, 1, 2):
            frame = case['path'] / 'public_frames' / f'{index:03d}.npz'
            if not frame.exists():
                rows.append(dict(case_id=case['case_id'], task=case['task'], frame_index=index,
                    passed=False, reason='Native_early_terminal_frame_not_recorded', NN_called=False))
                continue
            try:
                meta = read(frame.with_suffix('.json'))
                if sha(frame) != meta['sha256']: raise ValueError('Public frame changed')
                with np.load(frame, allow_pickle=False) as arrays:
                    packet = pack_public_observation(timestamp=meta['timestamp'],
                        rgb=arrays['rgb'], depth_m=arrays['depth_m'], K=arrays['K'], T=arrays['T'],
                        camera_clips=arrays['camera_clips'], q=arrays['q'], tcp_pose=arrays['tcp_pose'])
                # Only this strict public packet is passed to the trained bridge.
                reference = bridge.predict_reference(packet, frame_index=index)
                value = asdict(reference)
                row = dict(case_id=case['case_id'], task=case['task'], frame_index=index,
                    public_frame=pointer(frame), passed=True, reference=value,
                    inference=bridge.last_private_diagnostic,
                    opening_intent_issued=False, task_success_evaluated=False)
            except BaseException as error:
                row = dict(case_id=case['case_id'], task=case['task'], frame_index=index,
                    public_frame=pointer(frame), passed=False, error_type=type(error).__name__, error=str(error))
            rows.append(row)
        write_new(out / (case['case_id'] + '.json'), rows[-3:])
        print(json.dumps(dict(event='case_finished', case_id=case['case_id'], passed=sum(r['passed'] for r in rows[-3:]), frames=3)), flush=True)
    native_gate(args.native_root)
    report = dict(complete=True, cases=16, rows=len(rows), passed=sum(r['passed'] for r in rows),
        encoder_forward_calls=bridge.encoder_forward_calls, world_forward_calls=bridge.world_forward_calls,
        per_task={c['task']:dict(frames=3, passed=sum(r['passed'] for r in rows if r['task'] == c['task'])) for c in cases},
        wall_seconds=time.monotonic()-started, device='cpu', GPU_used=False, Native_started=False,
        real_public_feedback_used=True, original_large_encoder_compatibility_test=True,
        compact_task_branch_tested=False, task_policy_actuated=False,
        task_success_rate_evaluated=False, opening_intent_issued=False)
    write_new(out / 'report.json', report)
    print(json.dumps(report))
    return 0


if __name__ == '__main__': raise SystemExit(main())
