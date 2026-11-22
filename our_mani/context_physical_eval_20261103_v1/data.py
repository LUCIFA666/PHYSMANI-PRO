"""Compile real public histories only; do not open Native/evaluation labels."""
from pathlib import Path
from collections import Counter
import json

import numpy as np

from ..context_gap_dropout_20261103_v1.data import ROOT, Window, sha

INVENTORY = ROOT/'artifacts/framework/local_mpc_20261102/P4_TRAIN16_public_physical_clock_inventory_v1/receipt.json'
INVENTORY_SHA = '314f030b07796e817e9e7f5c33c7ba1001e2b61196bdbc8d876aaba89b8bce6d'
CORRECTION = INVENTORY.with_name('GT_read_scope_correction.json')
CORRECTION_SHA = 'be3a928d1dd3838bb7e504d2e430e31d04935666f1b4679287613cdeb17a232c'
PLAN = ROOT/'config/framework/P4_teacher_train_20261103_v1/plan.json'


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def compile_public_parent(parent_id, rows):
    """Every frame retained in ledger; anchor eligibility never reads a label.

    Future Window fields contain later public observations only. A missing or
    manipulated future terminates the evaluation mask, not the anchor cohort.
    """
    measured, ledger, windows = [], [], []
    previous = None
    for index, row in enumerate(rows):
        if set(row) != {'index', 'result', 'public_robot', 'close_requested', 'rgbd'}:
            raise ValueError('Exact original public observer row required')
        result, robot = row['result'], row['public_robot']
        now = result['timestamp']
        if (isinstance(now, bool) or not isinstance(now, (int, float)) or not np.isfinite(now)
                or abs(float(robot['timestamp'])-now) > 1e-8
                or previous is not None and abs(now-previous-.05) > 1e-6):
            raise ValueError('Actual synchronized contiguous50ms public/robot clock required')
        if type(row['close_requested']) is not bool:
            raise ValueError('Actual command intent boolean required')
        for key in ('Native_inputs_used', 'future_observations_used', 'observations_fabricated', 'forecaster_is_injected_test_double'):
            if result.get(key) is not False:
                raise ValueError('Native/predicted/fabricated inputs forbidden: '+key)
        previous = now
        obj = result.get('object_observation')
        eligible = (result['mode'] == 'FREE' and not row['close_requested'] and obj is not None
                    and obj.get('visible', True) is True and obj.get('confidence', 1.) >= .5)
        parsed = None
        if eligible:
            if obj.get('object_id') != 'target' or abs(obj['timestamp']-now) > 1e-8:
                raise ValueError('Only the same-frame current observed target may enter history')
            p, r = np.asarray(obj['pose']['position'], float), np.asarray(obj['pose']['rotation'], float)
            v, w = np.asarray(obj['linear_velocity'], float), np.asarray(obj['angular_velocity'], float)
            if (p.shape != (3,) or r.shape != (3, 3) or v.shape != (3,) or w.shape != (3,)
                    or not all(np.isfinite(x).all() for x in (p, r, v, w))
                    or not np.allclose(r.T@r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .9999):
                raise ValueError('Finite current observed proper-SE3 pose/twist required')
            if type(result['rotation_observable']) is not bool:
                raise ValueError('Current orientation observability must be explicit')
            parsed = dict(time=now, position=p, rotation=r, velocity=v, angular=w,
                          confidence=float(obj['confidence']), observable=result['rotation_observable'])
        measured.append(parsed)
        ledger.append(dict(index=index, time=now, mode=result['mode'], close_requested=row['close_requested'],
            current_visible=obj is not None and obj.get('visible', True), eligible_free_current=eligible,
            dense_history_anchor=False, reason='eligible_current' if eligible else 'not_current_open_FREE_observation'))
    for anchor, current in enumerate(measured):
        if current is None:
            continue
        begin = anchor
        while begin > max(0, anchor-19) and measured[begin-1] is not None and current['time']-measured[begin-1]['time'] <= 1.+1e-8:
            begin -= 1
        if anchor-begin+1 < 5:
            ledger[anchor]['reason'] = 'current_visible_but_fewer_than5_contiguous_real_history_frames'
            continue
        history = measured[begin:anchor+1]
        fp, fr, valid = np.zeros((17, 3), np.float32), np.tile(np.eye(3), (17, 1, 1)).astype(np.float32), np.zeros(17, bool)
        missing_reason = 'complete_public800ms_future'
        for step in range(17):
            if anchor+step >= len(measured):
                missing_reason = 'capture_ended_before800ms'
                break
            future = measured[anchor+step]
            if future is None:
                missing_reason = 'public_unknown_or_not_open_FREE_future_boundary'
                break
            fp[step], fr[step], valid[step] = future['position'], future['rotation'], step > 0
        window = Window((parent_id, 'target'), 'physical_TRAIN_development', anchor,
            np.asarray([r['time'] for r in history]), np.stack([r['position'] for r in history]),
            np.stack([r['rotation'] for r in history]), np.stack([r['velocity'] for r in history]),
            np.stack([r['angular'] for r in history]), np.asarray([r['confidence'] for r in history]),
            np.asarray([r['observable'] for r in history], bool), fp, fr, valid)
        windows.append(window)
        ledger[anchor].update(dense_history_anchor=True, dense_history_times=window.times.tolist(),
            public_future_valid_steps=int(valid.sum()), public_future_complete800ms=bool(valid[1:].all()),
            public_future_missing_reason=missing_reason)
    return windows, ledger


def load_public_dataset():
    if sha(INVENTORY) != INVENTORY_SHA or sha(CORRECTION) != CORRECTION_SHA:
        raise ValueError('Exact Root physical-clock inventory and disclosure correction required')
    inventory = json.loads(INVENTORY.read_text())
    sources = inventory['source_sha256']
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Closed source/public evidence changed')
    if sources.get(str(PLAN)) != sha(PLAN):
        raise ValueError('Exact preassigned original TRAIN16 plan required')
    jobs = json.loads(PLAN.read_text())['jobs']
    if len(jobs) != 16 or len(set(job['parent_id'] for job in jobs)) != 16 or any(job['split'] != 'train' for job in jobs):
        raise ValueError('All16 preassigned TRAIN parents required; no subset/success filtering')
    windows, ledgers, parent_info, cases = [], {}, [], {}
    for job, expected in zip(jobs, inventory['parents']):
        case = ROOT/job['path']
        path = case/'public_shadow/public_outputs.jsonl'
        if job['parent_id'] != expected['parent_id'] or expected['all_actual_exit0_and_nativeclosed'] is not True:
            raise ValueError('Exact closed parent order required')
        worker = json.loads((case/'worker_exit.json').read_text())
        if type(worker['exit_code']) is not int or worker['exit_code'] != 0 or worker['native_closed'] is not True:
            raise ValueError('Actual closed0 required before development evaluation')
        rows = read_rows(path)
        own_windows, ledger = compile_public_parent(job['parent_id'], rows)
        counts = dict(frames=len(rows), eligible_current_public_free_frames=sum(r['eligible_free_current'] for r in ledger),
            dense5_history_anchors=len(own_windows), anchors_with_complete800ms_public_future=sum(w.future_valid[1:].all() for w in own_windows))
        if any(int(counts[key]) != expected[key] for key in counts):
            raise ValueError('Compiled public cohort differs from predeclared Root inventory')
        windows.extend(own_windows)
        ledgers[job['parent_id']] = ledger
        parent_info.append(dict(parent_id=job['parent_id'], scene=job['scene'], seed=job['seed'], **{k:int(v) for k,v in counts.items()}))
        cases[job['parent_id']] = dict(case=str(case), scene=job['scene'], public_rows=rows)
    if (len(windows), sum(len(v) for v in ledgers.values())) != (969, 1300):
        raise ValueError('Exact full969 anchors/1300 frames required')
    identity = dict(inventory=dict(path=str(INVENTORY), sha256=INVENTORY_SHA),
        inventory_GT_scope_correction=dict(path=str(CORRECTION), sha256=CORRECTION_SHA),
        public_and_closed_source_sha256=sources, parent_ids=[job['parent_id'] for job in jobs],
        parents=parent_info, physical_clock_evidence='archived Native measured tick deltas and passive-shadow no-time-advance binding; original Root audit',
        old_teacher_and_TF_prior_development_exposure_not_claimed_independent=True,
        only_ordinary_pick_task=True, high_speed_task_evaluated=False, qualification=False)
    return windows, ledgers, cases, identity
