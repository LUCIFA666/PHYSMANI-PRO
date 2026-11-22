"""Pure compiler for separated public histories and private future labels.

This module does not open a simulator, checkpoint, image, HDF or TEST source.
The capture owner projects its public target rows into PUBLIC_KEYS. Complete raw
capture rows and failures remain with that owner; this compiler returns a ledger
for every input row and never replaces missing observations using labels.
"""
from collections import Counter
import numpy as np

DATA_SCHEMA = 'Cube_real50ms_public_history_private_future_20261104_v1'
PUBLIC_KEYS = frozenset(('timestamp', 'available', 'position_world', 'confidence'))
LABEL_KEYS = frozenset(('timestamp', 'target_position_world', 'free_motion', 'segment_id'))


def _time(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative measured timestamp required')
    return float(value)


def compile_episode(public_rows, label_rows, *, parent_id):
    """Compile one physical episode, never combining resets/parents.

    A future validity mask ends at the first missing/nonfree/different-segment
    label. Future observation validity is independent of future GT availability:
    a later visual miss does not erase a valid *training label*. The input anchor
    still requires five actually visible consecutive past/current observations.
    """
    if type(parent_id) is not str or not parent_id or not isinstance(public_rows, list) or not public_rows:
        raise ValueError('one named complete public episode required')
    if not isinstance(label_rows, list):
        raise ValueError('separate private training-label rows required')
    previous = None
    for row in public_rows:
        if not isinstance(row, dict) or set(row) != PUBLIC_KEYS or type(row['available']) is not bool:
            raise ValueError('exact public target whitelist required; no labels/Native channels')
        now = _time(row['timestamp'])
        if previous is not None and abs(now-previous-.05) > 1e-6:
            raise ValueError('complete actual50ms public ledger required; include unavailable rows')
        previous = now
        if row['available']:
            p = np.asarray(row['position_world'])
            c = row['confidence']
            if (p.shape != (3,) or not np.isfinite(p).all() or isinstance(c, (bool,np.bool_))
                    or not np.isscalar(c) or not np.isfinite(c) or not 0 < c <= 1):
                raise ValueError('available public target needs finite position/confidence')
        elif row['position_world'] is not None or row['confidence'] is not None:
            raise ValueError('unavailable target remains null, not GT/forecast-filled')
    labels = {}
    previous = None
    for row in label_rows:
        if (not isinstance(row, dict) or set(row) != LABEL_KEYS or type(row['free_motion']) is not bool
                or type(row['segment_id']) not in (str,int) or isinstance(row['segment_id'],bool)):
            raise ValueError('exact independent private future-label contract required')
        now = _time(row['timestamp'])
        if previous is not None and now <= previous:
            raise ValueError('private actual label timestamps must be unique/causal')
        previous = now
        p = np.asarray(row['target_position_world'])
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('private measured target xyz required')
        tick = int(round((now-public_rows[0]['timestamp'])/.05))
        expected = public_rows[0]['timestamp'] + tick*.05
        if abs(now-expected) > 1e-6 or tick in labels:
            raise ValueError('labels must refer to actual public physical-clock query grid')
        labels[tick] = row
    windows, ledger = [], []
    maximum_label_query_alignment_seconds = 0.
    visible = []
    for tick, row in enumerate(public_rows):
        now = float(row['timestamp'])
        if not row['available']:
            visible.clear()
            ledger.append(dict(parent_id=parent_id,timestamp=now,eligible=False,reason='current_public_unavailable'))
            continue
        visible.append(row)
        visible = visible[-20:]
        current_label = labels.get(tick)
        reason = ('fewer_than_five_current_visible_frames' if len(visible)<5 else
                  'current_private_free_label_unavailable' if current_label is None else
                  'current_not_externally_free' if not current_label['free_motion'] else None)
        if reason:
            ledger.append(dict(parent_id=parent_id,timestamp=now,eligible=False,reason=reason))
            continue
        target = np.full((17,3), np.nan, dtype=np.float32)
        valid = np.zeros(17, dtype=bool)
        # Nominal values are query/padding placeholders, never claimed as
        # measured label times. Every available label keeps its actual clock.
        future_times = now + np.arange(17, dtype=np.float64)*.05
        future_times[0] = float(current_label['timestamp'])
        for future in range(1,17):
            label = labels.get(tick+future)
            if (label is None or not label['free_motion'] or label['segment_id'] != current_label['segment_id']):
                break
            target[future] = label['target_position_world']
            valid[future] = True
            future_times[future] = float(label['timestamp'])
            maximum_label_query_alignment_seconds = max(maximum_label_query_alignment_seconds,
                abs(future_times[future]-(now+future*.05)))
        if not valid.any():
            ledger.append(dict(parent_id=parent_id,timestamp=now,eligible=False,reason='no_uninterrupted_free_future_label'))
            continue
        n = len(visible)
        positions = np.zeros((20,3),dtype=np.float32)
        timestamps = np.zeros(20,dtype=np.float64)
        confidence = np.zeros(20,dtype=np.float32)
        positions[:n] = [item['position_world'] for item in visible]
        timestamps[:n] = [item['timestamp'] for item in visible]
        confidence[:n] = [item['confidence'] for item in visible]
        windows.append(dict(parent_id=parent_id, anchor_time=now,
            inputs=dict(positions=positions,timestamps=timestamps,confidence=confidence,lengths=np.int64(n)),
            targets=dict(target_positions=target,target_valid=valid,target_times=future_times)))
        ledger.append(dict(parent_id=parent_id,timestamp=now,eligible=True,history_frames=n,
            valid_future_frames=int(valid.sum()),reason=None))
    return windows, dict(schema=DATA_SCHEMA,parent_id=parent_id,rows=len(public_rows),
        windows=len(windows),ledger=ledger,reasons=dict(Counter(row['reason'] for row in ledger if row['reason'])),
        maximum_actual_label_vs_nominal_query_alignment_seconds=maximum_label_query_alignment_seconds,
        valid_target_times='actual private label timestamps; nominal query placeholders only where mask is false',
        future_labels_are_inputs=False,interpolated_observations=0,heldout_role_decision='caller must enforce whole-parent registry')
