"""Calibrate visible marker frame on TRAIN, then report held-out RGB-D accuracy."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.goal_perception import RedGoalMarker
from our_mani.physmani_data import RecordedEpisode

TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')


def samples(split, limit):
    rows = json.loads((ROOT/f'artifacts/framework/protocol/{split}_manifest.json').read_text())['episodes']
    counts = {}
    for row in rows:
        if row['task'] not in TASKS or counts.get(row['task'], 0) >= limit:
            continue
        with RecordedEpisode(row['path']) as episode:
            stages = episode.file['low_dim_obs/expert_trajectory_stage'][:].astype('U').reshape(-1)
            indices = np.flatnonzero(np.char.lower(stages) == 'lift')
            if not len(indices):
                continue
            truth = episode.file['low_dim_obs/expert_trajectory'][indices[0]].reshape(-1)[:3]
            try:
                marker, _ = RedGoalMarker((0, 0, 0)).observe(episode.observation(0))
            except ValueError:
                continue
            counts[row['task']] = counts.get(row['task'], 0)+1
            yield {'task': row['task'], 'episode': row['raw_episode_id'],
                   'visual_center': marker.position, 'label_goal': truth}


def main():
    train = list(samples('train', 8))
    offset = np.median([r['label_goal']-r['visual_center'] for r in train], axis=0)
    validation = list(samples('validation', 20))
    errors = [np.linalg.norm(r['visual_center']+offset-r['label_goal']) for r in validation]
    record = {'created_utc': datetime.now(timezone.utc).isoformat(), 'offset_train_calibrated_m': offset.tolist(),
              'train_episodes': len(train), 'validation_episodes': len(validation),
              'validation_mean_error_m': float(np.mean(errors)), 'validation_p95_error_m': float(np.quantile(errors, .95)),
              'formal_test_used': False, 'inputs': 'current RGB-D only',
              'labels': 'TRAIN expert lift goal for calibration; validation lift goal for metrics only'}
    out = ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
