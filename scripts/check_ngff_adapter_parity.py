"""Compare the framework adapter against previously saved real GPU demo rollouts.

Uses original GSCollision coordinate units. This check establishes code parity,
not physical SI calibration or transfer to manipulation tasks.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from our_mani.contracts import ObjectState, Pose
from our_mani.dynamics import NGFFDynamics, NGFFFrame


def main():
    torch.set_num_threads(4)
    frame = NGFFFrame(1., 1., np.zeros(3), np.array([1.94, -1.94, 1.94, -1.94, 2.94, -.94]),
                      'original_GSCollision_coordinates_parity_only_not_SI_calibration')
    model = NGFFDynamics(ROOT/'artifacts/framework/checkpoints/ngff_best.pt', frame, keypoints=2048)
    rows = []
    for name in ('typical', 'difficult'):
        with np.load(ROOT/f'artifacts/demos/ngff-20261017/{name}.npz') as z:
            first, reference = z['truth'][0].copy(), z['ngff'].copy()
        states = [ObjectState(str(i), 0., Pose(points.mean(0)), points, np.zeros(3), np.zeros(3))
                  for i, points in enumerate(first)]
        prediction = model.predict(states, horizon=1.58, dt=.02)
        difference = prediction.points_world-reference
        row = {'scene': name, 'frames': len(prediction.times), 'objects': 3, 'points_per_object': 2048,
               'maximum_coordinate_difference_original_units': float(np.max(np.abs(difference))),
               'coordinate_rmse_original_units': float(np.sqrt(np.mean(difference**2))),
               'inference_seconds': prediction.metadata['latency_seconds']}
        # Centroid rounding and GPU neighbour ordering can perturb contact
        # integration, so parity is tested numerically with an explicit tolerance.
        row['passed'] = row['coordinate_rmse_original_units'] < 1e-4
        rows.append(row)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'original_gpu_rollout_adapter_parity',
              'checkpoint_sha256': model.checkpoint_sha256, 'rows': rows,
              'passed': all(r['passed'] for r in rows), 'physmani_transfer_demonstrated': False}
    destination = ROOT/'artifacts/framework/smoke/ngff_parity_report.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
