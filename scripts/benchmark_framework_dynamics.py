"""Measure frozen NGFF latency/accuracy vs point count on two existing dev scenes.

These are GSCollision examples already inspected in the dynamics demo, not a
PhysMani validation/test benchmark. CPU/GPU model calls are fully synchronized.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from our_mani.contracts import ObjectState, Pose
from our_mani.dynamics import NGFFDynamics, NGFFFrame


def main():
    torch.set_num_threads(4)
    def gpu_state():
        return subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    gpu_before = gpu_state()
    scenes = {}
    for name in ('typical', 'difficult'):
        with np.load(ROOT/f'artifacts/demos/ngff-20261017/{name}.npz') as z:
            scenes[name] = {k: z[k].copy() for k in ('truth', 'baseline')}
    frame = NGFFFrame(1., 1., np.zeros(3), np.array([1.94, -1.94, 1.94, -1.94, 2.94, -.94]),
                      'original_GSCollision_coordinates_latency_study_only')
    rows = []
    for points in (64, 128, 256, 512, 2048):
        model = NGFFDynamics(ROOT/'artifacts/framework/checkpoints/ngff_best.pt', frame, keypoints=points)
        indices = np.linspace(0, 2047, points, dtype=int)
        for horizon in (.3, .6):
            for name, data in scenes.items():
                initial = data['truth'][0]
                states = [ObjectState(str(i), 0., Pose(cloud.mean(0)), cloud, np.zeros(3), np.zeros(3))
                          for i, cloud in enumerate(initial)]
                f = model.predict(states, horizon, .02)
                frames = len(f.times)
                truth = data['truth'][:frames, :, indices]
                baseline = data['baseline'][:frames, :, indices]
                error = f.points_world[1:]-truth[1:]
                baseline_error = baseline[1:]-truth[1:]
                row = {'scene': name, 'keypoints_per_object': points, 'objects': 3,
                       'horizon_seconds': horizon, 'frames': frames,
                       'latency_seconds': f.metadata['latency_seconds'],
                       'latency_over_horizon': f.metadata['latency_seconds']/horizon,
                       'point_rmse_original_units': float(np.sqrt(np.square(error).mean())),
                       'point_fpe_original_units': float(np.linalg.norm(error[-1], axis=-1).mean()),
                       'gravity_floor_rmse_original_units': float(np.sqrt(np.square(baseline_error).mean()))}
                rows.append(row)
                print(json.dumps(row), flush=True)
        del model
    output = ROOT/'artifacts/framework/dynamics_latency'
    output.mkdir(parents=True, exist_ok=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'GSCollision_development_latency_accuracy_study',
              'scenes': 'two previously inspected examples; not independent test', 'rows': rows,
              'model_checkpoint_sha256': f.metadata['checkpoint_sha256'],
              'physmani_configuration_selected': False, 'torch_cpu_threads': torch.get_num_threads(),
              'gpu_before_util_percent_used_mib_free_mib': gpu_before,
              'gpu_after_util_percent_used_mib_free_mib': gpu_state(),
              'other_baseline_gpu_jobs_running': True,
              'latency_note': 'contended GPU measurement, two calls per point count/horizon; not isolated hardware throughput'}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (output/'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
