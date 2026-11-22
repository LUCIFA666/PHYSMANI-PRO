"""Convert locked TRAIN demonstrations to local LeRobot v2.1, never upload.

Future EE poses are supervised goal/action labels, not observations. This is an
initial goal-reaching BC dataset; it does not provide calibrated success Value
labels or establish a trained local capture region.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='local/our_mani_physmani_grasp2')
    parser.add_argument('--limit-episodes', type=int, default=0)
    parser.add_argument('--goal-steps', type=int, default=16)
    parser.add_argument('--scope', choices=('grasp2', 'all16'), default='grasp2')
    args = parser.parse_args()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
    manifest_path = ROOT/'artifacts/framework/protocol/train_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'train':
        raise RuntimeError('conversion must use training split only')
    tasks = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
    if args.scope == 'all16':
        tasks = tuple(sorted({r['task'] for r in manifest['episodes']}))
    rows = [r for r in manifest['episodes'] if r['task'] in tasks]
    if args.limit_episodes:
        # Balanced task selection for a training smoke, rather than sorted-prefix bias.
        rows = [r for task in tasks for r in [s for s in rows if s['task'] == task][:args.limit_episodes]]
    destination = Path(HF_LEROBOT_HOME)/args.repo_id
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite converted dataset: {destination}')
    features = {f'observation.images.{camera}': {'dtype': 'image', 'shape': (256, 256, 3),
                  'names': ['height', 'width', 'channel']} for camera in ('front', '8', '16', '36')}
    features.update({'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': ['state']},
                     'actions': {'dtype': 'float32', 'shape': (8,), 'names': ['action']},
                     'handover_goal': {'dtype': 'float32', 'shape': (17,), 'names': ['goal']}})
    dataset = LeRobotDataset.create(repo_id=args.repo_id, robot_type='panda', fps=20, features=features,
                                    use_videos=False, image_writer_threads=4, image_writer_processes=0)
    converted, frames = [], 0
    for row in rows:
        with h5py.File(row['path'], 'r') as f:
            poses = f['low_dim_obs/gripper_pose'][:].astype(np.float32)
            opens = f['low_dim_obs/gripper_open'][:].astype(np.float32)
            # Continuous quaternion signs are a representation convention, not an oracle input.
            for i in range(1, len(poses)):
                if poses[i-1, 3:]@poses[i, 3:] < 0:
                    poses[i, 3:] *= -1
            if len(poses) < 2:
                continue
            for i in range(len(poses)-1):
                j = min(i+args.goal_steps, len(poses)-1)
                before = max(0, j-1)
                elapsed = (j-before)*.05
                velocity = (poses[j, :3]-poses[before, :3])/elapsed
                turn = Rotation.from_quat(poses[j, 3:])*Rotation.from_quat(poses[before, 3:]).inv()
                angular = turn.as_rotvec()/elapsed
                goal = np.r_[poses[j], velocity, angular, [.04, .04, .04], (j-i)*.05].astype(np.float32)
                frame = {f'observation.images.{camera}': f[f'observations/{camera}/rgb'][i]
                         for camera in ('front', '8', '16', '36')}
                frame.update({'observation.state': np.r_[poses[i], opens[i]].astype(np.float32),
                              'actions': np.r_[poses[i+1], opens[i+1]].astype(np.float32),
                              'handover_goal': goal, 'task': row['task'].replace('_', ' ')})
                dataset.add_frame(frame)
                frames += 1
        dataset.save_episode()
        converted.append(row)
        print(f'converted {len(converted)}/{len(rows)} episodes; {frames} frames', flush=True)
    dataset.stop_image_writer()
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'repo_id': args.repo_id,
               'dataset_path': str(destination), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'episodes': converted, 'frames': frames, 'split': 'train', 'uploaded': False,
               'goal_label': f'future demonstrated EE pose at up to {args.goal_steps} frames; supervised label only',
               'action_label': 'next demonstrated EE world xyz xyzw and gripper-open, 20 Hz',
               'kind': 'goal_reaching_behavior_cloning_initialization_not_local_value_training'}
    receipt['training_scope'] = args.scope
    receipt['formal_full_system_comparison_requires_matched_all16_training_data'] = True
    out = ROOT/'artifacts/framework/faster_data'/f'{args.repo_id.replace("/", "_")}_receipt.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
