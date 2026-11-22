"""Stream released TRAIN HDF5 directly into the official FASTER data loader.

Avoids copying hundreds of thousands of RGB images to a second dataset format.
Only current RGB/state are observations; future demonstrated EE poses are
supervised goal/action labels. No recorded world-model predictions are read.
"""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ('front', '8', '16', '36')


class PhysManiFasterDataset:
    def __init__(self, *, scope='all16', action_horizon=16, split='train', limit_per_task=0):
        if split not in ('train', 'validation'):
            raise ValueError('formal test is not a training data source')
        manifest = json.loads((ROOT/f'artifacts/framework/protocol/{split}_manifest.json').read_text())
        entries = manifest['episodes']
        if scope == 'grasp2':
            entries = [r for r in entries if r['task'] in ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')]
        elif scope != 'all16':
            raise ValueError(scope)
        counts = {}
        self.entries, self.poses, self.opens, self.ends = [], [], [], []
        for row in entries:
            count = counts.get(row['task'], 0)
            if limit_per_task and count >= limit_per_task:
                continue
            with h5py.File(row['path'], 'r') as f:
                poses = f['low_dim_obs/gripper_pose'][:].astype(np.float32)
                opens = f['low_dim_obs/gripper_open'][:].astype(np.float32)
            if len(poses) < 2:
                continue
            poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
            for i in range(1, len(poses)):
                if poses[i-1, 3:]@poses[i, 3:] < 0:
                    poses[i, 3:] *= -1
            self.entries.append(row)
            self.poses.append(poses)
            self.opens.append(opens)
            self.ends.append((self.ends[-1] if self.ends else 0)+len(poses)-1)
            counts[row['task']] = count+1
        self.action_horizon = action_horizon
        self._files = OrderedDict()
        self.scope, self.split = scope, split

    def __len__(self):
        return self.ends[-1]

    def _locate(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        offset = index-(self.ends[episode-1] if episode else 0)
        return episode, offset

    def numeric(self, episode, offset, index):
        poses, opens = self.poses[episode], self.opens[episode]
        indices = np.minimum(offset+np.arange(1, self.action_horizon+1), len(poses)-1)
        actions = np.c_[poses[indices], opens[indices]].astype(np.float32)
        j = min(offset+4+int(index)%13, len(poses)-1)
        velocity = (poses[j, :3]-poses[j-1, :3])/.05
        angular = (Rotation.from_quat(poses[j, 3:])*Rotation.from_quat(poses[j-1, 3:]).inv()).as_rotvec()/.05
        goal = np.r_[poses[j], velocity, angular, [.04, .04, .04], (j-offset)*.05].astype(np.float32)
        return np.r_[poses[offset], opens[offset]].astype(np.float32), actions, goal

    def __getitem__(self, index):
        episode, offset = self._locate(index)
        if episode not in self._files:
            self._files[episode] = h5py.File(self.entries[episode]['path'], 'r')
            if len(self._files) > 8:
                _, old = self._files.popitem(last=False)
                old.close()
        file = self._files[episode]
        self._files.move_to_end(episode)
        state, actions, goal = self.numeric(episode, offset, index)
        return {'observation/images': {c: file[f'observations/{c}/rgb'][offset] for c in CAMERAS},
                'observation/state': state, 'actions': actions, 'handover_goal': goal,
                'prompt': self.entries[episode]['task'].replace('_', ' ')}

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_files'] = OrderedDict()
        return state

    def close(self):
        for file in self._files.values():
            file.close()
        self._files.clear()


def register_dataset_factory():
    from openpi.training import data_loader
    original = data_loader.create_torch_dataset
    def create(config, action_horizon, model_config):
        if config.repo_id and config.repo_id.startswith('hdf5/physmani_train_'):
            scope = config.repo_id.removeprefix('hdf5/physmani_train_')
            return PhysManiFasterDataset(scope=scope, action_horizon=action_horizon)
        return original(config, action_horizon, model_config)
    data_loader.create_torch_dataset = create


def compute_norm_stats(config):
    from openpi.shared import normalize
    scope = config.data.repo_id.removeprefix('hdf5/physmani_train_')
    dataset = PhysManiFasterDataset(scope=scope, action_horizon=config.model.action_horizon)
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    for episode, poses in enumerate(dataset.poses):
        states = np.c_[poses[:-1], dataset.opens[episode][:-1]]
        actions = []
        for offset in range(len(poses)-1):
            _, chunk, _ = dataset.numeric(episode, offset, offset)
            actions.append(chunk)
        state_stats.update(states)
        action_stats.update(np.asarray(actions).reshape(-1, 8))
        if episode % 100 == 0:
            print(f'normalization episodes {episode+1}/{len(dataset.poses)}', flush=True)
    destination = config.assets_dirs/config.data.assets.asset_id
    normalize.save(destination, {'state': state_stats.get_statistics(), 'actions': action_stats.get_statistics()})
    receipt = {'scope': scope, 'split': 'train', 'episodes': len(dataset.entries), 'frames': len(dataset),
               'source': 'released_HDF5_read_only', 'normalization': 'all current states and exact clamped action chunks',
               'formal_test_used': False}
    (destination/'hdf5_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    dataset.close()
