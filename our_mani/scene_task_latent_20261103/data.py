"""First real frame from author TRAIN only; grouped internal split before fit.

Only RGB/depth index0 and public camera parameters are read from HDF. Paths and
task attrs supply offline labels, never forward tensors. Expert actions/stages,
robot state and wm_predictions are not opened. No HDF clock is invented.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from ..hierarchy_20261030.data import decode_metric_depth
from .model import TASKS

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path('/home/ubuntu/桌面/phsimani/PhysMani/data/rmt/physmani_bench/train_package_compressed')
CAMERAS = ('front', '8', '16', '36')
SCHEMA = 'scene_task_firstframe_TRAIN_grouped_internal_split_20261103_v1'
SPLITS = ('fit', 'internal_val')
DEFAULT_SEED = 20261103


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([key, list(value.shape), value.dtype.str]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def semantic_task(task):
    family = task.removesuffix('_high_speed')
    if family not in TASKS or task not in (family, family + '_high_speed'):
        raise ValueError('exact declared two semantic tasks and their speed variants required')
    return family


def _episode_number(value):
    if not isinstance(value, str) or not value.startswith('episode') or not value[7:].isdigit():
        raise ValueError('real raw episodeN identity required, not sorted-list ordinal')
    return int(value[7:])


def assign_splits(rows, seed=DEFAULT_SEED):
    """Union same semantic/raw ID across speed and exact observed duplicates.

    Strata are semantic family and each union group's speed composition. The
    target is20% of parents in each stratum, never independent per-speed splits.
    """
    if type(seed) is not int or not rows:
        raise ValueError('nonempty TRAIN inventory and explicit integer split seed required')
    rows = [dict(row) for row in rows]
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def join(i, j):
        a, b = find(i), find(j)
        if a != b: parents[max(a, b)] = min(a, b)
    seen_parent, seen_hash, identities = {}, {}, set()
    for i, row in enumerate(rows):
        family = semantic_task(row['task']); number = _episode_number(row['raw_episode_id'])
        identity = (row['task'], number)
        if identity in identities: raise ValueError('duplicate task/source identity')
        identities.add(identity)
        row.update(semantic_task=family, semantic_label=TASKS.index(family),
                   speed='high' if row['task'].endswith('_high_speed') else 'normal', source_frame_index=0)
        key = (family, number)
        if key in seen_parent: join(i, seen_parent[key])
        seen_parent[key] = i
        for field in ('public_input_sha256', 'cached_input_sha256'):
            fingerprint = row.get(field)
            if fingerprint is not None:
                if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                        or any(c not in '0123456789abcdef' for c in fingerprint)):
                    raise ValueError('explicit SHA256 public input identity required')
                if (field, fingerprint) in seen_hash:
                    previous = seen_hash[field, fingerprint]
                    if rows[previous]['semantic_label'] != row['semantic_label']:
                        raise ValueError('identical public input has contradictory semantic labels; no fit/drop shortcut')
                    join(i, previous)
                seen_hash[field, fingerprint] = i
    groups = defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    strata = defaultdict(list)
    for members in groups.values():
        family = rows[members[0]]['semantic_task']
        key = family + ':' + ','.join(str(v) for v in sorted({_episode_number(rows[i]['raw_episode_id']) for i in members}))
        speeds = tuple(sorted({rows[i]['speed'] for i in members}))
        strata[(family, speeds)].append((key, members))
    for (family, speeds), entries in sorted(strata.items()):
        entries.sort(key=lambda value: hashlib.sha256(f'{seed}:{value[0]}'.encode()).hexdigest())
        number_dev = int(np.floor(len(entries) * .2 + .5))
        if len(entries) > 1: number_dev = min(len(entries)-1, max(1, number_dev))
        for order, (key, members) in enumerate(entries):
            for i in members:
                rows[i].update(parent_group=key, split='internal_val' if order < number_dev else 'fit')
    for split in SPLITS:
        if set(row['semantic_label'] for row in rows if row['split'] == split) != {0, 1}:
            raise ValueError('both semantic labels required in each parent-grouped internal split')
    split_parents = {split: {r['parent_group'] for r in rows if r['split'] == split} for split in SPLITS}
    if split_parents['fit'] & split_parents['internal_val']:
        raise AssertionError('parent leakage')
    summary = dict(seed=seed, requested_fit_fraction=.8, algorithm='semantic+speed-composition strata; SHA256 rank; nearest20percent parents internal_val',
        parent_definition='base semantic family + real episodeN across normal/high, union exact frame0 raw or64x64 input duplicates',
        duplicate_hashes_checked=all(all(k in r for k in ('public_input_sha256', 'cached_input_sha256')) for r in rows),
        source_count=len(rows), parent_count=len(groups),
        counts={split: dict(sources=sum(r['split'] == split for r in rows), parents=len(split_parents[split]),
            strata=dict(Counter(r['task'] for r in rows if r['split'] == split))) for split in SPLITS})
    return rows, summary


def read_first_public_rgbd(path, task):
    """Read only approved sensor datasets at index0; no task attr is a feature."""
    semantic_task(task)
    path = Path(path)
    raw, rgb, depth, calibration, accesses = {}, [], [], {}, []
    before = path.stat()
    with h5py.File(path, 'r') as source:
        if str(source.attrs['task']) != task:
            raise ValueError('offline source label differs from source task attr')
        if int(source.attrs['episode']) != _episode_number(path.parent.name):
            raise ValueError('source parent identity differs from HDF attr')
        for camera in CAMERAS:
            for field in ('rgb', 'depth'):
                key = f'observations/{camera}/{field}'; stream = source[key]
                if stream.dtype != np.uint8 or stream.shape[1:] != (256, 256, 3) or stream.shape[0] < 1:
                    raise ValueError('original256 uint8 RGB/RGB24 depth stream required')
                raw[camera + '/' + field] = stream[0]
                accesses.append(key + '[0]')
            params = {}
            for field in ('near', 'far', 'intrinsics', 'extrinsics'):
                key = f'cam_params/{camera}/{field}'; params[field] = np.asarray(source[key][()]).tolist()
                accesses.append(key + '[()]')
            calibration[camera] = params
            metric = decode_metric_depth(raw[camera + '/depth'], params['near'], params['far'])
            rgb.append(np.asarray(Image.fromarray(raw[camera + '/rgb']).resize((64, 64), Image.Resampling.BILINEAR)))
            depth.append(np.asarray(Image.fromarray(metric).resize((64, 64), Image.Resampling.NEAREST), dtype=np.float32))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('source changed during first-frame read')
    arrays = dict(rgb=np.stack(rgb), depth=np.stack(depth))
    if not np.isfinite(arrays['depth']).all() or not np.all(arrays['depth'] > 0):
        raise ValueError('positive finite metric depth required')
    audit = dict(source_frame_index=0, real_observation_frame=True, public_input_sha256=array_digest(raw),
        cached_input_sha256=array_digest(arrays), calibration_sha256=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest(),
        source_size_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
        HDF_value_accesses=accesses, HDF_attrs_accessed=['task','episode'],
        future_frames_read=False, expert_or_robot_datasets_read=False, camera_parameters_are_forward_features=False)
    return {**arrays, 'audit': audit}


def prepare_dataset(output, *, train_root=TRAIN_ROOT, seed=DEFAULT_SEED):
    if Path(train_root).resolve() != TRAIN_ROOT.resolve():
        raise ValueError('only the predeclared author TRAIN root is permitted, never official VAL/TEST')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    all_rgb, all_depth, rows = [], [], []
    for family in TASKS:
        for task in (family, family + '_high_speed'):
            files = sorted((Path(train_root)/task/'all_variations/episodes').glob('episode*/data.h5'), key=lambda p: _episode_number(p.parent.name))
            if len(files) != 100: raise ValueError('exact100 TRAIN files per task/speed variant required')
            for path in files:
                values = read_first_public_rgbd(path, task)
                rows.append(dict(index=len(rows),task=task,raw_episode_id=path.parent.name,source_path=str(path.resolve()),**values['audit']))
                all_rgb.append(values['rgb']); all_depth.append(values['depth'])
    rows, split = assign_splits(rows, seed)
    cache = output/'dataset.npz'
    with cache.open('xb') as stream: np.savez_compressed(stream,rgb=np.stack(all_rgb),depth=np.stack(all_depth))
    sources = list((ROOT/'our_mani/scene_task_latent_20261103').glob('*.py')) + [
        ROOT/'our_mani/hierarchy_20261030/data.py', ROOT/'scripts/prepare_scene_task_latent_20261103.py',
        ROOT/'scripts/train_scene_task_latent_20261103.py',ROOT/'configs/scene_task_latent_20261103/pretrain_v1.json']
    manifest = dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),complete=True,
        split=split,rows=rows,dataset_sha256=sha256(cache),source_sha256={str(p):sha256(p) for p in sources},
        input_arrays=['rgb','depth'],input_shape='N,4views,4channels,64,64 after fixed normalization',
        preprocessing=dict(rgb='uint8/255-.5',depth='decode original RGB24 to meters, nearest64, log1p',
                           normalization_fitted=False,all_statistics_fit_only=True),
        official_VAL_or_TEST_opened=False,Native_or_fit_started=False,
        limitations=['Two benchmark scenes have different camera calibration; no explicit calibration or cameraID enters forward, but view appearance shortcut remains.',
          'Scene family does not disambiguate a different requested intention within the same scene.',
          'This e is separate from motion c and cannot directly replace frozen single-task uniform-e P4 inputs.'])
    write_new(output/'manifest.json',manifest)
    return manifest


def load_prepared(output):
    output = Path(output);manifest = json.loads((output/'manifest.json').read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('complete') is not True
            or manifest.get('official_VAL_or_TEST_opened') is not False
            or sha256(output/'dataset.npz') != manifest['dataset_sha256']):
        raise ValueError('exact complete TRAIN-only prepared dataset identity required')
    for path, digest in manifest['source_sha256'].items():
        if sha256(path) != digest: raise ValueError('prepared source contract changed: ' + path)
    with np.load(output/'dataset.npz',allow_pickle=False) as data:
        if set(data.files) != {'rgb','depth'}: raise ValueError('only sensor arrays may enter cache')
        arrays = {k:np.array(data[k],copy=True) for k in data.files}
    n=len(manifest['rows'])
    expected_tasks={family+suffix for family in TASKS for suffix in ('','_high_speed')}
    if (n!=400 or dict(Counter(row['task'] for row in manifest['rows']))!={task:100 for task in expected_tasks}
            or not manifest['split'].get('duplicate_hashes_checked')):
        raise ValueError('exact400 TRAIN sources and full public-input duplicate audit required')
    for row in manifest['rows']:
        expected=TRAIN_ROOT/row['task']/'all_variations/episodes'/row['raw_episode_id']/'data.h5'
        if Path(row['source_path']).resolve()!=expected.resolve() or row.get('source_frame_index')!=0:
            raise ValueError('only first real frame of the declared author TRAIN source is permitted')
    if (arrays['rgb'].shape != (n,4,64,64,3) or arrays['rgb'].dtype != np.uint8
            or arrays['depth'].shape != (n,4,64,64) or not np.isfinite(arrays['depth']).all()
            or not np.all(arrays['depth']>0)):
        raise ValueError('finite declared sensor cache shape required')
    reassigned, summary = assign_splits(manifest['rows'],manifest['split']['seed'])
    if summary != manifest['split'] or any(any(a[key]!=b[key] for key in ('split','parent_group','semantic_task','semantic_label','speed','source_frame_index')) for a,b in zip(reassigned,manifest['rows'])):
        raise ValueError('preassigned parent split contract changed')
    return arrays, manifest


def normalized_rgbd(arrays, index, views=None):
    rgb=arrays['rgb'][index].astype(np.float32)/255.-.5
    depth=np.log1p(arrays['depth'][index].astype(np.float32))
    values=np.concatenate((rgb.transpose(0,1,4,2,3),depth[:,:,None]),2)
    if views is not None: values=values[:,list(views)]
    if not np.isfinite(values).all(): raise ValueError('finite sensor-only forward tensors required')
    return values
