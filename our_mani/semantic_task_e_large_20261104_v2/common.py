"""Immutable inputs, exact public-window shape and component-only receipts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import EncoderConfig, OPERATIONS, parameter_count

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'config/framework/semantic_task_e_large_20261104_v2/pilot.json'
COMPONENT = 'semantic_task_e8_only'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def pointer(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checked_pointer(value):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Exact path/SHA pointer required')
    path = (ROOT/value['path']).resolve(strict=True)
    if sha(path) != value['sha256']:
        raise ValueError('Bound file changed: '+str(path))
    return path


def fixed_config():
    value = read(CONFIG)
    expected = dict(schema='semantic_task_e207M_training_pilot_20261104_v2',
        model=dict(width=1024,layers=16,heads=16,feedforward_multiplier=4,cnn_base=64,
                   spatial_grid=2,maximum_history=5,dropout=.1),
        exact_parameters=207855640,image_size=256,history_frames=5,windows_per_parent=2,
        epochs=10,microbatch=4,gradient_accumulation=4,effective_batch_size=16,
        precision='BF16_autocast_FP32_parameters_AdamW',learning_rate=3e-5,
        weight_decay=.01,gradient_clip_norm=1.,seed=20261104,workers=0,
        selection='maximum_development_operation_accuracy_then_minimum_CE_then_earliest_epoch',
        automatic_Native_evaluation=False,fitted_component=COMPONENT,
        current_network_end_to_end_ready=False,
        minimum_learning_rate=3e-6,warmup_logical_steps=50,
        scheduler='linear_warmup_cosine_to_minimum',raw_group_auxiliary_weight=.5)
    if value != expected or parameter_count(EncoderConfig(**value['model'])) != value['exact_parameters']:
        raise ValueError('Root-fixed 207M/R256/BF16 pilot configuration changed')
    return value


def source_map():
    from ..multitask_data_preparation_20261104_v1 import data
    names = [
        '__init__.py','model.py','common.py','training.py','evaluation.py',
    ]
    paths = [Path(__file__).with_name(name) for name in names]
    paths += [CONFIG,
        ROOT/'scripts/train_semantic_task_e_large_20261104_v2.py',
        ROOT/'scripts/evaluate_semantic_task_e_large_20261104_v2.py',
        ROOT/'our_mani/semantic_task_e_20261104_v1/__init__.py',
        ROOT/'our_mani/semantic_task_e_20261104_v1/model.py',
        ROOT/'our_mani/hierarchy_20261030/networks.py']
    result = {str(path.resolve()):sha(path) for path in paths}
    result.update(data.source_map())
    return result


def verify_sources(expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Explicit source graph required')
    for name, checksum in expected.items():
        if sha(name) != checksum:
            raise ValueError('Source changed: '+name)


def root_authorization(path, expected_sha, manifest, sources):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected_sha:
        raise ValueError('External Root GPU authorization SHA required')
    value = read(path)
    if (value.get('schema')!='semantic_e207M_GPU_authorization_20261104_v2'
            or value.get('execution_authorized') is not True
            or value.get('method')!='current_network'
            or value.get('fitted_component')!=COMPONENT
            or value.get('split_manifest')!=pointer(manifest)
            or value.get('config')!=pointer(CONFIG)
            or value.get('source_sha256')!=sources):
        raise ValueError('Exact Root-authorized GPU component/config/data/source identity required')
    lifecycle=value.get('native_lifecycle',{})
    if (lifecycle.get('all_native_closed') is not True
            or lifecycle.get('all_native_children_reaped') is not True
            or not isinstance(lifecycle.get('owners'),list) or not lifecycle['owners']):
        raise ValueError('Root must attest all Native children closed and actually reaped')
    for item in lifecycle['owners']:
        owner=read(checked_pointer(item))
        if (type(owner.get('actual_process_returncode')) is not int
                or owner.get('actual_wait_completed') is not True):
            raise ValueError('Every prior Native owner needs actual typed exit/wait evidence')
    previous=value.get('prior_training_lifecycle',{})
    if (previous.get('all_training_children_reaped')is not True
            or not isinstance(previous.get('owners'),list)or not previous['owners']):
        raise ValueError('V2 starts only after the prior V1 training owner actually exits and is reaped')
    for item in previous['owners']:
        owner=read(checked_pointer(item))
        if type(owner.get('actual_process_returncode'))is not int or owner.get('actual_wait_completed')is not True:
            raise ValueError('Actual previous training exit/wait evidence required')
    return pointer(path)


class PublicOperationWindows:
    """Use public_episode; never call the adapter's expert/future label getter."""
    def __init__(self, manifest, data_root, role):
        from ..multitask_data_preparation_20261104_v1.data import SemanticTaskHistoryDataset
        self.base = SemanticTaskHistoryDataset(manifest,role=role,history_frames=5,
            windows_per_parent=2,image_size=256,data_root=data_root)
        self.role = role
        self.index = {row['source_parent_group']:i for i,row in enumerate(self.base.episodes)}
        if len(self.index) != len(self.base.episodes):
            raise ValueError('Unique source-parent episodes required')
        if {row['task'].removesuffix('_high_speed') for row in self.base.episodes} != set(OPERATIONS):
            raise ValueError('All eight operations required; never filter hard/missing classes')

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        record = self.base.records[index]
        group = record['episode']['source_parent_group']
        result = self.base.public_episode(self.index[group],record['current_frame_index'])
        if result['labels']:
            raise ValueError('Semantic pretraining requires public-only episode access')
        return result

    def summary(self):
        return dict(role=self.role,parents=len(self.base.episodes),windows=len(self),
            task_counts=dict(Counter(row['task'] for row in self.base.episodes)),
            source_parent_groups=sorted(self.index),image_size=256,history_frames=5,
            windows_per_parent=2,physical_clock_verified=False,
            label='offline base-operation metadata only; no HDF expert/target read')

    def verify(self): return self.base.source_verification()


def datasets(manifest, data_root):
    fit = PublicOperationWindows(manifest,data_root,'fit')
    selection = PublicOperationWindows(manifest,data_root,'selection')
    if set(fit.index)&set(selection.index):
        raise ValueError('Source-parent leakage between fit and selection')
    return fit, selection


def preprocess_gate(manifest, data_root):
    path = Path(data_root).resolve(strict=True)/'completion.json'
    value = read(path)
    if (any(value.get(k) is not True for k in ('complete','source_unchanged'))
            or value.get('method') != 'current_network' or value.get('stage') != 'preprocess'
            or value.get('split_manifest') != pointer(manifest)):
        raise ValueError('Actual preprocess completion bound to identical split required')
    checked_pointer(value['split_manifest'])
    return pointer(path)


def collate_public(samples):
    if not samples:
        raise ValueError('Nonempty public window batch required')
    images=[];robots=[];lengths=[];labels=[];identities=[]
    for sample in samples:
        x=np.asarray(sample['rgbd']);r=np.asarray(sample['robot_history'])
        length=sample['lengths'];indices=np.asarray(sample['frame_indices'])
        label=sample['operation_label'];mask=np.asarray(sample['robot_observed_mask'])
        if (x.dtype!=np.float32 or x.shape!=(5,4,4,256,256)
                or r.dtype!=np.float32 or r.shape!=(5,15)
                or isinstance(length,(bool,np.bool_)) or not isinstance(length,(int,np.integer))
                or not 1<=length<=5 or indices.shape!=(5,) or indices.dtype.kind not in 'iu'
                or mask.dtype!=np.bool_ or mask.shape!=(5,15)):
            raise ValueError('Original R256/history5 public dataset shape/dtype required; no resizing')
        n=int(length)
        if (not np.isfinite(x[:n]).all() or not np.isfinite(r[:n]).all()
                or not (r[:n,-1]==0).all() or mask[:n,-1].any()
                or not mask[:n,:14].all() or mask[n:].any()
                or (indices[:n]<0).any() or not (np.diff(indices[:n])==1).all()
                or not (indices[n:]==-1).all()
                or np.any(x[:n,:,:3]<-1) or np.any(x[:n,:,:3]>1)
                or not (x[:n,:,3]>0).all()):
            raise ValueError('Finite genuine prefix/unknown gripper/causal index contract violated')
        if (type(label) not in (int,np.int64,np.int32) or not 0<=int(label)<8
                or sample['task'].removesuffix('_high_speed')!=OPERATIONS[int(label)]):
            raise ValueError('Offline operation label/task mapping changed')
        images.append(x);robots.append(r);lengths.append(n);labels.append(int(label))
        identities.append(dict(task=sample['task'],source_parent_group=sample['source_parent_group'],
            frame_indices=indices.tolist(),length=n,current_frame_index=int(sample['current_frame_index'])))
    return dict(rgbd=torch.from_numpy(np.stack(images)),robot_history=torch.from_numpy(np.stack(robots)),
        lengths=torch.tensor(lengths,dtype=torch.long),operation_label=torch.tensor(labels,dtype=torch.long),
        identities=identities)


def device_inputs(batch, device):
    return {key:batch[key].to(device) for key in ('rgbd','robot_history','lengths')}


def check_preparation(manifest,data_root):
    config=fixed_config();sources=source_map();verify_sources(sources)
    fit,selection=datasets(manifest,data_root)
    for dataset in (fit,selection): dataset.verify()
    return dict(complete=True,method='current_network',stage='fit_preparation',
        config=pointer(CONFIG),split_manifest=pointer(manifest),source_sha256=sources,
        fit=fit.summary(),selection=selection.summary(),exact_parameters=config['exact_parameters'],
        image_size=256,model_forward_calls=0,optimizers_created=0,
        Native_started=False,GPU_started=False,TEST_payloads_read=0,
        fitted_component=COMPONENT,current_network_end_to_end_ready=False,
        execution_ready=False)
