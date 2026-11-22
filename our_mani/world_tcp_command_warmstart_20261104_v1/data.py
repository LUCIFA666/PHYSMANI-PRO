"""Expert command-run index with labels separate from current public frames."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ..multitask_data_preparation_20261104_v1 import data as official
from . import VERSION

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='world_TCP_expert_command_run_warmstart_index_20261104_v1'


def sha(path):return official.sha(path)
def pointer(path):return official.pointer(path)
def read(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def source_map():
    # The index contains only source/public-frame references and command
    # labels. A later fit source closure additionally binds its explicit
    # selected scene-encoder family/checkpoint; it is not chosen here.
    sources=official.source_map()
    paths=[Path(__file__).with_name(name)for name in ('__init__.py','model.py','features.py','data.py')]
    paths.extend((ROOT/'scripts/prepare_world_tcp_command_warmstart_20261104_v1.py',
                  ROOT/'our_mani/local_mpc_p4_20261103/model.py'))
    sources.update({str(path.resolve()):sha(path)for path in paths})
    return sources


def verify_sources(sources):
    if not sources or any(sha(name)!=value for name,value in sources.items()):
        raise ValueError('Bound command-warmstart source graph changed')


def command_run_starts(commands):
    values=np.asarray(commands)
    if values.ndim!=3 or values.shape[1:]!=(1,8)or len(values)<1 or not np.isfinite(values).all():
        raise ValueError('Actual recorded expert_trajectory[T,1,8] required')
    return np.flatnonzero(np.r_[True,np.any(values[1:,0]!=values[:-1,0],axis=1)]).astype(np.int64)


def command_target(command):
    command=np.asarray(command,dtype=np.float64)
    if (command.shape!=(8,)or not np.isfinite(command).all()
            or not np.isclose(np.linalg.norm(command[3:7]),1.,atol=1e-3,rtol=0)
            or not 0<=command[7]<=1):
        raise ValueError('Desired world TCP xyz+unitxyzw+opening intent required')
    return dict(world_tcp_position=command[:3].tolist(),world_tcp_quaternion_xyzw=command[3:7].tolist(),
        opening_intent=float(command[7]),semantics='future_desired_expert_command_not_observed_TCP',
        observed_object_pose=False,handoff_tau_seconds=None,time_loss_weight=0.)


def metadata_check(manifest):
    value=official.read_manifest(manifest)
    fit=official.SemanticTaskHistoryDataset(manifest,role='fit',image_size=256)
    selection=official.SemanticTaskHistoryDataset(manifest,role='selection',image_size=256)
    if {r['source_parent_group']for r in fit.episodes}&{r['source_parent_group']for r in selection.episodes}:
        raise ValueError('Whole source-parent separation required')
    sources=source_map();verify_sources(sources)
    return dict(complete=True,version=VERSION,split_manifest=pointer(manifest),source_sha256=sources,
        fit_parents=len(fit.episodes),selection_parents=len(selection.episodes),image_size=256,
        command_label_payloads_read=0,RGBD_payloads_read=0,NN_forwards=0,Native_started=False,
        raw_TEST_payloads_read=0,index_prepared=False,physical_handoff_supervised=False,
        current_network_end_to_end_ready=False)


def prepare_index(manifest,output):
    """Explicit CPU-only preparation; no RGBD or trained encoder is opened."""
    import h5py
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh command index only')
    checked=metadata_check(manifest);sources=checked['source_sha256']
    output.mkdir(parents=True,exist_ok=False);records=[];targets=[];parents=[]
    try:
        for role in ('fit','selection'):
            dataset=official.SemanticTaskHistoryDataset(manifest,role=role,image_size=256)
            for episode in dataset.episodes:
                path=official.check_stat(episode)
                with h5py.File(path,'r')as stored:
                    commands=np.asarray(stored['low_dim_obs/expert_trajectory'][()])
                    stages=np.asarray(stored['low_dim_obs/expert_trajectory_stage'][()]).reshape(-1)
                if len(commands)!=episode['frame_count']or len(stages)!=len(commands):
                    raise ValueError('Actual command/stage frame alignment required')
                starts=command_run_starts(commands)
                parent_index=len(parents);parents.append(dict(episode=episode,role=role,command_runs=len(starts)))
                for run,current in enumerate(starts):
                    stage=stages[current];stage=stage.decode('utf8')if isinstance(stage,bytes)else str(stage)
                    target=command_target(commands[current,0]);target.update(expert_stage_label=stage,
                        command_issue_frame_index=int(current),command_run_ordinal=run)
                    index=len(targets);targets.append(target)
                    records.append(dict(parent_index=parent_index,target_index=index,
                        current_frame_index=int(current),role=role,task=episode['task'],
                        source_parent_group=episode['source_parent_group']))
                official.check_stat(episode)
        verify_sources(sources)
        write_new(output/'targets.json',dict(labels_only=True,targets=targets,
            targets_may_be_future_desired_commands=True,Native_or_debug_target_positions_read=False))
        index=dict(schema=SCHEMA,version=VERSION,complete=True,split_manifest=pointer(manifest),
            source_sha256=sources,parents=parents,records=records,targets=pointer(output/'targets.json'),
            role_counts=dict(Counter(row['role']for row in records)),
            task_counts=dict(Counter(row['task']for row in records)),
            index_rule='all exact expert8-vector run starts; selection identical; no success/quality/result filtering',
            label_frame='world_TCP_desired_command',runtime_input='current/past RGBD,K/T,measuredq/TCP only',
            physical_clock_verified=False,physical_handoff_supervised=False,
            handoff_tau_seconds=None,time_loss_weight=0.,NN_forwards=0,Native_started=False,
            RGBD_payloads_read=0,raw_TEST_payloads_read=0,current_network_end_to_end_ready=False)
        write_new(output/'index.json',index)
        return dict(complete=True,index=pointer(output/'index.json'),labels=pointer(output/'targets.json'),
            role_counts=index['role_counts'],task_counts=index['task_counts'],Native_started=False,
            NN_forwards=0,current_network_end_to_end_ready=False)
    except BaseException as error:
        write_new(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            Native_started=False,NN_forwards=0,automatic_retry=False))
        raise


class CommandWarmstartDataset:
    def __init__(self,index,role='fit'):
        if role not in ('fit','selection'):raise ValueError('Only declared fit/selection roles supported')
        self.path=Path(index).resolve(strict=True);self.metadata=read(self.path)
        value=self.metadata
        if value.get('schema')!=SCHEMA or value.get('complete')is not True:
            raise ValueError('Actual complete expert-command index required')
        verify_sources(value['source_sha256'])
        if sha(value['split_manifest']['path'])!=value['split_manifest']['sha256']or sha(value['targets']['path'])!=value['targets']['sha256']:
            raise ValueError('Bound source manifest/labels changed')
        original=official.read_manifest(value['split_manifest']['path'])
        canonical={row['source_parent_group']:row for row in original['episodes']}
        self.targets=read(value['targets']['path'])['targets'];self.parents=value['parents']
        self.records=[row for row in value['records']if row['role']==role];self.role=role
        if not self.records:raise ValueError('No source role may disappear')
        for row in self.records:
            parent=self.parents[row['parent_index']]
            episode=parent['episode']
            if (parent['role']!=role or episode['source_parent_group']!=row['source_parent_group']
                    or canonical.get(row['source_parent_group'])!=episode
                    or not episode['offline_eligible']
                    or (role=='fit'and not(episode['split']=='train'and episode['role']=='fit'))
                    or (role=='selection'and not(episode['role']=='selection'or episode['split']=='validation'))):
                raise ValueError('Parent/record/source identity mismatch')
            official.check_stat(parent['episode'])

    def __len__(self):return len(self.records)

    def public_sample(self,index):
        row=self.records[index];episode=self.parents[row['parent_index']]['episode']
        return official.episode_window(episode,row['current_frame_index'],image_size=256,include_labels=False)

    def __getitem__(self,index):
        # A caller must pass only sample['public'] through the feature adapter;
        # labels remain a separate loss-side branch, never a merged dictionary.
        return dict(public=self.public_sample(index),label=self.targets[self.records[index]['target_index']],
                    identity=self.records[index])
