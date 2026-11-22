"""Fixed stage-preserving resource pilot over the immutable full v1 index."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from ..world_tcp_command_warmstart_20261104_v1 import data as full
from ..multitask_data_preparation_20261104_v1 import data as official
from . import VERSION

ROOT=Path(__file__).resolve().parents[2]
FULL_INDEX_SHA='1c8c292bab1d6075e0c4362eec899b786794b56a80fc9cc07c86a30c71cfd18c'
TARGET_SHA='15cf1029857a3233c04e3aa4fbb3a9b2f2f81e08a9080c0e0e8df57b37287181'
SCHEMA='world_TCP_command_stage_preserving_max16_pilot_20261104_v2'


def source_map():
    sources=full.source_map()
    paths=[Path(__file__).with_name(name)for name in ('__init__.py','model.py','runtime_adapter.py','data.py')]
    paths.extend((ROOT/'scripts/prepare_world_tcp_command_warmstart_20261104_v2.py',
                  ROOT/'our_mani/generic_reference_mpc_20261104_v1/contracts.py'))
    sources.update({str(p.resolve()):full.sha(p)for p in paths})
    return sources


def checked_full_index(path):
    path=Path(path).resolve(strict=True)
    if full.sha(path)!=FULL_INDEX_SHA:raise ValueError('Immutable full v1 index required; no regeneration')
    value=full.read(path);full.verify_sources(value['source_sha256'])
    if value['targets']['sha256']!=TARGET_SHA or full.sha(value['targets']['path'])!=TARGET_SHA:
        raise ValueError('Original complete expert labels must remain unchanged')
    official.read_manifest(value['split_manifest']['path'])
    return path,value


def select_parent_runs(global_indices,stage_labels,cap=16):
    if cap!=16 or not global_indices or len(global_indices)!=len(stage_labels):
        raise ValueError('Fixed max16 and all ordered command-run metadata required')
    if sorted(set(global_indices))!=list(global_indices):raise ValueError('Unique ordered full-index rows required')
    by_stage=defaultdict(list)
    for ordinal,stage in enumerate(stage_labels):by_stage[str(stage)].append(ordinal)
    if len(by_stage)>8:
        raise ValueError('More than8 stages requires a new Root rule; never drop an entire stage')
    mandatory=sorted({i for positions in by_stage.values()for i in (positions[0],positions[-1])})
    count=min(cap,len(global_indices));need=count-len(mandatory)
    remaining=[i for i in range(len(global_indices))if i not in set(mandatory)]
    if need<0:raise ValueError('Cannot preserve every stage first/last under fixed cap')
    fill=[] if need==0 else [remaining[int(np.floor((j+.5)*len(remaining)/need))]for j in range(need)]
    selected=sorted(mandatory+fill)
    if len(selected)!=count or len(set(selected))!=count:
        raise ValueError('Deterministic midpoint fill must never duplicate or remove a mandatory row')
    return dict(selected_global_indices=[global_indices[i]for i in selected],
        mandatory_global_indices=[global_indices[i]for i in mandatory],
        stage_count=len(by_stage),stage_counts=dict(Counter(map(str,stage_labels))),
        rule='first_last_each_distinct_stage_then_midpoints_of_remaining_run_ordinals')


def prepare_subset(index,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh pilot manifest only')
    path,value=checked_full_index(index);sources=source_map();full.verify_sources(sources)
    # Only previously frozen TRAIN/development labels are decoded for the
    # predeclared sampling rule. No RGBD, predictions or quality values enter.
    targets=full.read(value['targets']['path'])['targets'];groups=defaultdict(list)
    for i,row in enumerate(value['records']):groups[(row['role'],row['source_parent_group'])].append(i)
    if len(groups)!=1860:raise ValueError('All1537fit+323selection parents must remain')
    parents=[];selected=[]
    for (role,group),indices in groups.items():
        stages=[targets[value['records'][i]['target_index']]['expert_stage_label']for i in indices]
        choice=select_parent_runs(indices,stages)
        row=value['records'][indices[0]]
        parents.append(dict(role=role,source_parent_group=group,task=row['task'],
            full_command_runs=len(indices),**choice))
        selected.extend(choice['selected_global_indices'])
    selected=sorted(selected)
    counts=Counter(value['records'][i]['role']for i in selected)
    if counts!=dict(fit=24592,selection=5168):raise ValueError('Fixed all-parent16-window denominator changed')
    full.verify_sources(sources)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema=SCHEMA,version=VERSION,complete=True,full_index=full.pointer(path),
        full_targets=value['targets'],full_source_sha256=value['source_sha256'],source_sha256=sources,
        split_manifest=value['split_manifest'],parents=parents,selected_global_record_indices=selected,
        role_counts=dict(counts),parent_counts=dict(Counter(p['role']for p in parents)),
        stages_used_only_for_labelside_sampling_and_reports=True,
        actual_command_targets_used_as_model_input=False,quality_or_success_or_predictions_used=False,
        maximum_stage_count=max(p['stage_count']for p in parents),RGBD_reads=0,NN_calls=0,GPU=False,Native=False,TEST_reads=0,
        physical_clock_verified=False,handoff_tau_seconds=None,current_network_end_to_end_ready=False)
    full.write_new(output/'subset.json',manifest)
    return manifest


class PublicSubsetWindows:
    """Decode only public frames. Command-pose labels are NOT loaded here."""
    def __init__(self,subset,role='fit'):
        if role not in ('fit','selection'):raise ValueError('Only fixed fit/selection roles allowed')
        self.path=Path(subset).resolve(strict=True);self.metadata=full.read(self.path);self.role=role
        if self.metadata.get('schema')!=SCHEMA or self.metadata.get('complete')is not True:
            raise ValueError('Complete fixed stage-preserving subset required')
        full.verify_sources(self.metadata['source_sha256'])
        self.full_path,self.full=checked_full_index(self.metadata['full_index']['path'])
        if self.metadata['full_index']!=full.pointer(self.full_path):raise ValueError('Subset/full index identity changed')
        self.global_indices=[i for i in self.metadata['selected_global_record_indices']if self.full['records'][i]['role']==role]
        expected=24592 if role=='fit' else 5168
        if len(self.global_indices)!=expected or len(set(self.global_indices))!=expected:
            raise ValueError('Every fixed pilot window remains in its role')
        original=official.read_manifest(self.full['split_manifest']['path'])
        canonical={row['source_parent_group']:row for row in original['episodes']}
        for index in self.global_indices:
            row=self.full['records'][index];parent=self.full['parents'][row['parent_index']]
            episode=parent['episode']
            if (canonical.get(row['source_parent_group'])!=episode or not episode['offline_eligible']
                    or parent['role']!=role
                    or(role=='fit'and not(episode['split']=='train'and episode['role']=='fit'))
                    or(role=='selection'and not(episode['role']=='selection'or episode['split']=='validation'))):
                raise ValueError('No reserved/TEST/excluded/cross-role episode may enter pilot')
        self.targets_decoded=False

    def __len__(self):return len(self.global_indices)

    def __getitem__(self,index):
        global_index=self.global_indices[index];row=self.full['records'][global_index]
        episode=self.full['parents'][row['parent_index']]['episode']
        sample=official.episode_window(episode,row['current_frame_index'],image_size=256,include_labels=False)
        if sample['labels']:raise ValueError('Feature-cache public input must not include command labels')
        return dict(public=sample,identity=dict(global_record_index=global_index,
            current_frame_index=row['current_frame_index'],source_parent_group=row['source_parent_group'],task=row['task'],role=self.role))


def metadata_check(index):
    path,value=checked_full_index(index);sources=source_map();full.verify_sources(sources)
    return dict(complete=True,full_index=full.pointer(path),full_targets=value['targets'],source_sha256=sources,
        full_windows=143323,fit_parents=1537,selection_parents=323,cap=16,
        subset_not_built=True,RGBD_reads=0,command_targets_decoded=False,NN_calls=0,GPU=False,Native=False,TEST_reads=0)
