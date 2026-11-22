"""Public-only feature freeze, then explicit selected-command label join."""
from collections import OrderedDict,defaultdict
from pathlib import Path
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch

from ..world_tcp_command_warmstart_20261104_v2.data import PublicSubsetWindows
from ..world_tcp_command_warmstart_20261104_v2.model import quaternion_rotation
from .features import FrozenFinePublicSceneMemoryV3
from .model import CandidateConfig,WorldTCPCommandTransformerV3,command_imitation_loss
from .common import (ROOT,CONFIG,COMPONENT,fixed_config,source_map,verify_sources,pointer,
    checked_pointer,read,write_new,sha,root_authorization,checked_canary,resources)

INPUT_KEYS=('visual_memory','visual_memory_mask','current_robot','camera_calibration','task_e',
            'fine_visual','fine_geometry','fine_token_mask')
CACHE_SCHEMA='world_fine_v3_public_feature_cache_20261104_v1'
JOIN_SCHEMA='world_fine_v3_feature_command_join_20261104_v1'


def ordinary_cpu(value,dtype=None):
    # encode is inference-only; clone OUTSIDE inference_mode gives ordinary
    # tensors which a later trainable Linear can safely save for backward.
    with torch.inference_mode(False):
        out=value.detach().to(device='cpu',dtype=dtype or value.dtype).clone()
    if torch.is_inference(out):raise ValueError('Cache tensor must be ordinary, not inference tensor')
    return out


def cache_inputs(encoded):
    result={}
    for key in INPUT_KEYS:
        value=encoded[key]
        dtype=torch.bool if key.endswith('_mask') else (torch.bfloat16 if key in ('visual_memory','fine_visual')else torch.float32)
        result[key]=ordinary_cpu(value,dtype)
    return result


def validate_cached_inputs(tensors,rows):
    shapes=dict(visual_memory=(rows,86,1024),visual_memory_mask=(rows,86),current_robot=(rows,14),
        camera_calibration=(rows,4,23),task_e=(rows,16),fine_visual=(rows,512,512),
        fine_geometry=(rows,512,6),fine_token_mask=(rows,512))
    if set(tensors)!=set(INPUT_KEYS):raise ValueError('Exact public-only cache input whitelist required')
    for key,value in tensors.items():
        dtype=torch.bool if key.endswith('_mask')else(torch.bfloat16 if key in ('visual_memory','fine_visual')else torch.float32)
        if not isinstance(value,torch.Tensor)or value.shape!=shapes[key]or value.dtype!=dtype or torch.is_inference(value):
            raise ValueError('Ordinary same-spec cache dtype/shape required: '+key)


def collate_public(items):
    samples=[item['public']for item in items]
    result={key:torch.from_numpy(np.stack([np.asarray(s[key])for s in samples]))
            for key in ('rgbd','robot_history','frame_indices')}
    result['lengths']=torch.tensor([int(s['lengths'])for s in samples],dtype=torch.long)
    for key,source in (('K','K'),('camera_to_world','T'),('near_far','near_far')):
        result[key]=torch.from_numpy(np.stack([np.asarray(s['calibration'][source])for s in samples])).float()
    if result['rgbd'].shape[1:]!=(5,4,4,256,256):raise ValueError('Actual original256/current-public history required')
    return result


def selected_targets(path,wanted):
    """Decode only chosen target objects. Other command values are skipped lexically."""
    wanted=set(map(int,wanted))
    if not wanted or min(wanted)<0:raise ValueError('Explicit nonempty target indices required')
    result={};maximum=max(wanted)
    with Path(path).open()as stream:
        prefix=''
        while True:
            chunk=stream.read(4096)
            if not chunk:raise ValueError('Target array missing')
            prefix+=chunk
            marker=prefix.find('"targets"')
            if marker>=0:
                start=prefix.find('[',marker)
                if start>=0:break
        def characters():
            yield from prefix[start+1:]
            while True:
                chunk=stream.read(65536)
                if not chunk:return
                yield from chunk
        chars=iter(characters());index=0
        while index<=maximum:
            char=next(chars,None)
            while char is not None and(char.isspace()or char==','):char=next(chars,None)
            if char!='{':raise ValueError('Target object missing before requested fixed index')
            keep=index in wanted;buffer=['{']if keep else None;depth=1;quoted=False;escaped=False
            while depth:
                char=next(chars,None)
                if char is None:raise ValueError('Truncated command-label JSON')
                if keep:buffer.append(char)
                if quoted:
                    if escaped:escaped=False
                    elif char=='\\':escaped=True
                    elif char=='"':quoted=False
                elif char=='"':quoted=True
                elif char=='{':depth+=1
                elif char=='}':depth-=1
            if keep:result[index]=json.loads(''.join(buffer))
            index+=1
    if set(result)!=wanted:raise ValueError('All selected command labels must be retained')
    return result


def targets_tensor(targets):
    p=torch.tensor([t['world_tcp_position']for t in targets],dtype=torch.float32)
    q=torch.tensor([t['world_tcp_quaternion_xyzw']for t in targets],dtype=torch.float32)
    opening=torch.tensor([t['opening_intent']for t in targets],dtype=torch.float32)
    if (not all(bool(torch.isfinite(v).all())for v in (p,q,opening))
            or not torch.allclose(q.norm(dim=-1),torch.ones(len(q)),atol=1e-3,rtol=0)
            or bool(((opening<0)|(opening>1)).any())):raise ValueError('Actual desired command label quality failed')
    return dict(position=p,rotation=quaternion_rotation(q),opening=opening)


def first_full_windows(dataset):
    choices=[i for i,g in enumerate(dataset.global_indices)if dataset.full['records'][g]['current_frame_index']>=4]
    if len(choices)<4:raise ValueError('Canonical four full-history TRAIN windows required')
    return choices[:4]


def gpu_canary(subset,encoder_checkpoint,encoder_sha,output,authorization,authorization_sha,*,microbatch=4,oom_predecessor=None,oom_sha=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh actual GPU canary only')
    config=fixed_config();sources=source_map();verify_sources(sources)
    subset_ptr=pointer(subset);encoder_ptr=dict(path=str(Path(encoder_checkpoint).resolve()),sha256=encoder_sha)
    auth=root_authorization(authorization,authorization_sha,sources=sources,subset_pointer=subset_ptr,encoder_pointer=encoder_ptr)
    approval=read(authorization)
    if approval.get('allow_first4_fit_canary_command_labels')is not True:
        raise ValueError('Root must explicitly admit only canonical first4 TRAIN command labels for discarded canary')
    if microbatch not in (2,4)or type(microbatch)is not int:raise ValueError('Fixed microbatch4 or actualOOM2 only')
    if microbatch==2:
        if oom_predecessor is None or sha(oom_predecessor)!=oom_sha:raise ValueError('Preserved actualOOM4 required')
        previous=read(oom_predecessor)
        if(previous.get('cuda_oom')is not True or previous.get('microbatch')!=4 or previous.get('source_sha256')!=sources
                or previous.get('subset')!=subset_ptr or previous.get('encoder_checkpoint')!=encoder_ptr):
            raise ValueError('Only same-source actualOOM4 permits2, never model/data changes')
    dataset=PublicSubsetWindows(subset,'fit');canonical=first_full_windows(dataset);indices=canonical[:microbatch]
    with resources():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available()or not torch.cuda.is_bf16_supported():raise RuntimeError('Actual BF16 CUDA required')
        output.mkdir(parents=True,exist_ok=False)
        base=dict(schema='fine_world_v3_GPU_canary_20261104_v1',source_sha256=sources,config=pointer(CONFIG),subset=subset_ptr,
            encoder_checkpoint=encoder_ptr,cache_dtypes=config['cache_dtypes'],microbatch=microbatch,effective_batch_size=16,
            canonical_full_history_dataset_indices=canonical,selected_dataset_indices=indices,Root_authorization=auth)
        base['OOM_predecessor']=None if oom_predecessor is None else pointer(oom_predecessor)
        encoder=model=optimizer=None;start=time.perf_counter()
        try:
            random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
            data_start=time.perf_counter();items=[dataset[i]for i in indices];batch=collate_public(items)
            data_seconds=time.perf_counter()-data_start
            if not bool((batch['lengths']==5).all()):raise ValueError('Worst-case full20/10 actual frames required')
            init_start=time.perf_counter()
            encoder=FrozenFinePublicSceneMemoryV3(encoder_checkpoint,encoder_sha,dataset.full['split_manifest']['path'],device='cuda')
            encoder_init_seconds=time.perf_counter()-init_start
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            gpu={key:value.cuda()for key,value in batch.items()}
            before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
            call_start=time.perf_counter();before.record()
            with torch.autocast('cuda',dtype=torch.bfloat16):encoded=encoder.encode(**gpu)
            after.record();torch.cuda.synchronize();encoder_ms=before.elapsed_time(after)
            encoder_call_seconds=time.perf_counter()-call_start;materialize_start=time.perf_counter()
            # Freeze public canary features BEFORE the expressly permitted
            # first4TRAIN target join. It is not the full cohort feature cache.
            ordinary=cache_inputs(encoded)
            torch.save(dict(inputs=ordinary,identities=[i['identity']for i in items]),output/'public_canary.pt')
            write_new(output/'public_canary_freeze.json',dict(complete=True,public_features_frozen=True,
                features=pointer(output/'public_canary.pt'),decoded_command_labels=False,**base))
            materialize_seconds=time.perf_counter()-materialize_start
            target_ids=[dataset.full['records'][dataset.global_indices[i]]['target_index']for i in indices]
            picked=selected_targets(checked_pointer(dataset.full['targets']),target_ids)
            target=targets_tensor([picked[i]for i in target_ids]);target={k:v.cuda()for k,v in target.items()}
            train_inputs={key:value.cuda()for key,value in ordinary.items()}
            model=WorldTCPCommandTransformerV3(CandidateConfig(**config['model'])).cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(**train_inputs)
            with torch.autocast('cuda',enabled=False):
                loss=command_imitation_loss(prediction,**target)['loss'].float()
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();end.record();torch.cuda.synchronize();verify_sources(sources)
            report=dict(base,complete=True,passed=True,source_unchanged=True,exact_parameters=86772491,
                full_history=True,actual_valid_frames=microbatch*5,actual_fine_tokens_per_sample=512,
                candidate_optimizer_steps=1,canary_weights_reused=False,
                decoded_command_target_count=microbatch,decoded_selection_command_labels=0,
                public_canary_freeze=pointer(output/'public_canary_freeze.json'),
                encoder_spec=encoder.model.spec(),encoder_spec_sha256=sha_json(encoder.model.spec()),
                encoder_cuda_batch_ms=encoder_ms,candidate_cuda_forward_backward_step_ms=begin.elapsed_time(end),
                wall_seconds=time.perf_counter()-start,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                public_HDF_batch_seconds=data_seconds,encoder_initialization_seconds=encoder_init_seconds,
                encoder_call_wall_seconds=encoder_call_seconds,ordinary_copy_and_save_seconds=materialize_seconds,
                estimated29760_cache_seconds_one_cold_batch_extrapolation=(data_seconds+encoder_call_seconds+materialize_seconds)*29760/microbatch,
                cost_estimate_is_not_steady_state_benchmark=True,
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),loss=float(loss.detach()),gradient_norm=float(norm),
                Native_started=False,TEST_read=False,current_network_end_to_end_ready=False)
            report.update(deterministic_algorithms=True,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                cudnn_benchmark=False,candidate_training_mode=True,candidate_dropout=config['model']['dropout'],
                precision=config['precision'],actual_optimizer_step_batch=microbatch,
                canary_is_full_effective16_training=False)
            write_new(output/'canary.json',report);return report
        except BaseException as error:
            write_new(output/'failure.json',dict(base,complete=False,cuda_oom=isinstance(error,torch.cuda.OutOfMemoryError),error_type=type(error).__name__,error=str(error)))
            raise
        finally:
            del encoder,model,optimizer
            torch.cuda.empty_cache()


def sha_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _file_stat(path):
    value=Path(path).stat();return dict(size=value.st_size,mtime_ns=value.st_mtime_ns,inode=value.st_ino)


def _parent_groups(dataset):
    groups=OrderedDict()
    for local,global_index in enumerate(dataset.global_indices):
        row=dataset.full['records'][global_index]
        groups.setdefault(row['source_parent_group'],[]).append(local)
    if any(len(v)!=16 for v in groups.values()):raise ValueError('Every fixed parent must retain16 windows')
    return list(groups.items())


def build_public_cache(subset,encoder_checkpoint,encoder_sha,output,authorization,authorization_sha,*,preflight_report,preflight_sha):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh public cache only; no silent resume/retry')
    config=fixed_config();sources=source_map();verify_sources(sources)
    subset_ptr=pointer(subset);encoder_ptr=dict(path=str(Path(encoder_checkpoint).resolve()),sha256=encoder_sha)
    auth=root_authorization(authorization,authorization_sha,sources=sources,subset_pointer=subset_ptr,encoder_pointer=encoder_ptr)
    canary=checked_canary(preflight_report,preflight_sha,sources=sources,subset_pointer=subset_ptr,
                          encoder_pointer=encoder_ptr,cache_dtypes=config['cache_dtypes'])
    microbatch=canary['microbatch'];datasets={role:PublicSubsetWindows(subset,role)for role in ('fit','selection')}
    if shutil.disk_usage(output.parent).free<24*2**30:raise ValueError('At least24GiB free for fixed20GiB cache and metadata required')
    with resources():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available()or not torch.cuda.is_bf16_supported():raise RuntimeError('Actual Root BF16 CUDA required')
        output.mkdir(parents=True,exist_ok=False);chunks=[];records=[];encoder=None;calls=0;start=time.perf_counter()
        try:
            encoder=FrozenFinePublicSceneMemoryV3(encoder_checkpoint,encoder_sha,datasets['fit'].full['split_manifest']['path'],device='cuda')
            (output/'public_features').mkdir()
            for role,dataset in datasets.items():
                role_dir=output/'public_features'/role;role_dir.mkdir()
                for parent_ordinal,(group,indices)in enumerate(_parent_groups(dataset)):
                    tensors={key:[]for key in INPUT_KEYS};identities=[]
                    for offset in range(0,16,microbatch):
                        if (output/'STOP').exists()or(output.parent/'STOP').exists()or(ROOT/'runtime/framework/STOP').exists():
                            raise RuntimeError('Administrative STOP; preserve partial cache and do not retry')
                        items=[dataset[i]for i in indices[offset:offset+microbatch]]
                        batch=collate_public(items);gpu={k:v.cuda()for k,v in batch.items()}
                        with torch.autocast('cuda',dtype=torch.bfloat16):encoded=encoder.encode(**gpu)
                        copied=cache_inputs(encoded)
                        for key in INPUT_KEYS:tensors[key].append(copied[key])
                        for item,frames in zip(items,encoded['fine_actual_source_frame_indices']):
                            identities.append(dict(item['identity'],fine_actual_source_frame_indices=frames))
                        calls+=1;del encoded,gpu
                    tensors={key:torch.cat(values,0)for key,values in tensors.items()}
                    validate_cached_inputs(tensors,16)
                    filename=role_dir/f'parent{parent_ordinal:04d}.pt'
                    torch.save(dict(inputs=tensors,identities=identities),filename)
                    chunk_index=len(chunks)
                    chunks.append(dict(file=pointer(filename),stat=_file_stat(filename),role=role,source_parent_group=group,rows=16))
                    for row_in_chunk,identity in enumerate(identities):
                        records.append(dict(role=role,chunk_index=chunk_index,row_in_chunk=row_in_chunk,identity=identity))
                    if parent_ordinal%32==0:verify_sources(sources)
            if len(records)!=29760:raise ValueError('Both fixed24592/5168 public cohorts must be frozen together')
            verify_sources(sources);checked_pointer(encoder_ptr)
            manifest=dict(schema=CACHE_SCHEMA,complete=True,public_features_frozen=True,decoded_command_targets=False,
                source_sha256=sources,config=pointer(CONFIG),subset=subset_ptr,encoder_checkpoint=encoder_ptr,
                encoder_spec=encoder.model.spec(),encoder_spec_sha256=sha_json(encoder.model.spec()),
                encoder_family='semantic_e207M_v2',cache_dtypes=config['cache_dtypes'],microbatch=microbatch,
                chunks=chunks,records=records,role_counts=dict(fit=24592,selection=5168),parent_counts=dict(fit=1537,selection=323),
                real_encoder_forward_batches=calls,Root_authorization=auth,actual_canary=pointer(preflight_report),
                first_canary_fit_target_decodes=canary['decoded_command_target_count'],
                target_decode_semantics='this full cache process decodes0 command targets; prior Root miniFIT canary decoded documented4or2 only',
                source_view_order=['front','8','16','36'],ray_convention='u+.5/v+.5 raw_cameraZ, publicK/T unchanged',
                fine_geometry_not_object_state=True,physical_clock_verified=False,
                wall_seconds=time.perf_counter()-start,Native=False,TEST=False,current_network_end_to_end_ready=False)
            write_new(output/'cache_manifest.json',manifest)
            write_new(output/'public_feature_freeze.json',dict(complete=True,public_features_frozen=True,
                cache_manifest=pointer(output/'cache_manifest.json'),subset=subset_ptr,encoder_checkpoint=encoder_ptr,
                encoder_spec_sha256=manifest['encoder_spec_sha256'],cache_dtypes=config['cache_dtypes'],
                role_counts=manifest['role_counts'],decoded_command_targets_in_this_process=0,
                prior_miniFIT_canary_target_decodes=canary['decoded_command_target_count'],source_sha256=sources))
            return dict(complete=True,cache_manifest=pointer(output/'cache_manifest.json'),
                public_feature_freeze=pointer(output/'public_feature_freeze.json'),role_counts=manifest['role_counts'],
                real_encoder_forward_batches=calls,command_labels_joined=False,Native=False,TEST=False)
        except BaseException as error:
            write_new(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error),source_sha256=sources,
                actual_public_records_completed=len(records),decoded_command_targets=False,automatic_retry=False))
            raise
        finally:
            del encoder
            torch.cuda.empty_cache()


def checked_public_cache(path):
    path=Path(path).resolve(strict=True);value=read(path)
    if (value.get('schema')!=CACHE_SCHEMA or value.get('complete')is not True
            or value.get('public_features_frozen')is not True or value.get('decoded_command_targets')is not False
            or value.get('role_counts')!=dict(fit=24592,selection=5168)
            or len(value.get('records',[]))!=29760 or len(value.get('chunks',[]))!=1860):
        raise ValueError('Entire fixed29760 public feature cache required before any command-label join')
    sources=source_map();verify_sources(sources)
    if value['source_sha256']!=sources or value['cache_dtypes']!=fixed_config()['cache_dtypes']:
        raise ValueError('Same source/config/cache dtype required')
    freeze=read(path.parent/'public_feature_freeze.json')
    if (freeze.get('complete')is not True or freeze.get('public_features_frozen')is not True
            or freeze.get('cache_manifest')!=pointer(path)or freeze.get('source_sha256')!=sources):
        raise ValueError('Actual whole-public freeze must precede command targets')
    for chunk in value['chunks']:
        file=checked_pointer(chunk['file'])
        if _file_stat(file)!=chunk['stat']:raise ValueError('Immutable public feature chunk changed')
    checked_pointer(value['encoder_checkpoint']);checked_pointer(value['subset'])
    return path,value


def join_command_labels(public_cache_manifest,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh post-freeze label join only')
    path,cache=checked_public_cache(public_cache_manifest)
    subset=read(checked_pointer(cache['subset']))
    full=read(checked_pointer(subset['full_index']))
    target_ids=[full['records'][r['identity']['global_record_index']]['target_index']for r in cache['records']]
    selected=selected_targets(checked_pointer(full['targets']),target_ids)
    labels=[selected[index]for index in target_ids];target=targets_tensor(labels)
    output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(target=target,expert_stage=[row['expert_stage_label']for row in labels],
                    identities=[row['identity']for row in cache['records']]),output/'labels.pt')
    joined=dict(schema=JOIN_SCHEMA,complete=True,public_cache=pointer(path),
        public_feature_freeze=pointer(path.parent/'public_feature_freeze.json'),labels=pointer(output/'labels.pt'),
        source_sha256=cache['source_sha256'],config=cache['config'],subset=cache['subset'],
        encoder_checkpoint=cache['encoder_checkpoint'],encoder_spec_sha256=cache['encoder_spec_sha256'],
        cache_dtypes=cache['cache_dtypes'],role_counts=cache['role_counts'],
        command_target_join_after_whole_public_freeze=True,canary_miniFIT_target_decodes=cache['first_canary_fit_target_decodes'],
        targets_are_expert_future_intent=True,handoff_tau_seconds=None,Native=False,TEST=False)
    write_new(output/'joined_manifest.json',joined)
    return dict(complete=True,joined_manifest=pointer(output/'joined_manifest.json'),labels=pointer(output/'labels.pt'),
                role_counts=joined['role_counts'],Native=False,TEST=False)


class CachedCommandDataset:
    def __init__(self,cache_manifest,role='fit'):
        if role not in ('fit','selection'):raise ValueError('Fixed fit/selection only, never TEST')
        self.path=Path(cache_manifest).resolve(strict=True);self.metadata=read(self.path);self.role=role
        if(self.metadata.get('schema')!=JOIN_SCHEMA or self.metadata.get('complete')is not True
                or self.metadata.get('command_target_join_after_whole_public_freeze')is not True):
            raise ValueError('Actual post-full-public-freeze command join required')
        self.public_path,self.public=checked_public_cache(checked_pointer(self.metadata['public_cache']))
        if any(self.metadata.get(key)!=self.public[key]for key in ('source_sha256','config','subset','encoder_checkpoint','encoder_spec_sha256','cache_dtypes','role_counts')):
            raise ValueError('Joined targets and fixed public memory identity differ')
        self.labels_path=checked_pointer(self.metadata['labels'])
        with torch.inference_mode(False):
            self.labels=torch.load(self.labels_path,map_location='cpu',weights_only=True)
        if (len(self.labels['identities'])!=29760 or len(self.labels['expert_stage'])!=29760
                or any(self.labels['target'][key].shape!=shape for key,shape in
                       (('position',(29760,3)),('rotation',(29760,3,3)),('opening',(29760,))))):
            raise ValueError('Every public record must retain one exact command target')
        self.master_indices=[i for i,r in enumerate(self.public['records'])if r['role']==role]
        self.records=[self.public['records'][i]['identity']for i in self.master_indices]
        if len(self.records)!=(24592 if role=='fit' else 5168):raise ValueError('Fixed role denominator changed')
        grouped=OrderedDict()
        for local,master in enumerate(self.master_indices):
            record=self.public['records'][master]
            if record['identity']!=self.labels['identities'][master]:raise ValueError('Public/label row identity differs')
            grouped.setdefault(record['identity']['source_parent_group'],[]).append(local)
        self.parent_batches=list(grouped.values());self._lru=OrderedDict()
        self._stats={str(checked_pointer(c['file'])):c['stat']for c in self.public['chunks']}

    def __len__(self):return len(self.records)

    @torch.inference_mode(False)
    def _public_item(self,index):
        master=self.master_indices[index];row=self.public['records'][master];chunk_index=row['chunk_index'];offset=row['row_in_chunk']
        if chunk_index not in self._lru:
            chunk=self.public['chunks'][chunk_index];path=checked_pointer(chunk['file'])
            value=torch.load(path,map_location='cpu',weights_only=True)
            validate_cached_inputs(value['inputs'],16)
            if len(value['identities'])!=16:raise ValueError('Whole parent16 chunk required')
            self._lru[chunk_index]=value
            if len(self._lru)>2:self._lru.popitem(last=False)
        value=self._lru.pop(chunk_index);self._lru[chunk_index]=value
        if value['identities'][offset]!=row['identity']:raise ValueError('Chunk record order changed')
        tensors={key:value['inputs'][key][offset]for key in INPUT_KEYS}
        if any(torch.is_inference(v)for v in tensors.values()):raise ValueError('Cache must load as ordinary training tensors')
        return dict(inputs=tensors,identity=row['identity'])

    @torch.inference_mode(False)
    def __getitem__(self,index):
        value=self._public_item(index);master=self.master_indices[index]
        value.update(target={key:self.labels['target'][key][master]for key in ('position','rotation','opening')},
                     expert_stage=self.labels['expert_stage'][master])
        return value

    def public_item(self,index):return self._public_item(index)

    def verify(self):
        verify_sources(self.metadata['source_sha256']);checked_pointer(self.metadata['public_cache']);checked_pointer(self.metadata['labels'])
        for path,expected in self._stats.items():
            if _file_stat(path)!=expected:raise ValueError('Immutable feature chunk stat changed')
        return dict(source_unchanged=True,rows=len(self),RGBD_reads=0,NN_calls=0,TEST=False)


def collate_cached(items):
    if not items:raise ValueError('Nonempty ordinary cached batch required')
    return dict(inputs={key:torch.stack([item['inputs'][key]for item in items])for key in INPUT_KEYS},
        target={key:torch.stack([item['target'][key]for item in items])for key in ('position','rotation','opening')},
        identities=[item['identity']for item in items],stages=[item['expert_stage']for item in items])
