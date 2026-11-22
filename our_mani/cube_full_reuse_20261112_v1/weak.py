"""Explicit recorded-frame / desired-pose pretraining, never Native contact truth.

Only completed actual-clock TRAIN captures can create the subsequent runtime
checkpoints. Weak checkpoints have a different schema and are not runtime input.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .common import ROOT,OUT,sha,read,rows,write,append,pointer,verify,stopped,parents,training_sources,status

MOTION_FEATURES=('positions','timestamps','confidence','lengths')
CANDIDATE_FEATURES=('future_tokens','robot','geometry','context_c','task_e')


def admit(p):
    inputs={p['recorded_cache_registry']['path']:p['recorded_cache_registry']['sha256']}
    count=0
    for c in p['cases']:
        if c['source_split']!='train':continue
        e=c['existing_recorded_cache'];path=ROOT/e['cache']
        if sha(path)!=e['cache_sha256']:raise ValueError('Old recorded cache changed')
        inputs[str(path)]=e['cache_sha256'];inputs[c['source_path']]=c['source_sha256'];count+=1
    if count!=200:raise ValueError('All200 prior TRAIN sources required')
    for c in p['calibration_captures']:
        stopped(p);receipt=read(c['owner_receipt'])
        if not receipt['complete']or not receipt['actual_wait_completed']or receipt['actual_process_returncode']!=0 or receipt['unreaped_descendants']:
            raise ValueError('Actual old Native owner wait/reap required')
        folder=Path(c['capture_root']);r=read(folder/'result.json');closed=read(folder/'worker_exit.json')
        if not r['complete']or not closed['native_closed']or closed['error']is not None:raise ValueError('Complete old Native capture required')
        for f in (Path(c['owner_receipt']),folder/'result.json',folder/'worker_exit.json',folder/'private_TRAIN_labels.jsonl',folder/'public_observer/public_outputs.jsonl',folder/'public_observer/camera_calibration.json'):
            inputs[str(f)]=sha(f)
        for f in (folder/'public_observer').glob('frame_provenance.jsonl'):inputs[str(f)]=sha(f)
        for f in (folder/'public_observer/rgbd').glob('*.npz'):inputs[str(f)]=sha(f)
        public=rows(folder/'public_observer/public_outputs.jsonl');private=rows(folder/'private_TRAIN_labels.jsonl')
        if len(public)!=len(private)or len(private)!=r['actual_ticks']+1:raise ValueError('Complete same-clock original capture required')
        for index,(u,v)in enumerate(zip(public,private)):
            if u['index']!=index or abs(u['result']['timestamp']-v['time'])>1e-8 or v['episode_id']!=c['episode_id']:
                raise ValueError('Old capture clock/identity differs')
    verify(inputs)
    write(OUT/'reuse_admission.json',dict(complete=True,recorded_TRAIN_sources=200,actual_clock_Native_captures=len(p['calibration_captures']),
        all_reused_Native_owners_actual_waited=True,input_files_sha256=inputs,
        recorded_clock='nominal frame index, not actual physics time',recorded_contact_truth_available=False,
        whole200_Native_recollection=False,official_VAL_TEST_payloads_read=False))


def public_windows(c,with_future=True):
    """Read only the verified no-GT adapter cache; no original truth array opened."""
    e=c['existing_recorded_cache'];path=ROOT/e['cache']
    if sha(path)!=e['cache_sha256']:raise ValueError('Exact recorded public cache required')
    with np.load(path,allow_pickle=False)as f:
        pos=np.asarray(f['positions'][:,0],np.float32);confidence=np.asarray(f['confidence'][:,0],np.float32)
        valid=np.asarray(f['valid'][:,0],bool);times=np.asarray(f['times'],np.float64)
    if not np.allclose(times,np.arange(len(times))*.05,atol=1e-8,rtol=0):raise ValueError('Original nominal frame clock differs')
    hist=[];windows=[]
    for i in range(len(pos)):
        if not valid[i]or not np.isfinite(pos[i]).all()or confidence[i]<.5:hist.clear();continue
        hist.append(i);hist=hist[-20:]
        if len(hist)<5:continue
        x=np.zeros((20,3),np.float32);t=np.zeros(20,np.float64);cf=np.zeros(20,np.float32)
        x[:len(hist)]=pos[hist];t[:len(hist)]=times[hist];cf[:len(hist)]=confidence[hist]
        row=dict(positions=x,timestamps=t,confidence=cf,lengths=np.int64(len(hist)),anchor_index=np.int64(i))
        if with_future:
            y=np.zeros((17,3),np.float32);mask=np.zeros(17,bool)
            for k in range(1,17):
                j=i+k
                if j>=len(pos)or not valid[j]or not np.isfinite(pos[j]).all():break
                y[k]=pos[j];mask[k]=True
            if not mask.any():continue
            row.update(target_positions=y,target_valid=mask)
        windows.append(row)
    return windows


def save_role(folder,data,role,role_parents,**metadata):
    folder.mkdir(parents=True,exist_ok=False)
    if not data:raise ValueError('No recorded pretraining examples')
    arrays={k:np.stack([r[k]for r in data])for k in data[0]}
    assigned=set(arrays['parent_index'].tolist())
    if assigned!=set(range(len(role_parents))):raise ValueError('Every original assigned TRAIN parent must contribute pretraining examples')
    with(folder/'dataset.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    write(folder/'report.json',dict(role=role,parents=role_parents,windows=len(data),dataset_sha256=sha(folder/'dataset.npz'),
        all_assigned_parents_present=True,actual_physics_time_claimed=False,TEST_read=False,VAL_read=False,**metadata))


def compile_motion(p):
    destination=OUT/'motion_weak_data';destination.mkdir(exist_ok=False)
    for role in('fit','selection'):
        data=[];names=parents(p,role)
        for c in [c for c in p['cases']if c['role']==role]:
            stopped(p);w=public_windows(c)
            if not w:raise ValueError('No recorded sequence windows '+c['parent_id'])
            for row in w:
                row.pop('anchor_index');row.update(parent_index=np.int64(names.index(c['parent_id'])),sample_weight=np.float32(1/len(w)));data.append(row)
        save_role(destination/role,data,role,names,supervision='future observed RGBD positions in nominal recorded-frame domain; not Native future/contact labels')


def compile_candidate_features(p):
    """Every public feature is frozen before expert desired-pose labels are read."""
    from ..cube_motion_residual_20261104_v1.model import model_from_spec
    from ..cube_candidate_matched_20261104_v2.data import rot6
    checkpoint=OUT/'motion_fit/selected.pt';digest=sha(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model=model_from_spec(payload['model_spec'])
    model.load_state_dict(payload['model_state']);model.eval().requires_grad_(False);torch.set_num_threads(2)
    destination=OUT/'candidate_weak_features';destination.mkdir(exist_ok=False);files={}
    for c in [c for c in p['cases']if c['source_split']=='train']:
        stopped(p);w=public_windows(c,with_future=False)
        if not w:raise ValueError('Every original source needs recorded public history')
        inputs={k:torch.from_numpy(np.stack([r[k]for r in w]))for k in MOTION_FEATURES}
        with torch.inference_mode():out=model.forward_public(**inputs)
        predicted=out['positions'].numpy();rotations=out['rotations'].numpy();velocity=out['velocities'].numpy();angular=out['angular_velocities'].numpy()
        features=[]
        for i,row in enumerate(w):
            confidence=float(row['confidence'][int(row['lengths'])-1])
            features.append(dict(future_tokens=np.concatenate((np.arange(17,dtype=np.float32)[:,None]/16.,predicted[i]-predicted[i,:1],rot6(rotations[i]),velocity[i],angular[i]),axis=1).astype(np.float32),
                robot=np.zeros(25,np.float32),geometry=np.array([.05,.05,.05,confidence],np.float32),context_c=out['context'][i].numpy().astype(np.float32),task_e=np.full(16,.125,np.float32)))
        arrays={k:np.stack([r[k]for r in features])for k in CANDIDATE_FEATURES};arrays['anchor_index']=np.asarray([r['anchor_index']for r in w],np.int64)
        path=destination/(c['id']+'.npz')
        with path.open('xb')as f:np.savez_compressed(f,**arrays)
        files[str(path)]=sha(path)
    if sha(checkpoint)!=digest:raise ValueError('Frozen physical forecast changed during recorded-domain feature extraction')
    write(destination/'frozen.json',dict(complete=True,source_parents=200,files=files,forecast_checkpoint=pointer(checkpoint),
        expert_label_arrays_opened=False,Native_future_or_pose_inputs=False,actual_physical_time_claimed=False,
        robot_conditioning='explicit unavailable zero placeholder, not measured robot/finger sensors; robot projection frozen in weak stage',
        source='recorded causal RGBD positions; nominal frame clock, public predicted futures only'))


def compile_candidate_labels(p):
    import h5py
    destination=OUT/'candidate_weak_data';destination.mkdir(exist_ok=False)
    frozen=read(OUT/'candidate_weak_features/frozen.json');verify(frozen['files'])
    for role in('fit','selection'):
        data=[];names=parents(p,role);events=[]
        for c in [c for c in p['cases']if c['role']==role]:
            stopped(p)
            with np.load(OUT/'candidate_weak_features'/(c['id']+'.npz'),allow_pickle=False)as f:features={k:np.array(f[k])for k in CANDIDATE_FEATURES}
            # Same teacher reference is a desired-pose prior, never actual contact truth.
            if sha(c['source_path'])!=c['source_sha256']:raise ValueError('Original TRAIN HDF changed')
            with h5py.File(c['source_path'],'r')as f:
                stages=[v.decode()if isinstance(v,bytes)else str(v)for v in f['low_dim_obs/expert_trajectory_stage'][:].reshape(-1)]
                at=next(i for i,v in enumerate(stages)if v=='grasp')
                desired=np.asarray(f['low_dim_obs/expert_trajectory'][at,0],float)
                reference=np.asarray(f['low_dim_obs/expert_trajectory_target_position'][at],float)
            target_position=(desired[:3]-reference).astype(np.float32)
            target_rotation=Rotation.from_quat(desired[3:7]).as_matrix().astype(np.float32)
            n=len(features['robot'])
            for i in range(n):data.append(dict(**{k:features[k][i]for k in CANDIDATE_FEATURES},target_position=target_position,target_rotation=target_rotation,
                parent_index=np.int64(names.index(c['parent_id'])),sample_weight=np.float32(1/n)))
            events.append(dict(parent_id=c['parent_id'],teacher_frame_index=at,label='expert desired TCP relative to its own expert target reference; not actual Native contact',contact_time_known=False))
        save_role(destination/role,data,role,names,supervision='weak expert desired relative grasp pose only; no actual contact, task success or time labels',
            events=events,public_feature_freeze=pointer(OUT/'candidate_weak_features/frozen.json'))
    verify(frozen['files'])


def motion_loss(model,data,ids):
    out=model.forward_public(**{k:data[k][ids]for k in MOTION_FEATURES})
    mask=data['target_valid'][ids];target=torch.where(mask[...,None],data['target_positions'][ids],out['positions'].detach())
    error=torch.nn.functional.smooth_l1_loss((out['positions']-target)/.02,torch.zeros_like(target),reduction='none').mean(-1)
    return (error*mask).sum(1)/mask.sum(1).clamp_min(1)


def candidate_loss(model,data,ids):
    from ..local_mpc_p4_20261103.model import parallel_gripper_rotation_distance
    out=model(**{k:data[k][ids]for k in CANDIDATE_FEATURES})
    pe=torch.linalg.vector_norm(out['positions']-data['target_position'][ids,None,:],dim=-1)
    re=parallel_gripper_rotation_distance(out['rotation_matrices'],data['target_rotation'][ids,None,:,:])
    return ((pe/.03).square()+(re/.2).square()).min(1).values


@torch.no_grad()
def evaluate(model,data,kind):
    model.eval();n=len(data['parent_index']);values=[];loss=motion_loss if kind=='motion'else candidate_loss
    for start in range(0,n,128):values.append(loss(model,data,slice(start,start+128)).cpu().numpy())
    values=np.concatenate(values);weights=data['sample_weight'].cpu().numpy()
    return float((values*weights).sum()/weights.sum())


def fit(p,kind):
    from ..current_cube_rgbd_centre_large_20261104_v2.training import exclusive_resources
    from .fit import configure
    cfg=p['weak_pretraining'][kind];out=OUT/(kind+'_weak_fit');out.mkdir(exist_ok=False)
    sources=training_sources();write(out/'sources.json',sources)
    key='corrected_forecast_checkpoint'if kind=='motion'else'candidate_transformer_checkpoint'
    initial=ROOT/p['model_initialization'][key];inputs={str(initial):sha(initial)}
    if sha(initial)!=p['model_initialization'][key+'_sha256']:raise ValueError('Exact weak initializer required')
    folder=OUT/(kind+'_weak_data');arrays={}
    for role in('fit','selection'):
        report=read(folder/role/'report.json')
        if report['parents']!=parents(p,role)or report['dataset_sha256']!=sha(folder/role/'dataset.npz'):raise ValueError('Exact whole-role weak pretraining dataset required')
        with np.load(folder/role/'dataset.npz',allow_pickle=False)as f:arrays[role]={k:np.array(f[k])for k in f.files}
        for name in('dataset.npz','report.json'):inputs[str(folder/role/name)]=sha(folder/role/name)
    write(out/'request.json',dict(kind=kind,configuration=cfg,initializer=pointer(initial),source_parents=200,
        input_files_sha256=inputs,actual_physics_clock_claimed=False,Native_contact_supervision=False,runtime_admitted=False))
    started=time.perf_counter();steps=0
    with exclusive_resources()as resource:
        configure(cfg['seed']);verify(inputs);verify(sources);stopped(p)
        payload=torch.load(initial,map_location='cpu',weights_only=False)
        if kind=='motion':from ..cube_motion_residual_20261104_v1.model import model_from_spec
        else:from ..cube_matched_candidate_20261104_v1.model import model_from_spec
        model=model_from_spec(payload['model_spec']);model.load_state_dict(payload['model_state']);model.requires_grad_(True)
        frozen=[]
        if kind=='candidate':
            for name,module in model.named_children():
                if name in('robot_projection','time_head'):module.requires_grad_(False);frozen.append(name)
        model.cuda();data={role:{k:torch.from_numpy(v).cuda()for k,v in a.items()}for role,a in arrays.items()}
        initial_loss=evaluate(model,data['selection'],kind);write(out/'initial_selection.json',dict(loss=initial_loss))
        optimizer=torch.optim.AdamW([v for v in model.parameters()if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        groups={i:np.flatnonzero(arrays['fit']['parent_index']==i)for i in range(160)}
        if any(not len(v)for v in groups.values()):raise ValueError('Every160 fit parent must contribute weak gradients')
        rng=np.random.default_rng(cfg['seed']);best=float('inf');best_state=None;best_epoch=None;objective=motion_loss if kind=='motion'else candidate_loss
        for epoch in range(1,cfg['epochs']+1):
            stopped(p);model.train();total=0.
            for _ in range(cfg['steps_per_epoch']):
                ids=np.concatenate([rng.choice(v,cfg['windows_per_parent_per_step'],replace=True)for v in groups.values()]);rng.shuffle(ids)
                ids=torch.as_tensor(ids,device='cuda:0');optimizer.zero_grad(set_to_none=True)
                loss=objective(model,data['fit'],ids).mean()
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite recorded weak objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1;total+=float(loss.detach())
            entry=dict(epoch=epoch,optimizer_steps=steps,loss=total/cfg['steps_per_epoch'])
            if epoch%cfg['selection_interval']==0 or epoch==cfg['epochs']:
                score=evaluate(model,data['selection'],kind);entry['selection_loss']=score
                if score<best:best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
                status(kind+'_recorded_pretraining',epoch=epoch,epochs=cfg['epochs'],actual_optimizer_steps=steps,selection_loss=score)
                print(__import__('json').dumps(dict(kind=kind,**entry)),flush=True)
            append(out/'history.jsonl',entry)
        if best_state is None:raise RuntimeError('No trained weak snapshot')
        weak=dict(schema='Cube_recorded_weak_pretraining_only_20261112_v1',model_spec=payload['model_spec'],model_state=best_state,
            metadata=dict(training_complete=True,actual_physics_time_claimed=False,runtime_admitted=False,
                time_basis='recorded_nominal_frame_clock_not_actual_physics',labels='future public observations'if kind=='motion'else'expert desired relative pose, not contact',
                frozen_modules=frozen,train_parents=parents(p,'fit'),selection_parents=parents(p,'selection'),TEST_used=False,VAL_used=False),
            source_sha256=sources,input_sha256=inputs,selected_epoch=best_epoch,training_updates=steps)
        torch.save(weak,out/'selected.pt');verify(inputs);verify(sources)
        write(out/'fit_status.json',dict(complete=True,actual_optimizer_steps=steps,epochs_completed=cfg['epochs'],selected_epoch=best_epoch,
            checkpoint=pointer(out/'selected.pt'),TRAIN_source_parents=200,gradient_fit_parents=160,TRAIN_selection_parents=40,
            initial_selection_loss=initial_loss,selected_selection_loss=best,Native_contact_supervision=False,actual_physics_time_claimed=False,
            runtime_admitted=False,wall_seconds=time.perf_counter()-started,resources=resource))
