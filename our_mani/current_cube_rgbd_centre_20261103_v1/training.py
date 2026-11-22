"""Fixed pilot40 fit/selection only. Default check never constructs a model."""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import random
import numpy as np
from .data import ROOT,DualCropCubeDataset,check_frozen_definitions,sha256

VERSION='dual_crop_current_cube_centre_40epoch_two_stage_pilot40_20261103_v1'
CONFIG_PATH=Path(__file__).with_name('fixed_config.json')
PROTOCOL_PATH=ROOT/'artifacts/framework/local_mpc_20261102/current_cube_dual_crop_centre_pilot40_training_protocol_v1/protocol.json'
PROTOCOL_SHA='888273e7f6f8ffa1322cf927fe3a608331d14ad695a63e5f04ce3ee3adf4ba0d'
ADDENDUM_PATH=PROTOCOL_PATH.with_name('derived_layout_and_unproven_absence_addendum_v1.json')
ADDENDUM_SHA='c9212a319c63b6e602c106bcc8cbe5bc46d2c74e2ec1d9deeadd42bb5a2ce217'
SELECTED_ENCODER_SHA='dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa'


def write_new(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def pointer(path):p=Path(path).resolve(strict=True);return dict(path=str(p),sha256=sha256(p))


def fixed_config():
    check_frozen_definitions()
    if sha256(PROTOCOL_PATH)!=PROTOCOL_SHA:raise ValueError('Immutable training protocol changed')
    if sha256(ADDENDUM_PATH)!=ADDENDUM_SHA:raise ValueError('Immutable pre-capture layout/absence addendum changed')
    c=json.loads(CONFIG_PATH.read_text())
    if json.loads(PROTOCOL_PATH.read_text())['config']!=c:raise ValueError('Fixed config differs from protocol')
    return c


def training_source_map():
    names=['our_mani/current_cube_rgbd_centre_20261103_v1/'+n for n in ('data.py','features.py','model.py','training.py','evaluation.py','fixed_config.json')]
    names+=['scripts/train_current_cube_dual_crop_centre_20261103_v1.py','scripts/evaluate_current_cube_dual_crop_centre_20261103_v1.py','our_mani/near_goal_segmentation_20261103/data.py','our_mani/near_goal_depth_refiner_20261103/data.py','our_mani/near_goal_depth_refiner_20261103/model.py']
    result={str((ROOT/n).resolve()):sha256(ROOT/n)for n in names};result[str(PROTOCOL_PATH)]=PROTOCOL_SHA
    result[str(ADDENDUM_PATH)]=ADDENDUM_SHA
    return result


def check(manifest=None):
    c=fixed_config();sources=training_source_map()
    value=dict(version=VERSION,config=c,source_sha256=sources,protocol=pointer(PROTOCOL_PATH),dataset_ready=False,
        model_forward_calls=0,model_constructed=False,optimizer_constructed=False,GPU_used=False,Native_started=False,training_started=False,runtime_qualified=False)
    if manifest is not None:
        fit=DualCropCubeDataset(manifest,'fit');selection=DualCropCubeDataset(manifest,'selection')
        value.update(dataset_ready=True,manifest=pointer(manifest),fit_samples=len(fit),selection_samples=len(selection),reserved_getters=0,arrays_decoded=0,label_proof_metadata_validated=True)
        fit.revalidate();selection.revalidate()
    return value


def array_digest(arrays):
    h=hashlib.sha256()
    for name,a in sorted(arrays.items()):
        h.update((name+str(a.dtype)+str(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def cache_role(dataset):
    records=[dataset[i]for i in range(len(dataset))]
    keys=tuple(records[0]['model_inputs'])
    arrays={key:np.stack([r['model_inputs'][key]for r in records])for key in keys}
    arrays.update(centre_target_world=np.stack([r['targets']['centre_target_world']for r in records]),
        centre_valid=np.asarray([r['targets']['centre_valid']for r in records],bool),
        presence_target=np.asarray([r['targets']['presence_target']for r in records],np.int64))
    for a in arrays.values():a.setflags(write=False)
    return arrays,dict(role=dataset.role,samples=len(records),ordered_scenes=[r['scene_id']for r in records],
        ordered_content_sha256=array_digest(arrays),targets=[dict(scene_id=r['scene_id'],**{k:v for k,v in r['targets'].items()if k!='centre_target_world'})for r in records])


def tensor_batch(arrays,indices,device):
    import torch
    return {k:torch.as_tensor(np.array(a[indices],copy=True),device=device)for k,a in arrays.items()}


def loss_batch(model,batch):
    from .model import centre_presence_loss
    inputs={k:batch[k]for k in ('features','wide_features','K','camera_rotations','initial_sphere')}
    output=model(**inputs)
    return centre_presence_loss(output,batch['centre_target_world'],batch['centre_valid'],batch['presence_target'])


def selection_loss(model,arrays,batch_size,device):
    import torch
    totals=dict(centre=0.,presence=0.);nc=npres=0;calls=0
    model.eval()
    with torch.no_grad():
        for start in range(0,len(arrays['presence_target']),batch_size):
            value=loss_batch(model,tensor_batch(arrays,np.arange(start,min(start+batch_size,len(arrays['presence_target']))),device));calls+=1
            c,p=value['centre_samples'],value['known_presence_samples'];nc+=c;npres+=p
            totals['centre']+=float(value['centre'])*c;totals['presence']+=float(value['presence'])*p
    centre=totals['centre']/max(nc,1);presence=totals['presence']/max(npres,1)
    return dict(total=centre+presence,centre=centre,presence=presence,centre_samples=nc,known_presence_samples=npres,forward_batch_calls=calls)


def fit(manifest,encoder_checkpoint,expected_encoder_sha256,output,*,device='cuda'):
    c=fixed_config()
    if expected_encoder_sha256!=SELECTED_ENCODER_SHA or sha256(encoder_checkpoint)!=SELECTED_ENCODER_SHA:raise ValueError('Exact original dbf encoder initialization required')
    fit_data=DualCropCubeDataset(manifest,'fit');selection_data=DualCropCubeDataset(manifest,'selection')
    if len(fit_data)!=208 or len(selection_data)!=208:raise ValueError('Fixed16fit/16selection/13shots only')
    sources=training_source_map();initial_data=fit_data.revalidate();selection_data.revalidate()
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    identity=dict(version=VERSION,manifest=pointer(manifest),selected_encoder=pointer(encoder_checkpoint),source_sha256=sources,
        protocol=pointer(PROTOCOL_PATH),config=c,data_before=initial_data,reserved_getters=0,
        role_scope='fit208 optimize; selection208 choose epoch; reserved104 never getter here',runtime_qualified=False)
    write_new(out/'identity.json',identity)
    steps=train_calls=selection_calls=0;history=[];status=dict(training_complete=False,actual_optimizer_steps=0,model_qualified=False,runtime_or_control_admitted=False)
    try:
        fit_cache,fit_receipt=cache_role(fit_data);selection_cache,sel_receipt=cache_role(selection_data)
        fit_data.revalidate();selection_data.revalidate()
        before=dict(fit=fit_receipt,selection=sel_receipt)
        write_new(out/'cache_before.json',before)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from .model import DualCropCubeCentreHeadV1,ENCODER_PREFIXES
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.set_num_threads(2)
        if str(device).startswith('cuda')and not torch.cuda.is_available():raise ValueError('Requested GPU unavailable; do not silently change device')
        torch.use_deterministic_algorithms(True)
        model=DualCropCubeCentreHeadV1(encoder_trainable=True)
        initialization=model.load_selected_v5_encoder(encoder_checkpoint,SELECTED_ENCODER_SHA);model.to(device)
        encoder=[p for name,p in model.named_parameters()if name.startswith(ENCODER_PREFIXES)]
        heads=[p for name,p in model.named_parameters()if not name.startswith(ENCODER_PREFIXES)]
        optimizer=torch.optim.AdamW([dict(params=encoder,lr=c['encoder_learning_rate'],group_name='encoder'),dict(params=heads,lr=c['head_learning_rate'],group_name='heads')],weight_decay=c['weight_decay'])
        rng=np.random.default_rng(c['seed']);best=None;best_state=None;best_epoch=None
        for epoch in range(1,c['epochs']+1):
            encoder_enabled=epoch>c['head_only_epochs']
            for p in encoder:p.requires_grad_(encoder_enabled)
            model.train();order=rng.permutation(len(fit_data));sum_loss=0.
            for start in range(0,len(order),c['batch_size']):
                batch=tensor_batch(fit_cache,order[start:start+c['batch_size']],device)
                optimizer.zero_grad(set_to_none=True);loss=loss_batch(model,batch);train_calls+=1
                if not bool(torch.isfinite(loss['total'])):raise ValueError('Nonfinite supervised loss')
                loss['total'].backward();optimizer.step();steps+=1;sum_loss+=float(loss['total'].detach())
            selection=selection_loss(model,selection_cache,c['batch_size'],device);selection_calls+=selection['forward_batch_calls']
            if not np.isfinite(selection['total']):raise ValueError('Nonfinite selection loss')
            if best is None or selection['total']<best:
                best=selection['total'];best_epoch=epoch;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
            history.append(dict(epoch=epoch,stage='heads_only'if not encoder_enabled else'encoder_and_heads',encoder_trainable=encoder_enabled,
                optimizer_groups=[dict(name=g['group_name'],learning_rate=g['lr'],parameters=len(g['params']))for g in optimizer.param_groups],
                fit_batch_mean_loss=sum_loss/13,selection=selection,actual_optimizer_steps=steps,
                trainable_parameters=[name for name,p in model.named_parameters()if p.requires_grad]))
            print(json.dumps(dict(epoch=epoch,optimizer_steps=steps,selection_loss=selection['total'])),flush=True)
        if steps!=520 or len(history)!=40:raise ValueError('Complete fixed40epoch/520step run required')
        after=dict(fit=array_digest(fit_cache),selection=array_digest(selection_cache))
        if after['fit']!=before['fit']['ordered_content_sha256']or after['selection']!=before['selection']['ordered_content_sha256']:raise ValueError('Training cache changed')
        fit_data.revalidate();selection_data.revalidate()
        if sources!=training_source_map()or sha256(encoder_checkpoint)!=SELECTED_ENCODER_SHA:raise ValueError('Implementation/source/initializer changed during training')
        write_new(out/'cache_after.json',after);write_new(out/'history.json',history)
        write_new(out/'sample_access.json',dict(fit=fit_data.access,selection=selection_data.access,reserved_getters=0))
        write_new(out/'source_verification.json',dict(source_sha256=sources,data_after=fit_data.revalidate(),all_source_and_data_unchanged=True))
        metadata=dict(version=VERSION,stage='selected_unqualified_pending_reserved_evaluation',training_complete=True,
            epochs_completed=40,actual_optimizer_steps=steps,selected_epoch=best_epoch,selected_selection_loss=best,
            config=c,encoder_initialization=initialization,manifest=pointer(manifest),identity=pointer(out/'identity.json'),
            source_sha256=sources,protocol=pointer(PROTOCOL_PATH),model_qualified=False,position_measurement_qualified=False,
            evidence={name:pointer(out/(name+'.json'))for name in ('identity','cache_before','cache_after','history','sample_access','source_verification')},
            uncertainty_qualified=False,runtime_or_control_admitted=False,reserved_forward_calls=0,selected_threshold=None,
            model_spec=model.spec(),environment=dict(torch=str(torch.__version__),numpy=np.__version__,torch_num_threads=torch.get_num_threads(),training_device=str(device)),
            forward_counts=dict(fit_batch_calls=train_calls,selection_batch_calls=selection_calls,fit_rows=40*208,selection_rows=40*208))
        torch.save(dict(model_state_dict=best_state,metadata=metadata),out/'selected.pt')
        status.update(training_complete=True,actual_optimizer_steps=steps,selected_epoch=best_epoch,selected_checkpoint=pointer(out/'selected.pt'),
            epochs_completed=40,pending_reserved_evaluation=True,forward_counts=metadata['forward_counts'])
        return status
    except BaseException as error:
        status.update(actual_optimizer_steps=steps,error=type(error).__name__+': '+str(error))
        raise
    finally:
        write_new(out/'fit_status.json',status)
