"""Preparation checks and an explicitly invoked future fit, no automatic jobs."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import ROOT, CAMERAS, load_prepared, normalized_rgbd, sha256, write_new
from .model import SceneTaskEncoder


def parent_weights(rows, indices):
    """Each family has equal mass; parents equal within family, speeds share it."""
    selected = [rows[int(i)] for i in indices]
    sizes = Counter(row['parent_group'] for row in selected)
    family_parents = {label: {r['parent_group'] for r in selected if r['semantic_label'] == label} for label in (0, 1)}
    if not all(family_parents.values()): raise ValueError('both semantic families required')
    result = np.array([1./(len(family_parents[r['semantic_label']])*sizes[r['parent_group']]) for r in selected], np.float64)
    return result / result.mean()


def _tensors(arrays, rows, indices, device, views):
    sensor = torch.from_numpy(normalized_rgbd(arrays, indices, views)).to(device)
    labels = torch.tensor([rows[int(i)]['semantic_label'] for i in indices], dtype=torch.long, device=device)
    return sensor, labels


@torch.no_grad()
def evaluate(model, arrays, rows, indices, *, views=(0, 1, 2, 3), batch_size=32):
    model.eval();device=next(model.parameters()).device; losses=[];predictions=[];truth=[]
    for begin in range(0,len(indices),batch_size):
        index=indices[begin:begin+batch_size];sensor,labels=_tensors(arrays,rows,index,device,views)
        out=model(sensor);losses.extend(F.cross_entropy(out['task_logits'],labels,reduction='none').cpu().tolist())
        predictions.extend(out['task_logits'].argmax(-1).cpu().tolist());truth.extend(labels.cpu().tolist())
    losses=np.array(losses);predictions=np.array(predictions);truth=np.array(truth);weights=parent_weights(rows,indices)
    confusion=np.zeros((2,2),int)
    for label,prediction in zip(truth,predictions):confusion[label,prediction]+=1
    return dict(parent_balanced_cross_entropy=float(np.average(losses,weights=weights)),
        parent_balanced_accuracy=float(np.average(predictions==truth,weights=weights)),
        sample_accuracy=float(np.mean(predictions==truth)),confusion_true_rows_predicted_columns=confusion.tolist(),
        per_task_speed={task:dict(samples=len(chosen),accuracy=float(np.mean(predictions[chosen]==truth[chosen])))
            for task in sorted({rows[int(i)]['task'] for i in indices})
            if len(chosen:=[j for j,i in enumerate(indices) if rows[int(i)]['task']==task])},
        views=[CAMERAS[v] for v in views],scope='internal TRAIN-parent classification only; no semantic/generalization or manipulation success claim')


def check_prepared(dataset, *, cpu_threads=2):
    """Forward/backward contract smoke; no optimizer or parameter update."""
    torch.set_num_threads(cpu_threads);torch.manual_seed(20261103)
    arrays,manifest=load_prepared(dataset);rows=manifest['rows'];indices=np.array([i for i,r in enumerate(rows) if r['split']=='fit'])
    labels=[r['semantic_label'] for r in rows]
    chosen=np.array([next(i for i in indices if labels[i]==label) for label in (0,1)])
    model=SceneTaskEncoder().cpu().eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    sensor,target=_tensors(arrays,rows,chosen,'cpu',(0,1,2,3));out=model(sensor)
    loss=F.cross_entropy(out['task_logits'],target);loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError('finite meaningful scene-classifier gradients required')
    with torch.no_grad():
        permutation=model(sensor[:,[3,1,0,2]])
        torch.testing.assert_close(out['task_e'],permutation['task_e'],atol=2e-6,rtol=1e-5)
        for view in range(4):
            one=model(sensor[:,view:view+1])
            assert one['task_e'].shape==(2,16)
        for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    return dict(complete=True,mode='CPU_preflight_no_fit',dataset_sha256=manifest['dataset_sha256'],split=manifest['split'],
        forward_shapes={k:list(v.shape) for k,v in out.items()},SimNorm_group_sums=out['task_e'].detach().reshape(2,2,8).sum(-1).tolist(),
        loss_finite=True,backward_finite=True,optimizer_created=False,parameter_updates=0,
        model_checkpoint_saved=False,Native_started=False,official_VAL_or_TEST_opened=False,model_spec=model.spec())


def fit(dataset,output,config):
    """Only the CLI's explicit --fit branch may call this function."""
    torch.set_num_threads(int(config['cpu_threads']));seed=int(config['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    arrays,manifest=load_prepared(dataset);rows=manifest['rows'];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if int(config['epochs'])<1 or int(config['batch_size'])<1:raise ValueError('positive fit budget required')
    views=tuple(CAMERAS.index(name) for name in config['train_cameras'])
    if not views or len(set(views))!=len(views):raise ValueError('explicit distinct training cameras required')
    device=torch.device(config['device']);model=SceneTaskEncoder().to(device)
    warmstart=None
    if config.get('camera_warmstart'):
        path=Path(config['camera_warmstart']);payload=torch.load(path,map_location='cpu',weights_only=False)
        state=payload.get('encoder_state_dict',payload.get('model',payload.get('state_dict')))
        if not isinstance(state,dict):raise ValueError('explicit visual initializer state_dict required')
        warmstart=dict(path=str(path.resolve()),sha256=sha256(path),**model.warmstart_camera_encoder(state,config['camera_warmstart_prefix']))
    indices={split:np.array([i for i,r in enumerate(rows) if r['split']==split]) for split in ('fit','internal_val')}
    weights=parent_weights(rows,indices['fit']);optimizer=torch.optim.AdamW(model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']))
    identity=dict(dataset_sha256=manifest['dataset_sha256'],manifest_sha256=sha256(Path(dataset)/'manifest.json'),source_sha256=manifest['source_sha256'],
        parent_split=manifest['split'],normalization=manifest['preprocessing'],config=deepcopy(config),model_spec=model.spec(),camera_warmstart=warmstart,
        official_VAL_or_TEST_used=False,Native_started=False,frozen_P4_direct_e_substitution_allowed=False,
        selection='minimum internal TRAIN-dev parent-balanced cross entropy; no official VAL/TEST',camera_diagnostic='heldout camera means excluded from train_cameras; leave-one-view-out of a four-camera-trained model is only view-dropout robustness')
    write_new(output/'identity.json',identity)
    for source in manifest['source_sha256']:
        relative=Path(source).relative_to(ROOT);destination=output/'source_snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    history=[];best=float('inf');selected=None;started=time.perf_counter();rng=np.random.default_rng(seed)
    for epoch in range(1,int(config['epochs'])+1):
        model.train();order=rng.permutation(len(indices['fit']));total=0.
        for begin in range(0,len(order),int(config['batch_size'])):
            local=order[begin:begin+int(config['batch_size'])];index=indices['fit'][local]
            sensor,labels=_tensors(arrays,rows,index,device,views);out=model(sensor)
            ce=F.cross_entropy(out['task_logits'],labels,reduction='none');weight=torch.as_tensor(weights[local],device=device,dtype=ce.dtype)
            loss=(ce*weight).mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            optimizer.step();total+=float((ce.detach()*weight).sum())
        score=evaluate(model,arrays,rows,indices['internal_val'],views=views,batch_size=int(config['batch_size']))
        row=dict(epoch=epoch,fit_parent_balanced_CE=total/len(indices['fit']),internal_val=score);history.append(row)
        if score['parent_balanced_cross_entropy']<best:
            best=score['parent_balanced_cross_entropy'];selected=epoch
            torch.save(dict(model=model.state_dict(),model_spec=model.spec(),identity=identity,selected_epoch=epoch,selected_metrics=score,training_complete=False),output/'best.pt')
    selected_payload=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    selected_payload.update(training_complete=True,completed_epochs=int(config['epochs']),selection_history_unchanged=True)
    torch.save(selected_payload,output/'best.pt')
    torch.save(dict(model=model.state_dict(),model_spec=model.spec(),identity=identity,completed_epochs=int(config['epochs']),training_complete=True),output/'final.pt')
    model.load_state_dict(selected_payload['model'],strict=True)
    heldout=[v for v in range(4) if v not in views]
    diagnostics=dict(view_dropout_internal_val={CAMERAS[v]:evaluate(model,arrays,rows,indices['internal_val'],views=tuple(i for i in views if i!=v)) for v in views if len(views)>1},
        truly_excluded_camera_internal_val={CAMERAS[v]:evaluate(model,arrays,rows,indices['internal_val'],views=(v,)) for v in heldout})
    write_new(output/'history.json',history);write_new(output/'camera_diagnostics.json',diagnostics)
    status=dict(complete=True,training_complete=True,requested_epochs=int(config['epochs']),completed_epochs=int(config['epochs']),selected_epoch=selected,
        best_sha256=sha256(output/'best.pt'),final_sha256=sha256(output/'final.pt'),wall_seconds=time.perf_counter()-started,
        official_VAL_or_TEST_used=False,Native_started=False,scene_classification_only=True,frozen_P4_e_compatibility_qualified=False)
    write_new(output/'training_status.json',status)
    return status
