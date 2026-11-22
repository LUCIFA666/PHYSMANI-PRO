"""Fixed TRAIN-only pretraining, selection-epoch then calibration-threshold.

Importing this module never fits anything. No old checkpoint is reused.
"""
from pathlib import Path
import hashlib,json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import CaptureDataset,FEATURE_NAMES,CLASS_NAMES,sha256
from .model import TinyNearGoalUNet,segmentation_loss,predict_probabilities
from .evaluation import oracle_diagnostic,evaluate_probabilities,choose_calibrated_threshold,THRESHOLD_GRID,QUALITY_PROTOCOL,MIN_ORACLE_ELIGIBLE_COVERAGE

SEED=261103700;EPOCHS=40;LR=3e-4;WEIGHT_DECAY=1e-4
VERSION='near_goal_current_visible_segmentation_20261103_v1'


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def fixed_config():
    return dict(seed=SEED,epochs=EPOCHS,optimizer='AdamW',lr=LR,weight_decay=WEIGHT_DECAY,batch_shots=16,
        loss='CE+foregroundDice',parent_weighting='eachparent8shots_equal_views_equal',
        feature_names=FEATURE_NAMES,class_names=CLASS_NAMES,unet_base_channels=8,near_scope_m=.08,
        threshold_grid=THRESHOLD_GRID,quality_protocol=QUALITY_PROTOCOL,minimum_oracle_eligible_coverage=MIN_ORACLE_ELIGIBLE_COVERAGE,augmentation='none',pretrained_weights=False)

def source_identity(manifest):
    package=Path(__file__).parent;files=sorted([*package.glob('*.py'),*package.glob('*.json')])
    cli=package.parents[1]/'scripts/train_near_goal_segmentation_20261103.py'
    if cli.exists():files.append(cli)
    return dict(manifest_path=str(Path(manifest).resolve()),manifest_sha256=sha256(manifest),
        code_sha256={str(p):sha256(p) for p in files},config=json.loads(json.dumps(fixed_config())))

def _collate(samples):
    return dict(features=torch.tensor(np.stack([s['features'] for s in samples]),dtype=torch.float32),
        labels=torch.tensor(np.stack([s['semantic'] for s in samples]),dtype=torch.long))

def _loss_batch(model,batch,device):
    x=batch['features'].to(device);y=batch['labels'].to(device);b=x.shape[0]
    logits=model(x.reshape(b*4,9,64,64))
    return segmentation_loss(logits,y.reshape(b*4,64,64)).reshape(b,4).mean(1).mean()

def train_capture(manifest,output,*,device='cpu',allow_pilot_smoke=False,workers=0):
    complete=CaptureDataset(manifest);scope=complete.scope
    if scope!='full32' and not allow_pilot_smoke:raise ValueError('pilot only explicit smoke; never32parentqualification')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);identity=source_identity(manifest)
    write_new(out/'identity.json',identity)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in identity['code_sha256']:
        source=Path(path);destination=snapshot/source.name
        if destination.exists():raise ValueError('ambiguous source archive name')
        destination.write_bytes(source.read_bytes())
    oracle=oracle_diagnostic(complete);write_new(out/'oracle_before_fit.json',oracle)
    if not oracle['ready_for_segmentation_fit']:
        write_new(out/'status.json',dict(complete=False,training_started=False,reason='oracle_geometry_gate_failed',model_qualified=False))
        raise ValueError('oracle visiblemask/depth centre fails5mm; investigate geometry before NN fit')
    torch.set_num_threads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(SEED)
    fit=CaptureDataset(manifest,'fit');selection=CaptureDataset(manifest,'selection');calib=CaptureDataset(manifest,'calib')
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(fit,batch_size=16,shuffle=True,generator=generator,num_workers=workers,collate_fn=_collate)
    select_loader=DataLoader(selection,batch_size=16,shuffle=False,num_workers=workers,collate_fn=_collate)
    model=TinyNearGoalUNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    best=float('inf');best_epoch=None;best_state=None;history=[];started=time.perf_counter()
    for epoch in range(EPOCHS):
        model.train();fit_sum=0.;fit_count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);loss=_loss_batch(model,batch,device)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss; no retry')
            loss.backward();optimizer.step();count=batch['features'].shape[0];fit_sum+=float(loss.detach())*count;fit_count+=count
        model.eval();select_sum=0.;select_count=0
        with torch.inference_mode():
            for batch in select_loader:
                loss=_loss_batch(model,batch,device);count=batch['features'].shape[0]
                select_sum+=float(loss)*count;select_count+=count
        value=select_sum/select_count
        if not np.isfinite(value):raise ValueError('nonfinite selection loss; no retry')
        if value<best:
            best=value;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch+1,fit_loss=fit_sum/fit_count,selection_loss=value))
    model.load_state_dict(best_state,strict=True);model.eval()
    samples=[calib[i] for i in range(len(calib))];probs=[predict_probabilities(model,s['features']) for s in samples]
    reports=[evaluate_probabilities(samples,probs,t) for t in THRESHOLD_GRID]
    calibration=choose_calibrated_threshold(reports);write_new(out/'calibration.json',dict(selection=calibration,reports=reports))
    qualified=scope=='full32' and calibration['calibration_passed']
    metadata=dict(identity=identity,training_complete=True,epochs_completed=EPOCHS,selected_epoch=best_epoch,
        model_config=dict(input_channels=9,classes=3,base_channels=8),dataset_split='TRAIN',scope=scope,
        pilot_is_smoke_only=scope!='full32',quality_protocol=QUALITY_PROTOCOL,model_qualified=qualified,selected_threshold=calibration['selected_threshold'],
        class_names=CLASS_NAMES,feature_names=FEATURE_NAMES,current_visible_pixels_only=True,completion=False,
        pose_regression=False,orientation_observed=False,absolute_worldXYZ_input=False,TCP_GRU_GT_input=False,
        raw_RGBD_modified=False,TEST_read=False,Native_control_qualified=False)
    checkpoint=out/'best.pt'
    for path,expected in identity['code_sha256'].items():
        if sha256(path)!=expected:raise ValueError('source changed during training')
    if sha256(manifest)!=identity['manifest_sha256']:raise ValueError('manifest changed during training')
    if sha256(complete.partition_path)!=complete.partition_sha256:raise ValueError('frozen pre-capture partition changed during training')
    for record in complete.all_records:
        for key in ('inputs','labels'):
            if sha256(record[key+'_path'])!=record[key]['sha256']:raise ValueError('capture changed during training')
    torch.save(dict(version=VERSION,model_state_dict=best_state,metadata=metadata),checkpoint)
    write_new(out/'history.json',history)
    write_new(out/'status.json',dict(version=VERSION,complete=True,training_started=True,training_complete=True,
        completed_epochs=EPOCHS,identity=identity,selected_epoch=best_epoch,selection_loss=best,
        best_checkpoint_sha256=sha256(checkpoint),model_qualified=qualified,scope=scope,TEST_read=False,
        Native_control_qualified=False,wall_seconds=time.perf_counter()-started))
    return checkpoint
