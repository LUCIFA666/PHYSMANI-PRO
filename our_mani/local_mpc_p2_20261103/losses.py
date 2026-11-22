"""One-step/recorded-reference rollout objectives; all physical units explicit."""
import numpy as np
import torch

from .dataset import DOF


def servo_loss(prediction,batch,*,q_scale=.02,dq_scale=.5,aperture_scale=.1,TCP_scale=.005,rotation_scale=.1):
    if not all(np.isfinite(s) and s>0 for s in (q_scale,dq_scale,aperture_scale,TCP_scale,rotation_scale)):
        raise ValueError("explicit positive physical loss scales required")
    valid=batch["valid"]
    if valid.ndim != 2 or valid.dtype!=torch.bool or not bool(valid.any(-1).all()):
        raise ValueError("each batched window needs real executed labels")
    if (batch['target_states'].shape != (*valid.shape, 17)
            or batch['target_tcp'].shape != (*valid.shape, 4, 4)):
        raise ValueError('aligned actual-state and TCP targets required')
    expected = {'q': (*valid.shape, DOF), 'dq': (*valid.shape, DOF),
                'opening': (*valid.shape, 2), 'tcp_matrix': (*valid.shape, 4, 4)}
    if any(prediction[key].shape != shape for key, shape in expected.items()):
        raise ValueError('predictions must preserve batch and horizon axes')
    target=torch.where(valid[...,None],batch["target_states"],torch.zeros_like(batch["target_states"]))
    tcp=torch.where(valid[...,None,None],batch["target_tcp"],torch.zeros_like(batch["target_tcp"]))
    errors={
        'q':((prediction['q']-target[...,:DOF])/q_scale).square().mean(-1),
        'dq':((prediction['dq']-target[...,DOF:2*DOF])/dq_scale).square().mean(-1),
        'opening':((prediction['opening']-target[...,2*DOF:2*DOF+2])/aperture_scale).square().mean(-1),
        'TCP_position':((prediction['tcp_matrix'][...,:3,3]-tcp[...,:3,3])/TCP_scale).square().sum(-1),
        'TCP_rotation':(prediction['tcp_matrix'][...,:3,:3]-tcp[...,:3,:3]).square().sum((-1,-2))/(2*rotation_scale**2)}
    value=errors['q']+.2*errors['dq']+.2*errors['opening']+errors['TCP_position']+.2*errors['TCP_rotation']
    value=torch.where(valid,value,torch.zeros_like(value)).sum(-1)/valid.sum(-1)
    weight=batch.get('weight',torch.ones_like(value))
    if (weight.shape != value.shape or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(weight).all())
            or not bool((weight>0).all())):raise ValueError("finite predictions/positive weights required")
    return (value*weight).sum()/weight.sum()


def teacher_forced_prediction(model,batch):
    # Input is current actual state at each real tick, not next-state truth.
    return model(batch['teacher_states'],batch['references'])


def horizon_metrics(prediction,batch):
    """Every declared finite label counted; tails explicitly lack target labels."""
    result={}
    pred={k:v.detach().cpu().numpy() for k,v in prediction.items() if isinstance(v,torch.Tensor)}
    target=batch['target_states'].detach().cpu().numpy();tcp=batch['target_tcp'].detach().cpu().numpy()
    valid=batch['valid'].cpu().numpy()
    for count in (1,2,4,8,16):
        if count>valid.shape[1]:continue
        mask=valid[:,count-1];n=int(mask.sum())
        row={'samples':n,'expected_samples':n,'time_s':count*.05,'finite_cohort':True}
        if n:
            q=pred['q'][mask,count-1];v=pred['dq'][mask,count-1];opening=pred['opening'][mask,count-1]
            T=pred['tcp_matrix'][mask,count-1];truth=target[mask,count-1];targetT=tcp[mask,count-1]
            if not all(np.isfinite(a).all() for a in (q,v,opening,T,truth,targetT)):
                raise ValueError("nonfinite available-horizon sample cannot be filtered")
            relative=np.swapaxes(T[:,:3,:3],-1,-2)@targetT[:,:3,:3]
            angle=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))
            arrays=dict(q_rms_rad=np.sqrt(((q-truth[:,:DOF])**2).mean(-1)),
                        dq_rms_rad_s=np.sqrt(((v-truth[:,DOF:2*DOF])**2).mean(-1)),
                        aperture_MAE=np.abs(opening-truth[:,2*DOF:2*DOF+2]).mean(-1),
                        TCP_error_m=np.linalg.norm(T[:,:3,3]-targetT[:,:3,3],axis=-1),rotation_error_rad=angle)
            for name,a in arrays.items():row[name]=dict(mean=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max()))
        result[str(count)]=row
    return result
