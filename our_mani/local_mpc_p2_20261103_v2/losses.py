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
    available=prediction.get("prediction_available",valid)
    if available.shape!=valid.shape or bool((valid & ~available).any()):
        raise ValueError("infeasible/missing known-command predictions cannot be dropped from the objective")
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
    for count in range(1,valid.shape[1]+1):
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


def position_precision_gate(prediction,batch):
    """All real samples and all 50ms horizons; never means alone."""
    horizons=horizon_metrics(prediction,batch)
    valid=batch['valid'].detach().cpu()
    errors=torch.linalg.vector_norm((prediction['tcp_matrix'][...,:3,3]-batch['target_tcp'][...,:3,3]).detach().cpu(),dim=-1)[valid]
    if not bool(torch.isfinite(errors).all()):
        raise ValueError('nonfinite eligible position error cannot be filtered')
    complete=all(row['samples']>0 for row in horizons.values())
    mean_pass=complete and all(row['TCP_error_m']['mean']<=.03 for row in horizons.values())
    bound_pass=bool(float(errors.max())<=.05)
    return dict(mean_target_m=.03,all_samples_upper_bound_m=.05,horizon_seconds='every_50ms_through_800ms',
                samples=int(valid.sum()),mean_m=float(errors.mean()),p95_m=float(torch.quantile(errors,.95)),
                max_m=float(errors.max()),above_5cm_fraction=float((errors>.05).double().mean()),
                all_horizons_have_labels=complete,mean_target_passed=bool(mean_pass),
                all_samples_bound_passed=bound_pass,passed=bool(mean_pass and bound_pass),
                position_only_not_grip_rotation_or_task_success=True,
                formal_TEST_used=False,Native_task_success_measured=False)
