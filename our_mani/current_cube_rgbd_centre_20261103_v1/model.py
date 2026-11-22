"""Unqualified dual-crop current-RGBD centre/availability head preparation.

The encoder must be explicitly initialized from the immutable selected V5
segmentation weights. New heads inherit no old model or position certificate.
Inputs are current public tensors only; privileged targets belong to the
separate loss function. This module supplies no runtime, threshold or policy.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet

VERSION='dual_crop_current_RGBD_Cube_centre_availability_head_20261103_v1'
SELECTED_V5_SHA256='dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa'
ENCODER_PREFIXES=('e0.','e1.','e2.','bottom.')
UNCERTAINTY_SCALE_M=.05
UNCERTAINTY_FLOOR_M=1e-6


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selected_encoder_tensors(state,expected_shapes):
    """Pure strict prefix extraction; never load a decoder or any new head."""
    if not isinstance(state,dict):raise TypeError('Selected model state dictionary required')
    selected={k:v for k,v in state.items()if k.startswith(ENCODER_PREFIXES)}
    if set(selected)!=set(expected_shapes):raise ValueError('All and only four original encoder prefixes required')
    for key,value in selected.items():
        if (not isinstance(value,torch.Tensor)or value.device.type!='cpu'
            or value.dtype!=torch.float32 or tuple(value.shape)!=tuple(expected_shapes[key])
            or not bool(torch.isfinite(value).all())):
            raise ValueError('Exact finite CPU float32 selected encoder tensor required: '+key)
    return {k:v.detach().clone()for k,v in selected.items()}


def fuse_public_views(delta_camera,uncertainty_camera,attention_logits,rotations,valid_views):
    """Four cameras, optionally two crop scales; never a confidence set."""
    if (delta_camera.ndim!=3 or delta_camera.shape[1]not in (4,8)or delta_camera.shape[-1]!=3
        or uncertainty_camera.shape!=delta_camera.shape
        or attention_logits.shape!=delta_camera.shape[:2]
        or rotations.shape!=delta_camera.shape[:2]+(3,3)
        or valid_views.shape!=delta_camera.shape[:2] or valid_views.dtype!=torch.bool):
        raise ValueError('Aligned Bx4 or dual-scale Bx8 public predictions/calibration required')
    if (not all(bool(torch.isfinite(x).all())for x in (delta_camera,uncertainty_camera,attention_logits,rotations))
        or bool((uncertainty_camera<=0).any())):
        raise ValueError('Finite positive uncertainty scores required')
    masked=attention_logits.masked_fill(~valid_views,torch.finfo(attention_logits.dtype).min)
    weights=torch.softmax(masked,dim=1)*valid_views.to(delta_camera.dtype)
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    world=torch.einsum('bvij,bvj->bvi',rotations,delta_camera)
    centre=(weights[...,None]*world).sum(1)
    variance=torch.einsum('bvij,bvj->bvi',rotations.square(),uncertainty_camera.square())
    variance=(weights[...,None]*(variance+(world-centre[:,None]).square())).sum(1)
    score=torch.sqrt(variance+UNCERTAINTY_FLOOR_M**2)
    # No current metric-depth support is UNKNOWN, never micrometre certainty.
    # This finite placeholder is explicitly invalid in the model output.
    score=torch.where(valid_views.any(1,keepdim=True),score,torch.full_like(score,UNCERTAINTY_SCALE_M))
    return centre,score,weights


class DualCropCubeCentreHeadV1(nn.Module):
    """Shared four-view near/wide encoder and fresh heads; no qualification."""
    def __init__(self,*,encoder_trainable=False):
        super().__init__()
        if type(encoder_trainable)is not bool:raise ValueError('Explicit encoder_trainable bool required')
        original=TinyDepthInfluenceUNet()
        self.e0,self.e1,self.e2,self.bottom=original.e0,original.e1,original.e2,original.bottom
        self.view_projection=nn.Sequential(nn.Linear(68,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.scale_embedding=nn.Parameter(torch.zeros(2,64))
        self.delta_camera_head=nn.Linear(64,3)
        self.uncertainty_camera_head=nn.Linear(64,3)
        self.attention_head=nn.Linear(64,1)
        self.availability_head=nn.Sequential(nn.Linear(65,32),nn.SiLU(),nn.Linear(32,1))
        self.encoder_trainable=encoder_trainable
        for name,p in self.named_parameters():
            if name.startswith(ENCODER_PREFIXES):p.requires_grad_(encoder_trainable)
        self._encoder_initialization=None

    def load_selected_v5_encoder(self,checkpoint,expected_sha256):
        """Explicit whole-SHA transfer of encoder only; no old qualification."""
        if self._encoder_initialization is not None:raise ValueError('Initialize encoder exactly once on a fresh model')
        if expected_sha256!=SELECTED_V5_SHA256:raise ValueError('Only the predeclared original selected V5 SHA accepted')
        path=Path(checkpoint).resolve(strict=True)
        if _sha(path)!=expected_sha256:raise ValueError('Selected V5 whole-checkpoint SHA mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload.get('metadata',{})
        if (meta.get('stage')!='selected_pending_fresh_calibration'
            or meta.get('training_complete')is not True or meta.get('epochs_completed')!=160
            or meta.get('actual_optimizer_steps')!=1920 or meta.get('model_qualified')is not False
            or meta.get('scope')!='full32' or meta.get('dataset_split')!='TRAIN'
            or meta.get('model_config')!=dict(input_channels=9,classes=2,base_channels=8)):
            raise ValueError('Exact completed unqualified selected V5 segmentation source required')
        own={k:tuple(v.shape)for k,v in self.state_dict().items()if k.startswith(ENCODER_PREFIXES)}
        selected=_selected_encoder_tensors(payload.get('model_state_dict'),own)
        for name in ('e0','e1','e2','bottom'):
            getattr(self,name).load_state_dict({k[len(name)+1:]:v for k,v in selected.items()if k.startswith(name+'.')},strict=True)
        digest=hashlib.sha256()
        for k,v in sorted(selected.items()):digest.update((k+str(v.shape)).encode());digest.update(v.numpy().tobytes())
        self._encoder_initialization=dict(path=str(path),sha256=expected_sha256,
            encoder_tensor_sha256=digest.hexdigest(),prefixes=list(ENCODER_PREFIXES),
            decoder_or_segmentation_head_loaded=False,new_heads_initialized_from_checkpoint=False,
            old_qualification_transferred=False)
        return deepcopy(self._encoder_initialization)

    def spec(self):
        return dict(version=VERSION,input_shapes=dict(features=[4,9,64,64],wide_features=[4,9,64,64],K=[4,3,3],
            camera_rotations=[4,3,3],initial_sphere=[3]),
            initialization=deepcopy(self._encoder_initialization),encoder_trainable=self.encoder_trainable,
            feature_contract='original near make_near_goal_crops unchanged; new wide128 stride2 real-pixel features use the same current public source',
            output_scope='proposed bounded near+transition current measurement availability; not world object existence',
            proposed_data_extent_m=.20,extent_or_coverage_qualified=False,
            no_depth_uncertainty_score_is_invalid_placeholder=True,
            uncertainty_is_model_score_not_CI=True,model_qualified=False,position_measurement_qualified=False,
            runtime_or_control_admitted=False)

    def forward(self,*,features,wide_features,K,camera_rotations,initial_sphere):
        if self._encoder_initialization is None:
            raise ValueError('Explicit selected V5 encoder whole-SHA initialization required before forward')
        if not isinstance(features,torch.Tensor)or features.ndim!=5 or features.shape[1:]!=(4,9,64,64):
            raise ValueError('Only current public tensor features[B,4,9,64,64]; no dictionaries or labels')
        n=features.shape[0]
        if n<1:raise ValueError('Nonempty current public batch required')
        values=((features,(n,4,9,64,64)),(wide_features,(n,4,9,64,64)),(K,(n,4,3,3)),
                (camera_rotations,(n,4,3,3)),(initial_sphere,(n,3)))
        for value,shape in values:
            if (not isinstance(value,torch.Tensor)or value.shape!=shape or value.dtype!=torch.float32
                or value.device!=features.device or not bool(torch.isfinite(value).all())):
                raise ValueError('Aligned finite float32 public feature/calibration tensors required')
        if K.requires_grad or camera_rotations.requires_grad or initial_sphere.requires_grad:
            raise ValueError('Public calibration/Sphere are fixed inputs, not supervised optimization variables')
        if (not torch.allclose(K[:,:,2,:],K.new_tensor([0.,0.,1.]).expand(n,4,3),rtol=0,atol=1e-6)
            or bool((torch.linalg.det(K).abs()<1e-8).any())
            or not torch.allclose(camera_rotations.transpose(-1,-2)@camera_rotations,
                                  torch.eye(3,dtype=K.dtype,device=K.device).expand(n,4,3,3),rtol=0,atol=1e-5)
            or bool((torch.linalg.det(camera_rotations)<1.-1e-5).any())):
            raise ValueError('Original signed K and proper camera-to-world rotations required')
        scales=torch.stack((features,wide_features),dim=1)
        flags=scales[:,:,:,7:9]
        if bool(((flags!=0)&(flags!=1)).any())or bool((flags[:,:,:,0]>flags[:,:,:,1]).any()):
            raise ValueError('Original binary depth/crop valid channels required')
        x=scales.reshape(n*8,9,64,64)
        x=self.e0(x);x=self.e1(F.max_pool2d(x,2));x=self.e2(F.max_pool2d(x,2));x=self.bottom(F.max_pool2d(x,2))
        pooled=x.mean((-1,-2)).reshape(n,2,4,64)
        signed_K=torch.stack((K[:,:,0,0],K[:,:,1,1],K[:,:,0,2],K[:,:,1,2]),-1)/256.
        hidden=self.view_projection(torch.cat((pooled,signed_K[:,None].expand(n,2,4,4)),-1))
        hidden=hidden+self.scale_embedding[None,:,None,:]
        hidden=hidden.reshape(n,8,64)
        valid=scales[:,:,:,7].bool().flatten(-2).any(-1).reshape(n,8)
        delta,score,weights=fuse_public_views(self.delta_camera_head(hidden),
            F.softplus(self.uncertainty_camera_head(hidden))*UNCERTAINTY_SCALE_M+UNCERTAINTY_FLOOR_M,
            self.attention_head(hidden).squeeze(-1),camera_rotations[:,None].expand(n,2,4,3,3).reshape(n,8,3,3),valid)
        fused=(weights[...,None]*hidden).sum(1)
        logit=self.availability_head(torch.cat((fused,valid.float().mean(1,keepdim=True)),-1)).squeeze(-1)
        return dict(centre_delta_world=delta,centre_world=initial_sphere+delta,
            measurement_availability_logit=logit,uncertainty_score_m=score,
            uncertainty_score_valid=valid.any(1),
            view_scale_weights=weights.reshape(n,2,4),current_depth_view_scale_mask=valid.reshape(n,2,4))


def centre_presence_loss(output,centre_target_world,centre_valid,presence_target,*,centre_scale_m=.05):
    """Separate label-side targets. Unknown presence masks both losses to zero.

    Presence=-1 is UNKNOWN, 0 is a proved in-scope negative, 1 a positive.
    Physical Shape existence and an available GT centre do not imply presence.
    The positive uncertainty is fitted by a normalized residual NLL, not a
    statistical coverage certificate. All-UNKNOWN batches have finite zero
    loss and zero gradients for centre, presence and uncertainty predictions.
    """
    centre=output['centre_world'];logit=output['measurement_availability_logit'];score=output['uncertainty_score_m']
    if (not all(isinstance(x,torch.Tensor)for x in (centre,logit,score))or centre.ndim!=2 or centre.shape[1]!=3
        or logit.shape!=centre.shape[:1]or score.shape!=centre.shape):
        raise ValueError('Aligned centre/presence/uncertainty tensors required')
    n=centre.shape[0]
    if (n<1 or not all(bool(torch.isfinite(x).all())for x in (centre,logit,score))
        or bool((score<=0).any())or not isinstance(centre_scale_m,(int,float))or not 0<centre_scale_m<1):
        raise ValueError('Finite predictions, positive scores and fixed positive loss scale required')
    if (not all(isinstance(x,torch.Tensor)for x in (centre_target_world,centre_valid,presence_target))
        or centre_target_world.shape!=centre.shape or centre_valid.shape!=(n,)or centre_valid.dtype!=torch.bool
        or presence_target.shape!=(n,)or presence_target.dtype not in (torch.int32,torch.int64)
        or any(x.device!=centre.device for x in (logit,score,centre_target_world,centre_valid,presence_target))
        or bool(((presence_target< -1)|(presence_target>1)).any())):
        raise ValueError('Explicit separate label-side centre_valid and presence[-1/0/1] required')
    known=presence_target>=0;valid=centre_valid&(presence_target==1)
    if not bool(torch.isfinite(centre_target_world[valid]).all()):raise ValueError('Finite actual valid centre labels required')
    target=torch.where(valid[:,None],centre_target_world,centre.detach())
    sigma=(score/centre_scale_m).clamp_min(1e-6)
    term=.5*((centre-target)/centre_scale_m/sigma).square()+torch.log(sigma)
    centre_loss=torch.where(valid[:,None],term,torch.zeros_like(term)).sum()/valid.sum().clamp_min(1)/3.
    bce=F.binary_cross_entropy_with_logits(logit,presence_target.clamp_min(0).to(logit.dtype),reduction='none')
    presence_loss=torch.where(known,bce,torch.zeros_like(bce)).sum()/known.sum().clamp_min(1)
    return dict(total=centre_loss+presence_loss,centre=centre_loss,presence=presence_loss,
        centre_samples=int(valid.sum()),known_presence_samples=int(known.sum()),
        unknown_samples=int((~known).sum()),uncertainty_qualified=False)
