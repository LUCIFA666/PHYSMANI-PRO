"""Full-resolution predicted foreground, public surface proposal, 25M residual.

All outputs are unqualified research candidates. A raw-depth weighted surface
is not a Cube centre or a physical surface certificate (including AA/Classic
renderer effects). Geometry support and predicted scope are necessary gates,
not empirical precision guarantees. No runtime or control integration exists.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from ..current_cube_rgbd_centre_20261103_v1.model import SELECTED_V5_SHA256
from ..current_cube_rgbd_centre_large_20261104_v2.model import (
    LargeDualCropCubeCentreV2, CentreConfig, fixed_average_pool2d)
from . import VERSION

DECODER_PREFIXES = ('d2.', 'd1.', 'd0.', 'head.')
MINIMUM_CURRENT_VIEWS = 2
SCOPE_M = .20
FOREGROUND_TIE_IS_UNKNOWN = True


def _decoder_tensors(state, shapes):
    if not isinstance(state, dict):
        raise TypeError('Explicit original V5 state required')
    chosen = {k: v for k, v in state.items() if k.startswith(DECODER_PREFIXES)}
    if set(chosen) != set(shapes):
        raise ValueError('Exactly the complete original V5 decoder/head required')
    for key, value in chosen.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float32 or tuple(value.shape) != tuple(shapes[key])
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Exact finite V5 decoder tensor required: '+key)
    return {k: v.detach().clone() for k, v in chosen.items()}


def surface_proposal(features, wide_features, camera_rotations, initial_sphere,
                     source_pixels_uv, foreground_probability):
    """Detached predicted weights; fixed near-before-wide original-UV union.

    Pixel selection is strict pCube>.5, real valid depth, real image, unique
    source UV. Two distinct current cameras are required. Equal camera means
    avoid making the dense near crop a fictitious extra camera. No GT enters.
    """
    if (not isinstance(features, torch.Tensor) or features.ndim != 5
            or features.shape[1:] != (4, 9, 64, 64)):
        raise ValueError('Public near tensor[B,4,9,64,64] required')
    n, device = len(features), features.device
    for value, shape in ((features, (n,4,9,64,64)),
                         (wide_features, (n,4,9,64,64)),
                         (camera_rotations, (n,4,3,3)), (initial_sphere, (n,3)),
                         (foreground_probability, (n,2,4,64,64))):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device != device
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Aligned finite float32 public proposal tensors required')
    if (n < 1 or bool(((foreground_probability < 0) | (foreground_probability > 1)).any())
            or not isinstance(source_pixels_uv, torch.Tensor)
            or tuple(source_pixels_uv.shape) != (n,2,4,64,64,2)
            or source_pixels_uv.dtype != torch.int64 or source_pixels_uv.device != device):
        raise ValueError('Normalized probability and exact integer source UV required')
    if any(v.requires_grad for v in (camera_rotations, initial_sphere)):
        raise ValueError('Public calibration/anchor cannot be optimized')
    with torch.autocast(device_type=device.type, enabled=False):
        rotation = camera_rotations.detach()
        if (not torch.allclose(rotation.transpose(-1,-2)@rotation,
                torch.eye(3,device=device).expand(n,4,3,3), atol=1e-5, rtol=0)
                or bool((torch.linalg.det(rotation) < 1.-1e-5).any())):
            raise ValueError('Proper current public camera rotations required')
        scales = torch.stack((features, wide_features), 1)
        flags = scales[:,:,:,7:9].detach()
        if bool(((flags != 0) & (flags != 1)).any()) or bool((flags[:,:,:,0] > flags[:,:,:,1]).any()):
            raise ValueError('Original binary crop/depth flags required')
        y,x = torch.meshgrid(torch.arange(64,device=device),
                             torch.arange(64,device=device),indexing='ij')
        for scale, stride in ((0,1),(1,2)):
            origin = source_pixels_uv[:,scale,:,0,0]
            expected = origin[:,:,None,None]+torch.stack((x,y),-1)*stride
            if not torch.equal(source_pixels_uv[:,scale], expected):
                raise ValueError('Original near stride1 / wide stride2 UV lattice required')
        uv = source_pixels_uv
        in_image = ((uv[...,0]>=0)&(uv[...,0]<256)&(uv[...,1]>=0)&(uv[...,1]<256))
        if not torch.equal(in_image, flags[:,:,:,1].bool()):
            raise ValueError('Source UV and original image validity differ')
        near_origin = uv[:,0,:,0,0]
        wu, wv = uv[:,1,...,0], uv[:,1,...,1]
        overlap = ((wu>=near_origin[:,:,None,None,0]) & (wu<near_origin[:,:,None,None,0]+64)
                 & (wv>=near_origin[:,:,None,None,1]) & (wv<near_origin[:,:,None,None,1]+64))
        unique = torch.stack((in_image[:,0], in_image[:,1]&~overlap),1)
        p = foreground_probability.detach()
        selected = (p>.5) & flags[:,:,:,0].bool() & unique
        weights = torch.where(selected,p,torch.zeros_like(p))
        relative = .1*scales[:,:,:,4:7].detach()
        world_delta = torch.einsum('bcij,bscjhw->bscihw',rotation,relative)
        mass = weights.sum((1,3,4))
        view_delta = (world_delta*weights[:,:,:,None]).sum((1,4,5))/mass.clamp_min(1e-12)[...,None]
        current_views = selected.sum((1,3,4))>0
        views = current_views.sum(1)
        delta = (view_delta*current_views[...,None]).sum(1)/views.clamp_min(1)[...,None]
        supported = views>=MINIMUM_CURRENT_VIEWS
        proposal = initial_sphere.detach()+delta
        # Finite tensor placeholder is explicitly unavailable, never a current measurement.
        proposal = torch.where(supported[:,None],proposal,initial_sphere.detach())
        return dict(proposal_world=proposal,proposal_supported=supported,
            current_views=views,unique_supported_pixels=selected.sum((1,2,3,4)),
            per_view_pixels=selected.sum((1,3,4)),per_view_mass=mass,
            detached_pixel_weights=weights,unique_source_pixel_mask=unique,
            proposal_is_surface_not_object_centre=True)


class LargeDualCropCubeCentreV3(LargeDualCropCubeCentreV2):
    def __init__(self, config=CentreConfig(), *, encoder_trainable=True):
        if config.spatial_grid != 2:
            raise ValueError('Single prototype preserves the original grid2 / 25M Transformer')
        super().__init__(config, encoder_trainable=encoder_trainable)
        decoder = TinyDepthInfluenceUNet()
        self.d2,self.d1,self.d0,self.head = decoder.d2,decoder.d1,decoder.d0,decoder.head
        self._decoder_initialization = None

    def load_selected_v5_components(self, checkpoint, expected_sha256):
        """Fresh V5 encoder+decoder initialization, never a V2 head warmstart."""
        if self._decoder_initialization is not None or expected_sha256 != SELECTED_V5_SHA256:
            raise ValueError('One explicit original V5 initialization required')
        path = Path(checkpoint).resolve(strict=True)
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        if original != expected_sha256:
            raise ValueError('Original V5 whole checkpoint SHA mismatch')
        encoder = super().load_selected_v5_encoder(path,expected_sha256)
        payload = torch.load(path,map_location='cpu',weights_only=True)
        shapes = {k:tuple(v.shape)for k,v in self.state_dict().items() if k.startswith(DECODER_PREFIXES)}
        chosen = _decoder_tensors(payload.get('model_state_dict'),shapes)
        for name in ('d2','d1','d0','head'):
            getattr(self,name).load_state_dict({k[len(name)+1:]:v for k,v in chosen.items()
                                               if k.startswith(name+'.')},strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != original:
            raise ValueError('Original V5 changed during initialization')
        self._decoder_initialization = dict(path=str(path),sha256=original,
            prefixes=list(DECODER_PREFIXES),parameters=sum(v.numel()for v in chosen.values()),
            old_qualification_transferred=False,centre_or_availability_head_transferred=False)
        return dict(encoder=encoder,decoder=deepcopy(self._decoder_initialization))

    def spec(self):
        value=super().spec()
        value.update(version=VERSION,decoder_initialization=deepcopy(self._decoder_initialization),
            input_contract='six public tensor keys: original five plus dual exact source_pixels_uv',
            output_frame='detached predicted current-depth surface proposal + learned world centre residual',
            foreground_semantics='CF current depth influence 0/1/255; not pure face or full instance',
            foreground_to_centre_gradient='detached; pixel supervision trains decoder',
            minimum_current_camera_support=MINIMUM_CURRENT_VIEWS,
            duplicate_UV_policy='fixed near precedence, wide only previously unsampled source UV',
            no_support_centre_is_unavailable_placeholder=True,
            predicted_scope_gate_is_NOT_true_visibility_or_accuracy_bound=True,
            training_runner_supplied=False)
        return value

    def forward(self, *, features, wide_features, K, camera_rotations,
                initial_sphere, source_pixels_uv):
        if self._encoder_initialization is None or self._decoder_initialization is None:
            raise ValueError('Explicit whole-SHA original V5 encoder+decoder initialization required')
        if (not isinstance(features,torch.Tensor) or features.ndim!=5
                or tuple(features.shape[1:])!=(4,9,64,64) or len(features)<1):
            raise ValueError('Nonempty current public near tensor required')
        n=len(features)
        for value in (features,wide_features):
            if (not isinstance(value,torch.Tensor) or value.shape!=(n,4,9,64,64)
                    or value.dtype!=torch.float32 or value.device!=features.device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Aligned finite original public near/wide tensors required')
        for value,shape in ((camera_rotations,(n,4,3,3)),(initial_sphere,(n,3))):
            if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape
                    or value.dtype!=torch.float32 or value.device!=features.device
                    or not bool(torch.isfinite(value).all()) or value.requires_grad):
                raise ValueError('Finite fixed original public calibration/anchor required')
        if (not isinstance(source_pixels_uv,torch.Tensor)
                or tuple(source_pixels_uv.shape)!=(n,2,4,64,64,2)
                or source_pixels_uv.dtype!=torch.int64 or source_pixels_uv.device!=features.device):
            raise ValueError('Exact original source UV required')
        if (not isinstance(K,torch.Tensor) or K.shape!=(n,4,3,3) or K.dtype!=torch.float32
                or K.device!=features.device or not bool(torch.isfinite(K).all()) or K.requires_grad):
            raise ValueError('Original public signed K required')
        with torch.autocast(device_type=features.device.type,enabled=False):
            if (not torch.allclose(K[:,:,2],K.new_tensor([0,0,1]).expand(n,4,3),atol=1e-6,rtol=0)
                    or bool((torch.linalg.det(K).abs()<1e-8).any())):
                raise ValueError('Original nonsingular signed K required')
        scales=torch.stack((features,wide_features),1)
        images=scales.reshape(n*8,9,64,64)
        a=self.e0(images);b=self.e1(F.max_pool2d(a,2));c=self.e2(F.max_pool2d(b,2));z=self.bottom(F.max_pool2d(c,2))
        def up(v):return F.interpolate(v,scale_factor=2,mode='bilinear',align_corners=False)
        d=self.d2(torch.cat((up(z),c),1));d=self.d1(torch.cat((up(d),b),1))
        logits=self.head(self.d0(torch.cat((up(d),a),1)))
        pixel_logits=logits.reshape(n,2,4,2,64,64)
        probability=pixel_logits.float().softmax(3)[:,:,:,1]
        geometry=surface_proposal(features,wide_features,camera_rotations,initial_sphere,
                                  source_pixels_uv,probability)
        weights=geometry['detached_pixel_weights'].reshape(n*8,1,64,64)
        grid=2
        image_tokens=fixed_average_pool2d(z,grid).flatten(2).transpose(1,2)
        with torch.autocast(device_type=features.device.type,enabled=False):
            mass=fixed_average_pool2d(weights,grid)
            metric=fixed_average_pool2d(images[:,4:7].float()*weights,grid)/mass.clamp_min(1e-12)
            metric=metric.flatten(2).transpose(1,2)
            fractions=torch.cat((mass,fixed_average_pool2d(images[:,8:9].float(),grid)),1).flatten(2).transpose(1,2)
            signed_K=torch.stack((K[:,:,0,0],K[:,:,1,1],K[:,:,0,2],K[:,:,1,2]),-1)/256.
            calibration=torch.cat((signed_K,camera_rotations.flatten(-2)),-1)
            calibration=calibration[:,None].expand(n,2,4,13).reshape(n*8,1,13).expand(n*8,4,13)
        tokens=self.token_projection(torch.cat((image_tokens,metric,fractions,calibration),-1))
        tokens=tokens.reshape(n,2,4,4,self.config.width)
        tokens=tokens+self.scale_embedding[None,:,None,None]+self.spatial_embedding[None,None,None]
        tokens=tokens.reshape(n,32,self.config.width)
        valid=(mass.flatten(1)>0).reshape(n,32)
        mask=torch.cat((torch.zeros(n,1,device=features.device,dtype=torch.bool),~valid),1)
        fused=self.output_norm(self.transformer(torch.cat((self.query.expand(n,-1,-1),tokens),1),src_key_padding_mask=mask)[:,0])
        residual=self.config.position_bound_m*torch.tanh(self.centre_head(fused).float())
        centre=geometry['proposal_world']+residual
        score=F.softplus(self.uncertainty_head(fused).float())*.05+1e-6
        supported=geometry['proposal_supported']
        score=torch.where(supported[:,None],score,torch.full_like(score,.05))
        logit=self.availability_head(torch.cat((fused,valid.float().mean(1,keepdim=True)),-1)).squeeze(-1).float()
        predicted_in_scope=torch.linalg.vector_norm(centre-initial_sphere,dim=-1)<=SCOPE_M
        return dict(centre_world=centre,centre_delta_world=centre-initial_sphere,
            centre_residual_world=residual,measurement_availability_logit=logit,
            uncertainty_score_m=score,uncertainty_score_valid=supported,
            candidate_available=(logit.sigmoid()>=.5)&supported&predicted_in_scope,
            predicted_in_scope=predicted_in_scope,pixel_logits=pixel_logits,
            proposal_world=geometry['proposal_world'],proposal_supported=supported,
            proposal_current_views=geometry['current_views'],proposal_pixels=geometry['unique_supported_pixels'],
            model_qualified=False,position_measurement_qualified=False,runtime_or_control_admitted=False)


def parameter_count(config=CentreConfig()):
    with torch.device('meta'):
        model=LargeDualCropCubeCentreV3(config)
    return sum(p.numel()for p in model.parameters())
