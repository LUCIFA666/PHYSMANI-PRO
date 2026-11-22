"""Large cross-view centre model; the frozen V1 and its data remain unchanged."""
from dataclasses import asdict, dataclass
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from ..current_cube_rgbd_centre_20261103_v1.model import (
    DualCropCubeCentreHeadV1, ENCODER_PREFIXES, SELECTED_V5_SHA256,
    centre_presence_loss)
from . import VERSION


@dataclass(frozen=True)
class CentreConfig:
    width: int = 512
    layers: int = 8
    heads: int = 8
    spatial_grid: int = 2
    dropout: float = .1
    position_bound_m: float = .25

    def __post_init__(self):
        if (any(type(v)is not int or v<1 for v in (self.width,self.layers,self.heads,self.spatial_grid))
            or self.width%self.heads or self.spatial_grid>4 or not 0<=self.dropout<1
            or not .20<self.position_bound_m<=.5):
            raise ValueError('Valid explicit large centre architecture required')


class LargeDualCropCubeCentreV1(nn.Module):
    def __init__(self, config=CentreConfig(), *, encoder_trainable=True):
        super().__init__()
        if type(config)is not CentreConfig or type(encoder_trainable)is not bool:
            raise ValueError('Explicit typed architecture/trainability required')
        self.config=config;self.encoder_trainable=encoder_trainable;self._encoder_initialization=None
        source=TinyDepthInfluenceUNet()
        self.e0,self.e1,self.e2,self.bottom=source.e0,source.e1,source.e2,source.bottom
        # 64 frozen-transfer image features; relative metric xyz; two current
        # validity fractions; signed K4 and camera-to-world R9.
        self.token_projection=nn.Linear(82,config.width)
        self.scale_embedding=nn.Parameter(torch.empty(2,config.width))
        self.spatial_embedding=nn.Parameter(torch.empty(config.spatial_grid**2,config.width))
        self.query=nn.Parameter(torch.empty(1,1,config.width))
        layer=nn.TransformerEncoderLayer(config.width,config.heads,config.width*4,
            dropout=config.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.output_norm=nn.LayerNorm(config.width)
        self.centre_head=nn.Linear(config.width,3)
        self.uncertainty_head=nn.Linear(config.width,3)
        self.availability_head=nn.Sequential(nn.Linear(config.width+1,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.centre_head.weight);nn.init.zeros_(self.centre_head.bias)
        for value in (self.scale_embedding,self.spatial_embedding,self.query):nn.init.normal_(value,std=.02)
        for name,p in self.named_parameters():
            if name.startswith(ENCODER_PREFIXES):p.requires_grad_(encoder_trainable)

    def load_selected_v5_encoder(self, checkpoint, expected_sha256):
        # Exactly the old explicit prefix-only transfer and evidence checks.
        # No old centre head, decoder, or old certificate is inherited.
        return DualCropCubeCentreHeadV1.load_selected_v5_encoder(self,checkpoint,expected_sha256)

    def spec(self):
        return dict(version=VERSION,architecture=asdict(self.config),
            parameters=sum(p.numel()for p in self.parameters()),
            encoder_initialization=deepcopy(self._encoder_initialization),
            input_contract='unchanged public_model_arrays near/wide current RGBD9 and public K/R/Sphere',
            output_frame='world centre delta relative to initial public Sphere',
            proposed_scope_m=.20,model_qualified=False,position_measurement_qualified=False,
            uncertainty_is_model_score_not_CI=True,runtime_or_control_admitted=False)

    def forward(self, *, features,wide_features,K,camera_rotations,initial_sphere):
        if self._encoder_initialization is None:raise ValueError('Explicit original V5 prefix transfer required')
        if not isinstance(features,torch.Tensor)or features.ndim!=5 or features.shape[1:]!=(4,9,64,64):
            raise ValueError('Current public features[B,4,9,64,64] required')
        n=features.shape[0]
        for value,shape in ((features,(n,4,9,64,64)),(wide_features,(n,4,9,64,64)),
            (K,(n,4,3,3)),(camera_rotations,(n,4,3,3)),(initial_sphere,(n,3))):
            if (not isinstance(value,torch.Tensor)or value.shape!=shape or value.dtype!=torch.float32
                or value.device!=features.device or not bool(torch.isfinite(value).all())):
                raise ValueError('Aligned finite original float32 public features/calibration required')
        if any(value.requires_grad for value in (K,camera_rotations,initial_sphere)):
            raise ValueError('Public calibration/anchor cannot be supervised variables')
        with torch.autocast(device_type=features.device.type,enabled=False):
            if (not torch.allclose(K[:,:,2],K.new_tensor([0,0,1]).expand(n,4,3),atol=1e-6,rtol=0)
                or bool((torch.linalg.det(K).abs()<1e-8).any())
                or not torch.allclose(camera_rotations.transpose(-1,-2)@camera_rotations,
                    torch.eye(3,device=K.device).expand(n,4,3,3),atol=1e-5,rtol=0)
                or bool((torch.linalg.det(camera_rotations)<1.-1e-5).any())):
                raise ValueError('Original signed K and proper public world-camera rotations required')
        scales=torch.stack((features,wide_features),1)
        flags=scales[:,:,:,7:9]
        if bool(((flags!=0)&(flags!=1)).any())or bool((flags[:,:,:,0]>flags[:,:,:,1]).any()):
            raise ValueError('Original binary depth/crop validity required')
        images=scales.reshape(n*8,9,64,64);grid=self.config.spatial_grid
        x=self.e0(images);x=self.e1(F.max_pool2d(x,2));x=self.e2(F.max_pool2d(x,2));x=self.bottom(F.max_pool2d(x,2))
        image_tokens=F.adaptive_avg_pool2d(x,(grid,grid)).flatten(2).transpose(1,2)
        metric=F.adaptive_avg_pool2d(images[:,4:7],(grid,grid)).flatten(2).transpose(1,2)
        fractions=F.adaptive_avg_pool2d(images[:,7:9],(grid,grid)).flatten(2).transpose(1,2)
        signed_K=torch.stack((K[:,:,0,0],K[:,:,1,1],K[:,:,0,2],K[:,:,1,2]),-1)/256.
        calibration=torch.cat((signed_K,camera_rotations.flatten(-2)),-1)
        calibration=calibration[:,None].expand(n,2,4,13).reshape(n*8,1,13).expand(n*8,grid**2,13)
        tokens=self.token_projection(torch.cat((image_tokens,metric,fractions,calibration),-1))
        tokens=tokens.reshape(n,2,4,grid**2,self.config.width)
        tokens=tokens+self.scale_embedding[None,:,None,None]+self.spatial_embedding[None,None,None]
        tokens=tokens.reshape(n,8*grid**2,self.config.width)
        valid=F.adaptive_max_pool2d(images[:,7:8],(grid,grid)).flatten(1).bool().reshape(n,8*grid**2)
        mask=torch.cat((torch.zeros(n,1,device=features.device,dtype=torch.bool),~valid),1)
        tokens=torch.cat((self.query.expand(n,-1,-1),tokens),1)
        fused=self.output_norm(self.transformer(tokens,src_key_padding_mask=mask)[:,0])
        delta=self.config.position_bound_m*torch.tanh(self.centre_head(fused).float())
        score=F.softplus(self.uncertainty_head(fused).float())*.05+1e-6
        score=torch.where(valid.any(1,keepdim=True),score,torch.full_like(score,.05))
        logit=self.availability_head(torch.cat((fused,valid.float().mean(1,keepdim=True)),-1)).squeeze(-1).float()
        return dict(centre_world=initial_sphere+delta,centre_delta_world=delta,
            measurement_availability_logit=logit,uncertainty_score_m=score,
            uncertainty_score_valid=valid.any(1))


def parameter_count(config=CentreConfig()):
    with torch.device('meta'):model=LargeDualCropCubeCentreV1(config)
    return sum(p.numel()for p in model.parameters())
