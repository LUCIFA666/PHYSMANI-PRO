"""Independent v2 checkpoint family: normalized branches and TCP residual R."""
import torch
from torch import nn

from ..world_tcp_command_warmstart_20261104_v1.model import (
    CandidateConfig,WorldTCPCommandTransformerV1,command_imitation_loss,
)
from . import VERSION


def quaternion_rotation(q):
    if q.shape[-1]!=4 or not bool(torch.isfinite(q).all()):
        raise ValueError('Measured finite TCP quaternion xyzw required')
    q=q.float();q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


class WorldTCPCommandTransformerV2(WorldTCPCommandTransformerV1):
    def __init__(self,config=CandidateConfig()):
        super().__init__(config)
        for name in ('visual_projection','robot_projection','calibration_projection','task_projection'):
            setattr(self,name,nn.Sequential(getattr(self,name),nn.LayerNorm(config.width)))
        nn.init.normal_(self.rotation_head.weight,std=1e-3)
        with torch.no_grad():self.rotation_head.bias.copy_(torch.tensor([1.,0.,0.,0.,1.,0.]))

    def forward(self,*,visual_memory,visual_memory_mask,current_robot,camera_calibration,task_e):
        result=super().forward(visual_memory=visual_memory,visual_memory_mask=visual_memory_mask,
            current_robot=current_robot,camera_calibration=camera_calibration,task_e=task_e)
        with torch.autocast(current_robot.device.type,enabled=False):
            actual=quaternion_rotation(current_robot[:,10:14])
            result['world_tcp_rotations']=actual[:,None]@result['world_tcp_rotations'].float()
        return result

    def spec(self):
        value=super().spec()
        value.update(version=VERSION,checkpoint_family='v2_only_not_v1_state_dict_compatible',
            branch_normalization='LayerNorm(width) after each of four input projections',
            orientation='R_world=measured_current_TCP_R@predicted_residual_R6',
            rotation_weight_initialization_std=1e-3,rotation_bias='identity_two_columns',
            position='measured_TCP_position+unbounded_world_displacement',
            source_of_TCP_frame='current measured public proprioception, not GT object frame')
        return value


def parameter_count(config=CandidateConfig()):
    with torch.device('meta'):model=WorldTCPCommandTransformerV2(config)
    return sum(value.numel()for value in model.parameters())
