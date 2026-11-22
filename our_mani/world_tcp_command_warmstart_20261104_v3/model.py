"""598 public memory tokens: coarse history plus fine surface-feature stream."""
import torch
from torch import nn
from ..world_tcp_command_warmstart_20261104_v2.model import (CandidateConfig,WorldTCPCommandTransformerV2,quaternion_rotation,command_imitation_loss)
from ..local_mpc_p4_20261103.model import rotation_6d_to_matrix
from . import VERSION

class WorldTCPCommandTransformerV3(WorldTCPCommandTransformerV2):
    def __init__(self,config=CandidateConfig()):
        super().__init__(config);d=config.width
        self.fine_visual_projection=nn.Linear(512,d)
        self.fine_geometry_projection=nn.Linear(6,d)
        self.fine_positions=nn.Parameter(torch.empty(1,512,d))
        self.fine_type=nn.Parameter(torch.empty(1,1,d))
        self.fine_token_norm=nn.LayerNorm(d)
        nn.init.normal_(self.fine_positions,std=.02);nn.init.normal_(self.fine_type,std=.02)
        self.register_buffer('fine_geometry_scale',torch.tensor([2.,2.,2.,1.,1.,1.]))

    def _fine_tokens(self,visual,geometry,mask,lengths,current_robot):
        b=len(current_robot);device=current_robot.device
        if (not isinstance(visual,torch.Tensor)or not visual.is_floating_point()or visual.shape!=(b,512,512)
                or not isinstance(geometry,torch.Tensor)or geometry.dtype!=torch.float32 or geometry.shape!=(b,512,6)
                or not isinstance(mask,torch.Tensor)or mask.dtype!=torch.bool or mask.shape!=(b,512)
                or any(v.device!=device for v in (visual,geometry,mask))):
            raise ValueError('Actual finevisual[B,512,512],FP32nominalgeometry[B,512,6] and frame mask required')
        expected=torch.stack((lengths>=2,torch.ones_like(lengths,dtype=torch.bool)),1)[:,:,None].expand(b,2,256).reshape(b,512)
        if not torch.equal(mask,expected):raise ValueError('Two actual nearest frame groups required; never duplicate missing previous')
        observed_geometry=geometry[mask]
        if (not bool(torch.isfinite(visual[mask]).all())or not bool(torch.isfinite(observed_geometry).all())
                or bool(((observed_geometry[:,3:]<0)|(observed_geometry[:,3:]>1)).any())):
            raise ValueError('Finite fine surfaces and validFraction/UV in[0,1] required')
        clean_v=torch.where(mask[...,None],visual,torch.zeros_like(visual))
        clean_g=torch.where(mask[...,None],geometry,torch.zeros_like(geometry))
        depth_valid=(clean_g[...,3]>0)&mask
        relative=torch.where(depth_valid[...,None],clean_g[...,:3]-current_robot[:,None,7:10].float(),torch.zeros_like(clean_g[...,:3]))
        geo=torch.cat((relative,clean_g[...,3:]),-1)*self.fine_geometry_scale
        return self.fine_token_norm(self.fine_visual_projection(clean_v)+self.fine_geometry_projection(geo)+self.fine_positions+self.fine_type)

    def forward(self,*,visual_memory,visual_memory_mask,current_robot,camera_calibration,task_e,fine_visual,fine_geometry,fine_token_mask):
        if (not isinstance(visual_memory,torch.Tensor) or not visual_memory.is_floating_point()
                or visual_memory.ndim!=3 or visual_memory.shape[1:]!=(86,1024)
                or visual_memory.shape[0]<1):raise ValueError('Actual frozen public1024-token memory[B,86,1024] required')
        b=visual_memory.shape[0];device=visual_memory.device
        if (not isinstance(visual_memory_mask,torch.Tensor) or visual_memory_mask.dtype!=torch.bool
                or visual_memory_mask.shape!=(b,86) or visual_memory_mask.device!=device
                or not bool(visual_memory_mask[:,0].all())):
            raise ValueError('Actual causal prefix mask[B,86], including cls token, required')
        grouped=visual_memory_mask[:,1:].reshape(b,5,17)
        if (not bool((grouped==grouped[:,:,:1]).all()) or not bool(grouped[:,0].all())
                or bool((grouped[:,1:,0]&~grouped[:,:-1,0]).any())):
            raise ValueError('Whole observed frame groups must be a causal prefix; no invented future tokens')
        shapes=(('current_robot',current_robot,(b,14)),('camera_calibration',camera_calibration,(b,4,23)),('task_e',task_e,(b,16)))
        for name,value,shape in shapes:
            if (not isinstance(value,torch.Tensor) or not value.is_floating_point()
                    or value.shape!=shape or value.device!=device or not bool(torch.isfinite(value).all())):
                raise ValueError('Finite public '+name+' shape '+str(shape)+' required')
        K=camera_calibration[:,:,:9].float().reshape(b,4,3,3)
        pose=camera_calibration[:,:,9:21].float().reshape(b,4,3,4)
        R=pose[:,:,:3,:3];clips=camera_calibration[:,:,21:23].float()
        with torch.autocast(device.type,enabled=False):
            calibration_valid=(not bool((torch.linalg.det(K).abs()<1e-8).any())
                and torch.allclose(R.transpose(-1,-2)@R,torch.eye(3,device=device).expand(b,4,3,3),atol=1e-4,rtol=0)
                and torch.allclose(torch.linalg.det(R),torch.ones(b,4,device=device),atol=1e-4,rtol=0)
                and not bool((clips[:,:,0]<=0).any())and not bool((clips[:,:,1]<=clips[:,:,0]).any())
                and torch.allclose(current_robot[:,10:14].float().norm(dim=-1),torch.ones(b,device=device),atol=1e-3,rtol=0))
        if not calibration_valid:
            raise ValueError('Real public signedK/camera-to-world/clip and unit TCP quaternion required')
        if (not bool(torch.isfinite(visual_memory[visual_memory_mask]).all())
                or bool((task_e<0).any())
                or not torch.allclose(task_e.reshape(b,2,8).sum(-1),task_e.new_ones(b,2),atol=1e-5,rtol=0)):
            raise ValueError('Finite observed memory and actual predicted SimNorm e16 required')
        # No dq, gripper-open, contact, c, object-frame or future-clock channel.
        # Camera features are the measured/public K9 + camera-to-world3x4 +clip2.
        clean=torch.where(visual_memory_mask[...,None],visual_memory,torch.zeros_like(visual_memory))
        memory=self.visual_projection(clean)+self.memory_positions
        fine_tokens=self._fine_tokens(fine_visual,fine_geometry,fine_token_mask,grouped[:,:,0].sum(-1),current_robot)
        calibration=self.calibration_projection(camera_calibration*self.calibration_scale)
        conditions=torch.cat((self.robot_projection(current_robot*self.robot_scale)[:,None],
                              calibration,self.task_projection(task_e)[:,None]),1)+self.condition_types
        tokens=torch.cat((memory,fine_tokens,conditions,self.queries.expand(b,-1,-1)),1)
        padding=torch.cat((~visual_memory_mask,~fine_token_mask,torch.zeros(b,14,dtype=torch.bool,device=device)),1)
        result=self.output_norm(self.transformer(tokens,src_key_padding_mask=padding)[:,-8:])
        # These are unbounded world-frame command displacements, not object
        # observations and not a bounded physical actuator action.
        positions=current_robot[:,7:10,None].transpose(1,2)+self.position_head(result).float()
        rotations=rotation_6d_to_matrix(self.rotation_head(result).float())
        with torch.autocast(current_robot.device.type,enabled=False):
            rotations=quaternion_rotation(current_robot[:,10:14])[:,None]@rotations
        opening_logits=self.opening_head(result).float().squeeze(-1)
        return dict(world_tcp_positions=positions,world_tcp_rotations=rotations,
            opening_intent=opening_logits.sigmoid(),opening_logits=opening_logits,
            proposal_logits=self.demo_likelihood_head(result).float().squeeze(-1),handoff_tau_seconds=None,
            proposal_score_semantics='expert_command_demonstration_mixture_not_task_success',
            output_semantics='expert_command_intent_not_observed_state',
            physical_handoff_qualified=False,calibrated_success_probability=False)

    def spec(self):
        value=super().spec();value.update(version=VERSION,checkpoint_family='v3_only_coarse_plus_fine',
            fine_visual_shape=[512,512],fine_geometry_shape=[512,6],fine_mask_shape=[512],memory_tokens=598,
            fine_geometry='nominal all-surface rawZ worldXYZmean/validFraction/pixelCenterUVmean; not object_state',
            geometry_cache_dtype='FP32',geometry_projection_frame='worldXYZ minus measured currentTCP; divided by.5m; unknownzero remainsunknown',
            pixel_convention='u+.5/v+.5, publicK/T unchanged',previous_frame_missing='maskFalse; no duplicated observation',
            fine_encoder='frozen e2 finalpool-input16x16map ->8x8, same one actual encoder forward',
            Classic_pure_plane_correction_applied=False,pose_3cm_guaranteed=False)
        return value

def parameter_count(config=CandidateConfig()):
    with torch.device('meta'):model=WorldTCPCommandTransformerV3(config)
    return sum(value.numel()for value in model.parameters())
