"""K8 world TCP pose/opening intentions without a fictitious contact time."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..local_mpc_p4_20261103.model import rotation_6d_to_matrix, rotation_distance
from . import VERSION


@dataclass(frozen=True)
class CandidateConfig:
    width:int=768
    layers:int=12
    heads:int=12
    feedforward_multiplier:int=4
    dropout:float=.1
    memory_width:int=1024
    memory_tokens:int=86
    candidates:int=8

    def __post_init__(self):
        if (any(type(v)is not int or v<1 for v in (self.width,self.layers,self.heads,self.feedforward_multiplier))
                or self.width%self.heads or self.memory_width!=1024 or self.memory_tokens!=86
                or self.candidates!=8 or isinstance(self.dropout,bool) or not 0<=self.dropout<1):
            raise ValueError('Explicit positive divisible Transformer and real1024/86/K8 contract required')


class WorldTCPCommandTransformerV1(nn.Module):
    def __init__(self,config=CandidateConfig()):
        super().__init__()
        if type(config)is not CandidateConfig:raise ValueError('Exact CandidateConfig required')
        self.config=config;d=config.width
        self.visual_projection=nn.Linear(1024,d)
        self.robot_projection=nn.Linear(14,d)
        self.calibration_projection=nn.Linear(23,d)
        self.task_projection=nn.Linear(16,d)
        self.memory_positions=nn.Parameter(torch.empty(1,86,d))
        self.condition_types=nn.Parameter(torch.empty(1,6,d))
        self.queries=nn.Parameter(torch.empty(1,8,d))
        layer=nn.TransformerEncoderLayer(d,config.heads,d*config.feedforward_multiplier,
            dropout=config.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.output_norm=nn.LayerNorm(d)
        self.position_head=nn.Linear(d,3)
        self.rotation_head=nn.Linear(d,6)
        self.opening_head=nn.Linear(d,1)
        self.demo_likelihood_head=nn.Linear(d,1)
        self.register_buffer('calibration_scale',torch.tensor([*([1/256]*6),1.,1.,1.,*([1.]*14)]))
        self.register_buffer('robot_scale',torch.tensor([*([1/3.141592653589793]*7),*([1.]*7)]))
        for parameter in (self.memory_positions,self.condition_types,self.queries):nn.init.normal_(parameter,std=.02)
        for head in (self.position_head,self.rotation_head,self.opening_head,self.demo_likelihood_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        with torch.no_grad():self.rotation_head.bias.copy_(torch.tensor([1.,0.,0.,0.,1.,0.]))

    def forward(self,*,visual_memory,visual_memory_mask,current_robot,camera_calibration,task_e):
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
        calibration=self.calibration_projection(camera_calibration*self.calibration_scale)
        conditions=torch.cat((self.robot_projection(current_robot*self.robot_scale)[:,None],
                              calibration,self.task_projection(task_e)[:,None]),1)+self.condition_types
        tokens=torch.cat((memory,conditions,self.queries.expand(b,-1,-1)),1)
        padding=torch.cat((~visual_memory_mask,torch.zeros(b,14,dtype=torch.bool,device=device)),1)
        result=self.output_norm(self.transformer(tokens,src_key_padding_mask=padding)[:,-8:])
        # These are unbounded world-frame command displacements, not object
        # observations and not a bounded physical actuator action.
        positions=current_robot[:,7:10,None].transpose(1,2)+self.position_head(result).float()
        rotations=rotation_6d_to_matrix(self.rotation_head(result).float())
        opening_logits=self.opening_head(result).float().squeeze(-1)
        return dict(world_tcp_positions=positions,world_tcp_rotations=rotations,
            opening_intent=opening_logits.sigmoid(),opening_logits=opening_logits,
            proposal_logits=self.demo_likelihood_head(result).float().squeeze(-1),handoff_tau_seconds=None,
            proposal_score_semantics='expert_command_demonstration_mixture_not_task_success',
            output_semantics='expert_command_intent_not_observed_state',
            physical_handoff_qualified=False,calibrated_success_probability=False)

    def spec(self):
        return dict(version=VERSION,architecture=asdict(self.config),
            inputs=dict(visual_memory=[86,1024],visual_memory_mask=[86],current_robot=[14],camera_calibration=[4,23],task_e=[16]),
            scene_encoder_frozen=True,task_e_is_predicted_not_metadata=True,
            motion_c_input=False,body_local_frame_input=False,physical_clock_input=False,
            world_TCP_command_output=True,opening_is_command_not_observation=True,
            proposal_logits_trainable=True,proposal_score_semantics='demonstration_mixture_likelihood_not_success',
            handoff_tau_seconds=None,time_loss_weight=0.,success_probability=False,
            P4_strict_handoff_consumer_compatible=False,current_network_end_to_end_ready=False)


def parameter_count(config=CandidateConfig()):
    with torch.device('meta'):model=WorldTCPCommandTransformerV1(config)
    return sum(value.numel()for value in model.parameters())


def command_imitation_loss(prediction,position,rotation,opening):
    """One recorded desired command; no physical-arrival/contact-time label."""
    p=prediction['world_tcp_positions'];r=prediction['world_tcp_rotations']
    logits=prediction['opening_logits'];b=len(p)
    if (position.shape!=(b,3) or rotation.shape!=(b,3,3) or opening.shape!=(b,)
            or not all(bool(torch.isfinite(v).all())for v in (position,rotation,opening))
            or bool(((opening<0)|(opening>1)).any())):raise ValueError('Finite world expert-command targets required')
    with torch.autocast(p.device.type,enabled=False):
        rotation=rotation.float()
        if (not torch.allclose(rotation.transpose(-1,-2)@rotation,torch.eye(3,device=rotation.device).expand(b,3,3),atol=1e-4,rtol=0)
                or not torch.allclose(torch.linalg.det(rotation),torch.ones(b,device=rotation.device),atol=1e-4,rtol=0)):
            raise ValueError('Expert orientation label must be proper SO3')
        position_error=(p.float()-position.float()[:,None]).square().sum(-1)
        angle=rotation_distance(r.float(),rotation[:,None])
    opening_error=torch.nn.functional.binary_cross_entropy_with_logits(logits,opening[:,None].expand_as(logits),reduction='none')
    per_candidate=position_error/(2*.03**2)+angle.square()/(2*.2**2)+opening_error
    log_weights=torch.nn.functional.log_softmax(prediction['proposal_logits'].float(),-1)
    loss=-torch.logsumexp(log_weights-per_candidate,-1).mean()
    best=(log_weights-per_candidate).argmax(-1);rows=torch.arange(b,device=p.device)
    return dict(loss=loss,selected_candidate=best,
        position_error_m=torch.sqrt(position_error[rows,best]),orientation_error_rad=angle[rows,best],
        opening_intent_error=(prediction['opening_intent'][rows,best]-opening).abs(),
        time_loss=p.sum()*0,physical_handoff_supervised=False,
        position_training_scale_m=.03,rotation_training_scale_rad=.2,
        training_scales_are_confidence_intervals=False)
