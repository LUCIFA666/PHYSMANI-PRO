"""Single-task P4 proposal pilot; all forward inputs are causal public features.

The motion context c and the independent task embedding e are separate inputs.
This one-task pilot cannot learn automatic task identification. Its K outputs
are unranked imitation proposals, not calibrated grasp-success probabilities.
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SCORE_SEMANTICS = 'constant_untrained_proposal_logits_not_success_probabilities'
POSITION_BOUND_M = .12
TIME_MIN_SECONDS = .05
TIME_MAX_SECONDS = .8
INITIAL_POSITION_M = (0., 0., -.015)
EXPLICIT_PRIORS = dict(pregrasp_offset_m=.07, approach_direction=(0., 0., 1.),
                       opening=0., required_opening=1., required_width_m=.08,
                       allowed_relative_speed_m_s=.14)


def _finite(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(name + ' must be a floating-point Tensor')
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError('finite ' + name + ' with shape ' + str(shape) + ' required')
    return value


def rotation_6d_to_matrix(value: Tensor) -> Tensor:
    """Two packed columns -> SO(3), including deterministic degenerate inputs.

    Column packing is [R[:,0], R[:,1]], never row-major R[:,:2].reshape(6).
    The fallback only handles zero/collinear predictions; it is not a rotation
    observation or a learned symmetry declaration.
    """
    if (not isinstance(value, Tensor) or not value.is_floating_point()
            or value.shape[-1:] != (6,) or not bool(torch.isfinite(value).all())):
        raise ValueError('finite packed two-column rotation Tensor[...,6] required')
    first, second = value[..., :3], value[..., 3:]
    eps = 1e-7
    n1 = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    axis1 = torch.zeros_like(first);axis1[..., 0] = 1.
    b1 = torch.where(n1 > eps, first / n1.clamp_min(eps), axis1)
    orthogonal = second - (b1 * second).sum(-1, keepdim=True) * b1
    n2 = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    least_aligned = F.one_hot(b1.abs().argmin(-1), num_classes=3).to(value.dtype)
    fallback = least_aligned - (least_aligned * b1).sum(-1, keepdim=True) * b1
    fallback = F.normalize(fallback, dim=-1, eps=eps)
    b2 = torch.where(n2 > eps, orthogonal / n2.clamp_min(eps), fallback)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(value: Tensor) -> Tensor:
    if value.shape[-2:] != (3, 3):
        raise ValueError('rotation matrix[...,3,3] required')
    return torch.cat((value[..., :, 0], value[..., :, 1]), dim=-1)


def _proper_rotation(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    _finite(value, shape, name)
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if (not torch.allclose(value.transpose(-1, -2) @ value, identity.expand(*shape[:-2], 3, 3), atol=1e-4, rtol=0)
            or not torch.allclose(torch.linalg.det(value), torch.ones(shape[:-2], dtype=value.dtype, device=value.device), atol=1e-4, rtol=0)):
        raise ValueError(name + ' must contain proper SO(3) matrices')
    return value


def rotation_distance(first: Tensor, second: Tensor) -> Tensor:
    """Geodesic radians with broadcast leading dimensions; stable at identity."""
    relative = first.transpose(-1, -2) @ second
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.) / 2.).clamp(-1., 1.)
    sine_vector = .5 * torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0], relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine = torch.linalg.vector_norm(sine_vector, dim=-1)
    return torch.atan2(sine, cosine)


def parallel_gripper_rotation_distance(predicted: Tensor, target: Tensor) -> Tensor:
    """R_target and R_target @ Rz(pi) swap symmetric parallel gripper fingers.

    This is a TCP-local Z postmultiplication. World/object Z premultiplication
    is not equivalent for a general non-axis-aligned target orientation.
    """
    symmetry = torch.diag(target.new_tensor([-1., -1., 1.]))
    return torch.minimum(rotation_distance(predicted, target),
                         rotation_distance(predicted, target @ symmetry))


class CandidateTransformerPilot(nn.Module):
    """K8 object-relative pose/time proposals from one current causal anchor.

    Input features remain in physical units at the dataset boundary. The first
    future column alone is already dt/.8. Four condition tokens have separate
    projections for robot, geometry, motion context c, and task embedding e.
    Learned query tokens share two Transformer blocks with these memory tokens.
    """
    def __init__(self, *, d_model=128, num_layers=2, num_heads=4,
                 num_candidates=8, dropout=0.):
        super().__init__()
        if (type(d_model) is not int or type(num_layers) is not int or type(num_heads) is not int
                or d_model < 16 or num_layers < 1 or num_heads < 1 or d_model % num_heads
                or num_candidates != 8 or not 0 <= dropout < 1):
            raise ValueError('positive divisible Transformer dimensions and K8 required')
        self.d_model,self.num_layers,self.num_heads,self.num_candidates = d_model,num_layers,num_heads,num_candidates
        self.dropout = float(dropout)
        # Multipliers: source features and labels are never normalized in-place.
        self.register_buffer('future_input_scale', torch.tensor([1., *([10.]*3), *([1.]*6), *([5.]*3), *([1.]*3)]))
        self.register_buffer('robot_input_scale', torch.tensor([*([1./math.pi]*7), *([1./3.]*7), *([2.]*3), *([1.]*6), 1., 1.]))
        self.register_buffer('geometry_input_scale', torch.tensor([20.,20.,20.,1.]))
        self.future_projection = nn.Linear(16, d_model)
        self.robot_projection = nn.Linear(25, d_model)
        self.geometry_projection = nn.Linear(4, d_model)
        self.context_projection = nn.Linear(16, d_model)
        self.task_projection = nn.Linear(16, d_model)
        self.future_position_embedding = nn.Parameter(torch.empty(17,d_model))
        self.condition_type_embedding = nn.Parameter(torch.empty(4,d_model))
        self.queries = nn.Parameter(torch.empty(8,d_model))
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=num_heads,dim_feedforward=4*d_model,
            dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.position_head = nn.Linear(d_model,3)
        self.rotation_head = nn.Linear(d_model,6)
        self.time_head = nn.Linear(d_model,1)
        nn.init.normal_(self.future_position_embedding,std=.02)
        nn.init.normal_(self.condition_type_embedding,std=.02)
        nn.init.normal_(self.queries,std=.02)
        for head in (self.position_head,self.rotation_head,self.time_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.position_head.bias.copy_(torch.atanh(torch.tensor(INITIAL_POSITION_M)/POSITION_BOUND_M))
            self.rotation_head.bias.copy_(torch.tensor([1.,0.,0.,0.,-1.,0.]))

    def spec(self):
        return dict(schema='p4_single_task_causal_proposal_transformer_pilot_20261103_v1',
            architecture=dict(d_model=self.d_model,num_layers=self.num_layers,num_heads=self.num_heads,
                num_candidates=8,dropout=self.dropout,memory_tokens=21,query_tokens=8,batch_first=True),
            input_shapes=dict(future_tokens=[17,16],robot=[25],geometry=[4],context_c=[16],task_e=[16]),
            input_units='SI radians/metres/seconds; future first column already dt/.8',
            fixed_input_divisors=dict(future_position_m=.1,future_velocity_m_s=.2,future_angular_velocity_rad_s=1.,
                robot_q_rad=math.pi,robot_dq_rad_s=3.,robot_TCP_minus_object_m=.5,geometry_size_m=.05),
            context_and_task=dict(independent_input_projections=True,simnorm_group_size=8,
                task_embedding_source='caller supplied; single-task pilot uses two groups of8 uniform1/8',
                automatic_task_classification_learned=False),
            rotation_packing='concat(R[..., :,0],R[..., :,1]); columns, not row-major slice reshape',
            output_position_bounds_m=[-POSITION_BOUND_M,POSITION_BOUND_M],position_initial_m=list(INITIAL_POSITION_M),
            output_pose_frame='body-fixed object coordinates; instantiate at the predicted object pose at t0+relative_time',
            output_rotation_frame='object-local TCP orientation at candidate contact',
            rotation_initial_matrix=[[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
            output_relative_time_bounds_seconds=[TIME_MIN_SECONDS,TIME_MAX_SECONDS],
            proposal_score_semantics=SCORE_SEMANTICS,proposal_logits_trainable=False,
            explicit_priors=deepcopy(EXPLICIT_PRIORS),
            explicit_prior_semantics=dict(opening='binary actuator command:0 closed,1 open',
                required_opening='normalized pre-grasp aperture,1 fully open; not a width in metres',
                required_width_m='declared grasp-clearance width in metres',
                approach_direction='object-frame backoff from contact toward the pre-grasp region; +objectZ, opposite the nominal downward TCP+Z',
                pregrasp_reference='object_contact_point+approach_direction*pregrasp_offset_m'),
            gripper_pi_symmetry='target @ Rz(pi), TCP local Z',
            orientation_observed_by_public_perceptor=False,arbitrary_rotated_object_generalization_qualified=False,
            calibrated_success_probability=False)

    def forward(self, future_tokens:Tensor, robot:Tensor, geometry:Tensor, context_c:Tensor, task_e:Tensor):
        if not isinstance(future_tokens,Tensor) or future_tokens.ndim!=3:
            raise ValueError('batched current+16 predicted future tokens required')
        batch=future_tokens.shape[0]
        if batch<1:raise ValueError('nonempty feature batch required')
        values=(('future_tokens',future_tokens,(batch,17,16)),('robot',robot,(batch,25)),
                ('geometry',geometry,(batch,4)),('context_c',context_c,(batch,16)),('task_e',task_e,(batch,16)))
        for name,value,shape in values:
            _finite(value,shape,name)
            if value.device!=future_tokens.device or value.dtype!=future_tokens.dtype:
                raise ValueError('feature tensors must share dtype and device')
        clock=torch.arange(17,dtype=future_tokens.dtype,device=future_tokens.device)/16.
        if not torch.allclose(future_tokens[:,:,0],clock.expand(batch,17),atol=1e-5,rtol=0):
            raise ValueError('future clock must be current index0 plus16 actual50ms model steps, dt/.8')
        if not torch.allclose(future_tokens[:,0,1:4],torch.zeros_like(future_tokens[:,0,1:4]),atol=1e-5,rtol=0):
            raise ValueError('future position index0 must be relative to the current object anchor')
        if bool((geometry[:,:3]<=0).any()) or bool(((geometry[:,3]<0)|(geometry[:,3]>1)).any()):
            raise ValueError('positive geometry size and confidence in[0,1] required')
        if bool(((robot[:,-2:]<0)|(robot[:,-2:]>1)).any()):
            raise ValueError('actual normalized two-finger sensors required')
        for name,value in (('context_c',context_c),('task_e',task_e)):
            groups=value.reshape(batch,2,8)
            if bool((value<0).any()) or not torch.allclose(groups.sum(-1),torch.ones_like(groups.sum(-1)),atol=1e-4,rtol=0):
                raise ValueError(name+' must preserve two SimNorm groups of8; no task-ID input')
        future=self.future_projection(future_tokens*self.future_input_scale)+self.future_position_embedding
        conditions=torch.stack((self.robot_projection(robot*self.robot_input_scale),
            self.geometry_projection(geometry*self.geometry_input_scale),
            self.context_projection(context_c),self.task_projection(task_e)),dim=1)+self.condition_type_embedding
        query=self.queries.unsqueeze(0).expand(batch,-1,-1)
        encoded=self.transformer(torch.cat((future,conditions,query),dim=1))
        features=self.output_norm(encoded[:,-8:])
        positions=POSITION_BOUND_M*torch.tanh(self.position_head(features))
        raw_rotation=self.rotation_head(features)
        rotations=rotation_6d_to_matrix(raw_rotation)
        times=TIME_MIN_SECONDS+(TIME_MAX_SECONDS-TIME_MIN_SECONDS)*torch.sigmoid(self.time_head(features).squeeze(-1))
        return dict(positions=positions,rotation_matrices=rotations,rotation_6d=matrix_to_rotation_6d(rotations),
            rotation_6d_raw=raw_rotation,times=times,proposal_logits=positions.new_zeros((batch,8)),
            proposal_score_semantics=SCORE_SEMANTICS)


def matching_loss(output, target_position:Tensor, target_rotation:Tensor, target_time:Tensor,
                  *, position_scale=.03, rotation_scale=.2, time_scale=.2):
    """One positive per example: supervise only its nearest pose-time proposal.

    Other proposals have unknown labels, not negative or failed outcomes. The
    caller performs parent-balanced aggregation of ``per_example``; this helper
    never hides a batch reduction or adds a ranking/success classification loss.
    """
    for scale in (position_scale,rotation_scale,time_scale):
        if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite physical loss scales required')
    if not isinstance(output,dict) or not isinstance(output.get('positions'),Tensor):
        raise TypeError('proposal output dictionary required')
    batch=output['positions'].shape[0]
    positions=_finite(output['positions'],(batch,8,3),'proposal positions')
    rotations=_proper_rotation(output['rotation_matrices'],(batch,8,3,3),'proposal rotations')
    times=_finite(output['times'],(batch,8),'proposal times')
    _finite(target_position,(batch,3),'positive target position')
    _proper_rotation(target_rotation,(batch,3,3),'positive target rotation')
    _finite(target_time,(batch,),'positive target time')
    if any(t.device!=positions.device or t.dtype!=positions.dtype for t in (rotations,times,target_position,target_rotation,target_time)):
        raise ValueError('proposal and target tensors must share dtype and device')
    if bool(((times<TIME_MIN_SECONDS)|(times>TIME_MAX_SECONDS)).any()) or bool(((target_time<TIME_MIN_SECONDS)|(target_time>TIME_MAX_SECONDS)).any()):
        raise ValueError('predicted and positive relative times must remain in[.05,.8] seconds')
    position_error=torch.linalg.vector_norm(positions-target_position[:,None,:],dim=-1)
    rotation_error=parallel_gripper_rotation_distance(rotations,target_rotation[:,None,:,:])
    time_error=(times-target_time[:,None]).abs()
    pairwise=(position_error/position_scale).square()+(rotation_error/rotation_scale).square()+(time_error/time_scale).square()
    index=pairwise.argmin(dim=1)
    def select(value):return value.gather(1,index[:,None]).squeeze(1)
    return dict(per_example=select(pairwise),matching_index=index,pairwise_loss=pairwise,
        pairwise_position_error_m=position_error,pairwise_rotation_error_rad=rotation_error,
        pairwise_time_error_seconds=time_error,matched_position_error_m=select(position_error),
        matched_rotation_error_rad=select(rotation_error),matched_time_error_seconds=select(time_error),
        unmatched_candidates_are_negative_labels=False,proposal_score_loss_used=False,
        physical_scales=dict(position_m=position_scale,rotation_rad=rotation_scale,time_seconds=time_scale))
