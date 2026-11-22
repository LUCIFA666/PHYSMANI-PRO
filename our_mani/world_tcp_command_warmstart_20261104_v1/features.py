"""Read-only frozen scene-feature extraction through the real encoder call."""
import torch


def causal_memory_mask(lengths):
    if (not isinstance(lengths,torch.Tensor)or lengths.ndim!=1
            or lengths.dtype not in (torch.int32,torch.int64)
            or not bool(((lengths>=1)&(lengths<=5)).all())):
        raise ValueError('Actual prefix lengths1..5 required')
    b=len(lengths);frame=torch.arange(5,device=lengths.device)[None]<lengths[:,None]
    return torch.cat((torch.ones(b,1,dtype=torch.bool,device=lengths.device),
                      frame[:,:,None].expand(b,5,17).reshape(b,85)),1)

def frozen_encoder_binding(family):
    # The family is an explicit source/spec contract, never a state-dict shape
    # compatibility fallback. A failed/untrained family is not substituted.
    if family=='semantic_e207M_v1':
        from ..semantic_task_e_large_20261104_v1.common import source_map,verify_sources
        from ..semantic_task_e_large_20261104_v1.evaluation import load_selected
        from ..semantic_task_e_large_20261104_v1.model import LargeSemanticTaskEncoderV1 as model_type
    elif family=='semantic_e207M_v2':
        from ..semantic_task_e_large_20261104_v2.common import source_map,verify_sources
        from ..semantic_task_e_large_20261104_v2.evaluation import load_selected
        from ..semantic_task_e_large_20261104_v2.model import LargeSemanticTaskEncoderV2 as model_type
    else:raise ValueError('Explicit supported semantic encoder family required; no auto fallback')
    return source_map,verify_sources,load_selected,model_type


class FrozenPublicSceneMemoryV1:
    def __init__(self,checkpoint,expected_sha256,manifest,*,encoder_family,device='cpu'):
        source_map,verify_sources,load_selected,model_type=frozen_encoder_binding(encoder_family)
        self.sources=source_map();verify_sources(self.sources)
        model,metadata=load_selected(checkpoint,expected_sha256,manifest,self.sources)
        self.model=model.to(device).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.checkpoint_sha256=expected_sha256;self.metadata=metadata
        self.encoder_family=encoder_family;self.model_type=model_type

    @torch.inference_mode()
    def encode(self,*,rgbd,robot_history,lengths,K,camera_to_world,near_far):
        source_map,verify_sources,_,model_type=frozen_encoder_binding(self.encoder_family)
        if (self.model_type is not model_type or type(self.model)is not model_type
                or getattr(self.model.forward,'__func__',None)is not model_type.forward
                or self.model.training or any(p.requires_grad for p in self.model.parameters())):
            raise ValueError('Exact actual frozen semantic encoder required; no substituted actor/feature model')
        if rgbd.shape[1:]!=(5,4,4,256,256):raise ValueError('Real original256/history5 public inputs required')
        b=rgbd.shape[0]
        if (K.shape!=(b,4,3,3) or camera_to_world.shape!=(b,4,4,4) or near_far.shape!=(b,4,2)
                or not all(bool(torch.isfinite(v).all())for v in (K,camera_to_world,near_far))):
            raise ValueError('Original public per-view K/T/clip required')
        if self.model.transformer._forward_hooks:
            raise ValueError('No foreign feature-changing hooks allowed')
        captured=[]
        def copy_output(module,arguments,value):
            if not isinstance(value,torch.Tensor):raise ValueError('Real Transformer tensor output required')
            captured.append(value.detach().clone())
        # A temporary read-only hook copies the actual uncompressed hidden
        # tensor. It never changes inputs, outputs, weights or runtime classes.
        handle=self.model.transformer.register_forward_hook(copy_output)
        try:prediction=self.model(rgbd=rgbd,robot_history=robot_history,lengths=lengths)
        finally:handle.remove()
        if len(captured)!=1 or captured[0].shape!=(b,86,1024):
            raise ValueError('Exactly one actual 1024-wide causal feature pass required')
        mask=causal_memory_mask(lengths)
        memory=torch.where(mask[...,None],captured[0],torch.zeros_like(captured[0]))
        rows=torch.arange(b,device=lengths.device)
        current_robot=robot_history[rows,lengths-1,:14].clone()
        calibration=torch.cat((K.reshape(b,4,9),camera_to_world[:,:,:3].reshape(b,4,12),near_far),-1)
        verify_sources(self.sources)
        return dict(visual_memory=memory,visual_memory_mask=mask,current_robot=current_robot,
            camera_calibration=calibration,task_e=prediction['task_e'].detach().clone(),
            frozen_scene_model_sha256=self.checkpoint_sha256,physical_clock_verified=False,
            frozen_scene_encoder_family=self.encoder_family,
            motion_c=None,body_local_frame=None,privileged_or_future_inputs=False)
