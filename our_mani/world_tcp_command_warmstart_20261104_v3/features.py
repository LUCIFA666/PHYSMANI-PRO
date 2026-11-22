"""One real frozen e2 pass; read-only prepool and coarse-output hooks."""
import torch
from torch import nn

from ..world_tcp_command_warmstart_20261104_v1.features import FrozenPublicSceneMemoryV1
from .geometry import previous_current_indices,selected_depths,nominal_cell_geometry


def pool_recent_maps(original,lengths):
    """Pure tensor indexing/pooling; accepts only real valid-frame ordering."""
    slots=previous_current_indices(lengths);sizes=lengths.detach().cpu().tolist()
    if original.shape!=(sum(sizes)*4,512,16,16):
        raise ValueError('Actual flattened valid-frame four-view prepool map required')
    starts=[];n=0
    for size in sizes:starts.append(n);n+=size
    chosen=original.new_zeros(len(sizes),2,4,512,16,16)
    for batch,size in enumerate(sizes):
        for time in range(2):
            local=int(slots[batch,time])
            if local>=0:chosen[batch,time]=original[(starts[batch]+local)*4:(starts[batch]+local+1)*4]
    fine=chosen.reshape(-1,512,8,2,8,2).mean((3,5))
    return fine.reshape(len(sizes),2,4,512,64).permute(0,1,2,4,3).reshape(len(sizes),512,512)


class FrozenFinePublicSceneMemoryV3(FrozenPublicSceneMemoryV1):
    def __init__(self,checkpoint,expected_sha256,manifest,*,device='cpu'):
        super().__init__(checkpoint,expected_sha256,manifest,encoder_family='semantic_e207M_v2',device=device)

    @torch.inference_mode()
    def encode(self,*,rgbd,robot_history,lengths,K,camera_to_world,near_far,frame_indices):
        pool=self.model.camera_encoder[-1]
        if (type(pool)is not nn.AdaptiveAvgPool2d or tuple(pool.output_size)!=(2,2)
                or pool._forward_pre_hooks or pool._forward_hooks):
            raise ValueError('Exact frozen e2 final2x2pool with no foreign hooks required')
        slots=previous_current_indices(lengths);present=slots>=0
        if (frame_indices.shape!=(len(lengths),5)or frame_indices.dtype not in (torch.int32,torch.int64)
                or frame_indices.device!=lengths.device):raise ValueError('Actual causal source frame indices[B,5] required')
        actual_indices=[]
        for batch,length in enumerate(lengths.detach().cpu().tolist()):
            values=frame_indices[batch,:length]
            if (bool((values<0).any())or not bool((values[1:]-values[:-1]==1).all())
                    or not bool((frame_indices[batch,length:]==-1).all())):
                raise ValueError('Actual noninterpolated recorded source indices required')
            actual_indices.append([None if int(slot)<0 else int(frame_indices[batch,int(slot)])for slot in slots[batch]])
        sizes=lengths.detach().cpu().tolist();starts=[];n=0
        for size in sizes:starts.append(n);n+=size
        captured=[]
        def copy_prepool(module,arguments):
            if len(arguments)!=1 or arguments[0].shape!=(n*4,512,16,16):
                raise ValueError('Actual frozen e2 prepool map shape required')
            # Exact non-overlapping2x2mean for16→8. Encoder is inference-only;
            # no adaptive-pooling CUDA backward is introduced by this hook.
            captured.append(pool_recent_maps(arguments[0],lengths).detach().clone())
        handle=pool.register_forward_pre_hook(copy_prepool)
        try:
            result=super().encode(rgbd=rgbd,robot_history=robot_history,lengths=lengths,
                K=K,camera_to_world=camera_to_world,near_far=near_far)
        finally:handle.remove()
        if len(captured)!=1:raise ValueError('Exactly one actual CNN prepool capture per real encoder pass required')
        depths,depth_present,actual_slots=selected_depths(rgbd,lengths)
        geometry,mask=nominal_cell_geometry(depths,K,camera_to_world,near_far,depth_present)
        visual=torch.where(mask[...,None],captured[0],torch.zeros_like(captured[0]))
        result.update(fine_visual=visual,fine_geometry=geometry,fine_token_mask=mask,
            fine_prefix_slots=actual_slots.detach().cpu().tolist(),
            fine_actual_source_frame_indices=actual_indices,view_order=['front','8','16','36'],
            fine_geometry_semantics='all_surface_nominal_rawZ_XYZmean_validFraction_pixelCenterUVmean',
            fine_object_or_face_ownership=False,Classic_pure_plane_correction_applied=False,
            additional_trained_NN_forwards=0)
        return result
