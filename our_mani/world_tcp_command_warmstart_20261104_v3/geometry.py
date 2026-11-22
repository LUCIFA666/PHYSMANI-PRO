"""FP32 nominal raw-camera-Z surface features, with pixel centres u+.5/v+.5.

All valid surfaces contribute. This is not a selected body, face, corrected
Classic hit point, or observed object state. Geometry does not manufacture c.
"""
import torch


def previous_current_indices(lengths):
    if (not isinstance(lengths,torch.Tensor)or lengths.ndim!=1
            or lengths.dtype not in (torch.int32,torch.int64)
            or not bool(((lengths>=1)&(lengths<=5)).all())):
        raise ValueError('Actual valid recorded prefix lengths1..5 required')
    previous=torch.where(lengths>=2,lengths-2,torch.full_like(lengths,-1))
    return torch.stack((previous,lengths-1),1)


def selected_depths(rgbd,lengths):
    if rgbd.shape[1:]!=(5,4,4,256,256):raise ValueError('Original256/history5 public RGBD required')
    slots=previous_current_indices(lengths);present=slots>=0
    result=rgbd.new_zeros(len(rgbd),2,4,256,256)
    for b in range(len(rgbd)):
        for t in range(2):
            if bool(present[b,t]):result[b,t]=rgbd[b,int(slots[b,t]),:,3]
    return result,present,slots


def nominal_cell_geometry(depths,K,T,clips,present):
    """B,2,4,256,256 -> B,512,6; invalid XYZ/UV zero is unknown, not origin."""
    b=len(depths);device=depths.device
    if (depths.shape!=(b,2,4,256,256)or K.shape!=(b,4,3,3)or T.shape!=(b,4,4,4)
            or clips.shape!=(b,4,2)or present.shape!=(b,2)or present.dtype!=torch.bool
            or any(v.device!=device for v in (K,T,clips,present))):
        raise ValueError('Exact actual frame/view geometry shapes and devices required')
    with torch.autocast(device.type,enabled=False):
        K=K.float();T=T.float();clips=clips.float();depths=depths.float()
        if (not all(bool(torch.isfinite(v).all())for v in (K,T,clips))
                or bool((torch.linalg.det(K).abs()<1e-8).any())
                or not torch.allclose(T[:,:,3],T.new_tensor([0,0,0,1]).expand(b,4,4),atol=1e-6,rtol=0)
                or not torch.allclose(T[:,:,:3,:3].transpose(-1,-2)@T[:,:,:3,:3],torch.eye(3,device=device).expand(b,4,3,3),atol=1e-5,rtol=0)
                or not torch.allclose(torch.linalg.det(T[:,:,:3,:3]),torch.ones(b,4,device=device),atol=1e-5,rtol=0)
                or bool((clips[:,:,0]<=0).any())or bool((clips[:,:,1]<=clips[:,:,0]).any())):
            raise ValueError('Real public signedK/camera-to-world/clip required')
        v,u=torch.meshgrid(torch.arange(256,device=device,dtype=torch.float32)+.5,
                           torch.arange(256,device=device,dtype=torch.float32)+.5,indexing='ij')
        pixels=torch.stack((u,v,torch.ones_like(u)),-1).reshape(-1,3)
        rays=torch.einsum('bcij,pj->bcpi',torch.linalg.inv(K),pixels)
        if not torch.allclose(rays[...,2],torch.ones_like(rays[...,2]),atol=1e-6,rtol=0):
            raise ValueError('Metric camera-Z ray convention required')
        d=depths.reshape(b,2,4,-1)
        valid=torch.isfinite(d)&(d>clips[:,None,:,0,None])&(d<clips[:,None,:,1,None])&present[:,:,None,None]
        cam=rays[:,None]*torch.where(valid,d,torch.zeros_like(d))[...,None]
        world=torch.einsum('bcij,btcpj->btcpi',T[:,:,:3,:3],cam)+T[:,None,:,:3,3][:,:,:,None]
        uv=(pixels[:,:2]/256.).expand(b,2,4,-1,-1)
        values=torch.cat((world,uv),-1)*valid[...,None]
        sums=values.reshape(b,2,4,8,32,8,32,5).sum((4,6))
        counts=valid.reshape(b,2,4,8,32,8,32).sum((4,6))
        means=sums/counts.clamp_min(1)[...,None]
        geometry=torch.cat((means[...,:3],counts[...,None]/1024.,means[...,3:]),-1)
        mask=present[:,:,None,None,None].expand(b,2,4,8,8).reshape(b,512)
        return geometry.reshape(b,512,6).float(),mask
