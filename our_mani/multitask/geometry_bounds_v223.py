"""Diagnostic bounds also support planar collision meshes and exact zero extent."""
import numpy as np


def box_corners(bounds):
    b=np.asarray(bounds,float)
    if b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[::2]>b[1::2]):
        raise ValueError('finite ordered local bounds required; planar meshes are valid')
    return np.array([[x,y,z] for x in b[:2] for y in b[2:4] for z in b[4:6]])


def world_bounds(points,matrix):
    p=np.asarray(points,float);m=np.asarray(matrix,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or m.shape!=(4,4)
            or not np.isfinite(np.r_[p.ravel(),m.ravel()]).all()):
        raise ValueError('finite nonempty points and homogeneous transform required')
    world=p@m[:3,:3].T+m[:3,3]
    return dict(min=world.min(0).tolist(),max=world.max(0).tolist())
