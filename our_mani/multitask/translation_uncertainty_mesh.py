"""Enlarge fixed all-rotation cup ghosts to cover current position hypotheses."""
from copy import deepcopy
import itertools
import numpy as np
from .collision_geometry import mesh_sha256


def expanded_cube(receipt,old_sha,translation_bound):
    if (receipt.get('collision_geometry_kind')!='rotation_uncertainty_cube'
        or receipt.get('registered_mesh_sha256')!=old_sha or translation_bound<=0):
        raise ValueError('original immutable all-rotation fallback and positive translation bound required')
    previous=float(receipt['registered_cube_half_extent_m'])
    extent=float(np.nextafter(np.float32(previous+translation_bound),np.float32(np.inf)))
    vertices=np.asarray(list(itertools.product((-extent,extent),repeat=3)),dtype=np.float32)
    faces=np.asarray(((0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
        (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)),dtype=np.int32)
    updated=dict(deepcopy(receipt),registered_mesh_sha256=mesh_sha256(vertices,faces),
        registered_cube_half_extent_m=extent,original_fallback_mesh_sha256=old_sha,
        original_cube_half_extent_m=previous,additional_translation_uncertainty_m=float(translation_bound),
        construction_rule='original all-rotation cube Minkowski sum with current-translation hypothesis ball; enclosing cube rounded upward to float32',
        geometry_strictly_enlarged=True,native_pose_or_future_state_used=False)
    return vertices,faces,updated


def install(actor,bound):
    service=actor.collision_port.map;clock=service._clock();rows=[]
    for name,body in service.estimated_bodies.items():
        if name not in ('cup0','cup1','cup2'):continue
        if body.fallback_ghost is None:raise ValueError('every cup needs its original all-rotation fallback')
        vertices,faces,receipt=expanded_cube(body.fallback_geometry_receipt,body.fallback_geometry_sha256,bound)
        new=service._new_ghost(vertices,faces)
        try:creation=new.get_matrix().copy()
        except BaseException:new.remove();raise
        old=body.fallback_ghost
        body.fallback_ghost=new;body.fallback_creation_matrix=creation
        body.fallback_geometry_sha256=receipt['registered_mesh_sha256'];body.fallback_geometry_receipt=receipt
        old.remove();rows.append(dict(body=name,receipt=receipt))
    if len(rows)!=3:raise ValueError('all three public cup fallback geometries required')
    service._clock_check(clock,'derived public cup translation uncertainty mesh construction')
    return rows
